"""
Single Firecracker microVM management.

Each FirecrackerVM instance corresponds to exactly one Firecracker OS process.
Communication happens over a Unix domain socket using Firecracker's REST API.

Lifecycle:
    1. Create VMConfig with kernel, rootfs, etc.
    2. Instantiate FirecrackerVM (spawns the firecracker process)
    3. Configure boot source, drives, machine config via API
    4. Start the VM (InstanceStart action)
    5. Communicate with guest over vsock/serial
    6. Snapshot / restore / destroy as needed

For snapshot-restored VMs, the lifecycle starts at step 5 — the VM resumes
exactly where the snapshot was taken, with all guest processes intact.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import http.client


class VMState(Enum):
    """Tracks the VM lifecycle from the host's perspective."""
    CREATED = "created"       # Firecracker process started, not yet configured
    CONFIGURED = "configured"  # Boot source + drives set, not yet running
    RUNNING = "running"       # Guest is executing
    PAUSED = "paused"         # Guest frozen (required before snapshot)
    STOPPED = "stopped"       # Process terminated


@dataclass
class VMConfig:
    """
    Everything needed to boot a fresh Firecracker VM.

    Fields:
        firecracker_bin: Path to the firecracker binary
        kernel_path:     Path to the guest kernel (vmlinux)
        rootfs_path:     Path to the root filesystem image (ext4)
        vcpu_count:      Number of virtual CPUs
        mem_size_mib:    Guest memory in MiB
        boot_args:       Kernel command line arguments
        track_dirty:     Enable dirty page tracking for diff snapshots
    """
    firecracker_bin: str = "firecracker"
    kernel_path: str = ""
    rootfs_path: str = ""
    vcpu_count: int = 1
    mem_size_mib: int = 256
    boot_args: str = "console=ttyS0 reboot=k panic=1 pci=off random.trust_cpu=on init=/opt/init.sh"
    track_dirty: bool = True
    extra_drives: list[dict] = field(default_factory=list)


class FirecrackerVM:
    """
    Manages a single Firecracker microVM process.

    This is the lowest-level abstraction. It owns:
    - The firecracker OS process
    - A working directory for socket, logs, etc.
    - The API connection over the Unix socket

    Thread safety: Not thread-safe. Each VM should be managed from a single
    thread. Use VMPool for concurrent VM management.
    """

    def __init__(self, config: VMConfig, work_dir: Optional[str] = None,
                 vm_id: Optional[str] = None):
        self.config = config
        self.vm_id = vm_id or f"vm-{os.getpid()}-{id(self):x}"
        self.state = VMState.CREATED

        # Each VM gets its own working directory to avoid socket/file collisions
        if work_dir:
            self.work_dir = Path(work_dir).resolve()
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._owns_work_dir = False
        else:
            self.work_dir = Path(tempfile.mkdtemp(
                prefix=f"fc-{self.vm_id}-")).resolve()
            self._owns_work_dir = True

        self.socket_path = self.work_dir / "firecracker.sock"
        self._process: Optional[subprocess.Popen] = None
        self._log_path = self.work_dir / "firecracker.log"
        # Track which rootfs this VM is using (may differ from config for forks)
        self.rootfs_path = config.rootfs_path

    def boot(self) -> None:
        """
        Full boot sequence: spawn process, configure, start.

        Communication with the guest uses vsock (virtio socket) —
        much faster than serial, supports full-bandwidth data transfer.
        """
        self._spawn_process()
        self._configure_machine()
        self._configure_boot_source()
        self._configure_rootfs()
        self._configure_vsock()
        for drive in self.config.extra_drives:
            self._add_drive(drive)
        self._start_instance()

    def _spawn_process(self) -> None:
        """
        Start the Firecracker binary as a child process.

        The process listens on a Unix domain socket for API commands.
        We pipe stdin/stdout for serial console communication with the guest.
        Firecracker connects the guest's /dev/ttyS0 to the process's
        stdin/stdout, giving us a bidirectional byte stream to the guest.
        """
        # Clean up any stale socket
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Resolve paths to absolute since we set cwd to work_dir
        fc_bin = str(Path(self.config.firecracker_bin).resolve())

        self._process = subprocess.Popen(
            [
                fc_bin,
                "--api-sock", str(self.socket_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(self._log_path, "w"),
            cwd=str(self.work_dir),  # vsock "v.sock" resolves relative to this
        )

        # Wait for the socket to become available
        for _ in range(100):  # 10 second timeout
            if self.socket_path.exists():
                try:
                    self._api_call("GET", "/")
                    return
                except Exception:
                    pass
            time.sleep(0.1)

        raise RuntimeError(
            f"Firecracker failed to start. Check {self._log_path}"
        )

    def _spawn_process_for_restore(self) -> None:
        """
        Start a Firecracker process that will be used for snapshot restore.

        Same as _spawn_process but the VM won't be configured from scratch —
        instead, the caller will load a snapshot into it.
        """
        self._spawn_process()

    def _api_call(self, method: str, path: str,
                  body: Optional[dict] = None) -> dict[str, Any]:
        """
        Make an HTTP request to the Firecracker API over the Unix socket.

        Firecracker's API is a simple REST interface:
        - PUT for creating/setting resources (idempotent)
        - PATCH for modifying resources
        - GET for reading state

        Returns the parsed JSON response body, or raises on error.

        Under the hood this uses Python's http.client with a Unix socket
        connection. In a Rust implementation, you'd use hyper with a
        UnixStream transport — same idea, different plumbing.
        """
        conn = _UnixHTTPConnection(str(self.socket_path))
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        body_str = json.dumps(body) if body else None

        conn.request(method, path, body=body_str, headers=headers)
        response = conn.getresponse()
        response_body = response.read().decode("utf-8")
        conn.close()

        if response.status >= 400:
            raise RuntimeError(
                f"Firecracker API error {response.status} on {method} {path}: "
                f"{response_body}"
            )

        if response_body:
            try:
                return json.loads(response_body)
            except json.JSONDecodeError:
                return {}
        return {}

    def _configure_machine(self) -> None:
        """Set vCPU count, memory, and dirty page tracking."""
        self._api_call("PUT", "/machine-config", {
            "vcpu_count": self.config.vcpu_count,
            "mem_size_mib": self.config.mem_size_mib,
            "track_dirty_pages": self.config.track_dirty,
        })

    def _configure_boot_source(self) -> None:
        """Set the guest kernel and boot arguments."""
        kernel = str(Path(self.config.kernel_path).resolve())
        self._api_call("PUT", "/boot-source", {
            "kernel_image_path": kernel,
            "boot_args": self.config.boot_args,
        })

    def _configure_rootfs(self) -> None:
        """Attach the root filesystem as a virtio-blk device."""
        rootfs = str(Path(self.config.rootfs_path).resolve())
        self._api_call("PUT", "/drives/rootfs", {
            "drive_id": "rootfs",
            "path_on_host": rootfs,
            "is_root_device": True,
            "is_read_only": False,
        })

    def _configure_vsock(self) -> None:
        """
        Attach a virtio-vsock device for host↔guest communication.

        We use a RELATIVE uds_path ("v.sock") so that each Firecracker
        process resolves it relative to its own cwd (the VM's work_dir).
        This is critical for snapshot/restore: the snapshot bakes the
        uds_path, so if we used an absolute path, all restored children
        would try to bind to the same socket and collide.
        """
        self._api_call("PUT", "/vsock", {
            "vsock_id": "1",
            "guest_cid": 3,
            "uds_path": "v.sock",
        })

    @property
    def vsock_uds_path(self) -> str:
        """Absolute path to the vsock UDS (resolved from work_dir + relative path)."""
        return str(self.work_dir / "v.sock")

    @property
    def serial_stdin(self):
        """
        Writable pipe to the guest's /dev/ttyS0 (via Firecracker's stdin).

        Whatever you write here appears on the guest's serial input.
        """
        if self._process is None:
            raise RuntimeError("VM process not started")
        return self._process.stdin

    @property
    def serial_stdout(self):
        """
        Readable pipe from the guest's /dev/ttyS0 (via Firecracker's stdout).

        Whatever the guest writes to /dev/ttyS0 appears here. Note: this
        includes kernel boot messages, init output, AND agent protocol data.
        The client must parse through the noise to find protocol lines.
        """
        if self._process is None:
            raise RuntimeError("VM process not started")
        return self._process.stdout

    def _add_drive(self, drive_config: dict) -> None:
        """Attach an additional virtio-blk device."""
        drive_id = drive_config.get("drive_id", "extra")
        self._api_call("PUT", f"/drives/{drive_id}", drive_config)

    def _start_instance(self) -> None:
        """Send InstanceStart action — this boots the guest kernel."""
        self._api_call("PUT", "/actions", {
            "action_type": "InstanceStart",
        })
        self.state = VMState.RUNNING

    def pause(self) -> None:
        """
        Pause the VM — freezes all vCPUs.

        This is a prerequisite for snapshot creation. The guest doesn't
        know it's paused; from its perspective, time simply stops.
        """
        if self.state != VMState.RUNNING:
            raise RuntimeError(
                f"Can only pause a running VM, got {self.state}")
        self._api_call("PATCH", "/vm", {"state": "Paused"})
        self.state = VMState.PAUSED

    def resume(self) -> None:
        """
        Resume a paused VM — unfreezes all vCPUs.

        From the guest's perspective, time jumps forward by however long
        the VM was paused.
        """
        if self.state != VMState.PAUSED:
            raise RuntimeError(
                f"Can only resume a paused VM, got {self.state}")
        self._api_call("PATCH", "/vm", {"state": "Resumed"})
        self.state = VMState.RUNNING

    def create_snapshot(self, snapshot_dir: str,
                        snapshot_type: str = "Full") -> "Snapshot":
        """
        Create a snapshot of the current VM state.

        Produces two files:
        - vmstate: serialized CPU registers, device state, KVM state
        - memory:  full guest RAM contents (or dirty pages for Diff)

        The VM must be paused before calling this. After snapshot creation,
        the VM can be resumed and continues running normally.

        The disk (rootfs) is NOT included in the snapshot — the caller
        must manage disk file copies separately if needed for forking.

        Returns a Snapshot object that can be used to restore new VMs.
        """
        if self.state != VMState.PAUSED:
            raise RuntimeError("VM must be paused before snapshotting")

        from firecracker_vm.snapshot import Snapshot

        snap_dir = Path(snapshot_dir).resolve()
        snap_dir.mkdir(parents=True, exist_ok=True)
        vmstate_path = snap_dir / "vmstate"
        memory_path = snap_dir / "memory"

        self._api_call("PUT", "/snapshot/create", {
            "snapshot_type": snapshot_type,
            "snapshot_path": str(vmstate_path),
            "mem_file_path": str(memory_path),
        })

        return Snapshot(
            vmstate_path=str(vmstate_path),
            memory_path=str(memory_path),
            rootfs_path=str(Path(self.config.rootfs_path).resolve()),
            config=self.config,
        )

    def destroy(self) -> None:
        """
        Kill the Firecracker process and clean up.

        This is the nuclear option — the guest gets no chance to shut down
        gracefully. For our use case (disposable RL environment VMs), this
        is fine.
        """
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=5)
        self.state = VMState.STOPPED

        # Clean up socket
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    def is_alive(self) -> bool:
        """Check if the Firecracker process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    @classmethod
    def restore_from_snapshot(
        cls,
        snapshot: "Snapshot",
        work_dir: Optional[str] = None,
        vm_id: Optional[str] = None,
        rootfs_path: Optional[str] = None,
        resume: bool = True,
    ) -> "FirecrackerVM":
        """
        Create a new VM by restoring from a snapshot.

        This is the core of the forking mechanism. The new VM:
        1. Gets a fresh Firecracker process
        2. Loads the snapshot's vmstate and memory (MAP_PRIVATE / CoW)
        3. Optionally resumes immediately

        The rootfs_path parameter allows using a copy-on-write copy of
        the original disk, which is important for forking — each child
        needs its own writable disk.

        Performance note: The memory file is mmap'd MAP_PRIVATE, so only
        pages the child modifies are physically copied. This makes
        fork+restore very fast (~30ms) and memory-efficient.
        """
        from firecracker_vm.snapshot import Snapshot

        config = snapshot.config
        vm = cls(config, work_dir=work_dir, vm_id=vm_id)
        vm._spawn_process_for_restore()

        # Use provided rootfs or the snapshot's original
        actual_rootfs = rootfs_path or snapshot.rootfs_path
        vm.rootfs_path = actual_rootfs

        vm._api_call("PUT", "/snapshot/load", {
            "snapshot_path": snapshot.vmstate_path,
            "mem_backend": {
                "backend_type": "File",
                "backend_path": snapshot.memory_path,
            },
            "enable_diff_snapshots": config.track_dirty,
            "resume_vm": resume,
        })

        vm.state = VMState.RUNNING if resume else VMState.PAUSED
        return vm

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()

    def __repr__(self):
        return f"FirecrackerVM(id={self.vm_id}, state={self.state.value})"


class _UnixHTTPConnection(http.client.HTTPConnection):
    """
    HTTP connection over a Unix domain socket.

    Firecracker's API server listens on a Unix socket, not TCP.
    This adapter lets us use Python's standard http.client with it.

    The trick: we override connect() to create a Unix socket and
    connect to the file path, but everything else (HTTP framing,
    request/response parsing) works identically to TCP.
    """

    def __init__(self, socket_path: str, timeout: float = 10.0):
        # The host parameter doesn't matter for Unix sockets,
        # but http.client requires it
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._socket_path)
