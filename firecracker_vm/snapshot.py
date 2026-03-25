"""
Snapshot management for Firecracker VMs.

A snapshot captures the complete state of a running microVM at a point in time:
- vmstate file: CPU registers, device state, KVM state (~1-2 MB)
- memory file:  guest RAM contents (= mem_size_mib, but sparse)
- rootfs:       disk image (not captured by Firecracker, managed separately)

The memory file is the big one. For a 256 MiB VM, it's up to 256 MiB on disk.
However, Firecracker writes it as a sparse file, so only pages that have been
touched take real disk space.

When restoring, Firecracker mmap's the memory file with MAP_PRIVATE. This gives
us copy-on-write semantics for free: the physical pages are shared between the
snapshot file and the restored VM, and only pages the guest modifies afterward
get copied. This is what makes forking fast.

For disk forking, we need to handle the rootfs ourselves. Options:
1. cp --reflink=auto: uses CoW if the filesystem supports it (btrfs, xfs)
2. Regular copy: safe fallback, but slow for large rootfs images
3. Overlay: use a thin overlay on top of a shared base (most efficient)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from firecracker_vm.vm import VMConfig


@dataclass
class Snapshot:
    """
    An immutable capture of VM state that can spawn new VMs.

    Once created, a Snapshot should be treated as immutable. Multiple VMs
    can be restored from the same snapshot concurrently (each gets its own
    CoW copy of memory via MAP_PRIVATE).

    The rootfs_path points to the disk image at the time of snapshot.
    When forking, callers should create a copy of the rootfs for each
    child VM (see fork_rootfs()).
    """
    vmstate_path: str   # Path to the serialized VM state file
    memory_path: str    # Path to the guest memory dump
    rootfs_path: str    # Path to the rootfs image used by the source VM
    config: VMConfig    # Original VM configuration (vcpus, mem, etc.)

    def fork_rootfs(self, dest_path: str, use_reflink: bool = True) -> str:
        """
        Create a writable copy of the rootfs for a child VM.

        If the host filesystem supports reflinks (btrfs, xfs with reflink=1),
        this is nearly instant — the copy shares physical blocks with the
        original and only diverges on write (CoW at the filesystem level).

        Otherwise falls back to a regular copy, which is slower but always works.

        Returns the path to the new rootfs copy.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if use_reflink:
            try:
                subprocess.run(
                    ["cp", "--reflink=auto", self.rootfs_path, str(dest)],
                    check=True,
                    capture_output=True,
                )
                return str(dest)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass  # Fall through to regular copy

        shutil.copy2(self.rootfs_path, str(dest))
        return str(dest)

    def copy_to(self, dest_dir: str) -> "Snapshot":
        """
        Create a full copy of this snapshot in a new directory.

        Useful when you need to modify snapshot files (e.g., for diff snapshot
        rebasing) without affecting the original.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        new_vmstate = str(dest / "vmstate")
        new_memory = str(dest / "memory")

        shutil.copy2(self.vmstate_path, new_vmstate)
        shutil.copy2(self.memory_path, new_memory)

        return Snapshot(
            vmstate_path=new_vmstate,
            memory_path=new_memory,
            rootfs_path=self.rootfs_path,
            config=self.config,
        )

    def size_bytes(self) -> dict[str, int]:
        """
        Report the on-disk size of snapshot components.

        Note: the memory file may be sparse, so apparent size != disk usage.
        We report both for transparency.
        """
        vmstate_stat = os.stat(self.vmstate_path)
        memory_stat = os.stat(self.memory_path)

        return {
            "vmstate_apparent": vmstate_stat.st_size,
            "vmstate_disk": vmstate_stat.st_blocks * 512,
            "memory_apparent": memory_stat.st_size,
            "memory_disk": memory_stat.st_blocks * 512,
        }

    def cleanup(self) -> None:
        """Remove snapshot files from disk."""
        for path in [self.vmstate_path, self.memory_path]:
            try:
                os.unlink(path)
            except OSError:
                pass

    def __repr__(self):
        return (
            f"Snapshot(vmstate={self.vmstate_path}, "
            f"memory={self.memory_path})"
        )
