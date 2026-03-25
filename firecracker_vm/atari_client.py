"""
Host-side client for the Atari guest agent — vsock version.

Connection flow:
1. Wait for AGENT_READY on serial (firecracker stdout) — confirms
   the guest agent has started and is listening on vsock
2. Connect to Firecracker's AF_UNIX vsock socket
3. Send "CONNECT 5005\n", wait for "OK <port>\n"
4. After handshake, the socket carries raw data to/from the guest

Vsock gives us virtio-speed transfers — no frame size constraints.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
from typing import IO, Optional

AGENT_READY_SENTINEL = "AGENT_READY"
VSOCK_GUEST_PORT = 5005


class AtariClient:
    def __init__(self, vsock_uds_path: str, serial_stdout: IO[bytes]):
        """
        Args:
            vsock_uds_path: Path to Firecracker's AF_UNIX vsock socket
            serial_stdout:  Firecracker's stdout pipe (for AGENT_READY detection)
        """
        self._vsock_uds_path = vsock_uds_path
        self._serial_stdout = serial_stdout
        self._sock: Optional[socket.socket] = None
        self._ready = threading.Event()
        self._serial_thread: Optional[threading.Thread] = None

    def connect(self, timeout: float = 60.0, skip_ready: bool = False) -> None:
        """
        Wait for AGENT_READY on serial, then connect via vsock.

        For restored VMs (skip_ready=True), skip the serial wait and
        connect to vsock directly — the agent is already listening.
        """
        if not skip_ready:
            # Start a thread to watch serial for AGENT_READY
            self._serial_thread = threading.Thread(
                target=self._watch_serial, daemon=True
            )
            self._serial_thread.start()

            if not self._ready.wait(timeout=timeout):
                raise ConnectionError(
                    f"Guest agent did not send {AGENT_READY_SENTINEL} "
                    f"within {timeout}s."
                )

        # Now connect to vsock
        self._connect_vsock(timeout=timeout if skip_ready else 10.0)

    def _watch_serial(self) -> None:
        """Read serial stdout until we see AGENT_READY."""
        while True:
            try:
                raw = self._serial_stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if AGENT_READY_SENTINEL in line:
                    self._ready.set()
                    return
            except Exception:
                break

    def _connect_vsock(self, timeout: float = 10.0) -> None:
        """
        Connect to guest agent via Firecracker's vsock AF_UNIX proxy.

        Handshake: connect to UDS, send "CONNECT <port>\n", expect "OK ...\n".
        Retries until timeout — the guest may need a moment after restore.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(self._vsock_uds_path)

                # Firecracker vsock handshake
                sock.sendall(f"CONNECT {VSOCK_GUEST_PORT}\n".encode("ascii"))

                response = b""
                while b"\n" not in response:
                    chunk = sock.recv(256)
                    if not chunk:
                        raise ConnectionError(
                            "Firecracker closed vsock connection")
                    response += chunk

                resp_line = response.split(b"\n")[0].decode("ascii")
                if not resp_line.startswith("OK"):
                    raise ConnectionError(
                        f"Vsock handshake failed: {resp_line}")

                self._sock = sock
                return

            except (ConnectionRefusedError, FileNotFoundError, OSError,
                    ConnectionError):
                try:
                    sock.close()
                except Exception:
                    pass
                time.sleep(0.3)

        raise ConnectionError(
            f"Could not connect to guest via vsock at "
            f"{self._vsock_uds_path} within {timeout}s"
        )

    def _send_command(self, cmd: dict, timeout: float = 30.0) -> dict:
        """Send a JSON command over vsock and wait for the response."""
        if self._sock is None:
            raise RuntimeError("Not connected")

        msg = json.dumps(cmd).encode("utf-8") + b"\n"
        self._sock.sendall(msg)

        # Read response
        buf = b""
        self._sock.settimeout(timeout)
        while b"\n" not in buf:
            chunk = self._sock.recv(1024 * 1024)  # 1 MiB — vsock is fast
            if not chunk:
                raise ConnectionError("Guest disconnected")
            buf += chunk

        line = buf.split(b"\n", 1)[0]
        response = json.loads(line.decode("utf-8"))

        if response.get("status") == "error":
            raise RuntimeError(f"Guest error: {response.get('message')}")
        return response

    def reset(self) -> bytes:
        """Reset environment, return observation as PNG bytes."""
        response = self._send_command({"cmd": "reset"})
        return base64.b64decode(response["obs"])

    def step(self, action: int) -> tuple[bytes, float, bool, dict]:
        """Step environment. Returns (obs_bytes, reward, done, info)."""
        response = self._send_command({"cmd": "step", "action": action})
        obs = base64.b64decode(response["obs"])
        return obs, response["reward"], response["done"], response.get("info", {})

    def get_legal_actions(self) -> list[int]:
        response = self._send_command({"cmd": "get_actions"})
        return response["actions"]

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
