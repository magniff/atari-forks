#!/usr/bin/env python3
"""
Guest agent for the Atari environment — vsock version.

Communicates over virtio-vsock (AF_VSOCK), which provides high-bandwidth
host↔guest communication via virtio DMA ring buffers. Much faster than
the 8250 UART serial console — easily handles full uncompressed frames.

The agent listens on AF_VSOCK port 5005. The host connects via
Firecracker's AF_UNIX vsock proxy.

Protocol (newline-delimited JSON):
    Host → Guest: {"cmd": "reset"} / {"cmd": "step", "action": N} / {"cmd": "get_actions"}
    Guest → Host: {"status": "ok", "obs": "<base64 png>", ...}

Also writes "AGENT_READY" to /dev/ttyS0 so the host knows when to connect.
"""

import base64
import io
import json
import os
import signal
import socket
import sys

VSOCK_PORT = 5005
AF_VSOCK = 40
VMADDR_CID_ANY = 0xFFFFFFFF


# ── ALE Environment ─────────────────────────────────────────────────

class AtariEnv:
    def __init__(self, rom_path=None):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        self.ale = None
        self.legal_actions = [0, 1, 2, 3]

        def _timeout(signum, frame):
            raise TimeoutError("ALE init timed out")

        try:
            old = signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(10)

            from ale_py import ALEInterface
            self.ale = ALEInterface()
            self.ale.setInt("random_seed", 42)
            self.ale.setFloat("repeat_action_probability", 0.0)
            self.ale.setBool("display_screen", False)
            self.ale.setBool("sound", False)
            if rom_path:
                self.ale.loadROM(rom_path)
            else:
                import ale_py.roms as roms
                self.ale.loadROM(roms.get_rom_path("breakout"))
            self.legal_actions = [int(a)
                                  for a in self.ale.getMinimalActionSet()]

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        except Exception:
            signal.alarm(0)
            self.ale = None

    def reset(self) -> bytes:
        if self.ale:
            self.ale.reset_game()
            return self._capture()
        return self._fallback_frame()

    def step(self, action) -> tuple:
        if self.ale:
            reward = self.ale.act(action)
            done = self.ale.game_over()
            return self._capture(), float(reward), done, {"lives": self.ale.lives()}
        return self._fallback_frame(), 0.0, False, {"mock": True}

    def _capture(self) -> bytes:
        try:
            from PIL import Image
            screen = self.ale.getScreenRGB()
            img = Image.fromarray(screen)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return self._fallback_frame()

    @staticmethod
    def _fallback_frame() -> bytes:
        """Minimal 1x1 PNG."""
        return (
            b'\x89PNG\r\n\x1a\n'
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde'
            b'\x00\x00\x00\x0cIDATx'
            b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05'
            b'\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )


# ── Dispatch ─────────────────────────────────────────────────────────

def dispatch(env, cmd):
    try:
        action = cmd.get("cmd")
        if action == "reset":
            obs = env.reset()
            return {
                "status": "ok",
                "obs": base64.b64encode(obs).decode("ascii"),
                "reward": 0.0, "done": False, "info": {},
            }
        elif action == "step":
            obs, reward, done, info = env.step(cmd["action"])
            return {
                "status": "ok",
                "obs": base64.b64encode(obs).decode("ascii"),
                "reward": reward, "done": done, "info": info,
            }
        elif action == "get_actions":
            return {"status": "ok", "actions": env.legal_actions}
        elif action == "shutdown":
            return {"status": "ok"}
        else:
            return {"status": "error", "message": f"Unknown: {action}"}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    env = AtariEnv(rom_path=sys.argv[1] if len(sys.argv) > 1 else None)

    # Signal readiness over serial (host watches for this)
    try:
        with open("/dev/ttyS0", "w") as serial:
            serial.write("AGENT_READY\n")
            serial.flush()
    except Exception:
        pass

    # Listen on vsock
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((VMADDR_CID_ANY, VSOCK_PORT))
    sock.listen(5)

    while True:
        conn, addr = sock.accept()
        try:
            handle_connection(conn, env)
        except Exception:
            pass
        finally:
            conn.close()


def handle_connection(conn, env):
    """Handle one client connection — read commands, send responses."""
    buf = b""
    while True:
        data = conn.recv(65536)
        if not data:
            break
        buf += data

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                cmd = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            response = dispatch(env, cmd)
            resp_bytes = json.dumps(response).encode("utf-8") + b"\n"
            conn.sendall(resp_bytes)

            if cmd.get("cmd") == "shutdown":
                return


if __name__ == "__main__":
    main()
