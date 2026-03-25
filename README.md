# Atari Forking Challenge

A library for snapshotting and forking Firecracker microVMs, with a demo
that performs tree search over an Atari Breakout environment.

## What This Does

Runs the Atari Breakout environment inside a Firecracker microVM, snapshots
the VM state, forks it into 4 children (one per action), steps each child,
selects a random successor, and repeats. Produces `1 + 4n` observation
frames showing the game state diverging across different action sequences.

```
     ┌──────────┐
     │  Root VM │  (Breakout, initial state)
     │  obs_0   │
     └────┬─────┘
          │ snapshot
    ┌─────┼─────┬─────┐
    ▼     ▼     ▼     ▼
 NOOP   FIRE  RIGHT  LEFT     ← 4 children, each takes a different action
 obs_1  obs_2  obs_3  obs_4

          │ randomly pick one
          ▼
    ┌─────┼─────┬─────┐       ← repeat n times
    ▼     ▼     ▼     ▼
```

## Project Structure

```
atari-fork/
├── firecracker_vm/          # Generic VM management library
│   ├── __init__.py          #   Re-exports
│   ├── vm.py                #   FirecrackerVM: single VM lifecycle
│   ├── snapshot.py          #   Snapshot: captured VM state
│   ├── pool.py              #   VMPool: parallel fork orchestration
│   └── atari_client.py      #   Host-side vsock client for Atari agent
├── guest/
│   └── atari_agent.py       #   Guest-side agent (runs inside the VM)
├── tree_search.py           #   Main benchmark script
├── setup.sh                 #   Downloads Firecracker, kernel, builds rootfs
└── README.md
```

## Quick Start

### Prerequisites

- Linux with KVM support (`ls -la /dev/kvm`)
- Python 3.10+
- Docker (for rootfs building)

### Setup

```bash
./setup.sh
```

This downloads:

- Firecracker v1.10.1
- Linux kernel 6.1.102 (with `random.trust_cpu=on` support)
- Builds a rootfs with Python, ALE, gymnasium, pillow, numpy

### Run

```bash
python3 tree_search.py \
    --firecracker ./deps/firecracker \
    --kernel ./deps/vmlinux \
    --rootfs ./deps/rootfs.ext4 \
    --iterations 10
```

Produces 41 frames (`1 + 4×10`) in `./frames/`.

### Options

```
--iterations N    Number of fork-step-select cycles (default: 10)
--output DIR      Where to save frames (default: ./frames)
--vcpus N         vCPUs per VM (default: 1)
--mem N           Memory per VM in MiB (default: 256)
--seed N          Random seed for child selection (default: 42)
```

## Architecture

### Library (`firecracker_vm/`)

Three layers, each with a single responsibility:

**`FirecrackerVM`** — Manages one Firecracker process. Handles boot,
pause/resume, snapshot creation, and snapshot restore. Each instance maps
to exactly one OS process. Uses relative vsock UDS paths so that restored
VMs don't collide on the host-side socket.

**`Snapshot`** — Represents captured VM state (vmstate file + memory file).
Immutable after creation — multiple VMs can restore from the same snapshot.
Handles rootfs copying with reflink support for CoW filesystems.

**`VMPool`** — Orchestrates parallel fork operations. `fork(parent, n)`
pauses the parent, snapshots it, and spawns n children in parallel.

### Communication

The guest agent communicates with the host over **vsock** (virtio socket),
which provides high-bandwidth virtio DMA transport. The host connects to
Firecracker's AF_UNIX vsock proxy with a `CONNECT <port>\n` handshake.

The agent signals readiness by writing `AGENT_READY` to the serial console,
which the host monitors on Firecracker's stdout.

### Guest Init

The VM boots with `init=/opt/init.sh` — a 5-line script that mounts
`/proc`, `/sys`, `/dev`, sets `PYTHONPATH`, and execs the agent directly
as PID 1. No systemd, no init system.

### Key Design Decisions

- **Relative vsock `uds_path`**: The snapshot bakes the vsock path. Using
  a relative path (`"v.sock"`) with per-VM `cwd` ensures restored children
  don't collide on the host socket.
- **`random.trust_cpu=on`**: Seeds the kernel entropy pool from RDRAND at
  boot, preventing ALE from blocking on `/dev/random` in the headless VM.
- **Kernel 6.1**: Required for the entropy fix (4.14 didn't support it).
- **PNG over vsock**: Atari frames are 210×160 with flat color regions —
  PNG compresses them to ~3-8KB, fast enough even for serial but instant
  over vsock.

## How Forking Works

Each iteration:

1. **Pause** — `ioctl(KVM_PAUSE_VCPU)` freezes all vCPUs
2. **Snapshot** — serialize CPU registers, device state, write dirty pages
3. **Restore** — new process, `mmap(memory_file, MAP_PRIVATE)` for CoW,
   restore KVM state. Each child gets copy-on-write memory pages.
4. **Step** — each child takes a different action, producing a different
   game frame
5. **Select** — one child becomes the new root, others are destroyed

## Performance

Typical numbers on a modern x86_64 machine:

- Boot + ALE init: ~500-600ms
- Fork (4 children): ~400-700ms (dominated by rootfs copy)
- Step all 4 children: ~15-25ms
- Total per iteration: ~500-800ms

### Optimization ideas

- **btrfs/xfs reflinks** for rootfs copy — near-instant CoW
- **Diff snapshots** with dirty page tracking — smaller memory files
- **tmpfs** for snapshot files — avoid disk I/O
- **Parallel steps** — step all children concurrently (currently sequential)

## Extending

The `firecracker_vm` library is environment-agnostic:

```python
from firecracker_vm import FirecrackerVM, VMConfig, VMPool

config = VMConfig(
    firecracker_bin="./firecracker",
    kernel_path="./vmlinux",
    rootfs_path="./my-rootfs.ext4",
)

vm = FirecrackerVM(config)
vm.boot()

pool = VMPool(base_dir="/tmp/pool")
result = pool.fork(vm, num_children=3)

# Each child is a full VM restored from the same point
for child in result.children:
    pass  # do work

selected = pool.select_and_cleanup(result, selected_index=1)
```

## License

MIT
