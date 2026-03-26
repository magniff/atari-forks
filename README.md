# Atari Forking Challenge

A Rust library for snapshotting and forking Firecracker microVMs, with a
benchmark that performs tree search over an Atari Breakout environment.

## What this does

Runs Atari Breakout inside a Firecracker microVM, snapshots the VM state,
forks it into 4 children (one per legal action), steps each child in
parallel, selects a random successor as the new root, and repeats.
Produces `1 + 4n` observation frames.

```
     ┌──────────┐
     │  Root VM │  (Breakout, initial state)
     │  obs_0   │
     └────┬─────┘
          │ pause + diff snapshot
    ┌─────┼─────┬─────┐
    ▼     ▼     ▼     ▼
 NOOP   FIRE  RIGHT  LEFT     ← 4 pre-warmed children, each takes a different action
 obs_1  obs_2  obs_3  obs_4

          │ randomly pick one
          ▼
    ┌─────┼─────┬─────┐       ← repeat n times
    ▼     ▼     ▼     ▼
```

## Project structure

```
atari-fork/
├── src/
│   ├── main.rs              # Benchmark entry point
│   ├── vm.rs                # FirecrackerVM: single VM lifecycle
│   ├── snapshot.rs          # Snapshot: captured VM state
│   ├── pool.rs              # VMPool: fork orchestration + diff snapshots
│   ├── process_pool.rs      # ProcessPool: pre-warmed FC processes
│   └── atari_client.rs      # Host-side vsock client for the guest agent
├── guest/
│   └── atari_agent.py       # Guest-side agent (runs inside the VM as PID 1)
├── setup.sh                 # Downloads Firecracker, kernel, builds rootfs
├── Cargo.toml
└── README.md
```

## Quick start

### Prerequisites

- Linux with KVM support (`ls -la /dev/kvm`)
- Rust toolchain (stable)
- Docker (for rootfs building, or `debootstrap` + sudo)

### Setup

```bash
./setup.sh
```

Downloads Firecracker v1.10.1, a Linux 6.1 guest kernel, and builds a
Debian rootfs with Python, ALE, gymnasium, pillow, and numpy.

### Build and run

```bash
cargo build --release

./target/release/tree-search \
    --firecracker ./deps/firecracker \
    --kernel ./deps/vmlinux \
    --rootfs ./deps/rootfs.ext4 \
    --iterations 100 \
    --pool-dir /dev/shm/vm-pool
```

### Options

```
--firecracker PATH   Path to Firecracker binary (default: firecracker)
--kernel PATH        Path to guest kernel image
--rootfs PATH        Path to rootfs ext4 image
--iterations N       Number of fork-step-select cycles (default: 100)
--output DIR         Where to save observation frames (default: ./frames)
--vcpus N            vCPUs per VM (default: 1)
--mem N              Guest memory in MiB (default: 128)
--seed N             Random seed for child selection (default: 42)
--pool-dir DIR       Work directory for snapshots and child VMs
                     (default: ./vm-pool, recommend /dev/shm for tmpfs)
```

## Architecture

### Library layers

**`FirecrackerVM`** (`vm.rs`) — Manages one Firecracker OS process.
Handles boot (configure machine, kernel, rootfs, vsock, start), pause,
snapshot creation (full or diff), and snapshot restore. Communicates
with Firecracker over its Unix socket HTTP API using hyper.

**`Snapshot`** (`snapshot.rs`) — A captured VM state: vmstate file +
memory file + rootfs path + config. Immutable after creation — multiple
children restore from the same snapshot concurrently via `MAP_PRIVATE`.

**`VMPool`** (`pool.rs`) — Orchestrates parallel fork operations with
two key optimizations: diff snapshots that reuse a persistent base
memory file (writing only dirty pages each iteration), and a process
pool for pre-warmed Firecracker processes.

**`ProcessPool`** (`process_pool.rs`) — Maintains a buffer of
pre-spawned Firecracker processes with their API sockets ready.
Replenishes consumed slots in the background while the current
iteration's children are stepping the environment.

**`AtariClient`** (`atari_client.rs`) — Host-side client for the
guest agent. Connects via Firecracker's vsock AF_UNIX proxy with a
health-check probe to handle post-restore connection races.

### Communication

The guest agent (`atari_agent.py`) communicates over **virtio-vsock**
(`AF_VSOCK`, `SOCK_STREAM`) — a reliable byte stream transported via
virtio DMA ring buffers. Much faster than serial. The host connects
through Firecracker's AF_UNIX vsock proxy with a `CONNECT <port>\n`
handshake.

Protocol is newline-delimited JSON:

```
Host → Guest:  {"cmd": "reset"} | {"cmd": "step", "action": N} | {"cmd": "get_actions"}
Guest → Host:  {"status": "ok", "obs": "<base64 png>", "reward": 0.0, "done": false}
```

The agent signals readiness by writing `AGENT_READY` to `/dev/ttyS0`
(serial console), which the host monitors on Firecracker's stdout pipe.

### Guest init

The VM boots with `init=/opt/init.sh` — a minimal script that mounts
`/proc`, `/sys`, `/dev`, `/tmp`, `/var`, sets `PYTHONPATH`, and execs
the agent as PID 1. No systemd, no init system, no module loading.

### Key design decisions

- **Relative vsock `uds_path`** (`"v.sock"`): The snapshot bakes the
  vsock path. Using a relative path with per-VM `cwd` ensures restored
  children don't collide on the host socket.
- **`random.trust_cpu=on`**: Seeds the kernel entropy pool from RDRAND
  at boot, preventing the guest from blocking on `/dev/random`.
- **`mitigations=off`**: Disables Spectre/Meltdown mitigations in the
  guest kernel — not needed for a benchmark workload, saves cycles.
- **`MAP_PRIVATE` memory sharing**: Children restored from the same
  snapshot share the same physical pages until they write. Each write
  triggers a kernel CoW fault, giving each child its own copy of only
  the pages it modifies.

## How forking works

Each iteration:

1. **Pause** the parent VM (`PATCH /vm {"state": "Paused"}`)
2. **Diff snapshot** — Firecracker writes only pages dirtied since the
   last snapshot into the persistent base memory file (~1-2 MB vs 128 MB
   for a full dump). Dirty page tracking via KVM's bitmap, reset after
   each snapshot.
3. **Take pre-warmed processes** from the process pool — 4 Firecracker
   processes already spawned with API sockets ready
4. **Load snapshot** into each process in parallel (`PUT /snapshot/load`)
   — Firecracker mmaps the memory file with `MAP_PRIVATE`, sets up KVM
   vCPU state, configures virtio devices
5. **Step** each child over vsock — connect, health-check probe, send
   action, receive observation
6. **Select** one child as the new root, SIGKILL the rest, clean up
   directories in a background thread

The process pool replenishes in the background while step 5 executes,
so fresh processes are ready before the next iteration's fork.

## Performance

Typical numbers on a modern x86_64 machine with `--pool-dir /dev/shm/vm-pool`:

| Metric                | Time        |
| --------------------- | ----------- |
| Boot + ALE init       | ~500-600 ms |
| Fork (4 children)     | ~56 ms      |
| Step all 4 (parallel) | ~6 ms       |
| Total per iteration   | ~56 ms      |
| 1000 iterations       | ~56 s       |

### Optimizations applied

| Optimization                                            | Effect                                    |
| ------------------------------------------------------- | ----------------------------------------- |
| `sync_file_range(SYNC_FILE_RANGE_WRITE)` after snapshot | Prevents dirty page accumulation stalls   |
| Async filesystem cleanup via `spawn_blocking`           | Unlinks don't block the fork path         |
| Early parent kill after fork                            | Frees MAP_PRIVATE mappings sooner         |
| tmpfs (`/dev/shm`) for pool directory                   | Eliminates block I/O entirely             |
| Diff snapshots with persistent base memory file         | ~1-2 MB written per snapshot vs 128 MB    |
| Pre-warmed process pool with background replenishment   | Eliminates ~25 ms spawn overhead per fork |

### Further optimization ideas

- **Skip PNG encoding** in the guest — send raw RGB bytes over vsock,
  encode on the host in Rust with the `png` crate
- **Persistent HTTP connections** to Firecracker API — skip per-call
  handshake overhead
- **Custom VMM with `fork()`** — clone the VMM process directly, let
  Linux CoW handle memory sharing without snapshot files at all

## Extending

The VM management library is environment-agnostic. Replace
`atari_client.rs` and `atari_agent.py` with your own guest agent and
host client to fork arbitrary environments:

```rust
let mut vm = FirecrackerVM::new(config, None, "root")?;
vm.boot().await?;
// ... set up your environment ...

let mut pool = VMPool::try_new(pool_dir, &config, num_children).await?;

let fork_result = pool.fork(&mut vm, num_children).await?;
// fork_result.children: Vec<FirecrackerVM>, each restored from the same state

let selected = pool.select_and_cleanup(fork_result, chosen_index);
// selected is now the new root VM for the next iteration
```
