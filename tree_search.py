#!/usr/bin/env python3
"""
Atari Forking Tree Search

This script implements the tree search benchmark described in the challenge:

    1. Start at the root with a freshly initialized Breakout environment
    2. Perform all 4 actions at the root in parallel (fork the VM 4 ways)
    3. Select a random successor as the new root
    4. Repeat: fork the new root into 4 children, take actions, pick one
    5. Save all observation frames to a directory

The result is 1 + 4n frames total, where n is the number of tree iterations.
The initial frame comes from resetting the environment, and each iteration
produces 4 frames (one per action).

Architecture:
    ┌─────────────┐
    │  Root VM     │  (running Atari Breakout via guest agent)
    │  obs_0       │
    └──────┬──────┘
           │ pause + snapshot
           ├──────────────────────────────┐
    ┌──────┴──────┐  ┌──────┴──────┐     ... (4 children total)
    │ Child 0     │  │ Child 1     │
    │ action=NOOP │  │ action=FIRE │
    │ obs_1       │  │ obs_2       │
    └─────────────┘  └──────┬──────┘  ← randomly selected
                            │ becomes new root
                            │ pause + snapshot
                     ┌──────┴──────┐  ...
                     │ Child 0     │
                     │ action=NOOP │
                     │ obs_5       │
                     └─────────────┘

Performance notes:
    - Each fork operation involves: pause (~1ms) + snapshot (~10-50ms depending
      on dirty pages) + 4× restore (~30ms each, parallelized) + 4× connect (~5ms)
    - Total per iteration: ~80-150ms with parallelism
    - The main bottleneck is the rootfs copy (unless using reflink/btrfs)
    - Memory pressure scales with snapshot_mem × num_concurrent_children
"""

from __future__ import annotations
from firecracker_vm.atari_client import AtariClient
from firecracker_vm import FirecrackerVM, VMConfig, VMPool

import argparse
import os
import random
import sys
import time
from pathlib import Path

# Add parent dir to path so we can import our library
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def save_frame(frame_data: bytes, path: str) -> None:
    """Save a PNG observation frame to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(frame_data)


def run_tree_search(
    config: VMConfig,
    num_iterations: int = 10,
    output_dir: str = "./frames",
    seed: int = 42,
) -> dict:
    """
    Run the forking tree search benchmark.

    Args:
        config:         VM configuration (kernel, rootfs, etc.)
        num_iterations: Number of fork-step-select iterations
        output_dir:     Directory to save observation frames
        seed:           Random seed for child selection

    Returns:
        Dictionary with timing statistics
    """
    random.seed(seed)
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    # Track the action history for each saved frame
    # Maps filename -> list of actions taken to reach that state
    frame_history: dict[str, list[int]] = {}
    # The action path leading to the current root state
    current_path: list[int] = []

    stats = {
        "iteration_times_ms": [],
        "fork_times_ms": [],
        "step_times_ms": [],
        "total_frames": 0,
    }

    print(f"Starting tree search: {num_iterations} iterations")
    print(f"Output directory: {frames_dir}")
    print(f"Expected frames: {1 + 4 * num_iterations}")
    print()

    # --- Phase 1: Boot the root VM and reset the environment ---
    print("Booting root VM...")
    t_boot = time.monotonic()

    root_vm = FirecrackerVM(config, vm_id="root")
    root_vm.boot()

    # Connect to the guest agent over vsock
    client = AtariClient(root_vm.vsock_uds_path, root_vm.serial_stdout)
    client.connect()

    # Get legal actions (should be 4 for Breakout: NOOP, FIRE, RIGHT, LEFT)
    actions = client.get_legal_actions()
    num_actions = len(actions)
    print(f"Legal actions: {actions} ({num_actions} total)")

    # Reset environment and save initial observation
    initial_obs = client.reset()
    frame_name = f"frame_{frame_count:06d}.png"
    save_frame(initial_obs, str(frames_dir / frame_name))
    frame_history[frame_name] = []  # root frame, no actions taken
    frame_count += 1
    client.close()

    print(f"Boot + reset took {(time.monotonic() - t_boot)*1000:.1f}ms")
    print()

    # --- Phase 2: Tree search iterations ---
    pool = VMPool(
        base_dir=str(frames_dir.parent / "vm-pool"),
        max_parallel=num_actions,
    )

    current_vm = root_vm

    try:
        for iteration in range(num_iterations):
            t_iter = time.monotonic()
            print(f"--- Iteration {iteration + 1}/{num_iterations} ---")

            # Fork the current VM into num_actions children
            t_fork = time.monotonic()
            fork_result = pool.fork(current_vm, num_children=num_actions)
            fork_ms = (time.monotonic() - t_fork) * 1000
            stats["fork_times_ms"].append(fork_ms)
            print(f"  Fork: {fork_ms:.1f}ms")

            # Step each child with a different action and collect frames
            t_step = time.monotonic()
            child_frames = []

            for child_idx, (child_vm, action) in enumerate(
                zip(fork_result.children, actions)
            ):
                child_client = AtariClient(
                    child_vm.vsock_uds_path, child_vm.serial_stdout,
                )
                child_client.connect(skip_ready=True)

                obs, reward, done, info = child_client.step(action)
                child_frames.append(obs)

                # Save frame with history
                frame_name = f"frame_{frame_count:06d}.png"
                save_frame(obs, str(frames_dir / frame_name))
                frame_history[frame_name] = current_path + [action]
                frame_count += 1

                child_client.close()

                print(f"  Action {action}: reward={reward:.1f}, "
                      f"done={done}")

            step_ms = (time.monotonic() - t_step) * 1000
            stats["step_times_ms"].append(step_ms)
            print(f"  Step all: {step_ms:.1f}ms")

            # Select a random child as the new root
            selected_idx = random.randint(0, num_actions - 1)
            selected_action = actions[selected_idx]
            print(f"  Selected child {selected_idx} "
                  f"(action {selected_action})")

            # Update the current path — the selected child's action
            # becomes part of the path for all future frames
            current_path = current_path + [selected_action]

            # Destroy the current parent VM (no longer needed)
            current_vm.destroy()

            # The selected child becomes the new current VM
            current_vm = pool.select_and_cleanup(fork_result, selected_idx)

            iter_ms = (time.monotonic() - t_iter) * 1000
            stats["iteration_times_ms"].append(iter_ms)
            print(f"  Total iteration: {iter_ms:.1f}ms")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        current_vm.destroy()
        pool.shutdown()

    # Save frame history
    import json
    history_path = frames_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(frame_history, f, indent=2)
    print(f"Frame history saved to {history_path}")

    stats["total_frames"] = frame_count
    return stats


def print_stats(stats: dict) -> None:
    """Print summary statistics."""
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total frames saved: {stats['total_frames']}")

    if stats["iteration_times_ms"]:
        times = stats["iteration_times_ms"]
        print(f"Iterations: {len(times)}")
        print(f"  Mean iteration time: {sum(times)/len(times):.1f}ms")
        print(f"  Min:  {min(times):.1f}ms")
        print(f"  Max:  {max(times):.1f}ms")

    if stats["fork_times_ms"]:
        times = stats["fork_times_ms"]
        print(f"Fork times:")
        print(f"  Mean: {sum(times)/len(times):.1f}ms")
        print(f"  Min:  {min(times):.1f}ms")
        print(f"  Max:  {max(times):.1f}ms")

    if stats["step_times_ms"]:
        times = stats["step_times_ms"]
        print(f"Step times (all {len(times)} children):")
        print(f"  Mean: {sum(times)/len(times):.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Atari Forking Tree Search Benchmark"
    )
    parser.add_argument(
        "--firecracker", default="firecracker",
        help="Path to firecracker binary"
    )
    parser.add_argument(
        "--kernel", required=True,
        help="Path to guest kernel (vmlinux)"
    )
    parser.add_argument(
        "--rootfs", required=True,
        help="Path to root filesystem image"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of tree search iterations (default: 10)"
    )
    parser.add_argument(
        "--output", default="./frames",
        help="Directory for output frames (default: ./frames)"
    )
    parser.add_argument(
        "--vcpus", type=int, default=1,
        help="Number of vCPUs per VM (default: 1)"
    )
    parser.add_argument(
        "--mem", type=int, default=256,
        help="Memory per VM in MiB (default: 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for child selection (default: 42)"
    )

    args = parser.parse_args()

    config = VMConfig(
        firecracker_bin=args.firecracker,
        kernel_path=args.kernel,
        rootfs_path=args.rootfs,
        vcpu_count=args.vcpus,
        mem_size_mib=args.mem,
    )

    stats = run_tree_search(
        config=config,
        num_iterations=args.iterations,
        output_dir=args.output,
        seed=args.seed,
    )

    print_stats(stats)


if __name__ == "__main__":
    main()
