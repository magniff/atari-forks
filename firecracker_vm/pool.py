"""
VM Pool for managing parallel fork operations.

This is the high-level orchestration layer. It handles the messy details of:
- Creating multiple child VMs from a single snapshot in parallel
- Managing working directories and rootfs copies for each child
- Cleaning up child VMs after they're no longer needed
- Providing a simple interface for the tree search algorithm

The forking workflow:
    1. Parent VM is running, reaches a decision point
    2. Parent pauses and creates a snapshot
    3. Pool spawns N child VMs from that snapshot (in parallel)
    4. Each child takes a different action and steps the environment
    5. One child is selected as the new "root"
    6. Other children are destroyed
    7. The selected child becomes the new parent, repeat from step 1

Performance considerations:
- Child VMs can be spawned in parallel (each is an independent Firecracker process)
- Memory is CoW via MAP_PRIVATE, so N children don't use N× the memory
- The rootfs copy is the bottleneck if reflinks aren't available
- For best performance, use a btrfs or xfs filesystem for the working directory
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from firecracker_vm.snapshot import Snapshot
from firecracker_vm.vm import FirecrackerVM, VMConfig


@dataclass
class ForkResult:
    """Result of forking a VM into multiple children."""
    children: list[FirecrackerVM]
    snapshot: Snapshot
    fork_time_ms: float  # Wall-clock time for the fork operation


class VMPool:
    """
    Manages a pool of Firecracker VMs for parallel tree search.

    Usage:
        pool = VMPool(base_dir="/tmp/vm-pool", max_parallel=4)

        # Fork a parent VM into 4 children
        result = pool.fork(parent_vm, num_children=4)

        # Do work with children...
        for child in result.children:
            send_action(child, ...)

        # Select one child as the new root, destroy the rest
        pool.select_and_cleanup(result, selected_index=2)
    """

    def __init__(self, base_dir: str, max_parallel: int = 4):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self._fork_counter = 0
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)

    def fork(self, parent: FirecrackerVM, num_children: int) -> ForkResult:
        """
        Fork a running VM into multiple children.

        Steps:
        1. Pause the parent
        2. Create a snapshot
        3. Resume the parent (it can keep running if needed)
        4. Spawn num_children VMs from the snapshot, in parallel

        Each child gets:
        - Its own Firecracker process
        - A CoW copy of the parent's memory (via MAP_PRIVATE)
        - A copy of the parent's rootfs (reflink if possible)
        - Its own working directory

        Returns a ForkResult with the children and timing info.
        """
        t0 = time.monotonic()
        self._fork_counter += 1
        fork_id = self._fork_counter

        # 1. Pause and snapshot
        parent.pause()
        snap_dir = str(self.base_dir / f"snap-{fork_id}")
        snapshot = parent.create_snapshot(snap_dir)

        # 2. Spawn children in parallel
        children = self._spawn_children(snapshot, fork_id, num_children)

        elapsed_ms = (time.monotonic() - t0) * 1000
        return ForkResult(
            children=children,
            snapshot=snapshot,
            fork_time_ms=elapsed_ms,
        )

    def fork_without_parent(self, snapshot: Snapshot,
                            num_children: int) -> ForkResult:
        """
        Fork from an existing snapshot without needing the parent VM.

        Useful when the parent has already been destroyed and you just
        have the snapshot files.
        """
        t0 = time.monotonic()
        self._fork_counter += 1
        fork_id = self._fork_counter

        children = self._spawn_children(snapshot, fork_id, num_children)

        elapsed_ms = (time.monotonic() - t0) * 1000
        return ForkResult(
            children=children,
            snapshot=snapshot,
            fork_time_ms=elapsed_ms,
        )

    def _spawn_children(self, snapshot: Snapshot, fork_id: int,
                        num_children: int) -> list[FirecrackerVM]:
        """
        Spawn multiple child VMs from a snapshot, in parallel.

        Each child gets its own directory, rootfs copy, and Firecracker process.
        """
        def spawn_one(child_idx: int) -> FirecrackerVM:
            child_id = f"fork{fork_id}-child{child_idx}"
            child_dir = self.base_dir / child_id

            # Each child needs its own rootfs copy
            child_rootfs = str(child_dir / "rootfs.ext4")
            snapshot.fork_rootfs(child_rootfs)

            return FirecrackerVM.restore_from_snapshot(
                snapshot=snapshot,
                work_dir=str(child_dir),
                vm_id=child_id,
                rootfs_path=child_rootfs,
                resume=True,
            )

        # Spawn in parallel using thread pool
        futures = {
            self._executor.submit(spawn_one, i): i
            for i in range(num_children)
        }

        children = [None] * num_children
        errors = []
        for future in as_completed(futures):
            idx = futures[future]
            try:
                children[idx] = future.result()
            except Exception as e:
                errors.append((idx, e))

        if errors:
            # Clean up any children that did start
            for child in children:
                if child is not None:
                    child.destroy()
            raise RuntimeError(
                f"Failed to spawn {len(errors)} children: {errors}"
            )

        return children

    def select_and_cleanup(self, fork_result: ForkResult,
                           selected_index: int) -> FirecrackerVM:
        """
        Select one child as the new root and destroy all others.

        The selected child continues running. All other children are
        killed and their working directories cleaned up.

        Returns the selected child VM.
        """
        selected = fork_result.children[selected_index]

        for i, child in enumerate(fork_result.children):
            if i != selected_index:
                child.destroy()
                # Clean up the child's working directory
                if child.work_dir.exists():
                    shutil.rmtree(child.work_dir, ignore_errors=True)

        # Clean up snapshot files (no longer needed)
        fork_result.snapshot.cleanup()

        return selected

    def cleanup_all(self, fork_result: ForkResult) -> None:
        """Destroy all children and clean up snapshot."""
        for child in fork_result.children:
            child.destroy()
            if child.work_dir.exists():
                shutil.rmtree(child.work_dir, ignore_errors=True)
        fork_result.snapshot.cleanup()

    def shutdown(self) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
