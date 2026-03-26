use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::atari_client::{AtariClient, StepResult};
use crate::snapshot::Snapshot;
use crate::vm::FirecrackerVM;

/// Result of forking a VM into multiple children.
///
/// Owns the child VMs and snapshot metadata. If dropped without
/// calling `VMPool::select_and_cleanup`, all children are destroyed.
/// The memory file is NOT cleaned up here — it's owned by the pool
/// and reused across iterations for diff snapshots.
pub struct ForkResult {
    pub children: Vec<FirecrackerVM>,
    pub snapshot: Snapshot,
    pub fork_time_ms: f64,
}

impl Drop for ForkResult {
    fn drop(&mut self) {
        for child in &mut self.children {
            child.destroy();
        }
        // Only clean vmstate — memory file is managed by the pool.
        self.snapshot.cleanup_vmstate_only();
    }
}

/// Manages parallel VM forking with diff-snapshot acceleration.
///
/// Maintains a persistent base memory file under `base_dir`. The
/// first fork creates a full snapshot (writing all guest memory).
/// Subsequent forks create diff snapshots: Firecracker only writes
/// pages dirtied since the last snapshot into the same memory file,
/// overwriting those offsets in-place. Clean pages retain their
/// content from previous iterations.
///
/// This works because:
///   - Children mmap the file with MAP_PRIVATE (CoW). Modifying the
///     file after they've mapped it doesn't affect their pages — the
///     kernel already copied them on first write.
///   - Between iterations, all children except the selected one are
///     destroyed. The selected child has its own CoW pages. So it's
///     safe to overwrite the memory file for the next diff.
///   - Firecracker's dirty page tracking (enabled via
///     `enable_diff_snapshots: true` on restore) resets the bitmap
///     after each snapshot, so each diff captures exactly one
///     iteration's worth of changes.
pub struct VMPool {
    pub base_dir: PathBuf,
    fork_counter: AtomicU64,
    /// Persistent memory file reused across diff snapshots.
    /// None before the first fork, Some after.
    base_memory_path: Option<PathBuf>,
}

impl VMPool {
    pub async fn try_new(base_dir: PathBuf) -> Result<Self> {
        // Ignore the error if the dir doesn't exist yet.
        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        tokio::fs::create_dir_all(&base_dir).await?;
        Ok(Self {
            base_dir: std::fs::canonicalize(&base_dir)?,
            fork_counter: AtomicU64::new(0),
            base_memory_path: None,
        })
    }

    /// Fork a running VM into `num_children` children.
    ///
    /// First call: full snapshot (writes all guest memory).
    /// Subsequent calls: diff snapshot (writes only dirty pages into
    /// the existing memory file).
    ///
    /// 1. Pause the parent
    /// 2. Create a full or diff snapshot
    /// 3. Spawn `num_children` VMs from the snapshot in parallel
    pub async fn fork(
        &mut self,
        parent: &mut FirecrackerVM,
        num_children: usize,
    ) -> Result<ForkResult> {
        let t0 = std::time::Instant::now();
        let fork_id = self.fork_counter.fetch_add(1, Ordering::SeqCst) + 1;

        parent.pause().await?;

        let snap_dir = self.base_dir.join(format!("snap-{fork_id}"));
        let is_diff = self.base_memory_path.is_some();

        let snapshot = parent
            .create_snapshot(&snap_dir, self.base_memory_path.as_deref(), is_diff)
            .await?;

        // After the first full snapshot, remember the memory path
        // for subsequent diffs to write into.
        if self.base_memory_path.is_none() {
            self.base_memory_path = Some(snapshot.memory_path.clone());
        }

        let children = self
            .spawn_children(&snapshot, fork_id, num_children)
            .await?;

        Ok(ForkResult {
            children,
            snapshot,
            fork_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn spawn_children(
        &self,
        snapshot: &Snapshot,
        fork_id: u64,
        num_children: usize,
    ) -> Result<Vec<FirecrackerVM>> {
        let mut handles = Vec::with_capacity(num_children);

        for i in 0..num_children {
            let child_id = format!("fork{fork_id}-child{i}");
            let child_dir = self.base_dir.join(&child_id);
            let snap = snapshot.clone();

            handles.push(tokio::spawn(async move {
                FirecrackerVM::restore_from_snapshot(&snap, child_dir, &child_id, true).await
            }));
        }

        let mut children = Vec::with_capacity(num_children);
        let mut errors = Vec::new();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(vm)) => children.push(vm),
                Ok(Err(e)) => errors.push(format!("child {i}: {e}")),
                Err(e) => errors.push(format!("child {i}: task panicked: {e}")),
            }
        }

        if !errors.is_empty() {
            for mut child in children {
                child.destroy();
            }
            bail!(
                "Failed to spawn {} of {} children:\n{}",
                errors.len(),
                num_children,
                errors.join("\n")
            );
        }

        Ok(children)
    }

    /// Step all children in parallel, each with a different action.
    ///
    /// Connects to each child's vsock concurrently and sends a step
    /// command. Returns results in the same order as `actions`.
    pub async fn step_all(children: &[FirecrackerVM], actions: &[i64]) -> Result<Vec<StepResult>> {
        assert_eq!(children.len(), actions.len());

        let mut handles = Vec::with_capacity(children.len());

        for (child, &action) in children.iter().zip(actions.iter()) {
            let vsock_path = child.vsock_uds_path();
            handles.push(tokio::spawn(async move {
                let mut client = AtariClient::connect(&vsock_path, Duration::from_secs(10)).await?;
                client.step(action).await
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        let mut errors = Vec::new();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => errors.push(format!("child {i}: {e}")),
                Err(e) => errors.push(format!("child {i}: task panicked: {e}")),
            }
        }

        if !errors.is_empty() {
            bail!(
                "Failed to step {} children:\n{}",
                errors.len(),
                errors.join("\n")
            );
        }

        Ok(results)
    }

    /// Select one child as the new "current" VM, destroy the rest.
    ///
    /// Kills unselected Firecracker processes immediately (SIGKILL),
    /// then moves filesystem cleanup to a background task. The base
    /// memory file is preserved for the next diff snapshot.
    pub fn select_and_cleanup(
        &self,
        mut result: ForkResult,
        selected_index: usize,
    ) -> FirecrackerVM {
        let selected = result.children.remove(selected_index);

        // SIGKILL all unselected children — non-blocking syscall.
        let mut dirs_to_remove: Vec<PathBuf> = Vec::with_capacity(result.children.len());
        for mut child in result.children.drain(..) {
            dirs_to_remove.push(child.work_dir.clone());
            child.destroy();
        }

        // Clean vmstate only — memory file is reused for diffs.
        let snap_vmstate = result.snapshot.vmstate_path.clone();

        // Move filesystem cleanup off the hot path.
        tokio::task::spawn_blocking(move || {
            let _ = std::fs::remove_file(&snap_vmstate);
            if let Some(parent) = snap_vmstate.parent() {
                let _ = std::fs::remove_dir(parent);
            }
            for dir in dirs_to_remove {
                let _ = std::fs::remove_dir_all(&dir);
            }
        });

        selected
    }
}

impl Drop for VMPool {
    fn drop(&mut self) {
        if self.base_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.base_dir);
        }
    }
}
