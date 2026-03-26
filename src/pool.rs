use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::atari_client::{AtariClient, StepResult};
use crate::snapshot::Snapshot;
use crate::vm::FirecrackerVM;

/// Result of forking a VM into multiple children.
///
/// Owns the child VMs and snapshot. If dropped without calling
/// `VMPool::select_and_cleanup`, all children are destroyed and
/// snapshot files are removed automatically.
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
        self.snapshot.cleanup();
    }
}

/// Manages parallel VM forking operations.
///
/// Handles the lifecycle of snapshot directories and child VM
/// work directories under `base_dir`.
pub struct VMPool {
    pub base_dir: PathBuf,
    fork_counter: AtomicU64,
}

impl VMPool {
    pub async fn try_new(base_dir: PathBuf) -> Result<Self> {
        tokio::fs::remove_dir_all(&base_dir).await?;
        tokio::fs::create_dir_all(&base_dir).await?;
        Ok(Self {
            base_dir: std::fs::canonicalize(&base_dir)?,
            fork_counter: AtomicU64::new(0),
        })
    }

    /// Fork a running VM into `num_children` children.
    ///
    /// 1. Pause the parent
    /// 2. Create a full snapshot (vmstate + memory)
    /// 3. Spawn `num_children` VMs from the snapshot in parallel
    ///
    /// Each child shares the same read-only rootfs and gets its own
    /// CoW memory pages via MAP_PRIVATE on the memory snapshot file.
    pub async fn fork(
        &self,
        parent: &mut FirecrackerVM,
        num_children: usize,
    ) -> Result<ForkResult> {
        let t0 = std::time::Instant::now();
        let fork_id = self.fork_counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Pause and snapshot
        parent.pause().await?;
        let snap_dir = self.base_dir.join(format!("snap-{fork_id}"));
        let snapshot = parent.create_snapshot(&snap_dir).await?;

        // Spawn all children concurrently via tokio tasks
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
            // Clean up any successfully spawned children
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
    /// Also cleans up the snapshot files and child work directories.
    pub fn select_and_cleanup(
        &self,
        mut result: ForkResult,
        selected_index: usize,
    ) -> FirecrackerVM {
        // Take the selected child out before dropping the rest
        let selected = result.children.remove(selected_index);

        // Destroy remaining children and clean up their directories
        for mut child in result.children.drain(..) {
            let dir = child.work_dir.clone();
            child.destroy();
            let _ = std::fs::remove_dir_all(&dir);
        }

        // Clean up snapshot files.
        // ForkResult's Drop will run after this, but it's a no-op:
        // children vec is drained, and snapshot.cleanup() is
        // idempotent (ignores already-removed files).
        result.snapshot.cleanup();

        selected
    }
}

impl Drop for VMPool {
    fn drop(&mut self) {
        // Best-effort cleanup of the pool's base directory.
        // Any remaining child dirs or snapshot dirs get removed.
        if self.base_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.base_dir);
        }
    }
}
