use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::atari_client::{AtariClient, StepResult};
use crate::process_pool::ProcessPool;
use crate::snapshot::Snapshot;
use crate::vm::{FirecrackerVM, VMConfig};

/// Result of forking a VM into multiple children.
///
/// Owns the child VMs and snapshot metadata. If dropped without
/// calling `VMPool::select_and_cleanup`, all children are destroyed.
/// The memory file is NOT cleaned up here — it's owned by the pool
/// and reused across iterations for diff snapshots.
pub struct ForkResult {
    pub children: Vec<FirecrackerVM>,
    pub snapshot: Snapshot,
}

impl Drop for ForkResult {
    fn drop(&mut self) {
        for child in &mut self.children {
            child.destroy();
        }
        self.snapshot.cleanup_vmstate();
    }
}

/// Manages parallel VM forking with diff snapshots and pre-warmed processes.
///
/// Combines two optimizations:
///
/// 1. **Diff snapshots**: After the first full snapshot, subsequent
///    forks write only dirty pages (~1-2MB vs 128MB).
///
/// 2. **Process pool**: Firecracker processes are pre-spawned in the
///    background. When a fork needs children, it grabs already-running
///    processes and sends a single `/snapshot/load` API call — skipping
///    the ~30-40ms spawn+init overhead per child.
///
/// Together these reduce the fork critical path to roughly:
///   pause (~1ms) + diff snapshot (~3-5ms) + snapshot load × N in
///   parallel (~15-20ms) ≈ 20-25ms total.
pub struct VMScheduler {
    pub base_dir: PathBuf,
    fork_counter: AtomicU64,
    /// Persistent memory file reused across diff snapshots.
    base_memory_path: Option<PathBuf>,
    /// Pre-spawned Firecracker processes.
    process_pool: ProcessPool,
}

impl VMScheduler {
    /// Create a new VMPool with a pre-warmed process pool.
    ///
    /// `pool_size` controls how many Firecracker processes to keep
    /// warm. Set this to the number of children per fork (typically
    /// the action count).
    pub async fn try_new(base_dir: PathBuf, config: &VMConfig, pool_size: usize) -> Result<Self> {
        let _ = tokio::fs::remove_dir_all(&base_dir).await;
        tokio::fs::create_dir_all(&base_dir).await?;
        let base_dir = std::fs::canonicalize(&base_dir)?;

        let proc_pool_dir = base_dir.join("proc-pool");
        let process_pool = ProcessPool::new(config.clone(), proc_pool_dir, pool_size).await?;

        Ok(Self {
            base_dir,
            fork_counter: AtomicU64::new(0),
            base_memory_path: None,
            process_pool,
        })
    }

    /// Fork a running VM into `num_children` children.
    ///
    /// 1. Pause the parent
    /// 2. Create a full or diff snapshot
    /// 3. Grab pre-spawned processes from the pool
    /// 4. Load the snapshot into each process in parallel
    ///
    /// Steps 3-4 are fast because the processes are already running
    /// and just need a single `/snapshot/load` API call each.
    pub async fn fork(
        &mut self,
        parent: &mut FirecrackerVM,
        num_children: usize,
    ) -> Result<ForkResult> {
        let fork_id = self.fork_counter.fetch_add(1, Ordering::SeqCst) + 1;

        parent.pause().await?;

        let snap_dir = self.base_dir.join(format!("snap-{fork_id}"));
        let is_diff = self.base_memory_path.is_some();

        let snapshot = parent
            .create_snapshot(&snap_dir, self.base_memory_path.as_deref(), is_diff)
            .await?;

        if self.base_memory_path.is_none() {
            self.base_memory_path = Some(snapshot.memory_path.clone());
        }

        // Grab pre-warmed processes — this also kicks off background
        // replenishment for the next iteration.
        let processes = self.process_pool.take(num_children).await?;

        // Load the snapshot into each process in parallel.
        let children = self
            .load_snapshot_into_processes(processes, &snapshot)
            .await?;

        Ok(ForkResult { children, snapshot })
    }

    /// Load a snapshot into pre-spawned processes in parallel.
    async fn load_snapshot_into_processes(
        &self,
        processes: Vec<FirecrackerVM>,
        snapshot: &Snapshot,
    ) -> Result<Vec<FirecrackerVM>> {
        let mut handles = Vec::with_capacity(processes.len());

        for mut vm in processes {
            let snap = snapshot.clone();
            handles.push(tokio::spawn(async move {
                vm.load_snapshot(&snap, true).await?;
                Ok::<_, anyhow::Error>(vm)
            }));
        }

        let mut children = Vec::with_capacity(handles.len());
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
                "Failed to load snapshot into {} processes:\n{}",
                errors.len(),
                errors.join("\n")
            );
        }

        Ok(children)
    }

    /// Step all children in parallel, each with a different action.
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
    pub fn select_and_cleanup(
        &self,
        mut result: ForkResult,
        selected_index: usize,
    ) -> FirecrackerVM {
        let selected = result.children.remove(selected_index);

        let mut dirs_to_remove: Vec<PathBuf> = Vec::with_capacity(result.children.len());
        for mut child in result.children.drain(..) {
            dirs_to_remove.push(child.work_dir.clone());
            child.destroy();
        }

        let snap_vmstate = result.snapshot.vmstate_path.clone();

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

impl Drop for VMScheduler {
    fn drop(&mut self) {
        if self.base_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.base_dir);
        }
    }
}
