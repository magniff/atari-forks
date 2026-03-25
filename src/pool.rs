use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::snapshot::Snapshot;
use crate::vm::FirecrackerVM;

pub struct ForkResult {
    pub children: Vec<FirecrackerVM>,
    pub snapshot: Snapshot,
    pub fork_time_ms: f64,
}

/// Manages parallel VM forking operations.
pub struct VMPool {
    pub base_dir: PathBuf,
    pub max_parallel: usize,
    fork_counter: AtomicU64,
}

impl VMPool {
    pub fn new(base_dir: PathBuf, max_parallel: usize) -> Result<Self> {
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self {
            base_dir: std::fs::canonicalize(&base_dir)?,
            max_parallel,
            fork_counter: AtomicU64::new(0),
        })
    }

    /// Fork a running VM into multiple children.
    ///
    /// 1. Pause the parent
    /// 2. Create a snapshot
    /// 3. Spawn num_children VMs from the snapshot, in parallel
    pub fn fork(&self, parent: &mut FirecrackerVM, num_children: usize) -> Result<ForkResult> {
        let t0 = std::time::Instant::now();
        let fork_id = self.fork_counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Pause and snapshot
        parent.pause()?;
        let snap_dir = self.base_dir.join(format!("snap-{fork_id}"));
        let snapshot = parent.create_snapshot(&snap_dir)?;

        // Spawn children in parallel using std::thread
        let children = self.spawn_children(&snapshot, fork_id, num_children)?;

        Ok(ForkResult {
            children,
            snapshot,
            fork_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn spawn_children(
        &self,
        snapshot: &Snapshot,
        fork_id: u64,
        num_children: usize,
    ) -> Result<Vec<FirecrackerVM>> {
        use std::thread;

        // Prepare child configs
        let child_configs: Vec<_> = (0..num_children)
            .map(|i| {
                let child_id = format!("fork{fork_id}-child{i}");
                let child_dir = self.base_dir.join(&child_id);
                (child_id, child_dir)
            })
            .collect();

        // No rootfs copy needed — rootfs is read-only and shared across all VMs.
        // This eliminates the ~300-400ms per fork that was spent copying the 1GB image.

        // Spawn VMs in parallel
        let handles: Vec<_> = child_configs
            .into_iter()
            .map(|(child_id, child_dir)| {
                let snap = snapshot.clone();
                thread::spawn(move || {
                    FirecrackerVM::restore_from_snapshot(
                        &snap, child_dir, &child_id,
                        None, // use snapshot's rootfs directly (read-only, shared)
                        true,
                    )
                })
            })
            .collect();

        let mut children = Vec::with_capacity(num_children);
        let mut errors = Vec::new();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(vm)) => children.push(vm),
                Ok(Err(e)) => errors.push((i, e)),
                Err(_) => errors.push((i, anyhow::anyhow!("thread panicked"))),
            }
        }

        if !errors.is_empty() {
            // Clean up any children that did start
            for mut child in children {
                child.destroy();
            }
            let msg: Vec<_> = errors
                .iter()
                .map(|(i, e)| format!("child {i}: {e}"))
                .collect();
            bail!(
                "Failed to spawn {} children:\n{}",
                errors.len(),
                msg.join("\n")
            );
        }

        Ok(children)
    }

    /// Select one child, destroy the rest, clean up snapshot.
    pub fn select_and_cleanup(
        &self,
        mut result: ForkResult,
        selected_index: usize,
    ) -> FirecrackerVM {
        let selected = result.children.remove(selected_index);

        for mut child in result.children {
            let dir = child.work_dir.clone();
            child.destroy();
            let _ = std::fs::remove_dir_all(&dir);
        }

        result.snapshot.cleanup();
        selected
    }
}
