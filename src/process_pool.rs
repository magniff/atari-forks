use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::vm::{FirecrackerVM, VMConfig};

/// A pool of pre-spawned Firecracker processes ready for snapshot load.
///
/// Each pooled process has completed the expensive part of startup:
/// `fork()`+`exec()` of the Firecracker binary, VMM initialization,
/// API socket creation, and the initial HTTP health check. When a
/// snapshot load is needed, the caller grabs a process from the pool
/// and sends a single `/snapshot/load` API call — skipping ~30-40ms
/// of per-child spawn overhead.
///
/// The pool replenishes consumed slots in the background via
/// `tokio::spawn`, so new processes are warming up while the current
/// iteration's children are stepping the environment.
///
/// # Lifecycle
///
/// ```text
/// spawn_process()          load_snapshot()
///      │                        │
///      ▼                        ▼
/// ┌─────────┐  take()   ┌────────────┐  step()  ┌─────────┐
/// │  IDLE   │ ────────► │  LOADING   │ ───────► │ RUNNING │
/// │ (pool)  │           │ (snapshot) │          │ (child) │
/// └─────────┘           └────────────┘          └─────────┘
///      ▲                                              │
///      │              replenish()                      │
///      └──────────────────────────────────────── destroy()
/// ```
pub struct ProcessPool {
    /// Pre-spawned, idle Firecracker VMs waiting for a snapshot load.
    ready: Arc<Mutex<Vec<FirecrackerVM>>>,
    /// Base directory for process work dirs.
    base_dir: PathBuf,
    /// Config template for spawning new processes.
    config: VMConfig,
    /// Monotonic counter for unique process IDs.
    counter: AtomicU64,
    /// How many processes to keep warm.
    target_size: usize,
}

impl ProcessPool {
    /// Create a new pool and pre-spawn `size` Firecracker processes.
    ///
    /// All processes are spawned concurrently. Returns once every
    /// process has its API socket ready.
    pub async fn new(config: VMConfig, base_dir: PathBuf, size: usize) -> Result<Self> {
        std::fs::create_dir_all(&base_dir)?;
        let base_dir = std::fs::canonicalize(&base_dir)?;

        let pool = Self {
            ready: Arc::new(Mutex::new(Vec::with_capacity(size))),
            base_dir,
            config,
            counter: AtomicU64::new(0),
            target_size: size,
        };

        // Initial fill — spawn all processes in parallel.
        pool.fill(size).await?;

        Ok(pool)
    }

    /// Spawn `count` processes in parallel and add them to the pool.
    async fn fill(&self, count: usize) -> Result<()> {
        let mut handles = Vec::with_capacity(count);

        for _ in 0..count {
            let id = self.counter.fetch_add(1, Ordering::Relaxed);
            let vm_id = format!("pool-{id}");
            let work_dir = self.base_dir.join(&vm_id);
            let config = self.config.clone();

            handles.push(tokio::spawn(async move {
                let mut vm = FirecrackerVM::new(config, Some(work_dir), &vm_id)?;
                vm.spawn_process().await?;
                Ok::<_, anyhow::Error>(vm)
            }));
        }

        let mut vms = Vec::with_capacity(count);
        let mut errors = Vec::new();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(vm)) => vms.push(vm),
                Ok(Err(e)) => errors.push(format!("pool process {i}: {e}")),
                Err(e) => errors.push(format!("pool process {i}: task panicked: {e}")),
            }
        }

        if !errors.is_empty() {
            for mut vm in vms {
                vm.destroy();
            }
            bail!(
                "Failed to spawn {} pool processes:\n{}",
                errors.len(),
                errors.join("\n")
            );
        }

        let mut ready = self.ready.lock().await;
        ready.extend(vms);

        Ok(())
    }

    /// Take `count` pre-spawned processes from the pool.
    ///
    /// If the pool doesn't have enough, spawns the missing ones
    /// on the spot (slower fallback). After taking, kicks off
    /// background replenishment so the pool is full again by the
    /// time the next iteration needs processes.
    pub async fn take(&self, count: usize) -> Result<Vec<FirecrackerVM>> {
        let mut ready = self.ready.lock().await;
        let available = ready.len();

        let taken: Vec<FirecrackerVM> = if available >= count {
            // Fast path: drain from the back (O(1) per element).
            let split_at = available - count;
            ready.split_off(split_at)
        } else {
            // Take whatever we have, spawn the rest synchronously.
            let mut result = ready.drain(..).collect::<Vec<_>>();
            drop(ready); // release lock before spawning

            let deficit = count - result.len();
            let mut handles = Vec::with_capacity(deficit);
            for _ in 0..deficit {
                let id = self.counter.fetch_add(1, Ordering::Relaxed);
                let vm_id = format!("pool-{id}");
                let work_dir = self.base_dir.join(&vm_id);
                let config = self.config.clone();

                handles.push(tokio::spawn(async move {
                    let mut vm = FirecrackerVM::new(config, Some(work_dir), &vm_id)?;
                    vm.spawn_process().await?;
                    Ok::<_, anyhow::Error>(vm)
                }));
            }

            for handle in handles {
                result.push(handle.await??);
            }
            result
        };

        // Ensure we return exactly `count`.
        assert_eq!(taken.len(), count);

        // Kick off background replenishment. The spawned task fills
        // the pool back to target_size while the caller proceeds
        // with snapshot loading and stepping.
        self.replenish_background();

        Ok(taken)
    }

    /// Spawn replacement processes in the background.
    ///
    /// Non-blocking — returns immediately. The spawned processes
    /// will be ready by the time the next `take()` is called (~70ms
    /// later, which is more than enough for process startup).
    fn replenish_background(&self) {
        let ready = Arc::clone(&self.ready);
        let base_dir = self.base_dir.clone();
        let config = self.config.clone();
        let counter = &self.counter;
        let target = self.target_size;

        // Allocate IDs now (on the caller's thread) to avoid needing
        // &self in the spawned task.
        let mut spawn_ids = Vec::new();
        // We don't hold the lock here, so we estimate. Worst case
        // we spawn a few extra — they'll sit idle and get cleaned up.
        for _ in 0..target {
            spawn_ids.push(counter.fetch_add(1, Ordering::Relaxed));
        }

        tokio::spawn(async move {
            // Check how many we actually need.
            let current = ready.lock().await.len();
            let needed = target.saturating_sub(current);
            if needed == 0 {
                return;
            }

            let ids = &spawn_ids[..needed];
            let mut handles = Vec::with_capacity(needed);

            for &id in ids {
                let vm_id = format!("pool-{id}");
                let work_dir = base_dir.join(&vm_id);
                let config = config.clone();

                handles.push(tokio::spawn(async move {
                    let mut vm = FirecrackerVM::new(config, Some(work_dir), &vm_id)?;
                    vm.spawn_process().await?;
                    Ok::<_, anyhow::Error>(vm)
                }));
            }

            let mut vms = Vec::with_capacity(needed);
            for handle in handles {
                match handle.await {
                    Ok(Ok(vm)) => vms.push(vm),
                    _ => { /* log or ignore — best effort */ }
                }
            }

            if !vms.is_empty() {
                let mut ready = ready.lock().await;
                ready.extend(vms);
            }
        });
    }
}

impl Drop for ProcessPool {
    fn drop(&mut self) {
        // Synchronous cleanup — kill all idle processes.
        if let Ok(mut ready) = self.ready.try_lock() {
            for vm in ready.drain(..) {
                drop(vm); // triggers FirecrackerVM::drop → destroy
            }
        }
    }
}
