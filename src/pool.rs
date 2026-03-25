use anyhow::{bail, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::snapshot::Snapshot;
use crate::vm::{FirecrackerVM, VMConfig};

pub struct ForkResult {
    pub children: Vec<FirecrackerVM>,
    pub snapshot: Snapshot,
    pub fork_time_ms: f64,
}

/// VM pool with background pre-spawning of Firecracker processes.
///
/// The pool keeps `pool_size` idle Firecracker processes ready at all times.
/// A background thread replenishes the pool as VMs are consumed. At fork
/// time, we just load the snapshot into pre-spawned processes — no fork+exec.
pub struct VMPool {
    pub base_dir: PathBuf,
    pool_size: usize,
    fork_counter: AtomicU64,
    dir_counter: Arc<AtomicU64>,
    warm_pool: Arc<Mutex<Vec<FirecrackerVM>>>,
    config: VMConfig,
    shutdown: Arc<AtomicBool>,
    _replenish_handle: Option<thread::JoinHandle<()>>,
}

impl VMPool {
    pub fn new(base_dir: PathBuf, pool_size: usize, config: VMConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_dir)?;
        let base_dir = std::fs::canonicalize(&base_dir)?;

        let warm_pool: Arc<Mutex<Vec<FirecrackerVM>>> = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let dir_counter = Arc::new(AtomicU64::new(0));

        // Pre-spawn initial batch in parallel
        let initial = Self::spawn_batch(&base_dir, &config, &dir_counter, pool_size);
        warm_pool.lock().unwrap().extend(initial);

        // Background replenisher
        let wp = warm_pool.clone();
        let shut = shutdown.clone();
        let dc = dir_counter.clone();
        let bd = base_dir.clone();
        let cfg = config.clone();
        let ps = pool_size;

        let handle = thread::spawn(move || {
            while !shut.load(Ordering::Relaxed) {
                let current = wp.lock().unwrap().len();
                if current < ps {
                    let batch = Self::spawn_batch(&bd, &cfg, &dc, ps - current);
                    wp.lock().unwrap().extend(batch);
                } else {
                    thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        });

        Ok(Self {
            base_dir,
            pool_size,
            fork_counter: AtomicU64::new(0),
            dir_counter,
            warm_pool,
            config,
            shutdown,
            _replenish_handle: Some(handle),
        })
    }

    /// Spawn `count` idle Firecracker processes in parallel.
    fn spawn_batch(
        base_dir: &PathBuf,
        config: &VMConfig,
        dir_counter: &Arc<AtomicU64>,
        count: usize,
    ) -> Vec<FirecrackerVM> {
        let handles: Vec<_> = (0..count)
            .map(|_| {
                let id = dir_counter.fetch_add(1, Ordering::SeqCst);
                let dir = base_dir.join(format!("warm-{id}"));
                let cfg = config.clone();
                thread::spawn(move || {
                    let _ = std::fs::remove_dir_all(&dir);
                    let _ = std::fs::create_dir_all(&dir);
                    FirecrackerVM::new_idle(cfg, dir, &format!("warm-{id}"))
                })
            })
            .collect();

        let mut vms = Vec::new();
        for h in handles {
            match h.join() {
                Ok(Ok(vm)) => vms.push(vm),
                Ok(Err(e)) => eprintln!("[warm-pool] spawn error: {e}"),
                Err(_) => eprintln!("[warm-pool] spawn thread panicked"),
            }
        }
        vms
    }

    /// Take `n` idle VMs from the warm pool, blocking until available.
    fn take_warm(&self, n: usize) -> Vec<FirecrackerVM> {
        loop {
            {
                let mut pool = self.warm_pool.lock().unwrap();
                if pool.len() >= n {
                    let start = pool.len() - n;
                    return pool.drain(start..).collect();
                }
            }
            // Not enough ready — spin briefly, the background thread is spawning
            thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Fork a running VM into multiple children.
    pub fn fork(&self, parent: &mut FirecrackerVM, num_children: usize) -> Result<ForkResult> {
        let t0 = std::time::Instant::now();
        let fork_id = self.fork_counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Pause and snapshot
        parent.pause()?;
        let shm_base = std::path::Path::new("/dev/shm");
        let snap_dir = if shm_base.exists() {
            shm_base.join(format!("atari-fork-snap-{fork_id}"))
        } else {
            self.base_dir.join(format!("snap-{fork_id}"))
        };
        let snapshot = parent.create_snapshot(&snap_dir)?;

        // Grab pre-spawned processes from the warm pool
        let idle_vms = self.take_warm(num_children);

        // Load snapshot into each in parallel — this is the only work at fork time
        let handles: Vec<_> = idle_vms
            .into_iter()
            .map(|mut vm| {
                let snap = snapshot.clone();
                thread::spawn(move || -> Result<FirecrackerVM> {
                    vm.load_snapshot(&snap, true)?;
                    Ok(vm)
                })
            })
            .collect();

        let mut children = Vec::with_capacity(num_children);
        let mut errors = Vec::new();
        for (i, h) in handles.into_iter().enumerate() {
            match h.join() {
                Ok(Ok(vm)) => children.push(vm),
                Ok(Err(e)) => errors.push((i, e)),
                Err(_) => errors.push((i, anyhow::anyhow!("thread panicked"))),
            }
        }

        if !errors.is_empty() {
            for mut c in children {
                c.destroy();
            }
            let msg: Vec<_> = errors
                .iter()
                .map(|(i, e)| format!("child {i}: {e}"))
                .collect();
            bail!("Failed: {}", msg.join(", "));
        }

        Ok(ForkResult {
            children,
            snapshot,
            fork_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
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

impl Drop for VMPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Ok(mut pool) = self.warm_pool.lock() {
            for mut vm in pool.drain(..) {
                vm.destroy();
            }
        }
    }
}
