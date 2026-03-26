use anyhow::Result;
use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

mod atari_client;
mod pool;
mod snapshot;
mod vm;

use atari_client::AtariClient;
use pool::VMPool;
use vm::{FirecrackerVM, VMConfig};

#[derive(Parser)]
#[command(name = "tree-search", about = "Atari Forking Tree Search Benchmark")]
struct Args {
    #[arg(long, default_value = "firecracker")]
    firecracker: PathBuf,
    #[arg(long)]
    kernel: PathBuf,
    #[arg(long)]
    rootfs: PathBuf,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    #[arg(long, default_value = "./frames")]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    vcpus: u32,
    #[arg(long, default_value_t = 64)]
    mem: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn save_frame(data: &[u8], path: &PathBuf) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, data)?;
    Ok(())
}

fn save_history(path: &PathBuf, history: &HashMap<String, serde_json::Value>) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(history)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let config = VMConfig {
        firecracker_bin: args.firecracker,
        kernel_path: args.kernel,
        rootfs_path: args.rootfs,
        vcpu_count: args.vcpus,
        mem_size_mib: args.mem,
        ..Default::default()
    };

    let frames_dir = &args.output;
    std::fs::create_dir_all(frames_dir)?;
    let history_path = frames_dir.join("history.json");

    let mut frame_count = 0usize;
    let mut frame_history: HashMap<String, serde_json::Value> = HashMap::new();
    let mut current_actions: Vec<i64> = Vec::new();
    let mut current_rewards: Vec<f64> = Vec::new();

    println!("Starting tree search: {} iterations", args.iterations);
    println!("Output directory: {:?}", frames_dir);
    println!("Expected frames: {}", 1 + 4 * args.iterations);
    println!();

    // --- Phase 1: Boot root VM ---
    println!("Booting root VM...");
    let t_boot = Instant::now();

    let mut root_vm = FirecrackerVM::new(config.clone(), None, "root")?;
    root_vm.boot()?;

    // Wait for AGENT_READY in background thread (keeps stdout alive)
    let vsock_path = root_vm.vsock_uds_path();
    let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ready_clone = ready.clone();

    if let Some(stdout) = root_vm.process_stdout() {
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if line.contains("AGENT_READY") {
                    ready_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
        });
    }

    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    while !ready.load(std::sync::atomic::Ordering::SeqCst) {
        if Instant::now() > deadline {
            anyhow::bail!("Guest agent did not send AGENT_READY within 60s");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let mut client = AtariClient::connect_vsock(&vsock_path, std::time::Duration::from_secs(10))?;

    let actions = client.get_legal_actions()?;
    let num_actions = actions.len();
    println!("Legal actions: {:?} ({} total)", actions, num_actions);

    let initial_obs = client.reset()?;
    let frame_name = format!("frame_{:06}.png", frame_count);
    save_frame(&initial_obs, &frames_dir.join(&frame_name))?;
    frame_history.insert(frame_name, json!({"actions": [], "rewards": []}));
    frame_count += 1;
    drop(client);
    save_history(&history_path, &frame_history)?;

    println!(
        "Boot + reset took {:.1}ms",
        t_boot.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    // --- Phase 2: Tree search ---
    let pool_dir = frames_dir.parent().unwrap_or(frames_dir).join("vm-pool");
    let pool = VMPool::new(pool_dir, num_actions, config.clone())?;

    let mut current_vm = root_vm;

    // Background cleanup with bounded channel — blocks if cleanup falls behind,
    // preventing FD accumulation
    let (cleanup_tx, cleanup_rx) = std::sync::mpsc::sync_channel::<Vec<FirecrackerVM>>(1);
    let _cleanup_thread = std::thread::spawn(move || {
        for vms in cleanup_rx {
            for mut vm in vms {
                let d = vm.work_dir.clone();
                vm.destroy();
                let _ = std::fs::remove_dir_all(&d);
            }
        }
    });

    // Background frame saver
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<(PathBuf, Vec<u8>)>();
    let _frame_thread = std::thread::spawn(move || {
        for (path, data) in frame_rx {
            let _ = std::fs::write(&path, &data);
        }
    });

    // Pipeline: pre-create snapshot so fork only needs to load
    let shm_base = std::path::Path::new("/dev/shm");
    let mut snap_counter = 0u64;

    // Pre-create initial snapshot
    current_vm.pause()?;
    snap_counter += 1;
    let snap_dir = if shm_base.exists() {
        shm_base.join(format!("atari-fork-snap-{snap_counter}"))
    } else {
        PathBuf::from(format!("vm-pool/snap-{snap_counter}"))
    };
    let mut current_snapshot = current_vm.create_snapshot(&snap_dir)?;

    for iteration in 0..args.iterations {
        let t_iter = Instant::now();

        // Fork — snapshot is already created, just load into warm VMs
        let t_fork = Instant::now();
        let idle_vms = pool.take_warm(num_actions);
        let snap = current_snapshot.clone();
        let load_handles: Vec<_> = idle_vms
            .into_iter()
            .map(|mut vm| {
                let s = snap.clone();
                std::thread::spawn(move || -> Result<FirecrackerVM> {
                    vm.load_snapshot(&s, true)?;
                    Ok(vm)
                })
            })
            .collect();

        let mut children: Vec<FirecrackerVM> = Vec::with_capacity(num_actions);
        for (idx, h) in load_handles.into_iter().enumerate() {
            match h.join() {
                Ok(Ok(vm)) => children.push(vm),
                Ok(Err(e)) => {
                    eprintln!(
                        "[fork] snapshot load failed for child {} at iter {}: {}",
                        idx, iteration, e
                    );
                    // Destroy any children that did load
                    for mut c in children {
                        c.destroy();
                    }
                    return Err(e);
                }
                Err(_) => {
                    for mut c in children {
                        c.destroy();
                    }
                    anyhow::bail!("snapshot load thread panicked at iter {}", iteration);
                }
            }
        }
        let fork_ms = t_fork.elapsed().as_secs_f64() * 1000.0;

        // Step each child in parallel
        let t_step = Instant::now();
        let step_handles: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(idx, action)| {
                let vp = children[idx].vsock_uds_path();
                let vm_id = children[idx].vm_id.clone();
                let a = *action;
                let iter = iteration;
                std::thread::spawn(move || -> Result<(i64, atari_client::StepResult)> {
                    for attempt in 0..2 {
                        match (|| -> Result<_> {
                            let mut c =
                                AtariClient::connect_vsock(&vp, std::time::Duration::from_secs(5))?;
                            Ok((a, c.step(a)?))
                        })() {
                            Ok(r) => return Ok(r),
                            Err(e) if attempt == 0 => {
                                eprintln!(
                                    "[step] iter={} vm={} action={} attempt=0 err: {}",
                                    iter, vm_id, a, e
                                );
                                std::thread::sleep(std::time::Duration::from_millis(50));
                            }
                            Err(e) => {
                                eprintln!(
                                    "[step] iter={} vm={} action={} FAILED: {}",
                                    iter, vm_id, a, e
                                );
                                return Err(e);
                            }
                        }
                    }
                    unreachable!()
                })
            })
            .collect();

        let mut step_results = Vec::new();
        for (idx, h) in step_handles.into_iter().enumerate() {
            match h.join() {
                Ok(Ok(r)) => step_results.push(r),
                Ok(Err(e)) => {
                    eprintln!("[step] child {} failed at iter {}: {}", idx, iteration, e);
                    return Err(e);
                }
                Err(_) => anyhow::bail!("step thread {} panicked at iter {}", idx, iteration),
            }
        }
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;

        // Record results
        let mut child_rewards = Vec::new();
        let mut child_done = Vec::new();
        for (action, result) in &step_results {
            child_rewards.push(result.reward);
            child_done.push(result.done);
            let fname = format!("frame_{:06}.png", frame_count);
            let _ = frame_tx.send((frames_dir.join(&fname), result.obs.clone()));
            frame_history.insert(
                fname,
                json!({
                    "action": action,
                    "reward": result.reward,
                    "done": result.done,
                    "depth": current_actions.len(),
                }),
            );
            frame_count += 1;
        }

        // Select — prefer children that aren't game-over
        let alive: Vec<usize> = (0..num_actions).filter(|&i| !child_done[i]).collect();
        let selected_idx = if alive.is_empty() {
            // All children are game-over — clean up and stop
            for mut child in children {
                child.destroy();
            }
            current_snapshot.cleanup();

            // Save final history chunk
            if !frame_history.is_empty() {
                let chunk_start = (iteration / 1000) * 1000;
                let chunk_path =
                    frames_dir.join(format!("history_{}-{}.json", chunk_start, iteration + 1));
                frame_history.insert(
                    "_selected_path".to_string(),
                    json!({
                        "actions": &current_actions,
                        "rewards": &current_rewards,
                    }),
                );
                let _ = save_history(&chunk_path, &frame_history);
            }

            println!(
                "Game over on all branches at iteration {}. Total reward: {:.0}",
                iteration + 1,
                current_rewards.iter().sum::<f64>()
            );
            break;
        } else {
            alive[rng.gen_range(0..alive.len())]
        };

        current_actions.push(actions[selected_idx]);
        current_rewards.push(child_rewards[selected_idx]);

        // Kill discarded children and wait for them to die — this releases
        // their MAP_PRIVATE mmaps so tmpfs has room for the next snapshot
        let t_clean = Instant::now();
        let selected = children.remove(selected_idx);

        // SIGKILL discarded children + old parent (instant, no waitpid)
        let old_vm = std::mem::replace(&mut current_vm, selected);
        let mut to_cleanup: Vec<FirecrackerVM> = children;
        for vm in &mut to_cleanup {
            vm.kill_no_wait();
        }
        to_cleanup.push(old_vm); // old parent gets killed by destroy() in background

        current_snapshot.cleanup();

        // Pause+snapshot the selected child for the next iteration
        current_vm.pause()?;
        snap_counter += 1;
        let snap_dir = if shm_base.exists() {
            shm_base.join(format!("atari-fork-snap-{snap_counter}"))
        } else {
            PathBuf::from(format!("vm-pool/snap-{snap_counter}"))
        };
        current_snapshot = current_vm.create_snapshot(&snap_dir)?;

        // Send to bounded channel — blocks if previous batch isn't done yet,
        // which prevents FD/memory accumulation
        let _ = cleanup_tx.send(to_cleanup);
        let clean_ms = t_clean.elapsed().as_secs_f64() * 1000.0;

        let iter_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Iter {}/{}: fork={:.1}ms step={:.1}ms clean={:.1}ms total={:.1}ms",
            iteration + 1,
            args.iterations,
            fork_ms,
            step_ms,
            clean_ms,
            iter_ms
        );

        // Save history in chunks of 1000 iterations, then clear
        if (iteration + 1) % 1000 == 0 || iteration + 1 == args.iterations {
            let chunk_start = (iteration / 1000) * 1000;
            let chunk_end = iteration + 1;
            let chunk_path = frames_dir.join(format!("history_{}-{}.json", chunk_start, chunk_end));
            // Include the selected action path for just this chunk
            let path_start = if current_actions.len() > 1000 {
                current_actions.len() - 1000
            } else {
                0
            };
            frame_history.insert(
                "_selected_path".to_string(),
                json!({
                    "actions": &current_actions[path_start..],
                    "rewards": &current_rewards[path_start..],
                    "total_depth": current_actions.len(),
                    "total_reward": current_rewards.iter().sum::<f64>(),
                }),
            );
            save_history(&chunk_path, &frame_history)?;
            frame_history.clear();

            // Trim the action/reward vectors to prevent unbounded growth
            if current_actions.len() > 1000 {
                current_actions = current_actions.split_off(current_actions.len() - 1000);
                current_rewards = current_rewards.split_off(current_rewards.len() - 1000);
            }
        }
    }

    current_vm.destroy();
    current_snapshot.cleanup();

    println!("\nTotal frames saved: {}", frame_count);

    Ok(())
}
