use anyhow::Result;
use clap::Parser;
use rand::Rng;
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

    #[arg(long, default_value_t = 256)]
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
    let json = serde_json::to_string_pretty(history)?;
    std::fs::write(path, json)?;
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

    // Wait for AGENT_READY on serial in a background thread.
    // We keep stdout alive to prevent SIGPIPE to the Firecracker process.
    let vsock_path = root_vm.vsock_uds_path();
    let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ready_clone = ready.clone();

    if let Some(stdout) = root_vm.process_stdout() {
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if line.contains("AGENT_READY") {
                        ready_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    // Keep reading to prevent pipe buffer from filling up
                }
            }
        });
    }

    // Wait for ready signal
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !ready.load(std::sync::atomic::Ordering::SeqCst) {
        if std::time::Instant::now() > deadline {
            anyhow::bail!("Guest agent did not send AGENT_READY within 60s");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Connect to vsock
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

    // Background cleanup — waitpid + remove directories
    let (cleanup_tx, cleanup_rx) =
        std::sync::mpsc::channel::<(FirecrackerVM, Vec<FirecrackerVM>)>();
    let _cleanup_thread = std::thread::spawn(move || {
        for (mut parent, children) in cleanup_rx {
            let parent_dir = parent.work_dir.clone();
            parent.destroy();
            let _ = std::fs::remove_dir_all(&parent_dir);
            for mut child in children {
                let dir = child.work_dir.clone();
                child.destroy(); // waitpid happens here
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    });

    // Background frame saver — write PNGs without blocking the hot path
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<(PathBuf, Vec<u8>)>();
    let _frame_thread = std::thread::spawn(move || {
        for (path, data) in frame_rx {
            let _ = std::fs::write(&path, &data);
        }
    });

    for iteration in 0..args.iterations {
        let t_iter = Instant::now();

        // Fork
        let t_fork = Instant::now();
        let fork_result = pool.fork(&mut current_vm, num_actions)?;
        let fork_ms = t_fork.elapsed().as_secs_f64() * 1000.0;

        // Step each child in parallel
        let t_step = Instant::now();

        let step_handles: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(child_idx, action)| {
                let vsock_path = fork_result.children[child_idx].vsock_uds_path();
                let action = *action;
                std::thread::spawn(move || -> anyhow::Result<(i64, atari_client::StepResult)> {
                    // Retry once on connection failure
                    for attempt in 0..2 {
                        match (|| -> anyhow::Result<_> {
                            let mut client = AtariClient::connect_vsock(
                                &vsock_path,
                                std::time::Duration::from_secs(5),
                            )?;
                            let result = client.step(action)?;
                            Ok((action, result))
                        })() {
                            Ok(r) => return Ok(r),
                            Err(e) if attempt == 0 => {
                                std::thread::sleep(std::time::Duration::from_millis(50));
                                continue;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    unreachable!()
                })
            })
            .collect();

        let mut step_results = Vec::new();
        for handle in step_handles {
            step_results.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("step thread panicked"))??,
            );
        }

        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;

        // Record results
        let t_record = Instant::now();
        let mut child_rewards = Vec::new();
        for (action, result) in &step_results {
            child_rewards.push(result.reward);

            let frame_name = format!("frame_{:06}.png", frame_count);
            let _ = frame_tx.send((frames_dir.join(&frame_name), result.obs.clone()));

            let mut hist_actions = current_actions.clone();
            hist_actions.push(*action);
            let mut hist_rewards = current_rewards.clone();
            hist_rewards.push(result.reward);
            frame_history.insert(
                frame_name,
                json!({
                    "actions": hist_actions,
                    "rewards": hist_rewards,
                }),
            );
            frame_count += 1;
        }
        let record_ms = t_record.elapsed().as_secs_f64() * 1000.0;

        // Select random child
        let selected_idx = rng.gen_range(0..num_actions);
        let selected_action = actions[selected_idx];
        let selected_reward = child_rewards[selected_idx];

        current_actions.push(selected_action);
        current_rewards.push(selected_reward);

        // Select child, send everything else to background cleanup
        let t_cleanup = Instant::now();
        let (selected, discarded, snapshot) = pool.select(fork_result, selected_idx);
        let old_vm = std::mem::replace(&mut current_vm, selected);
        // SIGKILL all discarded children immediately (just a syscall, no wait).
        // This releases their MAP_PRIVATE mmaps so tmpfs pages are freed.
        let mut dead_children: Vec<FirecrackerVM> = discarded;
        for child in &mut dead_children {
            child.kill_no_wait();
        }
        // Snapshot files can go now that children are killed
        snapshot.cleanup();
        // Defer waitpid + directory removal to background
        let _ = cleanup_tx.send((old_vm, dead_children));
        let cleanup_ms = t_cleanup.elapsed().as_secs_f64() * 1000.0;

        let iter_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Iter {}/{}: fork={:.1}ms step={:.1}ms rec={:.1}ms clean={:.1}ms total={:.1}ms",
            iteration + 1,
            args.iterations,
            fork_ms,
            step_ms,
            record_ms,
            cleanup_ms,
            iter_ms
        );

        // Save history periodically
        if (iteration + 1) % 50 == 0 || iteration + 1 == args.iterations {
            save_history(&history_path, &frame_history)?;
        }
    }

    current_vm.destroy();

    save_history(&history_path, &frame_history)?;
    println!("Frame history saved to {:?}", history_path);

    // Print stats
    println!();
    println!("============================================================");
    println!("BENCHMARK RESULTS");
    println!("============================================================");
    println!("Total frames saved: {}", frame_count);

    Ok(())
}

use rand::SeedableRng;
