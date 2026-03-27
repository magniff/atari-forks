use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand::SeedableRng;
use tokio::io::AsyncBufReadExt;

mod atari_client;
mod process_pool;
mod scheduler;
mod snapshot;
mod vm;

use atari_client::AtariClient;
use scheduler::VMScheduler;
use vm::{FirecrackerVM, VMConfig};

#[derive(Parser)]
#[command(name = "tree-search", about = "Atari Forking Tree Search Benchmark")]
struct Arguments {
    #[arg(long, default_value = "firecracker")]
    firecracker: PathBuf,

    #[arg(long)]
    kernel: PathBuf,

    #[arg(long)]
    rootfs: PathBuf,

    #[arg(long, default_value_t = 100)]
    iterations: usize,

    #[arg(long, default_value = "./frames")]
    output: PathBuf,

    #[arg(long, default_value_t = 1)]
    vcpus: u32,

    #[arg(long, default_value_t = 128)]
    mem: u32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value = "/dev/shm/atari-pool")]
    pool_dir: Option<PathBuf>,
}

async fn save_frame(data: &[u8], path: &PathBuf) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, data).await?;
    Ok(())
}

/// Wait for AGENT_READY on the VM's serial console (stdout).
///
/// Spawns a background task that drains stdout to prevent the pipe
/// buffer from filling up and blocking Firecracker. Returns once the
/// sentinel line is seen, or errors on timeout.
async fn wait_for_agent_ready(
    stdout: tokio::process::ChildStdout,
    timeout: std::time::Duration,
) -> Result<()> {
    let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ready_clone = ready.clone();

    // Drain stdout in background — must keep reading even after we
    // see AGENT_READY, otherwise the pipe buffer fills and FC stalls.
    tokio::spawn(async move {
        let mut reader = tokio::io::BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if line.contains("AGENT_READY") {
                        ready_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        }
    });

    let deadline = tokio::time::Instant::now() + timeout;
    while !ready.load(std::sync::atomic::Ordering::SeqCst) {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("Guest agent did not send AGENT_READY within {:?}", timeout);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Arguments::parse();
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

    // ── Set up pool directory (optionally on tmpfs) ─────────────────

    let pool_dir = args
        .pool_dir
        .unwrap_or_else(|| frames_dir.parent().unwrap_or(frames_dir).join("vm-pool"));
    std::fs::create_dir_all(&pool_dir)?;

    let mut frame_count = 0usize;

    println!("Starting tree search: {} iterations", args.iterations);
    println!("Guest memory: {}MB", args.mem);
    println!("Output directory: {:?}", frames_dir);
    println!("Expected frames: {}", 1 + 4 * args.iterations);
    println!();

    // ── Phase 1: Boot root VM ───────────────────────────────────────

    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]);

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(spinner_style.clone());
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));
    spinner.set_message("Booting root VM...");
    let t_boot = Instant::now();

    let mut root_vm = FirecrackerVM::new(config.clone(), None, "root")?;
    root_vm.boot().await?;

    // Wait for the guest agent to be ready
    if let Some(stdout) = root_vm.take_stdout() {
        wait_for_agent_ready(stdout, std::time::Duration::from_secs(60)).await?;
    }

    // Connect and initialize the environment
    let vsock_path = root_vm.vsock_uds_path();
    let mut client = AtariClient::connect(&vsock_path, std::time::Duration::from_secs(10)).await?;

    let actions = client.get_legal_actions().await?;
    let num_actions = actions.len();

    let initial_obs = client.reset().await?;
    let frame_name = format!("frame_{:06}.png", frame_count);
    save_frame(&initial_obs, &frames_dir.join(&frame_name)).await?;
    frame_count += 1;
    drop(client);

    spinner.finish_with_message(format!(
        "Root VM ready — actions: {:?}  boot+reset: {:.1}ms",
        actions,
        t_boot.elapsed().as_secs_f64() * 1000.0
    ));

    // ── Phase 2: Tree search ────────────────────────────────────────
    let mut pool = VMScheduler::try_new(pool_dir, &config, num_actions * 4).await?;
    let mut current_vm = root_vm;

    let progress_style =
        ProgressStyle::with_template("{spinner:.cyan} iter {pos}/{len}  fork {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]);

    let pb = ProgressBar::new(args.iterations as u64);
    pb.set_style(progress_style);
    pb.enable_steady_tick(std::time::Duration::from_millis(80));

    for _ in 0..args.iterations {
        let t_iter = Instant::now();

        // Fork: pause → snapshot → restore N children in parallel
        let t_fork = Instant::now();
        pb.set_message("forking...");
        let fork_result = pool.fork(&mut current_vm, num_actions).await?;
        // Parent is paused and no longer needed — kill it now to
        // release its memory mappings before we step the children.
        current_vm.destroy();
        let fork_ms = t_fork.elapsed().as_secs_f64() * 1000.0;

        // Step all children in parallel — each takes a different action
        let t_step = Instant::now();
        pb.set_message(format!("{fork_ms:.1}ms  stepping..."));
        let step_results = VMScheduler::step_all(&fork_result.children, &actions).await?;
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;

        // Record results
        for result in step_results.iter() {
            let frame_name = format!("frame_{:06}.png", frame_count);
            save_frame(&result.obs, &frames_dir.join(&frame_name)).await?;
            frame_count += 1;
        }

        // Select a random child as the new root
        let selected_idx = rng.gen_range(0..num_actions);
        let selected_action = actions[selected_idx];

        // Destroy unselected children, keep selected as next root
        current_vm = pool.select_and_cleanup(fork_result, selected_idx);

        let total_ms = t_iter.elapsed().as_secs_f64() * 1000.0;
        pb.set_message(format!(
            "{fork_ms:.1}ms  step {step_ms:.1}ms  total {total_ms:.1}ms  selected action {selected_action}"
        ));
        pb.inc(1);
    }

    pb.finish_and_clear();
    current_vm.destroy();

    println!();
    println!("============================================================");
    println!("BENCHMARK RESULTS");
    println!("============================================================");
    println!("Total frames saved: {}", frame_count);

    Ok(())
}
