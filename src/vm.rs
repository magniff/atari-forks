use anyhow::{bail, Context, Result};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::Request;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::net::UnixStream;
use tokio::process::{Child, Command};

use crate::snapshot::Snapshot;

/// VM lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VMState {
    Created,
    Running,
    Paused,
    Stopped,
}

/// Configuration for booting a fresh Firecracker VM.
#[derive(Debug, Clone)]
pub struct VMConfig {
    pub firecracker_bin: PathBuf,
    pub kernel_path: PathBuf,
    pub rootfs_path: PathBuf,
    pub vcpu_count: u32,
    pub mem_size_mib: u32,
    pub boot_args: String,
    pub track_dirty: bool,
}

impl Default for VMConfig {
    fn default() -> Self {
        Self {
            firecracker_bin: PathBuf::from("firecracker"),
            kernel_path: PathBuf::new(),
            rootfs_path: PathBuf::new(),
            vcpu_count: 1,
            mem_size_mib: 128,
            boot_args: [
                "console=ttyS0",
                "reboot=k",
                "panic=1",
                "pci=off",
                "random.trust_cpu=on", // avoid entropy starvation
                "init=/opt/init.sh",
                "mitigations=off", // skip spectre/meltdown — not needed for benchmarking
                "audit=0",         // disable audit subsystem
                "quiet",           // reduce kernel log noise
                "loglevel=1",      // only critical kernel messages
                "nomodule",        // skip module loading
            ]
            .join(" "),
            track_dirty: true,
        }
    }
}

/// Manages a single Firecracker microVM process.
///
/// Each instance owns one Firecracker OS process and communicates
/// with it over a Unix domain socket (HTTP API via hyper).
pub struct FirecrackerVM {
    pub config: VMConfig,
    pub state: VMState,
    pub work_dir: PathBuf,
    pub rootfs_path: PathBuf,
    socket_path: PathBuf,
    log_path: PathBuf,
    process: Option<Child>,
}

impl FirecrackerVM {
    pub fn new(config: VMConfig, work_dir: Option<PathBuf>, vm_id: &str) -> Result<Self> {
        let work_dir = match work_dir {
            Some(d) => {
                std::fs::create_dir_all(&d)?;
                std::fs::canonicalize(&d).context("canonicalize work_dir")?
            }
            None => {
                let td = tempfile::Builder::new()
                    .prefix(&format!("fc-{}-", vm_id))
                    .tempdir()?;
                let d = td.path().to_path_buf();
                std::mem::forget(td);
                d
            }
        };
        std::fs::create_dir_all(&work_dir)?;
        let rootfs_path = std::fs::canonicalize(&config.rootfs_path)
            .unwrap_or_else(|_| config.rootfs_path.clone());

        Ok(Self {
            config,
            state: VMState::Created,
            socket_path: work_dir.join("firecracker.sock"),
            log_path: work_dir.join("firecracker.log"),
            rootfs_path,
            work_dir,
            process: None,
        })
    }

    /// Full boot: spawn process, configure, start.
    pub async fn boot(&mut self) -> Result<()> {
        self.spawn_process().await?;
        self.configure_machine().await?;
        self.configure_boot_source().await?;
        self.configure_rootfs().await?;
        self.configure_vsock().await?;
        self.start_instance().await?;
        Ok(())
    }

    /// Spawn the Firecracker process and wait for the API socket.
    ///
    /// After this returns, the process is idle and ready to accept
    /// API calls (boot config or snapshot load). Public so that
    /// `ProcessPool` can pre-warm processes.
    pub async fn spawn_process(&mut self) -> Result<()> {
        let _ = std::fs::remove_file(&self.socket_path);

        let fc_bin = std::fs::canonicalize(&self.config.firecracker_bin)
            .context("resolve firecracker binary path")?;

        let log_file = std::fs::File::create(&self.log_path)?;
        let child = Command::new(&fc_bin)
            .arg("--api-sock")
            .arg(&self.socket_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log_file)
            .current_dir(&self.work_dir)
            .spawn()
            .context("spawn firecracker")?;

        self.process = Some(child);

        // Wait for API socket to become available
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if self.socket_path.exists() && self.call_firecracker("GET", "/", None).await.is_ok() {
                return Ok(());
            }
            if tokio::time::Instant::now() > deadline {
                bail!("Firecracker failed to start. Check {:?}", self.log_path);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Make an HTTP request to the Firecracker API over the Unix socket.
    ///
    /// Opens a fresh connection per call. Firecracker's API is
    /// single-threaded and low-frequency — connection reuse isn't
    /// worth the complexity of keeping an idle connection alive.
    async fn call_firecracker(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .context("connect to API socket")?;
        let io = TokioIo::new(stream);

        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .context("HTTP handshake")?;

        // Drive the connection in the background.
        // The spawned task exits when the response is fully read
        // and `sender` is dropped.
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                eprintln!("[api] connection error: {e}");
            }
        });

        let body_bytes = match &body {
            Some(v) => v.to_string().into_bytes(),
            None => Vec::new(),
        };

        let mut builder = Request::builder()
            .method(method)
            .uri(path)
            .header("Host", "localhost")
            .header("Accept", "application/json");

        if body.is_some() {
            builder = builder
                .header("Content-Type", "application/json")
                .header("Content-Length", body_bytes.len());
        }

        let req = builder
            .body(Full::new(Bytes::from(body_bytes)))
            .context("build request")?;

        let response = sender.send_request(req).await.context("send request")?;
        let status = response.status();
        let resp_bytes = response
            .into_body()
            .collect()
            .await
            .context("read response body")?
            .to_bytes();
        let body_text = String::from_utf8_lossy(&resp_bytes);

        if status.as_u16() >= 400 {
            bail!(
                "Firecracker API error {} on {} {}: {}",
                status.as_u16(),
                method,
                path,
                body_text.trim()
            );
        }

        if body_text.trim().is_empty() {
            Ok(Value::Null)
        } else {
            Ok(serde_json::from_str(body_text.trim()).unwrap_or(Value::Null))
        }
    }

    async fn configure_machine(&self) -> Result<()> {
        self.call_firecracker(
            "PUT",
            "/machine-config",
            Some(json!({
                "vcpu_count": self.config.vcpu_count,
                "mem_size_mib": self.config.mem_size_mib,
                "track_dirty_pages": self.config.track_dirty,
            })),
        )
        .await?;
        Ok(())
    }

    async fn configure_boot_source(&self) -> Result<()> {
        let kernel = std::fs::canonicalize(&self.config.kernel_path)?;
        self.call_firecracker(
            "PUT",
            "/boot-source",
            Some(json!({
                "kernel_image_path": kernel.to_str().unwrap(),
                "boot_args": self.config.boot_args,
            })),
        )
        .await?;
        Ok(())
    }

    async fn configure_rootfs(&self) -> Result<()> {
        self.call_firecracker(
            "PUT",
            "/drives/rootfs",
            Some(json!({
                "drive_id": "rootfs",
                "path_on_host": self.rootfs_path.to_str().unwrap(),
                "is_root_device": true,
                "is_read_only": true,
            })),
        )
        .await?;
        Ok(())
    }

    async fn configure_vsock(&self) -> Result<()> {
        self.call_firecracker(
            "PUT",
            "/vsock",
            Some(json!({
                "vsock_id": "1",
                "guest_cid": 3,
                "uds_path": "v.sock",
            })),
        )
        .await?;
        Ok(())
    }

    async fn start_instance(&mut self) -> Result<()> {
        self.call_firecracker(
            "PUT",
            "/actions",
            Some(json!({
                "action_type": "InstanceStart",
            })),
        )
        .await?;
        self.state = VMState::Running;
        Ok(())
    }

    pub async fn pause(&mut self) -> Result<()> {
        if self.state != VMState::Running {
            bail!("Can only pause a running VM, got {:?}", self.state);
        }
        self.call_firecracker("PATCH", "/vm", Some(json!({"state": "Paused"})))
            .await?;
        self.state = VMState::Paused;
        Ok(())
    }

    /// Create a snapshot of this (paused) VM.
    ///
    /// When `memory_path` is `None`, creates a full snapshot: Firecracker
    /// dumps all guest memory to `<snapshot_dir>/memory`. When `memory_path`
    /// is `Some(base)`, creates a diff snapshot: FC writes only pages dirtied
    /// since the last snapshot/restore into `base`, overwriting those offsets
    /// in-place. The caller must ensure `base` already contains a valid full
    /// image so that clean pages retain their correct content.
    ///
    /// After the write, we kick off async writeback via
    /// `sync_file_range(SYNC_FILE_RANGE_WRITE)` to prevent dirty page
    /// accumulation from causing stalls on later iterations.
    pub async fn create_snapshot(
        &self,
        snapshot_dir: &Path,
        memory_path: Option<&Path>,
    ) -> Result<Snapshot> {
        if self.state != VMState::Paused {
            bail!("VM must be paused before snapshotting");
        }
        std::fs::create_dir_all(snapshot_dir)?;
        let snap_dir = std::fs::canonicalize(snapshot_dir)?;
        let vmstate_path = snap_dir.join("vmstate");
        let is_diff = memory_path.is_some();
        let memory_path = memory_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| snap_dir.join("memory"));

        let snapshot_type = if is_diff { "Diff" } else { "Full" };

        self.call_firecracker(
            "PUT",
            "/snapshot/create",
            Some(json!({
                "snapshot_type": snapshot_type,
                "snapshot_path": vmstate_path.to_str().unwrap(),
                "mem_file_path": memory_path.to_str().unwrap(),
            })),
        )
        .await?;

        // Kick off async writeback of the memory file immediately.
        //
        // Firecracker writes guest memory into the page cache. Without
        // this, the kernel accumulates dirty pages and eventually hits
        // dirty_ratio, causing a *future* write/mmap call to block
        // synchronously for writeback — random multi-second stalls.
        //
        // SYNC_FILE_RANGE_WRITE is non-blocking: starts writeback in
        // the background and returns immediately.
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            if let Ok(f) = std::fs::File::open(&memory_path) {
                unsafe {
                    libc::sync_file_range(f.as_raw_fd(), 0, 0, libc::SYNC_FILE_RANGE_WRITE);
                }
            }
        }

        Ok(Snapshot {
            vmstate_path,
            memory_path,
            rootfs_path: self.rootfs_path.clone(),
            config: self.config.clone(),
        })
    }

    /// Load a snapshot into an already-spawned Firecracker process.
    ///
    /// The process must be freshly spawned (no prior boot or load).
    /// This is the fast path used by the process pool — the spawn
    /// and API socket wait have already happened, so this is just
    /// one HTTP round-trip to Firecracker.
    pub async fn load_snapshot(&mut self, snapshot: &Snapshot, resume: bool) -> Result<()> {
        self.rootfs_path = snapshot.rootfs_path.clone();
        self.call_firecracker(
            "PUT",
            "/snapshot/load",
            Some(json!({
                "snapshot_path": snapshot.vmstate_path.to_str().unwrap(),
                "mem_backend": {
                    "backend_type": "File",
                    "backend_path": snapshot.memory_path.to_str().unwrap(),
                },
                "enable_diff_snapshots": snapshot.config.track_dirty,
                "resume_vm": resume,
            })),
        )
        .await?;

        self.state = if resume {
            VMState::Running
        } else {
            VMState::Paused
        };
        Ok(())
    }

    /// Absolute path to the vsock UDS.
    pub fn vsock_uds_path(&self) -> PathBuf {
        self.work_dir.join("v.sock")
    }

    /// Take ownership of the process's stdout handle (serial console).
    pub fn take_stdout(&mut self) -> Option<tokio::process::ChildStdout> {
        self.process.as_mut().and_then(|p| p.stdout.take())
    }

    /// Kill the Firecracker process and clean up the socket file.
    pub fn destroy(&mut self) {
        if let Some(mut child) = self.process.take() {
            // start_kill() sends SIGKILL — non-async, just a syscall.
            let _ = child.start_kill();
            // The Child is dropped here; tokio reaps the zombie via
            // a background SIGCHLD handler.
        }
        self.state = VMState::Stopped;
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

impl Drop for FirecrackerVM {
    fn drop(&mut self) {
        self.destroy();
    }
}
