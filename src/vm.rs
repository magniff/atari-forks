use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

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
            mem_size_mib: 256,
            boot_args:
                "console=ttyS0 reboot=k panic=1 pci=off random.trust_cpu=on ro init=/opt/init.sh"
                    .into(),
            track_dirty: true,
        }
    }
}

/// Manages a single Firecracker microVM process.
///
/// Each instance owns one Firecracker OS process and communicates
/// with it over a Unix domain socket (REST API).
pub struct FirecrackerVM {
    pub config: VMConfig,
    pub vm_id: String,
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
            Some(d) => std::fs::canonicalize(&d)
                .or_else(|_| {
                    std::fs::create_dir_all(&d)?;
                    std::fs::canonicalize(&d)
                })
                .context("canonicalize work_dir")?,
            None => {
                let td = tempfile::Builder::new()
                    .prefix(&format!("fc-{}-", vm_id))
                    .tempdir()?;
                let d = td.path().to_path_buf();
                std::mem::forget(td); // prevent cleanup on drop
                d
            }
        };
        std::fs::create_dir_all(&work_dir)?;
        let rootfs_path = std::fs::canonicalize(&config.rootfs_path)
            .unwrap_or_else(|_| config.rootfs_path.clone());

        Ok(Self {
            config,
            vm_id: vm_id.to_string(),
            state: VMState::Created,
            socket_path: work_dir.join("firecracker.sock"),
            log_path: work_dir.join("firecracker.log"),
            rootfs_path,
            work_dir,
            process: None,
        })
    }

    /// Full boot: spawn process, configure, start.
    pub fn boot(&mut self) -> Result<()> {
        self.spawn_process()?;
        self.configure_machine()?;
        self.configure_boot_source()?;
        self.configure_rootfs()?;
        self.configure_vsock()?;
        self.start_instance()?;
        Ok(())
    }

    fn spawn_process(&mut self) -> Result<()> {
        // Clean up stale socket
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

        // Wait for API socket to appear — poll every 5ms instead of 100ms
        for _ in 0..2000 {
            if self.socket_path.exists() {
                // Try a quick API call to confirm Firecracker is listening
                if self.api_call("GET", "/", None).is_ok() {
                    return Ok(());
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        bail!("Firecracker failed to start. Check {:?}", self.log_path)
    }

    /// Make an HTTP request to the Firecracker API over the Unix socket.
    fn api_call(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value> {
        use std::io::{BufRead, BufReader, Read, Write as _};
        use std::os::unix::net::UnixStream;

        let stream = UnixStream::connect(&self.socket_path).context("connect to API socket")?;
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;

        let body_str = body.map(|b| b.to_string());
        let cl = body_str.as_ref().map(|b| b.len()).unwrap_or(0);

        let request = if let Some(ref b) = body_str {
            format!(
                "{method} {path} HTTP/1.1\r\n\
Host: localhost\r\n\
Accept: application/json\r\n\
Content-Type: application/json\r\n\
Content-Length: {cl}\r\n\
\r\n{b}"
            )
        } else {
            format!(
                "{method} {path} HTTP/1.1\r\n\
Host: localhost\r\n\
Accept: application/json\r\n\
\r\n"
            )
        };

        (&stream).write_all(request.as_bytes())?;

        let mut reader = BufReader::new(&stream);

        // Read status line
        let mut status_line = String::new();
        reader.read_line(&mut status_line)?;
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Read headers, extract Content-Length
        let mut content_length: usize = 0;
        loop {
            let mut header = String::new();
            reader.read_line(&mut header)?;
            let trimmed = header.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some(val) = trimmed.strip_prefix("Content-Length:") {
                content_length = val.trim().parse().unwrap_or(0);
            }
            // Also check lowercase
            if let Some(val) = trimmed.strip_prefix("content-length:") {
                content_length = val.trim().parse().unwrap_or(0);
            }
        }

        // Read exactly content_length bytes of body
        let body_text = if content_length > 0 {
            let mut buf = vec![0u8; content_length];
            reader.read_exact(&mut buf)?;
            String::from_utf8_lossy(&buf).to_string()
        } else {
            String::new()
        };

        // Explicitly close the connection
        drop(reader);
        drop(stream);

        if status_code >= 400 {
            bail!(
                "Firecracker API error {} on {} {}: {}",
                status_code,
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

    fn configure_machine(&self) -> Result<()> {
        self.api_call(
            "PUT",
            "/machine-config",
            Some(json!({
                "vcpu_count": self.config.vcpu_count,
                "mem_size_mib": self.config.mem_size_mib,
                "track_dirty_pages": self.config.track_dirty,
            })),
        )?;
        Ok(())
    }

    fn configure_boot_source(&self) -> Result<()> {
        let kernel = std::fs::canonicalize(&self.config.kernel_path)?;
        self.api_call(
            "PUT",
            "/boot-source",
            Some(json!({
                "kernel_image_path": kernel.to_str().unwrap(),
                "boot_args": self.config.boot_args,
            })),
        )?;
        Ok(())
    }

    fn configure_rootfs(&self) -> Result<()> {
        let rootfs = std::fs::canonicalize(&self.config.rootfs_path)?;
        self.api_call(
            "PUT",
            "/drives/rootfs",
            Some(json!({
                "drive_id": "rootfs",
                "path_on_host": rootfs.to_str().unwrap(),
                "is_root_device": true,
                "is_read_only": true,
            })),
        )?;
        Ok(())
    }

    fn configure_vsock(&self) -> Result<()> {
        // Relative path — each VM's cwd makes it unique
        self.api_call(
            "PUT",
            "/vsock",
            Some(json!({
                "vsock_id": "1",
                "guest_cid": 3,
                "uds_path": "v.sock",
            })),
        )?;
        Ok(())
    }

    fn start_instance(&mut self) -> Result<()> {
        self.api_call(
            "PUT",
            "/actions",
            Some(json!({
                "action_type": "InstanceStart",
            })),
        )?;
        self.state = VMState::Running;
        Ok(())
    }

    pub fn pause(&mut self) -> Result<()> {
        if self.state != VMState::Running {
            bail!("Can only pause a running VM, got {:?}", self.state);
        }
        self.api_call("PATCH", "/vm", Some(json!({"state": "Paused"})))?;
        self.state = VMState::Paused;
        Ok(())
    }

    pub fn resume(&mut self) -> Result<()> {
        if self.state != VMState::Paused {
            bail!("Can only resume a paused VM, got {:?}", self.state);
        }
        self.api_call("PATCH", "/vm", Some(json!({"state": "Resumed"})))?;
        self.state = VMState::Running;
        Ok(())
    }

    pub fn create_snapshot(&self, snapshot_dir: &Path) -> Result<Snapshot> {
        if self.state != VMState::Paused {
            bail!("VM must be paused before snapshotting");
        }
        let snap_dir =
            std::fs::canonicalize(std::fs::create_dir_all(snapshot_dir).map(|_| snapshot_dir)?)?;
        let vmstate_path = snap_dir.join("vmstate");
        let memory_path = snap_dir.join("memory");

        self.api_call(
            "PUT",
            "/snapshot/create",
            Some(json!({
                "snapshot_type": "Full",
                "snapshot_path": vmstate_path.to_str().unwrap(),
                "mem_file_path": memory_path.to_str().unwrap(),
            })),
        )?;

        Ok(Snapshot {
            vmstate_path,
            memory_path,
            rootfs_path: self.rootfs_path.clone(),
            config: self.config.clone(),
        })
    }

    /// Restore a VM from a snapshot.
    pub fn restore_from_snapshot(
        snapshot: &Snapshot,
        work_dir: PathBuf,
        vm_id: &str,
        rootfs_path: Option<PathBuf>,
        resume: bool,
    ) -> Result<Self> {
        let mut vm = Self::new(snapshot.config.clone(), Some(work_dir), vm_id)?;
        vm.rootfs_path = rootfs_path.unwrap_or_else(|| snapshot.rootfs_path.clone());
        vm.spawn_process()?;
        vm.load_snapshot(snapshot, resume)?;
        Ok(vm)
    }

    /// Create a VM with a spawned Firecracker process, ready for snapshot load.
    /// The process is started and the API socket is verified, but no VM is configured.
    pub fn new_idle(config: VMConfig, work_dir: PathBuf, vm_id: &str) -> Result<Self> {
        let mut vm = Self::new(config, Some(work_dir), vm_id)?;
        vm.spawn_process()?;
        Ok(vm)
    }

    /// Load a snapshot into an already-spawned Firecracker process.
    pub fn load_snapshot(&mut self, snapshot: &Snapshot, resume: bool) -> Result<()> {
        self.rootfs_path = snapshot.rootfs_path.clone();
        self.api_call(
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
        )?;
        self.state = if resume {
            VMState::Running
        } else {
            VMState::Paused
        };
        Ok(())
    }

    /// Get the absolute path to the vsock UDS.
    pub fn vsock_uds_path(&self) -> PathBuf {
        self.work_dir.join("v.sock")
    }

    /// Get a handle to the process's stdout (serial console).
    pub fn serial_stdout(&mut self) -> Option<impl BufRead + '_> {
        self.process
            .as_mut()
            .and_then(|p| p.stdout.take())
            .map(BufReader::new)
    }

    /// Take ownership of the process's stdout handle.
    /// Unlike serial_stdout(), returns the raw ChildStdout without BufReader wrapping.
    pub fn process_stdout(&mut self) -> Option<std::process::ChildStdout> {
        self.process.as_mut().and_then(|p| p.stdout.take())
    }

    pub fn destroy(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
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
