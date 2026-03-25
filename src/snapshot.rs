use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

use crate::vm::VMConfig;

/// A captured VM state that can spawn new VMs via restore.
///
/// Immutable after creation — multiple VMs can restore from the same
/// snapshot concurrently (each gets CoW memory via MAP_PRIVATE).
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub vmstate_path: PathBuf,
    pub memory_path: PathBuf,
    pub rootfs_path: PathBuf,
    pub config: VMConfig,
}

impl Snapshot {
    /// Create a writable copy of the rootfs for a child VM.
    /// Uses `cp --reflink=auto` for CoW if the filesystem supports it.
    pub fn fork_rootfs(&self, dest: &PathBuf) -> Result<()> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let status = Command::new("cp")
            .args([
                "--reflink=auto",
                self.rootfs_path.to_str().unwrap(),
                dest.to_str().unwrap(),
            ])
            .status()?;
        if !status.success() {
            // Fallback to regular copy
            std::fs::copy(&self.rootfs_path, dest)?;
        }
        Ok(())
    }

    /// Remove snapshot files from disk.
    pub fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.vmstate_path);
        let _ = std::fs::remove_file(&self.memory_path);
    }
}
