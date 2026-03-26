use std::path::PathBuf;

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
    /// Remove snapshot files from disk.
    pub fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.vmstate_path);
        let _ = std::fs::remove_file(&self.memory_path);
    }
}
