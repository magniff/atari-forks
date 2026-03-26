use std::path::PathBuf;

use crate::vm::VMConfig;

/// A captured VM state that can spawn new VMs via restore.
///
/// Immutable after creation — multiple VMs can restore from the same
/// snapshot concurrently:
///   - Memory: Firecracker maps the file MAP_PRIVATE, giving each
///     child its own CoW pages. No copy needed.
///   - Rootfs: mounted read-only by all children. The Atari agent
///     doesn't write to disk, so this is safe.
///   - VM state: small file (~1KB), read-only after creation.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub vmstate_path: PathBuf,
    pub memory_path: PathBuf,
    pub rootfs_path: PathBuf,
    pub config: VMConfig,
}

impl Snapshot {
    /// Remove snapshot files (vmstate + memory dump) from disk.
    ///
    /// Call this after all children restored from this snapshot have
    /// been destroyed or have finished loading. The memory file may
    /// still be mapped by child processes (MAP_PRIVATE), so the OS
    /// keeps the pages alive until the last mapping is dropped — it's
    /// safe to unlink eagerly.
    pub fn cleanup(&self) {
        let _ = std::fs::remove_file(&self.vmstate_path);
        let _ = std::fs::remove_file(&self.memory_path);
        // Also clean the snapshot directory if it's now empty
        if let Some(parent) = self.vmstate_path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
    }
}
