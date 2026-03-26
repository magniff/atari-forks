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
    /// Remove only the vmstate file, keeping the memory file alive.
    ///
    /// Used with diff snapshots: the memory file is reused across
    /// iterations as the base for subsequent diffs, so we can't
    /// delete it. The vmstate is tiny and unique per snapshot.
    pub fn cleanup_vmstate(&self) {
        let _ = std::fs::remove_file(&self.vmstate_path);
    }
}
