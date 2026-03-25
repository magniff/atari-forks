#!/bin/bash
#
# setup.sh - Prepare everything needed to run the Atari forking benchmark.
#
# This script:
#   1. Downloads the Firecracker binary        (no root)
#   2. Downloads a minimal guest kernel         (no root)
#   3. Builds a rootfs image with Python, ALE, and the guest agent
#   4. Verifies KVM is available
#
# Step 3 (rootfs) requires privileged operations (mount, chroot). You have
# two options:
#
#   Option A — Docker (recommended, no root needed if your user is in the
#              docker group):
#       ./setup.sh --rootfs-method=docker
#
#   Option B — Direct host mount+chroot (needs sudo, only for step 3):
#       ./setup.sh --rootfs-method=chroot
#       The script will call sudo only for the mount/chroot/umount phase.
#
# Prerequisites:
#   - Ubuntu 22.04+ (tested on Ubuntu 24.04)
#   - KVM access (check: ls -la /dev/kvm)
#   - docker OR (debootstrap + sudo) for rootfs creation
#   - ~2GB free disk space
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh                        # auto-detects best method
#   ./setup.sh --rootfs-method=docker # explicit docker
#   ./setup.sh --rootfs-method=chroot # explicit sudo mount+chroot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${SCRIPT_DIR}/deps"
ARCH=$(uname -m)

# Firecracker version to use
FC_VERSION="v1.10.1"

# Parse --rootfs-method flag
ROOTFS_METHOD=""
for arg in "$@"; do
    case "$arg" in
        --rootfs-method=*) ROOTFS_METHOD="${arg#*=}" ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err() { echo -e "${RED}[!]${NC} $*" >&2; }

# ── Step 0: Check prerequisites ──────────────────────────────────────

check_prerequisites() {
    log "Checking prerequisites..."

    if [ ! -e /dev/kvm ]; then
        err "KVM not available. Make sure:"
        err "  1. Your CPU supports virtualization (VT-x/AMD-V)"
        err "  2. It's enabled in BIOS"
        err "  3. The kvm module is loaded: sudo modprobe kvm kvm_intel (or kvm_amd)"
        exit 1
    fi

    if [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
        warn "/dev/kvm exists but is not readable/writable by $(whoami)."
        warn "Fix with: sudo usermod -aG kvm $(whoami)  (then re-login)"
    fi

    for cmd in curl tar dd; do
        if ! command -v "$cmd" &>/dev/null; then
            err "Required command not found: $cmd"
            exit 1
        fi
    done

    log "Prerequisites OK"
}

# ── Step 1: Download Firecracker ──────────────────────────────────────

download_firecracker() {
    local fc_bin="${DEPS_DIR}/firecracker"

    if [ -f "$fc_bin" ]; then
        log "Firecracker already downloaded"
        return
    fi

    log "Downloading Firecracker ${FC_VERSION} for ${ARCH}..."
    mkdir -p "$DEPS_DIR"

    local url="https://github.com/firecracker-microvm/firecracker/releases/download/${FC_VERSION}/firecracker-${FC_VERSION}-${ARCH}.tgz"
    curl -fSL "$url" | tar -xz -C "$DEPS_DIR"

    # The tarball extracts to a versioned directory
    mv "${DEPS_DIR}/release-${FC_VERSION}-${ARCH}/firecracker-${FC_VERSION}-${ARCH}" "$fc_bin"
    chmod +x "$fc_bin"

    # Also grab the jailer (not strictly needed but useful)
    if [ -f "${DEPS_DIR}/release-${FC_VERSION}-${ARCH}/jailer-${FC_VERSION}-${ARCH}" ]; then
        mv "${DEPS_DIR}/release-${FC_VERSION}-${ARCH}/jailer-${FC_VERSION}-${ARCH}" "${DEPS_DIR}/jailer"
        chmod +x "${DEPS_DIR}/jailer"
    fi

    rm -rf "${DEPS_DIR}/release-${FC_VERSION}-${ARCH}"
    log "Firecracker downloaded to ${fc_bin}"
}

# ── Step 2: Download guest kernel ────────────────────────────────────

download_kernel() {
    local kernel="${DEPS_DIR}/vmlinux"

    if [ -f "$kernel" ]; then
        log "Guest kernel already downloaded"
        return
    fi

    log "Downloading guest kernel (6.1)..."
    mkdir -p "$DEPS_DIR"

    # Use Firecracker's CI 6.1 kernel — supports random.trust_cpu=on
    # which is needed to avoid entropy starvation in minimal VMs
    local url="https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/v1.10/${ARCH}/vmlinux-6.1.102"
    curl -fSL "$url" -o "$kernel"

    log "Kernel downloaded to ${kernel}"
}

# ── Step 3: Build rootfs with ALE ────────────────────────────────────

build_rootfs() {
    local rootfs="${DEPS_DIR}/rootfs.ext4"

    if [ -f "$rootfs" ]; then
        log "Rootfs already built"
        return
    fi

    # Auto-detect method if not specified
    local method="$ROOTFS_METHOD"
    if [ -z "$method" ]; then
        if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
            method="docker"
        elif command -v podman &>/dev/null; then
            method="podman"
        elif command -v debootstrap &>/dev/null; then
            method="chroot"
        else
            err "Cannot build rootfs. Install one of:"
            err "  - podman (pacman -S podman, no root needed)"
            err "  - docker"
            err "  - debootstrap (needs sudo for mount/chroot)"
            exit 1
        fi
    fi

    log "Building rootfs via method: ${method}"

    case "$method" in
        docker)  build_rootfs_docker "$rootfs" ;;
        podman)  CONTAINER_CMD=podman build_rootfs_docker "$rootfs" ;;
        chroot)  build_rootfs_chroot "$rootfs" ;;
        *)       err "Unknown rootfs method: $method"; exit 1 ;;
    esac
}

# ── Method A: Build rootfs using Docker/Podman (no root needed) ──────
#
# This creates a container with all dependencies, exports its
# filesystem as a tarball, then packs it into an ext4 image.
# Works with docker or podman. No mount/chroot/sudo required
# if fuse2fs is available for the final step.

build_rootfs_docker() {
    local rootfs="$1"
    local cmd="${CONTAINER_CMD:-docker}"

    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd not found."
        exit 1
    fi

    log "Creating rootfs via ${cmd}..."

    # Write a Dockerfile that installs everything the guest needs
    local build_dir
    build_dir=$(mktemp -d)
    cp "${SCRIPT_DIR}/guest/atari_agent.py" "$build_dir/"

    cat > "$build_dir/Dockerfile" << 'DOCKERFILE'
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv && \
    pip3 install --break-system-packages ale-py gymnasium pillow numpy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify ALE actually works
RUN PYTHONPATH=/usr/local/lib/python3.11/dist-packages \
    python3 -c "from ale_py import ALEInterface; print('ALE OK')"

COPY atari_agent.py /usr/local/bin/atari_agent.py
RUN chmod +x /usr/local/bin/atari_agent.py

# Create a minimal init script.
# Sets PYTHONPATH so pip-installed packages in /usr/local are found.
# Mounts tmpfs at /tmp and /run so the read-only rootfs works.
RUN printf '#!/bin/sh\n\
mount -t proc proc /proc\n\
mount -t sysfs sys /sys\n\
mount -t devtmpfs dev /dev\n\
mount -t tmpfs tmpfs /tmp\n\
mount -t tmpfs tmpfs /run\n\
export PYTHONPATH=/usr/local/lib/python3.11/dist-packages\n\
exec /usr/bin/python3 /usr/local/bin/atari_agent.py\n' > /opt/init.sh && \
    chmod +x /opt/init.sh
DOCKERFILE

    # Build the container
    local tag="atari-fork-rootfs"
    $cmd build -t "$tag" "$build_dir"

    # Export the container filesystem as a tarball
    local container_id
    container_id=$($cmd create "$tag")
    local tarball="$build_dir/rootfs.tar"
    $cmd export "$container_id" -o "$tarball"
    $cmd rm "$container_id" > /dev/null

    # Create an ext4 image and unpack the tarball into it.
    # This part needs either:
    #   a) fuse2fs (from e2fsprogs, no root needed), or
    #   b) sudo mount -o loop (fallback)
    local img_size_mb=1024
    dd if=/dev/zero of="$rootfs" bs=1M count=$img_size_mb status=progress 2>&1
    mkfs.ext4 -F "$rootfs" > /dev/null 2>&1

    if command -v fuse2fs &>/dev/null; then
        # fuse2fs: mount ext4 image in userspace — no root!
        log "Using fuse2fs to populate rootfs (no root needed)"
        local fuse_mount
        fuse_mount=$(mktemp -d)
        fuse2fs -o fakeroot "$rootfs" "$fuse_mount"
        tar -xf "$tarball" -C "$fuse_mount"
        fusermount -u "$fuse_mount"
        rmdir "$fuse_mount"
    elif command -v debugfs &>/dev/null; then
        # debugfs fallback: slower but also rootless
        log "fuse2fs not found, using sudo mount as fallback"
        log "  (install fuse2fs to avoid sudo: sudo apt install fuse2fs)"
        local mount_point
        mount_point=$(mktemp -d)
        sudo mount -o loop "$rootfs" "$mount_point"
        sudo tar -xf "$tarball" -C "$mount_point"
        sudo umount "$mount_point"
        rmdir "$mount_point"
    else
        log "Falling back to sudo mount to populate the ext4 image"
        local mount_point
        mount_point=$(mktemp -d)
        sudo mount -o loop "$rootfs" "$mount_point"
        sudo tar -xf "$tarball" -C "$mount_point"
        sudo umount "$mount_point"
        rmdir "$mount_point"
    fi

    # Cleanup
    $cmd rmi "$tag" > /dev/null 2>&1 || true
    rm -rf "$build_dir"

    log "Rootfs built at ${rootfs}"
}

# ── Method B: Build rootfs via debootstrap + chroot (needs sudo) ─────
#
# This is the traditional approach: create an ext4 image, mount it,
# debootstrap a Debian system into it, chroot to install packages.
# Requires: sudo, debootstrap, mkfs.ext4
#
# The script calls sudo explicitly for just the privileged operations
# (mount, chroot, umount) — it does NOT require running the whole
# script as root.

build_rootfs_chroot() {
    local rootfs="$1"

    for cmd in mkfs.ext4 debootstrap; do
        if ! command -v "$cmd" &>/dev/null; then
            err "$cmd not found. Install it: sudo apt install ${cmd}"
            exit 1
        fi
    done

    log "Creating rootfs via debootstrap + chroot (will use sudo for mount/chroot/umount)..."

    # Create the ext4 image — no root needed for this
    local img_size_mb=1024
    dd if=/dev/zero of="$rootfs" bs=1M count=$img_size_mb status=progress 2>&1
    mkfs.ext4 -F "$rootfs" > /dev/null 2>&1

    # Everything below needs root. We use sudo per-command so the user
    # sees exactly what runs as root.
    local mount_point
    mount_point=$(mktemp -d)

    log "Mounting image (sudo required)..."
    sudo mount -o loop "$rootfs" "$mount_point"

    # Trap to ensure we always unmount, even on failure
    trap "sudo umount '$mount_point' 2>/dev/null; rmdir '$mount_point' 2>/dev/null" EXIT

    log "Running debootstrap (sudo required)..."
    sudo debootstrap --variant=minbase \
        --include=python3,python3-pip,python3-venv \
        bookworm "$mount_point" http://deb.debian.org/debian

    log "Installing ALE and dependencies (sudo required for chroot)..."
    sudo chroot "$mount_point" /bin/bash -c "
        pip3 install --break-system-packages ale-py gymnasium pillow numpy
        PYTHONPATH=/usr/local/lib/python3.11/dist-packages python3 -c 'from ale_py import ALEInterface; print(\"ALE installed successfully\")'
    "

    # Copy guest agent
    sudo cp "${SCRIPT_DIR}/guest/atari_agent.py" "$mount_point/usr/local/bin/atari_agent.py"
    sudo chmod +x "$mount_point/usr/local/bin/atari_agent.py"

    # Create minimal init script (kernel boots directly into this via init=)
    sudo mkdir -p "$mount_point/opt"
    sudo tee "$mount_point/opt/init.sh" > /dev/null << 'INITEOF'
#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev
mount -t tmpfs tmpfs /tmp
mount -t tmpfs tmpfs /run
export PYTHONPATH=/usr/local/lib/python3.11/dist-packages
exec /usr/bin/python3 /usr/local/bin/atari_agent.py
INITEOF
    sudo chmod +x "$mount_point/opt/init.sh"

    log "Unmounting image..."
    sudo umount "$mount_point"
    rmdir "$mount_point"
    trap - EXIT

    log "Rootfs built at ${rootfs}"
}

# ── Step 4: Verify setup ─────────────────────────────────────────────

verify_setup() {
    log "Verifying setup..."

    local fc_bin="${DEPS_DIR}/firecracker"
    local kernel="${DEPS_DIR}/vmlinux"
    local rootfs="${DEPS_DIR}/rootfs.ext4"

    local ok=true

    for f in "$fc_bin" "$kernel" "$rootfs"; do
        if [ -f "$f" ]; then
            log "  ✓ $(basename "$f") ($(du -h "$f" | cut -f1))"
        else
            err "  ✗ $(basename "$f") not found"
            ok=false
        fi
    done

    if $ok; then
        log ""
        log "Setup complete! Run the benchmark with:"
        log ""
        log "  python3 tree_search.py \\"
        log "    --firecracker ${fc_bin} \\"
        log "    --kernel ${kernel} \\"
        log "    --rootfs ${rootfs} \\"
        log "    --iterations 10"
        log ""
    else
        err "Setup incomplete. Check errors above."
        exit 1
    fi
}

# ── Main ─────────────────────────────────────────────────────────────

main() {
    log "Atari Forking Challenge - Setup"
    log "==============================="
    echo

    check_prerequisites
    download_firecracker
    download_kernel
    build_rootfs
    verify_setup
}

main "$@"