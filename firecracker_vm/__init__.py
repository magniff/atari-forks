"""
firecracker_vm - A library for managing Firecracker microVM snapshots and forks.

This library provides a clean abstraction over Firecracker's REST API for:
- Booting and configuring microVMs
- Creating snapshots (full VM state: memory + CPU + devices)
- Restoring VMs from snapshots (with copy-on-write memory mapping)
- Forking: creating multiple child VMs from a single parent snapshot

The key insight for performance: Firecracker uses MAP_PRIVATE for snapshot
memory, meaning restored VMs get copy-on-write pages. Only pages that the
child actually modifies get physically copied. This makes forking very cheap
in terms of both time and memory.

Architecture:
    FirecrackerVM   - manages a single Firecracker process and its API socket
    Snapshot        - represents a captured VM state (vmstate + memory files)
    VMPool          - manages multiple VMs for parallel forking operations
"""

from firecracker_vm.vm import FirecrackerVM, VMConfig
from firecracker_vm.snapshot import Snapshot
from firecracker_vm.pool import VMPool

__all__ = ["FirecrackerVM", "VMConfig", "Snapshot", "VMPool"]
