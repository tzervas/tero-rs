//! Optional **memory-gate-rs** integration (`join/tero-memory-feature`).
//!
//! Learned memory (dense retrieval) is separate from Layer-1 citations: MG hits are never surfaced
//! as L1 [`crate::Citation`]s.

#[cfg(feature = "memory")]
mod handle;

#[cfg(feature = "memory")]
pub use handle::{
    envelope_consolidated, envelope_hits, envelope_memory_disabled, envelope_stored, MemoryHandle,
    MemoryOpenError,
};
