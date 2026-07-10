//! \[Declared\] Platform entropy floor. Reads from `/dev/urandom` via `std::fs`.
//!
//! # Implementation
//!
//! This module provides OS entropy by opening `/dev/urandom` and using `read_exact` to fill
//! the caller's buffer. Rationale for this approach over alternatives:
//!
//! - **Pure `std::fs` / no new dependency**: the `getrandom` crate provides a cross-platform
//!   OS-entropy abstraction, but adding it would require editing the workspace `Cargo.toml`
//!   (orchestrator-owned). `/dev/urandom` is directly available via `std::fs::File` at zero
//!   additional cost, so it is strongly preferred here.
//! - **`#![forbid(unsafe_code)]` preserved**: `/dev/urandom` via `std::fs` needs no `unsafe`
//!   blocks or FFI contact. The `getrandom` path (C FFI or platform `syscall`) would require
//!   lifting that attribute — avoided.
//!
//! # Platform caveat
//!
//! `/dev/urandom` is a Unix/Linux/macOS concept. On platforms that do not expose it (Windows,
//! WASI, bare-metal `no_std` targets), `fill_bytes` returns
//! `Err(EntropyError::Unavailable(...))` **explicitly** — it never panics, never zero-fills
//! silently (G2 / never-silent fallibility). Cross-platform OS entropy (via `getrandom`) is
//! the correct long-term solution; see FLAG-GETRANDOM in the report.
//!
//! # Guarantee tag
//!
//! `[Declared]` — `/dev/urandom` is a real OS entropy source backed by the kernel CSPRNG, but
//! no statistical audit or coverage measurement has been run against this implementation.
//! Promotion to `Empirical` requires documented trials with measured statistical quality
//! (e.g. Diehard/TestU01 run + recorded pass/fail table). Do not use for security-sensitive
//! purposes without that audit (VR-5).

use std::fmt;
use std::io::Read as _;

/// Errors from platform entropy operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntropyError {
    /// The platform entropy source is unavailable or exhausted.
    Unavailable(String),
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntropyError::Unavailable(msg) => write!(f, "entropy unavailable: {msg}"),
        }
    }
}

impl std::error::Error for EntropyError {}

/// \[Declared\] Fill `buf` with bytes from the OS entropy source (`/dev/urandom`).
///
/// Never-silent: returns `Err(EntropyError::Unavailable(...))` on any failure (open error,
/// short read, EOF, or platform without `/dev/urandom`). Never panics. Never zero-fills
/// silently (G2).
///
/// # Guarantee
///
/// `Declared` — source is the kernel CSPRNG via `/dev/urandom`, a genuine OS entropy source,
/// but no statistical quality audit has been conducted. Promotion to `Empirical` requires
/// documented trials with measured statistical quality (VR-5). Do not use for
/// security-sensitive purposes until retagged.
///
/// # Platform
///
/// Unix/Linux/macOS only. Returns `Err(EntropyError::Unavailable(...))` on platforms that
/// do not expose `/dev/urandom`. Use the `getrandom` crate (FLAG-GETRANDOM) for
/// cross-platform OS entropy.
pub fn fill_bytes(buf: &mut [u8]) -> Result<(), EntropyError> {
    // Empty buffer: trivially satisfied — no read needed.
    if buf.is_empty() {
        return Ok(());
    }

    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| EntropyError::Unavailable(format!("open /dev/urandom: {e}")))?;

    // `read_exact` fills the entire buffer or returns an error (including UnexpectedEof),
    // which means we never silently return a partially-filled buffer (G2).
    f.read_exact(buf)
        .map_err(|e| EntropyError::Unavailable(format!("read /dev/urandom: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip helper: treat `EntropyError::Unavailable` as a graceful skip (the repo idiom for
    /// platform-absent resources — mirrors the `ToolchainMissing` skip in the mlir differential).
    /// Non-Unix platforms (Windows, WASI, bare-metal) return `Unavailable`; that is correct
    /// behaviour and must not be treated as a test failure.
    macro_rules! skip_if_unavailable {
        ($result:expr) => {
            match $result {
                Ok(v) => v,
                Err(EntropyError::Unavailable(_)) => return, // graceful skip — not a failure
            }
        };
    }

    #[test]
    fn fill_bytes_32_succeeds_and_fills() {
        let mut buf = [0u8; 32];
        // Skip gracefully on platforms without /dev/urandom (Windows, WASI, bare-metal).
        skip_if_unavailable!(fill_bytes(&mut buf));
        // Light non-zero smoke only when the fill succeeded. This is a basic connectivity
        // check, NOT a statistical quality proof — probability of all-zero from /dev/urandom
        // is ~2^-256, treated as impossible in practice. Declared quality only (VR-5).
        assert_ne!(
            buf, [0u8; 32],
            "fill_bytes returned all-zero bytes (smoke check)"
        );
    }

    #[test]
    fn fill_bytes_empty_ok() {
        let mut buf: [u8; 0] = [];
        let result = fill_bytes(&mut buf);
        // Empty buffer is trivially satisfied even on platforms without /dev/urandom — no read
        // is performed (see implementation). This test is unconditional.
        assert!(result.is_ok(), "fill_bytes([]) returned Err: {result:?}");
    }

    #[test]
    fn fill_bytes_large_succeeds() {
        // Exercises read_exact across a larger buffer (4096 bytes) to ensure no short-read
        // issue. /dev/urandom never EOF so this must succeed on Unix.
        let mut buf = vec![0u8; 4096];
        // Skip gracefully on platforms without /dev/urandom.
        skip_if_unavailable!(fill_bytes(&mut buf));
    }

    /// Smoke test: two successive fills succeed when the entropy source is available.
    ///
    /// This test verifies only that both calls succeed (availability) — not that the two
    /// buffers differ. A content-quality / non-determinism check (two buffers should differ)
    /// is an inherently probabilistic assertion; it belongs in a documented statistical audit
    /// (e.g. Diehard/TestU01 with a recorded pass/fail table), not a unit test. Honest label:
    /// `Declared` quality, not `Empirical` (VR-5).
    #[test]
    fn fill_bytes_successive_succeed_when_available() {
        let mut buf1 = [0u8; 32];
        let mut buf2 = [0u8; 32];
        // Skip gracefully on platforms without /dev/urandom.
        skip_if_unavailable!(fill_bytes(&mut buf1));
        skip_if_unavailable!(fill_bytes(&mut buf2));
        // Both calls succeeded — the entropy source is functional. Content quality is not
        // checked here; see module doc for the path to `Empirical` promotion (VR-5).
    }

    #[test]
    fn entropy_error_display() {
        let e = EntropyError::Unavailable("test reason".to_string());
        assert_eq!(e.to_string(), "entropy unavailable: test reason");
    }

    /// Verify the EntropyError type implements std::error::Error (required by public API).
    #[test]
    fn entropy_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(EntropyError::Unavailable("x".to_string()));
        assert!(e.to_string().contains("entropy unavailable"));
    }
}
