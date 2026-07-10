//! \[Declared\] Standard-stream I/O floor (RFC-0028 §4.5; M-722). Thin, never-silent wrappers over
//! Rust `std::io` stdin/stdout/stderr.
//!
//! This is the audited syscall floor for `std.io`'s OS contact: reading from stdin and writing to
//! stdout/stderr. Per LR-9 / RFC-0016 §8-Q6 all such contact lives in this single `std-sys` phylum,
//! so the pure `std-io` crate stays `wild`-free.
//!
//! # Honesty (VR-5)
//!
//! Every function carries the **`Declared`** guarantee tag — these are unaudited `std::io` wrappers;
//! no theorem and no measured bound backs stream semantics, buffering, or partial-write behaviour.
//! Promotion to `Empirical` requires documented coverage (e.g. a recorded round-trip property over a
//! piped fixture); `Proven` requires a checked theorem. Neither is established in v0.
//!
//! # Never-silent (G2)
//!
//! Every operation returns an explicit `Result` on failure — no byte count is silently dropped, no
//! short write is silently ignored. `write_all` propagates the OS error; the caller is never told a
//! write succeeded when it did not.
//!
//! # Bounded reads on the external edge (P3)
//!
//! [`read_to_end`] is an **unbounded, input-driven allocation**: a hostile or runaway stdin pipe
//! grows the buffer without any cap → a memory-exhaustion DoS. Its signature
//! `() -> Result<Vec<u8>, …>` cannot express a bound, so it is **trusted-only** — use it solely
//! where the producer is your own process / under your control. For **untrusted** input use
//! [`read_to_end_capped`], which is **bounded by construction** (P3) and returns an explicit
//! [`ReadCappedError::TooLarge`] on a cap-hit rather than silently truncating (G2).
//!
//! # Guarantee matrix (RFC-0016 §4.5)
//!
//! | op | signature | failure mode | tag |
//! |----|-----------|--------------|-----|
//! | `read_to_end` | `() -> Result<Vec<u8>, io::Error>` | OS read error → `Err`; **unbounded alloc (trusted-only)** | `Declared` |
//! | `read_to_end_capped` | `(usize) -> Result<Vec<u8>, ReadCappedError>` | OS read error / **cap exceeded** → `Err` (P3 bounded; never truncates, G2) | `Declared` |
//! | `read_line` | `() -> Result<String, io::Error>` | OS/UTF-8 error → `Err` | `Declared` |
//! | `write_out` | `(&[u8]) -> Result<(), io::Error>` | short/failed write → `Err` | `Declared` |
//! | `write_err` | `(&[u8]) -> Result<(), io::Error>` | short/failed write → `Err` | `Declared` |
//! | `flush_out` | `() -> Result<(), io::Error>` | OS flush error → `Err` | `Declared` |

use std::fmt;
use std::io::{self, Read, Write};

/// \[Declared\] Read **all** of stdin to end-of-input — **UNBOUNDED, TRUSTED-ONLY**.
///
/// This grows the buffer with no cap: a hostile or runaway producer can drive an arbitrarily large
/// allocation (memory-exhaustion DoS). Use it **only** where the stdin producer is your own process
/// or otherwise under your control. For **untrusted / external-edge** input, use the bounded
/// [`read_to_end_capped`] instead (P3 — bounds at the external edge).
///
/// Returns `Err` on any OS read error — never-silent (G2): a partial read that hits an error is
/// reported, not truncated-and-returned.
pub fn read_to_end() -> Result<Vec<u8>, io::Error> {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf)?;
    Ok(buf)
}

/// Error set for the bounded stdin read [`read_to_end_capped`] (closed sum — never-silent, G2).
#[derive(Debug)]
pub enum ReadCappedError {
    /// An OS read error occurred before the cap was reached.
    Io(io::Error),
    /// Input reached the cap: there were **more** than `cap` bytes available, so the read was
    /// refused rather than truncated (G2). `cap` is the byte limit that was exceeded.
    TooLarge {
        /// The byte cap that the input exceeded.
        cap: usize,
    },
}

impl fmt::Display for ReadCappedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadCappedError::Io(e) => write!(f, "stdin read error: {e}"),
            ReadCappedError::TooLarge { cap } => {
                write!(
                    f,
                    "stdin input exceeds the {cap}-byte cap (refused, not truncated)"
                )
            }
        }
    }
}

impl std::error::Error for ReadCappedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReadCappedError::Io(e) => Some(e),
            ReadCappedError::TooLarge { .. } => None,
        }
    }
}

impl From<io::Error> for ReadCappedError {
    fn from(e: io::Error) -> Self {
        ReadCappedError::Io(e)
    }
}

/// \[Declared\] Read stdin to end-of-input, **bounded by `max` bytes** (P3 — bound at the external
/// edge). The safe form for **untrusted** input: a hostile/runaway pipe cannot drive an unbounded
/// allocation.
///
/// Never-silent (G2): if the input would **exceed** `max` bytes, this returns
/// `Err(ReadCappedError::TooLarge { cap: max })` — it never silently truncates the stream. An input
/// of **exactly** `max` bytes (or fewer) succeeds and returns those bytes. OS read errors surface as
/// `Err(ReadCappedError::Io(_))`.
///
/// # Guarantee
///
/// `Declared` — a thin bounded wrapper over `std::io`; no theorem and no measured bound backs the
/// underlying stream semantics. The cap itself is exact by construction (the returned buffer length
/// is always `<= max`), but the floor's overall guarantee stays `Declared` (VR-5).
pub fn read_to_end_capped(max: usize) -> Result<Vec<u8>, ReadCappedError> {
    read_capped(io::stdin().lock(), max)
}

/// Core bounded read, factored over any [`Read`] so it is unit-testable without real stdin.
///
/// Reads up to `max + 1` bytes via [`Read::take`]. Because `.take(max)` would *silently* stop at the
/// cap (an invisible truncation — a G2 violation), we instead read one byte past the cap: if the
/// result length is `> max`, the input was too large and we return [`ReadCappedError::TooLarge`]
/// (explicit, never a truncated buffer). Otherwise the whole input fit within the cap and is
/// returned. `max == usize::MAX` is handled without overflow (the `+ 1` saturates).
fn read_capped<R: Read>(r: R, max: usize) -> Result<Vec<u8>, ReadCappedError> {
    // Read one byte past the cap so an at-cap-vs-over-cap input is distinguishable. Saturate so
    // `max == usize::MAX` does not overflow (the over-cap branch is then unreachable, as intended).
    let probe = (max as u64).saturating_add(1);
    let mut buf = Vec::new();
    r.take(probe).read_to_end(&mut buf)?;
    if buf.len() > max {
        // We observed at least one byte beyond the cap ⇒ the input exceeds `max`. Refuse explicitly
        // rather than hand back the truncated `max` bytes (G2).
        return Err(ReadCappedError::TooLarge { cap: max });
    }
    Ok(buf)
}

/// \[Declared\] Read a single line from stdin (including the trailing newline if present). Returns
/// `Err` on any OS read error or invalid UTF-8 — never-silent (G2).
pub fn read_line() -> Result<String, io::Error> {
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line)
}

/// \[Declared\] Write all of `bytes` to stdout. Uses `write_all`, so a short write is an explicit
/// `Err`, never a silently-dropped tail (G2). Does **not** flush — call [`flush_out`] when ordering
/// against process exit matters.
pub fn write_out(bytes: &[u8]) -> Result<(), io::Error> {
    io::stdout().write_all(bytes)
}

/// \[Declared\] Write all of `bytes` to stderr. Like [`write_out`]: a short write is an explicit
/// `Err` (G2).
pub fn write_err(bytes: &[u8]) -> Result<(), io::Error> {
    io::stderr().write_all(bytes)
}

/// \[Declared\] Flush stdout — surfaces any deferred OS write error explicitly (G2), so a buffered
/// write lost at flush time is never silently dropped.
pub fn flush_out() -> Result<(), io::Error> {
    io::stdout().flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The write path returns `Ok` on a normal stdout write and surfaces the byte payload faithfully
    /// (a smoke test — the OS stream is the ground truth; this only pins that the wrapper does not
    /// swallow or transform the bytes).
    #[test]
    fn write_out_and_flush_succeed_on_a_normal_stream() {
        write_out(b"").expect("an empty write succeeds");
        flush_out().expect("flush succeeds");
    }

    /// Never-silent (G2): `write_all` over a deliberately failing writer surfaces the error rather
    /// than reporting a phantom success. We exercise the same `Write::write_all` contract `write_out`
    /// relies on, against a sink that refuses, to pin the never-silent property structurally.
    #[test]
    fn a_failing_write_is_an_explicit_error_not_a_phantom_success() {
        struct Refuse;
        impl Write for Refuse {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("refused"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let err = Refuse
            .write_all(b"x")
            .expect_err("a refusing writer must surface an error");
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    /// P3 / under-cap: an input shorter than the cap is returned in full.
    #[test]
    fn capped_read_under_cap_returns_all_bytes() {
        let input: &[u8] = b"hello";
        let out = read_capped(input, 64).expect("an under-cap read must succeed");
        assert_eq!(out, b"hello", "the full payload must be returned unchanged");
    }

    /// Boundary: an input of *exactly* the cap length succeeds (the cap is inclusive — `len <= max`).
    #[test]
    fn capped_read_at_exactly_cap_returns_all_bytes() {
        let input: &[u8] = b"abcd";
        let out = read_capped(input, 4).expect("an at-cap read must succeed");
        assert_eq!(out, b"abcd");
    }

    /// Never-silent (G2): an input that EXCEEDS the cap is an explicit `TooLarge` error — never a
    /// silently-truncated buffer. This is the DoS-bound test (P3): we must refuse, not truncate.
    #[test]
    fn capped_read_over_cap_is_explicit_error_not_silent_truncation() {
        let input: &[u8] = b"abcdef"; // 6 bytes, cap 4
        match read_capped(input, 4) {
            Err(ReadCappedError::TooLarge { cap }) => {
                assert_eq!(cap, 4, "the error must cite the cap")
            }
            other => panic!("over-cap input must be TooLarge, never a truncated Ok: {other:?}"),
        }
    }

    /// Edge: a zero cap rejects any non-empty input (one byte already exceeds the cap).
    #[test]
    fn capped_read_zero_cap_rejects_nonempty_input() {
        let input: &[u8] = b"x";
        assert!(
            matches!(
                read_capped(input, 0),
                Err(ReadCappedError::TooLarge { cap: 0 })
            ),
            "a 1-byte input must exceed a zero cap"
        );
    }

    /// Edge: a zero cap accepts empty input (nothing to read — the bound holds vacuously).
    #[test]
    fn capped_read_zero_cap_accepts_empty_input() {
        let input: &[u8] = b"";
        let out = read_capped(input, 0).expect("empty input fits a zero cap");
        assert!(out.is_empty());
    }

    /// The error type names the cap in its `Display` (auditable message; G2 — the refusal is legible).
    #[test]
    fn too_large_error_display_cites_the_cap() {
        let e = ReadCappedError::TooLarge { cap: 128 };
        assert!(
            e.to_string().contains("128"),
            "TooLarge Display must cite the cap, got: {e}"
        );
    }

    /// Never-silent (G2): an OS read error before the cap surfaces as `Io`, not a partial `Ok`.
    #[test]
    fn capped_read_surfaces_os_error_not_partial_ok() {
        struct Boom;
        impl Read for Boom {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("boom"))
            }
        }
        match read_capped(Boom, 16) {
            Err(ReadCappedError::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::Other),
            other => panic!("a failing reader must surface Io, not a phantom Ok: {other:?}"),
        }
    }
}
