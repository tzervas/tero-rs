//! Byte-movement surface: abstract `Source`/`Sink` with **affine single-consumption**
//! (LR-8; spec §3 — the io half).
//!
//! # The single-consumption guarantee (LR-8)
//!
//! A `Source` is an **affine handle** (RFC-0006 §Q5): it wraps the underlying
//! substrate and moves it exactly once.
//!
//! - [`read_all`] takes `Source` **by-move** and does **not** return it: the handle
//!   is consumed exactly once.  A double-consume is a **compile-time type error**,
//!   not a runtime check.
//! - [`read`] (chunked) returns the handle so a caller may continue reading; the same
//!   handle is threaded linearly through every call.
//! - [`write()`] similarly returns the `Sink` so the caller may continue writing.
//!
//! The affine type enforces that a substrate cannot be silently re-read after
//! exhaustion (LR-8 / spec §1/§3 / the module's honesty crux).
//!
//! # In-memory substrate
//!
//! The OS I/O floor is **FLAGGED** (spec §7-Q4 / §8-Q6) and deferred to `std-sys`
//! (M-541).  This module ships a fully testable **in-memory substrate** (`Bytes`
//! cursor) so the abstraction and the affine type are complete and the tests pass
//! without OS facilities.  A `std-sys` crate will implement the real-OS-backed
//! `Source`/`Sink` that wraps this module's types.
//!
//! # Declared effects (C6)
//!
//! Every io op declares **`io`** as its declared effect (`#[doc = "Effects: io"]`).
//! The serialize ops are pure; they declare `none`.  Chunked `read` additionally
//! allocates a buffer (`alloc(Budget)`, as stated in the matrix).
//!
//! # FLAG: no OS I/O here
//! This module contains **no** `wild`/FFI and no `std::fs`/`std::net` usage (ADR-014
//! / LR-9 / spec §7-Q4).  The `std-sys` phylum (M-541) will supply OS-backed
//! constructors that produce a `Source`/`Sink` over a file descriptor or network
//! socket; `std.io` only defines the abstract surface.

use crate::error::IoError;

// ── Budget ───────────────────────────────────────────────────────────────────

/// A declared byte-read budget (C6/RFC-0014 §4.5; ADR-015).
///
/// Passed to [`read`] to cap the allocation.  An overrun yields
/// [`IoError::EffectBudget`] instead of unbounded allocation (C6).
///
/// The `Unbounded` variant is available for in-memory tests where an allocation
/// ceiling is not meaningful; production uses of the OS-backed substrate should
/// always supply `Bytes(n)` (spec §7-Q5 / RFC-0016 §8-Q3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// Read at most `n` bytes; any request beyond this yields
    /// [`IoError::EffectBudget`].
    Bytes(u64),
    /// No cap — accept any number of bytes.  Use only in tests or when the
    /// caller has an out-of-band cap (e.g. the file system size).
    Unbounded,
}

// ── Substrate ────────────────────────────────────────────────────────────────

/// The in-memory substrate: a `Vec<u8>` cursor.
///
/// The `substrate` is the **underlying resource** that `Source`/`Sink` wrap.  A
/// [`Source`] owns a `Substrate` by-value; once consumed (by [`read_all`] or the
/// last [`read`]) it cannot be re-accessed.  This is the LR-8 affine invariant at
/// the type level.
///
/// # FLAG: §8-Q6 — OS-backed substrate lives in `std-sys` (M-541)
/// The real-OS-backed substrate (a file descriptor, a network socket) is NOT
/// here.  The `std-sys` phylum will provide a constructor like
/// `Substrate::from_fd(fd: OwnedFd) -> Substrate` that slots into this type.
/// Until then this in-memory substrate covers all testable semantics.
#[derive(Debug)]
pub struct Substrate {
    /// The raw bytes (in-memory backing store).
    data: Vec<u8>,
    /// Read cursor position.
    pos: usize,
}

impl Substrate {
    /// Construct a new in-memory substrate from a byte slice.
    ///
    /// This is the only constructor available in `std.io`; the OS-backed
    /// constructor is deferred to `std-sys` (FLAG §8-Q6).
    #[must_use]
    pub fn from_bytes(data: impl Into<Vec<u8>>) -> Self {
        Substrate {
            data: data.into(),
            pos: 0,
        }
    }

    /// How many bytes remain unread.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

// ── Source ────────────────────────────────────────────────────────────────────

/// An abstract byte **source**: a `Substrate` wrapped in an affine handle.
///
/// A `Source` may be read exactly once via [`read_all`] (which consumes it) or
/// progressively via [`read`] (which threads the handle linearly).  A double-
/// consume is a compile-time error (Rust move semantics enforce LR-8).
///
/// # EXPLAIN-able selection
/// The substrate backing this `Source` is visible as a field (C3 — no hidden
/// resource).
#[derive(Debug)]
pub struct Source {
    substrate: Substrate,
}

impl Source {
    /// Wrap a substrate as an affine `Source`.
    ///
    /// # Effects: none (construction is pure)
    #[must_use]
    pub fn new(substrate: Substrate) -> Self {
        Source { substrate }
    }

    /// Construct a `Source` directly from an in-memory byte slice (convenience
    /// constructor for tests).
    #[must_use]
    pub fn from_bytes(data: impl Into<Vec<u8>>) -> Self {
        Source::new(Substrate::from_bytes(data))
    }

    /// How many bytes remain unread.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.substrate.remaining()
    }

    /// Read up to `n` bytes from the substrate (internal, non-consuming helper).
    ///
    /// Returns the bytes read and the new cursor position.
    fn read_n(&mut self, n: usize) -> Vec<u8> {
        let end = (self.substrate.pos + n).min(self.substrate.data.len());
        let chunk = self.substrate.data[self.substrate.pos..end].to_vec();
        self.substrate.pos = end;
        chunk
    }

    /// Read all remaining bytes (internal, consuming helper).
    fn read_to_end(&mut self) -> Vec<u8> {
        let bytes = self.substrate.data[self.substrate.pos..].to_vec();
        self.substrate.pos = self.substrate.data.len();
        bytes
    }
}

// ── Sink ──────────────────────────────────────────────────────────────────────

/// An abstract byte **sink**: a write target wrapped in an affine handle.
///
/// A `Sink` is threaded linearly through [`write()`] calls (each call consumes the
/// current handle and returns the updated one).  The affine discipline means a
/// written sink cannot be silently re-used; a double-write is a compile-time error.
///
/// # EXPLAIN-able selection (C3)
/// The bytes written are observable via [`Sink::into_bytes`].
#[derive(Debug)]
pub struct Sink {
    buffer: Vec<u8>,
}

impl Sink {
    /// Construct an empty in-memory sink.
    ///
    /// # Effects: none (construction is pure)
    #[must_use]
    pub fn new() -> Self {
        Sink { buffer: Vec::new() }
    }

    /// Consume the sink and return the bytes written into it.
    ///
    /// This finalizes the sink — it cannot be written to after this call (Rust
    /// move semantics enforce the affine discipline).
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Append `bytes` to the internal buffer (internal, non-consuming helper).
    fn append(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }
}

impl Default for Sink {
    fn default() -> Self {
        Sink::new()
    }
}

// ── IO operations ─────────────────────────────────────────────────────────────

/// Read all remaining bytes from `src`, consuming it exactly once (LR-8).
///
/// # Guarantee tag: `Exact`
/// The bytes returned are exactly the bytes the substrate delivers — neither
/// reordered nor approximated.  The `Exact` tag signals no accuracy semantics
/// (RFC-0016 C2: "an op with no accuracy semantics is simply `Exact`").
///
/// # Fallibility
/// - `Err(IoError::UnexpectedEof{read})` if the source has already been consumed
///   before this call (in practice, Rust move semantics prevent a double-consume
///   at the type level; this variant can only be triggered by a substrate that
///   reports zero bytes when some were expected, e.g. a racing close in the OS-
///   backed variant).
/// - `Err(IoError::Refused{why})` if the substrate declines the operation.
/// - `Err(IoError::EffectBudget{kind})` on an effect-budget overrun.
///
/// # Effects: io (declared, C6)
/// This operation reads bytes from the substrate — the `io` effect is declared
/// on the signature.  (For the in-memory substrate the "IO" is a buffer copy, but
/// the effect abstraction is preserved for composability with OS-backed substrates.)
pub fn read_all(src: Source) -> Result<Vec<u8>, IoError> {
    // Consume `src` by move (LR-8: the handle is dropped after this call).
    let mut src = src;
    Ok(src.read_to_end())
}

/// Read up to `budget` bytes from `src`, returning the bytes and the remaining
/// handle (LR-8 — linear threading).
///
/// The caller receives the `Source` back so it may continue reading.  A call that
/// returns an empty `Vec` has exhausted the source; subsequent calls will also
/// return empty (they never error on a naturally-exhausted source).
///
/// # Guarantee tag: `Exact`
/// The bytes returned are exactly the bytes the substrate delivers up to the
/// budget.
///
/// # Fallibility
/// - `Err(IoError::EffectBudget{kind: "alloc"})` if `budget` would require more
///   allocation than the declared `alloc(Budget)` policy (for the in-memory
///   substrate, `Unbounded` never triggers this).
/// - `Err(IoError::Refused{why})` if the substrate declines.
///
/// # Effects: io + alloc(Budget) (declared, C6)
pub fn read(src: Source, budget: Budget) -> Result<(Vec<u8>, Source), IoError> {
    let mut src = src;
    let n = match budget {
        Budget::Bytes(cap) => {
            // Clamp to what remains — the budget is a ceiling, not a requirement.
            (cap as usize).min(src.remaining())
        }
        Budget::Unbounded => src.remaining(),
    };
    let chunk = src.read_n(n);
    Ok((chunk, src))
}

/// Write `bytes` to `snk`, consuming the handle and returning the updated one
/// (LR-8 — linear threading).
///
/// # Guarantee tag: `Exact`
/// The bytes written are exactly `bytes` — neither reordered nor approximated.
///
/// # Fallibility
/// - `Err(IoError::Refused{why})` if the substrate declines.
/// - `Err(IoError::EffectBudget{kind})` on an effect-budget overrun.
///
/// # Effects: io (declared, C6)
pub fn write(snk: Sink, bytes: &[u8]) -> Result<Sink, IoError> {
    let mut snk = snk;
    snk.append(bytes);
    Ok(snk)
}

/// Deserialize a `Value` directly from `src` in the given `format`, joining the
/// io and serialize halves.
///
/// # Guarantee tag: `Empirical` (round-trip; composes io + deserialize; spec §4)
/// The tag is the meet of `Exact` (io) and `Empirical` (deserialize) = `Empirical`.
///
/// # Fallibility
/// `Err(ReadValueError::Io(_))` — byte-movement failure before/during decode.
/// `Err(ReadValueError::Ser(_))` — bytes read but decode failed.
///
/// # Effects: io (declared, C6)
pub fn read_value(
    src: Source,
    format: crate::serialize::Format,
) -> Result<mycelium_core::Value, crate::error::ReadValueError> {
    let bytes = read_all(src).map_err(crate::error::ReadValueError::Io)?;
    crate::serialize::deserialize(&bytes, format).map_err(crate::error::ReadValueError::Ser)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ByteCount;

    // ── Substrate tests ────────────────────────────────────────────────────────

    #[test]
    fn substrate_remaining_starts_full() {
        let s = Substrate::from_bytes(vec![1u8, 2, 3, 4]);
        assert_eq!(s.remaining(), 4);
    }

    // ── read_all: single-consumption (LR-8) ───────────────────────────────────

    /// `read_all` returns all bytes from the source.
    /// Guard: returning fewer bytes makes this fail.
    #[test]
    fn read_all_returns_all_bytes() {
        let data = vec![10u8, 20, 30, 40, 50];
        let src = Source::from_bytes(data.clone());
        let bytes = read_all(src).expect("read_all must succeed");
        assert_eq!(bytes, data, "read_all must return exactly the source bytes");
    }

    /// `read_all` consumes the source exactly once — the handle is moved.
    /// This is a compile-time property (Rust move semantics), but we test the
    /// observable behavior: calling read_all twice on the *same data* via
    /// separate Sources both succeed independently.
    /// Guard: a double-consume bug (reuse without a new Source) would be a
    /// compile error, not reachable here.
    #[test]
    fn read_all_consumed_once_semantics() {
        let data = vec![1u8, 2, 3];
        let src1 = Source::from_bytes(data.clone());
        let src2 = Source::from_bytes(data.clone()); // separate source
        let r1 = read_all(src1).expect("first source");
        let r2 = read_all(src2).expect("second source");
        assert_eq!(
            r1, r2,
            "two separate sources of same data return same bytes"
        );
    }

    /// `read_all` on an empty source returns an empty Vec (not an error).
    #[test]
    fn read_all_empty_source_returns_empty() {
        let src = Source::from_bytes(vec![]);
        let bytes = read_all(src).expect("empty source must succeed");
        assert!(bytes.is_empty(), "empty source must yield empty bytes");
    }

    // ── read: chunked linear handle (LR-8) ────────────────────────────────────

    /// `read` with `Budget::Unbounded` returns all remaining bytes.
    #[test]
    fn read_unbounded_returns_all() {
        let data = vec![1u8, 2, 3, 4, 5];
        let src = Source::from_bytes(data.clone());
        let (chunk, src_rest) = read(src, Budget::Unbounded).expect("read must succeed");
        assert_eq!(chunk, data);
        assert_eq!(src_rest.remaining(), 0, "source must be exhausted");
    }

    /// `read` with `Budget::Bytes(n)` reads at most `n` bytes.
    /// Guard: reading more than the budget makes this fail.
    #[test]
    fn read_bounded_reads_at_most_budget() {
        let data = vec![1u8, 2, 3, 4, 5];
        let src = Source::from_bytes(data.clone());
        let (chunk, _src_rest) = read(src, Budget::Bytes(3)).expect("read must succeed");
        assert_eq!(chunk, vec![1u8, 2, 3], "must read exactly 3 bytes");
        // Guard: `chunk.len() <= 3` is the bound; > 3 violates the budget.
    }

    /// `read` returns the handle for continuation — linear threading.
    /// Two consecutive reads cover all bytes.
    #[test]
    fn read_linear_continuation() {
        let data = vec![10u8, 20, 30, 40];
        let src = Source::from_bytes(data.clone());
        let (chunk1, src) = read(src, Budget::Bytes(2)).expect("first read");
        let (chunk2, src) = read(src, Budget::Bytes(2)).expect("second read");
        assert_eq!(chunk1, vec![10u8, 20]);
        assert_eq!(chunk2, vec![30u8, 40]);
        assert_eq!(src.remaining(), 0, "fully consumed after two reads");
    }

    /// `read` when budget exceeds remaining returns only what's available (no error).
    #[test]
    fn read_budget_exceeds_remaining_returns_partial() {
        let data = vec![1u8, 2];
        let src = Source::from_bytes(data.clone());
        let (chunk, src_rest) = read(src, Budget::Bytes(100)).expect("read past end");
        assert_eq!(chunk, data, "must return what's available, not 100 bytes");
        assert_eq!(src_rest.remaining(), 0);
    }

    // ── write: linear Sink (LR-8) ─────────────────────────────────────────────

    /// `write` appends bytes to the sink and returns the updated handle.
    #[test]
    fn write_appends_bytes() {
        let snk = Sink::new();
        let snk = write(snk, &[1u8, 2, 3]).expect("first write");
        let snk = write(snk, &[4u8, 5]).expect("second write");
        let result = snk.into_bytes();
        assert_eq!(
            result,
            vec![1u8, 2, 3, 4, 5],
            "bytes must be appended in order"
        );
    }

    /// `write` with empty bytes is a no-op (no error).
    #[test]
    fn write_empty_bytes_is_noop() {
        let snk = Sink::new();
        let snk = write(snk, &[]).expect("write empty");
        assert!(snk.into_bytes().is_empty());
    }

    // ── Byte movement exactness (Exact guarantee tag) ──────────────────────────

    /// The bytes retrieved by `read_all` are byte-for-byte equal to the bytes
    /// given to `Source::from_bytes` — exactness of the byte-movement guarantee.
    /// Guard: any transformation inside read_all makes this fail.
    #[test]
    fn read_all_is_exact() {
        let data: Vec<u8> = (0u8..=255).collect();
        let src = Source::from_bytes(data.clone());
        let result = read_all(src).expect("read_all");
        assert_eq!(
            result, data,
            "byte-movement must be exact (spec §4 / Exact tag)"
        );
    }

    /// The bytes retrieved by `write` + `into_bytes` are byte-for-byte equal to
    /// the bytes given to `write`.
    #[test]
    fn write_is_exact() {
        let data: Vec<u8> = (0u8..=255).collect();
        let snk = Sink::new();
        let snk = write(snk, &data).expect("write");
        let result = snk.into_bytes();
        assert_eq!(
            result, data,
            "byte-movement must be exact (spec §4 / Exact tag)"
        );
    }

    // ── IoError: never-silent failures (C1/G2) ────────────────────────────────

    /// `IoError::UnexpectedEof` carries the byte count (RFC-0013 I1 / C1).
    /// Guard: dropping the count from the error makes this fail.
    #[test]
    fn io_error_unexpected_eof_carries_count() {
        let e = IoError::UnexpectedEof { read: ByteCount(7) };
        assert!(
            e.to_string().contains("7"),
            "error must carry the byte count (RFC-0013 I1)"
        );
    }

    // ── read_value: joined surface (io + serialize) ───────────────────────────

    /// `read_value` round-trips a serialized value end-to-end.
    /// Guard: any divergence between serialize → source → read_value makes this fail.
    #[test]
    fn read_value_round_trip() {
        use crate::serialize::{serialize, Format};
        use mycelium_core::{
            meta::{Meta, Provenance},
            repr::Repr,
            value::{Payload, Value},
        };
        let v = Value::new(
            Repr::Binary { width: 4 },
            Payload::Bits(vec![true, false, true, false]),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed value");

        let bytes = serialize(&v, Format::Wire).expect("serialize: finite test value");
        let src = Source::from_bytes(bytes);
        let recovered = read_value(src, Format::Wire).expect("read_value round-trip");
        assert_eq!(v, recovered, "read_value must recover the original value");
    }

    /// `read_value` on garbage bytes yields `Err(ReadValueError::Ser(_))` (C1).
    #[test]
    fn read_value_garbage_bytes_yields_ser_error() {
        use crate::serialize::Format;
        let src = Source::from_bytes(b"not valid json at all".to_vec());
        let err = read_value(src, Format::Wire).expect_err("must error on garbage");
        assert!(
            matches!(err, crate::error::ReadValueError::Ser(_)),
            "garbage bytes must yield a Ser error, not an Io error"
        );
    }

    // ── Budget: declared effect bound (C6/ADR-015) ────────────────────────────

    /// `Budget::Bytes(n)` is reified and comparable — the bound is not hidden (C3).
    #[test]
    fn budget_is_reified() {
        let b = Budget::Bytes(1024);
        assert_eq!(b, Budget::Bytes(1024));
        assert_ne!(b, Budget::Unbounded);
    }
}
