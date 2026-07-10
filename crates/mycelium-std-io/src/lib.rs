//! `std.io` + `serialize` — single-consumption IO + canonical serialization (M-514).
//!
//! Two coupled surfaces over Mycelium's content-addressed value model:
//!
//! **serialize** — project a [`mycelium_core::Value`] to/from a byte or text form:
//! - [`serialize::serialize`] / [`serialize::deserialize`] over the RFC-0001 §4.8
//!   self-describing wire form (`[Repr descriptor] ‖ [Meta] ‖ [payload]`,
//!   schema-travels-with-data, faithfully round-trippable including `Meta`).
//! - [`serialize::to_json`] / [`serialize::from_json`] — the **one canonical JSON
//!   projection** that `fmt.to_json` (M-533) delegates to (README §5 seam; FLAGGED
//!   §7-Q1 pending maintainer sign-off).
//!
//! **io** — move bytes over an abstract source/sink:
//! - [`io::read_all`] / [`io::read`] / [`io::write`] over [`io::Source`] /
//!   [`io::Sink`] — affine handles consumed **exactly once** (LR-8 / RFC-0006 §Q5).
//!   Single-consumption is a **type-level invariant** (Rust move semantics), not a
//!   runtime convention.
//! - [`io::read_value`] — the bridge joining io + serialize: deserialize a `Value`
//!   directly from a `Source`.
//!
//! # Honesty crux (twofold, structural)
//!
//! 1. **Round-trip** — `deserialize(serialize(v, f), f) ≡ v` including `Meta` is a
//!    **checked property** (proptest corpus; `Empirical` tag — not `Proven`, no
//!    checked theorem; VR-5 / spec §7-Q2).
//! 2. **Never-silent** — a truncated, malformed, or decode-failed input is an
//!    **explicit, located** error (`SerError` / `IoError` carrying a byte offset /
//!    field path — RFC-0013 I1); **no op returns a partially-filled `Value`** (C1/G2).
//!
//! # Serialization is a projection, not identity (ADR-003 / C4)
//! `serialize`/`to_json` borrow their `&Value` immutably.  The content hash is
//! unchanged by serialization.  The round-trip recovers a value with the **same
//! content-id**; the projection is not a re-keying.
//!
//! # Single-consumption (LR-8)
//! `read_all` takes [`io::Source`] **by-move** and does not return it — the affine
//! handle is consumed exactly once.  A double-consume is a **compile-time error**, not
//! a runtime check.  Chunked `read` threads the handle linearly; `write` threads
//! [`io::Sink`] linearly.
//!
//! # Declared effects (C6 / RFC-0014 §4.5)
//! - io ops declare the **`io`** effect; chunked `read` additionally declares
//!   `alloc(Budget)`.
//! - serialize/JSON ops are **pure** (`effects: none`).
//!
//! # In-memory substrate + OS I/O floor (FLAGGED §7-Q4 / §8-Q6)
//! The OS I/O floor is deferred to `std-sys` (M-541). This module ships a fully
//! testable **in-memory substrate** ([`io::Substrate`] backed by `Vec<u8>`) that
//! exercises the affine abstraction without OS facilities. The `std-sys` phylum will
//! provide OS-backed constructors for file descriptors and sockets.
//!
//! # No new trusted code (KC-3 / C5)
//! This crate wraps `mycelium-core`'s `serde::{Serialize, Deserialize}` for `Value`
//! (landed M-104). No new trusted serialization logic; no `unsafe`.
//!
//! # Guarantee matrix
//! Eight rows — one per exported op — in [`guarantee_matrix::MATRIX`] (RFC-0016 §4.5).
//! Asserted in tests.
//!
//! # Design spec
//! `docs/spec/stdlib/io.md` (M-514, #155); contract: RFC-0016 §4.1 (C1–C6).
//!
//! # Open questions (FLAGs from spec §7)
//!
//! - **(Q1) One canonical JSON — `fmt.to_json` delegation.** `to_json`/`from_json` are
//!   proposed as the canonical JSON projection that `fmt` (M-533) delegates to.  Needs
//!   maintainer sign-off (spec §7-Q1 / RFC-0016 §8-Q1).
//!
//! - **(Q2) Which round-trip reaches `Proven`.** The `Empirical` tag is the honest
//!   maximum for this implementation.  A `Proven` tag would require a checked
//!   injectivity/totality theorem over the closed `Repr`/`Meta` grammar (VR-5;
//!   spec §7-Q2).
//!
//! - **(Q3) The io ↔ fs build-on seam.** `fs` (M-528) *builds on* this module's
//!   `Source`/`Sink`; the exact path→substrate constructor surface is co-design with
//!   M-528 (spec §7-Q3).
//!
//! - **(Q4) The `wild`/FFI floor for the io half.** Byte movement over a real OS
//!   source/sink requires OS facilities → `wild`/FFI (ADR-014). Deferred to `std-sys`
//!   (M-541; spec §7-Q4 / RFC-0016 §8-Q6).
//!
//! - **(Q5) Format / Budget ergonomics at the call site.** Both `Format` and `Budget`
//!   are required-explicit until the per-ring ergonomics pass (M-540; spec §7-Q5 /
//!   RFC-0016 §8-Q3 tension A).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** I/O is representation-opaque at the byte level —
//! `Source`/`Sink` move raw bytes without interpreting `Repr`. Callers own the encoding
//! choice; the `Format` parameter to serialize ops is always explicit (never inferred from
//! context). The serialized form carries the `Repr` descriptor as part of its wire schema
//! (RFC-0001 §4.8 "schema-travels-with-data"), so no silent re-encoding occurs on round-trip.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/io.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.

#![forbid(unsafe_code)]

pub mod error;
pub mod guarantee_matrix;
pub mod io;
pub mod serialize;

// In-crate white-box unit tests extracted from logic files (M-797 as-touched). Named
// `unit_tests` to avoid colliding with this crate's own integration-level `mod tests` below.
#[cfg(test)]
#[path = "tests/mod.rs"]
mod unit_tests;

// ── Flat re-exports for convenience ────────────────────────────────────────────

// error types
pub use error::{ByteCount, ByteOffset, FieldPath, IoError, ReadValueError, SerError};

// serialize surface
pub use serialize::{deserialize, from_json, serialize, to_json, Format};

// io surface
pub use io::{read, read_all, read_value, write, Budget, Sink, Source, Substrate};

// guarantee matrix
pub use guarantee_matrix::MATRIX as GUARANTEE_MATRIX;

// ── Integration-level tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{
        meta::{Meta, Provenance},
        repr::Repr,
        value::{Payload, Value},
    };

    fn binary_value() -> Value {
        Value::new(
            Repr::Binary { width: 8 },
            Payload::Bits(vec![true, false, true, true, false, false, true, false]),
            Meta::exact(Provenance::Root),
        )
        .expect("well-formed binary value")
    }

    // ── Public surface smoke tests ─────────────────────────────────────────────

    /// The flat re-exports are usable without the sub-module path.
    #[test]
    fn flat_re_exports_work() {
        let v = binary_value();
        // serialize
        let bytes = serialize(&v, Format::Wire).expect("serialize: finite test value");
        let recovered = deserialize(&bytes, Format::Wire).expect("wire round-trip");
        assert_eq!(v, recovered);
        // json
        let text = to_json(&v).expect("to_json: finite test value");
        let recovered = from_json(&text).expect("JSON round-trip");
        assert_eq!(v, recovered);
    }

    /// End-to-end: serialize -> Source -> read_all -> deserialize.
    #[test]
    fn io_and_serialize_end_to_end() {
        let v = binary_value();
        let bytes = serialize(&v, Format::Wire).expect("serialize: finite test value");
        let src = Source::from_bytes(bytes);
        let raw = read_all(src).expect("read_all");
        let recovered = deserialize(&raw, Format::Wire).expect("deserialize after read_all");
        assert_eq!(v, recovered);
    }

    /// End-to-end: serialize -> Source -> read_value.
    #[test]
    fn read_value_end_to_end() {
        let v = binary_value();
        let bytes = serialize(&v, Format::Wire).expect("serialize: finite test value");
        let src = Source::from_bytes(bytes);
        let recovered = read_value(src, Format::Wire).expect("read_value");
        assert_eq!(v, recovered);
    }

    /// End-to-end: serialize -> write -> Sink -> deserialize.
    #[test]
    fn write_and_serialize_end_to_end() {
        let v = binary_value();
        let bytes = serialize(&v, Format::Wire).expect("serialize: finite test value");
        let snk = Sink::new();
        let snk = write(snk, &bytes).expect("write");
        let result = snk.into_bytes();
        let recovered = deserialize(&result, Format::Wire).expect("deserialize after write");
        assert_eq!(v, recovered);
    }

    // ── Guarantee matrix is well-formed ───────────────────────────────────────

    /// The guarantee matrix has exactly 8 rows (one per exported op; spec §4).
    #[test]
    fn guarantee_matrix_has_eight_rows() {
        assert_eq!(
            GUARANTEE_MATRIX.len(),
            8,
            "spec §4 guarantees eight op rows"
        );
    }

    // ── C1 — never-silent ─────────────────────────────────────────────────────

    /// Garbage bytes yield an explicit Err, never a silent partial value.
    #[test]
    fn deserialize_never_silent_on_garbage() {
        let result = deserialize(b"garbage", Format::Wire);
        assert!(result.is_err(), "garbage input must yield Err (C1/G2)");
    }

    /// Empty bytes yield Err (C1).
    #[test]
    fn deserialize_never_silent_on_empty() {
        assert!(
            deserialize(b"", Format::Wire).is_err(),
            "empty input must yield Err (C1)"
        );
    }

    // ── C4 — projection, not identity (ADR-003) ───────────────────────────────

    /// `serialize` does not change the content hash of the value.
    #[test]
    fn serialize_is_a_projection_not_identity() {
        let v = binary_value();
        let h_before = v.content_hash();
        let _bytes = serialize(&v, Format::Wire).expect("serialize: finite test value");
        assert_eq!(
            v.content_hash(),
            h_before,
            "ADR-003: projection must not change identity"
        );
    }

    // ── LR-8 — affine single-consumption ──────────────────────────────────────

    /// `read_all` exhausts the source (returns the full byte payload).
    #[test]
    fn read_all_exhausts_source() {
        let data = vec![1u8, 2, 3];
        let src = Source::from_bytes(data.clone());
        let bytes = read_all(src).expect("read_all");
        assert_eq!(bytes, data, "read_all must return all source bytes");
    }

    /// Chunked `read` threads the handle linearly.
    #[test]
    fn read_threads_linearly() {
        let src = Source::from_bytes(vec![1u8, 2, 3, 4, 5, 6]);
        let (a, src) = read(src, Budget::Bytes(2)).expect("first chunk");
        let (b, src) = read(src, Budget::Bytes(2)).expect("second chunk");
        let (c, _src) = read(src, Budget::Bytes(2)).expect("third chunk");
        assert_eq!([a, b, c], [vec![1, 2], vec![3, 4], vec![5, 6]]);
    }
}
