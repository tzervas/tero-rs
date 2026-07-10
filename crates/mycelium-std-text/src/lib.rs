//! `std.text` — Ring-2 UTF-8 string type and operations (M-524, #165).
//!
//! The UTF-8 string surface every program needs: a value-semantic, immutable [`Text`] type with
//! construction from bytes/chars, slicing and indexing on **validated** char boundaries, a parse
//! family (`str → T`), and encoding/transcoding between UTF-8 and other encodings.
//!
//! # Honesty crux (RFC-0016 §4.4 — two-part, structural)
//!
//! 1. **`parse` returns a `Result`, never a sentinel** (C1 / G2 "never-silent"). A failed parse
//!    is `Err(`[`ParseErr`]`)` carrying *where* it failed (the byte/char index) and
//!    *expected-vs-found* (the RFC-0013 §4.1 diagnostic record I1). A malformed input is **never**
//!    silently coerced to `0` / `false` / `""`.
//!
//! 2. **Lossy encoding/transcoding is explicit** — a lossy transcode is a **strict `Err`** in the
//!    default arm ([`to_latin1`], [`from_utf16`]), and the **only** path to U+FFFD substitution is
//!    the **distinct, named** [`to_latin1_lossy`] op whose `Lossy<Vec<u8>>` return type carries the
//!    substitution count. A caller cannot lose data silently (G2 / C1).
//!
//! # Guarantee matrix
//!
//! Every exported op has a row in [`guarantee_matrix::MATRIX`] (RFC-0016 §4.5 / spec §4). All ops
//! are `Exact` (no accuracy/precision semantics) and effect-free. The matrix is asserted in tests,
//! not prose-only (C2 / VR-5).
//!
//! # §4.1 contract conformance
//!
//! - **C1 — never-silent (G2):** `parse_*` returns `Result`, never a sentinel; `from_utf8`,
//!   `to_latin1`, `from_utf16` return `Err` on lossy/invalid input, never U+FFFD; `slice`/`char_at`
//!   return `Err` on off-boundary/out-of-range, never a snap/truncation. No sentinel anywhere.
//! - **C2 — honest per-op tag (VR-5):** Every op is `Exact` — `text` has no accuracy/precision
//!   semantics, so there is nothing to downgrade. The honesty is the fallibility column.
//! - **C3 — no black boxes / EXPLAIN (SC-3/G11):** Every op that can reject carries an inspectable
//!   error that names *where* and *what* (RFC-0013 §4.1 I1 — additive, still propagating).
//! - **C4 — content-addressed, value-semantic (ADR-003 / RFC-0001):** `Text` is immutable; every
//!   transform returns a **new** `Text` (never in-place). Two `Text`s with equal bytes are equal.
//! - **C5 — above the small kernel (KC-3):** Ring 2. No trusted code added; no `unsafe`.
//! - **C6 — declared, bounded effects:** All ops are pure (no IO, time, randomness, global state).
//!
//! # Open questions (FLAGs — resolve before ratification)
//!
//! - **(Q1) `Lossy<T>` shape:** distinct type with configurable `marker` (per spec §7-Q1 proposed
//!   disposition). Pending maintainer decision on whether `marker` is configurable or fixed at
//!   U+FFFD. The `Lossy` type anticipates the configurable direction.
//! - **(Q2) Grapheme segmentation table version:** [`len_graphemes`] uses a scalar-value count
//!   placeholder (exact for ASCII/simple Latin; may differ for ZWJ/emoji). Full Unicode grapheme-
//!   break segmentation requires a versioned table reified in `Meta` (spec §7-Q2). Not asserted
//!   as `Proven` (VR-5 — stay honest; the tag stays `Exact` against the placeholder definition).
//! - **(Q3) `parse` ↔ `math`/`numerics` value-semantics seam:** `parse_int` returns `i64` as a
//!   placeholder; the authoritative numeric type and range semantics live in `math`/`numerics`
//!   (M-525/M-512). `ParseErr::OutOfRange` is the `text`-layer hand-off (spec §7-Q3).
//! - **(Q4) UTF-8 validation `wild`/FFI floor:** this crate uses `std::str::from_utf8` (Rust
//!   stdlib — pure, no FFI). The `wild`/FFI question (spec §7-Q4) is resolved at this layer:
//!   no FFI is needed; the C5 "no `wild`" claim holds for this implementation.
//! - **(Q-iter) `chars` return type:** returns `Vec<char>` (value-semantic) rather than
//!   `Iter<Char>` to avoid depending on the unimplemented `std.iter` (M-526) sibling crate.
//!   FLAG to orchestrator: update once M-526 lands.
//!
//! # Ring & layering
//!
//! Ring 2 (RFC-0016 §4.2). `text` is new library code written to the §4.1 contract over Ring 0
//! (`mycelium-core` / `mycelium-std-core`). It adds **no trusted code** and no `unsafe`
//! (KC-3 / `#![forbid(unsafe_code)]`).
//!
//! # Design spec
//! `docs/spec/stdlib/text.md` (M-524, #165). Contract: RFC-0016 §4.1 (C1–C6).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Text ops are representation-opaque at the byte level —
//! `Text` is a UTF-8 byte sequence; callers are responsible for any encoding decision (e.g.,
//! which `Repr` their parsed numeric values use). No ambient encoding is inferred; lossy
//! transcoding is an explicit named op (`to_latin1_lossy`), never a silent substitution.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/text.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; the same-named `lib/std/text.myc` prototype is a narrower, structurally distinct surface (DN-66 S3.1) — the D6 retirement trigger has not fired, so no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod error;
pub mod guarantee_matrix;
pub mod ops;
pub mod types;

// ─── Public re-exports (the flat API surface) ────────────────────────────────

pub use error::{BoundaryError, EncodeError, ParseErr, TranscodeError, Utf8Error};
pub use types::{Lossy, Text};

// Construction
pub use ops::{concat, from_chars, from_utf8, join};

// Immutable transforms (each returns a NEW Text — never in-place)
pub use ops::{replace, to_lower, to_upper, trim};

// Length / iteration
pub use ops::{chars, len_bytes, len_chars, len_graphemes};

// Slicing / indexing on validated boundaries
pub use ops::{char_at, slice};

// Parse: str → T is a Result, NEVER a sentinel
pub use ops::{parse_bool, parse_int};

// Encoding / transcoding
pub use ops::{encode_utf8, from_utf16, to_latin1, to_latin1_lossy, to_utf16};

// ─── Crate-level integration tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Crate-level integration tests that exercise the public API through the flat re-exports.
    //!
    //! Module-level unit tests live in their respective submodules (`ops::tests`,
    //! `error::tests`, `types::tests`, `guarantee_matrix::tests`). These tests verify
    //! the *integration* of the public API surface and assert the guarantee matrix
    //! conformance from the crate root.

    use super::*;
    use crate::guarantee_matrix::MATRIX;

    // ─── Guarantee matrix conformance ─────────────────────────────────────────

    /// The guarantee matrix is non-empty (at least one exported op).
    #[test]
    fn guarantee_matrix_is_non_empty() {
        assert!(
            !MATRIX.is_empty(),
            "guarantee matrix must have at least one row"
        );
    }

    /// Every row of the guarantee matrix is `Exact` and effect-free (C2 / C6).
    /// This is the crate-root assertion of the matrix invariant (asserted redundantly in
    /// `guarantee_matrix::tests`, duplicated here as an integration-level safety net).
    #[test]
    fn guarantee_matrix_all_exact_effect_free() {
        for row in MATRIX {
            assert_eq!(
                row.guarantee, "Exact",
                "op {:?} must be Exact (text has no accuracy semantics — C2 / VR-5)",
                row.op
            );
            assert_eq!(
                row.effects, "none",
                "op {:?} must be effect-free (C6 — all text ops are pure)",
                row.op
            );
        }
    }

    // ─── End-to-end honesty crux tests ────────────────────────────────────────

    /// The `parse_int` honesty crux: `Result`, **never** a sentinel `0`.
    ///
    /// This is the central RFC-0016 §4.4 honesty obligation for `std.text`.
    /// Guard: if `parse_int` returned `0` instead of `Err` for malformed input, this fails.
    #[test]
    fn honesty_crux_parse_int_never_sentinel_zero() {
        // Mutant witness: replacing parse_int with |_| Ok(0) makes this fail.
        let cases: &[&str] = &["", "abc", "1.2", "true", " 1", "1 "];
        for s in cases {
            let result = parse_int(&Text::new(s));
            assert!(
                result.is_err(),
                "parse_int({s:?}) must be Err, not a sentinel 0 (RFC-0016 §4.4 honesty crux)"
            );
        }
    }

    /// The `parse_bool` honesty crux: `Result`, **never** a sentinel `false`.
    ///
    /// Guard: if `parse_bool` returned `false` instead of `Err` for malformed input, this fails.
    #[test]
    fn honesty_crux_parse_bool_never_sentinel_false() {
        // Mutant witness: replacing parse_bool with |_| Ok(false) makes this fail.
        let cases: &[&str] = &["", "True", "False", "1", "0", "yes", "no", "TRUE"];
        for s in cases {
            let result = parse_bool(&Text::new(s));
            assert!(
                result.is_err(),
                "parse_bool({s:?}) must be Err, not a sentinel false (RFC-0016 §4.4 honesty crux)"
            );
        }
    }

    /// The `from_utf8` honesty crux: invalid bytes → `Err`, **never** silent U+FFFD.
    ///
    /// Guard: if `from_utf8` inserted U+FFFD for invalid bytes instead of returning `Err`, this fails.
    #[test]
    fn honesty_crux_from_utf8_never_silent_fffd() {
        // 0xFF is never valid UTF-8.
        let result = from_utf8(&[0xFF]);
        assert!(
            result.is_err(),
            "invalid UTF-8 must be Err, never silent U+FFFD (C1 / G2)"
        );
        // Verify it's not accidentally returning U+FFFD.
        if let Ok(t) = result {
            assert_ne!(
                t.as_str(),
                "\u{FFFD}",
                "must not silently substitute U+FFFD for invalid UTF-8"
            );
        }
    }

    /// The `to_latin1` honesty crux: non-Latin-1 chars → `Err`, **never** silent U+FFFD.
    ///
    /// Guard: returning Ok with '?' for unrepresentable chars makes this fail.
    #[test]
    fn honesty_crux_to_latin1_never_silent_fffd() {
        // '€' (U+20AC) is not in Latin-1.
        let t = Text::new("€");
        let result = to_latin1(&t);
        assert!(
            result.is_err(),
            "non-Latin-1 char must be Err, never silent U+FFFD (C1 / G2)"
        );
    }

    /// The `to_latin1_lossy` honesty crux: lossiness is **un-droppable** (G2 / C1).
    ///
    /// The caller cannot get lossy output without seeing the substitution count; it is always
    /// in the `Lossy` return value.
    ///
    /// Guard: if `to_latin1_lossy` returned `Vec<u8>` directly (hiding the count), this fails
    /// to compile (the type enforces the guarantee).
    #[test]
    fn honesty_crux_to_latin1_lossy_count_is_not_droppable() {
        let t = Text::new("hello €");
        let result: Lossy<Vec<u8>> = to_latin1_lossy(&t, '?');
        // The caller must use `result.substituted` or `result.value`; they cannot drop the count.
        assert_eq!(
            result.substituted, 1,
            "substitution count must be in the Lossy value"
        );
        assert_eq!(result.marker, '?');
    }

    /// The `slice` honesty crux: out-of-range → `Err`, **never** a silent truncation.
    ///
    /// Guard: if `slice` silently truncated to the string length instead of returning `Err`,
    /// this fails.
    #[test]
    fn honesty_crux_slice_never_silent_truncation() {
        let t = Text::new("hi");
        let result = slice(&t, 0, 100);
        assert!(
            result.is_err(),
            "out-of-range slice must be Err, never a silent truncation (C1 / G2)"
        );
    }

    /// The `char_at` honesty crux: out-of-range → `Err`, **never** a sentinel char.
    ///
    /// Guard: returning '\0' for an out-of-range index makes this fail.
    #[test]
    fn honesty_crux_char_at_never_sentinel_char() {
        let t = Text::new("hi");
        let result = char_at(&t, 100);
        assert!(
            result.is_err(),
            "out-of-range char_at must be Err, never a sentinel char (C1 / G2)"
        );
    }

    // ─── C4 — value-semantic / immutability ────────────────────────────────────

    /// Every transform returns a NEW Text; the original is unmodified (C4 — value-semantic).
    #[test]
    fn c4_transforms_never_mutate_original() {
        let original = Text::new("Hello World");
        let _upper = to_upper(&original);
        let _lower = to_lower(&original);
        let _trimmed = trim(&original);
        let _replaced = replace(&original, &Text::new("World"), &Text::new("Rust"));
        // All transforms above have no side effects; the original must be unchanged.
        assert_eq!(
            original.as_str(),
            "Hello World",
            "C4 — transforms must never mutate the original"
        );
    }

    // ─── C6 — effect-free ──────────────────────────────────────────────────────

    /// Repeated calls to any deterministic op return the same result (C6 / C2 — no randomness,
    /// no global state, no time dependency).
    #[test]
    fn c6_all_ops_are_deterministic() {
        let t = Text::new("hello");
        // Call each op twice; results must be equal.
        assert_eq!(to_upper(&t), to_upper(&t));
        assert_eq!(to_lower(&t), to_lower(&t));
        assert_eq!(trim(&t), trim(&t));
        assert_eq!(len_bytes(&t), len_bytes(&t));
        assert_eq!(len_chars(&t), len_chars(&t));
        assert_eq!(chars(&t), chars(&t));
        assert_eq!(encode_utf8(&t), encode_utf8(&t));
        assert_eq!(to_utf16(&t), to_utf16(&t));
        assert_eq!(parse_int(&Text::new("42")), parse_int(&Text::new("42")));
        assert_eq!(
            parse_bool(&Text::new("true")),
            parse_bool(&Text::new("true"))
        );
    }
}
