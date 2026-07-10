//! Explicit error types for `std.text` (C1 — never-silent; RFC-0016 §4.1).
//!
//! Every fallible operation in this crate returns a typed error or `Option`, **never** a sentinel
//! value, a clamp, or a silent best-effort result (G2 / C1). The error types below are the
//! EXPLAIN-able artifacts (C3 / RFC-0013 §4.1): each one carries **where** the failure occurred
//! (byte/char index) and **what** was expected or found.
//!
//! # Error hierarchy
//!
//! - [`Utf8Error`] — `from_utf8`: an invalid byte sequence was encountered at a known index.
//! - [`BoundaryError`] — `slice` / `char_at`: an out-of-range or mid-codepoint/mid-grapheme index.
//! - [`ParseErr`] — `parse_int` / `parse_bool`: lexical parse failure; carries the span and
//!   expected-vs-found (RFC-0013 §4.1 diagnostic record I1 — additive over the error, never
//!   replacing it).
//! - [`EncodeError`] — `to_latin1`: an unrepresentable character at a known char index.
//! - [`TranscodeError`] — `from_utf16`: an unpaired surrogate or invalid unit at a known index.
//!
//! # FLAG — Q3 (parse ↔ math/numerics value-semantics seam)
//! `text` owns the *lexical* failure (malformed digits, the span). The numeric module owns
//! the *value-range* failure. `ParseErr::OutOfRange { at, target }` is the hand-off point: `text`
//! reports that the digit sequence was lexically valid but exceeded the target type's range;
//! the numeric module (M-525/M-512) owns the authoritative range semantics. Resolution of this
//! seam is deferred to the maintainer (spec §7-Q3; cmp §7-Q2). This crate does not guess.

use std::fmt;

// ─── Utf8Error ────────────────────────────────────────────────────────────────

/// An invalid UTF-8 byte sequence was found at a known byte index.
///
/// Produced by [`crate::from_utf8`] when the input bytes are not valid UTF-8.
///
/// # C1 compliance
/// The error is an explicit `Err(Utf8Error::Invalid { byte, reason })` — **never** a silent
/// U+FFFD replacement-character substitution (RFC-0016 §4.1 C1; G2 "never-silent"). A caller
/// who wants U+FFFD substitution must use a distinct, explicitly-named `*_lossy` variant
/// (unavailable here — see spec §2 scope; `from_utf8` is the strict op).
///
/// # C3 compliance
/// The `byte` index names **where** the invalid byte is, so the caller can locate it in the
/// input (the EXPLAIN artifact — RFC-0013 §4.1 I1 "additive and still propagating").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Utf8Error {
    /// The byte at index `byte` is not valid UTF-8.
    ///
    /// `reason` is a short human-readable description of why the byte is invalid (C3 / G11).
    Invalid {
        /// The byte index at which the invalid byte was found.
        byte: usize,
        /// A short human-readable reason (G11 dual projection; RFC-0013 §4.1).
        reason: &'static str,
    },
}

impl fmt::Display for Utf8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Utf8Error::Invalid { byte, reason } => {
                write!(f, "invalid UTF-8 at byte {byte}: {reason}")
            }
        }
    }
}

mycelium_std_core::impl_std_error!(Utf8Error);

// ─── BoundaryError ───────────────────────────────────────────────────────────

/// An index into a `Text` was out of range or fell on an invalid boundary.
///
/// Produced by [`crate::slice`] and [`crate::char_at`] when the requested index is not a valid
/// char/grapheme boundary (or is out of range).
///
/// # C1 compliance
/// The error is an explicit `Err(BoundaryError::…)` — **never** a silent snap to the nearest
/// boundary or a truncation (RFC-0016 §4.1 C1; G2).
///
/// # C3 compliance
/// Each variant carries the offending index and/or the valid length so the caller can diagnose
/// the failure without reparsing (RFC-0013 §4.1 I1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryError {
    /// The requested index or range end is past the end of the string.
    OutOfRange {
        /// The length of the string in bytes.
        len: usize,
        /// The index (or range end) that was requested.
        asked: usize,
    },
    /// The byte index falls in the middle of a multi-byte codepoint.
    NotCharBoundary {
        /// The offending byte index.
        byte: usize,
    },
    /// The byte index falls in the middle of a grapheme cluster.
    ///
    /// # FLAG — Q2 (grapheme segmentation)
    /// This variant depends on a versioned Unicode grapheme-break table (spec §7-Q2).
    /// The table version is not yet reified in `Meta` (a maintainer call). The variant
    /// exists so the error type is complete; `slice` with grapheme-boundary checking
    /// will be gated until the table versioning is resolved (spec §7-Q2).
    NotGraphemeBoundary {
        /// The offending byte index.
        byte: usize,
    },
    /// The requested range is inverted (`start > end`).
    ///
    /// Both endpoints can individually be in range yet describe an empty/negative span; reported
    /// explicitly rather than panicking on the slice index (C1 / G2 — never a runtime panic where
    /// the signature promises `Result`).
    InvalidRange {
        /// The requested start byte index.
        start: usize,
        /// The requested end byte index (`< start`).
        end: usize,
    },
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundaryError::OutOfRange { len, asked } => {
                write!(f, "index {asked} is out of range (string length {len})")
            }
            BoundaryError::NotCharBoundary { byte } => {
                write!(f, "byte index {byte} is not on a char boundary")
            }
            BoundaryError::NotGraphemeBoundary { byte } => {
                write!(f, "byte index {byte} is not on a grapheme-cluster boundary")
            }
            BoundaryError::InvalidRange { start, end } => {
                write!(f, "inverted range: start {start} is greater than end {end}")
            }
        }
    }
}

mycelium_std_core::impl_std_error!(BoundaryError);

// ─── ParseErr ─────────────────────────────────────────────────────────────────

/// A parse failure: the input string was empty, malformed, or the lexically-valid value was
/// out of range for the target type.
///
/// Produced by [`crate::parse_int`] and [`crate::parse_bool`].
///
/// # Honesty crux (C1 / RFC-0016 §4.4)
/// `parse` returns a **`Result`, never a sentinel** (`0` / `false` / `""`). A malformed input is
/// `Err(ParseErr)` carrying the byte/char index + expected-vs-found (the RFC-0013 §4.1 diagnostic
/// record I1). The caller cannot silently ignore the failure.
///
/// # C3 compliance
/// `Invalid { at, expected, found }` carries **where** the parse failed and **what** was expected
/// vs found — the EXPLAIN artifact (RFC-0013 §4.1 I1 "additive over the explicit error").
///
/// # FLAG — Q3 (parse ↔ numerics value-semantics seam)
/// `ParseErr::OutOfRange { at, target }` is the hand-off point: `text` owns the lexical failure,
/// `math`/`numerics` own the value-range semantics. See module-level FLAG Q3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErr {
    /// The input string was empty.
    Empty,
    /// The input was lexically invalid at the given byte index.
    Invalid {
        /// The byte index at which the unexpected character was found.
        at: usize,
        /// A short description of what was expected at this position (G11).
        expected: &'static str,
        /// A short description of what was actually found (G11).
        found: String,
    },
    /// The digit sequence was lexically valid but outside the range of `target`.
    ///
    /// # FLAG — Q3
    /// `text` reports the out-of-range condition; the numeric module (M-525/M-512) owns the
    /// authoritative range semantics. This is the `text`-layer hand-off only.
    OutOfRange {
        /// The byte index at which the out-of-range value starts.
        at: usize,
        /// The target type that the value was being parsed into.
        target: &'static str,
    },
}

impl fmt::Display for ParseErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErr::Empty => write!(f, "parse error: input is empty"),
            ParseErr::Invalid {
                at,
                expected,
                found,
            } => write!(
                f,
                "parse error at byte {at}: expected {expected}, found {found:?}"
            ),
            ParseErr::OutOfRange { at, target } => write!(
                f,
                "parse error at byte {at}: value is out of range for {target}"
            ),
        }
    }
}

mycelium_std_core::impl_std_error!(ParseErr);

// ─── EncodeError ─────────────────────────────────────────────────────────────

/// A character in the `Text` is not representable in the target encoding.
///
/// Produced by [`crate::to_latin1`] (the strict variant — never by `to_latin1_lossy`).
///
/// # C1 compliance
/// The error is an explicit `Err(EncodeError::Unrepresentable { ch, at, target_encoding })` —
/// **never** a silent U+FFFD substitution. A caller who wants substitution must call
/// [`crate::to_latin1_lossy`] explicitly (spec §2 / §3).
///
/// # C3 compliance
/// The variant carries the offending character, its char index, and the target encoding name
/// (RFC-0013 §4.1 I1 — the EXPLAIN artifact naming where and what).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// The character at char index `at` is not representable in `target_encoding`.
    Unrepresentable {
        /// The unrepresentable character.
        ch: char,
        /// The char-boundary index of the unrepresentable character.
        at: usize,
        /// The name of the target encoding (e.g. `"Latin-1"`).
        target_encoding: &'static str,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::Unrepresentable {
                ch,
                at,
                target_encoding,
            } => write!(
                f,
                "character {:?} at char index {} is not representable in {}",
                ch, at, target_encoding
            ),
        }
    }
}

mycelium_std_core::impl_std_error!(EncodeError);

// ─── TranscodeError ──────────────────────────────────────────────────────────

/// A UTF-16 unit sequence is invalid (unpaired surrogate or otherwise invalid unit).
///
/// Produced by [`crate::from_utf16`].
///
/// # C1 compliance
/// The error is an explicit `Err(TranscodeError::…)` — **never** a silent U+FFFD substitution
/// (RFC-0016 §4.1 C1; G2). A caller who wants U+FFFD substitution must use the distinct, named
/// `from_utf16_lossy` variant (not exposed in this crate — see spec §2).
///
/// # C3 compliance
/// Each variant carries the unit index at which the invalid sequence starts (RFC-0013 §4.1 I1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscodeError {
    /// A surrogate code unit at `at` is unpaired (no matching low/high surrogate follows/precedes).
    UnpairedSurrogate {
        /// The UTF-16 unit index of the unpaired surrogate.
        at: usize,
    },
    /// An invalid code unit was encountered at `at`.
    Invalid {
        /// The UTF-16 unit index of the invalid unit.
        at: usize,
    },
}

impl fmt::Display for TranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranscodeError::UnpairedSurrogate { at } => {
                write!(f, "unpaired UTF-16 surrogate at unit index {at}")
            }
            TranscodeError::Invalid { at } => {
                write!(f, "invalid UTF-16 unit at index {at}")
            }
        }
    }
}

mycelium_std_core::impl_std_error!(TranscodeError);

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Utf8Error ────────────────────────────────────────────────────────────

    /// `Utf8Error::Invalid` carries the byte index and reason (C3 — EXPLAIN artifact).
    #[test]
    fn utf8_error_display_carries_index_and_reason() {
        let e = Utf8Error::Invalid {
            byte: 3,
            reason: "continuation byte expected",
        };
        let s = e.to_string();
        assert!(s.contains("3"), "must include byte index");
        assert!(
            s.contains("continuation byte expected"),
            "must include reason"
        );
    }

    #[test]
    fn utf8_error_is_std_error() {
        let e = Utf8Error::Invalid {
            byte: 0,
            reason: "test",
        };
        let _: &dyn std::error::Error = &e;
    }

    // ─── BoundaryError ────────────────────────────────────────────────────────

    /// Out-of-range variant carries len and asked (C3).
    #[test]
    fn boundary_error_out_of_range_display() {
        let e = BoundaryError::OutOfRange { len: 5, asked: 10 };
        let s = e.to_string();
        assert!(s.contains("10"), "must include asked index");
        assert!(s.contains("5"), "must include length");
    }

    /// NotCharBoundary variant carries the byte index (C3).
    #[test]
    fn boundary_error_not_char_boundary_display() {
        let e = BoundaryError::NotCharBoundary { byte: 2 };
        let s = e.to_string();
        assert!(s.contains("2"), "must include byte index");
    }

    /// NotGraphemeBoundary variant carries the byte index (C3).
    #[test]
    fn boundary_error_not_grapheme_boundary_display() {
        let e = BoundaryError::NotGraphemeBoundary { byte: 4 };
        let s = e.to_string();
        assert!(s.contains("4"), "must include byte index");
    }

    #[test]
    fn boundary_error_is_std_error() {
        let e = BoundaryError::OutOfRange { len: 1, asked: 2 };
        let _: &dyn std::error::Error = &e;
    }

    // ─── ParseErr ────────────────────────────────────────────────────────────

    /// Empty parse error has no sentinel in its display.
    #[test]
    fn parse_err_empty_display() {
        let s = ParseErr::Empty.to_string();
        assert!(s.contains("empty"), "must mention empty");
    }

    /// Invalid variant carries the byte index, expected, and found (C3 — RFC-0013 §4.1 I1).
    #[test]
    fn parse_err_invalid_display_carries_where_and_what() {
        let e = ParseErr::Invalid {
            at: 7,
            expected: "digit",
            found: "x".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("7"), "must include byte index");
        assert!(s.contains("digit"), "must include expected");
        assert!(s.contains("x"), "must include found");
    }

    /// OutOfRange variant carries the byte index and target type (C3 / FLAG Q3).
    #[test]
    fn parse_err_out_of_range_display_carries_at_and_target() {
        let e = ParseErr::OutOfRange {
            at: 0,
            target: "i64",
        };
        let s = e.to_string();
        assert!(s.contains("i64"), "must include target type");
        assert!(s.contains("0"), "must include byte index");
    }

    #[test]
    fn parse_err_is_std_error() {
        let e = ParseErr::Empty;
        let _: &dyn std::error::Error = &e;
    }

    // ─── EncodeError ─────────────────────────────────────────────────────────

    /// EncodeError carries the unrepresentable char, its index, and the target encoding (C3).
    #[test]
    fn encode_error_unrepresentable_display() {
        let e = EncodeError::Unrepresentable {
            ch: '€',
            at: 5,
            target_encoding: "Latin-1",
        };
        let s = e.to_string();
        assert!(
            s.contains('€'.to_string().as_str()),
            "must include the char"
        );
        assert!(s.contains("5"), "must include char index");
        assert!(s.contains("Latin-1"), "must include target encoding");
    }

    #[test]
    fn encode_error_is_std_error() {
        let e = EncodeError::Unrepresentable {
            ch: 'X',
            at: 0,
            target_encoding: "Latin-1",
        };
        let _: &dyn std::error::Error = &e;
    }

    // ─── TranscodeError ──────────────────────────────────────────────────────

    /// UnpairedSurrogate carries the unit index (C3).
    #[test]
    fn transcode_error_unpaired_surrogate_display() {
        let e = TranscodeError::UnpairedSurrogate { at: 3 };
        let s = e.to_string();
        assert!(s.contains("3"), "must include unit index");
        assert!(s.contains("surrogate"), "must mention surrogate");
    }

    /// Invalid carries the unit index (C3).
    #[test]
    fn transcode_error_invalid_display() {
        let e = TranscodeError::Invalid { at: 8 };
        let s = e.to_string();
        assert!(s.contains("8"), "must include unit index");
    }

    #[test]
    fn transcode_error_is_std_error() {
        let e = TranscodeError::UnpairedSurrogate { at: 0 };
        let _: &dyn std::error::Error = &e;
    }

    // ─── Cross-variant: distinct variants produce distinct messages (G11) ─────

    /// Different `BoundaryError` variants produce distinct display strings (G11 dual projection).
    /// Guard: returning the same string for every variant makes this fail.
    #[test]
    fn boundary_error_variants_have_distinct_displays() {
        let oor = BoundaryError::OutOfRange { len: 10, asked: 20 }.to_string();
        let ncb = BoundaryError::NotCharBoundary { byte: 1 }.to_string();
        let ngb = BoundaryError::NotGraphemeBoundary { byte: 1 }.to_string();
        assert_ne!(oor, ncb);
        assert_ne!(oor, ngb);
        assert_ne!(ncb, ngb);
    }

    /// Different `ParseErr` variants produce distinct display strings (G11).
    /// Guard: returning the same string for every variant makes this fail.
    #[test]
    fn parse_err_variants_have_distinct_displays() {
        let empty = ParseErr::Empty.to_string();
        let invalid = ParseErr::Invalid {
            at: 0,
            expected: "digit",
            found: "a".to_string(),
        }
        .to_string();
        let oor = ParseErr::OutOfRange {
            at: 0,
            target: "i32",
        }
        .to_string();
        assert_ne!(empty, invalid);
        assert_ne!(empty, oor);
        assert_ne!(invalid, oor);
    }
}
