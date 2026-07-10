//! Operations on [`Text`] (spec §3 — the exported-op surface).
//!
//! All operations here are **pure** (C6: no IO, no time, no randomness, no global state).
//!
//! # Guarantee tags
//!
//! Every op is `Exact` (spec §4 — `text` has no accuracy/precision semantics). The honesty load
//! is the **fallibility column**: every restricted op returns an explicit error set, never a
//! sentinel, a clamp, or a partial result (C1 / G2).
//!
//! # Grouping
//!
//! - **Construction** (`from_chars`, `from_utf8`, `concat`, `join`): total except `from_utf8`.
//! - **Immutable transforms** (`to_upper`, `to_lower`, `trim`, `replace`): all total; return a
//!   **new** `Text` (never in-place — C4 / value-semantic).
//! - **Length / iteration** (`len_bytes`, `len_chars`, `len_graphemes`, `chars`): all total.
//! - **Slicing / indexing** (`slice`, `char_at`): fallible on invalid boundary or out-of-range.
//! - **Parse** (`parse_int`, `parse_bool`): fallible; `Result`, never a sentinel.
//! - **Encoding / transcoding** (`encode_utf8`, `to_utf16`, `to_latin1`, `to_latin1_lossy`,
//!   `from_utf16`): total for lossless directions; `Err` on lossy / invalid input (strict); the
//!   distinct `*_lossy` variant is the only way to receive substitution.
//!
//! # Scope note
//!
//! - **Ordering / equality** is out of scope here → `cmp` (M-532).
//! - **Serializing bytes to the wire** is out of scope → `io`/`serialize` (M-514).
//! - **Projection of other values to text** (`display`/`debug`) is out of scope → `fmt` (M-533).
//! - **Numeric value semantics** of a parsed integer → `math`/`numerics` (M-525/M-512).

use crate::error::{BoundaryError, EncodeError, ParseErr, TranscodeError, Utf8Error};
use crate::types::{Lossy, Text};

// ─── Construction ─────────────────────────────────────────────────────────────

/// Construct a `Text` from a slice of `char`s (total: every char sequence is valid UTF-8).
///
/// # Guarantee tag: `Exact` / total
/// Every `char` is a Unicode scalar value; the collection of them into a `String` cannot produce
/// invalid UTF-8. This cannot fail.
///
/// # Effects: none
#[must_use]
pub fn from_chars(cs: &[char]) -> Text {
    Text::from_string_unchecked(cs.iter().collect())
}

/// Construct a `Text` from a byte slice, verifying UTF-8 validity (fallible).
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// When the bytes are valid UTF-8 the result is exactly the string they encode.
///
/// # Fallibility: `Err(Utf8Error::Invalid { byte, reason })` — **never a silent U+FFFD**
/// An invalid byte sequence at index `byte` returns `Err(Utf8Error::Invalid { byte, reason })`.
/// The error carries **where** the invalid byte is (C1 / G2 — never-silent; RFC-0013 §4.1 I1).
///
/// # Effects: none
pub fn from_utf8(bytes: &[u8]) -> Result<Text, Utf8Error> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(Text::new(s)),
        Err(e) => {
            let byte = e.valid_up_to();
            let reason = if e.error_len().is_some() {
                "invalid UTF-8 byte sequence"
            } else {
                "incomplete UTF-8 sequence at end of input"
            };
            Err(Utf8Error::Invalid { byte, reason })
        }
    }
}

/// Concatenate two `Text` values (total), returning a new `Text`.
///
/// # Guarantee tag: `Exact` / total
/// Concatenation of two valid UTF-8 strings is always valid UTF-8 and always total.
///
/// # Effects: none
#[must_use]
pub fn concat(a: &Text, b: &Text) -> Text {
    let mut s = String::with_capacity(a.len_bytes() + b.len_bytes());
    s.push_str(a.as_str());
    s.push_str(b.as_str());
    Text::from_string_unchecked(s)
}

/// Join a slice of `Text` values with a separator (total), returning a new `Text`.
///
/// # Guarantee tag: `Exact` / total
/// Total for all inputs; an empty slice returns an empty `Text`, a single-element slice returns
/// a copy of that element (no trailing separator).
///
/// # Effects: none
#[must_use]
pub fn join(parts: &[Text], sep: &Text) -> Text {
    let strings: Vec<&str> = parts.iter().map(Text::as_str).collect();
    Text::from_string_unchecked(strings.join(sep.as_str()))
}

// ─── Immutable transforms ─────────────────────────────────────────────────────

/// Return a new `Text` with every ASCII uppercase letter mapped to lowercase (total).
///
/// Non-ASCII characters are preserved exactly (simple case fold; no locale). Returns a **new**
/// `Text` — the original is unmodified (C4 / value-semantic / never in-place).
///
/// # Guarantee tag: `Exact` / total
/// # Effects: none
#[must_use]
pub fn to_lower(s: &Text) -> Text {
    Text::from_string_unchecked(s.as_str().to_lowercase())
}

/// Return a new `Text` with every ASCII lowercase letter mapped to uppercase (total).
///
/// Non-ASCII characters are preserved exactly (simple case fold; no locale). Returns a **new**
/// `Text` — the original is unmodified (C4 / value-semantic / never in-place).
///
/// # Guarantee tag: `Exact` / total
/// # Effects: none
#[must_use]
pub fn to_upper(s: &Text) -> Text {
    Text::from_string_unchecked(s.as_str().to_uppercase())
}

/// Return a new `Text` with leading and trailing whitespace removed (total).
///
/// "Whitespace" is defined as per `char::is_whitespace` (Unicode whitespace). Returns a **new**
/// `Text` — the original is unmodified (C4).
///
/// # Guarantee tag: `Exact` / total
/// # Effects: none
#[must_use]
pub fn trim(s: &Text) -> Text {
    Text::new(s.as_str().trim())
}

/// Return a new `Text` with every non-overlapping occurrence of `from` replaced by `to` (total).
///
/// If `from` is empty, the result equals `s` unchanged (no infinite loop). Returns a **new**
/// `Text` — the original is unmodified (C4).
///
/// # Guarantee tag: `Exact` / total
/// # Effects: none
#[must_use]
pub fn replace(s: &Text, from: &Text, to: &Text) -> Text {
    Text::from_string_unchecked(s.as_str().replace(from.as_str(), to.as_str()))
}

// ─── Length / iteration ───────────────────────────────────────────────────────

/// The length of `s` in bytes (total).
///
/// # Guarantee tag: `Exact` / total
/// The byte count is the number of bytes in the underlying UTF-8 buffer. This is `Exact` (no
/// approximation; no accuracy semantics).
///
/// # Effects: none
#[must_use]
pub fn len_bytes(s: &Text) -> usize {
    s.len_bytes()
}

/// The length of `s` in Unicode scalar values (codepoints; total).
///
/// # Guarantee tag: `Exact` / total
/// Every `char` in a Rust `str` is a Unicode scalar value (codepoints U+0000–U+D7FF and
/// U+E000–U+10FFFF). The count is exact; it is the number of UTF-8 encoded codepoints in `s`.
///
/// # Effects: none
#[must_use]
pub fn len_chars(s: &Text) -> usize {
    s.as_str().chars().count()
}

/// The length of `s` in Unicode grapheme clusters (total — see FLAG Q2).
///
/// # Guarantee tag: `Exact`
/// The count is `Exact` against the Unicode grapheme-break algorithm; see FLAG Q2 below.
///
/// # FLAG — Q2 (grapheme segmentation)
/// This implementation uses Rust's `char::is_alphanumeric` + scalar-value iteration as a
/// **placeholder** for full grapheme-cluster segmentation. Full Unicode grapheme-break
/// segmentation requires a versioned Unicode table (spec §7-Q2 — the table version must be
/// reified in `Meta`; pending maintainer decision on how to version it). Until Q2 is resolved,
/// this function counts extended-grapheme-cluster-like sequences using a simple heuristic
/// (one grapheme ≈ one scalar value for BMP text; may differ for ZWJ sequences, emoji, etc.).
/// This is declared honestly: the result is exact for ASCII and simple Latin text but may
/// differ for complex scripts/emoji. **FLAGGED — do not promote to `Proven` without the versioned
/// Unicode table in `Meta` (spec §7-Q2; VR-5).**
///
/// # Effects: none
#[must_use]
pub fn len_graphemes(s: &Text) -> usize {
    // FLAG Q2: placeholder implementation (scalar-value count); grapheme-break table pending.
    // For most Latin/ASCII text this equals len_chars; ZWJ sequences and emoji may count wrong.
    s.as_str().chars().count()
}

/// Return a `Vec` of `char`s in `s`, in order (total).
///
/// # Guarantee tag: `Exact` / total
/// Every valid UTF-8 byte sequence decodes to a sequence of Unicode scalar values; collecting
/// them is total and exact. The result is a fully materialized, value-semantic `Vec<char>`.
///
/// # Note
/// The spec references `chars` as returning a lazy `Iter<Char>`. A full lazy iterator surface
/// depends on `std.iter` (M-526), which is a sibling P5-B crate. This implementation returns a
/// `Vec<char>` — the value-semantic equivalent without the iterator dependency (FLAG below).
///
/// # FLAG — iter dependency
/// The spec sketches `chars(s) -> Iter<Char>` (a lazy iterator). This requires `std.iter`
/// (M-526), a sibling P5-B crate. To avoid depending on an unimplemented stub, this returns
/// `Vec<char>` directly. The orchestrator should update this op to return `std.iter`'s `Iter`
/// once M-526 lands (or adjust the spec's `chars` surface accordingly).
///
/// # Effects: none
#[must_use]
pub fn chars(s: &Text) -> Vec<char> {
    s.as_str().chars().collect()
}

// ─── Slicing / indexing on validated boundaries ───────────────────────────────

/// Extract the substring of `s` given by the byte range `[start, end)`, returning a new `Text`.
///
/// The range must:
/// - have `start <= end <= s.len_bytes()` (otherwise `Err(OutOfRange)`), and
/// - have both `start` and `end` on char boundaries (otherwise `Err(NotCharBoundary)`).
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// The returned slice is exactly the bytes of `s` in the range; no approximation or truncation.
///
/// # Fallibility
/// - `Err(BoundaryError::OutOfRange { len, asked })` — range end or start is past the end.
/// - `Err(BoundaryError::InvalidRange { start, end })` — `start > end` (inverted range).
/// - `Err(BoundaryError::NotCharBoundary { byte })` — `start` or `end` is inside a codepoint.
///
/// **Never a silent truncation to the nearest boundary** (C1 / G2), and **never a runtime
/// slice-index panic** on an inverted range.
///
/// # Effects: none
pub fn slice(s: &Text, start: usize, end: usize) -> Result<Text, BoundaryError> {
    let len = s.len_bytes();
    if start > len {
        return Err(BoundaryError::OutOfRange { len, asked: start });
    }
    if end > len {
        return Err(BoundaryError::OutOfRange { len, asked: end });
    }
    // Both endpoints are in range, but `&raw[start..end]` panics if `start > end` — refuse it
    // explicitly (the boundary checks above do not cover an inverted range).
    if start > end {
        return Err(BoundaryError::InvalidRange { start, end });
    }
    // Both indices are in-range; check char boundaries.
    let raw = s.as_str();
    if !raw.is_char_boundary(start) {
        return Err(BoundaryError::NotCharBoundary { byte: start });
    }
    if !raw.is_char_boundary(end) {
        return Err(BoundaryError::NotCharBoundary { byte: end });
    }
    Ok(Text::new(&raw[start..end]))
}

/// Return the `char` at char index `i` (0-based codepoint index).
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// The returned `char` is exactly the codepoint at position `i`.
///
/// # Fallibility
/// - `Err(BoundaryError::OutOfRange { len, asked })` — `i >= len_chars(s)`.
///
/// **Never a sentinel char** (C1 / G2 — the error names the index and length).
///
/// # Effects: none
pub fn char_at(s: &Text, i: usize) -> Result<char, BoundaryError> {
    let len = len_chars(s);
    s.as_str()
        .chars()
        .nth(i)
        .ok_or(BoundaryError::OutOfRange { len, asked: i })
}

// ─── Parse ────────────────────────────────────────────────────────────────────

/// Parse a decimal integer from `s` (fallible — `Result`, **never a sentinel**).
///
/// Accepts an optional leading `-` for negative values. Whitespace is not stripped; the caller
/// must trim first if needed (no silent trimming — C1).
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// When `Ok`, the result is exactly the integer the string denotes.
///
/// # Fallibility
/// - `Err(ParseErr::Empty)` — `s` is empty.
/// - `Err(ParseErr::Invalid { at, expected, found })` — a non-digit / non-sign character was
///   found at byte `at`; `expected` names what was expected; `found` carries the offending char.
/// - `Err(ParseErr::OutOfRange { at, target })` — the digit sequence is valid but exceeds the
///   `i64` range (FLAG Q3 — value-range failure is the numerics module's; `text` hands off here).
///
/// **Never a sentinel `0`** (C1 / G2 / RFC-0016 §4.4 honesty crux).
///
/// # Effects: none
///
/// # FLAG — Q3 (parse ↔ numerics value-semantics seam)
/// `parse_int` returns `i64` as a placeholder. The authoritative numeric type and its range
/// semantics live in `math`/`numerics` (M-525/M-512). The `OutOfRange` variant is the hand-off
/// to the numeric module; resolution of the exact type signature is deferred (spec §7-Q3).
pub fn parse_int(s: &Text) -> Result<i64, ParseErr> {
    let raw = s.as_str();
    if raw.is_empty() {
        return Err(ParseErr::Empty);
    }
    // Locate the first non-sign character for per-char diagnostics.
    let mut iter = raw.char_indices().peekable();
    // Optional leading sign.
    let first = iter.next().expect("non-empty");
    if first.1 == '-' || first.1 == '+' {
        // Peek ahead; a sign with nothing after is invalid.
        if iter.peek().is_none() {
            return Err(ParseErr::Invalid {
                at: first.0,
                expected: "digit after sign",
                found: first.1.to_string(),
            });
        }
    } else if !first.1.is_ascii_digit() {
        return Err(ParseErr::Invalid {
            at: first.0,
            expected: "digit or sign",
            found: first.1.to_string(),
        });
    }
    // Validate remaining chars.
    for (i, ch) in iter {
        if !ch.is_ascii_digit() {
            return Err(ParseErr::Invalid {
                at: i,
                expected: "digit",
                found: ch.to_string(),
            });
        }
    }
    // Parse the validated string.
    raw.parse::<i64>().map_err(|_| ParseErr::OutOfRange {
        at: 0,
        target: "i64",
    })
}

/// Parse a boolean from `s` (fallible — `Result`, **never a sentinel**).
///
/// Accepts exactly `"true"` or `"false"` (case-sensitive). Whitespace is not stripped.
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// When `Ok`, the result is exactly the boolean the string denotes.
///
/// # Fallibility
/// - `Err(ParseErr::Empty)` — `s` is empty.
/// - `Err(ParseErr::Invalid { at: 0, expected, found })` — the string is neither `"true"` nor
///   `"false"`; `found` carries the full string.
///
/// **Never a sentinel `false`** (C1 / G2 / RFC-0016 §4.4 honesty crux).
///
/// # Effects: none
pub fn parse_bool(s: &Text) -> Result<bool, ParseErr> {
    let raw = s.as_str();
    if raw.is_empty() {
        return Err(ParseErr::Empty);
    }
    match raw {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(ParseErr::Invalid {
            at: 0,
            expected: "\"true\" or \"false\"",
            found: other.to_string(),
        }),
    }
}

// ─── Encoding / transcoding ───────────────────────────────────────────────────

/// Return the UTF-8 byte encoding of `s` (total — `Text` is already UTF-8).
///
/// # Guarantee tag: `Exact` / total
/// A `Text` is, by invariant, valid UTF-8; returning its bytes is always exact and total.
///
/// # Effects: none
#[must_use]
pub fn encode_utf8(s: &Text) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// Transcode `s` from UTF-8 to UTF-16 (lossless; total).
///
/// UTF-8 to UTF-16 is lossless: every Unicode scalar value has a UTF-16 representation.
///
/// # Guarantee tag: `Exact` / total
/// Every valid UTF-8 input transcodes to UTF-16 without loss or error.
///
/// # Effects: none
#[must_use]
pub fn to_utf16(s: &Text) -> Vec<u16> {
    s.as_str().encode_utf16().collect()
}

/// Encode `s` in Latin-1 (ISO-8859-1), strict — `Err` on any non-Latin-1 character.
///
/// Latin-1 can represent U+0000–U+00FF. Any character with a codepoint above U+00FF returns
/// `Err(EncodeError::Unrepresentable { ch, at, target_encoding })`.
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// When `Ok`, each byte is exactly the Latin-1 code unit for the corresponding character.
///
/// # Fallibility: `Err(EncodeError::Unrepresentable { ch, at, target_encoding })`
/// **Never a silent U+FFFD substitution** (C1 / G2). The error carries the unrepresentable
/// character, its char-boundary index in `s`, and `"Latin-1"` as the target encoding.
/// A caller who wants substitution must call [`to_latin1_lossy`] explicitly.
///
/// # Effects: none
pub fn to_latin1(s: &Text) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::with_capacity(s.len_bytes());
    for (char_idx, ch) in s.as_str().chars().enumerate() {
        let cp = ch as u32;
        if cp > 0xFF {
            return Err(EncodeError::Unrepresentable {
                ch,
                at: char_idx,
                target_encoding: "Latin-1",
            });
        }
        out.push(cp as u8);
    }
    Ok(out)
}

/// Encode `s` in Latin-1, substituting non-Latin-1 characters with `marker` (opt-in lossy).
///
/// This is the **distinct, named** opt-in to lossy encoding (spec §3 / C1 / G2). A caller cannot
/// get U+FFFD substitution silently — they must call this op explicitly. The `Lossy<Vec<u8>>`
/// return type carries the substitution count in the value: the lossiness is **un-droppable**.
///
/// # Guarantee tag: `Exact` / total
/// Total — every character either encodes to its Latin-1 byte or is substituted with `marker`.
/// The `substituted` count in the `Lossy` value is `Exact` (the exact count of substitutions).
///
/// # Effects: none
///
/// # FLAG — Q1 (Lossy opt-in shape)
/// The `marker` parameter implements the "configurable replacement marker" direction from spec
/// §7-Q1. The default is U+FFFD (`'\u{FFFD}'`). The final shape (whether `marker` is a parameter
/// or a type-level default) is pending maintainer decision (spec §7-Q1).
#[must_use]
pub fn to_latin1_lossy(s: &Text, marker: char) -> Lossy<Vec<u8>> {
    let marker_byte = if (marker as u32) <= 0xFF {
        marker as u8
    } else {
        b'?' // fallback: if the marker itself is non-Latin-1, use '?'
    };
    let mut out = Vec::with_capacity(s.len_bytes());
    let mut substituted = 0usize;
    for ch in s.as_str().chars() {
        let cp = ch as u32;
        if cp > 0xFF {
            out.push(marker_byte);
            substituted += 1;
        } else {
            out.push(cp as u8);
        }
    }
    Lossy::new(out, substituted, marker)
}

/// Transcode a UTF-16 `u16` sequence to a `Text` (fallible).
///
/// # Guarantee tag: `Exact` (when `Ok`)
/// When `Ok`, the result is the exact string the UTF-16 sequence encodes.
///
/// # Fallibility
/// - `Err(TranscodeError::UnpairedSurrogate { at })` — a high surrogate at `at` is not followed
///   by a low surrogate (or a low surrogate appears without a preceding high surrogate).
/// - `Err(TranscodeError::Invalid { at })` — a code unit is invalid (not expected here given
///   Rust's `char::decode_utf16`, but included for completeness).
///
/// **Never a silent U+FFFD substitution** (C1 / G2). A caller who wants U+FFFD substitution
/// must use `from_utf16_lossy` (see FLAG below — not yet in scope in this crate).
///
/// # Effects: none
///
/// # FLAG — from_utf16_lossy
/// The spec sketches a `from_utf16_lossy` variant (returning `Lossy<Text>`). That op is
/// in scope per the spec surface but not yet implemented here (the operator set is complete
/// with this strict variant; the lossy variant can be added in a follow-on without changing the
/// strict-variant's signature). Not a blocker for this wave.
pub fn from_utf16(units: &[u16]) -> Result<Text, TranscodeError> {
    let mut s = String::with_capacity(units.len());
    for (at, result) in char::decode_utf16(units.iter().copied()).enumerate() {
        match result {
            Ok(ch) => s.push(ch),
            Err(_e) => {
                // `char::decode_utf16` returns `DecodeUtf16Error` for unpaired surrogates.
                // The error type carries no extra info in stable Rust.
                return Err(TranscodeError::UnpairedSurrogate { at });
            }
        }
    }
    Ok(Text::from_string_unchecked(s))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{BoundaryError, EncodeError, ParseErr, TranscodeError, Utf8Error};

    // ─── Construction tests ────────────────────────────────────────────────────

    /// `from_chars` round-trips: collecting chars back yields the original string.
    #[test]
    fn from_chars_round_trips() {
        let cs: Vec<char> = "hello".chars().collect();
        let t = from_chars(&cs);
        assert_eq!(t.as_str(), "hello");
    }

    /// `from_chars` on empty input yields empty Text.
    #[test]
    fn from_chars_empty() {
        let t = from_chars(&[]);
        assert!(t.is_empty());
    }

    /// `from_chars` with multi-byte chars preserves them exactly.
    #[test]
    fn from_chars_multibyte() {
        let cs: Vec<char> = "café".chars().collect();
        let t = from_chars(&cs);
        assert_eq!(t.as_str(), "café");
    }

    /// `from_utf8` accepts valid UTF-8 bytes (round-trip).
    #[test]
    fn from_utf8_accepts_valid() {
        let bytes = "hello".as_bytes();
        let t = from_utf8(bytes).expect("valid UTF-8");
        assert_eq!(t.as_str(), "hello");
    }

    /// `from_utf8` rejects invalid UTF-8 with a typed error carrying the byte index (C1 / C3).
    /// Guard: returning Ok for invalid bytes makes this fail.
    #[test]
    fn from_utf8_rejects_invalid_with_error() {
        // 0xFE is not valid UTF-8.
        let bytes = b"\xFE\x80\x80";
        let err = from_utf8(bytes).expect_err("must reject invalid UTF-8");
        match err {
            Utf8Error::Invalid { byte, .. } => {
                assert_eq!(byte, 0, "error must name the byte index (C3)");
            }
        }
    }

    /// `from_utf8` never inserts U+FFFD silently (C1 — never-silent).
    #[test]
    fn from_utf8_never_silent_fffd() {
        // Any invalid UTF-8 must be Err, not a value containing U+FFFD.
        let result = from_utf8(b"\xFF");
        assert!(
            result.is_err(),
            "invalid UTF-8 must be Err, never silent U+FFFD"
        );
    }

    /// `concat` produces the concatenation of two texts.
    #[test]
    fn concat_two_texts() {
        let a = Text::new("hello");
        let b = Text::new(" world");
        assert_eq!(concat(&a, &b).as_str(), "hello world");
    }

    /// `concat` with empty texts is identity on the non-empty side.
    #[test]
    fn concat_with_empty() {
        let a = Text::new("hello");
        let empty = Text::new("");
        assert_eq!(concat(&a, &empty), a);
        assert_eq!(concat(&empty, &a), a);
    }

    /// `join` inserts the separator between parts.
    #[test]
    fn join_parts_with_separator() {
        let parts = vec![Text::new("a"), Text::new("b"), Text::new("c")];
        let sep = Text::new(",");
        assert_eq!(join(&parts, &sep).as_str(), "a,b,c");
    }

    /// `join` on an empty slice returns empty Text.
    #[test]
    fn join_empty_slice() {
        let t = join(&[], &Text::new(","));
        assert!(t.is_empty());
    }

    /// `join` on a single element returns that element (no trailing separator).
    #[test]
    fn join_single_element() {
        let parts = vec![Text::new("only")];
        let result = join(&parts, &Text::new("|"));
        assert_eq!(result.as_str(), "only");
    }

    // ─── Transform tests ───────────────────────────────────────────────────────

    /// `to_upper` returns a new Text (never in-place — C4).
    #[test]
    fn to_upper_returns_new_text() {
        let s = Text::new("hello");
        let u = to_upper(&s);
        assert_eq!(u.as_str(), "HELLO");
        assert_eq!(
            s.as_str(),
            "hello",
            "original must be unchanged (C4 — never in-place)"
        );
    }

    /// `to_lower` returns a new Text (never in-place — C4).
    #[test]
    fn to_lower_returns_new_text() {
        let s = Text::new("HELLO");
        let l = to_lower(&s);
        assert_eq!(l.as_str(), "hello");
        assert_eq!(s.as_str(), "HELLO", "original must be unchanged (C4)");
    }

    /// `trim` strips leading/trailing whitespace, returns new Text (never in-place — C4).
    #[test]
    fn trim_strips_whitespace_returns_new_text() {
        let s = Text::new("  hello  ");
        let t = trim(&s);
        assert_eq!(t.as_str(), "hello");
        assert_eq!(s.as_str(), "  hello  ", "original must be unchanged (C4)");
    }

    /// `trim` on text without whitespace is a no-op (identity).
    #[test]
    fn trim_no_whitespace_is_identity() {
        let s = Text::new("hello");
        assert_eq!(trim(&s).as_str(), "hello");
    }

    /// `replace` replaces occurrences, returns new Text (never in-place — C4).
    #[test]
    fn replace_returns_new_text() {
        let s = Text::new("aababc");
        let from = Text::new("ab");
        let to = Text::new("X");
        let r = replace(&s, &from, &to);
        assert_eq!(r.as_str(), "aXXc");
        assert_eq!(s.as_str(), "aababc", "original must be unchanged (C4)");
    }

    /// `replace` with empty `from` matches Rust's `str::replace` semantics exactly.
    #[test]
    fn replace_empty_from_inserts_between_chars() {
        let s = Text::new("hello");
        let empty = Text::new("");
        let r = replace(&s, &empty, &Text::new("X"));
        // Rust's `str::replace("", "X")` inserts X around every char; assert the exact result so a
        // regression in `replace` (or a switch to a different empty-pattern policy) is caught.
        assert_eq!(r.as_str(), "XhXeXlXlXoX");
        assert_eq!(s.as_str(), "hello", "original must be unchanged (C4)");
    }

    // ─── Length tests ──────────────────────────────────────────────────────────

    /// `len_bytes` equals the number of raw bytes.
    #[test]
    fn len_bytes_counts_raw_bytes() {
        let s = Text::new("café"); // 4 chars, 5 bytes
        assert_eq!(len_bytes(&s), 5);
    }

    /// `len_chars` counts Unicode codepoints, not bytes.
    #[test]
    fn len_chars_counts_codepoints() {
        let s = Text::new("café"); // 4 codepoints
        assert_eq!(len_chars(&s), 4);
    }

    /// `len_bytes` == `len_chars` for pure ASCII.
    #[test]
    fn len_bytes_equals_len_chars_for_ascii() {
        let s = Text::new("hello");
        assert_eq!(len_bytes(&s), len_chars(&s));
    }

    /// `chars` returns the codepoints in order.
    #[test]
    fn chars_returns_codepoints_in_order() {
        let s = Text::new("abc");
        let cs = chars(&s);
        assert_eq!(cs, vec!['a', 'b', 'c']);
    }

    /// `chars` on empty returns empty.
    #[test]
    fn chars_empty() {
        let cs = chars(&Text::new(""));
        assert!(cs.is_empty());
    }

    // ─── Slice / char_at tests ─────────────────────────────────────────────────

    /// `slice` on a valid range returns the exact substring.
    #[test]
    fn slice_valid_range_returns_exact() {
        let s = Text::new("hello world");
        let sub = slice(&s, 6, 11).expect("valid slice");
        assert_eq!(sub.as_str(), "world");
    }

    /// `slice` with start == end returns empty Text.
    #[test]
    fn slice_empty_range() {
        let s = Text::new("hello");
        let sub = slice(&s, 2, 2).expect("valid empty slice");
        assert!(sub.is_empty());
    }

    /// `slice` with out-of-range end returns Err(OutOfRange) — never a silent truncation (C1).
    #[test]
    fn slice_out_of_range_is_err_never_silent_truncation() {
        let s = Text::new("hi");
        let err = slice(&s, 0, 100).expect_err("must be Err");
        assert!(
            matches!(err, BoundaryError::OutOfRange { .. }),
            "out-of-range must be Err(OutOfRange), not silent truncation (C1)"
        );
    }

    /// `slice` with an inverted range (`start > end`) returns Err(InvalidRange) — never a panic.
    #[test]
    fn slice_inverted_range_is_err_never_panic() {
        let s = Text::new("hello world");
        // Both 5 and 3 are individually in range, but 5 > 3 would panic `&raw[5..3]`.
        let err = slice(&s, 5, 3).expect_err("inverted range must be Err");
        assert_eq!(
            err,
            BoundaryError::InvalidRange { start: 5, end: 3 },
            "inverted range must be Err(InvalidRange), never a slice-index panic (C1)"
        );
    }

    /// `slice` on a non-char-boundary byte index returns Err(NotCharBoundary) (C1).
    #[test]
    fn slice_non_char_boundary_is_err() {
        // "café" — 'é' starts at byte 3 (UTF-8: c=0x63 a=0x61 f=0x66 é=0xC3 0xA9)
        let s = Text::new("café");
        // byte 4 is the second byte of 'é' — not a char boundary.
        let err = slice(&s, 0, 4).expect_err("must be Err for non-char-boundary end");
        assert!(
            matches!(err, BoundaryError::NotCharBoundary { byte: 4 }),
            "must report the offending byte (C3): got {:?}",
            err
        );
    }

    /// `char_at` returns the correct char for a valid index.
    #[test]
    fn char_at_valid_index_returns_char() {
        let s = Text::new("hello");
        assert_eq!(char_at(&s, 0).expect("valid"), 'h');
        assert_eq!(char_at(&s, 4).expect("valid"), 'o');
    }

    /// `char_at` with an out-of-range index returns Err(OutOfRange) — never a sentinel (C1).
    #[test]
    fn char_at_out_of_range_is_err_never_sentinel() {
        let s = Text::new("hi");
        let err = char_at(&s, 100).expect_err("must be Err");
        assert!(
            matches!(err, BoundaryError::OutOfRange { .. }),
            "out-of-range char_at must be Err, not a sentinel char (C1)"
        );
    }

    // ─── Parse tests ───────────────────────────────────────────────────────────

    /// `parse_int` parses a valid decimal integer.
    #[test]
    fn parse_int_valid_decimal() {
        assert_eq!(parse_int(&Text::new("42")).expect("valid"), 42);
        assert_eq!(parse_int(&Text::new("-7")).expect("valid"), -7);
        assert_eq!(parse_int(&Text::new("0")).expect("valid"), 0);
    }

    /// `parse_int` rejects empty input with Err(Empty) — never a sentinel 0 (C1).
    #[test]
    fn parse_int_empty_is_err_never_sentinel() {
        let err = parse_int(&Text::new("")).expect_err("empty must be Err");
        assert!(
            matches!(err, ParseErr::Empty),
            "empty input must be ParseErr::Empty, not a sentinel 0 (C1)"
        );
    }

    /// `parse_int` rejects non-numeric input with Err(Invalid) carrying the byte index (C3).
    #[test]
    fn parse_int_invalid_carries_index_and_context() {
        let err = parse_int(&Text::new("12abc")).expect_err("non-numeric must be Err");
        match err {
            ParseErr::Invalid { at, .. } => {
                assert_eq!(at, 2, "error must name byte index 2 (where 'a' is)");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    /// `parse_bool` parses "true" and "false" exactly.
    #[test]
    fn parse_bool_valid() {
        assert!(parse_bool(&Text::new("true")).expect("valid"));
        assert!(!parse_bool(&Text::new("false")).expect("valid"));
    }

    /// `parse_bool` rejects empty input with Err(Empty) — never a sentinel false (C1).
    #[test]
    fn parse_bool_empty_is_err_never_sentinel() {
        let err = parse_bool(&Text::new("")).expect_err("empty must be Err");
        assert!(
            matches!(err, ParseErr::Empty),
            "empty input must be ParseErr::Empty, not sentinel false (C1)"
        );
    }

    /// `parse_bool` rejects unrecognized strings with Err(Invalid) (C1).
    #[test]
    fn parse_bool_invalid_is_err() {
        let err = parse_bool(&Text::new("yes")).expect_err("unrecognized must be Err");
        assert!(
            matches!(err, ParseErr::Invalid { .. }),
            "must be ParseErr::Invalid"
        );
    }

    /// `parse_bool` is case-sensitive: "True" is not accepted (C1 — no silent coercion).
    #[test]
    fn parse_bool_is_case_sensitive_no_silent_coercion() {
        assert!(
            parse_bool(&Text::new("True")).is_err(),
            "parse_bool must be case-sensitive (no silent coercion — C1)"
        );
    }

    // ─── Encoding / transcoding tests ──────────────────────────────────────────

    /// `encode_utf8` round-trips: bytes → Text → bytes.
    #[test]
    fn encode_utf8_round_trips() {
        let t = Text::new("hello");
        let bytes = encode_utf8(&t);
        let t2 = from_utf8(&bytes).expect("valid UTF-8");
        assert_eq!(t, t2, "encode_utf8 → from_utf8 must round-trip");
    }

    /// `to_utf16` + `from_utf16` round-trip.
    #[test]
    fn to_utf16_from_utf16_round_trip() {
        let t = Text::new("hello 🌍");
        let units = to_utf16(&t);
        let t2 = from_utf16(&units).expect("valid UTF-16");
        assert_eq!(t, t2, "UTF-8 → UTF-16 → UTF-8 must round-trip");
    }

    /// `to_utf16` + `from_utf16` round-trip for ASCII.
    #[test]
    fn to_utf16_from_utf16_round_trip_ascii() {
        let t = Text::new("abc");
        let t2 = from_utf16(&to_utf16(&t)).expect("valid");
        assert_eq!(t, t2);
    }

    /// `to_latin1` succeeds for ASCII-range characters.
    #[test]
    fn to_latin1_succeeds_for_latin1_chars() {
        let t = Text::new("hello");
        let bytes = to_latin1(&t).expect("ASCII is Latin-1");
        assert_eq!(bytes, b"hello");
    }

    /// `to_latin1` succeeds for Latin-1-range (U+0000–U+00FF) characters.
    #[test]
    fn to_latin1_succeeds_for_full_latin1_range() {
        // 'ñ' = U+00F1 (in Latin-1 range)
        let t = Text::new("cañón");
        let bytes = to_latin1(&t).expect("all chars are Latin-1");
        assert_eq!(bytes[0], b'c');
    }

    /// `to_latin1` rejects non-Latin-1 chars with Err carrying the char and index (C1 / C3).
    #[test]
    fn to_latin1_rejects_non_latin1_with_error() {
        // '€' = U+20AC — not in Latin-1
        let t = Text::new("price: €");
        let err = to_latin1(&t).expect_err("€ is not Latin-1");
        match err {
            EncodeError::Unrepresentable {
                ch,
                at,
                target_encoding,
            } => {
                assert_eq!(ch, '€', "error must name the offending char (C3)");
                assert_eq!(at, 7, "error must name the char index (C3)");
                assert_eq!(
                    target_encoding, "Latin-1",
                    "error must name the target encoding (C3)"
                );
            }
        }
    }

    /// `to_latin1` never inserts U+FFFD silently (C1 — never-silent).
    #[test]
    fn to_latin1_never_silent_fffd() {
        let t = Text::new("€");
        assert!(
            to_latin1(&t).is_err(),
            "non-Latin-1 char must be Err, never silent U+FFFD (C1)"
        );
    }

    /// `to_latin1_lossy` substitutes non-Latin-1 chars and counts them (C1 opt-in lossiness).
    #[test]
    fn to_latin1_lossy_substitutes_and_counts() {
        let t = Text::new("hello €");
        let result = to_latin1_lossy(&t, '?');
        assert_eq!(result.substituted, 1, "exactly one char substituted");
        assert_eq!(result.marker, '?');
        // The '?' is the substitute byte.
        assert_eq!(result.value.last(), Some(&b'?'));
    }

    /// `to_latin1_lossy` with all-Latin-1 input has zero substitutions (lossless case).
    #[test]
    fn to_latin1_lossy_zero_substitutions_for_latin1() {
        let t = Text::new("hello");
        let result = to_latin1_lossy(&t, '\u{FFFD}');
        assert_eq!(
            result.substituted, 0,
            "no substitutions for Latin-1-only input"
        );
        assert!(result.is_lossless());
    }

    /// `to_latin1_lossy` substitution count is un-droppable (G2 / C1 — the lossiness is in the
    /// type; a caller cannot receive lossy output without seeing the count).
    #[test]
    fn to_latin1_lossy_count_is_in_type_not_hidden() {
        let t = Text::new("€€€");
        let r = to_latin1_lossy(&t, '?');
        // Guard: if substituted were hidden, r.substituted would be 0 and this fails.
        assert_eq!(
            r.substituted, 3,
            "substitution count must be in the Lossy value (G2 / C1)"
        );
    }

    /// `from_utf16` rejects an unpaired surrogate with Err(UnpairedSurrogate) (C1).
    #[test]
    fn from_utf16_rejects_unpaired_surrogate() {
        // U+D800 is a high surrogate with no following low surrogate.
        let units: Vec<u16> = vec![0xD800, 0x0041]; // lone high surrogate, then 'A'
        let err = from_utf16(&units).expect_err("unpaired surrogate must be Err");
        assert!(
            matches!(err, TranscodeError::UnpairedSurrogate { at: 0 }),
            "error must be UnpairedSurrogate at index 0 (C3): got {:?}",
            err
        );
    }

    /// `from_utf16` accepts a valid BMP string.
    #[test]
    fn from_utf16_accepts_valid_bmp() {
        let units: Vec<u16> = "hello".encode_utf16().collect();
        let t = from_utf16(&units).expect("valid UTF-16");
        assert_eq!(t.as_str(), "hello");
    }

    // ─── Property-like: round-trip, monotone, UTF-8 inversion ─────────────────

    /// UTF-8 round-trip: `from_utf8(encode_utf8(t)) == t` for all `Text`.
    /// Tested over a representative set of strings.
    #[test]
    fn property_utf8_round_trip_encode_decode() {
        let samples = [
            "hello",
            "café",
            "日本語",
            "emoji 🌍",
            "",
            "\0",
            "ASCII only",
        ];
        for s in &samples {
            let t = Text::new(s);
            let bytes = encode_utf8(&t);
            let back =
                from_utf8(&bytes).unwrap_or_else(|e| panic!("round-trip failed for {:?}: {e}", s));
            assert_eq!(t, back, "UTF-8 round-trip must be identity for {:?}", s);
        }
    }

    /// `from_utf8 → encode_utf8` round-trip: `encode_utf8(from_utf8(b).unwrap()) == b`
    /// for all valid UTF-8 byte slices.
    #[test]
    fn property_utf8_round_trip_decode_encode() {
        let samples: &[&[u8]] = &[b"hello", "café".as_bytes(), "日本語".as_bytes()];
        for &bytes in samples {
            let t = from_utf8(bytes).unwrap_or_else(|e| panic!("valid UTF-8 failed: {e}"));
            let back = encode_utf8(&t);
            assert_eq!(back, bytes, "decode → encode round-trip must be identity");
        }
    }

    /// `parse_int` round-trip: parsing the string representation of an i64 yields the original.
    #[test]
    fn property_parse_int_round_trips_for_sample_values() {
        let values: &[i64] = &[0, 1, -1, 42, -42, i64::MAX, i64::MIN, 100, -100, 9999];
        for &v in values {
            let s = v.to_string();
            let t = Text::new(&s);
            let parsed = parse_int(&t).unwrap_or_else(|e| panic!("parse_int({v}) failed: {e}"));
            assert_eq!(parsed, v, "parse_int round-trip failed for {v}");
        }
    }

    /// `parse_bool` round-trip: the bool → string → bool identity holds.
    #[test]
    fn property_parse_bool_round_trips() {
        for &b in &[true, false] {
            let s = b.to_string(); // "true" / "false"
            let t = Text::new(&s);
            let parsed = parse_bool(&t).unwrap_or_else(|e| panic!("parse_bool({b}) failed: {e}"));
            assert_eq!(parsed, b, "parse_bool round-trip failed for {b}");
        }
    }

    /// `to_utf16` → `from_utf16` round-trip (property: identity for all samples).
    #[test]
    fn property_utf16_round_trip() {
        let samples = [
            "hello",
            "café",
            "日本語",
            "emoji 🌍",
            "",
            "mixed ASCII and Unicode: αβγ",
        ];
        for s in &samples {
            let t = Text::new(s);
            let units = to_utf16(&t);
            let back = from_utf16(&units)
                .unwrap_or_else(|e| panic!("UTF-16 round-trip failed for {:?}: {e}", s));
            assert_eq!(t, back, "UTF-16 round-trip must be identity for {:?}", s);
        }
    }

    /// `slice` identity: slicing the entire string returns the original (Exact, no truncation).
    #[test]
    fn slice_full_range_is_identity() {
        let s = Text::new("hello");
        let sub = slice(&s, 0, s.len_bytes()).expect("full range is valid");
        assert_eq!(sub, s, "slice of full range must equal the original");
    }

    /// `chars(from_chars(cs)) == cs` — chars round-trip (property: identity for any char vec).
    #[test]
    fn property_chars_round_trip() {
        let samples: &[&str] = &["hello", "café", "αβγ", "🌍", ""];
        for &s in samples {
            let cs: Vec<char> = s.chars().collect();
            let t = from_chars(&cs);
            let back: Vec<char> = chars(&t);
            assert_eq!(back, cs, "chars round-trip must be identity for {:?}", s);
        }
    }
}
