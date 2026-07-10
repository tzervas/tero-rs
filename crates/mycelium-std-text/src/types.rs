//! Core value types for `std.text`: [`Text`] and [`Lossy<T>`].
//!
//! # `Text`
//! An immutable, UTF-8 encoded string value. Value-semantic: every transform returns a **new**
//! `Text`; there is no in-place mutation (C4 / ADR-003). Two `Text` values with equal content
//! are the same value (content-addressed identity — RFC-0001 §4.6); metadata is not identity
//! (ADR-003).
//!
//! # `Lossy<T>`
//! The **type-level opt-in** to lossy transcoding (spec §3 / C1). A caller cannot receive a
//! lossy result silently — they must explicitly call a `*_lossy` operation that returns
//! `Lossy<T>`. The `substituted` count and the `marker` character are always in the value, so
//! the lossiness is un-droppable (G2 / C1).
//!
//! # FLAG — Q1 (Lossy opt-in shape)
//! Whether `Lossy` is a distinct type (as here) or a `Meta`-attached substitution count, and
//! whether the replacement marker is configurable, is deferred to spec §7-Q1. This implementation
//! uses a distinct `Lossy<T>` type with a default U+FFFD marker, following the spec's proposed
//! disposition (spec §7-Q1: "distinct `Lossy` type with U+FFFD-default-but-overridable marker").

use std::fmt;

// ─── Text ─────────────────────────────────────────────────────────────────────

/// An immutable, UTF-8 encoded string value (spec §1 / §3).
///
/// Invariant: the internal bytes are always valid UTF-8.
///
/// Value-semantic: two `Text` values with identical bytes compare as equal (content-addressed,
/// RFC-0001 §4.6); every transform returns a new `Text` (C4 / ADR-003 — metadata is not
/// identity; immutability is the structural form of C4 for text).
///
/// # Guarantee tag: `Exact`
/// `Text` carries no accuracy/precision semantics. Construction either succeeds (valid UTF-8)
/// or returns an explicit `Err` (C1 — never-silent).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Text {
    inner: String,
}

impl Text {
    /// Construct a `Text` from a `&str` slice (total — any `&str` is valid UTF-8).
    ///
    /// This is a pure convenience constructor. The `FromStr` trait implementation delegates here.
    ///
    /// # Guarantee: `Exact` / total
    /// Every `&str` is already validated UTF-8 by Rust's type system; this cannot fail.
    #[must_use]
    pub fn new(s: &str) -> Text {
        Text {
            inner: s.to_owned(),
        }
    }

    /// View the internal UTF-8 bytes as a `&str` (total, by-invariant).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// View the internal bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// The length in bytes (C2: `Exact`; total).
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.inner.len()
    }

    /// Is the text empty (zero bytes)?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Decompose into the inner `String`, consuming the `Text`.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.inner
    }

    /// Construct directly from a validated `String` (crate-internal use; bypass UTF-8 check).
    ///
    /// # Safety / invariant
    /// The caller must guarantee `s` is valid UTF-8. Rust `String` already enforces this, so
    /// this is safe by the type system; the function is `pub(crate)` to prevent external bypass.
    pub(crate) fn from_string_unchecked(s: String) -> Text {
        // Rust `String` invariant guarantees UTF-8; this is pure construction, not unsafe.
        Text { inner: s }
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner)
    }
}

impl From<String> for Text {
    fn from(s: String) -> Self {
        Text { inner: s }
    }
}

impl From<&str> for Text {
    fn from(s: &str) -> Self {
        Text::new(s)
    }
}

impl std::str::FromStr for Text {
    /// `Text::new` is infallible; `Text`'s `FromStr` impl is also infallible: every `&str` is valid UTF-8 by Rust's type system.
    /// The error type is `std::convert::Infallible` (the never-fails error).
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Text::new(s))
    }
}

impl AsRef<str> for Text {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<str> for Text {
    fn eq(&self, other: &str) -> bool {
        self.inner == other
    }
}

impl PartialEq<&str> for Text {
    fn eq(&self, other: &&str) -> bool {
        self.inner == *other
    }
}

// ─── Lossy<T> ─────────────────────────────────────────────────────────────────

/// The **type-level opt-in** to lossy transcoding (spec §3 / C1 / G2).
///
/// A `Lossy<T>` is the only way to receive a value that was produced by substituting characters
/// that were not representable in the target encoding. The `substituted` count and `marker`
/// character are always present in the value — the lossiness is **un-droppable** (G2 / C1).
///
/// The default replacement character is U+FFFD (the Unicode replacement character). A caller
/// who wants a different replacement character passes it explicitly to `*_lossy` operations.
///
/// # FLAG — Q1 (Lossy opt-in shape)
/// This is a distinct type per spec §7-Q1's proposed disposition. Whether the replacement marker
/// should be configurable (as here, via `marker`) vs fixed at U+FFFD is pending maintainer
/// decision. The `marker` field exists to anticipate the configurable direction without guessing
/// (G2 / VR-5).
///
/// # Guarantee tag: `Exact` / total
/// The `Lossy` wrapper itself carries no accuracy semantics — it is a total, value-semantic
/// record of what was substituted. The `substituted` count is `Exact` (an exact integer count
/// of the substitutions that occurred).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lossy<T> {
    /// The transcoded/encoded value (with substitutions applied).
    pub value: T,
    /// The count of characters that were substituted.
    pub substituted: usize,
    /// The replacement character that was used (default U+FFFD).
    pub marker: char,
}

impl<T> Lossy<T> {
    /// Construct a `Lossy<T>` with a given value, substitution count, and marker.
    #[must_use]
    pub fn new(value: T, substituted: usize, marker: char) -> Self {
        Lossy {
            value,
            substituted,
            marker,
        }
    }

    /// Construct a `Lossy<T>` using the default U+FFFD replacement marker.
    #[must_use]
    pub fn with_default_marker(value: T, substituted: usize) -> Self {
        Lossy {
            value,
            substituted,
            marker: '\u{FFFD}',
        }
    }

    /// Whether any substitutions occurred (convenience; the caller can also inspect `substituted`).
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.substituted == 0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Text ─────────────────────────────────────────────────────────────────

    /// `Text::new` round-trips: the content is exactly the input (C4 — exact).
    #[test]
    fn text_new_round_trips() {
        let t = Text::new("hello");
        assert_eq!(t.as_str(), "hello");
    }

    /// Empty text is empty and reports zero bytes.
    #[test]
    fn text_empty() {
        let t = Text::new("");
        assert!(t.is_empty());
        assert_eq!(t.len_bytes(), 0);
    }

    /// Multi-byte UTF-8 strings are preserved exactly (C4).
    #[test]
    fn text_new_preserves_multibyte_utf8() {
        // "café" has 5 chars but 6 bytes (é is 2 bytes in UTF-8).
        let s = "café";
        let t = Text::new(s);
        assert_eq!(t.as_str(), s);
        assert_eq!(t.len_bytes(), s.len());
    }

    /// `Text` equality is content-addressed: two texts with equal bytes are equal (RFC-0001 §4.6 / C4).
    #[test]
    fn text_equality_is_content_addressed() {
        let a = Text::new("hello");
        let b = Text::new("hello");
        assert_eq!(
            a, b,
            "texts with equal content must be equal (C4 / RFC-0001 §4.6)"
        );
    }

    /// Texts with different content are not equal.
    #[test]
    fn text_different_content_not_equal() {
        let a = Text::new("hello");
        let b = Text::new("world");
        assert_ne!(a, b);
    }

    /// `Display` renders the text content verbatim.
    #[test]
    fn text_display_is_content() {
        let t = Text::new("greet");
        assert_eq!(t.to_string(), "greet");
    }

    /// `From<String>` and `From<&str>` both yield the correct text.
    #[test]
    fn text_from_conversions() {
        let t1 = Text::from("hello");
        let t2 = Text::from(String::from("hello"));
        assert_eq!(t1, t2);
    }

    /// `AsRef<str>` exposes the underlying string slice.
    #[test]
    fn text_as_ref_str() {
        let t = Text::new("test");
        let s: &str = t.as_ref();
        assert_eq!(s, "test");
    }

    // ─── Lossy<T> ─────────────────────────────────────────────────────────────

    /// `Lossy::with_default_marker` uses U+FFFD (the spec's proposed default marker).
    #[test]
    fn lossy_default_marker_is_replacement_char() {
        let l = Lossy::<Vec<u8>>::with_default_marker(vec![0x3F], 2);
        assert_eq!(l.marker, '\u{FFFD}');
        assert_eq!(l.substituted, 2);
    }

    /// `Lossy::is_lossless` returns true iff `substituted == 0`.
    #[test]
    fn lossy_is_lossless_iff_zero_substitutions() {
        let lossless = Lossy::<Vec<u8>>::with_default_marker(vec![], 0);
        let lossy = Lossy::<Vec<u8>>::with_default_marker(vec![0x3F], 1);
        assert!(lossless.is_lossless(), "zero substitutions → lossless");
        assert!(!lossy.is_lossless(), "one substitution → not lossless");
    }

    /// `Lossy` carries the substitution count in the value — it is un-droppable (G2 / C1).
    /// Guard: a struct with no `substituted` field would make this assertion fail.
    #[test]
    fn lossy_substitution_count_is_in_the_value() {
        let l = Lossy::new(vec![b'?'], 3_usize, '?');
        assert_eq!(
            l.substituted, 3,
            "substitution count must be in the Lossy value (G2 / C1)"
        );
        assert_eq!(l.marker, '?');
    }

    /// Equality: two `Lossy` values are equal iff their `value`, `substituted`, and `marker` are
    /// equal (content-addressed).
    #[test]
    fn lossy_equality_considers_all_fields() {
        let a = Lossy::new(vec![b'x'], 1_usize, '\u{FFFD}');
        let b = Lossy::new(vec![b'x'], 1_usize, '\u{FFFD}');
        let c = Lossy::new(vec![b'x'], 2_usize, '\u{FFFD}'); // different count
        assert_eq!(a, b);
        assert_ne!(a, c, "different substitution counts must not be equal");
    }
}
