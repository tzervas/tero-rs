//! Explicit error types for `std.content` (C1 — never-silent; RFC-0016 §4.1).
//!
//! Every fallible operation in this crate returns a typed error or `Option`, never a sentinel
//! value. [`MalformedDigest`] is the only error variant the module produces — [`crate::parse_ref`]
//! is the only parse-fallible entry point (spec §3).

use std::fmt;

/// The content-address string is not well-formed (`<algo>:<digest>` shape; RFC-0001 §4.6).
///
/// # C1 compliance
/// [`crate::parse_ref`] returns `Err(MalformedDigest)` rather than a silently-coerced or
/// zeroed digest (RFC-0016 §4.1 C1; G2 "never-silent").
///
/// The `description` field carries a human-readable explanation of *why* the string is rejected,
/// enabling callers to surface the error without stripping information (G11 dual projection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalformedDigest {
    /// The rejected input.
    pub input: String,
    /// Why the input was rejected (G11: a human-readable explanation for the caller to surface).
    pub description: &'static str,
}

impl MalformedDigest {
    pub(crate) fn new(input: impl Into<String>, description: &'static str) -> Self {
        MalformedDigest {
            input: input.into(),
            description,
        }
    }
}

impl fmt::Display for MalformedDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "malformed content address {:?}: {}",
            self.input, self.description
        )
    }
}

impl std::error::Error for MalformedDigest {}

#[cfg(test)]
mod tests {
    use super::MalformedDigest;

    #[test]
    fn malformed_digest_display_includes_input_and_description() {
        // Guard: mutation of `input` or `description` makes this fail.
        let e = MalformedDigest::new("bad", "missing colon");
        let s = e.to_string();
        assert!(s.contains("bad"), "display must include the rejected input");
        assert!(
            s.contains("missing colon"),
            "display must include the description"
        );
    }

    #[test]
    fn malformed_digest_is_an_std_error() {
        // Compile-time check: it satisfies the std::error::Error bound.
        let e = MalformedDigest::new("x", "test");
        let _: &dyn std::error::Error = &e;
    }
}
