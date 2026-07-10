//! Explicit parse diagnostics (never-silent; S5/G2): every failure is a [`ParseError`] carrying a
//! source position and a message — the parser never panics on malformed input and never silently
//! accepts it.

use crate::token::Pos;

/// A parse/lex failure at a source position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Where the error was detected.
    pub pos: Pos,
    /// What went wrong.
    pub message: String,
}

impl ParseError {
    /// Build an error at `pos`.
    #[must_use]
    pub fn new(pos: Pos, message: String) -> Self {
        ParseError { pos, message }
    }

    /// Ergonomic alias for [`ParseError::new`] taking any `impl Into<String>` message (so a `&str`
    /// literal needs no `.to_owned()` at the call site). Additive: [`new`](ParseError::new) is
    /// unchanged and still the canonical constructor.
    #[must_use]
    pub fn at(pos: Pos, message: impl Into<String>) -> Self {
        ParseError {
            pos,
            message: message.into(),
        }
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "parse error at {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for ParseError {}
