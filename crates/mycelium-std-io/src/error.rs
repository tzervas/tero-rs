//! Explicit error types for `std.io` + `serialize` (C1 — never-silent; RFC-0016 §4.1).
//!
//! Every fallible operation returns a typed error carrying an RFC-0013 diagnostic
//! **locus** (byte offset / field path) — never a sentinel value or a silent partial
//! result (C1/G2). Two distinct error families:
//!
//! - [`SerError`] — (de)serialization failures: truncated input, malformed grammar,
//!   unknown tag, value-model invariant violation, budget exceeded.
//! - [`IoError`] — byte-movement failures: unexpected EOF, substrate refusal, effect
//!   budget overrun.
//!
//! Both implement [`std::error::Error`] and carry a `Display` projection that names
//! the failure locus — the G11 dual human/machine projection (RFC-0013 §4.3 / I1).
//!
//! # Design spec
//! `docs/spec/stdlib/io.md` §3/§5 (C1); RFC-0013 §4.3 (the diagnostic record); C6
//! (declared bounded effects — `BudgetExceeded`/`EffectBudget` carry the `kind`).

use std::fmt;

// ── Locus types ──────────────────────────────────────────────────────────────

/// A byte offset into the input: the **locus** of a serialization failure (C1 /
/// RFC-0013 I1). Named `none` (0) when the locus cannot be determined (e.g. the
/// first byte of a completely empty input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteOffset(pub u64);

impl fmt::Display for ByteOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "byte {}", self.0)
    }
}

/// A field path into a structured value, naming *where* inside a composite the
/// failure was detected (C1/RFC-0013 I1). Encoded as a `/`-separated string of
/// field names — e.g. `"repr/width"`, `"meta/bound/delta"`.
///
/// This is the structural analogue of [`ByteOffset`] for grammar-structured failures
/// where the byte position is less informative than the semantic path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPath(pub String);

impl FieldPath {
    /// Construct from a static description.
    #[must_use]
    pub fn from_static(s: &'static str) -> Self {
        FieldPath(s.to_owned())
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "field path {}", self.0)
    }
}

// ── SerError ─────────────────────────────────────────────────────────────────

/// The explicit error set for (de)serialization failures (C1/RFC-0013; spec §3).
///
/// Every variant carries a locus (byte offset or field path) so the caller can
/// surface *where* the decode failed — never a locationless "parse error" (G2/C3).
/// An overrun yields `BudgetExceeded`, never an OOM or a hang (C6/ADR-015).
///
/// # C1 compliance
/// No variant of `SerError` is a sentinel, a clamp, or a partial-result indicator.
/// The presence of a `SerError` means the entire decode is rejected; the caller
/// never receives a partially-filled `Value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerError {
    /// The input ended before a complete value was decoded.
    ///
    /// `at` is the byte offset where the input terminated (the last byte
    /// successfully consumed before the truncation was detected).
    Truncated {
        /// Byte offset of the truncation.
        at: ByteOffset,
    },
    /// The bytes at `at` do not conform to the wire/JSON grammar.
    ///
    /// `why` is a human-readable explanation of the grammar rule that was
    /// violated (G11 dual projection — machine-parseable in the tag, human-
    /// legible in the description).
    Malformed {
        /// Byte offset of the malformed datum.
        at: ByteOffset,
        /// Why the datum was rejected.
        why: String,
    },
    /// An unrecognized `Repr`, constructor, or `Meta` tag was encountered at `path`.
    ///
    /// Carrying the rejected `tag` name enables the caller to log it and report
    /// it exactly (G11); an unknown tag is never silently skipped (C1/G2).
    UnknownTag {
        /// Structural path to the field that held the unknown tag.
        path: FieldPath,
        /// The unrecognized tag string.
        tag: String,
    },
    /// A field decoded successfully but violates a value-model invariant (RFC-0001
    /// §4.3 — e.g. payload length ≠ repr width, bound delta ∉ \[0,1\]).
    ///
    /// The invariant name is carried in `why` so the caller can surface it.
    OutOfDomain {
        /// Structural path to the violating field.
        path: FieldPath,
        /// The violated invariant.
        why: String,
    },
    /// A declared decode budget (depth limit, enumeration ceiling — ADR-015) was
    /// exceeded. `kind` names which budget: `"depth"`, `"enum"`, `"alloc"`, etc.
    ///
    /// An overrun is an explicit error, never a hang or an OOM (C6).
    BudgetExceeded {
        /// The budget kind that was exceeded (e.g. `"depth"`, `"enum"`).
        kind: String,
    },
}

impl fmt::Display for SerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerError::Truncated { at } => write!(f, "truncated input at {at}"),
            SerError::Malformed { at, why } => write!(f, "malformed input at {at}: {why}"),
            SerError::UnknownTag { path, tag } => {
                write!(f, "unknown tag {tag:?} at {path}")
            }
            SerError::OutOfDomain { path, why } => {
                write!(f, "value out of domain at {path}: {why}")
            }
            SerError::BudgetExceeded { kind } => {
                write!(f, "decode budget exceeded ({kind})")
            }
        }
    }
}

mycelium_std_core::impl_std_error!(SerError);

// ── IoError ──────────────────────────────────────────────────────────────────

/// The number of bytes successfully read before an error.
///
/// Carried by [`IoError::UnexpectedEof`] to name *how much* was read before the
/// source was exhausted — enabling the caller to report precisely without
/// re-reading (C1/RFC-0013 I1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteCount(pub u64);

impl fmt::Display for ByteCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} bytes", self.0)
    }
}

/// The explicit error set for byte-movement failures (C1/RFC-0013; spec §3).
///
/// Every variant is a declared, inspectable failure mode — never a silent partial
/// read (C1) and never a hang or OOM (C6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    /// The source/sink closed before the requested number of bytes were moved.
    ///
    /// `read` is the count of bytes that *were* moved before the EOF — enabling
    /// the caller to report the shortfall precisely (RFC-0013 I1).
    UnexpectedEof {
        /// Bytes successfully read before the EOF.
        read: ByteCount,
    },
    /// The underlying substrate refused the operation.
    ///
    /// `why` carries the refusal reason (G11 dual projection); for the in-memory
    /// substrate this is always the explicit reason the substrate declined.
    Refused {
        /// Why the operation was refused.
        why: String,
    },
    /// A bounded io/alloc effect budget was exceeded (RFC-0014 §4.5 / C6).
    ///
    /// `kind` names which budget: `"io"`, `"alloc"`, etc.
    EffectBudget {
        /// The budget kind that was exceeded.
        kind: String,
    },
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::UnexpectedEof { read } => write!(f, "unexpected EOF after {read}"),
            IoError::Refused { why } => write!(f, "substrate refused: {why}"),
            IoError::EffectBudget { kind } => write!(f, "effect budget exceeded ({kind})"),
        }
    }
}

mycelium_std_core::impl_std_error!(IoError);

// ── Combined error for read_value ────────────────────────────────────────────

/// A combined error from [`crate::io::read_value`]: either a byte-movement failure
/// (`Io`) or a (de)serialization failure (`Ser`). The two failure modes are kept
/// distinct so the caller can handle each class separately (C3 — no black box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadValueError {
    /// A byte-movement failure occurred before or during decode.
    Io(IoError),
    /// The bytes were read but the decode failed.
    Ser(SerError),
}

impl fmt::Display for ReadValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadValueError::Io(e) => write!(f, "io error: {e}"),
            ReadValueError::Ser(e) => write!(f, "serialize error: {e}"),
        }
    }
}

mycelium_std_core::impl_std_error!(
    ReadValueError,
    source = |this| {
        match this {
            ReadValueError::Io(e) => Some(e),
            ReadValueError::Ser(e) => Some(e),
        }
    }
);

impl From<IoError> for ReadValueError {
    fn from(e: IoError) -> Self {
        ReadValueError::Io(e)
    }
}

impl From<SerError> for ReadValueError {
    fn from(e: SerError) -> Self {
        ReadValueError::Ser(e)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SerError Display / locus tests ────────────────────────────────────────

    /// `SerError::Truncated` Display includes the byte offset (RFC-0013 I1 / G11).
    /// Guard: a display that drops the offset makes this fail.
    #[test]
    fn ser_error_truncated_display_includes_offset() {
        let e = SerError::Truncated { at: ByteOffset(42) };
        let s = e.to_string();
        assert!(s.contains("42"), "display must include the byte offset");
    }

    /// `SerError::Malformed` Display includes both the offset and the reason.
    /// Guard: dropping either field makes this fail.
    #[test]
    fn ser_error_malformed_display_includes_offset_and_why() {
        let e = SerError::Malformed {
            at: ByteOffset(7),
            why: "unexpected 0xFF in bit string".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("7"), "display must include the offset");
        assert!(
            s.contains("0xFF"),
            "display must include the rejection reason"
        );
    }

    /// `SerError::UnknownTag` Display includes both the path and the tag.
    /// Guard: dropping either field makes this fail.
    #[test]
    fn ser_error_unknown_tag_display_includes_path_and_tag() {
        let e = SerError::UnknownTag {
            path: FieldPath::from_static("repr/kind"),
            tag: "Quantum".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("repr/kind"), "must include the field path");
        assert!(s.contains("Quantum"), "must include the rejected tag");
    }

    /// `SerError::OutOfDomain` Display includes both the path and the reason.
    #[test]
    fn ser_error_out_of_domain_display_includes_path_and_why() {
        let e = SerError::OutOfDomain {
            path: FieldPath::from_static("meta/bound/delta"),
            why: "delta 1.5 ∉ [0,1]".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("meta/bound/delta"), "must include the path");
        assert!(s.contains("1.5"), "must include the why");
    }

    /// `SerError::BudgetExceeded` Display includes the budget kind.
    #[test]
    fn ser_error_budget_exceeded_display_includes_kind() {
        let e = SerError::BudgetExceeded {
            kind: "enum".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("enum"), "must include the budget kind");
    }

    /// `SerError` implements `std::error::Error` (compile-time check).
    #[test]
    fn ser_error_is_std_error() {
        let e = SerError::Truncated { at: ByteOffset(0) };
        let _: &dyn std::error::Error = &e;
    }

    // ── IoError Display tests ─────────────────────────────────────────────────

    /// `IoError::UnexpectedEof` Display includes the byte count (RFC-0013 I1).
    #[test]
    fn io_error_unexpected_eof_display_includes_count() {
        let e = IoError::UnexpectedEof {
            read: ByteCount(15),
        };
        let s = e.to_string();
        assert!(s.contains("15"), "must include the byte count");
    }

    /// `IoError::Refused` Display includes the reason.
    #[test]
    fn io_error_refused_display_includes_why() {
        let e = IoError::Refused {
            why: "substrate is consumed".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("consumed"), "must include the refusal reason");
    }

    /// `IoError::EffectBudget` Display includes the budget kind.
    #[test]
    fn io_error_effect_budget_display_includes_kind() {
        let e = IoError::EffectBudget {
            kind: "alloc".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("alloc"), "must include the budget kind");
    }

    /// `IoError` implements `std::error::Error`.
    #[test]
    fn io_error_is_std_error() {
        let e = IoError::Refused {
            why: "test".to_owned(),
        };
        let _: &dyn std::error::Error = &e;
    }

    // ── ReadValueError tests ──────────────────────────────────────────────────

    /// `ReadValueError` distinguishes the two error classes (C3).
    /// Guard: making From<IoError> and From<SerError> both map to the same variant
    /// would collapse the discrimination — this test catches that.
    #[test]
    fn read_value_error_distinguishes_io_and_ser() {
        let io_err: ReadValueError = IoError::Refused {
            why: "test".to_owned(),
        }
        .into();
        let ser_err: ReadValueError = SerError::Truncated { at: ByteOffset(0) }.into();
        assert!(matches!(io_err, ReadValueError::Io(_)));
        assert!(matches!(ser_err, ReadValueError::Ser(_)));
        assert_ne!(io_err, ser_err);
    }

    /// `ReadValueError` implements `std::error::Error` and exposes `source()`.
    #[test]
    fn read_value_error_has_source() {
        let e: ReadValueError = IoError::Refused {
            why: "x".to_owned(),
        }
        .into();
        assert!(
            std::error::Error::source(&e).is_some(),
            "source() must chain to the underlying IoError"
        );
    }
}
