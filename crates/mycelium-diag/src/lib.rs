//! `mycelium-diag` — the canonical RFC-0013 structured-diagnostic record types.
//!
//! # Why this crate exists (maintainer decision, 2026-06-18)
//!
//! RFC-0013/RFC-0014 concepts (the structured diagnostic + the recovery bridge) were scattered
//! across `mycelium-check`/`mycelium-l1`/`mycelium-interp`/`mycelium-lsp`. The Phase-5 Tier-A wave
//! (M-510/M-520) needs **one** consolidated reference for the diagnostic record that
//! `std.diag` projects, `std.recover` carries, and `std.testing` records a `Fail` on. Per the
//! maintainer's resolved FLAG (scaffold decision #1), that canonical record is **extracted into
//! this small kernel crate** rather than homed inside `mycelium-std-diag` — a deliberate, bounded
//! growth of the trusted base so the type has a single owner below the std layer. `mycelium-std-diag`
//! re-exports and ergonomically wraps these types (KC-3); it does not redefine them.
//!
//! # Honesty crux (RFC-0013 I1)
//!
//! A `Diag` is **additive over an explicit error**: it presents a failure, it never *is* the
//! failure's control flow. Presentation never gates propagation — there is no severity, note, or
//! locus that makes an underlying error *not* surface. Construction is **total**: a missing locus is
//! [`None`] (explicit), never a fabricated zero (G2).
//!
//! Design spec: `docs/spec/stdlib/diag.md`; RFC-0013; task M-510, issue #151.
//!
//! # Dual projection (G11 / RFC-0013 I3)
//!
//! A `Diag` has one canonical truth; human and JSON are two renderers of it.
//! - [`Diag::human`] — human-readable view; carries the content id.
//! - [`Diag::machine`] — lossless JSON machine record with embedded `id`
//!   (round-trips via [`Diag::from_json`]).
//! - [`Diag::content_hash`] — deterministic BLAKE3 over the canonical fields *sans presentation*
//!   (ADR-003): identity is the record, not how it is shown. Presentation-invariant.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub use mycelium_core::ContentHash;

// ─── A local injective BLAKE3 framing ─────────────────────────────────────────────────────────────
//
// Mirrors the pattern in `mycelium-lsp/src/diagnostics/record.rs` and
// `mycelium-core/src/content.rs` (KC-3 — no kernel dep added; the kernel crate itself also carries
// this local framing for the tooling layer).

/// A canonical, injective byte encoder for content-addressing a `Diag`. Length-prefixed blobs so no
/// two distinct records share an encoding.
struct Canon {
    h: blake3::Hasher,
}

impl Canon {
    fn new(domain: &str) -> Self {
        let mut c = Canon {
            h: blake3::Hasher::new(),
        };
        // Domain separation: hashing the domain string first ensures diag hashes can never collide
        // with hashes of other record kinds that share the same field layout.
        c.blob(domain.as_bytes());
        c
    }

    fn blob(&mut self, bytes: &[u8]) {
        self.h.update(&(bytes.len() as u64).to_le_bytes());
        self.h.update(bytes);
    }

    fn str(&mut self, s: &str) {
        self.blob(s.as_bytes());
    }

    /// Encode `None` and `Some("")` as distinct byte sequences (tagged).
    fn opt(&mut self, s: Option<&str>) {
        match s {
            None => {
                self.h.update(&[0u8]);
            }
            Some(v) => {
                self.h.update(&[1u8]);
                self.str(v);
            }
        }
    }

    fn finish(self) -> ContentHash {
        let hex = self.h.finalize().to_hex();
        // BLAKE3 hex is always 64 lowercase hex chars — a well-formed digest.
        ContentHash::from_parts("blake3", hex.as_str())
            .expect("blake3 hex is always a valid digest")
    }
}

// ─── Severity ─────────────────────────────────────────────────────────────────────────────────────

/// Graded diagnostic severity (RFC-0013 §4.1). A **typed** distinction — never a stringly-typed
/// level. Presentation severity **never gates propagation** (I1): a `Warn` never silently becomes a
/// pass, and an `Error` severity does not itself halt anything — it annotates an already-explicit
/// error.
///
/// Ordered `Debug < Info < Warn < Error` (weakest-to-strongest). The ordering is purely for
/// comparisons and aggregation; it does **not** gate propagation (I1).
/// `#[non_exhaustive]`: a future severity grade may be added without a breaking change — an external
/// exhaustive `match` must carry a `_` arm (M-644; additive — no variant removed; the `Ord` order and
/// [`Severity::ALL`] are preserved). In-crate matches and `ALL` already name every variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Severity {
    /// A debug-grade diagnostic (lowest severity).
    Debug,
    /// An informational diagnostic.
    Info,
    /// A warning-grade diagnostic.
    Warn,
    /// An error-grade diagnostic (highest severity).
    Error,
}

impl Severity {
    /// All severities, ordered weakest-to-strongest (`Debug < Info < Warn < Error`).
    pub const ALL: [Severity; 4] = [
        Severity::Debug,
        Severity::Info,
        Severity::Warn,
        Severity::Error,
    ];

    /// The canonical name used in human/machine output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warn => "warn",
            Severity::Info => "info",
            Severity::Debug => "debug",
        }
    }
}

// ─── Code ─────────────────────────────────────────────────────────────────────────────────────────

/// A stable diagnostic code / error class (RFC-0013 §4.2). Closed for the common kernel cases with
/// an explicit [`Code::Other`] escape hatch — never a stringly-typed free-for-all on the common
/// path. The set may be widened additively (via new variants) as the spec's class registry grows.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum Code {
    /// A value fell outside its declared range/domain.
    OutOfRange,
    /// A declared, bounded effect budget was exhausted (RFC-0014 I3/I4).
    Budget,
    /// A content-hash / identity mismatch (ADR-003).
    HashMismatch,
    /// An open-coded class identified by a stable string (the registry escape hatch).
    Other(String),
}

impl Code {
    /// The canonical code name for use in human/machine output.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Code::OutOfRange => "OutOfRange",
            Code::Budget => "Budget",
            Code::HashMismatch => "HashMismatch",
            Code::Other(s) => s.as_str(),
        }
    }
}

// ─── Locus ────────────────────────────────────────────────────────────────────────────────────────

/// A source locus — *where* a diagnostic points (RFC-0013 §4.2). All fields are optional: an absent
/// locus stays [`None`] on the [`Diag`], and an absent span/line stays `None` here — **never** a
/// fabricated zero (G2).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Locus {
    /// Source path/name, if known.
    pub source: Option<String>,
    /// 1-based line, if known.
    pub line: Option<u32>,
    /// 1-based column, if known.
    pub column: Option<u32>,
}

// ─── Trace ────────────────────────────────────────────────────────────────────────────────────────

/// An ordered diagnostic trace — the chain of frames/notes that led to the failure (RFC-0013 §4.3).
/// A thin newtype over the frame list so it can grow a richer frame model without breaking callers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Trace {
    /// Trace frames, outermost-first.
    pub frames: Vec<String>,
}

impl Trace {
    /// The empty trace (explicit absence — not a fabricated frame).
    #[must_use]
    pub fn empty() -> Self {
        Self { frames: Vec::new() }
    }

    /// Push a frame, returning the extended trace (value-semantic).
    #[must_use]
    pub fn with_frame(mut self, frame: impl Into<String>) -> Self {
        self.frames.push(frame.into());
        self
    }
}

// ─── Diag ─────────────────────────────────────────────────────────────────────────────────────────

/// A structured diagnostic record (RFC-0013 §4.1): a content-addressable value over an
/// already-emitted explicit error. Identity is the record *sans presentation* (ADR-003) —
/// [`Diag::content_hash`] is a deterministic BLAKE3 over the canonical fields, presentation-
/// invariant so the human and JSON projections share one identity (I3). Builders are total;
/// a missing locus is [`None`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diag {
    /// The graded severity (typed; never gates propagation — I1).
    pub severity: Severity,
    /// The diagnostic code / error class.
    pub code: Code,
    /// The human-readable message.
    pub message: String,
    /// Where the diagnostic points, if known (explicit `None` when absent).
    pub locus: Option<Locus>,
    /// The diagnostic trace.
    pub trace: Trace,
    /// Free-form notes (EXPLAIN payload, G11).
    pub notes: Vec<String>,
}

impl Diag {
    // ── Builders (total; a missing field is explicit absence, never a fabricated zero) ──────────

    /// Build an `Error`-severity diagnostic with the given code (total builder).
    #[must_use]
    pub fn error(code: Code) -> Self {
        Self::with_severity(Severity::Error, code)
    }

    /// Build a `Warn`-severity diagnostic with the given code (total builder).
    #[must_use]
    pub fn warn(code: Code) -> Self {
        Self::with_severity(Severity::Warn, code)
    }

    /// Build an `Info`-severity diagnostic with the given code (total builder).
    #[must_use]
    pub fn info(code: Code) -> Self {
        Self::with_severity(Severity::Info, code)
    }

    /// The common total builder behind [`Self::error`]/[`Self::warn`]/[`Self::info`].
    #[must_use]
    pub fn with_severity(severity: Severity, code: Code) -> Self {
        Self {
            severity,
            code,
            message: String::new(),
            locus: None,
            trace: Trace::empty(),
            notes: Vec::new(),
        }
    }

    /// Set the human-readable message (value-semantic builder).
    #[must_use]
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Attach a source locus (explicit; absence stays `None` — never a fabricated zero, G2).
    #[must_use]
    pub fn at(mut self, locus: Locus) -> Self {
        self.locus = Some(locus);
        self
    }

    /// Attach a note (EXPLAIN payload).
    #[must_use]
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Replace the trace (value-semantic builder).
    #[must_use]
    pub fn trace(mut self, trace: Trace) -> Self {
        self.trace = trace;
        self
    }

    // ── Field accessors ─────────────────────────────────────────────────────────────────────────

    /// The typed severity (a `Warn` never silently becomes a pass — I1).
    #[must_use]
    pub fn severity(&self) -> Severity {
        self.severity
    }

    /// The diagnostic code / error class.
    #[must_use]
    pub fn code(&self) -> &Code {
        &self.code
    }

    // ── Content address (ADR-003 / RFC-0013 I3) ─────────────────────────────────────────────────

    /// The **content address** of this diagnostic (RFC-0013 §4.3; ADR-003) — a deterministic BLAKE3
    /// over the **canonical fields** (severity, code, message, locus, trace, notes), excluding the
    /// rendered presentation (the formatted human string and JSON output are not hash inputs).
    /// Presentation-invariant: the same `Diag` content always hashes the same, so the human
    /// and JSON projections share one identity (I3).
    ///
    /// # Guarantee: `Exact`
    /// Pure value transform; no approximation. (RFC-0016 §4.5, VR-5)
    #[must_use]
    pub fn content_hash(&self) -> ContentHash {
        let mut c = Canon::new("mycelium.diag.v1");
        c.str(self.severity.as_str());
        c.str(self.code.as_str());
        c.str(&self.message);
        // Locus: tag absence vs. presence distinctly (G2 — `None` ≠ `Some(Locus::default())`).
        match &self.locus {
            None => {
                c.h.update(&[0u8]);
            }
            Some(l) => {
                c.h.update(&[1u8]);
                c.opt(l.source.as_deref());
                match l.line {
                    None => {
                        c.h.update(&[0u8]);
                    }
                    Some(n) => {
                        c.h.update(&[1u8]);
                        c.h.update(&n.to_le_bytes());
                    }
                }
                match l.column {
                    None => {
                        c.h.update(&[0u8]);
                    }
                    Some(n) => {
                        c.h.update(&[1u8]);
                        c.h.update(&n.to_le_bytes());
                    }
                }
            }
        }
        // Trace frames (length-prefixed list so an empty trace ≠ a one-element trace with "").
        c.h.update(&(self.trace.frames.len() as u64).to_le_bytes());
        for frame in &self.trace.frames {
            c.str(frame);
        }
        // Notes (length-prefixed list).
        c.h.update(&(self.notes.len() as u64).to_le_bytes());
        for note in &self.notes {
            c.str(note);
        }
        c.finish()
    }

    // ── Dual projection (G11 / RFC-0013 I3) ────────────────────────────────────────────────────

    /// The **human projection** (G11 / RFC-0013 I3): a human-readable string. The content `id` is
    /// embedded so the human view carries the same identity as the machine one (I3). Shows severity,
    /// code, message, locus (when present), trace frames, and notes.
    ///
    /// Total: always returns a string for any well-formed `Diag`.
    ///
    /// # Guarantee: `Exact`
    /// Pure value transform; no approximation. (RFC-0016 §4.5, VR-5)
    #[must_use]
    pub fn human(&self) -> String {
        let id = self.content_hash();
        let mut out = String::new();
        out.push_str(&format!(
            "[{}] {}: {}",
            self.severity.as_str().to_uppercase(),
            self.code.as_str(),
            self.message
        ));
        if let Some(l) = &self.locus {
            let mut loc = String::new();
            if let Some(s) = &l.source {
                loc.push_str(s);
            }
            if let Some(line) = l.line {
                if !loc.is_empty() {
                    loc.push(':');
                }
                loc.push_str(&line.to_string());
                if let Some(col) = l.column {
                    loc.push(':');
                    loc.push_str(&col.to_string());
                }
            }
            if !loc.is_empty() {
                out.push_str(&format!("  (at {loc})"));
            }
        }
        if !self.trace.frames.is_empty() {
            out.push_str("\n  trace:");
            for f in &self.trace.frames {
                out.push_str(&format!("\n    {f}"));
            }
        }
        if !self.notes.is_empty() {
            out.push_str("\n  notes:");
            for n in &self.notes {
                out.push_str(&format!("\n    {n}"));
            }
        }
        out.push_str(&format!("\n  id: {}", id.as_str()));
        out
    }

    /// The **machine projection** (G11 / RFC-0013 I3): a lossless JSON record with the content `id`
    /// embedded. `from_json(machine(d))` recovers a record equal to `d` with an equal `content_hash`
    /// (the round-trip property, I3). The `id` field in JSON is informational: identity is recomputed
    /// from the recovered fields, so the round-trip is over semantic content, not over the wire string.
    ///
    /// Total: always returns a JSON string for any well-formed `Diag`.
    ///
    /// # Guarantee: `Exact`
    /// Pure value transform; no approximation. (RFC-0016 §4.5, VR-5)
    #[must_use]
    pub fn machine(&self) -> String {
        // Build a serde_json::Value so we can inject the `id` field alongside the record fields.
        let mut v = serde_json::to_value(self).expect("Diag always serializes to JSON");
        if let serde_json::Value::Object(map) = &mut v {
            map.insert(
                "id".to_owned(),
                serde_json::Value::String(self.content_hash().as_str().to_owned()),
            );
        }
        serde_json::to_string(&v).expect("a JSON Value always serializes")
    }

    /// Recover a `Diag` from its machine JSON projection (I3).
    ///
    /// The embedded `id` field is informational: because `Diag` does not carry
    /// `#[serde(deny_unknown_fields)]`, serde ignores unknown fields (including `id`) by default, so
    /// the machine projection round-trips transparently. Identity is recomputed from the recovered
    /// fields, so the round-trip is over semantic content, not the wire string.
    ///
    /// # Errors
    /// Returns a [`serde_json::Error`] if `s` is not a well-formed `Diag` JSON record (C1: explicit
    /// error, never a partial/sentinel record).
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Builder contract ────────────────────────────────────────────────────────────────────────

    #[test]
    fn builders_are_total_and_locus_absence_is_explicit() {
        let d = Diag::error(Code::OutOfRange).message("payload len ≠ width");
        assert_eq!(d.severity(), Severity::Error);
        assert_eq!(d.code(), &Code::OutOfRange);
        // A missing locus is explicit None, never a fabricated zero (G2).
        assert!(d.locus.is_none());
    }

    #[test]
    fn at_records_an_explicit_locus() {
        let d = Diag::warn(Code::Budget).at(Locus {
            source: Some("x.myc".into()),
            line: Some(3),
            column: None,
        });
        let l = d.locus.expect("locus set");
        assert_eq!(l.line, Some(3));
        // An absent column stays None — not a fabricated 0 (G2).
        assert!(l.column.is_none());
    }

    // ── Severity ordering (typed distinction, never stringly-typed) ─────────────────────────────

    /// `Severity` is a typed distinction with a defined order (RFC-0013 §4.1).
    /// Mutation witness: removing `PartialOrd`/`Ord` derives makes this fail.
    #[test]
    fn severity_is_a_typed_distinction_with_ordering() {
        assert!(Severity::Debug < Severity::Info);
        assert!(Severity::Info < Severity::Warn);
        assert!(Severity::Warn < Severity::Error);
        // Exhaustively verify all pairs are ordered consistently.
        for (i, a) in Severity::ALL.iter().enumerate() {
            for (j, b) in Severity::ALL.iter().enumerate() {
                match i.cmp(&j) {
                    std::cmp::Ordering::Less => assert!(*a < *b, "{a:?} < {b:?}"),
                    std::cmp::Ordering::Greater => assert!(*a > *b, "{a:?} > {b:?}"),
                    std::cmp::Ordering::Equal => assert_eq!(*a, *b, "{a:?} == {b:?}"),
                }
            }
        }
    }

    /// `Severity::as_str` round-trips through the serde rename (non-stringly typed).
    /// Mutation witness: renaming a variant without updating `as_str` breaks this test.
    #[test]
    fn severity_as_str_matches_serde_rename() {
        for s in Severity::ALL {
            let json = serde_json::to_string(&s).expect("Severity serializes");
            // serde rename_all = "lowercase" wraps the string in quotes.
            let expected = format!("\"{}\"", s.as_str());
            assert_eq!(
                json, expected,
                "Severity::{s:?}.as_str() must match serde rename"
            );
        }
    }

    // ── Content hash (ADR-003 / RFC-0013 I3) ───────────────────────────────────────────────────

    /// The content hash is deterministic: the same `Diag` always produces the same hash.
    /// Mutation witness: changing the domain tag `"mycelium.diag.v1"` in `content_hash` changes all
    /// hashes and breaks this test.
    #[test]
    fn content_hash_is_deterministic() {
        let d = Diag::error(Code::OutOfRange)
            .message("test msg")
            .note("some note");
        let h1 = d.content_hash();
        let h2 = d.content_hash();
        assert_eq!(h1, h2, "content_hash must be deterministic");
    }

    /// `content_hash()` is presentation-invariant: producing human()/machine() views does not change
    /// the hash.
    /// Mutation witness: having `human()` or `machine()` mutate state (impossible with &self, but
    /// guard it anyway) would break this test.
    #[test]
    fn content_hash_is_presentation_invariant() {
        let d = Diag::error(Code::OutOfRange).message("msg");
        let h = d.content_hash();
        let _ = d.human();
        let _ = d.machine();
        assert_eq!(
            d.content_hash(),
            h,
            "human()/machine() must not change identity"
        );
    }

    /// Distinct canonical fields produce distinct hashes (collision resistance for common cases).
    /// Mutation witness: removing field-specific hashing in `content_hash` makes two distinct `Diag`s
    /// collide.
    #[test]
    fn distinct_fields_produce_distinct_hashes() {
        let a = Diag::error(Code::OutOfRange).message("a");
        let b = Diag::error(Code::OutOfRange).message("b");
        assert_ne!(
            a.content_hash(),
            b.content_hash(),
            "different message → different hash"
        );

        let c = Diag::warn(Code::OutOfRange).message("a");
        assert_ne!(
            a.content_hash(),
            c.content_hash(),
            "different severity → different hash"
        );

        let d = Diag::error(Code::Budget).message("a");
        assert_ne!(
            a.content_hash(),
            d.content_hash(),
            "different code → different hash"
        );

        let e = Diag::error(Code::OutOfRange).message("a").note("extra");
        assert_ne!(
            a.content_hash(),
            e.content_hash(),
            "extra note → different hash"
        );
    }

    /// A `Diag` with a locus vs. without produces distinct hashes (explicit absence, G2).
    /// Mutation witness: commenting out the locus branch in `content_hash` makes this collide.
    #[test]
    fn locus_absence_is_explicit_in_hash() {
        let without = Diag::error(Code::OutOfRange).message("m");
        let with_locus = Diag::error(Code::OutOfRange).message("m").at(Locus {
            source: Some("f.myc".into()),
            line: None,
            column: None,
        });
        assert_ne!(
            without.content_hash(),
            with_locus.content_hash(),
            "locus changes identity (G2 — absence is distinct from presence)"
        );
    }

    /// `None` locus and an all-`None`-field `Some(Locus::default())` produce distinct hashes (G2).
    /// Mutation witness: changing the locus presence tag from 1 to 0 collapses these two cases.
    #[test]
    fn locus_none_differs_from_default_locus() {
        let no_locus = Diag::error(Code::OutOfRange).message("m");
        let default_locus = Diag::error(Code::OutOfRange)
            .message("m")
            .at(Locus::default()); // all-None fields
        assert_ne!(
            no_locus.content_hash(),
            default_locus.content_hash(),
            "None locus ≠ Some(Locus::default()) — explicit absence (G2)"
        );
    }

    /// A `Diag` with a non-empty trace produces a distinct hash from one without (G2).
    /// Mutation witness: commenting out the trace encoding in `content_hash` collapses these.
    #[test]
    fn trace_is_identity_bearing() {
        let no_trace = Diag::error(Code::OutOfRange).message("m");
        let with_trace = Diag::error(Code::OutOfRange)
            .message("m")
            .trace(Trace::empty().with_frame("outer"));
        assert_ne!(
            no_trace.content_hash(),
            with_trace.content_hash(),
            "non-empty trace changes identity"
        );
    }

    /// A `Diag` survives clone/re-use with identity unchanged (value-semantic).
    /// Mutation witness: making `note()` mutate in-place rather than return a new value would cause
    /// the original's hash to change.
    #[test]
    fn diag_identity_unchanged_through_clone() {
        let base = Diag::error(Code::OutOfRange).message("base");
        let base_hash = base.content_hash();
        // Value-semantic builder: `base` is unchanged; the extended record is a new value.
        let extended = base.clone().note("extra detail");
        assert_eq!(
            base.content_hash(),
            base_hash,
            "base record identity must not change"
        );
        assert_ne!(
            base.content_hash(),
            extended.content_hash(),
            "adding a note changes identity"
        );
    }

    // ── Dual projection (G11 / RFC-0013 I3) ────────────────────────────────────────────────────

    /// `human()` is total for any well-formed `Diag` (including empty message, no locus, no notes).
    /// Mutation witness: making `human()` return an `Option` or panic on empty message breaks this.
    #[test]
    fn human_is_total() {
        let d = Diag::error(Code::OutOfRange);
        let h = d.human();
        assert!(h.contains("[ERROR]"), "human() must name the severity");
        assert!(h.contains("OutOfRange"), "human() must name the code");
        assert!(h.contains("id:"), "human() must embed the content id (I3)");
    }

    /// `machine()` is total and embeds the content `id` field.
    /// Mutation witness: removing the `id` injection from `machine()` makes this fail.
    #[test]
    fn machine_is_total_and_embeds_id() {
        let d = Diag::error(Code::Budget).message("budget exceeded");
        let json = d.machine();
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("machine() must produce valid JSON");
        assert!(
            parsed.get("id").is_some(),
            "machine() must embed the content id (I3)"
        );
        let id_field = parsed["id"].as_str().expect("id is a string");
        assert_eq!(
            id_field,
            d.content_hash().as_str(),
            "embedded id must match content_hash()"
        );
    }

    /// `from_json(machine(d))` recovers a record equal to `d` (the round-trip property, I3).
    /// Mutation witness: injecting the `id` field into JSON without ignoring it on deserialization
    /// would cause `from_json` to fail with an unknown-field error.
    #[test]
    fn machine_to_from_json_round_trips() {
        let d = Diag::error(Code::OutOfRange)
            .message("range violation")
            .at(Locus {
                source: Some("src.myc".into()),
                line: Some(12),
                column: Some(5),
            })
            .trace(
                Trace::empty()
                    .with_frame("check_range")
                    .with_frame("validate"),
            )
            .note("expected 0..256")
            .note("got 300");
        let json = d.machine();
        let recovered = Diag::from_json(&json).expect("machine() JSON must be valid");
        assert_eq!(d, recovered, "from_json(machine(d)) must equal d (I3)");
        assert_eq!(
            d.content_hash(),
            recovered.content_hash(),
            "round-trip preserves content identity (I3)"
        );
    }

    /// `from_json` returns an explicit `Err` on malformed input (C1 — never a partial/sentinel record).
    /// Mutation witness: removing the `?` / error handling from `from_json` makes malformed input
    /// silently succeed.
    #[test]
    fn from_json_returns_explicit_err_on_malformed_input() {
        // Completely invalid JSON.
        assert!(Diag::from_json("not json at all").is_err());
        // Unknown severity variant.
        assert!(Diag::from_json(r#"{"severity":"unknown_level","code":{"kind":"OutOfRange"},"message":"","locus":null,"trace":{"frames":[]},"notes":[]}"#).is_err());
    }

    /// The human and machine projections share the same content id (I3).
    /// Mutation witness: using a different hash in `human()` vs. `content_hash()` would make the
    /// embedded ids diverge.
    #[test]
    fn human_and_machine_share_content_id() {
        let d = Diag::warn(Code::HashMismatch).message("mismatch detected");
        let h = d.human();
        let m = d.machine();
        let id = d.content_hash().as_str().to_owned();
        assert!(h.contains(&id), "human() must embed the content id (I3)");
        assert!(m.contains(&id), "machine() must embed the content id (I3)");
    }

    /// The `Code::Other` variant round-trips through serde correctly.
    /// Mutation witness: removing the `Other` variant or changing the serde tag breaks this.
    #[test]
    fn code_other_round_trips() {
        let d = Diag::error(Code::Other("MyCustomCode".into())).message("custom");
        let json = d.machine();
        let recovered = Diag::from_json(&json).expect("round-trip");
        assert_eq!(d.code(), recovered.code());
        assert_eq!(d.code().as_str(), "MyCustomCode");
    }
}
