//! `std.fmt` — Ring-2 dual human/machine projection over one canonical form (M-533).
//!
//! # Summary
//!
//! `std.fmt` renders a [`Value`] into two views of **one canonical form**: a **human**
//! projection (`display`/`debug`) and a **machine** projection (`to_json`/`from_json`), exactly
//! as RFC-0013 §4.3 renders one diagnostic as "two renderers of one truth" (G11). A bounded
//! display (`display_bounded`) uses a [`Budget`] limit and returns a [`Rendering`] whose
//! [`Truncation`] record says what was elided — never a silent drop (C1/G2).
//!
//! # Honesty crux (spec §1 / §4.1)
//!
//! Two structural honesty facts:
//!
//! 1. **Display is a projection, not identity.** Formatting a borrowed `&Value` is a pure
//!    function that never mutates the value and never changes its content hash (ADR-003; C4).
//!    The human and machine views are two projections of *one* content-addressed canonical
//!    form; neither is "new truth".
//!
//! 2. **A truncated rendering says so.** `Truncation::Elided { omitted, marker }` is the
//!    never-silent guard made type-level: a bounded display that drops data *cannot* be
//!    constructed without the `omitted`/`marker` fields, so a silent truncation is
//!    unrepresentable rather than merely discouraged (C1/G2).
//!
//! # Guarantee matrix (RFC-0016 §4.5)
//!
//! Encoded as data in [`GUARANTEE_MATRIX`] and asserted in tests — never prose-only.
//!
//! # Contract conformance (RFC-0016 §4.1 C1–C6)
//!
//! - **C1 never-silent (G2):** `from_json` returns `Result` with explicit `Malformed` /
//!   `UnknownTag` / `OutOfDomain` variants; `display_bounded` returns `Truncation::Elided`
//!   (never a `Complete`-shaped struct when data was dropped).
//! - **C2 honest per-op tag (VR-5):** every op is `Exact` — `fmt` has no accuracy/precision
//!   semantics; the round-trip invariant is the one checked property, not a numeric bound.
//! - **C3 no black boxes / EXPLAIN (SC-3/G11):** `display_bounded` reifies *what* was elided
//!   in the inspectable [`Truncation`] record; other ops neither select, convert, nor
//!   approximate.
//! - **C4 content-addressed, value-semantic (ADR-003):** all ops are pure functions of a
//!   borrowed `&Value`; none mutates the value or its content hash.
//! - **C5 above the kernel (KC-3):** consumes `mycelium-core`; no `unsafe`, no new trusted
//!   code.
//! - **C6 declared bounded effects (RFC-0014):** pure ops declare `none`; `display_bounded`
//!   declares `alloc(budget)` — the bound is on the signature.
//!
//! # JSON delegation to `mycelium-std-io` (M-514) — wired (ratified 2026-06-19)
//!
//! `fmt.to_json`/`from_json` **delegate** to the **one canonical JSON projection** owned by
//! `io`/`serialize` (M-514) — one canonical JSON, two entry points (spec §7-Q1; `README.md §5`).
//! The maintainer ratified the converged delegation (2026-06-19), so the codec, the non-finite
//! refusal (`NaN`/`±∞` refused, never a silent `null`), and the never-silent decode-error
//! classification all live **once**, in `std.io`; this crate keeps only its thin display facade
//! (`Json`/`ToJsonError`/`FromJsonError`) over them. The round-trip invariant
//! (`from_json(to_json(v)).content_hash() == v.content_hash()`) is established once in `std.io`
//! and re-checked here.
//!
//! **Tag-framing note (honesty, VR-5 — RESOLVED 2026-06-19; DN-16, maintainer-ratified).** The two
//! `from_json` tags are **deliberately scope-distinct**, not a contradiction — each names a different
//! property of the shared op, and both are kept. `std.fmt` `from_json` = **`Exact`** claims *decode
//! determinism* (the same JSON text always decodes to the same `Value`, with no accuracy semantics —
//! an `Exact` structural property of the parse, RFC-0016 C2). `std.io` `from_json` = **`Empirical`**
//! claims *round-trip fidelity* (`from_json(to_json(v)) ≡ v`), established by a proptest corpus, not a
//! theorem (VR-5: no checked theorem ⇒ not `Proven`). Neither over-claims (`Proven`); the tags answer
//! different questions about the same call and are intentionally retained as-is. (Cross-ref: `std.io`
//! `guarantee_matrix.rs` `from_json` row.)
//!
//! Design spec: `docs/spec/stdlib/fmt.md`; contract: RFC-0016 §4.1 (C1–C6);
//! guarantee matrix: spec §4.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Format ops render to text; the source representation
//! is always named in the format receipt — `to_json` serializes the `Repr` tag as part of the
//! canonical JSON form (the `Value`'s `Repr` is an observable field, never omitted). A rendered
//! `Value` includes its `Repr`; an `EXPLAIN`-able format rendering is a first-class goal (G11).
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/fmt.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; the same-named `lib/std/fmt.myc` prototype is a narrower, structurally distinct surface (DN-66 S3.1) — the D6 retirement trigger has not fired, so no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

use mycelium_core::{GuaranteeStrength, Repr, Value};
use serde::{Deserialize, Serialize};

// ── Re-exports (convenience) ───────────────────────────────────────────────

pub use mycelium_core::{Payload, Trit};

// ── §1. Human projection types ────────────────────────────────────────────

/// A rendered text string (the output of a human projection).
///
/// A thin `String` wrapper so the type surface makes the projection explicit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text(pub String);

impl Text {
    /// Borrow the inner string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for Text {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── §2. Bounded display types ─────────────────────────────────────────────

/// A budget for `display_bounded`: the maximum number of *elements* (bits, trits, scalars,
/// or hypervector components) to render before eliding.
///
/// The budget is specified in element units so the contract is paradigm-uniform. A budget of
/// 0 is allowed and produces an immediately-elided rendering (all content elided).
///
/// Declared as the `alloc(budget)` effect in the C6 guarantee for `display_bounded` (spec §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Budget(pub usize);

/// Whether a [`Rendering`] is complete or whether some content was elided.
///
/// `Elided` is the never-silent guard made type-level (spec §3; C1/G2): a bounded display that
/// drops data **cannot** be constructed without the `omitted`/`marker` fields, so a silent
/// truncation is unrepresentable rather than merely discouraged. This is the EXPLAIN-able
/// artifact for `display_bounded` (C3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Truncation {
    /// The rendering is faithful and complete — nothing was elided.
    Complete,
    /// Some content was elided. The `omitted` count and the `marker` string are the
    /// reified record of *what* was dropped and *why* (C3 — inspectable EXPLAIN artifact).
    Elided {
        /// Number of elements omitted (always >= 1).
        omitted: usize,
        /// The elision marker embedded in the rendered text (e.g. `"...<N omitted>"`).
        /// This string is part of the rendered text, not a separate annotation, so it cannot
        /// be confused with real content.
        marker: String,
    },
}

/// The result of `display_bounded`: a rendered text paired with its truncation record.
///
/// When `truncation` is `Truncation::Elided`, the `text` field already contains the `marker`
/// verbatim so the output is self-describing without having to inspect the `Truncation` variant.
/// The `Truncation` variant is still the machine-readable EXPLAIN artifact (C3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendering {
    /// The bounded human-readable text (may contain the elision marker).
    pub text: Text,
    /// Whether and what was elided (the EXPLAIN-able artifact; C3).
    pub truncation: Truncation,
}

// ── §3. Machine projection types ──────────────────────────────────────────

/// The machine-projection JSON view of a [`Value`] (spec §3 / G11).
///
/// Produced by [`to_json`]; round-trips via [`from_json`]. This is **not** the canonical
/// transport codec (that is `io`/`serialize`, M-514) — it is the *display* machine projection.
///
/// # Delegation (M-514) — wired
/// `to_json`/`from_json` delegate the canonical projection to `mycelium-std-io` (the ratified
/// fmt→io seam; spec §7-Q1, README §5 — "one canonical JSON, two entry points"). This `Json` is
/// the thin display wrapper over that one canonical JSON; the round-trip guarantee
/// (`from_json(to_json(v)).content_hash() == v.content_hash()`) holds through the delegation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Json(pub serde_json::Value);

impl Json {
    /// Borrow the inner `serde_json::Value` for inspection.
    #[must_use]
    pub fn inner(&self) -> &serde_json::Value {
        &self.0
    }
}

/// Errors the `from_json` machine projection can raise.
///
/// Never-silent (C1): a malformed, unknown-tag, or out-of-domain input is an explicit `Err`,
/// never a coercion or a sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FromJsonError {
    /// The JSON structure does not match the expected `Value` wire schema (wrong shape, missing
    /// field, wrong type for a field). Carries a human-readable description of the span / cause.
    Malformed(String),
    /// The `repr.kind` tag is not one of `Binary|Ternary|Dense|VSA`. Carries the unknown name.
    UnknownTag(String),
    /// A field value is out of its stated domain (e.g. `width: 0`, negative dim, empty model).
    OutOfDomain(String),
}

impl core::fmt::Display for FromJsonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FromJsonError::Malformed(s) => write!(f, "malformed JSON value: {s}"),
            FromJsonError::UnknownTag(t) => write!(f, "unknown repr.kind tag: {t:?}"),
            FromJsonError::OutOfDomain(s) => write!(f, "field out of domain: {s}"),
        }
    }
}

mycelium_std_core::impl_std_error!(FromJsonError);

/// Error the `to_json` machine projection can raise.
///
/// Never-silent (C1): a `Value` that has no faithful JSON form is an explicit `Err`, never a
/// lossy coercion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToJsonError {
    /// A non-finite `f64` (`NaN`/`±∞`) in a `Dense`/`Vsa` payload has no JSON representation.
    ///
    /// `serde_json` would silently emit `null` (collapsing `NaN`/`±∞` together and breaking the
    /// round-trip), so it is refused (C1/G2). Carries the payload index of the first offender.
    NonFinite {
        /// Index of the first non-finite scalar in the payload.
        index: usize,
    },
}

impl core::fmt::Display for ToJsonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ToJsonError::NonFinite { index } => write!(
                f,
                "non-finite f64 at payload index {index} has no JSON form \
                 (refused — never a silent null)"
            ),
        }
    }
}

mycelium_std_core::impl_std_error!(ToJsonError);

/// The payload index of the first non-finite `f64`, if any (`Dense`/`Vsa` payloads only).
fn first_non_finite(v: &Value) -> Option<usize> {
    let scalars: &[f64] = match v.payload() {
        Payload::Scalars(s) | Payload::Hypervector(s) => s,
        Payload::Bits(_) | Payload::Trits(_) => return None,
        // A sequence (RFC-0032 D3) carries no flat f64 payload at this level; nested non-finite
        // scalars are caught by the recursive `mycelium-std-io` representability check, not here —
        // return None for the seq's own (absent) scalar payload. A byte string (RFC-0032 D4) has no
        // f64 either. A scalar float (ADR-040; M-896) is always JSON-representable — its wire form
        // is a *string* carrying the in-band specials faithfully — so it is never non-finite here.
        Payload::Seq(_) | Payload::Bytes(_) | Payload::Float(_) => return None,
    };
    scalars.iter().position(|x| !x.is_finite())
}

// ── §4. Human projection operations ───────────────────────────────────────

/// Render `v` as a human-readable string.
///
/// **Guarantee tag: `Exact`** (total, pure, no selection / approximation; C2).
/// **Fallibility: total** — every `Value` has a human display.
/// **Effects: none**.
/// **EXPLAIN: n/a** — no selection or approximation is hidden.
///
/// The output is a projection of a borrowed `&Value`; it never mutates `v` and never changes
/// its content hash (ADR-003; C4).
#[must_use]
pub fn display(v: &Value) -> Text {
    Text(format_value_human(v, false))
}

/// Render `v` as a structural debug string (more detailed than `display`).
///
/// **Guarantee tag: `Exact`** (total, pure; C2).
/// **Fallibility: total**.
/// **Effects: none**.
/// **EXPLAIN: n/a**.
///
/// Like `display` but includes representation metadata (repr kind, width/dim, guarantee
/// strength) so the output is useful for diagnostics. Still a projection, not identity.
#[must_use]
pub fn debug(v: &Value) -> Text {
    Text(format_value_human(v, true))
}

// ── §5. Bounded display operation ─────────────────────────────────────────

/// Render `v` within `limit` elements, emitting a typed `Truncation` record when content is
/// elided — **never a silent drop** (C1/G2).
///
/// **Guarantee tag: `Exact`** — the result is faithful to *what it claims to render*. An
/// `Elided` rendering does not assert completeness; it carries `{omitted, marker}` evidence.
/// **Fallibility: total** — always returns a `Rendering`, even for `limit = Budget(0)`.
/// **Effects: `alloc(budget)`** — the output size is capped at `limit.0` elements (C6).
/// **EXPLAIN: yes** — `Rendering::truncation` is the reified, inspectable artifact of *what*
/// was elided and *why* (C3).
#[must_use]
pub fn display_bounded(v: &Value, limit: Budget) -> Rendering {
    display_bounded_impl(v, limit)
}

// ── §6. Machine projection operations ─────────────────────────────────────

/// Project `v` to a machine-faithful JSON view (the `to_json` half of the dual projection, G11).
///
/// **Guarantee tag: `Exact`** (when `Ok`) — the machine view is a deterministic function of the
/// value's canonical form.
/// **Fallibility: `Err(ToJsonError::NonFinite)`** — a non-finite `f64` has no JSON form and is
/// refused rather than silently coerced to `null` (C1/G2). Every finite `Value` has a JSON view.
/// **Effects: none**.
/// **EXPLAIN: n/a** — no selection or approximation.
///
/// The JSON projection is the identity-preserving wire view: `from_json(to_json(v))` recovers a
/// `Value` with the **same content hash** as `v` (ADR-003; RFC-0001 §4.6). This is the checked
/// round-trip invariant (spec §4; RFC-0016 §4.5).
///
/// # Delegation (M-514) — wired
/// This op delegates the canonical Value→JSON projection (and the non-finite refusal) to
/// `mycelium-std-io::to_json` — the ratified fmt→io seam (spec §7-Q1; README §5). `fmt` reports
/// the first offending payload index as its typed `ToJsonError::NonFinite`.
pub fn to_json(v: &Value) -> Result<Json, ToJsonError> {
    Ok(Json(value_to_json(v)?))
}

/// Reconstruct a [`Value`] from its machine JSON view (the `from_json` half).
///
/// **Guarantee tag: `Exact`** — if the input is well-formed, the output is deterministic.
/// The round-trip property (`from_json(to_json(v))` has the same content hash as `v`) is
/// the one checked invariant of this module (spec §4; RFC-0001 §4.6 / ADR-003).
/// **Fallibility: `Err(Malformed | UnknownTag | OutOfDomain)`** — never a best-effort coercion
/// (C1).
/// **Effects: none**.
/// **EXPLAIN: n/a** — round-trip is the checked property, not a heuristic.
///
/// # Errors
///
/// - [`FromJsonError::Malformed`] — the JSON does not match the value wire schema.
/// - [`FromJsonError::UnknownTag`] — the `repr.kind` field is not `Binary|Ternary|Dense|VSA`.
/// - [`FromJsonError::OutOfDomain`] — a field value is out of its domain (e.g. `width: 0`).
///
/// # Delegation (M-514) — wired
/// Delegates the canonical decode (and its located, classified errors) to
/// `mycelium-std-io::from_json` — the ratified fmt→io seam (spec §7-Q1; README §5).
pub fn from_json(j: &Json) -> Result<Value, FromJsonError> {
    json_to_value(j.inner())
}

// ── §7. Guarantee matrix (RFC-0016 §4.5) — encoded as data, asserted in tests ──

/// One row of the `std.fmt` guarantee matrix (RFC-0016 §4.5; spec §4).
///
/// Encoded as data so tests can assert invariants rather than relying on prose only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation's name.
    pub op: &'static str,
    /// The honest guarantee tag on `Exact > Proven > Empirical > Declared`.
    pub tag: GuaranteeStrength,
    /// Whether the op is fallible (returns `Result`).
    pub fallible: bool,
    /// Whether this op surfaces an inspectable EXPLAIN artifact.
    pub explainable: bool,
    /// The declared effects (`"none"` or `"alloc(budget)"`).
    pub effects: &'static str,
}

/// The `std.fmt` guarantee matrix (spec §4 / RFC-0016 §4.5).
///
/// All five rows are `Exact` — `fmt` has no accuracy/precision semantics (C2). The one
/// substantive honest claim is the **round-trip invariant** (`from_json(to_json(v))` preserves
/// content hash), which is checked in the test suite, not merely stated here.
pub const GUARANTEE_MATRIX: &[MatrixRow] = &[
    MatrixRow {
        op: "display",
        // Exact: a faithful full render; no selection, no approximation (spec §4).
        tag: GuaranteeStrength::Exact,
        fallible: false,
        explainable: false,
        effects: "none",
    },
    MatrixRow {
        op: "debug",
        // Exact: a faithful structural render; no selection, no approximation.
        tag: GuaranteeStrength::Exact,
        fallible: false,
        explainable: false,
        effects: "none",
    },
    MatrixRow {
        op: "to_json",
        // Exact when Ok: the machine view of one canonical form, deterministic. Fallible: a
        // non-finite f64 has no JSON form and is refused (never a silent null) — C1/G2.
        tag: GuaranteeStrength::Exact,
        fallible: true,
        explainable: false,
        effects: "none",
    },
    MatrixRow {
        op: "from_json",
        // Exact: round-trip is the checked property; explicit error set on Err.
        tag: GuaranteeStrength::Exact,
        fallible: true,
        explainable: false,
        effects: "none",
    },
    MatrixRow {
        op: "display_bounded",
        // Exact: faithful to what it claims to render; Elided record is the EXPLAIN artifact.
        tag: GuaranteeStrength::Exact,
        fallible: false,
        explainable: true,
        effects: "alloc(budget)",
    },
];

/// Assert the structural invariants of the guarantee matrix — called from tests.
///
/// Discharges the RFC-0016 §4.5 obligation: "encoded as data, asserted in tests, never
/// prose-only." Panics with a descriptive message on any violation.
pub fn assert_matrix_invariants() {
    assert_eq!(
        GUARANTEE_MATRIX.len(),
        5,
        "spec §4 lists exactly 5 rows (display/debug/to_json/from_json/display_bounded)"
    );
    for row in GUARANTEE_MATRIX {
        assert!(
            !row.op.is_empty(),
            "every matrix row must have a non-empty op name"
        );
        assert_eq!(
            row.tag,
            GuaranteeStrength::Exact,
            "op '{}': every fmt row must be Exact (no accuracy semantics; C2 / spec §4)",
            row.op
        );
        // Only display_bounded has an EXPLAIN artifact and a budget effect.
        if row.op == "display_bounded" {
            assert!(row.explainable, "display_bounded must be EXPLAIN-able (C3)");
            assert_eq!(
                row.effects, "alloc(budget)",
                "display_bounded must declare alloc(budget) (C6)"
            );
        } else {
            assert!(
                !row.explainable,
                "op '{}': only display_bounded is EXPLAIN-able",
                row.op
            );
            assert_eq!(
                row.effects, "none",
                "op '{}': pure ops must declare 'none' effects (C6)",
                row.op
            );
        }
    }
    // The fallible ops are the two JSON ops: to_json (refuses non-finite f64 — never a silent
    // null) and from_json (explicit decode errors). The human projections are total.
    let fallible_ops: Vec<&str> = GUARANTEE_MATRIX
        .iter()
        .filter(|r| r.fallible)
        .map(|r| r.op)
        .collect();
    assert_eq!(
        fallible_ops,
        ["to_json", "from_json"],
        "the JSON ops (to_json, from_json) must be the fallible ones (spec §3 / C1)"
    );
}

// ── §8. Internal rendering helpers ────────────────────────────────────────

/// Render a value in human form (used by both `display` and `debug`).
fn format_value_human(v: &Value, detailed: bool) -> String {
    match v.repr() {
        Repr::Binary { width } => {
            let bits = match v.payload() {
                Payload::Bits(b) => b
                    .iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect::<String>(),
                _ => unreachable!("Binary value must have Bits payload"),
            };
            if detailed {
                format!("Binary<{width}>(0b{bits})")
            } else {
                format!("0b{bits}")
            }
        }
        Repr::Ternary { trits } => {
            let ts = match v.payload() {
                Payload::Trits(t) => t
                    .iter()
                    .map(|t| match t {
                        Trit::Neg => '-',
                        Trit::Zero => '0',
                        Trit::Pos => '+',
                    })
                    .collect::<String>(),
                _ => unreachable!("Ternary value must have Trits payload"),
            };
            if detailed {
                format!("Ternary<{trits}>({ts})")
            } else {
                format!("0t{ts}")
            }
        }
        Repr::Dense { dim, dtype } => {
            let xs = match v.payload() {
                Payload::Scalars(s) => s
                    .iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                _ => unreachable!("Dense value must have Scalars payload"),
            };
            if detailed {
                format!("Dense<{dim},{dtype:?}>([{xs}])")
            } else {
                format!("[{xs}]")
            }
        }
        Repr::Vsa { model, dim, .. } => {
            let xs = match v.payload() {
                Payload::Hypervector(h) => h
                    .iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                _ => unreachable!("Vsa value must have Hypervector payload"),
            };
            if detailed {
                format!("Vsa<{model},{dim}>([{xs}])")
            } else {
                format!("hv[{xs}]")
            }
        }
        Repr::Seq { elem, len } => {
            // RFC-0032 D3 (M-749): render each element recursively in human form, comma-joined.
            let xs = match v.payload() {
                Payload::Seq(elems) => elems
                    .iter()
                    .map(|e| format_value_human(e, detailed))
                    .collect::<Vec<_>>()
                    .join(", "),
                _ => unreachable!("Seq value must have Seq payload"),
            };
            if detailed {
                format!("Seq<{},{len}>([{xs}])", format_repr_short(elem))
            } else {
                format!("[{xs}]")
            }
        }
        Repr::Bytes => {
            // RFC-0032 D4 (M-750): render the bytes as a lowercase-hex string.
            let hex = match v.payload() {
                Payload::Bytes(b) => b.iter().map(|x| format!("{x:02x}")).collect::<String>(),
                _ => unreachable!("Bytes value must have Bytes payload"),
            };
            if detailed {
                format!("Bytes(0x{hex})")
            } else {
                format!("0x{hex}")
            }
        }
        Repr::Float { .. } => {
            // ADR-040 (M-896): shortest round-trip decimal (`{:?}`), so `-0.0` keeps its sign and
            // the in-band specials render as `inf`/`-inf`/`NaN` — deterministic, never lossy.
            let x = match v.payload() {
                Payload::Float(x) => *x,
                _ => unreachable!("Float value must have Float payload"),
            };
            if detailed {
                format!("Float<F64>({x:?})")
            } else {
                format!("{x:?}")
            }
        }
    }
}

/// A compact element-type label for a sequence's `Repr` (the head paradigm + width). Used only in
/// the *detailed* human rendering, so it is intentionally terse.
fn format_repr_short(r: &Repr) -> String {
    match r {
        Repr::Binary { width } => format!("Binary<{width}>"),
        Repr::Ternary { trits } => format!("Ternary<{trits}>"),
        Repr::Dense { dim, dtype } => format!("Dense<{dim},{dtype:?}>"),
        Repr::Vsa { model, dim, .. } => format!("Vsa<{model},{dim}>"),
        Repr::Seq { elem, len } => format!("Seq<{},{len}>", format_repr_short(elem)),
        Repr::Bytes => "Bytes".to_owned(),
        // ADR-040 (M-896): the frozen width registry has exactly F64 today; render it by name.
        Repr::Float { .. } => "Float<F64>".to_owned(),
    }
}

/// Implement `display_bounded` with the never-silent elision discipline (spec §3 / C1/G2).
fn display_bounded_impl(v: &Value, limit: Budget) -> Rendering {
    let Budget(max_elems) = limit;

    match v.repr() {
        Repr::Binary { .. } => {
            let bits = match v.payload() {
                Payload::Bits(b) => b,
                _ => unreachable!("Binary value must have Bits payload"),
            };
            let total = bits.len();
            let rendered_count = total.min(max_elems);
            let omitted = total - rendered_count;

            let rendered: String = bits[..rendered_count]
                .iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect();

            if omitted == 0 {
                Rendering {
                    text: Text(format!("0b{rendered}")),
                    truncation: Truncation::Complete,
                }
            } else {
                let marker = format!("...<{omitted} omitted>");
                Rendering {
                    text: Text(format!("0b{rendered}{marker}")),
                    truncation: Truncation::Elided { omitted, marker },
                }
            }
        }

        Repr::Ternary { .. } => {
            let trits = match v.payload() {
                Payload::Trits(t) => t,
                _ => unreachable!("Ternary value must have Trits payload"),
            };
            let total = trits.len();
            let rendered_count = total.min(max_elems);
            let omitted = total - rendered_count;

            let rendered: String = trits[..rendered_count]
                .iter()
                .map(|t| match t {
                    Trit::Neg => '-',
                    Trit::Zero => '0',
                    Trit::Pos => '+',
                })
                .collect();

            if omitted == 0 {
                Rendering {
                    text: Text(format!("0t{rendered}")),
                    truncation: Truncation::Complete,
                }
            } else {
                let marker = format!("...<{omitted} omitted>");
                Rendering {
                    text: Text(format!("0t{rendered}{marker}")),
                    truncation: Truncation::Elided { omitted, marker },
                }
            }
        }

        Repr::Dense { .. } => {
            let scalars = match v.payload() {
                Payload::Scalars(s) => s,
                _ => unreachable!("Dense value must have Scalars payload"),
            };
            let total = scalars.len();
            let rendered_count = total.min(max_elems);
            let omitted = total - rendered_count;

            let rendered: Vec<String> = scalars[..rendered_count]
                .iter()
                .map(|x| format!("{x}"))
                .collect();

            if omitted == 0 {
                Rendering {
                    text: Text(format!("[{}]", rendered.join(", "))),
                    truncation: Truncation::Complete,
                }
            } else {
                let marker = format!("...<{omitted} omitted>");
                let inner = if rendered.is_empty() {
                    marker.clone()
                } else {
                    format!("{}, {marker}", rendered.join(", "))
                };
                Rendering {
                    text: Text(format!("[{inner}]")),
                    truncation: Truncation::Elided { omitted, marker },
                }
            }
        }

        Repr::Vsa { .. } => {
            let hv = match v.payload() {
                Payload::Hypervector(h) => h,
                _ => unreachable!("Vsa value must have Hypervector payload"),
            };
            let total = hv.len();
            let rendered_count = total.min(max_elems);
            let omitted = total - rendered_count;

            let rendered: Vec<String> = hv[..rendered_count]
                .iter()
                .map(|x| format!("{x}"))
                .collect();

            if omitted == 0 {
                Rendering {
                    text: Text(format!("hv[{}]", rendered.join(", "))),
                    truncation: Truncation::Complete,
                }
            } else {
                let marker = format!("...<{omitted} omitted>");
                let inner = if rendered.is_empty() {
                    marker.clone()
                } else {
                    format!("{}, {marker}", rendered.join(", "))
                };
                Rendering {
                    text: Text(format!("hv[{inner}]")),
                    truncation: Truncation::Elided { omitted, marker },
                }
            }
        }

        Repr::Seq { .. } => {
            // RFC-0032 D3 (M-749): elide on the element *count*, same never-silent discipline as the
            // other paradigms. Each rendered element is shown in (non-detailed) human form.
            let elems = match v.payload() {
                Payload::Seq(e) => e,
                _ => unreachable!("Seq value must have Seq payload"),
            };
            let total = elems.len();
            let rendered_count = total.min(max_elems);
            let omitted = total - rendered_count;

            let rendered: Vec<String> = elems[..rendered_count]
                .iter()
                .map(|e| format_value_human(e, false))
                .collect();

            if omitted == 0 {
                Rendering {
                    text: Text(format!("[{}]", rendered.join(", "))),
                    truncation: Truncation::Complete,
                }
            } else {
                let marker = format!("...<{omitted} omitted>");
                let inner = if rendered.is_empty() {
                    marker.clone()
                } else {
                    format!("{}, {marker}", rendered.join(", "))
                };
                Rendering {
                    text: Text(format!("[{inner}]")),
                    truncation: Truncation::Elided { omitted, marker },
                }
            }
        }

        Repr::Bytes => {
            // RFC-0032 D4 (M-750): elide on the byte *count*, same never-silent discipline. Each
            // rendered byte is two lowercase-hex digits, prefixed `0x`.
            let bytes = match v.payload() {
                Payload::Bytes(b) => b,
                _ => unreachable!("Bytes value must have Bytes payload"),
            };
            let total = bytes.len();
            let rendered_count = total.min(max_elems);
            let omitted = total - rendered_count;

            let rendered: String = bytes[..rendered_count]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();

            if omitted == 0 {
                Rendering {
                    text: Text(format!("0x{rendered}")),
                    truncation: Truncation::Complete,
                }
            } else {
                let marker = format!("...<{omitted} omitted>");
                Rendering {
                    text: Text(format!("0x{rendered}{marker}")),
                    truncation: Truncation::Elided { omitted, marker },
                }
            }
        }

        Repr::Float { .. } => {
            // ADR-040 (M-896): a scalar float is one indivisible element — it always renders
            // complete (a budget can only elide *elements*, and there is exactly one; truncating
            // digits would be a silently-lossy rendering, G2).
            let x = match v.payload() {
                Payload::Float(x) => *x,
                _ => unreachable!("Float value must have Float payload"),
            };
            Rendering {
                text: Text(format!("{x:?}")),
                truncation: Truncation::Complete,
            }
        }
    }
}

// ── §9. JSON serialization helpers — delegated to `mycelium-std-io` (M-514) ──
//
// The canonical Value↔JSON projection, its non-finite refusal, and the never-silent, located
// decode-error classification all live ONCE in `mycelium-std-io` (M-514; `to_json`/`from_json`).
// Per the maintainer-ratified delegation (2026-06-19; spec §7-Q1, README §5 — "one canonical
// JSON, two entry points"), `std.fmt` no longer carries its own codec: it calls `std.io` and
// adapts the result to its thin, display-facing facade. The round-trip property
// (`from_json(to_json(v)).content_hash() == v.content_hash()`) is therefore established once, in
// `std.io` (where it is honestly tagged `Empirical`), and merely re-checked here.

/// Project a [`Value`] to a `serde_json::Value` for `fmt`'s display wrapper, delegating the
/// canonical projection to [`mycelium_std_io::to_json`].
///
/// Refuses a non-finite `f64` with `Err(ToJsonError::NonFinite { index })` — `fmt` reports the
/// first offending payload index for an ergonomic typed error; `std.io` independently refuses the
/// same domain (never a silent `null`), so the seam is consistent (C1/G2).
fn value_to_json(v: &Value) -> Result<serde_json::Value, ToJsonError> {
    if let Some(index) = first_non_finite(v) {
        return Err(ToJsonError::NonFinite { index });
    }
    // Delegate the canonical Value→JSON-text projection to std.io (M-104 grammar, owned by M-514).
    // The text is valid JSON, so re-parsing it into a serde_json::Value for the display wrapper is
    // total.
    let text = mycelium_std_io::to_json(v)
        .expect("io::to_json is total over finite Values (non-finite excluded above)");
    Ok(
        serde_json::from_str(&text)
            .expect("io::to_json emits valid JSON, so the re-parse is total"),
    )
}

/// Reconstruct a [`Value`] from `fmt`'s display JSON wrapper, delegating the canonical decode
/// (and its located, classified errors) to [`mycelium_std_io::from_json`].
///
/// Never-silent (C1): every failure is an explicit [`FromJsonError`] mapped from `std.io`'s
/// located [`mycelium_std_io::SerError`]; no partially-filled `Value` is ever returned.
fn json_to_value(j: &serde_json::Value) -> Result<Value, FromJsonError> {
    // Pre-check `repr.kind` before the full delegation. A `serde_json::Map` serialises its keys in
    // sorted order when the `preserve_order` feature is off (the default), so when this wrapper is
    // rendered via `serde_json::to_string(j)` below, `meta` precedes `repr` — and a missing-field
    // error in `meta` would surface in `std.io` before serde ever reaches `repr.kind`. Checking the
    // tag eagerly here preserves the pre-delegation error priority: unknown `repr.kind` →
    // `UnknownTag`, regardless of field order in the serialised text (C1 — never-silent, classified
    // error set, spec §3).
    if let Some(kind) = j
        .get("repr")
        .and_then(|r| r.get("kind"))
        .and_then(|k| k.as_str())
    {
        match kind {
            "Binary" | "Ternary" | "Dense" | "VSA" => {}
            other => return Err(FromJsonError::UnknownTag(other.to_owned())),
        }
    }
    // Render the display wrapper to canonical text, then delegate the decode to std.io.
    let text = serde_json::to_string(j).expect("a serde_json::Value always re-serializes to text");
    mycelium_std_io::from_json(&text).map_err(from_ser_error)
}

/// Map `std.io`'s located [`mycelium_std_io::SerError`] onto `fmt`'s display [`FromJsonError`],
/// preserving the never-silent error class (C1). The structured locus (byte offset / field path)
/// is folded into the human-readable description `fmt` carries.
///
/// Classification note: `std.io`'s `is_domain_error` heuristic catches `"missing field payload"`
/// as `OutOfDomain` (because "payload" is in its domain-keyword list). For `fmt`, a missing
/// required field is a structural/grammar failure (`Malformed`), not a value-model invariant
/// violation (`OutOfDomain`). We reclassify accordingly.
fn from_ser_error(e: mycelium_std_io::SerError) -> FromJsonError {
    use mycelium_std_io::SerError;
    match &e {
        SerError::UnknownTag { tag, .. } => FromJsonError::UnknownTag(tag.clone()),
        SerError::OutOfDomain { why, .. } => {
            // Reclassify "missing field" as Malformed: missing a required JSON field is a
            // structural grammar failure (C1 — wrong shape), not a domain invariant violation.
            // std.io's domain-keyword heuristic over-classifies these (e.g. "missing field
            // `payload`" → OutOfDomain) because "payload" is in its domain word list.
            if why.contains("missing field") {
                FromJsonError::Malformed(e.to_string())
            } else {
                FromJsonError::OutOfDomain(why.clone())
            }
        }
        SerError::Truncated { .. }
        | SerError::Malformed { .. }
        | SerError::BudgetExceeded { .. } => FromJsonError::Malformed(e.to_string()),
    }
}

// ── §10. Tests ─────────────────────────────────────────────────────────────
//
// White-box unit tests live in a dedicated in-crate module (`src/tests.rs`) per the
// "no tests in logic files" house rule (M-797 as-touched), not inline here.

#[cfg(test)]
mod tests;
