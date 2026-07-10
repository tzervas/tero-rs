//! Guarantee matrix for `std.testing` (M-534, #174) — RFC-0016 §4.5.
//!
//! Encoded as checked data + asserted in tests (never prose only — spec §4 / README §2).
//!
//! # Tag justification (VR-5 — downgrade rather than overclaim)
//! The harness ops are **`Exact` mechanisms**: a `Verdict` is an exact, deterministic function
//! of the run (seeded property → reproducible; golden compare → exact equality; differential →
//! exact equality of observables). This `Exact` is about the *verdict mechanism*, **not** a
//! claim about the subject under test.
//!
//! The harness **never inflates the subject's tag** (C2 crux): a passing `for_all` backs
//! `Empirical`, not `Proven`. There is no operation in this module that turns
//! "passed N trials" into `Proven` — that would be the VR-5 violation the module exists to
//! prevent (spec §4 tag justification).

use mycelium_core::GuaranteeStrength;

/// One row of the `std.testing` guarantee matrix.
///
/// Mirrors the `GuaranteeRow` shape from `mycelium-std-core` (spec §4 / README §2) — kept
/// local to avoid a circular crate dependency while `std.core` is not yet a dep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
    /// The exported op name.
    pub op: &'static str,
    /// Guarantee tag on the `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` lattice (RFC-0001 §4.3).
    pub tag: GuaranteeStrength,
    /// Explicit fallibility: `"total"` or the `Option`/`Result`/`Verdict` shape returned.
    pub fallibility: &'static str,
    /// Declared effects (`"none"` for pure ops; `"io (baseline read)"` for golden).
    pub effects: &'static str,
    /// Whether the op surfaces an EXPLAIN artifact (C3/SC-3/G11).
    pub explainable: bool,
}

/// The `std.testing` guarantee matrix (spec §4).
///
/// Every row is `Exact` (the verdict mechanism is exact) and either effect-free or declares
/// its IO effect explicitly (C6). EXPLAIN coverage is on every harness op that produces a
/// non-`Pass` verdict (C3).
///
/// | Op | Tag | Fallibility | Effects | EXPLAIN |
/// |---|---|---|---|---|
/// | `for_all` | Exact | `Fail{Diag}` / `Skipped{...}` as Verdict | none (pure; seeded) | yes |
/// | `golden` | Exact | `Fail{diff}` / `Skipped{NeedsRecord}` | io (baseline read) | yes |
/// | `differential` | Exact | `Fail{lhs,rhs}` / `Skipped{BackendUnavailable}` | none / io per backend | yes |
/// | `summarize` | Exact | total | none | yes |
/// | `is_green` | Exact | total | none | yes |
pub const MATRIX: &[Row] = &[
    Row {
        op: "for_all",
        tag: GuaranteeStrength::Exact,
        fallibility: "Verdict::Fail{record} / Verdict::Skipped{NeedsRecord}",
        effects: "none (pure; seeded — C6/RT3)",
        explainable: true, // counterexample + seed (C3/G11)
    },
    Row {
        op: "golden",
        tag: GuaranteeStrength::Exact,
        fallibility: "Verdict::Fail{diff} / Verdict::Skipped{NeedsRecord}",
        effects: "io (baseline read — declared, C6)",
        explainable: true, // diff + baseline hash (C3/G11)
    },
    Row {
        op: "differential",
        tag: GuaranteeStrength::Exact,
        fallibility: "Verdict::Fail{lhs,rhs} / Verdict::Skipped{BackendUnavailable}",
        effects: "none / io per backend (declared, C6)",
        explainable: true, // both outputs + input (C3/G11)
    },
    Row {
        op: "summarize",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: true, // per-class counts (C3)
    },
    Row {
        op: "is_green",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: true, // caller can inspect Summary for skip/undetermined counts (C3)
    },
];
