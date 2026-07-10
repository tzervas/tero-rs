//! `std.select` — Ring 1 / Tier A capability surface (M-519).
//!
//! The **ergonomic library surface** over [`mycelium_select`]: a total, non-learned,
//! content-addressed selection-policy DSL with a **mandatory EXPLAIN record** on every
//! selection.
//!
//! # Contract (RFC-0016 §4.1, C1–C6)
//!
//! - **C1 never-silent (G2):** every fallible op returns an explicit `Result`; an empty
//!   candidate set, a non-total policy, an override outside the candidate set, and a wrong-kind
//!   candidate at a site adapter are all explicit errors — never a silent default or clamp.
//! - **C2 honest per-op tag (VR-5):** all ops are `Exact` — the policy is a total predicate
//!   over *exact* metadata; nothing is probabilistic, learned, or estimated. The `Exact` tag
//!   covers the *selection decision*, not downstream op accuracy.
//! - **C3 no black boxes / EXPLAIN (SC-3/G11):** this is the module's reason to exist.
//!   [`select`] and [`select_with_override`] **always** emit a valid [`Explanation`]; there is
//!   no code path that returns a choice without one. [`explain`] *is* the artifact.
//! - **C4 content-addressed, value-semantic (ADR-003):** a [`SelectionPolicy`] is an immutable
//!   value; its identity is its [`PolicyRef`] content hash. [`select`] and [`explain`] are pure
//!   functions of their inputs.
//! - **C5 above the kernel (KC-3):** this crate re-exports and wraps `mycelium-select`; it adds
//!   no trusted code and uses no `unsafe`/FFI.
//! - **C6 declared, bounded effects (RFC-0014):** all ops are pure — no IO, time, randomness,
//!   or unbounded allocation. The policy language is not Turing-complete; selection terminates.
//!
//! # Guarantee matrix (RFC-0016 §4.5)
//!
//! Encoded as data in [`GUARANTEE_MATRIX`] and asserted in the test suite.
//!
//! | Op | Tag | Fallibility | Effects | EXPLAIN-able |
//! |---|---|---|---|---|
//! | [`build`] | `Exact` | `Err(PolicyError)` — non-total / malformed table | none | n/a |
//! | [`policy_ref`] | `Exact` | total | none | n/a |
//! | [`select`] | `Exact` | `Err(SelectError)` — empty set, wrong kind | none | **yes** |
//! | [`explain`] | `Exact` | total over valid policy | none | **yes** |
//! | [`select_with_override`] | `Exact` | `Err(SelectError)` — out-of-range forced index | none | **yes** |
//!
//! # Design notes
//!
//! - Signatures follow the spec sketch in `docs/spec/stdlib/select.md §3`; exact field names of
//!   [`Explanation`] are owned by the landed `mycelium-select` crate (M-221) and re-exported
//!   here without fabrication (resolves spec open question Q1).
//! - Cost units are storage **bits** as declared (M-220; RFC-0005 §2) — no "arbitrary internal
//!   units". Q2 (multi-unit cost) is out of scope for v0.
//! - EXPLAIN ergonomics: every call site returns `(choice, explanation)` — the mandatory-EXPLAIN
//!   posture (C3 / SC-3). The §8-Q3 implicit-but-inspectable direction (M-540) is a follow-on
//!   per-ring design pass; v0 makes the record explicit at every call.
//! - Policy composition (Q4) is deferred to v1 — no first-class composition op here.
//!
//! # FLAG: Q3 ergonomics-vs-contract
//!
//! Per spec §7-Q3 / RFC-0016 §8-Q3: whether the default return is `(choice, explanation)` or
//! `choice` with `explanation` on demand is deferred to the M-540 per-ring design pass. This
//! v0 surface always returns `(choice, explanation)` — the mandatory-EXPLAIN posture (C3). A
//! future M-540 pass may add an ergonomic `select_choice`-only wrapper, but that cannot suppress
//! the EXPLAIN record's *existence* — only its return position. Tracked: spec §7-Q3.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A):** Selection records the chosen packing and representation
//! in the mandatory [`Explanation`] artifact (the `meta.physical` of the selected candidate,
//! EXPLAIN-able via `explain`); there is no code path that returns a layout decision without
//! an accompanying record. The representation choice is never a silent internal layout decision.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/select.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

// Re-export the full landed kernel surface so callers need only this crate.
pub use mycelium_select::{
    Action, Candidate, CandidateCost, CostModel, DecodeFacts, DecodeMethod, Explanation,
    ParadigmKind, PolicyError, PolicyRegistry, Predicate, Rule, SelectError, SelectionInputs,
    SelectionPolicy,
};

// Re-export the site adapters that wire one mechanism to each RFC-0005 §4 site.
pub use mycelium_select::{
    bitnet_packing_policy, layout_of, record_packing_layout, select_decode_method, select_layout,
    select_packing, select_swap_target, BITNET_PACKINGS,
};

// Re-export core types that appear in the public API.
pub use mycelium_core::{ContentHash, Meta, PackScheme, PhysicalLayout, Provenance, Repr, Value};

/// A **content hash** that identifies a [`SelectionPolicy`] — recorded in `Meta.policy_used` so
/// "which policy chose this?" is always answerable (RFC-0005 §3; ADR-003).
///
/// This is a type alias for [`mycelium_core::ContentHash`] for use at selection sites.
pub type PolicyRef = ContentHash;

// ──────────────────────────────────────────────────────────────────────────────────────────────
// §3 Exported-op surface
// ──────────────────────────────────────────────────────────────────────────────────────────────

/// Build and validate a [`SelectionPolicy`] from a name, candidates, rules, a default arm, and
/// a cost model (C1).
///
/// **Guarantee: `Exact`** — either the policy is accepted (total, well-formed) or it is refused
/// with an explicit [`PolicyError`]; there is no silent completion of a non-total table.
///
/// # Errors
///
/// - [`PolicyError::NoCandidates`] — the candidate set is empty.
/// - [`PolicyError::IndexOutOfRange`] — a rule's `Choose(i)` or the default arm points outside
///   the candidate list.
/// - [`PolicyError::BadCost`] — `cost.storage_weight` is non-finite or ≤ 0.
/// - [`PolicyError::BadPredicateLiteral`] — a predicate carries a non-finite `f64` literal
///   (would collide distinct policies under content addressing — A5-01).
///
/// # Examples
///
/// ```rust
/// use mycelium_std_select::{
///     build, Action, Candidate, CostModel, Predicate, Rule,
/// };
/// use mycelium_core::{Repr, ScalarKind};
///
/// let policy = build(
///     "example.v1",
///     vec![Candidate::Repr(Repr::Dense { dim: 128, dtype: ScalarKind::F32 })],
///     vec![Rule { when: Predicate::Always, action: Action::Cheapest }],
///     0,
///     CostModel { storage_weight: 1.0 },
/// ).expect("well-formed policy");
///
/// assert_eq!(policy.name(), "example.v1");
/// ```
pub fn build(
    name: impl Into<String>,
    candidates: Vec<Candidate>,
    rules: Vec<Rule>,
    default_choice: usize,
    cost: CostModel,
) -> Result<SelectionPolicy, PolicyError> {
    SelectionPolicy::new(name, candidates, rules, default_choice, cost)
}

/// Return the content address of a [`SelectionPolicy`] — its [`PolicyRef`] (RFC-0005 §3).
///
/// **Guarantee: `Exact`** — total; a valid policy always has a deterministic `PolicyRef`.
/// The hash is over the policy's canonical JSON serialization (RFC-0001 §4.6; ADR-003).
#[must_use]
pub fn policy_ref(p: &SelectionPolicy) -> PolicyRef {
    p.policy_ref()
}

/// The **one selection mechanism** (RFC-0005 §4; C3): evaluate the decision table and return
/// the chosen candidate **with its mandatory [`Explanation`]** — there is no code path that
/// returns a choice without one.
///
/// **Guarantee: `Exact`** — the policy is a total predicate over *exact* metadata (RFC-0005
/// §2.5): same `(policy, inputs)` → same `(choice, explanation)`, deterministically.
/// The `Exact` tag covers the *selection decision*, not the accuracy of the downstream op.
///
/// # Errors
///
/// - [`SelectError::WrongSiteKind`] — the chosen candidate does not match the expected kind at
///   a typed site adapter. Use [`select_swap_target`], [`select_packing`], or
///   [`select_decode_method`] to enforce kind at the type level.
///
/// # EXPLAIN guarantee (C3)
///
/// The returned [`Explanation`] contains:
/// - `inputs` — the exact [`SelectionInputs`] that were considered.
/// - `costs` — every candidate's cost in declared storage **bits** (RFC-0005 §2; M-220).
/// - `matched_rule` — the rule index that fired, or `None` when the default arm decided.
/// - `chosen_index` / `chosen` — the selected candidate.
/// - `overridden` — `false` on this path (see [`select_with_override`]).
/// - `policy` / `policy_name` — the content address and name of the deciding policy.
///
/// # Examples
///
/// ```rust
/// use mycelium_std_select::{build, select, Action, Candidate, CostModel, Predicate, Provenance, Rule};
/// use mycelium_core::{Repr, ScalarKind, Meta};
/// use mycelium_select::SelectionInputs;
///
/// let policy = build(
///     "trivial.v1",
///     vec![Candidate::Repr(Repr::Dense { dim: 64, dtype: ScalarKind::F32 })],
///     vec![Rule { when: Predicate::Always, action: Action::Cheapest }],
///     0,
///     CostModel { storage_weight: 1.0 },
/// ).unwrap();
///
/// let meta = Meta::exact(Provenance::Root);
/// let inputs = SelectionInputs::from_meta(Repr::Binary { width: 64 }, &meta);
///
/// let (choice, explanation) = select(&policy, &inputs).unwrap();
/// assert!(!explanation.overridden);
/// assert_eq!(explanation.chosen, choice);
/// ```
pub fn select(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
) -> Result<(Candidate, Explanation), SelectError> {
    mycelium_select::select(policy, inputs, None)
}

/// The **explain capability** (RFC-0005 §4): derive the mandatory [`Explanation`] for a
/// `(policy, inputs)` pair without performing a selection — **total and deterministic** over
/// any valid [`SelectionPolicy`] (M-221).
///
/// **Guarantee: `Exact`** — total; an un-overridden selection on a validated policy cannot
/// fail. The returned [`Explanation`] is re-derivable from `(policy, inputs)` alone — it
/// carries the content address of the deciding policy so the record is self-describing.
///
/// # EXPLAIN guarantee (C3)
///
/// The `Explanation` *is* the artifact: it reveals which rule matched, per-candidate costs in
/// declared bits, the chosen option, and the `PolicyRef` — answering *"why this choice?"* from
/// the policy alone.
///
/// # Examples
///
/// ```rust
/// use mycelium_std_select::{build, explain, policy_ref, Action, Candidate, CostModel, Predicate, Provenance, Rule};
/// use mycelium_core::{Repr, ScalarKind, Meta};
/// use mycelium_select::SelectionInputs;
///
/// let policy = build(
///     "explain-demo.v1",
///     vec![Candidate::Repr(Repr::Dense { dim: 32, dtype: ScalarKind::F16 })],
///     vec![Rule { when: Predicate::Always, action: Action::Cheapest }],
///     0,
///     CostModel { storage_weight: 1.0 },
/// ).unwrap();
///
/// let meta = Meta::exact(Provenance::Root);
/// let inputs = SelectionInputs::from_meta(Repr::Binary { width: 32 }, &meta);
///
/// let explanation = explain(&policy, &inputs);
/// // The explanation's policy field matches the policy's content address.
/// assert_eq!(explanation.policy, policy_ref(&policy));
/// ```
#[must_use]
pub fn explain(policy: &SelectionPolicy, inputs: &SelectionInputs) -> Explanation {
    mycelium_select::explain(policy, inputs)
}

/// A **first-class deterministic override**: force a specific candidate by index and record
/// the override state *in* the [`Explanation`] — the overridden selection remains fully
/// inspectable (M-221; RFC-0005 §2.4).
///
/// **Guarantee: `Exact`** — the forced choice is deterministic and the `Explanation` marks
/// `overridden: true`, so no selection is ever silently coerced (C1/C3).
///
/// # Errors
///
/// - [`SelectError::OverrideOutOfRange`] — `forced_index` is outside the candidate list; this
///   is an explicit refusal, never a snap to the nearest legal choice (C1).
/// - [`SelectError::WrongSiteKind`] — if the forced candidate does not fit the call site; use
///   the typed site adapters where kind enforcement matters.
///
/// # EXPLAIN guarantee (C3)
///
/// `explanation.overridden == true` on every successful path — the override state is recorded,
/// never hidden.
///
/// # Examples
///
/// ```rust
/// use mycelium_std_select::{
///     build, select_with_override, Action, Candidate, CostModel, Predicate, Provenance, Rule,
/// };
/// use mycelium_core::{Repr, ScalarKind, Meta};
/// use mycelium_select::SelectionInputs;
///
/// let policy = build(
///     "override-demo.v1",
///     vec![
///         Candidate::Repr(Repr::Binary { width: 8 }),
///         Candidate::Repr(Repr::Dense { dim: 8, dtype: ScalarKind::F32 }),
///     ],
///     vec![Rule { when: Predicate::Always, action: Action::Choose(0) }],
///     0,
///     CostModel { storage_weight: 1.0 },
/// ).unwrap();
///
/// let meta = Meta::exact(Provenance::Root);
/// let inputs = SelectionInputs::from_meta(Repr::Binary { width: 8 }, &meta);
///
/// // Force the second candidate even though the rule would choose index 0.
/// let (choice, explanation) = select_with_override(&policy, &inputs, 1).unwrap();
/// assert!(explanation.overridden);
/// assert_eq!(explanation.chosen_index, 1);
/// let _ = choice;
/// ```
pub fn select_with_override(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced_index: usize,
) -> Result<(Candidate, Explanation), SelectError> {
    mycelium_select::select(policy, inputs, Some(forced_index))
}

// ──────────────────────────────────────────────────────────────────────────────────────────────
// Guarantee matrix (RFC-0016 §4.5) — encoded as data, asserted in tests.
// ──────────────────────────────────────────────────────────────────────────────────────────────

/// One row of the guarantee matrix (RFC-0016 §4.5; spec §4).
///
/// The matrix is **data** — asserted in tests — not prose only. Every exported selection op has
/// a row; the `tag` column is the honest per-op guarantee (VR-5); `explain_able` is the
/// module's honesty crux (C3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// The exported op name.
    pub op: &'static str,
    /// The honest guarantee tag (`Exact ⊐ Proven ⊐ Empirical ⊐ Declared`; VR-5).
    pub tag: GuaranteeTag,
    /// Whether the op is fallible (returns `Result`).
    pub fallible: bool,
    /// Declared effects (none for this pure module — C6).
    pub effects: &'static str,
    /// Whether the op emits a valid, inspectable `Explanation` — the C3 crux.
    pub explain_able: ExplainAble,
}

/// The honest guarantee tag — C2 / VR-5.
///
/// `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.
///
/// All ops in `std.select` are `Exact` — the policy is a total predicate over *exact* metadata
/// (RFC-0005 §2.5). Nothing is probabilistic, learned, or estimated; `Exact` is the honest tag,
/// not an overclaim (spec §4, "Tag justification"). `Proven`/`Empirical`/`Declared` are present
/// for the complete lattice; they are not used by any op in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuaranteeTag {
    /// Deterministic and lossless over exact inputs.
    Exact,
    /// Proven by a theorem whose side-conditions are checked.
    /// Not used in this module — present for lattice completeness.
    #[allow(dead_code)]
    Proven,
    /// Empirically measured.
    /// Not used in this module — present for lattice completeness.
    #[allow(dead_code)]
    Empirical,
    /// Declared but not machine-checked.
    /// Not used in this module — present for lattice completeness.
    #[allow(dead_code)]
    Declared,
}

/// Whether an op emits a valid, inspectable `Explanation` (the C3 / SC-3 crux).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainAble {
    /// Yes — the op emits a valid `Explanation` on every successful path.
    Yes,
    /// N/a — the op constructs or hashes a policy; it is not itself a selection.
    NotApplicable,
}

/// The loaded guarantee matrix — 5 rows, all `Exact`, EXPLAIN-able = yes for every selection op.
///
/// Asserted in unit tests; never prose only (RFC-0016 §4.5).
pub static GUARANTEE_MATRIX: &[GuaranteeRow] = &[
    GuaranteeRow {
        op: "build",
        tag: GuaranteeTag::Exact,
        fallible: true,
        effects: "none",
        explain_able: ExplainAble::NotApplicable,
    },
    GuaranteeRow {
        op: "policy_ref",
        tag: GuaranteeTag::Exact,
        fallible: false,
        effects: "none",
        explain_able: ExplainAble::NotApplicable,
    },
    GuaranteeRow {
        op: "select",
        tag: GuaranteeTag::Exact,
        fallible: true,
        effects: "none",
        explain_able: ExplainAble::Yes,
    },
    GuaranteeRow {
        op: "explain",
        tag: GuaranteeTag::Exact,
        fallible: false,
        effects: "none",
        explain_able: ExplainAble::Yes,
    },
    GuaranteeRow {
        op: "select_with_override",
        tag: GuaranteeTag::Exact,
        fallible: true,
        effects: "none",
        explain_able: ExplainAble::Yes,
    },
];

// ──────────────────────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use mycelium_core::{Meta, Provenance, Repr, ScalarKind};
    use mycelium_select::SelectionInputs;

    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────────────────────

    fn exact_meta() -> Meta {
        Meta::exact(Provenance::Root)
    }

    fn binary_inputs(width: u32) -> SelectionInputs {
        SelectionInputs::from_meta(Repr::Binary { width }, &exact_meta())
    }

    fn dense_inputs(dim: u32) -> SelectionInputs {
        SelectionInputs::from_meta(
            Repr::Dense {
                dim,
                dtype: ScalarKind::F32,
            },
            &exact_meta(),
        )
    }

    fn one_repr_policy(name: &str, repr: Repr) -> SelectionPolicy {
        build(
            name,
            vec![Candidate::Repr(repr)],
            vec![Rule {
                when: Predicate::Always,
                action: Action::Cheapest,
            }],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .expect("well-formed policy")
    }

    fn two_repr_policy() -> SelectionPolicy {
        build(
            "two-repr.v1",
            vec![
                Candidate::Repr(Repr::Binary { width: 8 }),
                Candidate::Repr(Repr::Dense {
                    dim: 8,
                    dtype: ScalarKind::F32,
                }),
            ],
            vec![Rule {
                when: Predicate::Always,
                action: Action::Cheapest,
            }],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .expect("well-formed two-repr policy")
    }

    // ── guarantee matrix (RFC-0016 §4.5) ─────────────────────────────────────────────────────

    /// GUARANTEE_MATRIX has exactly 5 rows, covering every exported op.
    #[test]
    fn matrix_has_five_rows() {
        // Mutation witness: removing one row from GUARANTEE_MATRIX breaks this assertion.
        assert_eq!(
            GUARANTEE_MATRIX.len(),
            5,
            "matrix must have exactly 5 rows (one per exported op)"
        );
    }

    /// Every matrix row carries the `Exact` tag — the honest tag for a total predicate over
    /// exact metadata (spec §4 "Tag justification"; VR-5).
    #[test]
    fn matrix_all_rows_exact() {
        for row in GUARANTEE_MATRIX {
            // Mutation witness: changing any tag to non-Exact breaks this.
            assert_eq!(
                row.tag,
                GuaranteeTag::Exact,
                "op `{}` must be tagged Exact (VR-5 — total predicate over exact metadata)",
                row.op
            );
        }
    }

    /// Every selection op in the matrix is EXPLAIN-able = yes (C3 crux).
    #[test]
    fn matrix_selection_ops_explain_able() {
        let selection_ops = ["select", "explain", "select_with_override"];
        for op_name in selection_ops {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == op_name)
                .unwrap_or_else(|| panic!("matrix missing row for op `{op_name}`"));
            // Mutation witness: changing explain_able to NotApplicable breaks this.
            assert_eq!(
                row.explain_able,
                ExplainAble::Yes,
                "op `{op_name}` must be EXPLAIN-able = yes (C3 crux)"
            );
        }
    }

    /// Non-selection ops (`build`, `policy_ref`) are not themselves selections.
    #[test]
    fn matrix_non_selection_ops_not_applicable() {
        let non_selection_ops = ["build", "policy_ref"];
        for op_name in non_selection_ops {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == op_name)
                .unwrap_or_else(|| panic!("matrix missing row for op `{op_name}`"));
            assert_eq!(
                row.explain_able,
                ExplainAble::NotApplicable,
                "op `{op_name}` is not a selection; explain_able must be NotApplicable"
            );
        }
    }

    /// Every op in the matrix declares no effects (C6 — pure functions).
    #[test]
    fn matrix_all_ops_no_effects() {
        for row in GUARANTEE_MATRIX {
            // Mutation witness: changing effects to non-"none" breaks this.
            assert_eq!(
                row.effects, "none",
                "op `{}` must declare no effects (C6 — pure)",
                row.op
            );
        }
    }

    /// Fallibility matches spec §4: `build`, `select`, `select_with_override` are fallible;
    /// `policy_ref` and `explain` are total.
    #[test]
    fn matrix_fallibility_matches_spec() {
        let fallible_ops = ["build", "select", "select_with_override"];
        let total_ops = ["policy_ref", "explain"];
        for op in fallible_ops {
            let row = GUARANTEE_MATRIX.iter().find(|r| r.op == op).unwrap();
            assert!(row.fallible, "op `{op}` should be fallible per spec §4");
        }
        for op in total_ops {
            let row = GUARANTEE_MATRIX.iter().find(|r| r.op == op).unwrap();
            assert!(
                !row.fallible,
                "op `{op}` should be total (non-fallible) per spec §4"
            );
        }
    }

    // ── C1 never-silent: build refusals ──────────────────────────────────────────────────────

    /// `build` refuses an empty candidate set with the exact `NoCandidates` variant (C1).
    #[test]
    fn build_refuses_empty_candidates() {
        let err = build(
            "empty",
            vec![],
            vec![],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .expect_err("empty candidate set must be refused");
        // Mutation witness: removing the NoCandidates check lets this through.
        assert_eq!(
            err,
            PolicyError::NoCandidates,
            "must be NoCandidates, not a silent default"
        );
    }

    /// `build` refuses a rule whose `Choose(i)` is out of range (C1 — never a silent clamp).
    #[test]
    fn build_refuses_out_of_range_index() {
        let err = build(
            "bad-index",
            vec![Candidate::Repr(Repr::Binary { width: 1 })],
            vec![Rule {
                when: Predicate::Always,
                action: Action::Choose(99), // out of range
            }],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .expect_err("out-of-range Choose index must be refused");
        // Mutation witness: removing the index range check lets this through.
        assert!(
            matches!(err, PolicyError::IndexOutOfRange { index: 99 }),
            "must be IndexOutOfRange {{index: 99}}, got {err:?}"
        );
    }

    /// `build` refuses a non-positive cost weight (C1 — never a silent clamp).
    #[test]
    fn build_refuses_bad_cost() {
        let err = build(
            "bad-cost",
            vec![Candidate::Repr(Repr::Binary { width: 1 })],
            vec![],
            0,
            CostModel {
                storage_weight: -1.0,
            },
        )
        .expect_err("non-positive cost weight must be refused");
        // Mutation witness: removing the cost guard lets negative weights through.
        assert_eq!(err, PolicyError::BadCost, "must be BadCost, not silent");
    }

    /// `build` refuses a non-finite float literal in a predicate (A5-01 content-addressing
    /// safety — two policies with NaN/∞ literals would collide on the same PolicyRef).
    #[test]
    fn build_refuses_non_finite_predicate_literal() {
        let err = build(
            "nan-pred",
            vec![Candidate::Repr(Repr::Binary { width: 1 })],
            vec![Rule {
                when: Predicate::ErrorEpsAtMost(f64::NAN),
                action: Action::Choose(0),
            }],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .expect_err("non-finite predicate literal must be refused");
        // Mutation witness: removing the literals_finite check admits NaN.
        assert_eq!(
            err,
            PolicyError::BadPredicateLiteral,
            "must be BadPredicateLiteral (A5-01 — NaN collapses content identity)"
        );
    }

    // ── C3 every selection emits a valid Explanation ──────────────────────────────────────────

    /// `select` always returns an `Explanation`; its `policy` field matches the policy's
    /// `policy_ref` (C3 + C4 — content-addressed, self-describing).
    #[test]
    fn select_emits_explanation_with_correct_policy_ref() {
        let policy = one_repr_policy("explain-check.v1", Repr::Binary { width: 32 });
        let inputs = binary_inputs(32);
        let (_, explanation) = select(&policy, &inputs).expect("select must succeed");
        // Mutation witness: stripping the Explanation from the return type breaks this.
        assert_eq!(
            explanation.policy,
            policy_ref(&policy),
            "explanation.policy must match policy_ref (C4 — content-addressed)"
        );
    }

    /// `select` emits `overridden: false` on a normal (non-overridden) selection.
    #[test]
    fn select_emits_not_overridden() {
        let policy = one_repr_policy("no-override.v1", Repr::Binary { width: 16 });
        let inputs = binary_inputs(16);
        let (_, explanation) = select(&policy, &inputs).unwrap();
        // Mutation witness: hardcoding overridden=true would break this.
        assert!(
            !explanation.overridden,
            "normal select must not set overridden=true (C3 — override state recorded faithfully)"
        );
    }

    /// `explain` always returns a valid Explanation; the policy field matches `policy_ref` (C3).
    #[test]
    fn explain_emits_explanation_with_correct_policy_ref() {
        let policy = one_repr_policy(
            "explain-op.v1",
            Repr::Dense {
                dim: 64,
                dtype: ScalarKind::F16,
            },
        );
        let inputs = dense_inputs(64);
        let explanation = explain(&policy, &inputs);
        // Mutation witness: returning a zeroed Explanation breaks the policy-ref check.
        assert_eq!(
            explanation.policy,
            policy_ref(&policy),
            "explain.policy must match policy_ref (C3 + C4)"
        );
    }

    /// `select_with_override` records `overridden: true` in the Explanation — the override
    /// state is never hidden (C3; RFC-0005 §2.4).
    #[test]
    fn select_with_override_records_override_in_explanation() {
        let policy = two_repr_policy();
        let inputs = binary_inputs(8);
        // Force the second candidate (index 1) even though Cheapest would choose index 0.
        let (_, explanation) = select_with_override(&policy, &inputs, 1).unwrap();
        // Mutation witness: not setting overridden=true would hide the forced choice.
        assert!(
            explanation.overridden,
            "select_with_override must set overridden=true (C3 — override state recorded)"
        );
        assert_eq!(
            explanation.chosen_index, 1,
            "forced index must be the chosen index"
        );
    }

    /// `select_with_override` refuses an out-of-range forced index with `OverrideOutOfRange`
    /// — never snaps to the nearest legal choice (C1).
    #[test]
    fn select_with_override_refuses_out_of_range() {
        let policy = one_repr_policy("override-range.v1", Repr::Binary { width: 4 });
        let inputs = binary_inputs(4);
        let err = select_with_override(&policy, &inputs, 99)
            .expect_err("out-of-range override must be refused");
        // Mutation witness: silently clamping to the last candidate would pass a wrong index.
        assert!(
            matches!(
                err,
                SelectError::OverrideOutOfRange {
                    index: 99,
                    candidates: 1
                }
            ),
            "must be OverrideOutOfRange {{index: 99, candidates: 1}}, got {err:?}"
        );
    }

    // ── C2 + determinism ─────────────────────────────────────────────────────────────────────

    /// `select` is deterministic: same `(policy, inputs)` → same `(choice, explanation)`.
    /// Exhaustive over a small candidate set.
    #[test]
    fn select_is_deterministic() {
        let policy = two_repr_policy();
        let inputs = binary_inputs(8);
        // Run twice and assert equality — one deterministic pair of calls.
        let (choice_a, expl_a) = select(&policy, &inputs).unwrap();
        let (choice_b, expl_b) = select(&policy, &inputs).unwrap();
        // Mutation witness: introducing any nondeterminism (e.g. HashMap ordering) breaks this.
        assert_eq!(choice_a, choice_b, "select must be deterministic (C2/C4)");
        assert_eq!(
            expl_a, expl_b,
            "explanation must be deterministic (same inputs → same record)"
        );
    }

    /// `explain` is deterministic: same `(policy, inputs)` → same `Explanation`.
    #[test]
    fn explain_is_deterministic() {
        let policy = two_repr_policy();
        let inputs = dense_inputs(8);
        let expl_a = explain(&policy, &inputs);
        let expl_b = explain(&policy, &inputs);
        // Mutation witness: adding a timestamp to Explanation would break this.
        assert_eq!(expl_a, expl_b, "explain must be deterministic (C4)");
    }

    /// `select_with_override` is deterministic for the same forced index.
    #[test]
    fn select_with_override_is_deterministic() {
        let policy = two_repr_policy();
        let inputs = binary_inputs(8);
        let (choice_a, expl_a) = select_with_override(&policy, &inputs, 1).unwrap();
        let (choice_b, expl_b) = select_with_override(&policy, &inputs, 1).unwrap();
        assert_eq!(
            choice_a, choice_b,
            "select_with_override must be deterministic (C2/C4)"
        );
        assert_eq!(expl_a, expl_b, "explanation must be deterministic");
    }

    // ── C4 content-addressed identity ────────────────────────────────────────────────────────

    /// Two policies built from identical data have the same `PolicyRef` (ADR-003 / C4).
    #[test]
    fn identical_policies_have_same_policy_ref() {
        let p1 = one_repr_policy("content-id.v1", Repr::Binary { width: 8 });
        let p2 = one_repr_policy("content-id.v1", Repr::Binary { width: 8 });
        // Mutation witness: using an address/pointer for identity instead of content.
        assert_eq!(
            policy_ref(&p1),
            policy_ref(&p2),
            "identical policies must have the same PolicyRef (ADR-003)"
        );
    }

    /// Two policies with different names have different `PolicyRef`s.
    #[test]
    fn different_policies_have_different_policy_ref() {
        let p1 = one_repr_policy("name-a.v1", Repr::Binary { width: 8 });
        let p2 = one_repr_policy("name-b.v1", Repr::Binary { width: 8 });
        assert_ne!(
            policy_ref(&p1),
            policy_ref(&p2),
            "policies with different names must have different PolicyRefs"
        );
    }

    /// The `PolicyRef` in an Explanation matches `policy_ref(&policy)` — the policy is always
    /// recoverable from the record alone (RFC-0005 §3; C4).
    #[test]
    fn explanation_policy_ref_matches_policy() {
        let policy = two_repr_policy();
        let inputs = binary_inputs(8);
        let explanation = explain(&policy, &inputs);
        assert_eq!(
            explanation.policy,
            policy_ref(&policy),
            "Explanation.policy must equal policy_ref (RFC-0005 §3 — self-describing record)"
        );
    }

    // ── explanation completeness (C3) ──────────────────────────────────────────────────────────

    /// Every candidate appears in `explanation.costs` — the EXPLAIN record is complete, not
    /// just the winner (RFC-0005 §2.2 — full ranking, not just the chosen cost).
    #[test]
    fn explanation_costs_covers_all_candidates() {
        let policy = two_repr_policy();
        let inputs = binary_inputs(8);
        let (_, explanation) = select(&policy, &inputs).unwrap();
        // Mutation witness: returning only the winner's cost would fail this count check.
        assert_eq!(
            explanation.costs.len(),
            policy.candidates().len(),
            "costs must have one entry per candidate (C3 — complete ranking)"
        );
    }

    /// The `explanation.chosen` matches `explanation.costs[chosen_index]` — self-consistent.
    #[test]
    fn explanation_chosen_consistent_with_costs() {
        let policy = two_repr_policy();
        let inputs = binary_inputs(8);
        let (_, explanation) = select(&policy, &inputs).unwrap();
        assert_eq!(
            explanation.chosen, explanation.costs[explanation.chosen_index].candidate,
            "chosen candidate must be consistent with chosen_index in costs"
        );
    }

    /// `explain` and `select` produce consistent explanations for the same `(policy, inputs)`.
    #[test]
    fn explain_and_select_consistent() {
        let policy = two_repr_policy();
        let inputs = dense_inputs(8);
        let (_, select_expl) = select(&policy, &inputs).unwrap();
        let explain_expl = explain(&policy, &inputs);
        assert_eq!(
            select_expl, explain_expl,
            "explain and select must produce identical Explanation for the same inputs"
        );
    }

    // ── cost model in declared bits (RFC-0005 §2; M-220) ──────────────────────────────────────

    /// `Cheapest` selects the lower-cost candidate; costs are in declared storage bits.
    /// Binary{width=8} costs 8 bits; Dense{dim=8, F32} costs 256 bits → Binary wins.
    #[test]
    fn cheapest_selects_minimum_cost_candidate() {
        let policy = two_repr_policy(); // Binary{8} vs Dense{8, F32}
        let inputs = binary_inputs(8);
        let (choice, explanation) = select(&policy, &inputs).unwrap();
        // Binary: 8 bits; Dense F32: 8 * 32 = 256 bits → Binary wins by cost.
        assert_eq!(
            choice,
            Candidate::Repr(Repr::Binary { width: 8 }),
            "Cheapest must select Binary (8 bits) over Dense F32 (256 bits)"
        );
        let binary_cost = explanation
            .costs
            .iter()
            .find(|c| c.candidate == Candidate::Repr(Repr::Binary { width: 8 }))
            .expect("Binary must appear in costs");
        let dense_cost = explanation
            .costs
            .iter()
            .find(|c| {
                c.candidate
                    == Candidate::Repr(Repr::Dense {
                        dim: 8,
                        dtype: ScalarKind::F32,
                    })
            })
            .expect("Dense must appear in costs");
        // Mutation witness: swapping the cost formula makes Binary appear more expensive.
        assert!(
            binary_cost.cost < dense_cost.cost,
            "Binary ({} bits) must cost less than Dense F32 ({} bits)",
            binary_cost.cost,
            dense_cost.cost
        );
    }

    /// Rule predicates fire in table order (first match wins — RFC-0005 §2.3 fixed precedence).
    #[test]
    fn predicate_rule_first_match_wins() {
        let policy = build(
            "first-match.v1",
            vec![
                Candidate::Repr(Repr::Binary { width: 1 }),
                Candidate::Repr(Repr::Dense {
                    dim: 1,
                    dtype: ScalarKind::F32,
                }),
            ],
            vec![
                // First rule matches Binary source, chooses index 0.
                Rule {
                    when: Predicate::SrcKindIs(ParadigmKind::Binary),
                    action: Action::Choose(0),
                },
                // Second rule always fires, chooses index 1.
                Rule {
                    when: Predicate::Always,
                    action: Action::Choose(1),
                },
            ],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .unwrap();

        let inputs = binary_inputs(1);
        let (_, explanation) = select(&policy, &inputs).unwrap();
        // Mutation witness: reversing rule evaluation order makes index 1 win.
        assert_eq!(
            explanation.matched_rule,
            Some(0),
            "first matching rule must win (fixed declared precedence)"
        );
        assert_eq!(explanation.chosen_index, 0, "rule 0 chooses index 0");
    }

    /// Default arm fires when no rule matches.
    #[test]
    fn default_arm_fires_when_no_rule_matches() {
        let policy = build(
            "default-arm.v1",
            vec![
                Candidate::Repr(Repr::Binary { width: 1 }),
                Candidate::Repr(Repr::Dense {
                    dim: 1,
                    dtype: ScalarKind::F32,
                }),
            ],
            vec![
                // Only matches Binary — Dense inputs fall through to default.
                Rule {
                    when: Predicate::SrcKindIs(ParadigmKind::Binary),
                    action: Action::Choose(0),
                },
            ],
            1, // default: Dense
            CostModel {
                storage_weight: 1.0,
            },
        )
        .unwrap();

        let inputs = dense_inputs(1);
        let (_, explanation) = select(&policy, &inputs).unwrap();
        // Mutation witness: making the rule always match would hide the default arm.
        assert_eq!(
            explanation.matched_rule, None,
            "no rule matched — default arm should fire (matched_rule = None)"
        );
        assert_eq!(
            explanation.chosen_index, 1,
            "default arm chose index 1 (Dense)"
        );
    }

    // ── PolicyRegistry ─────────────────────────────────────────────────────────────────────────

    /// A registered policy is retrievable by its `PolicyRef` — the operational form of RFC-0005
    /// §3 ("which policy chose this?").
    #[test]
    fn policy_registry_roundtrip() {
        let mut registry = PolicyRegistry::new();
        let policy = one_repr_policy("registry-test.v1", Repr::Binary { width: 32 });
        let pref = registry.register(policy.clone());
        let retrieved = registry
            .get(&pref)
            .expect("registered policy must be retrievable");
        assert_eq!(
            retrieved.name(),
            policy.name(),
            "retrieved policy must match by name"
        );
        assert_eq!(
            policy_ref(retrieved),
            pref,
            "retrieved policy's content hash must match the registered PolicyRef"
        );
    }

    /// An unregistered `PolicyRef` returns `None` — the registry never fabricates a policy.
    #[test]
    fn policy_registry_missing_returns_none() {
        let registry = PolicyRegistry::new();
        let fake_ref = policy_ref(&one_repr_policy("ghost.v1", Repr::Binary { width: 1 }));
        // Mutation witness: returning a default policy for missing refs breaks this.
        assert!(
            registry.get(&fake_ref).is_none(),
            "unregistered PolicyRef must return None (C1 — never a silent fabrication)"
        );
    }

    // ── site adapters ─────────────────────────────────────────────────────────────────────────

    /// `select_swap_target` refuses a packing candidate at the repr site (`WrongSiteKind` — C1).
    #[test]
    fn swap_target_adapter_refuses_packing_candidate() {
        use mycelium_core::PackScheme;
        let policy = build(
            "mixed-kinds.v1",
            vec![Candidate::Packing(PackScheme::I2S)],
            vec![Rule {
                when: Predicate::Always,
                action: Action::Choose(0),
            }],
            0,
            CostModel {
                storage_weight: 1.0,
            },
        )
        .unwrap();
        let meta = exact_meta();
        let inputs = SelectionInputs::from_meta(Repr::Ternary { trits: 4 }, &meta);
        let err = select_swap_target(&policy, &inputs, None)
            .expect_err("Packing candidate must be refused at the swap-target site");
        // Mutation witness: silently coercing to a default Repr would not error.
        assert!(
            matches!(
                err,
                SelectError::WrongSiteKind {
                    site: "swap-target",
                    ..
                }
            ),
            "must be WrongSiteKind {{site: swap-target}}, got {err:?}"
        );
    }

    // ── bitnet packing policy ─────────────────────────────────────────────────────────────────

    /// The built-in `bitnet_packing_policy` is well-formed and selects a packing candidate.
    #[test]
    fn bitnet_packing_policy_selects_packing() {
        let policy = bitnet_packing_policy();
        let meta = exact_meta();
        let inputs = SelectionInputs::from_meta(Repr::Ternary { trits: 64 }, &meta);
        let (scheme, explanation) = select_packing(&policy, &inputs, None)
            .expect("bitnet_packing_policy must select a valid packing");
        // Mutation witness: returning a Repr candidate breaks the PackScheme assertion.
        assert!(
            BITNET_PACKINGS.contains(&scheme),
            "selected packing must be in BITNET_PACKINGS"
        );
        assert_eq!(
            explanation.policy,
            policy_ref(&policy),
            "packing explanation policy must match the policy ref (C4)"
        );
    }

    /// `select_layout` refuses a non-ternary source (A5-02 well-formedness; C1 — never a
    /// silent mis-tag).
    #[test]
    fn select_layout_refuses_non_ternary_source() {
        let policy = bitnet_packing_policy();
        let meta = exact_meta();
        let inputs = SelectionInputs::from_meta(Repr::Binary { width: 8 }, &meta);
        let err = select_layout(&policy, &inputs, None)
            .expect_err("non-ternary source must be refused at the layout site");
        // Mutation witness: removing the ternary check produces a mismatched PhysicalLayout.
        assert!(
            matches!(err, SelectError::NonTernarySource { .. }),
            "must be NonTernarySource, got {err:?}"
        );
    }
}
