//! M-220/M-221 acceptance — the decision-table `SelectionPolicy` (RFC-0005 §2/§3) and the
//! mandatory EXPLAIN (RFC-0005 §2.2/§4): totality (default arm), fixed declared precedence,
//! determinism, content-addressing, first-class overrides, explicit costs, and the serializable
//! `Explanation` with the expected candidate ranking.

use mycelium_core::{
    Bound, BoundBasis, BoundKind, GuaranteeStrength, Meta, NormKind, PackScheme, Provenance, Repr,
    ScalarKind,
};
use mycelium_select::{
    explain, select, select_packing, select_placement, select_swap_target, Action, Candidate,
    CostModel, Explanation, NodeRef, ParadigmKind, PolicyError, Predicate, Rule, SelectError,
    SelectionInputs, SelectionPolicy,
};

fn unit_cost() -> CostModel {
    CostModel {
        storage_weight: 1.0,
    }
}

/// The worked swap-target policy used throughout: Dense F32 sources prefer BF16 (rule 0);
/// anything already-approximate beyond ε=0.01 keeps F32 (rule 1, listed *after* — precedence
/// check); default keeps F32.
fn swap_policy() -> SelectionPolicy {
    SelectionPolicy::new(
        "prefer-bf16-for-exact-f32",
        vec![
            Candidate::Repr(Repr::Dense {
                dim: 4,
                dtype: ScalarKind::Bf16,
            }),
            Candidate::Repr(Repr::Dense {
                dim: 4,
                dtype: ScalarKind::F32,
            }),
        ],
        vec![
            Rule {
                when: Predicate::All(vec![
                    Predicate::DtypeIs(ScalarKind::F32),
                    Predicate::GuaranteeAtLeast(GuaranteeStrength::Exact),
                ]),
                action: Action::Choose(0),
            },
            Rule {
                when: Predicate::DtypeIs(ScalarKind::F32),
                action: Action::Choose(1),
            },
        ],
        1,
        unit_cost(),
    )
    .unwrap()
}

fn exact_f32_inputs() -> SelectionInputs {
    SelectionInputs::from_meta(
        Repr::Dense {
            dim: 4,
            dtype: ScalarKind::F32,
        },
        &Meta::exact(Provenance::Root),
    )
}

fn approx_f32_inputs() -> SelectionInputs {
    let meta = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Empirical,
        Some(Bound {
            kind: BoundKind::Error {
                eps: 0.5,
                norm: NormKind::Rel,
            },
            basis: BoundBasis::EmpiricalFit {
                trials: 1000,
                method: "fixture".into(),
            },
        }),
        None,
        None,
        None,
    )
    .unwrap();
    SelectionInputs::from_meta(
        Repr::Dense {
            dim: 4,
            dtype: ScalarKind::F32,
        },
        &meta,
    )
}

// ---------- M-220: the decision table ----------

/// Fixed declared precedence: both rules match an exact F32 input; the *first* wins.
#[test]
fn first_matching_rule_wins() {
    let (chosen, expl) = select(&swap_policy(), &exact_f32_inputs(), None).unwrap();
    assert_eq!(
        chosen,
        Candidate::Repr(Repr::Dense {
            dim: 4,
            dtype: ScalarKind::Bf16
        })
    );
    assert_eq!(expl.matched_rule, Some(0));
}

/// An approximate input falls through rule 0's guard to rule 1 — same table, different row.
#[test]
fn guards_inspect_the_exact_metadata() {
    let (chosen, expl) = select(&swap_policy(), &approx_f32_inputs(), None).unwrap();
    assert_eq!(
        chosen,
        Candidate::Repr(Repr::Dense {
            dim: 4,
            dtype: ScalarKind::F32
        })
    );
    assert_eq!(expl.matched_rule, Some(1));
}

/// Totality: an input matching no rule lands on the mandatory default arm.
#[test]
fn default_arm_makes_the_table_total() {
    let inputs =
        SelectionInputs::from_meta(Repr::Binary { width: 8 }, &Meta::exact(Provenance::Root));
    let (_, expl) = select(&swap_policy(), &inputs, None).unwrap();
    assert_eq!(expl.matched_rule, None);
    assert_eq!(expl.chosen_index, 1);
    assert!(!expl.overridden);
}

/// Determinism: same inputs → bit-identical choice and explanation, across repeated runs and
/// over a sweep of inputs (RFC-0005 §2.3).
#[test]
fn selection_is_deterministic() {
    let policy = swap_policy();
    for inputs in [exact_f32_inputs(), approx_f32_inputs()] {
        let a = select(&policy, &inputs, None).unwrap();
        for _ in 0..100 {
            assert_eq!(select(&policy, &inputs, None).unwrap(), a);
        }
    }
}

/// Content-addressing (RFC-0005 §3): structurally equal policies share a `PolicyRef`; any
/// semantic edit (a rule, the default, the cost) changes it.
#[test]
fn policies_are_content_addressed() {
    assert_eq!(swap_policy().policy_ref(), swap_policy().policy_ref());
    let mut rules_changed = swap_policy().rules().to_vec();
    rules_changed.pop();
    let edited = SelectionPolicy::new(
        "prefer-bf16-for-exact-f32",
        swap_policy().candidates().to_vec(),
        rules_changed,
        1,
        unit_cost(),
    )
    .unwrap();
    assert_ne!(swap_policy().policy_ref(), edited.policy_ref());
}

/// First-class override: forces a candidate deterministically and is recorded as such.
#[test]
fn override_forces_a_candidate() {
    let (chosen, expl) = select(&swap_policy(), &exact_f32_inputs(), Some(1)).unwrap();
    assert_eq!(
        chosen,
        Candidate::Repr(Repr::Dense {
            dim: 4,
            dtype: ScalarKind::F32
        })
    );
    assert!(expl.overridden);
    assert_eq!(expl.matched_rule, None);
    // Out of range is an explicit error, never a clamped silent choice.
    assert_eq!(
        select(&swap_policy(), &exact_f32_inputs(), Some(7)),
        Err(SelectError::OverrideOutOfRange {
            index: 7,
            candidates: 2
        })
    );
}

/// `Cheapest` picks the explicit-cost minimum; ties break to the lowest index.
#[test]
fn cheapest_minimizes_the_explicit_cost() {
    let policy = SelectionPolicy::new(
        "cheapest-storage",
        vec![
            Candidate::Repr(Repr::Dense {
                dim: 4,
                dtype: ScalarKind::F32,
            }), // 128 bits
            Candidate::Repr(Repr::Dense {
                dim: 4,
                dtype: ScalarKind::Bf16,
            }), // 64 bits ← cheapest
            Candidate::Repr(Repr::Dense {
                dim: 4,
                dtype: ScalarKind::F16,
            }), // 64 bits (tie, higher index)
        ],
        vec![Rule {
            when: Predicate::Always,
            action: Action::Cheapest,
        }],
        0,
        unit_cost(),
    )
    .unwrap();
    let (chosen, expl) = select(&policy, &exact_f32_inputs(), None).unwrap();
    assert_eq!(expl.chosen_index, 1, "tie breaks to the lowest index");
    assert_eq!(
        chosen,
        Candidate::Repr(Repr::Dense {
            dim: 4,
            dtype: ScalarKind::Bf16
        })
    );
}

/// Construction validates totality up front: empty candidates, dangling indices, bad cost.
#[test]
fn invalid_policies_are_rejected_at_construction() {
    assert_eq!(
        SelectionPolicy::new("e", vec![], vec![], 0, unit_cost()),
        Err(PolicyError::NoCandidates)
    );
    let one = vec![Candidate::Packing(PackScheme::I2S)];
    assert_eq!(
        SelectionPolicy::new("e", one.clone(), vec![], 3, unit_cost()),
        Err(PolicyError::IndexOutOfRange { index: 3 })
    );
    assert_eq!(
        SelectionPolicy::new(
            "e",
            one.clone(),
            vec![Rule {
                when: Predicate::Always,
                action: Action::Choose(9)
            }],
            0,
            unit_cost()
        ),
        Err(PolicyError::IndexOutOfRange { index: 9 })
    );
    assert_eq!(
        SelectionPolicy::new(
            "e",
            one,
            vec![],
            0,
            CostModel {
                storage_weight: 0.0
            }
        ),
        Err(PolicyError::BadCost)
    );
}

// ---------- M-221: the mandatory EXPLAIN ----------

/// The EXPLAIN record carries everything RFC-0005 §2.2 demands: inputs considered, the cost of
/// *each* candidate, the chosen option, and the override state — and the ranking matches the
/// declared cost function (bits × weight), hand-computed.
#[test]
fn explanation_matches_the_expected_ranking() {
    let expl: Explanation = explain(&swap_policy(), &exact_f32_inputs());
    assert_eq!(expl.policy, swap_policy().policy_ref());
    assert_eq!(expl.inputs, exact_f32_inputs());
    assert_eq!(expl.costs.len(), 2, "every candidate is costed");
    // Hand-computed: Dense{4, BF16} = 4×16 = 64 bits; Dense{4, F32} = 4×32 = 128 bits.
    assert!((expl.costs[0].cost - 64.0).abs() < 1e-12);
    assert!((expl.costs[1].cost - 128.0).abs() < 1e-12);
    assert_eq!(expl.chosen_index, 0);
    assert!(!expl.overridden);
}

/// `explain` is total and deterministic, and the record serde round-trips (serializable).
#[test]
fn explain_is_total_deterministic_and_serializable() {
    let policy = swap_policy();
    for inputs in [
        exact_f32_inputs(),
        approx_f32_inputs(),
        SelectionInputs::from_meta(Repr::Ternary { trits: 6 }, &Meta::exact(Provenance::Root)),
    ] {
        let e1 = explain(&policy, &inputs);
        let e2 = explain(&policy, &inputs);
        assert_eq!(e1, e2);
        let json = serde_json::to_string(&e1).unwrap();
        let back: Explanation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e1);
    }
}

/// A policy itself serde round-trips (inspectable/diffable), and a tampered wire form with a
/// dangling index is rejected on deserialize — never silently trusted.
#[test]
fn policy_round_trips_and_rejects_malformed_wire() {
    let policy = swap_policy();
    let json = serde_json::to_string(&policy).unwrap();
    let back: SelectionPolicy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, policy);
    assert_eq!(back.policy_ref(), policy.policy_ref());
    let bad = json.replace("\"default_choice\":1", "\"default_choice\":99");
    assert!(serde_json::from_str::<SelectionPolicy>(&bad).is_err());
}

/// One mechanism, two sites: the same `select` drives the packing adapter, and a candidate of
/// the wrong kind at a site is an explicit refusal.
#[test]
fn site_adapters_share_one_mechanism() {
    let packing = SelectionPolicy::new(
        "cheapest-packing",
        vec![
            Candidate::Packing(PackScheme::Unpacked), // 8 b/t
            Candidate::Packing(PackScheme::I2S),      // 2 b/t
            Candidate::Packing(PackScheme::Tl2),      // 1.67 b/t ← cheapest
        ],
        vec![Rule {
            when: Predicate::SrcKindIs(ParadigmKind::Ternary),
            action: Action::Cheapest,
        }],
        0,
        unit_cost(),
    )
    .unwrap();
    let inputs =
        SelectionInputs::from_meta(Repr::Ternary { trits: 64 }, &Meta::exact(Provenance::Root));
    let (scheme, expl) = select_packing(&packing, &inputs, None).unwrap();
    assert_eq!(scheme, PackScheme::Tl2);
    assert!((expl.costs[2].cost - 1.67 * 64.0).abs() < 1e-9);
    // The swap-target adapter refuses a packing candidate — explicit, never coerced.
    assert!(matches!(
        select_swap_target(&packing, &inputs, None),
        Err(SelectError::WrongSiteKind {
            site: "swap-target",
            ..
        })
    ));

    // The fourth site (M-906 D-lite `forage`; RFC-0008 RT3): `select_placement` shares the SAME
    // `select` engine — no new mechanism — and refuses a non-`Node` candidate the same way the
    // other three adapters refuse a wrong-kind candidate.
    let placement = SelectionPolicy::new(
        "forage.dlite.v0",
        vec![Candidate::Node(NodeRef("worker-0".to_owned()))],
        Vec::new(),
        0,
        unit_cost(),
    )
    .unwrap();
    let (node, node_expl) = select_placement(&placement, &inputs, None).unwrap();
    assert_eq!(node, NodeRef("worker-0".to_owned()));
    assert_eq!(
        node_expl.chosen,
        Candidate::Node(NodeRef("worker-0".to_owned()))
    );
    assert!(matches!(
        select_placement(&packing, &inputs, None),
        Err(SelectError::WrongSiteKind {
            site: "placement",
            ..
        })
    ));
}

/// A5-01/B2-02 regression: non-finite predicate `f64` literals are refused at construction, so two
/// materially different policies cannot collide on one content-addressed `policy_ref` (NaN and +∞
/// both serialize to JSON `null`). Mutant-witness: removing the `literals_finite` check in
/// `SelectionPolicy::new` lets these construct (and `eps ≤ NaN` vs `eps ≤ ∞` then share a ref).
#[test]
fn non_finite_predicate_literals_are_refused() {
    let with_eps = |eps: f64| {
        SelectionPolicy::new(
            "p",
            vec![Candidate::Repr(Repr::Dense {
                dim: 4,
                dtype: ScalarKind::F32,
            })],
            vec![Rule {
                when: Predicate::ErrorEpsAtMost(eps),
                action: Action::Choose(0),
            }],
            0,
            unit_cost(),
        )
    };
    assert_eq!(
        with_eps(f64::NAN).unwrap_err(),
        PolicyError::BadPredicateLiteral
    );
    assert_eq!(
        with_eps(f64::INFINITY).unwrap_err(),
        PolicyError::BadPredicateLiteral
    );
    assert_eq!(
        with_eps(f64::NEG_INFINITY).unwrap_err(),
        PolicyError::BadPredicateLiteral
    );
    // Finite is fine; the check recurses through Not/Any/All.
    assert!(with_eps(0.01).is_ok());
    let nested = SelectionPolicy::new(
        "p",
        vec![Candidate::Repr(Repr::Dense {
            dim: 4,
            dtype: ScalarKind::F32,
        })],
        vec![Rule {
            when: Predicate::Not(Box::new(Predicate::Any(vec![Predicate::ErrorEpsAtMost(
                f64::NAN,
            )]))),
            action: Action::Choose(0),
        }],
        0,
        unit_cost(),
    );
    assert_eq!(nested.unwrap_err(), PolicyError::BadPredicateLiteral);
}
