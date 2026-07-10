//! M-140 acceptance — a **scripted client** drives the LSP feedback facade and reads all four
//! semantic-feedback artifact kinds over one surface (FR-S5; Foundation §5.8; SC-5).

use mycelium_cert::SwapCertificate;
use mycelium_core::{
    Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, Node, Payload, Provenance,
    Repr, SparsityClass, Value,
};
use mycelium_lsp::{analyze, Severity};

fn policy() -> ContentHash {
    ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A program a client would author: `let a = <byte> in swap(a -> Ternary{6})`.
fn swap_program() -> Node {
    Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte([
            true, false, true, true, false, false, true, false,
        ]))),
        body: Box::new(Node::Swap {
            src: Box::new(Node::Const(byte([
                true, false, true, true, false, false, true, false,
            ]))),
            target: Repr::Ternary { trits: 6 },
            policy: policy(),
        }),
    }
}

/// The headline acceptance: one `analyze` call surfaces all four artifact kinds.
#[test]
fn facade_exposes_all_four_artifact_kinds() {
    let fb = analyze(&swap_program());

    // (1) Diagnostics channel exists (this clean program has no errors).
    assert!(!mycelium_lsp::has_errors(&fb.diagnostics));

    // (2) Swap certificates: the binary→ternary swap over a const yields a Bijective certificate.
    assert_eq!(fb.swaps.len(), 1);
    match &fb.swaps[0].certificate {
        Some(SwapCertificate::Bijective { params, .. }) => {
            assert_eq!(params.width, 8);
            assert_eq!(params.trits, 6);
        }
        other => panic!("expected a Bijective certificate, got {other:?}"),
    }

    // (3) Bound/guarantee annotations: one per Const (the bound value + the swap source).
    assert_eq!(fb.guarantees.len(), 2);
    assert!(fb
        .guarantees
        .iter()
        .all(|a| a.guarantee == GuaranteeStrength::Exact && a.bound.is_none()));

    // (4) Lowering-stage dumps: core → substrate, both non-empty.
    assert_eq!(fb.stages.len(), 2);
    assert_eq!(fb.stages[0].name, "core");
    assert_eq!(fb.stages[1].name, "substrate");
    assert!(!fb.stages[0].text.is_empty() && !fb.stages[1].text.is_empty());
}

/// M-310: the structured `summary` rolls up the artifact-kind counts and the worst severity — the
/// at-a-glance health signal an AI co-author's feedback loop or an IDE status line consumes.
#[test]
fn summary_rolls_up_counts_and_worst_severity() {
    // A clean program: counts match the channels, no diagnostics, worst is None.
    let clean = analyze(&swap_program()).summary();
    assert_eq!(clean.errors, 0);
    assert_eq!(clean.swaps, 1);
    assert_eq!(clean.guarantees, 2);
    assert_eq!(clean.stages, 2);
    assert!(clean.is_clean());
    assert_eq!(clean.worst, None);

    // A program with an out-of-range swap: the summary reports the error and worst = Error.
    // Mutant-witness: if `summary` miscounted severities (or `worst` ignored Error), is_clean()
    // would stay true here.
    let all_plus = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![mycelium_core::Trit::Pos; 6]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let bad = analyze(&Node::Swap {
        src: Box::new(Node::Const(all_plus)),
        target: Repr::Binary { width: 8 },
        policy: policy(),
    })
    .summary();
    assert!(bad.errors >= 1);
    assert!(!bad.is_clean());
    assert_eq!(bad.worst, Some(Severity::Error));
}

/// A client linting a sloppy program sees the diagnostics channel light up.
#[test]
fn diagnostics_channel_surfaces_invariant_violations() {
    // An op mixing a binary and a declared-dense operand → implicit-swap (error) + unverified-bound.
    let declared = Value::new(
        Repr::Dense {
            dim: 1,
            dtype: mycelium_core::ScalarKind::F32,
        },
        Payload::Scalars(vec![1.0]),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Declared,
            Some(Bound {
                kind: BoundKind::Probability { delta: 0.2 },
                basis: BoundBasis::UserDeclared,
            }),
            None,
            None,
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let prog = Node::Op {
        prim: "mix".into(),
        args: vec![Node::Const(byte([false; 8])), Node::Const(declared.clone())],
    };
    let fb = analyze(&prog);
    let codes: Vec<&str> = fb.diagnostics.iter().map(|d| d.code).collect();
    assert!(codes.contains(&"implicit-swap"));
    assert!(codes.contains(&"unverified-bound"));

    // The guarantee channel reflects the Declared value with its bound.
    let declared_ann = fb
        .guarantees
        .iter()
        .find(|a| a.guarantee == GuaranteeStrength::Declared)
        .expect("declared value annotated");
    assert!(declared_ann.bound.is_some());
}

/// An out-of-range / unsupported swap is reported on the diagnostics channel — never silent.
#[test]
fn failed_swap_is_surfaced_not_silent() {
    // Decoding an all-`+` 6-trit value (364) to Binary{8} is out of range (P4).
    let all_plus = Value::new(
        Repr::Ternary { trits: 6 },
        Payload::Trits(vec![mycelium_core::Trit::Pos; 6]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();
    let prog = Node::Swap {
        src: Box::new(Node::Const(all_plus)),
        target: Repr::Binary { width: 8 },
        policy: policy(),
    };
    let fb = analyze(&prog);
    assert!(fb.swaps[0].certificate.is_none());
    assert!(fb
        .diagnostics
        .iter()
        .any(|d| d.code == "swap-error" && d.severity == Severity::Error));
}

/// A6-05 regression: a statically-known source whose swap pair has *no* implemented certifier
/// (e.g. Binary→Dense) must surface an `unsupported-swap-pair` diagnostic — the empty certificate
/// channel is never silent for a known source. Mutant-witness: dropping the diagnostic on the
/// `None` arm (back to a bare `None`) would leave `certificate: None` with zero diagnostics.
#[test]
fn unsupported_swap_pair_is_surfaced_not_silent() {
    let prog = Node::Swap {
        src: Box::new(Node::Const(byte([true; 8]))),
        target: Repr::Dense {
            dim: 4,
            dtype: mycelium_core::ScalarKind::F32,
        },
        policy: policy(),
    };
    let fb = analyze(&prog);
    // The source *is* statically known, yet no certifier exists for this pair.
    assert!(fb.swaps[0].certificate.is_none());
    assert!(
        fb.diagnostics
            .iter()
            .any(|d| d.code == "unsupported-swap-pair" && d.severity == Severity::Error),
        "an unsupported but statically-known swap pair must surface a diagnostic, got {:?}",
        fb.diagnostics
    );
}

/// A VSA value's Proven capacity bound is visible on the guarantee channel.
#[test]
fn proven_bound_is_visible_on_the_guarantee_channel() {
    let proven = Value::new(
        Repr::Vsa {
            model: "MAP-I".into(),
            dim: 4,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(vec![1.0, 0.0, 0.0, -1.0]),
        Meta::new(
            Provenance::Root,
            GuaranteeStrength::Proven,
            Some(Bound {
                kind: BoundKind::Capacity { items: 2, dim: 4 },
                basis: BoundBasis::ProvenThm {
                    citation: "Clarkson-Ubaru-Yang 2023".into(),
                },
            }),
            None,
            None,
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let fb = analyze(&Node::Const(proven));
    assert_eq!(fb.guarantees.len(), 1);
    assert_eq!(fb.guarantees[0].guarantee, GuaranteeStrength::Proven);
    assert!(matches!(
        fb.guarantees[0].bound.as_ref().map(|b| &b.basis),
        Some(BoundBasis::ProvenThm { .. })
    ));
}

/// M-221 acceptance — the facade surfaces the **EXPLAIN channel** (RFC-0005 §4; SC-5): with a
/// registered policy, `analyze_with` re-derives the selection at the swap site (deterministic),
/// and a target that disagrees with the policy's choice raises a `policy-divergence` warning —
/// surfaced, never silent.
#[test]
fn facade_surfaces_selection_explain() {
    use mycelium_core::ScalarKind;
    use mycelium_lsp::analyze_with;
    use mycelium_select::{
        Action, Candidate, CostModel, PolicyRegistry, Predicate, Rule, SelectionPolicy,
    };

    let select_policy = SelectionPolicy::new(
        "bf16-when-f32",
        vec![
            Candidate::Repr(Repr::Dense {
                dim: 2,
                dtype: ScalarKind::Bf16,
            }),
            Candidate::Repr(Repr::Dense {
                dim: 2,
                dtype: ScalarKind::F32,
            }),
        ],
        vec![Rule {
            when: Predicate::DtypeIs(ScalarKind::F32),
            action: Action::Choose(0),
        }],
        1,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .unwrap();
    let mut registry = PolicyRegistry::new();
    let policy_ref = registry.register(select_policy.clone());

    let f32_const = Value::new(
        Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        Payload::Scalars(vec![1.5, -2.0]),
        Meta::exact(Provenance::Root),
    )
    .unwrap();

    // A swap whose target agrees with the policy's choice: EXPLAIN surfaced, no divergence.
    let agreeing = Node::Swap {
        src: Box::new(Node::Const(f32_const.clone())),
        target: Repr::Dense {
            dim: 2,
            dtype: ScalarKind::Bf16,
        },
        policy: policy_ref.clone(),
    };
    let fb = analyze_with(&agreeing, &registry);
    assert_eq!(fb.explanations.len(), 1, "the EXPLAIN channel is populated");
    let ex = &fb.explanations[0].explanation;
    assert_eq!(ex.policy, policy_ref);
    assert_eq!(ex.costs.len(), 2, "every candidate is costed in the trace");
    assert!(!fb.diagnostics.iter().any(|d| d.code == "policy-divergence"));
    // Deterministic: re-analyzing yields the identical trace (same Meta in, same trace out).
    assert_eq!(
        analyze_with(&agreeing, &registry).explanations,
        fb.explanations
    );

    // A swap whose recorded target disagrees with the policy: surfaced as a warning.
    let diverging = Node::Swap {
        src: Box::new(Node::Const(f32_const)),
        target: Repr::Dense {
            dim: 2,
            dtype: ScalarKind::F32,
        },
        policy: policy_ref,
    };
    let fb = analyze_with(&diverging, &registry);
    assert!(fb
        .diagnostics
        .iter()
        .any(|d| d.code == "policy-divergence" && d.severity == Severity::Warning));

    // Plain `analyze` (no registry) keeps the channel empty — nothing to resolve against.
    assert!(analyze(&swap_program()).explanations.is_empty());
}
