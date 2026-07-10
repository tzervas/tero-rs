//! M-251 — the **E3 wrong-layout soundness differential** (RFC-0004 §8; NFR-7; M-I5; RR-12).
//!
//! E3 (DN-01 §7; RFC-0004 §8) asks: can a wrong `Meta.physical`/schedule tag cause a memory
//! misread, and does the NFR-7 reference-equivalence check catch it? This extends the M-151
//! interp↔AOT differential to the **schedule-staged packing** stage: the AOT path packs a ternary
//! result into a physical buffer and reads it back under its recorded layout tag
//! (`mycelium_mlir::run_with_layout`). The reference (interpreter) is layout-agnostic.
//!
//! - **A correctly-labeled layout passes:** packed-as == tag ⇒ the read-back is the identity ⇒ the
//!   AOT result is observably equal to the interpreter's, and the M-210 shared checker
//!   (`ObservationalEquiv`) **validates** the pair.
//! - **A mislabeled layout is caught:** packed-as ≠ tag ⇒ the buffer is misread ⇒ the payload
//!   differs ⇒ the **same** checker reports an explicit `NotValidated{ Diverged }` (the
//!   circuit-breaker fires; the swap/result is refused, fall back to the reference — ADR-007).
//!
//! Honesty: the layout record (chosen by the M-250 selector) is trusted **only because a wrong one
//! is caught here**. The true scheme used below is the one the M-250 `bitnet_packing_policy`
//! actually selects, tying the soundness check to the selector it guards.

mod common;
use common::{byte, A};

use mycelium_cert::{check, BinaryTernarySwapEngine, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{
    GuaranteeStrength, Meta, Node, PackScheme, PhysicalLayout, Provenance, Repr, Value,
};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_numerics::Certificate;
use mycelium_select::{bitnet_packing_policy, layout_of, select_layout, SelectionInputs};

// Local: named policy_ref (vs policy() in differential.rs) — kept local to avoid unifying names.
fn policy_ref() -> mycelium_core::ContentHash {
    mycelium_core::ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

/// A corpus of ternary-*producing* programs (the ones a packing layout applies to): a bare swap,
/// a swap fed by an op, and a swap through a let.
fn ternary_corpus() -> Vec<Node> {
    let cst = |v: Value| Node::Const(v);
    vec![
        Node::Swap {
            src: Box::new(cst(byte(A))),
            target: Repr::Ternary { trits: 6 },
            policy: policy_ref(),
        },
        Node::Let {
            id: "c".into(),
            bound: Box::new(cst(byte(A))),
            body: Box::new(Node::Swap {
                src: Box::new(Node::Op {
                    prim: "bit.not".into(),
                    args: vec![Node::Var("c".into())],
                }),
                target: Repr::Ternary { trits: 6 },
                policy: policy_ref(),
            }),
        },
    ]
}

/// The scheme the M-250 packing selector actually chooses for a ternary value — the layout this
/// soundness check guards.
fn selected_scheme(trits: u32) -> PackScheme {
    let policy = bitnet_packing_policy();
    let meta = Meta::exact(Provenance::Root);
    let inputs = SelectionInputs::from_meta(Repr::Ternary { trits }, &meta);
    let (layout, _explain) = select_layout(&policy, &inputs, None).unwrap();
    match layout {
        PhysicalLayout::TritPacked { scheme } => scheme,
        other => panic!("a ternary value packs as TritPacked, got {other:?}"),
    }
}

fn observe(
    interp: &Interpreter,
    prims: &PrimRegistry,
    node: &Node,
    packed_as: PackScheme,
    tag: PackScheme,
) -> (Value, Value) {
    let reference = interp.eval(node).expect("interp must evaluate the corpus");
    let aot = mycelium_mlir::run_with_layout(node, prims, &BinaryTernarySwapEngine, packed_as, tag)
        .expect("aot must evaluate the corpus");
    (reference, aot)
}

#[test]
fn a_correctly_labeled_layout_passes_the_differential() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let scheme = selected_scheme(6); // what M-250 chose (TL2)

    for (i, node) in ternary_corpus().iter().enumerate() {
        // Correct tag: the buffer is read under the same scheme it was packed in (identity).
        let (reference, aot) = observe(&interp, &prims, node, scheme, scheme);

        // The recorded layout is present and matches the selector's choice (M-I5 lossless record).
        assert_eq!(aot.meta().physical(), Some(layout_of(scheme)));
        // The layout did not change the value: same trits as the layout-agnostic reference.
        assert_eq!(aot.payload(), reference.payload(), "program #{i}");

        // The M-210 shared checker validates the observational-equivalence pair (RFC-0004 §3).
        assert_eq!(
            check(
                &reference,
                &aot,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "program #{i}: a correct layout must pass the NFR-7 check"
        );
    }
}

#[test]
fn a_mislabeled_layout_is_caught_by_the_differential() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let packed_as = selected_scheme(6); // TL2: what the buffer is actually packed in
    let wrong_tag = PackScheme::I2S; // a deliberate mislabel (≠ TL2)
    assert_ne!(packed_as, wrong_tag);

    for (i, node) in ternary_corpus().iter().enumerate() {
        // Mislabel: pack under the true scheme, but tag (and read) as the wrong one — a misread.
        let (reference, aot) = observe(&interp, &prims, node, packed_as, wrong_tag);

        // The misread genuinely changed the payload (the soundness hazard is real).
        assert_ne!(
            aot.payload(),
            reference.payload(),
            "program #{i}: a wrong layout must misread the buffer"
        );

        // The circuit-breaker fires: the same M-210 checker reports an explicit divergence — never
        // a silent miscompute (NFR-7; the wrong tag is *caught*).
        let verdict = check(
            &reference,
            &aot,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        );
        assert!(
            matches!(
                verdict,
                CheckVerdict::NotValidated {
                    reason: mycelium_cert::NotValidatedReason::Diverged { .. },
                    ..
                }
            ),
            "program #{i}: a mislabeled layout must fail the NFR-7 check, got {verdict:?}"
        );
    }
}

/// The differential discriminates only on the *layout tag*: holding everything else fixed, flipping
/// the tag from correct to wrong flips the verdict from `Validated` to `NotValidated`. So a passing
/// E3 is meaningful (not vacuous) and is *about the layout*, nothing else.
#[test]
fn the_verdict_flips_solely_on_the_layout_tag() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let node = &ternary_corpus()[0];
    let truth = selected_scheme(6);

    let validates = |tag: PackScheme| {
        let (r, a) = observe(&interp, &prims, node, truth, tag);
        matches!(
            check(
                &r,
                &a,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated { .. }
        )
    };

    assert!(validates(truth), "correct tag validates");
    assert!(!validates(PackScheme::Tl1), "wrong tag (TL1) is caught");
    assert!(!validates(PackScheme::I2S), "wrong tag (I2_S) is caught");
}
