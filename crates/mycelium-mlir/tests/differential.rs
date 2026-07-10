//! M-151 — interp↔AOT **differential** testing (NFR-7; VR-4; RR-12).
//!
//! Runs a kernel corpus under both execution paths — the M-110 reference interpreter (small-step
//! substitution over the nested tree) and the M-150 AOT artifact (`mycelium_mlir::run`, a big-step
//! env-machine over the lowered A-normal form) — and asserts **observable equivalence**: same
//! representation, same payload, same guarantee tag. Divergence fails the test (and thus CI). This
//! is the cheap baseline preceding per-artifact translation validation in Phase 2.
//!
//! *Observable* = `repr + payload + guarantee`. Dynamic metadata (provenance, `policy_used`) is
//! path-dependent and intentionally excluded — "two execution paths must never mean two semantics"
//! is about results, not derivation records (NFR-7).
//!
//! Since M-210 the differential is **folded into the single shared TV checker** (RFC-0002 §2 /
//! RFC-0004 §3): each corpus pair also validates through
//! `mycelium_cert::check(interp, aot, ObservationalEquiv, {0,0,Exact}, Observational)` — the
//! interp↔AOT instance of the one checker that also validates swap certificates.

mod common;
use common::{byte, observable, A, ONES};

use mycelium_cert::{check, BinaryTernarySwapEngine, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{ternary, GuaranteeStrength, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_numerics::Certificate;

// Local variant: uses `int_to_trits(i64, u32)` — differs from the Vec<Trit> form in common.
fn tern(value: i64, m: u32) -> Value {
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(ternary::int_to_trits(value, m).unwrap()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

// Local: name differs from wrong_layout.rs's policy_ref() (same value, different call sites).
fn policy() -> mycelium_core::ContentHash {
    mycelium_core::ContentHash::parse("blake3:po1icy_Ref00").unwrap()
}

/// The kernel corpus: closed programs runnable by both paths (const/let/op/swap subset).
fn corpus() -> Vec<Node> {
    let cst = |v: Value| Node::Const(v);
    vec![
        // bare constant
        cst(byte(A)),
        // let / var
        Node::Let {
            id: "a".into(),
            bound: Box::new(cst(byte(A))),
            body: Box::new(Node::Var("a".into())),
        },
        // bit ops
        Node::Op {
            prim: "bit.not".into(),
            args: vec![cst(byte(A))],
        },
        Node::Op {
            prim: "bit.xor".into(),
            args: vec![cst(byte(A)), cst(byte(ONES))],
        },
        // ternary arithmetic
        Node::Op {
            prim: "trit.add".into(),
            args: vec![cst(tern(5, 4)), cst(tern(-3, 4))],
        },
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![cst(tern(6, 4)), cst(tern(-6, 4))],
        },
        // nested let + op
        Node::Let {
            id: "x".into(),
            bound: Box::new(cst(tern(4, 4))),
            body: Box::new(Node::Op {
                prim: "trit.add".into(),
                args: vec![Node::Var("x".into()), Node::Var("x".into())],
            }),
        },
        // certified swap, binary → ternary
        Node::Swap {
            src: Box::new(cst(byte(A))),
            target: Repr::Ternary { trits: 6 },
            policy: policy(),
        },
        // round-trip swap through a let
        Node::Let {
            id: "b".into(),
            bound: Box::new(cst(byte([
                false, false, true, false, true, false, true, false,
            ]))),
            body: Box::new(Node::Swap {
                src: Box::new(Node::Swap {
                    src: Box::new(Node::Var("b".into())),
                    target: Repr::Ternary { trits: 6 },
                    policy: policy(),
                }),
                target: Repr::Binary { width: 8 },
                policy: policy(),
            }),
        },
        // op feeding a swap
        Node::Let {
            id: "c".into(),
            bound: Box::new(cst(byte(A))),
            body: Box::new(Node::Swap {
                src: Box::new(Node::Op {
                    prim: "bit.not".into(),
                    args: vec![Node::Var("c".into())],
                }),
                target: Repr::Ternary { trits: 6 },
                policy: policy(),
            }),
        },
    ]
}

#[test]
fn interp_and_aot_are_observably_equivalent_on_the_corpus() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    for (i, node) in corpus().iter().enumerate() {
        let r = interp.eval(node);
        let a = mycelium_mlir::run(node, &prims, &engine);
        match (r, a) {
            (Ok(rv), Ok(av)) => {
                assert_eq!(
                    observable(&rv),
                    observable(&av),
                    "program #{i} diverged: interp {:?} vs aot {:?}",
                    rv.payload(),
                    av.payload()
                );
                // M-210: the same pair validates through the shared TV checker — the
                // interp↔AOT observational-equivalence instance (RFC-0004 §3).
                assert_eq!(
                    check(
                        &rv,
                        &av,
                        RefinementRelation::ObservationalEquiv,
                        Certificate::exact(),
                        &Evidence::Observational,
                    ),
                    CheckVerdict::Validated {
                        strength: GuaranteeStrength::Exact
                    },
                    "program #{i}: the shared checker must validate the differential pair"
                );
            }
            (re, ae) => panic!("program #{i}: interp={re:?} aot={ae:?} — both paths must agree"),
        }
    }
}

/// Sanity: the harness actually discriminates — two different programs are NOT observably equal
/// (so a passing differential is meaningful, not vacuous), and the shared checker reports the
/// same divergence explicitly (never a silent pass).
#[test]
fn differential_distinguishes_different_programs() {
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let x = mycelium_mlir::run(&Node::Const(byte(A)), &prims, &engine).unwrap();
    let y = mycelium_mlir::run(&Node::Const(byte(ONES)), &prims, &engine).unwrap();
    assert_ne!(observable(&x), observable(&y));
    let verdict = check(
        &x,
        &y,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    );
    assert!(
        matches!(verdict, CheckVerdict::NotValidated { .. }),
        "the checker must reject a genuinely divergent pair, got {verdict:?}"
    );
}
