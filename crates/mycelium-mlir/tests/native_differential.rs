//! M-302 — interp↔**native** differential testing (NFR-7; VR-4; RR-12; phase-3.md §2 Batch J).
//!
//! Extends the M-151 differential (`differential.rs`, interp vs the `aot::run` env-machine) to the
//! **genuinely compiled** path: each bit-subset program is run under the M-110 reference interpreter
//! *and* `mycelium_mlir::compile_and_run` (LLVM IR → `llc` → `clang` → native → read-back), and the
//! pair must be **observably equivalent** (`repr + payload + guarantee`) and **validate through the
//! single shared M-210 checker** (`ObservationalEquiv`). A deliberately divergent pair must be
//! caught — so a passing differential is meaningful, not vacuous.
//!
//! The compiled path needs `llc`/`clang`; where they are absent `compile_and_run` returns
//! `AotError::ToolchainMissing` and the test **skips** (the house "skip gracefully" idiom), exactly
//! as the proofs/api/supply-chain checks do — never a false failure.
//!
//! M-373 (Increment-1): extends coverage to the `Construct`/`Match` data fragment — non-recursive,
//! bounded, stack-alloca lowering (DN-15 §4.1 / RFC-0004 §11.2). Guarantee tag: `Declared`
//! (hand-written IR lowering; the differential is empirical evidence, not a proof — VR-5). The
//! `App`/`Lam`/`Fix`/`FixGroup` nodes must still return explicit `AotError::UnsupportedNode`.

mod common;
use common::{byte, observable, tern, A, B, ONES};

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{
    Alt, CtorSpec, DataRegistry, DeclSpec, FieldSpec, GuaranteeStrength, Node, Payload, Repr, Trit,
    Value,
};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::AotError;
use mycelium_numerics::Certificate;
use std::collections::BTreeMap;

/// The bit/trit-subset corpus: straight-line bit logic + balanced-ternary `neg` the direct-LLVM
/// backend lowers (no swaps, no trit *arithmetic* yet — those are out of subset and tested for
/// refusal in the unit tests). A small *deterministic* set of programs, not a statistical sample.
fn bit_corpus() -> Vec<Node> {
    let cst = |bits: [bool; 8]| Node::Const(byte(bits));
    vec![
        // bare constant
        cst(A),
        // core.id (passthrough)
        Node::Op {
            prim: "core.id".into(),
            args: vec![cst(A)],
        },
        // let / var alias
        Node::Let {
            id: "a".into(),
            bound: Box::new(cst(A)),
            body: Box::new(Node::Var("a".into())),
        },
        // each bit op
        Node::Op {
            prim: "bit.not".into(),
            args: vec![cst(A)],
        },
        Node::Op {
            prim: "bit.and".into(),
            args: vec![cst(A), cst(B)],
        },
        Node::Op {
            prim: "bit.or".into(),
            args: vec![cst(A), cst(B)],
        },
        Node::Op {
            prim: "bit.xor".into(),
            args: vec![cst(A), cst(ONES)], // xor with all-ones == complement
        },
        // nested: not(a xor b) through a let
        Node::Let {
            id: "x".into(),
            bound: Box::new(Node::Op {
                prim: "bit.xor".into(),
                args: vec![cst(A), cst(B)],
            }),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("x".into())],
            }),
        },
        // M-301 trit slice: balanced-ternary negation (a Ternary lane, end-to-end).
        Node::Op {
            prim: "trit.neg".into(),
            args: vec![Node::Const(tern(vec![
                Trit::Pos,
                Trit::Zero,
                Trit::Neg,
                Trit::Pos,
            ]))],
        },
        // trit.neg through a let / core.id passthrough on a ternary value.
        Node::Let {
            id: "t".into(),
            bound: Box::new(Node::Const(tern(vec![Trit::Neg, Trit::Neg, Trit::Pos]))),
            body: Box::new(Node::Op {
                prim: "core.id".into(),
                args: vec![Node::Op {
                    prim: "trit.neg".into(),
                    args: vec![Node::Var("t".into())],
                }],
            }),
        },
        // M-301 trit *carry* arithmetic (in range, so no overflow): add 5+4=9, sub 9-4=5, mul 2*3=6
        // over 3 trits. These exercise the ripple-carry / shifted-accumulate codegen end-to-end.
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
            ],
        },
        Node::Op {
            prim: "trit.sub".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Zero, Trit::Zero])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
            ],
        },
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
            ],
        },
        // nested arithmetic through a let: (5 + 4) - 4 = 5, mixing the adder and subtractor.
        Node::Let {
            id: "s".into(),
            bound: Box::new(Node::Op {
                prim: "trit.add".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
                ],
            }),
            body: Box::new(Node::Op {
                prim: "trit.sub".into(),
                args: vec![
                    Node::Var("s".into()),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
                ],
            }),
        },
    ]
}

/// Run a program through the interpreter; the bit subset uses no swaps, so an identity swap engine
/// suffices on the reference side.
fn interp_eval(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate the bit-subset corpus")
}

#[test]
fn interp_and_native_are_observably_equivalent_on_the_bit_corpus() {
    for (i, node) in bit_corpus().iter().enumerate() {
        let native = match mycelium_mlir::compile_and_run(node) {
            Ok(v) => v,
            // Environment skip: no native toolchain → cannot run the compiled path here.
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("program #{i}: native path errored: {e}"),
        };
        let interp = interp_eval(node);
        // Mutant-witness: if emit_op used the wrong LLVM instruction (e.g. `or` for `bit.and`), the
        // native payload would diverge from the interpreter's and this assertion would fail.
        assert_eq!(
            observable(&interp),
            observable(&native),
            "program #{i} diverged: interp {:?} vs native {:?}",
            interp.payload(),
            native.payload()
        );
        // M-210: the same pair validates through the single shared TV checker.
        assert_eq!(
            check(
                &interp,
                &native,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "program #{i}: the shared checker must validate the interp↔native pair"
        );
    }
}

/// M-301 overflow parity: a fixed-width balanced-ternary result out of range must be **refused** by
/// *both* the interpreter (`EvalError::Overflow`) and the native path (`AotError::Overflow`) — never
/// a silent wrap on either side (SC-3/G2). 4 + 4 = 8 overflows the 2-trit range (max magnitude 4).
#[test]
fn interp_and_native_agree_on_overflow_refusal() {
    let overflow = Node::Op {
        prim: "trit.add".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
            Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
        ],
    };
    // The interpreter refuses with an explicit overflow.
    let interp_err = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(&overflow);
    assert!(
        matches!(interp_err, Err(mycelium_interp::EvalError::Overflow { .. })),
        "interpreter must refuse the out-of-range sum, got {interp_err:?}"
    );
    // The native path computes the overflow at runtime and refuses through the read-back protocol.
    match mycelium_mlir::compile_and_run(&overflow) {
        Ok(v) => panic!(
            "native path silently wrapped the overflow: {:?}",
            v.payload()
        ),
        Err(AotError::Overflow(_)) => { /* expected — parity with the interpreter */ }
        Err(AotError::ToolchainMissing(_)) => { /* environment skip */ }
        Err(e) => panic!("native path errored unexpectedly: {e}"),
    }
}

/// Sanity: the compiled path actually discriminates — two different bit programs are NOT observably
/// equal and the shared checker reports the divergence explicitly (never a silent pass). So a passing
/// differential above is meaningful, not vacuous.
#[test]
fn native_differential_distinguishes_different_programs() {
    let not_a = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let id_a = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(byte(A))],
    };
    let (x, y) = match (
        mycelium_mlir::compile_and_run(&not_a),
        mycelium_mlir::compile_and_run(&id_a),
    ) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(AotError::ToolchainMissing(_)), _) | (_, Err(AotError::ToolchainMissing(_))) => return,
        (x, y) => panic!("native path errored: {x:?} / {y:?}"),
    };
    // not(A) != id(A) for a non-self-complementary A.
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
        "the checker must reject the genuinely divergent native pair, got {verdict:?}"
    );
}

// ─── M-373: Construct / Match data-fragment differential (Increment-1) ────────────────────────

/// Build the shared `DataRegistry` for the data corpus: `type Box = Mk(Binary{8})` (a single
/// constructor wrapping one 8-bit field). Non-recursive — no `FieldSpec::Data` back-reference —
/// so this is firmly within the DN-15 §4.1 Increment-1 subset.
fn box_registry() -> DataRegistry {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Box".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec {
                fields: vec![FieldSpec::Repr(Repr::Binary { width: 8 })],
            }],
        },
    );
    DataRegistry::build(&specs).expect("Box registry must build")
}

/// Build the shared `DataRegistry` for a two-constructor type: `type Color = Red | Blue`.
/// Both constructors carry no fields — the tag alone is the payload. Used to exercise the
/// `switch i64` dispatch with two arms, one of which produces `A` and the other `B`.
fn color_registry() -> DataRegistry {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Color".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] }, // Red  (tag 0)
                CtorSpec { fields: vec![] }, // Blue (tag 1)
            ],
        },
    );
    DataRegistry::build(&specs).expect("Color registry must build")
}

/// The data-fragment corpus (M-373 Increment-1): non-recursive `Construct`/`Match` programs whose
/// final result is a repr lane (bit vector). Each program is valid under both the interpreter
/// (`Interpreter::eval`) and the LLVM data-fragment path (`compile_and_run`).
///
/// Guarantee: `Declared` — hand-written IR lowering, empirically validated by the differential
/// (VR-5: never upgraded to `Proven` without a checked proof).
fn data_corpus() -> Vec<Node> {
    let reg = box_registry();
    let col = color_registry();
    let mk_box = |bits: [bool; 8]| Node::Construct {
        ctor: reg.ctor_ref("Box", 0).unwrap(),
        args: vec![Node::Const(byte(bits))],
    };
    let red = || Node::Construct {
        ctor: col.ctor_ref("Color", 0).unwrap(),
        args: vec![],
    };
    let blue = || Node::Construct {
        ctor: col.ctor_ref("Color", 1).unwrap(),
        args: vec![],
    };

    vec![
        // 1. Construct Box(A), match to extract the inner field b → return b unchanged.
        //    Tests: stack alloca for a 1-field type; tag-load + switch dispatch; field-load + phi.
        Node::Match {
            scrutinee: Box::new(mk_box(A)),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                binders: vec!["b".to_owned()],
                body: Node::Var("b".to_owned()),
            }],
            default: None,
        },
        // 2. Construct Box(A), match and apply bit.not to the extracted field.
        //    Tests: bit op in a match arm body; arm body uses a binder (not just a constant).
        Node::Match {
            scrutinee: Box::new(mk_box(A)),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                binders: vec!["b".to_owned()],
                body: Node::Op {
                    prim: "bit.not".into(),
                    args: vec![Node::Var("b".to_owned())],
                },
            }],
            default: None,
        },
        // 3. Let-bound Construct, then match. Tests that a Construct result in the env (Datum)
        //    can be looked up as the scrutinee of a later Match — the full env-lookup path.
        Node::Let {
            id: "box_a".into(),
            bound: Box::new(mk_box(A)),
            body: Box::new(Node::Match {
                scrutinee: Box::new(Node::Var("box_a".into())),
                alts: vec![Alt::Ctor {
                    ctor: reg.ctor_ref("Box", 0).unwrap(),
                    binders: vec!["b".to_owned()],
                    body: Node::Op {
                        prim: "bit.and".into(),
                        args: vec![Node::Var("b".to_owned()), Node::Const(byte(B))],
                    },
                }],
                default: None,
            }),
        },
        // 4. Two-constructor Color type: match Red → return A; match Blue → return B.
        //    Tests the switch with two real arms (the phi merge collects two (label, Lane) pairs).
        Node::Match {
            scrutinee: Box::new(red()),
            alts: vec![
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 0).unwrap(), // Red
                    binders: vec![],
                    body: Node::Const(byte(A)),
                },
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 1).unwrap(), // Blue
                    binders: vec![],
                    body: Node::Const(byte(B)),
                },
            ],
            default: None,
        },
        // 5. Same two-constructor Color type but select Blue → return B (mutant-witness that the
        //    switch dispatches on the correct tag, not always on arm 0).
        Node::Match {
            scrutinee: Box::new(blue()),
            alts: vec![
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 0).unwrap(), // Red
                    binders: vec![],
                    body: Node::Const(byte(A)),
                },
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 1).unwrap(), // Blue
                    binders: vec![],
                    body: Node::Const(byte(B)),
                },
            ],
            default: None,
        },
    ]
}

/// Evaluate a `data_corpus` program through the reference interpreter, returning a repr `Value`.
/// Programs in the corpus always reduce to a repr value (never a datum), so `eval` suffices.
fn interp_eval_core(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate every data_corpus program to a repr value")
}

/// M-373 Increment-1: interp ↔ native are observably equivalent on the `data_corpus`.
///
/// Guarantee: `Declared` — the differential is empirical evidence (VR-5). The design rationale
/// (stack `alloca`, `switch i64`, `@abort()` default) is in `llvm.rs` §M-373 / DN-15 §4.1.
#[test]
fn interp_and_native_are_observably_equivalent_on_the_data_corpus() {
    for (i, node) in data_corpus().iter().enumerate() {
        let native = match mycelium_mlir::compile_and_run(node) {
            Ok(v) => v,
            // Environment skip: no native toolchain → cannot run the compiled path here.
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("data program #{i}: native path errored: {e}"),
        };
        let interp = interp_eval_core(node);
        // Mutant-witness: if the Construct stored the wrong tag, or Match loaded from the wrong
        // slot, or the phi merge picked the wrong arm, the payloads would diverge here.
        assert_eq!(
            observable(&interp),
            observable(&native),
            "data program #{i} diverged: interp {:?} vs native {:?}",
            interp.payload(),
            native.payload()
        );
        // M-210: the same pair validates through the single shared TV checker.
        assert_eq!(
            check(
                &interp,
                &native,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "data program #{i}: the shared checker must validate the interp↔native pair"
        );
    }
}

// ─── M-378: closure (App/Lam) + heap differential (Increment-2) ───────────────────────────────

/// The closure corpus (M-378 Increment-2; DN-15 §7 / RFC-0004 §11.5): `App`/`Lam` programs over the
/// **narrow `Binary{8}`-packed-`i64` closure ABI**. Each program applies a closure fully and reduces
/// to a bit vector, so it is valid under both the interpreter (full v0 calculus) and the LLVM
/// closure path (`compile_and_run`, free-var analysis → heap closure record → indirect call).
///
/// Guarantee: `Declared` — hand-written IR lowering, empirically validated by the differential (VR-5:
/// never upgraded to `Proven` without a checked proof).
fn closure_corpus() -> Vec<Node> {
    // λx. <op over x and named vars>, applied to `arg`.
    let lam = |param: &str, body: Node| Node::Lam {
        param: param.to_owned(),
        body: Box::new(body),
    };
    let app = |f: Node, a: Node| Node::App {
        func: Box::new(f),
        arg: Box::new(a),
    };
    let var = |x: &str| Node::Var(x.to_owned());
    let op2 = |prim: &str, a: Node, b: Node| Node::Op {
        prim: prim.into(),
        args: vec![a, b],
    };
    let not = |a: Node| Node::Op {
        prim: "bit.not".into(),
        args: vec![a],
    };
    let let_ = |id: &str, bound: Node, body: Node| Node::Let {
        id: id.to_owned(),
        bound: Box::new(bound),
        body: Box::new(body),
    };

    vec![
        // 1. Identity: (λx. x) A → A. The minimal closure (no captures) + indirect call.
        app(lam("x", var("x")), Node::Const(byte(A))),
        // 2. Capture + xor: let y = B in (λx. x ⊕ y) A → A ⊕ B. One captured free var.
        let_(
            "y",
            Node::Const(byte(B)),
            app(
                lam("x", op2("bit.xor", var("x"), var("y"))),
                Node::Const(byte(A)),
            ),
        ),
        // 3. Capture + and: let y = A in (λx. x ∧ y) B → A ∧ B (capture distinct from the argument).
        let_(
            "y",
            Node::Const(byte(A)),
            app(
                lam("x", op2("bit.and", var("x"), var("y"))),
                Node::Const(byte(B)),
            ),
        ),
        // 4. Two captures: let y = A in let z = B in (λx. (x ⊕ y) ∨ z) ONES. Record with k = 2.
        let_(
            "y",
            Node::Const(byte(A)),
            let_(
                "z",
                Node::Const(byte(B)),
                app(
                    lam(
                        "x",
                        op2("bit.or", op2("bit.xor", var("x"), var("y")), var("z")),
                    ),
                    Node::Const(byte(ONES)),
                ),
            ),
        ),
        // 5. Const inside the body, no capture: (λx. ¬(x ⊕ A)) B → ¬(B ⊕ A). Body-local const stays
        //    body-local (not a capture) — exercises the empty-capture record with a non-trivial body.
        app(
            lam("x", not(op2("bit.xor", var("x"), Node::Const(byte(A))))),
            Node::Const(byte(B)),
        ),
        // 6. Closure result feeds an enclosing op: let y = A in ¬((λx. x ∧ y) B) → ¬(B ∧ A).
        //    The App result (unpacked lane) flows into a following `bit.not`.
        let_(
            "y",
            Node::Const(byte(A)),
            not(app(
                lam("x", op2("bit.and", var("x"), var("y"))),
                Node::Const(byte(B)),
            )),
        ),
        // 7. Nested capturing lambda, applied inside the body: (λx. let p = x ⊕ B in
        //    (λw. w ⊕ p) p) A → (A⊕B) ⊕ (A⊕B) = 0. The body-local const `B` pins the (M-851 boxed-ABI)
        //    param shape via `x ⊕ B`, so `p` is a concretely-typed `Binary{8}` lane; the inner closure
        //    captures `p`, is allocated + called within the outer body, and returns `Binary{8}`.
        //    Exercises recursion of the closure-conversion machinery (free-var analysis descending into
        //    a nested lambda; a record built inside a closure function) — the boundary case the widened
        //    ABI resolves end-to-end. (The *fully* polymorphic nested-capture `(λx. (λw. w⊕x) x)` —
        //    with no body-local const to pin the param shape — also lowers under M-851, because the
        //    call-site argument pins `x`'s shape at inlining time; see `nested_capture_of_outer_param_lowers`
        //    and `closure_widening_differential::nested_capture_of_outer_param`. The corpus case here uses
        //    the let-`p` form only to cross-cover the in-body-binding path — not because the polymorphic
        //    form is refused; G2/VR-5: no guessed width in either form.)
        app(
            lam(
                "x",
                let_(
                    "p",
                    op2("bit.xor", var("x"), Node::Const(byte(B))),
                    app(lam("w", op2("bit.xor", var("w"), var("p"))), var("p")),
                ),
            ),
            Node::Const(byte(A)),
        ),
    ]
}

/// Evaluate a `closure_corpus` program through the reference interpreter (the oracle). Each program
/// applies its closures fully, reducing to a repr `Value`, so `eval` suffices.
fn interp_eval_closure(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate every closure_corpus program to a repr value")
}

/// M-378 Increment-2: interp ↔ native are observably equivalent on the `closure_corpus`.
///
/// The gate (NFR-7): closures lowered to heap records + indirect calls in textual LLVM IR must agree
/// with the reference interpreter, element for element, through the single shared M-210 checker.
/// Guarantee: `Declared` (the differential is empirical evidence — VR-5; DN-15 §7).
#[test]
fn interp_and_native_are_observably_equivalent_on_the_closure_corpus() {
    for (i, node) in closure_corpus().iter().enumerate() {
        let native = match mycelium_mlir::compile_and_run(node) {
            Ok(v) => v,
            // Environment skip: no native toolchain → cannot run the compiled path here.
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("closure program #{i}: native path errored: {e}"),
        };
        let interp = interp_eval_closure(node);
        // Mutant-witness: if the closure captured the wrong slot, the indirect call passed the env
        // and arg in the wrong order, or pack/unpack mis-encoded the bits, the payloads would diverge.
        assert_eq!(
            observable(&interp),
            observable(&native),
            "closure program #{i} diverged: interp {:?} vs native {:?}",
            interp.payload(),
            native.payload()
        );
        // M-210: the same pair validates through the single shared TV checker.
        assert_eq!(
            check(
                &interp,
                &native,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "closure program #{i}: the shared checker must validate the interp↔native pair"
        );
    }
}

/// Sanity: the closure path actually discriminates — two closures with the *same* capture but
/// different bodies (`x ⊕ y` vs `x ∧ y`) produce different results, and the shared checker reports
/// the divergence (never a vacuous pass). Guards specifically the closure machinery.
#[test]
fn native_closure_differential_distinguishes_different_bodies() {
    let mk = |prim: &str| Node::Let {
        id: "y".to_owned(),
        bound: Box::new(Node::Const(byte(A))),
        body: Box::new(Node::App {
            func: Box::new(Node::Lam {
                param: "x".to_owned(),
                body: Box::new(Node::Op {
                    prim: prim.into(),
                    args: vec![Node::Var("x".to_owned()), Node::Var("y".to_owned())],
                }),
            }),
            arg: Box::new(Node::Const(byte(B))),
        }),
    };
    let (x, y) = match (
        mycelium_mlir::compile_and_run(&mk("bit.xor")),
        mycelium_mlir::compile_and_run(&mk("bit.and")),
    ) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(AotError::ToolchainMissing(_)), _) | (_, Err(AotError::ToolchainMissing(_))) => return,
        (x, y) => panic!("native closure path errored: {x:?} / {y:?}"),
    };
    assert_ne!(
        observable(&x),
        observable(&y),
        "A⊕B and A∧B must differ for these A/B"
    );
    let verdict = check(
        &x,
        &y,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    );
    assert!(
        matches!(verdict, CheckVerdict::NotValidated { .. }),
        "the checker must reject the divergent closure pair, got {verdict:?}"
    );
}

/// Refusal parity after the **M-851 widening**: even with closures over any repr/width, currying, and
/// returned closures now lowering, the native path must still refuse — explicitly, never silently
/// (G2) — a **closure-valued program result** (a closure is not printable by the read-back protocol)
/// and a degenerate `Fix`. A bare `Lam`, and a curried `(λx.λy.x) A` (whose result is the inner
/// closure `λy.x`), both reduce to a closure value at the program result and so are refused **at the
/// result** (DN-15 §7.4) — note the boundary moved: currying itself now lowers; only a closure left
/// on the *printable program result* is refused.
#[test]
fn closure_valued_program_result_and_degenerate_fix_are_refused() {
    // A bare Lam: lowers to a closure value, which cannot be the printable program result.
    let bare_lam = Node::Lam {
        param: "x".to_owned(),
        body: Box::new(Node::Var("x".to_owned())),
    };
    // Currying *now lowers*, but its result here IS a closure (`λy. x`), refused as the program result.
    let curry = Node::App {
        func: Box::new(Node::Lam {
            param: "x".to_owned(),
            body: Box::new(Node::Lam {
                param: "y".to_owned(),
                body: Box::new(Node::Var("x".to_owned())),
            }),
        }),
        arg: Box::new(Node::Const(byte(A))),
    };
    // Self-referential degenerate Fix — not a `λparam.Match` recursion shape.
    let fix = Node::Fix {
        name: "f".to_owned(),
        body: Box::new(Node::Var("f".to_owned())),
    };
    for (label, node) in [("bare Lam", &bare_lam), ("curry", &curry), ("Fix", &fix)] {
        match mycelium_mlir::compile_and_run(node) {
            Err(AotError::UnsupportedNode(_)) => { /* expected explicit refusal */ }
            Err(AotError::ToolchainMissing(_)) => { /* environment skip */ }
            Ok(v) => panic!(
                "{label} must be refused; native path returned {:?}",
                v.payload()
            ),
            Err(e) => panic!("{label} errored with an unexpected variant: {e}"),
        }
    }
}

/// M-851: a value crossing the closure boundary that is **not** `Binary{8}` now **lowers** — the
/// widened uniform pointer-boxed ABI carries any repr/width. Here the identity closure applied to a
/// `Ternary` argument returns that ternary value, value-equal to the interpreter (the refusal this
/// replaced — `closures_over_non_binary8_values_are_explicitly_refused` — is removed by the widening).
#[test]
fn closures_over_ternary_values_now_lower() {
    // (λx. x) applied to a balanced-ternary value — inside the widened boxed-value closure ABI.
    let t = tern(vec![Trit::Pos, Trit::Zero, Trit::Neg]);
    let prog = Node::App {
        func: Box::new(Node::Lam {
            param: "x".to_owned(),
            body: Box::new(Node::Var("x".to_owned())),
        }),
        arg: Box::new(Node::Const(t.clone())),
    };
    let native = match mycelium_mlir::compile_and_run(&prog) {
        Ok(v) => v,
        Err(AotError::ToolchainMissing(_)) => return, // env skip
        Err(e) => panic!("ternary-argument identity closure must lower now; errored: {e}"),
    };
    // The identity over a ternary value returns the same ternary value (Mutant-witness: a mis-encoded
    // box header or a wrong sext/trunc would diverge from the interpreter's oracle).
    assert_eq!(
        native.payload(),
        t.payload(),
        "identity over ternary must echo the input"
    );
    assert_eq!(native.repr(), t.repr());
}

/// M-851 specialize-at-application: a **nested capture of the outer param** — `(λx. (λw. w ⊕ x) x) A
/// → A ⊕ A = 0` — lowers natively. The inner closure captures the outer parameter `x` and is applied
/// to it; the inlining resolves it fully (the call-site argument `A` pins `x`, the inner closure
/// inlines with `w ← x`), so the shape flows in directly with no guessed width. interp ≡ native.
#[test]
fn nested_capture_of_outer_param_lowers() {
    let prog = Node::App {
        func: Box::new(Node::Lam {
            param: "x".to_owned(),
            body: Box::new(Node::App {
                func: Box::new(Node::Lam {
                    param: "w".to_owned(),
                    body: Box::new(Node::Op {
                        prim: "bit.xor".into(),
                        args: vec![Node::Var("w".to_owned()), Node::Var("x".to_owned())],
                    }),
                }),
                arg: Box::new(Node::Var("x".to_owned())),
            }),
        }),
        arg: Box::new(Node::Const(byte(A))),
    };
    let native = match mycelium_mlir::compile_and_run(&prog) {
        Ok(v) => v,
        Err(AotError::ToolchainMissing(_)) => return, // env skip
        Err(e) => panic!("nested capture of outer param must lower; errored: {e}"),
    };
    let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(&prog)
        .expect("interp must evaluate the nested-capture program");
    assert_eq!(
        observable(&interp),
        observable(&native),
        "nested capture: interp {:?} vs native {:?}",
        interp.payload(),
        native.payload()
    );
    // A ⊕ A == 0.
    assert_eq!(
        native.payload(),
        &Payload::Bits(vec![false; 8]),
        "A ⊕ A must be 0"
    );
}

/// M-851 capture boundary (DN-15 §7.4; G2): a **`Datum`** (constructed data value) may *not* be
/// captured by a closure in the widened ABI — capturing it would require runtime closure dispatch
/// (a vtable / tagged-union wire type) that is a separate future increment. The refusal is
/// **explicit** (`UnsupportedNode`) and **test-pinned** here so the boundary never silently shifts.
///
/// Pattern: `λx. (λy. <expr using constructed datum d>) v` where the inner closure captures `d`.
/// The outer closure body constructs a `Box(A)`; the inner `λy.` tries to capture it. `lower_lam`
/// refuses that at capture time — never at App time (G2: the refusal is at the well-defined boundary,
/// not silently deferred until a mis-encode would produce wrong output).
#[test]
fn datum_capture_by_closure_is_explicitly_refused() {
    // Construct a Box(A) datum, then try to capture it in a closure. The closure never applies it,
    // so the refusal is purely at capture time — not because of what the body does with it.
    let reg = box_registry();
    let box_ctor = reg.ctor_ref("Box", 0).unwrap();
    // Program: (λouter. let d = Box(outer) in (λinner. outer) A) A
    // Here the inner closure captures `d` (a Datum) — must refuse at lower_lam time.
    let prog = Node::App {
        func: Box::new(Node::Lam {
            param: "outer".to_owned(),
            body: Box::new(Node::Let {
                id: "d".to_owned(),
                bound: Box::new(Node::Construct {
                    ctor: box_ctor,
                    args: vec![Node::Var("outer".to_owned())],
                }),
                body: Box::new(Node::App {
                    func: Box::new(Node::Lam {
                        param: "inner".to_owned(),
                        body: Box::new(Node::Var("d".to_owned())), // captures datum `d`
                    }),
                    arg: Box::new(Node::Const(byte(A))),
                }),
            }),
        }),
        arg: Box::new(Node::Const(byte(A))),
    };
    match mycelium_mlir::compile_and_run(&prog) {
        Err(AotError::UnsupportedNode(msg)) => {
            // The error must name the offending capture — never a silent mis-encode.
            assert!(
                msg.contains("datum") || msg.contains("data value"),
                "datum-capture refusal message must name the issue; got: {msg}"
            );
        }
        Err(AotError::ToolchainMissing(_)) => { /* env skip — the refusal is at lower_lam, before llc */
        }
        Ok(v) => panic!("datum capture must be refused; got value {:?}", v.payload()),
        Err(e) => panic!("datum capture refused with unexpected variant: {e}"),
    }
}

// ─── M-379: Binary branch primitive (Match Lit-arms on a Binary lane) ─────────────────────────

/// The Binary-branch corpus (M-379 Increment-3; DN-15 §8.3): `Match` over a `Binary{8}` *lane*
/// scrutinee with `Lit` arms — the native branch primitive (pack the lane, `switch i64` on the packed
/// literals). Distinct from the Increment-1 `Match` over a `Datum` scrutinee with `Ctor` arms. Each
/// program reduces to a bit vector and is value-checked interp ≡ native.
fn binary_branch_corpus() -> Vec<Node> {
    // Match `scrut` against a single `Lit` pattern → `hit` on match, else `miss` (the default).
    let lit_match =
        |scrut: [bool; 8], pat: [bool; 8], hit: [bool; 8], miss: [bool; 8]| Node::Match {
            scrutinee: Box::new(Node::Const(byte(scrut))),
            alts: vec![Alt::Lit {
                value: byte(pat),
                body: Node::Const(byte(hit)),
            }],
            default: Some(Box::new(Node::Const(byte(miss)))),
        };
    vec![
        // 1. scrutinee A == pattern A → take the Lit arm (B).
        lit_match(A, A, B, ONES),
        // 2. scrutinee B != pattern A → fall through to the default (ONES). Mutant-witness that the
        //    branch compares the value, not always-takes the first arm.
        lit_match(B, A, B, ONES),
        // 3. Two Lit arms: the scrutinee matches the *second* pattern (dispatch on the right value).
        Node::Match {
            scrutinee: Box::new(Node::Const(byte(B))),
            alts: vec![
                Alt::Lit {
                    value: byte(A),
                    body: Node::Const(byte(ONES)),
                },
                Alt::Lit {
                    value: byte(B),
                    body: Node::Const(byte(A)),
                },
            ],
            default: Some(Box::new(Node::Const(byte(ONES)))),
        },
    ]
}

/// M-379 Increment-3: interp ↔ native are observably equivalent on the `binary_branch_corpus`.
/// Guarantee: `Declared` — the differential is empirical evidence (VR-5; DN-15 §8).
#[test]
fn interp_and_native_are_observably_equivalent_on_the_binary_branch_corpus() {
    for (i, node) in binary_branch_corpus().iter().enumerate() {
        let native = match mycelium_mlir::compile_and_run(node) {
            Ok(v) => v,
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("binary-branch program #{i}: native path errored: {e}"),
        };
        let interp = interp_eval_closure(node);
        // Mutant-witness: a wrong literal compare (or always taking arm 0) would diverge here.
        assert_eq!(
            observable(&interp),
            observable(&native),
            "binary-branch program #{i} diverged: interp {:?} vs native {:?}",
            interp.payload(),
            native.payload()
        );
        assert_eq!(
            check(
                &interp,
                &native,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "binary-branch program #{i}: the shared checker must validate the interp↔native pair"
        );
    }
}

// ─── Regression: Issue 1 fix — Match default arm taken (PR #213 review) ────────────────────────

/// Build a three-constructor registry: `type Signal = A | B | C`.
/// All three constructors carry no fields — tag alone is the discriminant.
fn signal_registry() -> DataRegistry {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Signal".to_owned(),
        DeclSpec {
            ctors: vec![
                CtorSpec { fields: vec![] }, // A (tag 0)
                CtorSpec { fields: vec![] }, // B (tag 1)
                CtorSpec { fields: vec![] }, // C (tag 2)
            ],
        },
    );
    DataRegistry::build(&specs).expect("Signal registry must build")
}

/// Regression test for Issue 1 (PR #213 / Copilot review): the ANF `Match` `default` arm must be
/// **lowered into the switch's default block**, not silently replaced by `abort()`.
///
/// Program structure:
///   match C {
///     A → const B-bits        (explicit arm — tag 0, not taken)
///     | default → const A-bits  (ANF default — taken because scrutinee is C / tag 2)
///   }
///
/// **Before the fix:** the native path would call `abort()` when the `default` arm was taken
/// (tag 2 hits the switch default ⇒ abort ⇒ `AotError::Run`), while the interpreter returned
/// the default body's value. That was a **silent semantic divergence**.
///
/// **After the fix:** both paths return the default body's value (`A` bits) and the M-210
/// checker reports `Validated { strength: Exact }`.
///
/// Guarantee: `Declared` (same as the rest of the Increment-1 data fragment; VR-5).
#[test]
fn match_default_arm_is_taken_and_observationally_equivalent() {
    let sig = signal_registry();
    // Construct C (tag 2) — the scrutinee. No explicit arm for tag 2.
    let construct_c = Node::Construct {
        ctor: sig.ctor_ref("Signal", 2).unwrap(),
        args: vec![],
    };
    // Match: one explicit arm for A (tag 0) → B-bits; default → A-bits.
    // The scrutinee is C (tag 2), so the explicit arm is NOT taken and the default IS taken.
    let program = Node::Match {
        scrutinee: Box::new(construct_c),
        alts: vec![Alt::Ctor {
            ctor: sig.ctor_ref("Signal", 0).unwrap(), // A — not taken
            binders: vec![],
            body: Node::Const(byte(B)), // would return B-bits if taken
        }],
        default: Some(Box::new(Node::Const(byte(A)))), // taken → returns A-bits
    };

    // Run through the reference interpreter.
    let interp_val = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(&program)
        .expect("interpreter must evaluate the default-arm Match to a repr value");

    // Interpreter must return A-bits (the default body), not B-bits.
    assert_eq!(
        interp_val.payload(),
        &Payload::Bits(A.to_vec()),
        "interpreter must return the default arm's value (A-bits) when scrutinee tag is unmatched"
    );

    // Run through the native compiled path.
    let native_val = match mycelium_mlir::compile_and_run(&program) {
        Ok(v) => v,
        // Environment skip: no native toolchain → cannot run the compiled path here.
        Err(AotError::ToolchainMissing(_)) => return,
        // Before the fix: native would abort() here (AotError::Run). After the fix: it must not.
        Err(e) => panic!(
            "native path errored when the Match default arm was taken: {e}\n\
             (Before the PR-213 fix this would be an AotError::Run from abort(); \
             after the fix the default block must be lowered correctly.)"
        ),
    };

    // Native must return the same A-bits as the interpreter.
    assert_eq!(
        observable(&interp_val),
        observable(&native_val),
        "interp and native diverged on the default arm: interp {:?} vs native {:?}",
        interp_val.payload(),
        native_val.payload()
    );

    // M-210: the pair validates through the single shared TV checker.
    assert_eq!(
        check(
            &interp_val,
            &native_val,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "the shared checker must validate the interp↔native pair when the default arm is taken"
    );
}
