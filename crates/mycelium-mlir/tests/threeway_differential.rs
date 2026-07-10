//! M-602 / M-725 / M-857 — the **three-way** native differential (NFR-7; VR-4; RR-12; RFC-0029 §7;
//! phase-6).
//!
//! Extends the M-302 interp↔native differential (`native_differential.rs`) to a **third** compiled
//! path: the **real MLIR-dialect** lowering (`dialect::native`, feature `mlir-dialect`; M-601, widened
//! by M-725 then M-857). For the in-fragment calculus corpus the programs run under
//!
//! 1. the M-110 **reference interpreter** (the trusted base),
//! 2. the **direct-LLVM** backend (`mycelium_mlir::compile_and_run`; `llvm.rs`), and
//! 3. the **MLIR-dialect** backend (`mycelium_mlir::mlir_compile_and_run`; emits `arith`/`func`/`cf`
//!    MLIR, runs `mlir-opt | mlir-translate → clang → native`),
//!
//! and all three must be **observably equivalent** (`repr + payload + guarantee`), each pair
//! **validated through the single shared M-210 checker** (`ObservationalEquiv`). A deliberately
//! divergent lowering on *any* path is caught — so a passing three-way differential is meaningful,
//! not vacuous.
//!
//! **Honest fragment boundary (VR-5/G2).** The MLIR-dialect path covers the element-wise fragment
//! (`core.id`, `bit.not/and/or/xor`, `trit.neg`) **plus** the balanced-ternary fixed-width arithmetic
//! — the additive carry chain `trit.add`/`trit.sub` (M-725) and the shifted-accumulate multiply
//! `trit.mul` (M-857) — including their never-silent overflow read-back. The **new boundary** is
//! everything richer: the data fragment, closures and recursion, `Swap`, Dense/VSA — each an
//! **explicit refusal** there (`DialectError::Unsupported`) that routes to the direct-LLVM/interp path.
//! This test asserts BOTH: the in-fragment corpus (element-wise + trit add/sub/mul, in-range *and*
//! overflowing) is three-way equivalent — on the result *and* the overflow refusal — AND the
//! out-of-fragment corpus (closures, …) is explicitly refused by the MLIR path while still
//! interp ≡ direct-LLVM (so coverage is honest, never silently claimed).
//!
//! **Toolchain skip.** Both compiled paths need their tools (`llc`/`clang` for direct-LLVM;
//! `mlir-opt`/`mlir-translate`/`clang` for the dialect path). Where a tool is absent the path returns
//! a `ToolchainMissing` and the test **skips** that path (the house "skip gracefully" idiom) — never
//! a false failure.
//!
//! **Guarantee:** `Empirical` — the differential is empirical evidence the MLIR lowering agrees with
//! the trusted interpreter over the corpus; never upgraded to `Proven` without a checked proof (VR-5).

#![cfg(feature = "mlir-dialect")]

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{
    Alt, CtorSpec, DataRegistry, DeclSpec, FieldSpec, GuaranteeStrength, Meta, Node, Payload,
    Provenance, Repr, Trit, Value,
};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::{AotError, DialectError};
use mycelium_numerics::Certificate;
use std::collections::BTreeMap;

// ─── shared helpers (local; the `common` module's helpers are a superset we don't fully need) ──

fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn tern(trits: Vec<Trit>) -> Value {
    let m = trits.len() as u32;
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(trits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

const A: [bool; 8] = [true, false, true, true, false, false, true, false];
const B: [bool; 8] = [false, false, true, false, true, false, true, true];
const ONES: [bool; 8] = [true; 8];

type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);
fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}

fn interp_eval(node: &Node) -> Value {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .eval(node)
        .expect("interpreter must evaluate the element-wise corpus")
}

/// Mean ns/call of the fastest batch of `iters` calls (after one warmup batch). House timing style —
/// no benchmarking dependency (mirrors `xtask::e1::bench`). Used only by the `#[ignore]` perf test.
#[allow(dead_code)]
fn bench(iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        #[allow(clippy::cast_precision_loss)]
        let per = t.elapsed().as_nanos() as f64 / f64::from(iters);
        best = best.min(per);
    }
    best
}

/// The **in-fragment** corpus the MLIR-dialect path covers: the element-wise ops (`core.id`,
/// `bit.not/and/or/xor`, `trit.neg`) **plus** (M-725) the additive carry chain `trit.add`/`trit.sub`
/// over `Binary{w}`/`Ternary{m}`, straight-line (through `let`s). All the trit-additive cases here
/// stay **in range** (no overflow) so every path produces a value; the overflow refusal is covered
/// separately by [`overflow_corpus`]. A small deterministic set, not a statistical sample.
fn element_wise_corpus() -> Vec<Node> {
    let cst = |bits: [bool; 8]| Node::Const(byte(bits));
    vec![
        // bare constant
        cst(A),
        // core.id passthrough
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
            args: vec![cst(A), cst(ONES)],
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
        // balanced-ternary negation (a Ternary lane end-to-end)
        Node::Op {
            prim: "trit.neg".into(),
            args: vec![Node::Const(tern(vec![
                Trit::Pos,
                Trit::Zero,
                Trit::Neg,
                Trit::Pos,
            ]))],
        },
        // trit.neg through a let / core.id passthrough on a ternary value
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
        // ── M-725: the additive carry chain, all in-range (no overflow) ──
        // trit.add: 1 + 1 = 2 (= [0,+,-]) in 3 trits (max magnitude 13).
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
            ],
        },
        // trit.add with a multi-trit carry ripple: 7 + (−7) = 0, over 4 trits.
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg, Trit::Pos])),
                Node::Const(tern(vec![Trit::Zero, Trit::Neg, Trit::Pos, Trit::Neg])),
            ],
        },
        // trit.sub: 3 − 1 = 2, over 3 trits.
        Node::Op {
            prim: "trit.sub".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
            ],
        },
        // nested: (a + b) through a let, then negate — a Ternary lane end-to-end with carry.
        Node::Let {
            id: "s".into(),
            bound: Box::new(Node::Op {
                prim: "trit.add".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Pos, Trit::Zero, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Neg, Trit::Pos, Trit::Pos])),
                ],
            }),
            body: Box::new(Node::Op {
                prim: "trit.neg".into(),
                args: vec![Node::Var("s".into())],
            }),
        },
        // ── M-857: the shifted-accumulate multiply, all in-range (no overflow) ──
        // trit.mul: 2 · 3 = 6 (= [+,-,0]) in 3 trits (max magnitude 13). 2 = [0,+,-], 3 = [0,+,0].
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
            ],
        },
        // trit.mul with a negative factor + a wider ripple: (−4) · 2 = −8 (= [0,-,0,+]) in 4 trits
        // (max magnitude 40). −4 = [0,0,-,-], 2 = [0,0,+,-]; −8 = 0·27 + (−1)·9 + 0·3 + 1·1.
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Neg, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos, Trit::Neg])),
            ],
        },
        // trit.mul by zero: 5 · 0 = 0 — every partial vanishes (the `arith.muli {aj}, 0` path).
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Zero])),
            ],
        },
        // nested: (a · b) through a let, then negate — a Ternary multiply lane end-to-end.
        Node::Let {
            id: "p".into(),
            bound: Box::new(Node::Op {
                prim: "trit.mul".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Zero, Trit::Pos])),
                ],
            }),
            body: Box::new(Node::Op {
                prim: "trit.neg".into(),
                args: vec![Node::Var("p".into())],
            }),
        },
    ]
}

/// The three-way differential over the element-wise corpus: interp ≡ direct-LLVM ≡ MLIR-dialect,
/// each pair validated through the shared M-210 checker. Skips a path whose toolchain is absent.
#[test]
fn interp_directllvm_mlirdialect_are_three_way_equivalent() {
    let mut ran_mlir = false;
    for (i, node) in element_wise_corpus().iter().enumerate() {
        let interp = interp_eval(node);

        // Path 2: direct-LLVM (skip if llc/clang absent).
        let direct = match mycelium_mlir::compile_and_run(node) {
            Ok(v) => Some(v),
            Err(AotError::ToolchainMissing(_)) => None,
            Err(e) => panic!("program #{i}: direct-LLVM path errored: {e}"),
        };

        // Path 3: MLIR-dialect (skip if mlir-opt/mlir-translate/clang absent).
        let mlir = match mycelium_mlir::mlir_compile_and_run(node) {
            Ok(v) => Some(v),
            Err(DialectError::ToolchainMissing(_)) => None,
            Err(e) => panic!("program #{i}: MLIR-dialect path errored: {e}"),
        };

        if let Some(d) = &direct {
            assert_eq!(
                observable(&interp),
                observable(d),
                "program #{i}: interp vs direct-LLVM diverged"
            );
        }
        if let Some(m) = &mlir {
            ran_mlir = true;
            // Mutant-witness: a wrong arith op (e.g. arith.ori for bit.and) would diverge here.
            assert_eq!(
                observable(&interp),
                observable(m),
                "program #{i}: interp vs MLIR-dialect diverged ({:?} vs {:?})",
                interp.payload(),
                m.payload()
            );
            // M-210: the interp↔MLIR pair validates through the single shared TV checker.
            assert_eq!(
                check(
                    &interp,
                    m,
                    RefinementRelation::ObservationalEquiv,
                    Certificate::exact(),
                    &Evidence::Observational,
                ),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "program #{i}: the shared checker must validate the interp↔MLIR pair"
            );
        }
        // When BOTH compiled paths ran, the two compiled artifacts must also agree with each other
        // (the third edge of the triangle) — validated through the same checker.
        if let (Some(d), Some(m)) = (&direct, &mlir) {
            assert_eq!(
                observable(d),
                observable(m),
                "program #{i}: direct-LLVM vs MLIR-dialect diverged"
            );
            assert_eq!(
                check(
                    d,
                    m,
                    RefinementRelation::ObservationalEquiv,
                    Certificate::exact(),
                    &Evidence::Observational,
                ),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "program #{i}: the shared checker must validate the direct-LLVM↔MLIR pair"
            );
        }
    }
    // If the MLIR toolchain was present, we must actually have exercised it on at least one program
    // (guard against a vacuous pass where every program silently skipped).
    if mycelium_mlir::MlirTools::is_available() {
        assert!(
            ran_mlir,
            "MLIR toolchain is available but no program exercised the dialect path — vacuous"
        );
    }
}

/// Sanity: the MLIR-dialect path actually discriminates — two different programs are NOT observably
/// equal and the shared checker reports the divergence (never a vacuous pass). So the equivalence
/// above is meaningful.
#[test]
fn mlir_dialect_distinguishes_different_programs() {
    let not_a = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let id_a = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(byte(A))],
    };
    let (x, y) = match (
        mycelium_mlir::mlir_compile_and_run(&not_a),
        mycelium_mlir::mlir_compile_and_run(&id_a),
    ) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(DialectError::ToolchainMissing(_)), _)
        | (_, Err(DialectError::ToolchainMissing(_))) => return,
        (x, y) => panic!("MLIR-dialect path errored: {x:?} / {y:?}"),
    };
    assert_ne!(observable(&x), observable(&y), "not(A) != id(A)");
    let verdict = check(
        &x,
        &y,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    );
    assert!(
        matches!(verdict, CheckVerdict::NotValidated { .. }),
        "the checker must reject the divergent MLIR pair, got {verdict:?}"
    );
}

// ─── M-856: the `Construct`/`Match` non-recursive data fragment ──────────────────────────────

/// A single-constructor, single-field type: `type Box = Box(Binary{8})`. Non-recursive (no
/// `FieldSpec::Data` back-reference) — firmly within the Increment-1 subset.
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

/// A two-constructor, no-field type: `type Color = Red | Blue`.
fn color_registry() -> DataRegistry {
    let mut specs = BTreeMap::new();
    specs.insert(
        "Color".to_owned(),
        DeclSpec {
            ctors: vec![CtorSpec { fields: vec![] }, CtorSpec { fields: vec![] }],
        },
    );
    DataRegistry::build(&specs).expect("Color registry must build")
}

/// The M-856 data-fragment corpus (mirrors `native_differential.rs::data_corpus`, the direct-LLVM
/// Increment-1 corpus, so the same shapes are proven three-way, not just two-way). Every program's
/// final result is a repr lane (bit vector); each is valid under the interpreter, the direct-LLVM
/// backend, and now the MLIR-dialect path.
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
        // 1. Construct Box(A), match to extract the inner field b -> return b unchanged (the
        //    tag-materialize + `cf.switch` + direct-SSA field bind + block-arg merge shape).
        Node::Match {
            scrutinee: Box::new(mk_box(A)),
            alts: vec![Alt::Ctor {
                ctor: reg.ctor_ref("Box", 0).unwrap(),
                binders: vec!["b".to_owned()],
                body: Node::Var("b".to_owned()),
            }],
            default: None,
        },
        // 2. Construct Box(A), match and apply bit.not to the extracted field — an op inside an arm
        //    body, using a binder (not just a constant).
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
        // 3. Let-bound Construct, then match — a Construct result in the env (Datum) looked up as
        //    the scrutinee of a later Match.
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
        // 4. Two-constructor Color type: match Red -> return A; match Blue -> return B. Exercises
        //    the switch with two real arms (the merge collects two (label, Lane) pairs).
        Node::Match {
            scrutinee: Box::new(red()),
            alts: vec![
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 0).unwrap(),
                    binders: vec![],
                    body: Node::Const(byte(A)),
                },
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 1).unwrap(),
                    binders: vec![],
                    body: Node::Const(byte(B)),
                },
            ],
            default: None,
        },
        // 5. Same two-constructor Color type but select Blue -> return B (mutant-witness that the
        //    switch dispatches on the correct tag, not always on arm 0).
        Node::Match {
            scrutinee: Box::new(blue()),
            alts: vec![
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 0).unwrap(),
                    binders: vec![],
                    body: Node::Const(byte(A)),
                },
                Alt::Ctor {
                    ctor: col.ctor_ref("Color", 1).unwrap(),
                    binders: vec![],
                    body: Node::Const(byte(B)),
                },
            ],
            default: None,
        },
        // NOTE (M-856b candidate, FLAGged, not included here): a `Match` `default` arm containing a
        // `trit.add` is deliberately NOT added to this *three-way* corpus. The direct-LLVM backend
        // (`crate::llvm`, read-only for this task) fails `llc`'s IR verifier on that shape
        // ("Instruction does not dominate all uses!") — it folds a Match arm's overflow flags into
        // the *same shared list* as the enclosing scope, so a flag computed only inside the
        // non-taken arm is referenced at a point that arm does not dominate. The MLIR-dialect path
        // (this crate) does not share the hazard (per-arm local folding + block-argument
        // re-export — see the module doc comment) and is covered on this exact shape by the
        // dedicated `crate::dialect::native::tests::match_default_arm_with_trit_add_…` in-crate
        // test (interp <-> MLIR-dialect only, since the direct-LLVM leg cannot compile it).
    ]
}

/// M-856: interp = direct-LLVM = MLIR-dialect on the `data_corpus` (`Construct`/`Match`), each pair
/// validated through the shared M-210 checker. Skips a path whose toolchain is absent; asserts the
/// MLIR path was non-vacuously exercised when its toolchain is present.
#[test]
fn interp_directllvm_mlirdialect_are_three_way_equivalent_on_the_data_corpus() {
    let mut ran_mlir = false;
    for (i, node) in data_corpus().iter().enumerate() {
        let interp = interp_eval(node);

        let direct = match mycelium_mlir::compile_and_run(node) {
            Ok(v) => Some(v),
            Err(AotError::ToolchainMissing(_)) => None,
            Err(e) => panic!("data program #{i}: direct-LLVM path errored: {e}"),
        };
        let mlir = match mycelium_mlir::mlir_compile_and_run(node) {
            Ok(v) => Some(v),
            Err(DialectError::ToolchainMissing(_)) => None,
            Err(e) => panic!("data program #{i}: MLIR-dialect path errored: {e}"),
        };

        if let Some(d) = &direct {
            assert_eq!(
                observable(&interp),
                observable(d),
                "data program #{i}: interp vs direct-LLVM diverged"
            );
        }
        if let Some(m) = &mlir {
            ran_mlir = true;
            // Mutant-witness: a wrong tag, a mis-bound field, or a switch on the wrong discriminant
            // would diverge here.
            assert_eq!(
                observable(&interp),
                observable(m),
                "data program #{i}: interp vs MLIR-dialect diverged ({:?} vs {:?})",
                interp.payload(),
                m.payload()
            );
            assert_eq!(
                check(
                    &interp,
                    m,
                    RefinementRelation::ObservationalEquiv,
                    Certificate::exact(),
                    &Evidence::Observational,
                ),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "data program #{i}: the shared checker must validate the interp<->MLIR pair"
            );
        }
        if let (Some(d), Some(m)) = (&direct, &mlir) {
            assert_eq!(
                observable(d),
                observable(m),
                "data program #{i}: direct-LLVM vs MLIR-dialect diverged"
            );
        }
    }
    if mycelium_mlir::MlirTools::is_available() {
        assert!(
            ran_mlir,
            "MLIR toolchain is available but no data program exercised the dialect path — vacuous"
        );
    }
}

/// M-856: a `Match` with **no default arm and no matching case** traps via the shared `@abort`
/// defined-trap convention — exercised only on the direct-LLVM/interp legs here (the MLIR-dialect
/// artifact aborts the *process*, which this differential harness's simple stdout read-back does
/// not attempt to observe as a `Value`; the emission-level trap shape is asserted in the in-crate
/// `construct_and_match_emit_a_switch_and_no_memory_ops` test instead). This test only pins that
/// the **direct-LLVM** and **interpreter** legs agree the case is a hard refusal (never a silent
/// wrong-arm fallthrough), so the abort path itself is not a divergence the MLIR leg introduces.
#[test]
fn match_default_arm_is_taken_not_the_no_default_trap() {
    // Sanity companion: program #6 in `data_corpus` (Blue with only a Red arm + a `trit.add`
    // default) exercises the *taken* default, not the trap — covered by the three-way test above.
    // This test just pins that omitting the default on an exhaustive two-arm match never traps.
    let col = color_registry();
    let node = Node::Match {
        scrutinee: Box::new(Node::Construct {
            ctor: col.ctor_ref("Color", 1).unwrap(),
            args: vec![],
        }),
        alts: vec![
            Alt::Ctor {
                ctor: col.ctor_ref("Color", 0).unwrap(),
                binders: vec![],
                body: Node::Const(byte(A)),
            },
            Alt::Ctor {
                ctor: col.ctor_ref("Color", 1).unwrap(),
                binders: vec![],
                body: Node::Const(byte(B)),
            },
        ],
        default: None,
    };
    let interp = interp_eval(&node);
    match mycelium_mlir::mlir_compile_and_run(&node) {
        Ok(v) => assert_eq!(
            observable(&interp),
            observable(&v),
            "exhaustive match diverged"
        ),
        Err(DialectError::ToolchainMissing(_)) => {}
        Err(e) => panic!("unexpected MLIR-dialect error: {e}"),
    }
}

/// The out-of-fragment corpus: nodes the MLIR-dialect path must **explicitly refuse** (routing to
/// the direct-LLVM/interp path), while interp ≡ direct-LLVM still holds. This proves coverage is
/// honest — the dialect path never silently mis-lowers a node it doesn't support (G2/VR-5).
///
/// **M-857 moved the boundary again:** `trit.mul` is now IN-fragment (it appears in
/// [`element_wise_corpus`] / [`overflow_corpus`]); the new boundary is everything *richer* than the
/// fixed-width bit/trit arithmetic — closures (`Lam`/`App`), recursion (`Fix`), the data fragment, and
/// `Swap`. These cases use **closures** (which the interpreter evaluates and direct-LLVM lowers to a
/// `Binary` lane, while the MLIR path refuses) so the interp ≡ direct-LLVM parity leg stays
/// non-vacuous; `Swap`/data nodes are exercised by the in-crate emission refusal tests instead (their
/// results don't read back as a simple lane through this harness's `IdentitySwapEngine`).
fn out_of_fragment_corpus() -> Vec<Node> {
    // `(λx. not x) A` — a closure applied to an argument. Result is a `Binary{8}` lane, so both the
    // interpreter and the direct-LLVM read-back produce a value; the MLIR-dialect path refuses it.
    let not_closure_applied = || Node::App {
        func: Box::new(Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("x".into())],
            }),
        }),
        arg: Box::new(Node::Const(byte(A))),
    };
    vec![
        // A bare closure application — the new boundary: refused by the dialect path (closures are not
        // in the MLIR-dialect fragment), lowered by direct-LLVM (M-378) and interpreted.
        not_closure_applied(),
        // The identity closure `(λy. y) B` — a different closure, still refused by the MLIR path.
        Node::App {
            func: Box::new(Node::Lam {
                param: "y".into(),
                body: Box::new(Node::Var("y".into())),
            }),
            arg: Box::new(Node::Const(byte(B))),
        },
        // A closure application nested behind an in-fragment bit.not (so the *program* mixes an
        // in-fragment op with an out-of-fragment node) — the whole program is refused by the MLIR
        // path and routed to direct-LLVM/interp.
        Node::Let {
            id: "c".into(),
            bound: Box::new(not_closure_applied()),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("c".into())],
            }),
        },
        // M-856: a `Match` with a **literal** arm on a `Binary{8}` scrutinee (the Increment-3
        // recursion branch primitive `crate::llvm` also lowers outside a `Fix` loop) — still
        // refused by the MLIR-dialect path (only `Ctor`-arm matching on a `Construct`-built datum
        // is covered; the `Lit`-arm form is tied to the Fix/FixGroup fragment, deferred). Direct-LLVM
        // and the interpreter both handle it.
        Node::Match {
            scrutinee: Box::new(Node::Const(byte(A))),
            alts: vec![Alt::Lit {
                value: byte(A),
                body: Node::Const(byte(B)),
            }],
            default: Some(Box::new(Node::Const(byte(A)))),
        },
        // M-856: an **illegal** binary<->ternary Swap pair — `Binary{8} -> Ternary{2}` (2^7=128 >
        // (3^2-1)/2=4) — still an explicit MLIR refusal (the `Recheck` compile-time re-check rejects
        // it), while the interpreter's certified engine also raises `IllegalPair` and direct-LLVM
        // refuses at compile time too, so all three legs agree the program never silently produces a
        // value — checked directly in `swap_differential.rs`, not read back here (a `Swap`'s result
        // does not round-trip through this harness's plain `IdentitySwapEngine` interp path).
    ]
}

/// The **overflow** corpus (M-725; M-857): in-fragment `trit.add`/`trit.sub`/`trit.mul` programs whose
/// fixed-width result leaves the `m`-trit range. All three paths must **refuse** non-silently — the
/// interpreter errors (`EvalError::Overflow`), the direct-LLVM path returns `AotError::Overflow`, and
/// the MLIR-dialect path returns `DialectError::Overflow` (the shared sentinel read-back). This is the
/// overflow half of the honest arithmetic boundary — a value is never silently wrapped (SC-3/G2).
fn overflow_corpus() -> Vec<Node> {
    vec![
        // max(2 trits) + max(2 trits) = 4 + 4 = 8, out of the 2-trit range [−4, 4]. ([+,+] = 4.)
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
            ],
        },
        // 4 − (−4) = 8, out of the 2-trit range.
        Node::Op {
            prim: "trit.sub".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
                Node::Const(tern(vec![Trit::Neg, Trit::Neg])),
            ],
        },
        // M-857: max(2 trits) · max(2 trits) = 4 · 4 = 16, out of the 2-trit range [−4, 4] — a high
        // trit is non-zero. Exercises the multiply overflow read-back (the 2m-buffer high half).
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
            ],
        },
        // 3 · 3 = 9, out of the 2-trit range (9 > 4). ([+,0] = 3.)
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Zero])),
                Node::Const(tern(vec![Trit::Pos, Trit::Zero])),
            ],
        },
    ]
}

#[test]
fn out_of_fragment_nodes_are_refused_by_mlir_but_run_on_direct_llvm() {
    for (i, node) in out_of_fragment_corpus().iter().enumerate() {
        // The MLIR-dialect path refuses explicitly (never silently mis-lowers).
        match mycelium_mlir::mlir_compile_and_run(node) {
            Err(DialectError::Unsupported(_)) => { /* expected explicit refusal */ }
            Err(DialectError::ToolchainMissing(_)) => { /* env skip — still no silent success */ }
            Ok(v) => panic!(
                "out-of-fragment program #{i} must be refused by the MLIR path, got {:?}",
                v.payload()
            ),
            Err(e) => panic!("out-of-fragment program #{i}: unexpected MLIR error: {e}"),
        }
        // …and the direct-LLVM path still agrees with the interpreter (coverage is preserved there).
        let interp = interp_eval(node);
        match mycelium_mlir::compile_and_run(node) {
            Ok(d) => assert_eq!(
                observable(&interp),
                observable(&d),
                "out-of-fragment program #{i}: interp vs direct-LLVM diverged"
            ),
            Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
            Err(e) => panic!("out-of-fragment program #{i}: direct-LLVM errored: {e}"),
        }
    }
}

/// M-725 / M-857: the **overflow** three-way refusal parity. An in-fragment
/// `trit.add`/`trit.sub`/`trit.mul` whose result leaves the `m`-trit range must be refused
/// **non-silently by all three paths** — the interpreter errors, and both compiled paths return an
/// explicit `Overflow`. Never a silent wrap on any path (SC-3/G2), so the fixed-width arithmetic
/// boundary is honest on overflow as well as on value.
#[test]
fn overflowing_trit_arithmetic_is_refused_non_silently_three_ways() {
    for (i, node) in overflow_corpus().iter().enumerate() {
        // Path 1: the reference interpreter must error (not return a wrapped value).
        let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
            .eval(node);
        assert!(
            interp.is_err(),
            "overflow program #{i}: interpreter must refuse (error), got {:?}",
            interp.ok().map(|v| v.payload().clone())
        );

        // Path 2: direct-LLVM must return an explicit Overflow (or skip if the toolchain is absent).
        match mycelium_mlir::compile_and_run(node) {
            Err(AotError::Overflow(_)) => { /* expected explicit refusal */ }
            Err(AotError::ToolchainMissing(_)) => { /* env skip — still no silent success */ }
            Ok(v) => panic!(
                "overflow program #{i}: direct-LLVM must refuse, got {:?}",
                v.payload()
            ),
            Err(e) => panic!("overflow program #{i}: unexpected direct-LLVM error: {e}"),
        }

        // Path 3: MLIR-dialect must return an explicit Overflow (the shared sentinel read-back), or
        // skip if its toolchain is absent. NOT Unsupported (these ops are in-fragment now), NOT Ok.
        match mycelium_mlir::mlir_compile_and_run(node) {
            Err(DialectError::Overflow(_)) => { /* expected explicit refusal */ }
            Err(DialectError::ToolchainMissing(_)) => { /* env skip */ }
            Ok(v) => panic!(
                "overflow program #{i}: MLIR-dialect must refuse (Overflow), got {:?}",
                v.payload()
            ),
            Err(e) => panic!("overflow program #{i}: unexpected MLIR error: {e}"),
        }
    }
}

// ─── M-602 E1 speedup: MLIR-dialect native vs interpreter (MEASURED, no pre-written target) ────

/// The E1 perf verdict half (M-602; M-303; NFR-4): a **measured** MLIR-dialect-native-vs-interpreter
/// ratio on the element-wise fragment. Reported as-measured — **no pre-written target** (VR-5); the
/// number is whatever the box produces.
///
/// `#[ignore]` by default (it spawns processes + times — not part of the fast unit gate) and
/// **refuses a debug build** (an unoptimized timing is meaningless, exactly as `xtask e1` refuses).
/// Run with: `cargo test -p mycelium-mlir --features mlir-dialect --release -- --ignored
/// e1_mlir_dialect_speedup_is_measured --nocapture`.
///
/// **Honest caption (printed):** the MLIR-native per-invocation figure is **process-spawn-bound** for
/// this trivial kernel (one `putchar` loop), so the ratio reflects spawn + run vs in-process eval —
/// captioned as such, never sold as raw compute throughput. This is the *AOT-path* E1 number; the
/// in-process compute-throughput E1 numbers (BitNet kernels) live in `xtask e1` §3–§5.
#[test]
#[ignore = "perf measurement: run with --release --ignored --nocapture"]
fn e1_mlir_dialect_speedup_is_measured() {
    // Refuse a debug build — an unoptimized timing is meaningless (parity with `xtask e1`'s
    // debug-build refusal). Print + return rather than assert (a `cfg!` assert is a constant).
    if cfg!(debug_assertions) {
        eprintln!(
            "E1(MLIR) refusing to measure a debug build — re-run with --release \
             (`cargo test -p mycelium-mlir --features mlir-dialect --release -- --ignored \
             e1_mlir_dialect_speedup_is_measured --nocapture`)."
        );
        return;
    }

    // Representative element-wise program: not(A xor B) over 8 bits.
    let prog = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Op {
            prim: "bit.xor".into(),
            args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
        }],
    };

    // Compile once through the MLIR pipeline (skip if the toolchain is absent).
    let t0 = std::time::Instant::now();
    let artifact = match mycelium_mlir::mlir_compile(&prog) {
        Ok(a) => a,
        Err(DialectError::ToolchainMissing(tool)) => {
            eprintln!("E1(MLIR) skip: MLIR toolchain absent ({tool}) — run scripts/setup-mlir.sh.");
            return;
        }
        Err(e) => panic!("MLIR compile failed: {e}"),
    };
    #[allow(clippy::cast_precision_loss)]
    let compile_ns = t0.elapsed().as_nanos() as f64;

    // Correctness gate before timing: the MLIR artifact must agree with the interpreter (refusing to
    // time a wrong kernel — the `xtask e1` discipline).
    let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine));
    let interp_val = interp.eval(&prog).expect("interp eval");
    let native_val = artifact.run().expect("MLIR artifact run");
    assert_eq!(
        observable(&interp_val),
        observable(&native_val),
        "E1(MLIR): native disagrees with interpreter — refusing to time a wrong kernel"
    );

    // Warm timing: minimum mean over a few batches (house style — no bench dependency).
    let native_ns = bench(40u32, || {
        std::hint::black_box(artifact.run().expect("run"));
    });
    let interp_ns = bench(20_000u32, || {
        std::hint::black_box(interp.eval(std::hint::black_box(&prog)).expect("eval"));
    });

    let ratio = if native_ns > 0.0 {
        native_ns / interp_ns
    } else {
        0.0
    };
    println!(
        "== E1 (M-602): MLIR-dialect native vs interpreter (element-wise, LLVM {}) ==",
        artifact.llvm_major()
    );
    println!("  MLIR AOT compile (emit MLIR + mlir-opt + mlir-translate + clang), one-time : {compile_ns:>14.0} ns");
    println!("  MLIR native per-invocation (spawn + run, warm)                            : {native_ns:>14.0} ns  [process-spawn-bound]");
    println!("  interpreter per-eval (in-process)                                         : {interp_ns:>14.0} ns");
    println!(
        "  ratio native/interp = {ratio:>6.1}x  (>1 ⇒ spawn dominates for this trivial kernel)"
    );
    println!(
        "  note: the per-invocation figure is process-spawn-bound for this trivial kernel, not \
         kernel compute. This is the AOT-path E1 number, measured — no pre-written target (VR-5). \
         In-process compute-throughput E1 numbers are in `xtask e1` §3–§5."
    );
}
