//! M-858 — the **unified, mutant-witnessed three-way differential** (RFC-0029 §7.5; M-729; NFR-7;
//! VR-4; RR-12; ADR-009; G2/VR-5).
//!
//! Every native-codegen fragment so far has its own scattered differential file
//! (`threeway_differential.rs` for element-wise + carry arithmetic, `native_differential.rs`'s
//! `data_corpus` for `Construct`/`Match` on **direct-LLVM only**, `swap_differential.rs` for the
//! certified binary↔ternary `Swap`, `closure_widening_differential.rs`/
//! `recursion_trampoline_differential.rs` which only ever *claim* in a doc comment that the
//! MLIR-dialect leg refuses closures/recursion, never asserting it). This file is the **single
//! entrypoint** M-858 asks for: one shared corpus, run through
//!
//! 1. the M-110 **reference interpreter** (the trusted base),
//! 2. the **direct-LLVM** backend ([`mycelium_mlir::compile_and_run`]; `llvm.rs`),
//! 3. the **MLIR-dialect** backend ([`mycelium_mlir::mlir_compile_and_run`]; `dialect/native.rs`),
//!    and
//! 4. the **JIT** ([`mycelium_mlir::run_mode`] with [`ExecMode::Jit`]) for the bit/trit subset it
//!    covers,
//!
//! every pair validated through the **single shared M-210 checker**
//! ([`RefinementRelation::ObservationalEquiv`]), with a **`ran_mlir`/`ran_jit` non-vacuity guard**:
//! a skipped/absent-toolchain leg must never masquerade as agreement.
//!
//! **Fragments covered (one corpus each, run through [`assert_in_fragment_agreement`]):**
//! - [`element_wise_and_arithmetic_corpus`] — `core.id`, `bit.*`, `trit.neg`, and the fixed-width
//!   carry chain `trit.add`/`trit.sub` (M-725) / `trit.mul` (M-857) — **four-way** (interp ≡
//!   direct-LLVM ≡ MLIR-dialect ≡ JIT).
//! - [`data_fragment_corpus`] — `Construct`/`Match` (`Ctor`-arm; M-373/M-856) — **three-way**
//!   (interp ≡ direct-LLVM ≡ MLIR-dialect; JIT does not cover this fragment, M-727).
//! - [`certified_swap_corpus`] — the certified binary↔ternary `Swap` class (ADR-034/M-852/M-856) —
//!   **three-way**, always under `SwapCertMode::Recheck` (the default both compiled paths use for a
//!   bare `Swap` node).
//!
//! **Overflow parity** ([`overflow_refusal_is_three_way_honest`]) — a fixed-width result leaving its
//! range (`trit.add`/`trit.sub`/`trit.mul` M-725/M-857, or an out-of-range `Swap` `dec`) is refused
//! **non-silently** by every leg that ran — never a silent wrap (SC-3/G2).
//!
//! **Honest boundary, now *checked* not just claimed**
//! ([`dialect_honestly_refuses_closures_and_recursion`]) — closures (`App`/`Lam`) and object-level
//! recursion (`Fix`) are covered by direct-LLVM (M-378/M-850/M-851) but are an explicit, deliberate
//! MLIR-dialect refusal (`DialectError::Unsupported`); the two sibling differential files only ever
//! *asserted this in a doc comment* — this test turns it into a checked fact, so the three-way
//! **honestly reduces to two-way** here (interp ≡ direct-LLVM), never a faked three-way pass. The
//! other known deferred boundary — Dense/VSA as a `Swap` target or generic `Repr` (M-856b) — is
//! exercised in `swap_differential.rs`'s `swap_to_dense_is_refused_by_the_mlir_dialect_path` and
//! `dense_differential.rs`/`vsa_differential.rs`'s own dialect-refusal tests; not duplicated here.
//!
//! **Mutant-witness ([`mutant_witness_catches_a_divergence_in_every_codegen_leg`]).** The
//! differential is `Empirical` only if a codegen divergence is demonstrably **caught**. For each of
//! the three in-fragment categories above (arithmetic, data, certified swap) we take two *real*
//! compiled MLIR-dialect values from two distinct in-corpus programs and assert the **same shared
//! M-210 checker** the equivalence tests trust **rejects** the mismatched pair — modelling exactly
//! what a mis-lowering in `dialect/native.rs` would produce. This covers M-856's Construct/Match +
//! Swap dialect legs as well as the arithmetic leg, so **no separate M-856 mutant pass is needed**
//! (RFC-0029 §7.5).
//!
//! **Toolchain skip.** Direct-LLVM needs `llc`/`clang`; MLIR-dialect needs
//! `mlir-opt`/`mlir-translate`/`clang`; JIT needs `clang`. Where a tool is absent that leg returns a
//! `ToolchainMissing` and the test **skips** that leg (the house "skip gracefully" idiom) — never a
//! false failure — but the non-vacuity guards assert that when the toolchain **is** present, the leg
//! genuinely ran on at least one program.
//!
//! **Guarantee:** `Empirical` — this differential plus its mutant witness is the checked basis for
//! the whole native-codegen surface's correctness claim (interp ≡ direct-LLVM ≡ MLIR-dialect ≡ JIT
//! over the corpus, with a demonstrated divergence-catch); never upgraded to `Proven` without a
//! checked refinement proof (VR-5).

#![cfg(feature = "mlir-dialect")]

mod common;
use common::{byte, observable, tern, Observable, A, B, ONES};

use mycelium_cert::{
    binary_to_ternary, check, BinaryTernarySwapEngine, CheckVerdict, Evidence, RefinementRelation,
};
use mycelium_core::{
    Alt, ContentHash, CtorSpec, DataRegistry, DeclSpec, FieldSpec, GuaranteeStrength, Meta, Node,
    Payload, Provenance, Repr, Trit, Value,
};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::{AotError, DialectError, ExecMode, ModeError};
use mycelium_numerics::Certificate;
use std::collections::BTreeMap;

// ─── shared plumbing ──────────────────────────────────────────────────────────────────────────

/// One corpus case: a program plus which interpreter oracle evaluates it. `certified` selects the
/// M-120 certified binary↔ternary swap engine (the reference the compiled `Swap` legs must match);
/// every other fragment uses the identity swap engine (no `Swap` nodes in those corpora).
struct Case {
    label: String,
    node: Node,
    certified: bool,
}

impl Case {
    fn plain(label: impl Into<String>, node: Node) -> Self {
        Case {
            label: label.into(),
            node,
            certified: false,
        }
    }

    fn certified(label: impl Into<String>, node: Node) -> Self {
        Case {
            label: label.into(),
            node,
            certified: true,
        }
    }
}

/// Evaluate a case's node under its designated reference-interpreter oracle.
fn interp_of(c: &Case) -> Result<Value, mycelium_interp::EvalError> {
    if c.certified {
        Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(BinaryTernarySwapEngine),
        )
        .eval(&c.node)
    } else {
        Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine)).eval(&c.node)
    }
}

/// Assert the shared M-210 checker validates a pair as observationally equivalent.
fn assert_checker_validates(a: &Value, b: &Value, label: &str, edge: &str) {
    assert_eq!(
        check(
            a,
            b,
            RefinementRelation::ObservationalEquiv,
            Certificate::exact(),
            &Evidence::Observational,
        ),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "{label}: the shared M-210 checker must validate the {edge} pair"
    );
}

/// Assert the shared M-210 checker REJECTS a pair (the mutant-witness half — a real divergence must
/// be caught, never silently accepted).
fn assert_checker_rejects(a: &Value, b: &Value, label: &str) {
    let verdict = check(
        a,
        b,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    );
    assert!(
        matches!(verdict, CheckVerdict::NotValidated { .. }),
        "{label}: the shared M-210 checker must REJECT a genuine divergence, got {verdict:?}"
    );
}

/// Run one corpus through interp ≡ direct-LLVM ≡ MLIR-dialect (≡ JIT iff `allow_jit`), each pair
/// M-210-checked. Accumulates into the caller's `ran_mlir`/`ran_jit` non-vacuity flags — a skipped
/// leg never masquerades as agreement (G2).
fn assert_in_fragment_agreement(
    cases: &[Case],
    allow_jit: bool,
    ran_mlir: &mut bool,
    ran_jit: &mut bool,
) {
    for c in cases {
        let interp = interp_of(c).unwrap_or_else(|e| {
            panic!("{}: interpreter must evaluate the case, got {e:?}", c.label)
        });

        let direct = match mycelium_mlir::compile_and_run(&c.node) {
            Ok(v) => Some(v),
            Err(AotError::ToolchainMissing(_)) => None,
            Err(e) => panic!("{}: direct-LLVM errored: {e}", c.label),
        };
        let mlir = match mycelium_mlir::mlir_compile_and_run(&c.node) {
            Ok(v) => Some(v),
            Err(DialectError::ToolchainMissing(_)) => None,
            Err(e) => panic!("{}: MLIR-dialect errored: {e}", c.label),
        };

        if let Some(d) = &direct {
            assert_eq!(
                observable(&interp),
                observable(d),
                "{}: interp vs direct-LLVM diverged ({:?} vs {:?})",
                c.label,
                interp.payload(),
                d.payload()
            );
            assert_checker_validates(&interp, d, &c.label, "interp<->direct-LLVM");
        }
        if let Some(m) = &mlir {
            *ran_mlir = true;
            assert_eq!(
                observable(&interp),
                observable(m),
                "{}: interp vs MLIR-dialect diverged ({:?} vs {:?})",
                c.label,
                interp.payload(),
                m.payload()
            );
            assert_checker_validates(&interp, m, &c.label, "interp<->MLIR-dialect");
        }
        if let (Some(d), Some(m)) = (&direct, &mlir) {
            assert_eq!(
                observable(d),
                observable(m),
                "{}: direct-LLVM vs MLIR-dialect diverged",
                c.label
            );
            assert_checker_validates(d, m, &c.label, "direct-LLVM<->MLIR-dialect");
        }

        if allow_jit {
            let jit = match mycelium_mlir::run_mode(
                ExecMode::Jit,
                &c.node,
                PrimRegistry::with_builtins(),
                Box::new(IdentitySwapEngine),
            ) {
                Ok(v) => Some(v),
                Err(ModeError::ToolchainMissing(_)) => None,
                Err(e) => panic!("{}: JIT errored: {e}", c.label),
            };
            if let Some(j) = &jit {
                *ran_jit = true;
                assert_eq!(
                    observable(&interp),
                    observable(j),
                    "{}: interp vs JIT diverged ({:?} vs {:?})",
                    c.label,
                    interp.payload(),
                    j.payload()
                );
                assert_checker_validates(&interp, j, &c.label, "interp<->JIT");
            }
        }
    }
}

// ─── Fragment A: element-wise + fixed-width trit arithmetic (four-way) ──────────────────────────

/// `core.id`, `bit.*`, `trit.neg`, and the in-range fixed-width carry chain `trit.add`/`trit.sub`
/// (M-725) / `trit.mul` (M-857) — the fragment every leg (interp, direct-LLVM, MLIR-dialect, JIT)
/// covers. A small deterministic set, not a statistical sample.
fn element_wise_and_arithmetic_corpus() -> Vec<Case> {
    let cst = |bits: [bool; 8]| Node::Const(byte(bits));
    vec![
        Case::plain("const A", cst(A)),
        Case::plain(
            "core.id(A)",
            Node::Op {
                prim: "core.id".into(),
                args: vec![cst(A)],
            },
        ),
        Case::plain(
            "let a = A in a",
            Node::Let {
                id: "a".into(),
                bound: Box::new(cst(A)),
                body: Box::new(Node::Var("a".into())),
            },
        ),
        Case::plain(
            "bit.not(A)",
            Node::Op {
                prim: "bit.not".into(),
                args: vec![cst(A)],
            },
        ),
        Case::plain(
            "bit.and(A,B)",
            Node::Op {
                prim: "bit.and".into(),
                args: vec![cst(A), cst(B)],
            },
        ),
        Case::plain(
            "bit.or(A,B)",
            Node::Op {
                prim: "bit.or".into(),
                args: vec![cst(A), cst(B)],
            },
        ),
        Case::plain(
            "bit.xor(A,ONES)",
            Node::Op {
                prim: "bit.xor".into(),
                args: vec![cst(A), cst(ONES)],
            },
        ),
        Case::plain(
            "not(let x = A^B in x)",
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
        ),
        Case::plain(
            "trit.neg([+,0,-,+])",
            Node::Op {
                prim: "trit.neg".into(),
                args: vec![Node::Const(tern(vec![
                    Trit::Pos,
                    Trit::Zero,
                    Trit::Neg,
                    Trit::Pos,
                ]))],
            },
        ),
        Case::plain(
            "trit.add(5,4) over 3 trits",
            Node::Op {
                prim: "trit.add".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
                ],
            },
        ),
        Case::plain(
            "trit.sub(3,4) over 3 trits",
            Node::Op {
                prim: "trit.sub".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Pos, Trit::Zero, Trit::Zero])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
                ],
            },
        ),
        // M-857: trit.mul, the shifted-accumulate multiply.
        Case::plain(
            "trit.mul(2,3) over 3 trits",
            Node::Op {
                prim: "trit.mul".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
                ],
            },
        ),
        Case::plain(
            "let s = 5+4 in s-4",
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
        ),
    ]
}

// ─── Fragment B: Construct / Match data fragment (three-way; M-373/M-856) ──────────────────────

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

/// The `Construct`/`Match` data fragment (`Ctor`-arm only; M-373 Increment-1, widened to the
/// MLIR-dialect leg by M-856). Non-recursive, bounded — every case here is in the increment both
/// direct-LLVM and the dialect cover.
fn data_fragment_corpus() -> Vec<Case> {
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
        Case::plain(
            "match Box(A) { Mk(b) => b }",
            Node::Match {
                scrutinee: Box::new(mk_box(A)),
                alts: vec![Alt::Ctor {
                    ctor: reg.ctor_ref("Box", 0).unwrap(),
                    binders: vec!["b".to_owned()],
                    body: Node::Var("b".to_owned()),
                }],
                default: None,
            },
        ),
        Case::plain(
            "match Box(A) { Mk(b) => bit.not(b) }",
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
        ),
        Case::plain(
            "let box_a = Box(A) in match box_a { Mk(b) => b & B }",
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
        ),
        Case::plain(
            "match Red { Red=>A, Blue=>B }",
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
        ),
        Case::plain(
            "match Blue { Red=>A, Blue=>B }",
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
        ),
    ]
}

// ─── Fragment C: certified binary↔ternary Swap (three-way; ADR-034/M-852/M-856) ────────────────

fn binary(bits: Vec<bool>) -> Value {
    let width = bits.len() as u32;
    Value::new(
        Repr::Binary { width },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

fn swap_policy() -> ContentHash {
    ContentHash::parse("blake3:round_trip_safe").unwrap()
}

fn swap_node(src: Value, target: Repr) -> Node {
    Node::Swap {
        src: Box::new(Node::Const(src)),
        target,
        policy: swap_policy(),
    }
}

/// In-range certified binary↔ternary `Swap`s (both directions), always the default
/// `SwapCertMode::Recheck` both compiled paths use for a bare `Swap` node.
fn certified_swap_corpus() -> Vec<Case> {
    vec![
        Case::certified(
            "swap(-78:binary8 -> ternary6)",
            swap_node(
                binary(vec![true, false, true, true, false, false, true, false]),
                Repr::Ternary { trits: 6 },
            ),
        ),
        Case::certified(
            "swap(0:binary8 -> ternary6)",
            swap_node(binary(vec![false; 8]), Repr::Ternary { trits: 6 }),
        ),
        Case::certified(
            "swap(-5:binary4 -> ternary4)",
            swap_node(
                binary(vec![true, false, true, true]),
                Repr::Ternary { trits: 4 },
            ),
        ),
        Case::certified(
            "swap(2:ternary3 -> binary4)",
            swap_node(
                tern(vec![Trit::Zero, Trit::Pos, Trit::Neg]),
                Repr::Binary { width: 4 },
            ),
        ),
    ]
}

// ─── the unified entrypoint ─────────────────────────────────────────────────────────────────────

/// **The single M-858 differential entrypoint.** Runs every fragment's shared corpus through
/// interp ≡ direct-LLVM ≡ MLIR-dialect (≡ JIT for the arithmetic fragment), each pair M-210-checked,
/// then asserts the `ran_mlir`/`ran_jit` non-vacuity guards: with real `mlir-opt-18`/`clang` present
/// in this environment, a vacuous all-skipped pass must never be reported as agreement (G2).
#[test]
fn unified_interp_directllvm_mlirdialect_jit_differential() {
    let mut ran_mlir = false;
    let mut ran_jit = false;

    assert_in_fragment_agreement(
        &element_wise_and_arithmetic_corpus(),
        true, // JIT covers this fragment
        &mut ran_mlir,
        &mut ran_jit,
    );
    assert_in_fragment_agreement(&data_fragment_corpus(), false, &mut ran_mlir, &mut ran_jit);
    assert_in_fragment_agreement(&certified_swap_corpus(), false, &mut ran_mlir, &mut ran_jit);

    if mycelium_mlir::MlirTools::is_available() {
        assert!(
            ran_mlir,
            "MLIR toolchain is available but no fragment exercised the dialect leg — vacuous"
        );
    }
    if clang_present() {
        assert!(
            ran_jit,
            "clang is available but no fragment exercised the JIT leg — vacuous"
        );
    }
}

/// Probe whether the JIT toolchain (`clang`) is usable right now, by attempting a trivial in-subset
/// compile. Used by the non-vacuity guard above.
fn clang_present() -> bool {
    let trivial = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    !matches!(
        mycelium_mlir::run_mode(
            ExecMode::Jit,
            &trivial,
            PrimRegistry::with_builtins(),
            Box::new(IdentitySwapEngine),
        ),
        Err(ModeError::ToolchainMissing(_))
    )
}

// ─── overflow parity (three-way, non-silent refusal) ────────────────────────────────────────────

/// A fixed-width arithmetic/`Swap`-`dec` result leaving its range must be refused **non-silently**
/// by every leg that ran (SC-3/G2): the interpreter errors, direct-LLVM returns `AotError::Overflow`,
/// MLIR-dialect returns `DialectError::Overflow` (the shared sentinel read-back) — never a wrap.
#[test]
fn overflow_refusal_is_three_way_honest() {
    // trit.add: max(2 trits) + max(2 trits) = 4 + 4 = 8, out of the 2-trit range [-4, 4].
    let arith_overflow = Case::plain(
        "trit.add(4,4) over 2 trits (overflow)",
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
                Node::Const(tern(vec![Trit::Pos, Trit::Pos])),
            ],
        },
    );
    // trit.mul: the 3-trit range is [-13, 13]; 13 * 13 = 169 is far out of range.
    let mul_overflow = Case::plain(
        "trit.mul(13,13) over 3 trits (overflow)",
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Pos, Trit::Pos])),
                Node::Const(tern(vec![Trit::Pos, Trit::Pos, Trit::Pos])),
            ],
        },
    );
    // Swap dec: a large ternary value swapped down into too-narrow a binary width.
    let big_ternary = binary_to_ternary(
        &binary(vec![false, true, true, false, false, true, false, false]),
        6,
        &swap_policy(),
    )
    .unwrap()
    .0;
    let Payload::Trits(ts) = big_ternary.payload() else {
        unreachable!()
    };
    let swap_overflow = Case::certified(
        "swap(100:ternary6 -> binary4) (out-of-range dec)",
        swap_node(tern(ts.clone()), Repr::Binary { width: 4 }),
    );
    // **M-858 mutant witness (the swap-`dec` range boundary).** The `dec` range check in
    // `emit_swap_int_to_bits` computes `hi = (1 << (n-1)) - 1` — for `binary4`, `B_4 = [-8, 7]`, so
    // `hi = 7`. The value **8** is the *first* value past the boundary: the correct code refuses it
    // (`8 > 7`), while every mutation that widens or shifts that bound (`n-1 → n+1`/`n/1`,
    // `half-1 → half+1`/`half/1`) makes `8` wrongly *in-range* — so the MLIR-dialect artifact would
    // return `Ok` instead of `Overflow`, and this case catches it (the far-out `100` case above does
    // not, since it overflows every widened bound too). `8 = +1·9 + 0·3 − 1·1` over 3 trits (MSB-first),
    // and `(4,3)` is a legal pair (`2^3 = 8 ≤ (3^3−1)/2 = 13`).
    let swap_boundary_overflow = Case::certified(
        "swap(8:ternary3 -> binary4) (boundary: first value past B_4 max 7)",
        swap_node(
            tern(vec![Trit::Pos, Trit::Zero, Trit::Neg]),
            Repr::Binary { width: 4 },
        ),
    );

    for case in [
        arith_overflow,
        mul_overflow,
        swap_overflow,
        swap_boundary_overflow,
    ] {
        let interp = interp_of(&case);
        assert!(
            interp.is_err(),
            "{}: interpreter must refuse the out-of-range result, got {:?}",
            case.label,
            interp.ok().map(|v| v.payload().clone())
        );

        match mycelium_mlir::compile_and_run(&case.node) {
            Err(AotError::Overflow(_)) => { /* expected */ }
            Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
            Ok(v) => panic!(
                "{}: direct-LLVM must refuse, got {:?}",
                case.label,
                v.payload()
            ),
            Err(e) => panic!("{}: unexpected direct-LLVM error: {e}", case.label),
        }

        match mycelium_mlir::mlir_compile_and_run(&case.node) {
            Err(DialectError::Overflow(_)) => { /* expected */ }
            Err(DialectError::ToolchainMissing(_)) => { /* env skip */ }
            Ok(v) => panic!(
                "{}: MLIR-dialect must refuse, got {:?}",
                case.label,
                v.payload()
            ),
            Err(e) => panic!("{}: unexpected MLIR-dialect error: {e}", case.label),
        }
    }
}

// ─── honest boundary: checked, not just claimed ─────────────────────────────────────────────────

/// Closures (`App`/`Lam`) and object-level recursion (`Fix`) are covered by direct-LLVM
/// (M-378/M-850/M-851) but stay an explicit MLIR-dialect refusal
/// (`DialectError::Unsupported`) — the honest boundary `dialect/native.rs`'s own module doc
/// describes. `closure_widening_differential.rs`/`recursion_trampoline_differential.rs` only ever
/// *state this in a comment*; this test makes it a checked fact, so the three-way honestly
/// **reduces to two-way** for this fragment (never a faked pass, G2/VR-5).
#[test]
fn dialect_honestly_refuses_closures_and_recursion() {
    // A minimal closure: let y = B in (lambda x. x xor y) A -> A xor B.
    let closure = Node::Let {
        id: "y".into(),
        bound: Box::new(Node::Const(byte(B))),
        body: Box::new(Node::App {
            func: Box::new(Node::Lam {
                param: "x".into(),
                body: Box::new(Node::Op {
                    prim: "bit.xor".into(),
                    args: vec![Node::Var("x".into()), Node::Var("y".into())],
                }),
            }),
            arg: Box::new(Node::Const(byte(A))),
        }),
    };

    // A minimal non-tail single-`Fix` recursion: f = fix self. lambda n. match n {
    //   Lit 0 => A ; default => bit.not(self 0) }, applied to a nonzero n — the default arm fires
    // once (a non-tail call), then the base case returns A, so the observable result is bit.not(A).
    let zero_val = byte([false; 8]);
    let one_val = byte([true, false, false, false, false, false, false, false]);
    let recursion = Node::App {
        func: Box::new(Node::Fix {
            name: "self".into(),
            body: Box::new(Node::Lam {
                param: "n".into(),
                body: Box::new(Node::Match {
                    scrutinee: Box::new(Node::Var("n".into())),
                    alts: vec![Alt::Lit {
                        value: zero_val.clone(),
                        body: Node::Const(byte(A)),
                    }],
                    default: Some(Box::new(Node::Op {
                        prim: "bit.not".into(),
                        args: vec![Node::App {
                            func: Box::new(Node::Var("self".into())),
                            arg: Box::new(Node::Const(zero_val.clone())),
                        }],
                    })),
                }),
            }),
        }),
        arg: Box::new(Node::Const(one_val)),
    };

    for (label, node) in [("closure", closure), ("non-tail recursion", recursion)] {
        // Edge 1: interp ≡ direct-LLVM (the fragment direct-LLVM DOES cover).
        let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
            .eval(&node)
            .unwrap_or_else(|e| panic!("{label}: interpreter must evaluate, got {e:?}"));
        match mycelium_mlir::compile_and_run(&node) {
            Ok(native) => {
                assert_eq!(
                    observable(&interp),
                    observable(&native),
                    "{label}: interp vs direct-LLVM diverged"
                );
                assert_checker_validates(&interp, &native, label, "interp<->direct-LLVM");
            }
            Err(AotError::ToolchainMissing(_)) => { /* env skip */ }
            Err(e) => panic!("{label}: direct-LLVM must cover this fragment, got {e}"),
        }

        // Edge 2 (the honest boundary): the MLIR-dialect leg must EXPLICITLY refuse — never
        // silently succeed with a possibly-wrong value, and never silently skip as though it agreed.
        match mycelium_mlir::mlir_compile_and_run(&node) {
            Err(DialectError::Unsupported(_)) => { /* expected — the checked honest boundary */ }
            Err(DialectError::ToolchainMissing(_)) => { /* env skip — still not a silent success */
            }
            Ok(v) => panic!(
                "{label}: the MLIR-dialect leg must refuse (out of the M-856 fragment), got {:?}",
                v.payload()
            ),
            Err(e) => panic!("{label}: unexpected MLIR-dialect error: {e}"),
        }
    }
}

/// **M-858 mutant witness (the Match arm-shape-consistency guard).** `lower_match_dialect` refuses a
/// `Match` whose arms produce lanes of **different kind or width** (`a.lane.kind != kind ||
/// a.lane.vals.len() != width`) — the block-argument "phi" that merges the arms needs a uniform
/// shape, so a heterogeneous merge is an explicit, never-silent [`DialectError::Unsupported`] (G2).
/// For a **well-typed** `Match` both operands are always false, so the equivalence corpus can't
/// exercise this guard; this test builds a deliberately **same-kind, different-width** `Match`
/// (`Binary{8}` arm vs `Binary{4}` arm) — exactly one operand true — so the guard's `||` is the load
/// bearing operator: with `||` the dialect refuses (correct); a mutation to `&&` would let it fall
/// through and emit a malformed phi (a silent mis-lowering), which this assertion catches. The
/// interpreter runs such a `Match` fine (it returns only the taken arm), so the honest three-way
/// reduces to "direct paths cover it, the dialect refuses the ill-shaped merge" — never a silent pass.
#[test]
fn dialect_refuses_a_match_with_inconsistent_arm_shapes() {
    // A two-constructor tag-only type `Color = Red | Blue`, matched on `Red`. The two arms return
    // lanes of the **same kind** (Binary) but **different width** (8 vs 4) — so `kind != kind` is
    // false while `width_8 != width_4` is true: the `||`/`&&` distinction is observable here.
    let col = color_registry();
    let bin4 = Value::new(
        Repr::Binary { width: 4 },
        Payload::Bits(vec![true, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .expect("4-bit binary value");
    let node = Node::Match {
        scrutinee: Box::new(Node::Construct {
            ctor: col.ctor_ref("Color", 0).unwrap(), // Red
            args: vec![],
        }),
        alts: vec![
            Alt::Ctor {
                ctor: col.ctor_ref("Color", 0).unwrap(), // Red -> Binary{8}
                binders: vec![],
                body: Node::Const(byte(A)),
            },
            Alt::Ctor {
                ctor: col.ctor_ref("Color", 1).unwrap(), // Blue -> Binary{4}
                binders: vec![],
                body: Node::Const(bin4),
            },
        ],
        default: None,
    };

    match mycelium_mlir::mlir_compile_and_run(&node) {
        Err(DialectError::Unsupported(msg)) => {
            assert!(
                msg.contains("different kind or width") || msg.contains("same repr shape"),
                "the refusal must name the arm-shape-consistency guard; got: {msg}"
            );
        }
        // Compile-time refusal — reached before any toolchain is invoked, so there is NO
        // `ToolchainMissing` escape here: the guard must fire even on a box without libMLIR.
        Ok(v) => panic!(
            "an inconsistent-arm-shape Match must be refused by the dialect (arm-shape guard), got {:?}",
            v.payload()
        ),
        Err(e) => panic!("the arm-shape guard must refuse with Unsupported, got a different error: {e}"),
    }
}

// ─── mutant witness (RFC-0029 §7.5) ─────────────────────────────────────────────────────────────

/// **The mutant witness.** For each in-fragment category (arithmetic, data, certified swap) we
/// compile two *distinct* real in-corpus programs through the MLIR-dialect leg and assert the
/// **same shared M-210 checker** the equivalence test trusts **rejects** the cross pair — modelling
/// exactly what a mis-lowering (a wrong `arith` op, a swapped tag/field load, a wrong swap-direction
/// digit) would produce: a value that doesn't match its own program's interpreter oracle. A
/// differential that could not reject this would be vacuous — this demonstrates it is not, across
/// all three fragments (so M-856's Construct/Match + Swap dialect legs are witnessed here too — no
/// separate M-856 mutant pass is needed).
#[test]
fn mutant_witness_catches_a_divergence_in_every_codegen_leg() {
    if !mycelium_mlir::MlirTools::is_available() {
        eprintln!("mutant-witness: MLIR toolchain absent — skip (env skip, not a false pass)");
        return;
    }

    // ── Arithmetic leg: P = bit.not(A), Q = core.id(A) — different observables. ──
    let p_arith = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let q_arith = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(byte(A))],
    };
    witness_one_leg("arithmetic", &p_arith, &q_arith);

    // ── Data leg (M-856 Construct/Match): P = match Box(A){Mk(b)=>b} = A,
    //    Q = match Box(A){Mk(b)=>bit.not(b)} = not(A) — different observables. ──
    let reg = box_registry();
    let mk_box_a = || Node::Construct {
        ctor: reg.ctor_ref("Box", 0).unwrap(),
        args: vec![Node::Const(byte(A))],
    };
    let p_data = Node::Match {
        scrutinee: Box::new(mk_box_a()),
        alts: vec![Alt::Ctor {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            binders: vec!["b".to_owned()],
            body: Node::Var("b".to_owned()),
        }],
        default: None,
    };
    let q_data = Node::Match {
        scrutinee: Box::new(mk_box_a()),
        alts: vec![Alt::Ctor {
            ctor: reg.ctor_ref("Box", 0).unwrap(),
            binders: vec!["b".to_owned()],
            body: Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("b".to_owned())],
            },
        }],
        default: None,
    };
    witness_one_leg("data (Construct/Match)", &p_data, &q_data);

    // ── Swap leg (M-856): P = swap(-78 -> ternary6), Q = swap(0 -> ternary6) — different values. ──
    let p_swap = swap_node(
        binary(vec![true, false, true, true, false, false, true, false]),
        Repr::Ternary { trits: 6 },
    );
    let q_swap = swap_node(binary(vec![false; 8]), Repr::Ternary { trits: 6 });
    witness_one_leg("certified swap", &p_swap, &q_swap);
}

/// Compile `p` and `q` through the MLIR-dialect leg, assert they genuinely diverge (else the witness
/// would be vacuous), then assert the shared checker rejects the (p, q) pair — the witness proper.
fn witness_one_leg(leg: &str, p: &Node, q: &Node) {
    let (vp, vq) = match (
        mycelium_mlir::mlir_compile_and_run(p),
        mycelium_mlir::mlir_compile_and_run(q),
    ) {
        (Ok(vp), Ok(vq)) => (vp, vq),
        (Err(DialectError::ToolchainMissing(_)), _)
        | (_, Err(DialectError::ToolchainMissing(_))) => {
            return; // env skip
        }
        (rp, rq) => panic!("{leg}: MLIR-dialect errored: {rp:?} / {rq:?}"),
    };
    let obs_p: Observable<'_> = observable(&vp);
    let obs_q: Observable<'_> = observable(&vq);
    assert_ne!(
        obs_p, obs_q,
        "{leg}: witness setup broken — P and Q must diverge observably"
    );
    assert_checker_rejects(&vp, &vq, leg);
}
