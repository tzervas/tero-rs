//! M-729 — the unified **interp ≡ AOT ≡ JIT** three-way codegen differential, **mutant-witnessed**
//! (RFC-0029 §7.5; NFR-7; VR-4; RR-12; ADR-009; G2/VR-5).
//!
//! This is the closure/durability gate for E15-1. It drives the bit/trit subset through the **three
//! named execution modes** ([`mycelium_mlir::run_mode`], M-727) —
//!
//! 1. [`ExecMode::Interpreter`] — the M-110 reference interpreter (the trusted base),
//! 2. [`ExecMode::Aot`] — the [`mycelium_mlir::aot`] env-machine, and
//! 3. [`ExecMode::Jit`] — the M-340 in-process compiled path —
//!
//! and asserts all three are **observably equivalent** (`repr + payload + guarantee`), each pair
//! **validated through the single shared M-210 checker** ([`RefinementRelation::ObservationalEquiv`]).
//! It unifies `jit_differential.rs` (interp↔JIT) and the env-machine differential (interp↔AOT) into one
//! harness over a shared corpus, routed through the *formalized* mode API.
//!
//! **Mutant-witnessed (RFC-0029 §7.5; the `Empirical` bar).** The differential is meaningful only if a
//! codegen divergence would actually be **caught**. [`mutant_witness_three_way_diff_catches_a_divergence`]
//! demonstrates exactly that: a backend that mis-lowered a program (modelled by a value a *mutated*
//! codegen would produce — the value of a *different* in-corpus program) is **rejected** by the same
//! shared checker the equivalence test trusts. So the three-way agreement is non-vacuous, and the
//! M-729 claim is `Empirical` (a divergence is demonstrably caught), not `Declared`.
//!
//! **Toolchain skip.** The JIT path needs `clang`; where it is absent the JIT mode returns
//! `ModeError::ToolchainMissing` and the test **skips** that path (the house idiom) — but the
//! interp↔AOT edge still runs, and a non-vacuity guard asserts that *when* `clang` is present the JIT
//! genuinely ran (never a silent vacuous pass).
//!
//! **Guarantee:** `Empirical` — the differential is evidence the three backends agree over the corpus
//! and that a divergence is caught; never upgraded to `Proven` absent a checked proof (VR-5).

mod common;
use common::{byte, observable, tern, A, B, ONES};

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, Node, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, PrimRegistry, SwapEngine};
use mycelium_mlir::{ExecMode, ModeError};
use mycelium_numerics::Certificate;

/// Fresh interpreter/AOT config for one `run_mode` call (the dispatcher consumes it).
fn config() -> (PrimRegistry, Box<dyn SwapEngine>) {
    (PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
}

/// The bit/trit subset all three backends compile — the M-729 corpus. Element-wise bit ops, trit
/// negation, and the M-301 ternary carry arithmetic (in range). A small deterministic set.
fn corpus() -> Vec<Node> {
    let cst = |bits: [bool; 8]| Node::Const(byte(bits));
    vec![
        cst(A),
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
        // let / nested: not(a xor b)
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
        Node::Op {
            prim: "trit.neg".into(),
            args: vec![Node::Const(tern(vec![
                Trit::Pos,
                Trit::Zero,
                Trit::Neg,
                Trit::Pos,
            ]))],
        },
        // M-301 trit carry: 5 + 4 = 9 over 3 trits (in range).
        Node::Op {
            prim: "trit.add".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
            ],
        },
        // trit.mul: 2 * 3 = 6 over 3 trits.
        Node::Op {
            prim: "trit.mul".into(),
            args: vec![
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
            ],
        },
    ]
}

/// Run a node under an explicit mode; `None` if the JIT toolchain is absent (the only skippable
/// mode), panicking on any other unexpected error so a real bug is never hidden as a skip.
fn run(mode: ExecMode, node: &Node, i: usize) -> Option<Value> {
    let (p, s) = config();
    match mycelium_mlir::run_mode(mode, node, p, s) {
        Ok(v) => Some(v),
        Err(ModeError::ToolchainMissing(_)) => None, // JIT-only environment skip
        Err(e) => panic!("program #{i}: {} mode errored: {e}", mode.name()),
    }
}

/// Assert the shared M-210 checker validates a pair as observationally equivalent.
fn assert_checker_validates(a: &Value, b: &Value, label: &str, i: usize) {
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
        "program #{i}: the shared checker must validate the {label} pair"
    );
}

/// M-729: interp ≡ AOT ≡ JIT over the subset, every pair validated through the shared M-210 checker.
/// The JIT path skips gracefully if `clang` is absent; the interp↔AOT edge always runs.
#[test]
fn interp_aot_jit_are_three_way_equivalent() {
    let mut ran_jit = false;
    for (i, node) in corpus().iter().enumerate() {
        let interp = run(ExecMode::Interpreter, node, i).expect("interpreter always available");
        let aot = run(ExecMode::Aot, node, i).expect("aot env-machine always available");
        let jit = run(ExecMode::Jit, node, i);

        // Edge 1: interp ≡ AOT (always runs).
        assert_eq!(
            observable(&interp),
            observable(&aot),
            "program #{i}: interp ≠ aot"
        );
        assert_checker_validates(&interp, &aot, "interp↔AOT", i);

        if let Some(jit) = jit {
            ran_jit = true;
            // Edge 2: interp ≡ JIT. Mutant-witness: a wrong store offset / fn signature would diverge.
            assert_eq!(
                observable(&interp),
                observable(&jit),
                "program #{i}: interp ≠ jit"
            );
            assert_checker_validates(&interp, &jit, "interp↔JIT", i);
            // Edge 3: AOT ≡ JIT (the third edge of the triangle).
            assert_eq!(
                observable(&aot),
                observable(&jit),
                "program #{i}: aot ≠ jit"
            );
            assert_checker_validates(&aot, &jit, "AOT↔JIT", i);
        }
    }
    // Non-vacuity guard: if `clang` is present, the JIT must actually have run on ≥1 program (never a
    // silent vacuous pass where every node skipped). `clang` presence is probed via a trivial compile.
    if clang_present() {
        assert!(
            ran_jit,
            "clang is present but the JIT mode ran on no program — vacuous three-way pass"
        );
    }
}

/// **The mutant-witness (RFC-0029 §7.5).** The three-way differential is `Empirical` only if a codegen
/// divergence is demonstrably **caught**. Here we model a backend that mis-lowered program `P` (so it
/// produced the value of a *different* in-corpus program `Q`) and assert the **same shared M-210
/// checker** the equivalence test trusts **rejects** the (correct-`P`, mutant-`Q`) pair. A differential
/// that could not reject this would be vacuous — this proves it is not.
#[test]
fn mutant_witness_three_way_diff_catches_a_divergence() {
    // P = bit.not(A); Q = core.id(A) — two distinct in-subset programs with different observables.
    let p = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let q = Node::Op {
        prim: "core.id".into(),
        args: vec![Node::Const(byte(A))],
    };

    // The reference value of P (interpreter), and the value a backend that *mis-lowered P to Q's
    // behaviour* would produce (the JIT value of Q — a real compiled value, not a hand-built fake).
    let interp_p = run(ExecMode::Interpreter, &p, 0).expect("interp P");
    let mutant_value = match run(ExecMode::Jit, &q, 1) {
        Some(v) => v, // the JIT genuinely ran Q
        None => {
            // clang absent — fall back to the AOT value of Q (still a real, independent codegen path),
            // so the witness is non-vacuous even without the JIT toolchain.
            run(ExecMode::Aot, &q, 1).expect("aot Q")
        }
    };

    // Sanity: P and Q really do differ observably (else the witness would be vacuous).
    assert_ne!(
        observable(&interp_p),
        observable(&mutant_value),
        "witness setup broken: P and Q must differ observably"
    );

    // THE WITNESS: the shared checker REJECTS the divergent pair — so a backend that lowered P to Q's
    // behaviour would be caught by the exact comparison the equivalence test uses (G2; not vacuous).
    let verdict = check(
        &interp_p,
        &mutant_value,
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    );
    assert!(
        matches!(verdict, CheckVerdict::NotValidated { .. }),
        "the three-way differential's checker must REJECT a codegen divergence \
         (mutant-witness), got {verdict:?}"
    );
}

/// Probe whether `clang` can compile a trivial in-subset kernel right now (the JIT toolchain). Used by
/// the non-vacuity guard — `true` means the JIT genuinely should have run.
fn clang_present() -> bool {
    let trivial = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    let (p, s) = config();
    !matches!(
        mycelium_mlir::run_mode(ExecMode::Jit, &trivial, p, s),
        Err(ModeError::ToolchainMissing(_))
    )
}
