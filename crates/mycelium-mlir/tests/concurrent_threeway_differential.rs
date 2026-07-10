//! M-865 — the **concurrent-batch differential**: interp-sequential ≡ interp-parallel (M-862) ≡
//! AOT-parallel ≡ JIT-parallel (RFC-0008 §8; DN-61 §A.2; ADR-034; NFR-7; VR-4/VR-5; G2).
//!
//! [`mycelium_mlir::concurrent`] extends M-862's interpreter-side top-level pure-argument batch to the
//! direct-LLVM AOT and in-process JIT execution paths, dispatched at the Rust **harness level** through
//! the same [`mycelium_sched::scheduler::Scheduler::run_indexed`] entry point M-860's
//! `emit_llvm_ir_many` already uses. This file is that extension's differential: every corpus case is
//! run through
//!
//! 1. the M-110 **reference interpreter**, sequential ([`Interpreter::eval`]) — the trusted base,
//! 2. the M-862 **interpreter parallel-eval** path ([`Interpreter::eval_parallel`]),
//! 3. the **direct-LLVM AOT**, both sequential ([`mycelium_mlir::compile_and_run`]) and the new
//!    harness-parallel entry point ([`mycelium_mlir::compile_and_run_concurrent`]), and
//! 4. the **in-process JIT**, both sequential ([`mycelium_mlir::jit_run`]) and the new harness-parallel
//!    entry point ([`mycelium_mlir::jit_run_concurrent`]),
//!
//! every pair validated through the single shared M-210 checker
//! ([`RefinementRelation::ObservationalEquiv`]), with a `ran_aot`/`ran_jit` **non-vacuity guard**: a
//! skipped/absent-toolchain leg must never masquerade as agreement (mirroring
//! `unified_threeway_differential.rs`'s own `ran_mlir`/`ran_jit` guards), plus a **plan-level**
//! non-vacuity check ([`mycelium_mlir::plan_concurrent`]) that every corpus case genuinely selects the
//! `OpBatch` fan-out — never a silent fall-through to the sequential arm (G2).
//!
//! **Scope (honest, mirrors [`mycelium_mlir::concurrent`]'s own module docs).** The corpus is the
//! `Op`-headed bit/trit element-wise + fixed-width-arithmetic fragment (the same one
//! `unified_threeway_differential.rs`'s `element_wise_and_arithmetic_corpus` four-way-validates, here
//! narrowed to its ≥2-argument entries so every case is a genuine [`ConcurrentPlan::OpBatch`]) — the
//! **only** fragment [`mycelium_mlir::concurrent`] parallelizes. A `Construct`-headed batch is *not*
//! covered here: it is out of the concurrent dispatcher's own scope (see that module's docs for the
//! grounded reason — the direct-LLVM whole-program contract requires a top-level `Lane` result, which
//! a bare `Construct` cannot produce standalone), so there is nothing to differential-check for it yet.
//!
//! **Mutant-witness ([`mutant_witness_catches_a_wrong_index_compose_aot`]/`..._jit`).** A hand-built
//! "wrong-index compose" mutant of the harness dispatcher (run each argument job for real, then
//! recompose in the **wrong** order) models exactly the class of scatter/indexing bug the real
//! dispatcher's determinism argument exists to rule out. The witness asserts the mutant's output
//! genuinely diverges from the honest reference (never a vacuous no-op mutation) and that the same
//! shared M-210 checker the equivalence tests trust **rejects** the (honest, mutant) pair.
//!
//! **Toolchain skip.** AOT needs `llc`/`clang`; JIT needs `clang`. Where absent, that leg returns a
//! `ToolchainMissing` and the test **skips** it (house idiom) — never a false failure — but the
//! non-vacuity guards assert that when the toolchain **is** present, the leg genuinely ran.
//!
//! **Guarantee: `Empirical`** (differential-checked over this corpus plus a demonstrated
//! divergence-catch) — never upgraded to `Proven` without a checked refinement argument (VR-5).

mod common;
use common::{byte, observable, tern, Observable, A, B, ONES};

use mycelium_cert::{check, CheckVerdict, Evidence, RefinementRelation};
use mycelium_core::{GuaranteeStrength, Node, Trit, Value};
use mycelium_interp::{IdentitySwapEngine, Interpreter, PrimRegistry};
use mycelium_mlir::{plan_concurrent, AotError, ConcurrentPlan};
use mycelium_numerics::Certificate;
use mycelium_sched::scheduler::Scheduler;

/// A fresh interpreter over the shared builtin prims + the identity swap engine (no `Swap` nodes in
/// this corpus, so the swap engine choice is inert — mirrors the sibling differentials' convention).
fn interp() -> Interpreter {
    Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
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

/// Assert the shared M-210 checker REJECTS a pair (the mutant-witness half).
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

/// One corpus case: a label plus a **top-level, ≥2-argument, pure `Op`** node — every case here must
/// be a genuine [`ConcurrentPlan::OpBatch`] (asserted in [`assert_agreement`] itself), never a silent
/// fall-through to the sequential arm.
struct Case {
    label: &'static str,
    node: Node,
}

/// The `Op`-headed corpus: the ≥2-argument entries of the same bit/trit element-wise + fixed-width
/// carry-arithmetic fragment `unified_threeway_differential.rs::element_wise_and_arithmetic_corpus`
/// four-way-validates sequentially — the fragment [`mycelium_mlir::concurrent`] actually parallelizes.
/// Includes one non-commutative case ([`trit.sub`]) so a wrong-index compose is *observably* wrong,
/// not accidentally still correct.
fn op_batch_corpus() -> Vec<Case> {
    vec![
        Case {
            label: "bit.and(A,B)",
            node: Node::Op {
                prim: "bit.and".into(),
                args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
            },
        },
        Case {
            label: "bit.or(A,B)",
            node: Node::Op {
                prim: "bit.or".into(),
                args: vec![Node::Const(byte(A)), Node::Const(byte(B))],
            },
        },
        Case {
            label: "bit.xor(A,ONES)",
            node: Node::Op {
                prim: "bit.xor".into(),
                args: vec![Node::Const(byte(A)), Node::Const(byte(ONES))],
            },
        },
        Case {
            label: "trit.add(5,4) over 3 trits",
            node: Node::Op {
                prim: "trit.add".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
                ],
            },
        },
        Case {
            label: "trit.sub(5,4) over 3 trits (non-commutative)",
            node: Node::Op {
                prim: "trit.sub".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])),
                ],
            },
        },
        Case {
            label: "trit.mul(2,3) over 3 trits",
            node: Node::Op {
                prim: "trit.mul".into(),
                args: vec![
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Neg])),
                    Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Zero])),
                ],
            },
        },
        Case {
            label: "bit.and(let x = A in x, B) — a non-trivial closed argument",
            node: Node::Op {
                prim: "bit.and".into(),
                args: vec![
                    Node::Let {
                        id: "x".into(),
                        bound: Box::new(Node::Const(byte(A))),
                        body: Box::new(Node::Var("x".into())),
                    },
                    Node::Const(byte(B)),
                ],
            },
        },
    ]
}

/// Run one case through every leg, each pair M-210-checked, accumulating the `ran_aot`/`ran_jit`
/// non-vacuity flags. Also asserts the **plan-level** non-vacuity guard: this case's node must be a
/// genuine [`ConcurrentPlan::OpBatch`] with the expected width — never a silent fall-through to
/// sequential (G2).
fn assert_agreement(c: &Case, ran_aot: &mut bool, ran_jit: &mut bool) {
    let expected_width = match &c.node {
        Node::Op { args, .. } => args.len(),
        _ => panic!("{}: corpus case must be a top-level Op", c.label),
    };
    assert_eq!(
        plan_concurrent(&c.node),
        ConcurrentPlan::OpBatch {
            width: expected_width
        },
        "{}: must genuinely plan an OpBatch fan-out, not fall through to Sequential",
        c.label
    );

    let interp_seq = interp().eval(&c.node).unwrap_or_else(|e| {
        panic!(
            "{}: interpreter (sequential) must evaluate, got {e:?}",
            c.label
        )
    });
    let interp_par = interp().eval_parallel(&c.node).unwrap_or_else(|e| {
        panic!(
            "{}: interpreter (M-862 parallel) must evaluate, got {e:?}",
            c.label
        )
    });
    assert_eq!(
        observable(&interp_seq),
        observable(&interp_par),
        "{}: interp-sequential vs interp-parallel (M-862) diverged",
        c.label
    );
    assert_checker_validates(
        &interp_seq,
        &interp_par,
        c.label,
        "interp-sequential<->interp-parallel",
    );

    let aot_seq = match mycelium_mlir::compile_and_run(&c.node) {
        Ok(v) => Some(v),
        Err(AotError::ToolchainMissing(_)) => None,
        Err(e) => panic!("{}: AOT sequential errored: {e}", c.label),
    };
    let aot_par = match mycelium_mlir::compile_and_run_concurrent(&c.node) {
        Ok(v) => Some(v),
        Err(AotError::ToolchainMissing(_)) => None,
        Err(e) => panic!("{}: AOT-parallel (M-865) errored: {e}", c.label),
    };
    if let (Some(seq), Some(par)) = (&aot_seq, &aot_par) {
        *ran_aot = true;
        assert_eq!(
            observable(seq),
            observable(par),
            "{}: AOT-sequential vs AOT-parallel (M-865) diverged",
            c.label
        );
        assert_checker_validates(seq, par, c.label, "AOT-sequential<->AOT-parallel");
        assert_eq!(
            observable(&interp_seq),
            observable(par),
            "{}: interp vs AOT-parallel (M-865) diverged",
            c.label
        );
        assert_checker_validates(&interp_seq, par, c.label, "interp<->AOT-parallel");
    }

    let jit_seq = match mycelium_mlir::jit_run(&c.node) {
        Ok(v) => Some(v),
        Err(AotError::ToolchainMissing(_)) => None,
        Err(e) => panic!("{}: JIT sequential errored: {e}", c.label),
    };
    let jit_par = match mycelium_mlir::jit_run_concurrent(&c.node) {
        Ok(v) => Some(v),
        Err(AotError::ToolchainMissing(_)) => None,
        Err(e) => panic!("{}: JIT-parallel (M-865) errored: {e}", c.label),
    };
    if let (Some(seq), Some(par)) = (&jit_seq, &jit_par) {
        *ran_jit = true;
        assert_eq!(
            observable(seq),
            observable(par),
            "{}: JIT-sequential vs JIT-parallel (M-865) diverged",
            c.label
        );
        assert_checker_validates(seq, par, c.label, "JIT-sequential<->JIT-parallel");
        assert_eq!(
            observable(&interp_seq),
            observable(par),
            "{}: interp vs JIT-parallel (M-865) diverged",
            c.label
        );
        assert_checker_validates(&interp_seq, par, c.label, "interp<->JIT-parallel");
    }

    if let (Some(a), Some(j)) = (&aot_par, &jit_par) {
        assert_eq!(
            observable(a),
            observable(j),
            "{}: AOT-parallel vs JIT-parallel diverged",
            c.label
        );
        assert_checker_validates(a, j, c.label, "AOT-parallel<->JIT-parallel");
    }
}

/// **The M-865 unified concurrent-batch differential entrypoint.** interp-sequential ≡
/// interp-parallel (M-862) ≡ AOT-parallel ≡ JIT-parallel over the `Op`-headed batch corpus, each pair
/// M-210-checked, then the `ran_aot`/`ran_jit` non-vacuity guards: with real `llc`/`clang` present in
/// this environment, a vacuous all-skipped pass must never be reported as agreement (G2).
#[test]
fn concurrent_interp_aot_jit_differential() {
    let mut ran_aot = false;
    let mut ran_jit = false;
    for c in op_batch_corpus() {
        assert_agreement(&c, &mut ran_aot, &mut ran_jit);
    }
    if toolchain_present() {
        assert!(
            ran_aot,
            "llc/clang are available but no case exercised the AOT-parallel leg — vacuous"
        );
        assert!(
            ran_jit,
            "clang is available but no case exercised the JIT-parallel leg — vacuous"
        );
    }
}

/// Probe whether the native toolchain (`llc`+`clang`) is usable right now, via a trivial in-subset
/// compile. Used by the non-vacuity guard above (mirrors `unified_threeway_differential.rs`'s
/// `clang_present`).
fn toolchain_present() -> bool {
    let trivial = Node::Op {
        prim: "bit.not".into(),
        args: vec![Node::Const(byte(A))],
    };
    !matches!(
        mycelium_mlir::compile_and_run(&trivial),
        Err(AotError::ToolchainMissing(_))
    )
}

/// Repeated runs of the AOT-parallel and JIT-parallel entry points must not perturb the result —
/// Exact by construction (the M-865 DoD), not "usually agrees" (mirrors
/// `llvm.rs::tests::parallel_emit_is_stable_under_repeated_runs`, M-860's own stability check, one
/// level up).
#[test]
fn concurrent_dispatch_is_stable_under_repeated_runs() {
    if !toolchain_present() {
        eprintln!("stability check: llc/clang absent — skip (env skip, not a false pass)");
        return;
    }
    for c in op_batch_corpus() {
        let first_aot = mycelium_mlir::compile_and_run_concurrent(&c.node)
            .unwrap_or_else(|e| panic!("{}: AOT-parallel errored: {e}", c.label));
        let first_jit = mycelium_mlir::jit_run_concurrent(&c.node)
            .unwrap_or_else(|e| panic!("{}: JIT-parallel errored: {e}", c.label));
        for _ in 0..5 {
            assert_eq!(
                observable(&mycelium_mlir::compile_and_run_concurrent(&c.node).unwrap()),
                observable(&first_aot),
                "{}: AOT-parallel result is not stable across repeated runs",
                c.label
            );
            assert_eq!(
                observable(&mycelium_mlir::jit_run_concurrent(&c.node).unwrap()),
                observable(&first_jit),
                "{}: JIT-parallel result is not stable across repeated runs",
                c.label
            );
        }
    }
}

// ─── mutant witness: a wrong-index compose ──────────────────────────────────────────────────────

/// A hand-built "wrong-index compose" mutant of the AOT harness dispatcher: dispatches each argument
/// job through the scheduler exactly as `mycelium_mlir::concurrent`'s real dispatcher does, but
/// recomposes the results in the **wrong (reversed) order** — modelling exactly the scatter/indexing
/// bug class the real dispatcher's spawn-order + original-index discipline exists to rule out.
fn broken_wrong_index_compose_aot(node: &Node) -> Result<Value, AotError> {
    let Node::Op { prim, args } = node else {
        panic!("witness helper requires a top-level Op node")
    };
    let jobs: Vec<_> = args
        .iter()
        .map(|arg| {
            let arg = arg.clone();
            move || mycelium_mlir::compile_and_run(&arg)
        })
        .collect();
    let results: Vec<Result<Value, AotError>> = Scheduler::new().run_indexed(jobs, None, None);
    let mut values: Vec<Value> = Vec::with_capacity(results.len());
    for r in results {
        values.push(r?);
    }
    values.reverse(); // the injected bug: composes the batch in the wrong order
    let recomposed = Node::Op {
        prim: prim.clone(),
        args: values.into_iter().map(Node::Const).collect(),
    };
    mycelium_mlir::compile_and_run(&recomposed)
}

/// The JIT-path twin of [`broken_wrong_index_compose_aot`] — same injected bug, over `jit_run`.
fn broken_wrong_index_compose_jit(node: &Node) -> Result<Value, AotError> {
    let Node::Op { prim, args } = node else {
        panic!("witness helper requires a top-level Op node")
    };
    let jobs: Vec<_> = args
        .iter()
        .map(|arg| {
            let arg = arg.clone();
            move || mycelium_mlir::jit_run(&arg)
        })
        .collect();
    let results: Vec<Result<Value, AotError>> = Scheduler::new().run_indexed(jobs, None, None);
    let mut values: Vec<Value> = Vec::with_capacity(results.len());
    for r in results {
        values.push(r?);
    }
    values.reverse(); // the injected bug: composes the batch in the wrong order
    let recomposed = Node::Op {
        prim: prim.clone(),
        args: values.into_iter().map(Node::Const).collect(),
    };
    mycelium_mlir::jit_run(&recomposed)
}

/// **The mutant witness (AOT leg).** `trit.sub(5,4)` is non-commutative, so a wrong-index (reversed)
/// compose genuinely changes the observable result (5-4=1 vs 4-5=-1) — never a vacuous no-op mutation.
/// Asserts the shared M-210 checker **rejects** the (honest, mutant) pair.
#[test]
fn mutant_witness_catches_a_wrong_index_compose_aot() {
    if !toolchain_present() {
        eprintln!("mutant-witness (AOT): llc/clang absent — skip (env skip, not a false pass)");
        return;
    }
    let node = Node::Op {
        prim: "trit.sub".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])), // 5
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])), // 4
        ],
    };
    let honest = mycelium_mlir::compile_and_run_concurrent(&node).expect("honest AOT-parallel run");
    let mutant = broken_wrong_index_compose_aot(&node).expect("mutant AOT run");

    let obs_honest: Observable<'_> = observable(&honest);
    let obs_mutant: Observable<'_> = observable(&mutant);
    assert_ne!(
        obs_honest, obs_mutant,
        "witness setup broken — the wrong-index compose must genuinely diverge (non-commutative op)"
    );
    assert_checker_rejects(&honest, &mutant, "AOT wrong-index compose");
}

/// **The mutant witness (JIT leg).** Same shape as the AOT witness above, over `jit_run`.
#[test]
fn mutant_witness_catches_a_wrong_index_compose_jit() {
    if !toolchain_present() {
        eprintln!("mutant-witness (JIT): clang absent — skip (env skip, not a false pass)");
        return;
    }
    let node = Node::Op {
        prim: "trit.sub".into(),
        args: vec![
            Node::Const(tern(vec![Trit::Pos, Trit::Neg, Trit::Neg])), // 5
            Node::Const(tern(vec![Trit::Zero, Trit::Pos, Trit::Pos])), // 4
        ],
    };
    let honest = mycelium_mlir::jit_run_concurrent(&node).expect("honest JIT-parallel run");
    let mutant = broken_wrong_index_compose_jit(&node).expect("mutant JIT run");

    let obs_honest: Observable<'_> = observable(&honest);
    let obs_mutant: Observable<'_> = observable(&mutant);
    assert_ne!(
        obs_honest, obs_mutant,
        "witness setup broken — the wrong-index compose must genuinely diverge (non-commutative op)"
    );
    assert_checker_rejects(&honest, &mutant, "JIT wrong-index compose");
}
