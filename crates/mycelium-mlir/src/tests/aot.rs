//! In-crate white-box tests for `aot.rs` — the trampolined env-machine (M-150/M-342/M-347), its
//! budget behavior (DN-05/RFC-0014 §4.8/RFC-0041 W3½), and the M-996 TCO (RFC-0041 §4.0/§4.6).
//! Extracted from the former inline `#[cfg(test)] mod tests` per the CLAUDE.md test-layout rule
//! (as-touched, M-797); white-box access via `use crate::aot::*` (`Frame`/`TcoTrace`/the traced
//! runner are `pub(crate)`).
//!
//! **M-996 (maintainer decision 2026-07-06) — the two intentional expectation changes live here:**
//! a deep *terminating* tail loop is now `Ok(value)` where it was `DepthLimit` (see
//! `a_deep_match_driven_tail_loop_succeeds_in_bounded_depth`), and a *divergent* tail loop is now
//! `FuelExhausted` where it was `DepthLimit` (see
//! `a_divergent_tail_loop_is_fuel_exhausted_not_depth_limited`). The graceful depth-ceiling
//! property itself stays pinned — via the correct (non-tail) witness
//! (`the_depth_ceiling_is_an_explicit_graceful_error`). No property was deleted; the witnesses
//! moved to shapes the ratified §4.0 metric actually charges (VR-5/G2).
//!
//! Guarantee tags: behavioral assertions are `Empirical` (checked by running the machine); the
//! frame-size pin is a `Declared` baseline.

use crate::aot::*;

use mycelium_core::{Alt, Meta, Node, Payload, Provenance, Repr, Value};
use mycelium_interp::{
    Budgets, EffectBudget, EffectKind, EvalError, IdentitySwapEngine, Interpreter, PrimRegistry,
};

use crate::budget::DEFAULT_PER_FRAME_BYTES;

// ─── fixtures ───────────────────────────────────────────────────────────────────────────────────

fn byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A `Binary{16}` word constant (MSB-first bits) — wide enough for the 10_000-iteration loops.
fn word(n: u16) -> Value {
    let bits: Vec<bool> = (0..16).rev().map(|i| (n >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 16 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// `(fix f => λx. f x) c` — unfolds forever; the recursive call is in **tail** position (the lambda
/// body IS the self-application), so under the M-996 TCO it runs at O(1) control-stack depth and
/// the budget that trips is **fuel** (the designed non-termination backstop), never depth.
fn spin() -> Node {
    Node::App {
        func: Box::new(Node::Fix {
            name: "f".into(),
            body: Box::new(Node::Lam {
                param: "x".into(),
                body: Box::new(Node::App {
                    func: Box::new(Node::Var("f".into())),
                    arg: Box::new(Node::Var("x".into())),
                }),
            }),
        }),
        arg: Box::new(Node::Const(byte())),
    }
}

/// The M-996 deep-tail witness: a `match`-driven countdown
/// `count = fix count => λn. match n { 0 => n, default => count(bin.sub(n, 1)) }; count(n0)`.
/// The recursive call is the default arm's result and the match is the whole body, so every
/// iteration is a tail iteration (both the `Match` continuation and the `App` continuation are
/// passthroughs) — under §4.0 it charges **no depth**, only fuel.
fn countdown(n0: u16) -> Node {
    Node::App {
        func: Box::new(Node::Fix {
            name: "count".into(),
            body: Box::new(Node::Lam {
                param: "n".into(),
                body: Box::new(Node::Match {
                    scrutinee: Box::new(Node::Var("n".into())),
                    alts: vec![Alt::Lit {
                        value: word(0),
                        body: Node::Var("n".into()),
                    }],
                    default: Some(Box::new(Node::App {
                        func: Box::new(Node::Var("count".into())),
                        arg: Box::new(Node::Op {
                            prim: "bin.sub".into(),
                            args: vec![Node::Var("n".into()), Node::Const(word(1))],
                        }),
                    })),
                }),
            }),
        }),
        arg: Box::new(Node::Const(word(n0))),
    }
}

/// The M-996 non-tail correctness guard (the shape the L1 guard
/// `l1_eval_non_tail_self_call_still_refuses_depth` pins):
/// `sum = fix sum => λn. match n { 0 => n, default => bin.add(n, sum(bin.sub(n, 1))) }; sum(n0)`.
/// The recursive call's result is **consumed** by `bin.add`, so its continuation is NOT a
/// passthrough — every level keeps its frame and deep recursion must still refuse at the depth
/// ceiling (the TCO must never over-elide).
fn non_tail_sum(n0: u16) -> Node {
    Node::App {
        func: Box::new(Node::Fix {
            name: "sum".into(),
            body: Box::new(Node::Lam {
                param: "n".into(),
                body: Box::new(Node::Match {
                    scrutinee: Box::new(Node::Var("n".into())),
                    alts: vec![Alt::Lit {
                        value: word(0),
                        body: Node::Var("n".into()),
                    }],
                    default: Some(Box::new(Node::Op {
                        prim: "bin.add".into(),
                        args: vec![
                            Node::Var("n".into()),
                            Node::App {
                                func: Box::new(Node::Var("sum".into())),
                                arg: Box::new(Node::Op {
                                    prim: "bin.sub".into(),
                                    args: vec![Node::Var("n".into()), Node::Const(word(1))],
                                }),
                            },
                        ],
                    })),
                }),
            }),
        }),
        arg: Box::new(Node::Const(word(n0))),
    }
}

// ─── carried-over pins (unchanged from the pre-extraction inline module) ────────────────────────

/// RFC-0041 §4.2 / W2 residual: pin the AOT env-machine `Frame`'s heap footprint under the shared
/// per-machine baseline ([`mycelium_workstack::MAX_FRAME_BYTES`]). The AOT `Frame` is the largest of
/// the three machines' frame/value structs and *set* that 384-byte baseline (it is ~336 B here, the
/// pre-W3½ ~328 B plus the one-pointer `DepthGuard`); pinning it means a field addition that grows
/// it past the ceiling **fails CI here, not in production** (the ADR-041 frame-size lesson). On an
/// intended growth, re-measure all three machines and bump `MAX_FRAME_BYTES` (a `Declared` baseline).
#[test]
fn aot_frame_size_is_pinned_under_the_shared_baseline() {
    let frame = std::mem::size_of::<Frame<'static>>() as u64;
    assert!(
        frame <= mycelium_workstack::MAX_FRAME_BYTES,
        "AOT Frame is {frame} B, over the shared MAX_FRAME_BYTES ceiling of {} B — re-measure all \
         three machines and bump the baseline if this growth is intended (RFC-0041 §4.2)",
        mycelium_workstack::MAX_FRAME_BYTES
    );
}

#[test]
fn runs_a_let_op_program() {
    // let a = byte in bit.not(a)
    let node = Node::Let {
        id: "a".into(),
        bound: Box::new(Node::Const(byte())),
        body: Box::new(Node::Op {
            prim: "bit.not".into(),
            args: vec![Node::Var("a".into())],
        }),
    };
    let out = run(&node, &PrimRegistry::with_builtins(), &IdentitySwapEngine).unwrap();
    let expected: Vec<bool> = match byte().payload() {
        Payload::Bits(b) => b.iter().map(|&x| !x).collect(),
        _ => unreachable!(),
    };
    assert_eq!(out.payload(), &Payload::Bits(expected));
}

#[test]
fn free_variable_is_explicit() {
    let node = Node::Var("nope".into());
    assert_eq!(
        run(&node, &PrimRegistry::with_builtins(), &IdentitySwapEngine),
        Err(EvalError::FreeVariable("nope".into()))
    );
}

#[test]
fn applies_a_closure_in_the_env_machine() {
    // (λx. bit.not(x)) byte  — exercises Lam capture + App + closure-body eval (M-342).
    let node = Node::App {
        func: Box::new(Node::Lam {
            param: "x".into(),
            body: Box::new(Node::Op {
                prim: "bit.not".into(),
                args: vec![Node::Var("x".into())],
            }),
        }),
        arg: Box::new(Node::Const(byte())),
    };
    let out = run(&node, &PrimRegistry::with_builtins(), &IdentitySwapEngine).unwrap();
    let expected: Vec<bool> = match byte().payload() {
        Payload::Bits(b) => b.iter().map(|&x| !x).collect(),
        _ => unreachable!(),
    };
    assert_eq!(out.payload(), &Payload::Bits(expected));
}

#[test]
fn a_nonproductive_recursion_is_an_explicit_budget_error_not_an_abort() {
    // M-347: with the trampoline the env-machine is O(1) host stack, so a divergent recursion is a
    // graceful explicit budget error, never a host-stack abort and never a hang. (Pre-trampoline,
    // large fuel overflowed the stack.) Since M-996 the tail `spin` no longer accumulates depth, so
    // the budget that trips is deterministically FUEL; the assertion keeps the original disjunction
    // (the property under test is gracefulness, whichever budget bites). The fuel constant is
    // 2M (was 50M): pre-TCO the run stopped early at the dynamic depth ceiling, post-TCO it burns
    // the whole fuel — 50M of pure fuel burn is the same property at 25× the test cost.
    let r = run_core_with_fuel(
        &spin(),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        2_000_000,
    );
    assert!(
        matches!(
            r,
            Err(EvalError::DepthLimit { .. }) | Err(EvalError::FuelExhausted)
        ),
        "expected a graceful budget error, got {r:?}"
    );
}

#[test]
fn a_declared_alloc_effect_budget_overruns_gracefully_at_runtime() {
    // RFC-0014 §4.8 (completed): the recovery `Budgets` ledger is wired into the env-machine's
    // budget enforcement. A declared `alloc` effect budget bounds control-stack *memory* (the
    // opt-in sibling of the depth ceiling) and an overrun is the unified, graceful
    // `EvalError::EffectBudget` — the runtime-path extension of the RFC-0014 I4 bounded-overrun
    // test, on the *same* channel as `FuelExhausted`/`DepthLimit`, never an OOM/hang. (M-996 note:
    // TCO-elided frames allocate nothing and charge nothing, but `spin`'s per-iteration `Fix`
    // unfold still pushes its real `ApplyThen` frame, so the declared budget still overruns.)
    let frames = 10u64; // allow 10 frames' worth of alloc, then the next frame overruns
    let mut budgets = Budgets::new().with(EffectBudget::Bytes(frames * DEFAULT_PER_FRAME_BYTES));
    let r = run_core_with_effects(
        &spin(),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        1_000_000, // fuel ≫ alloc budget
        1_000_000, // depth ceiling ≫ alloc budget, so the *effect* budget bites first
        &mut budgets,
    );
    match r {
        Err(EvalError::EffectBudget(e)) => {
            assert_eq!(e.kind, EffectKind::Alloc);
            assert_eq!(e.requested, DEFAULT_PER_FRAME_BYTES);
            assert_eq!(e.remaining, 0);
        }
        other => panic!("expected a graceful EffectBudget overrun, got {other:?}"),
    }
}

#[test]
fn an_absent_alloc_budget_leaves_runtime_behaviour_unchanged() {
    // I5 (opt-in): the default empty ledger declares no `alloc` budget, so the env-machine charges
    // nothing and the fuel/depth budgets remain the sole guards — identical to `run_core_with_budget`.
    // (M-996: the shared outcome for the divergent TAIL `spin` is now `FuelExhausted` — the depth
    // ceiling no longer bites a tail loop; the point of THIS test is unchanged: an absent ledger
    // changes nothing relative to the budget-only entry.)
    let with_empty_ledger = run_core_with_effects(
        &spin(),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        10_000,
        64,
        &mut Budgets::new(),
    );
    let budget_only = run_core_with_budget(
        &spin(),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        10_000,
        64,
    );
    assert_eq!(
        with_empty_ledger, budget_only,
        "an absent alloc ledger must not change the outcome"
    );
    assert_eq!(with_empty_ledger, Err(EvalError::FuelExhausted));
}

// ─── the graceful depth ceiling — pinned via the correct (non-tail) witness (M-996) ─────────────

/// The depth ceiling stays an **explicit graceful error** — pinned via a **non-tail** witness
/// (`sum(n) = add(n, sum(n-1))`: the recursive call's value is consumed, so §4.0 charges every
/// level). Pre-M-996 this test's witness was the *tail* `spin()`, which the ratified §4.0 metric
/// says should never have charged depth; the property (an explicit `DepthLimit`, never an abort)
/// is unchanged — only the witness moved to a shape the metric actually charges. This doubles as
/// the **no-over-elision guard**: TCO must not elide a call whose result the caller still needs.
#[test]
fn the_depth_ceiling_is_an_explicit_graceful_error() {
    let r = run_core_with_budget(
        &non_tail_sum(10_000),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        1_000_000, // fuel ≫ depth, so the depth ceiling bites first
        64,
    );
    assert_eq!(r, Err(EvalError::DepthLimit { limit: 64 }));
}

// ─── M-996: the two intentional behavior shifts + the elision witness ───────────────────────────

/// **The rewritten divergence pin (maintainer decision 2026-07-06, M-996).** This test formerly
/// asserted `spin()` → `DepthLimit {{ 64 }}`: without TCO the tail self-application accumulated one
/// frame per iteration, so the *space* budget tripped first. With TCO the divergent **tail** loop
/// runs at O(1) depth — exactly as it long has on the L1 interpreter — and the budget that
/// correctly trips is **fuel**, the designed non-termination backstop (time, not space). The owner
/// explicitly approved this `DepthLimit → FuelExhausted` shift for divergent tail loops (the
/// pre-production freeze is not a delivered-core guarantee); the depth-ceiling property itself
/// remains pinned by `the_depth_ceiling_is_an_explicit_graceful_error` on the non-tail witness —
/// nothing was silently deleted (G2/VR-5).
#[test]
fn a_divergent_tail_loop_is_fuel_exhausted_not_depth_limited() {
    let r = run_core_with_budget(
        &spin(),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        10_000, // fuel is what trips: the tail loop no longer consumes depth
        64,     // a small ceiling the pre-TCO machine hit — now never reached
    );
    assert_eq!(r, Err(EvalError::FuelExhausted));
}

/// **The M-996 headline: a deep terminating tail loop succeeds in bounded depth** (was
/// `DepthLimit`). `count(10_000)` under a 64-frame ceiling completes and returns 0 — tail
/// iterations charge no depth (RFC-0041 §4.0), so the loop's cost is fuel, not space. The elision
/// is **witnessed, not inferred** (house rule #2): the `TcoTrace` must record at least one elision
/// per iteration (each iteration elides the tail-`Match` continuation *and* the tail-`App`
/// continuation, so ≥ 10_000 is a conservative floor). Also re-runs the exact M-996 investigation
/// probe — `count(500)` @ depth 1000, formerly `DepthLimit {{ 1000 }}` — as a pinned flip.
#[test]
fn a_deep_match_driven_tail_loop_succeeds_in_bounded_depth() {
    let prims = PrimRegistry::with_builtins();

    let (r, trace) = run_core_with_effects_traced(
        &countdown(10_000),
        &prims,
        &IdentitySwapEngine,
        100_000, // fuel ≫ 10_000 unfolds
        64,      // depth ≪ iterations: only TCO makes this succeed
        &mut Budgets::new(),
    );
    let v = r.expect("a 10_000-iteration tail countdown must complete under a 64-frame ceiling");
    match &v {
        mycelium_core::CoreValue::Repr(rv) => {
            assert_eq!(
                rv.payload(),
                &Payload::Bits(vec![false; 16]),
                "count(10_000) must terminate at 0"
            );
        }
        other => panic!("expected a Repr result, got {other:?}"),
    }
    assert!(
        trace.total_elided >= 10_000,
        "the TCO must be witnessed: expected >= 10_000 elided frames (one per tail iteration at \
         minimum), got {}",
        trace.total_elided
    );

    // The investigation's fixed-budget probe: count(500) @ depth 1000 was DepthLimit{1000} before
    // M-996 (measured 2026-07-06, debug profile); it must now pass.
    let probe = run_core_with_budget(
        &countdown(500),
        &prims,
        &IdentitySwapEngine,
        1_000_000,
        1000,
    );
    assert!(
        probe.is_ok(),
        "count(500) @ depth 1000 must now succeed (was DepthLimit {{ 1000 }}), got {probe:?}"
    );
}

/// **The §5.1 family-parity convergence this change exists for:** the same deep tail-loop program
/// now produces the SAME outcome on the reference interpreter and on the AOT env-machine — both
/// `Ok`, same observable (repr + payload + guarantee). Pre-M-996 the interpreter succeeded while
/// the env-machine refused `DepthLimit` at the same budget family — a live parity violation of the
/// ratified §4.0 metric. (This is the in-crate L0-interp ≡ AOT leg — the crate's M-151 differential
/// partner; the L1-eval leg lives in `mycelium-l1/tests/` (that crate depends on this one), where
/// `l1_eval_tco_match_arm_tail_call_is_elided` already pins the same `count(10_000)` shape `Ok`.)
#[test]
fn deep_tail_loop_interp_and_aot_env_machine_agree() {
    let prog = countdown(10_000);
    let prims = PrimRegistry::with_builtins();

    let interp = Interpreter::new(PrimRegistry::with_builtins(), Box::new(IdentitySwapEngine))
        .with_fuel(1_000_000)
        .eval_core(&prog)
        .expect("interp: the deep tail loop completes");
    let aot = run_core_with_budget(&prog, &prims, &IdentitySwapEngine, 1_000_000, 4096)
        .expect("AOT env-machine: the deep tail loop completes (M-996 — was DepthLimit)");

    match (&interp, &aot) {
        (mycelium_core::CoreValue::Repr(iv), mycelium_core::CoreValue::Repr(av)) => {
            assert_eq!(iv.repr(), av.repr(), "repr must agree");
            assert_eq!(iv.payload(), av.payload(), "payload must agree");
            assert_eq!(
                iv.meta().guarantee(),
                av.meta().guarantee(),
                "guarantee must agree"
            );
        }
        other => panic!("expected two Repr results, got {other:?}"),
    }
}

/// A tail call **at** the depth ceiling still succeeds: an elided call reserves no depth unit, so a
/// program whose non-tail nesting sits exactly at the ceiling can still iterate. (Depth 2 leaves no
/// headroom for a per-iteration net charge — a leak of even one guard per iteration fails this.)
#[test]
fn tail_iterations_do_not_leak_depth_guards() {
    // countdown's transient shape needs 1 frame (the Fix unfold's ApplyThen); a ceiling of 2 gives
    // exactly that plus zero slack across 1_000 iterations — any per-iteration depth leak refuses.
    let r = run_core_with_budget(
        &countdown(1_000),
        &PrimRegistry::with_builtins(),
        &IdentitySwapEngine,
        100_000,
        2,
    );
    assert!(
        r.is_ok(),
        "1_000 tail iterations under a 2-frame ceiling must succeed (no guard leak), got {r:?}"
    );
}

/// PR #1193 review (MEDIUM): the `result() == name` conjunct of `Cont::is_tail_passthrough` was
/// unreachable-false through the public `Node` API (the current `lower_to_anf` lowering's trailing
/// `Alias` invariant guarantees it) — so a mutation dropping the conjunct survived the whole suite.
/// This white-box pin tests the condition DIRECTLY, so the conjunct is locally witnessed rather
/// than silently dependent on another module's lowering shape: a completed block whose result is
/// NOT the bound name must NOT be treated as a passthrough (eliding it would return the wrong
/// value), and pending bindings must block transparency regardless of the name.
#[test]
fn is_tail_passthrough_requires_result_to_be_the_bound_name() {
    use mycelium_core::lower::{lower_to_anf, Atom};
    use std::rc::Rc;

    // M-999: the machine now runs the prepared (`Rc`-shared) mirror of the lowered ANF; the pinned
    // passthrough property is unchanged — only the block handle type moved (`Anf` -> `Code`).
    let block = Code::prepare(&lower_to_anf(&Node::Const(byte())));
    let done = block.bindings_len();
    let result_name = block.result().clone();

    // Completed block + result IS the bound name → a genuine passthrough (elide).
    assert!(
        Cont::probe(Rc::clone(&block), done, result_name).is_tail_passthrough(),
        "completed block whose result is the bound name must be tail-transparent"
    );

    // Completed block but the result is a DIFFERENT name → NOT a passthrough: resuming this
    // continuation returns the block's own result binding, not the incoming value, so eliding it
    // would substitute the wrong value. This is the conjunct the mutation attack found untested.
    assert!(
        !Cont::probe(
            Rc::clone(&block),
            done,
            Atom::Named("m996_review_probe".into())
        )
        .is_tail_passthrough(),
        "a completed block whose result is NOT the bound name must not be elided"
    );

    // Pending bindings → not a passthrough even with the matching name (real post-work remains).
    if done > 0 {
        assert!(
            !Cont::probe(Rc::clone(&block), 0, block.result().clone()).is_tail_passthrough(),
            "a block with pending bindings must not be elided"
        );
    }
}
