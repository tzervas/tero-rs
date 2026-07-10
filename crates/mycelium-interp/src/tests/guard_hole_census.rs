//! RFC-0041 §4.7/§5 — the guard-hole **census** (W0 safety net; RR-29 guard-hole inventory turned
//! into tracked failing tests, one per hole this crate owns).
//!
//! Real repros: each test constructs a genuinely deep `Node` and calls the hole's entry point.
//! Rust's default stack-overflow handler aborts the process directly (never through panic/unwind),
//! so none of this is `catch_unwind`-able — every test here stays `#[ignore = "Wn"]`d; running one
//! for real would crash the whole test binary. When the named wave lands, drop the `#[ignore]` and
//! the call must refuse cleanly instead. White-box access via `use crate::…` (CLAUDE.md test layout).

use crate::parallel::{is_pure, plan_parallel};
use crate::{EvalError, Interpreter};
use mycelium_core::{ContentHash, CtorRef, Meta, Node, Payload, Provenance, Repr, Value};

/// The canonical depth floor (RFC-0041 §4.2) the interp's shared budget defaults to — the ceiling a
/// deep value is refused at. `mycelium_workstack::RecursionBudget::DEFAULT_DEPTH_LIMIT` (4096).
const DEPTH_FLOOR: usize = mycelium_workstack::RecursionBudget::DEFAULT_DEPTH_LIMIT as usize;

fn byte() -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(vec![false; 8]),
        Meta::exact(Provenance::Root),
    )
    .expect("a well-formed Binary{8} const")
}

fn ctor() -> CtorRef {
    CtorRef::new(
        ContentHash::parse("blake3:round_trip_safe").expect("a well-formed content hash"),
        0,
    )
}

/// A right-nested `Node::Construct` chain, `n` deep, every leaf already a `Const` — i.e. already a
/// normal form, so evaluating it exercises pure TRAVERSAL recursion (no reduction), not fuel.
fn deep_construct(n: usize) -> Node {
    let mut acc = Node::Const(byte());
    for _ in 0..n {
        acc = Node::Construct {
            ctor: ctor(),
            args: vec![acc],
        };
    }
    acc
}

/// A single outer `Let` whose `bound` is already a value and whose `body` is a deep `Construct`
/// chain referencing the bound variable at its innermost leaf — `step` reduces `bound` in O(1) then
/// calls `subst(body, id, bound)`, which walks the whole `body` `n`-deep in one recursive call.
fn deep_let_body(n: usize) -> Node {
    let mut body = Node::Var("x".to_owned());
    for _ in 0..n {
        body = Node::Construct {
            ctor: ctor(),
            args: vec![body],
        };
    }
    Node::Let {
        id: "x".to_owned(),
        bound: Box::new(Node::Const(byte())),
        body: Box::new(body),
    }
}

#[test]
fn eval_core_deep_construct_refuses_cleanly() {
    // RFC-0041 W4 (RR-29 §0.1, LANDED): holes were `Interpreter::step`'s `Construct` arm (recurses into
    // non-value args) and the private `node_to_core_value` — both walk the deep chain via `eval_core`.
    // Now the shared `RecursionBudget` refuses past the floor with a constructed `DepthLimit`, and the
    // pass runs on the growable deep stack, so this is a clean refusal, never a `SIGABRT`.
    let deep = deep_construct(200_000);
    let err = Interpreter::default()
        .eval_core(&deep)
        .expect_err("a 200k-deep value must refuse, not succeed or SIGABRT");
    assert_eq!(
        err,
        EvalError::DepthLimit { limit: DEPTH_FLOOR },
        "the deep-construct value walk must refuse with the canonical DepthLimit at the floor"
    );
}

#[test]
fn eval_core_deep_subst_via_let_refuses_cleanly() {
    // RFC-0041 W4 (LANDED): hole was the private `subst`, invoked from `step`'s `Let`/E-Let-Bind case,
    // walking `body` (here `n` deep) to substitute the bound variable. Now budget-charged → `DepthLimit`.
    let deep = deep_let_body(200_000);
    let err = Interpreter::default()
        .eval_core(&deep)
        .expect_err("a 200k-deep Let-body subst must refuse, not SIGABRT");
    assert_eq!(
        err,
        EvalError::DepthLimit { limit: DEPTH_FLOOR },
        "the deep subst must refuse with the canonical DepthLimit at the floor"
    );
}

/// Hole (RFC-0041 W4, LANDED): `parallel::is_pure`. `is_pure` returns a plain `bool` — it is a *pure
/// analysis* of an already-materialized `Node`, not an evaluation of adversarial fuel, so a depth
/// *refusal* does not fit its signature. W4 closes the SIGABRT the honest way for this case: `is_pure`
/// is now an **explicit work-stack** traversal (O(1) host stack for any depth), so the call completes
/// on a deep `Node` instead of overflowing. The assertion is that it returns (no crash) — a deep
/// all-`Const` `Construct` spine is pure.
#[test]
fn is_pure_deep_recursion() {
    let deep = deep_construct(200_000);
    assert!(is_pure(&deep), "a deep Const/Construct spine is pure");
}

/// Hole (RFC-0041 W4, LANDED): `parallel::plan_parallel` — its only deep recursion was via `is_pure`
/// (now iterative), so it too completes on a deep `Node` rather than a SIGABRT. A single-arg deep
/// `Construct` is below `MIN_BATCH_WIDTH`, so the plan is `SequentialNoBatch`.
#[test]
fn plan_parallel_deep_recursion() {
    let deep = deep_construct(200_000);
    assert_eq!(plan_parallel(&deep), crate::ParallelPlan::SequentialNoBatch);
}
