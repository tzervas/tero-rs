//! RFC-0041 W7 — `Interpreter::with_depth` **uniform-budget parity** (a cheap checked-basis mechanism
//! check).
//!
//! The over-budget refusal is verified at the 4096 *floor* elsewhere (`guard_hole_census`), but the
//! budget→[`EvalError::DepthLimit`] mapping is a *family* that must hold at an **arbitrary** ceiling,
//! not just the floor. These tests set a small ceiling via [`Interpreter::with_depth`] and confirm a
//! controlled-depth value refuses with `DepthLimit` at *exactly* that ceiling — so the mapping is
//! uniform across the range, never-silent (G2) at every budget, not merely floor-checked.
//!
//! White-box access via `use crate::…` (CLAUDE.md test layout). Guarantee: `Empirical` (a mechanism
//! check over concrete small budgets), not `Proven` (no theorem over all `u32` — VR-5).

use crate::{EvalError, Interpreter};
use mycelium_core::{ContentHash, CtorRef, Meta, Node, Payload, Provenance, Repr, Value};

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

/// A right-nested `Node::Construct` chain, `n` deep, every leaf already a `Const` — a normal form, so
/// stepping it exercises pure structural-traversal recursion (the depth budget), not fuel.
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

/// The mapping is uniform: at a small custom ceiling, an over-ceiling value refuses with `DepthLimit`
/// reporting **that** ceiling — not the 4096 floor. Exercised at several small budgets so a single
/// off-by-one or a hard-coded floor would fail.
#[test]
fn with_depth_refuses_at_the_custom_ceiling_not_the_floor() {
    for limit in [1_u32, 2, 8, 100] {
        // The value is comfortably deeper than the ceiling, so the enter past `limit` must refuse.
        let deep = deep_construct(limit as usize + 500);
        let err = Interpreter::default()
            .with_depth(limit)
            .eval_core(&deep)
            .expect_err(
                "a value deeper than the custom ceiling must refuse, not succeed or SIGABRT",
            );
        assert_eq!(
            err,
            EvalError::DepthLimit {
                limit: limit as usize
            },
            "the custom-budget walk must refuse with DepthLimit at exactly the set ceiling ({limit}), \
             proving the budget->error mapping is uniform, not floor-only",
        );
    }
}

/// The complement: a value *within* the custom ceiling is **not** refused for depth (it evaluates to a
/// data value here — an explicit `DataResult` via `eval`, never a `DepthLimit`), so `with_depth` only
/// ever tightens/loosens the depth gate and does not spuriously refuse shallow input.
#[test]
fn with_depth_does_not_refuse_input_within_the_ceiling() {
    // Depth 4 value under a ceiling of 64: well within budget. `eval_core` reads it off as a data
    // value (a saturated `Construct` of a const) — the point is only that it is NOT a DepthLimit.
    let shallow = deep_construct(4);
    let result = Interpreter::default().with_depth(64).eval_core(&shallow);
    assert!(
        !matches!(result, Err(EvalError::DepthLimit { .. })),
        "a value within the custom ceiling must not be refused for depth; got {result:?}",
    );
}

/// A default interpreter (no `with_depth`) still refuses at the 4096 floor — the additive knob leaves
/// the established default behavior exactly as `guard_hole_census` verifies it.
#[test]
fn default_interpreter_still_refuses_at_the_floor() {
    const FLOOR: usize = mycelium_workstack::RecursionBudget::DEFAULT_DEPTH_LIMIT as usize;
    let deep = deep_construct(FLOOR + 500);
    let err = Interpreter::default()
        .eval_core(&deep)
        .expect_err("a value past the floor must refuse under the default budget");
    assert_eq!(err, EvalError::DepthLimit { limit: FLOOR });
}
