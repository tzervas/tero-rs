//! **In-isolation validation of the shared core (§4.1 common-mode risk).** Because the three real
//! machines will share this budget/guard code, the differential can no longer cross-validate it — so we
//! exercise it here against a **synthetic frame type**, independent of any real machine, plus
//! **mutant-witness** cases: a distinguishing assertion that fails if a `>`/`>=` guard is flipped or a
//! charge is removed (the crate is in `cargo-mutants` scope, so a remove-guard mutant must not survive).

use crate::{BudgetError, BudgetKind, RecursionBudget};

/// A stand-in for a machine's per-frame state — deliberately unrelated to any real `Frame`, so the test
/// validates the budget/guard core, not a specific machine.
struct SynthFrame {
    heap_bytes: u64,
}

/// A synthetic recursive descent that charges the budget exactly as a real machine's frame-push site
/// would: enter one depth frame (held across the recursive call — validating nested composition under
/// *real* host recursion), then charge the frame's heap bytes, then recurse. Returns the total bytes on
/// success, or the never-silent `BudgetError` on the first over-budget frame.
fn descend(budget: &RecursionBudget, frames: &[SynthFrame]) -> Result<u64, BudgetError> {
    match frames.split_first() {
        None => Ok(0),
        Some((frame, rest)) => {
            let _guard = budget.try_enter()?; // depth++ (released when this frame unwinds)
            budget.charge_bytes(frame.heap_bytes)?;
            let deeper = descend(budget, rest)?;
            Ok(deeper.saturating_add(frame.heap_bytes))
        }
    }
}

fn frames(n: usize, bytes_each: u64) -> Vec<SynthFrame> {
    (0..n)
        .map(|_| SynthFrame {
            heap_bytes: bytes_each,
        })
        .collect()
}

#[test]
fn a_within_budget_descent_succeeds_and_releases_all_depth() {
    let budget = RecursionBudget::new(16, 10_000, u64::MAX);
    let total = descend(&budget, &frames(10, 100)).expect("10 frames of 100 bytes fit");
    assert_eq!(total, 1000);
    // Every guard unwound: the live depth is back to zero after the recursion returns.
    assert_eq!(
        budget.current_depth(),
        0,
        "all depth frames released on unwind"
    );
}

/// Mutant witness (depth): with a depth ceiling of exactly `N`, a descent of `N` frames succeeds but
/// `N+1` frames refuses at `DepthExceeded { limit: N }`. Flipping `>`→`>=` (refuse a frame early) or
/// removing the depth guard (never refuse) both fail this pair.
#[test]
fn depth_ceiling_refuses_the_first_over_deep_frame() {
    const N: u32 = 5;
    let make = |mem| RecursionBudget::new(N, mem, u64::MAX);

    // Exactly N frames fit.
    assert!(descend(&make(u64::MAX), &frames(N as usize, 1)).is_ok());

    // N+1 frames refuse — never-silently, with the exact limit.
    match descend(&make(u64::MAX), &frames(N as usize + 1, 1)) {
        Err(BudgetError::DepthExceeded { limit }) => assert_eq!(limit, N),
        other => panic!("an over-deep descent must refuse with DepthExceeded, got {other:?}"),
    }
}

/// Mutant witness (bytes): the descent refuses on the frame whose cumulative byte charge first exceeds
/// the memory ceiling. Removing the byte-charge guard would let it run to completion — this fails that.
#[test]
fn memory_ceiling_refuses_when_cumulative_bytes_overrun() {
    // Ceiling 250 with 100-byte frames: frame 1 → 100, frame 2 → 200, frame 3 → 300 > 250 refuses.
    let budget = RecursionBudget::new(64, 250, u64::MAX);
    match descend(&budget, &frames(5, 100)) {
        Err(BudgetError::OutOfBudget {
            kind: BudgetKind::Bytes,
            limit,
            requested,
        }) => {
            assert_eq!(limit, 250);
            assert_eq!(
                requested, 300,
                "the third frame's cumulative charge is the over-budget total"
            );
        }
        other => panic!("a memory overrun must refuse with OutOfBudget(Bytes), got {other:?}"),
    }
    // Never-silent: the accounting reflects exactly the two frames that were charged before the refusal
    // (the third frame's refused charge did not apply), and all depth unwound.
    assert_eq!(budget.current_bytes(), 200);
    assert_eq!(budget.current_depth(), 0);
}

/// The isolation property in the positive direction: for any admissible frame list (count within the
/// depth ceiling, total within the memory ceiling), the descent accepts and reports the exact byte sum —
/// so the guard core neither over- nor under-charges a well-formed input.
#[test]
fn accepts_every_admissible_frame_list() {
    struct Case {
        count: usize,
        bytes_each: u64,
    }
    let cases = [
        Case {
            count: 0,
            bytes_each: 0,
        },
        Case {
            count: 1,
            bytes_each: 4096,
        },
        Case {
            count: 32,
            bytes_each: 8,
        },
        Case {
            count: 100,
            bytes_each: 1,
        },
    ];
    for Case { count, bytes_each } in cases {
        // Ceilings chosen to comfortably admit the case.
        let budget =
            RecursionBudget::new(count as u32 + 1, (count as u64) * bytes_each + 1, u64::MAX);
        let total = descend(&budget, &frames(count, bytes_each)).expect("admissible list accepts");
        assert_eq!(total, count as u64 * bytes_each);
        assert_eq!(budget.current_depth(), 0);
    }
}
