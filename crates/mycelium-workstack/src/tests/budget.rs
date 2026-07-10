//! The per-invocation budget: refuses exactly at each limit (never-silently), the RAII guard decrements
//! on drop, nested enters compose, and `DepthExceeded` carries the right `limit`. Mutant-witness cases
//! pin the exact `>`/`>=` boundary and the presence of each charge guard.

use crate::{BudgetError, BudgetKind, RecursionBudget};

#[test]
fn default_depth_limit_is_4096() {
    assert_eq!(RecursionBudget::DEFAULT_DEPTH_LIMIT, 4096);
    assert_eq!(RecursionBudget::default().depth_limit(), 4096);
}

// ── Depth: refuse exactly at the limit; carry the right `limit`. (Mutant witness: `>` vs `>=`.) ──────

/// For a depth ceiling `N`, exactly `N` enters succeed and the `N+1`-th refuses with
/// `DepthExceeded { limit: N }`. A `>`→`>=` mutant refuses one enter early; a removed guard never
/// refuses — both fail this table.
#[test]
fn depth_refuses_exactly_at_the_limit() {
    for limit in [1u32, 2, 5, 64] {
        let budget = RecursionBudget::new(limit, u64::MAX, u64::MAX);
        // Hold the guards so the depth actually accumulates (nested enters compose).
        let mut guards = Vec::new();
        for expected_depth in 1..=limit {
            let guard = budget.try_enter().unwrap_or_else(|e| {
                panic!("enter {expected_depth}/{limit} must succeed, got {e:?}")
            });
            assert_eq!(budget.current_depth(), expected_depth);
            guards.push(guard);
        }
        // The (limit+1)-th enter refuses never-silently with the exact limit.
        match budget.try_enter() {
            Err(BudgetError::DepthExceeded { limit: hit }) => assert_eq!(hit, limit),
            other => panic!(
                "enter past the limit must be DepthExceeded {{ limit: {limit} }}, got {other:?}"
            ),
        }
        // Depth is unchanged by the refused enter.
        assert_eq!(budget.current_depth(), limit);
        drop(guards);
    }
}

// ── The RAII guard decrements on drop (a scope test). ────────────────────────────────────────────────

#[test]
fn depth_guard_decrements_on_drop() {
    let budget = RecursionBudget::new(8, u64::MAX, u64::MAX);
    assert_eq!(budget.current_depth(), 0);
    {
        let _g = budget.try_enter().expect("enter");
        assert_eq!(budget.current_depth(), 1);
    } // guard drops here
    assert_eq!(
        budget.current_depth(),
        0,
        "the guard must release its frame on drop"
    );
}

/// Nested guards compose and unwind in LIFO order — the property a `&mut`-borrowing guard could not
/// provide (the outer borrow would lock the budget).
#[test]
fn nested_enters_compose_and_unwind() {
    let budget = RecursionBudget::new(8, u64::MAX, u64::MAX);
    let g1 = budget.try_enter().expect("enter 1");
    let g2 = budget.try_enter().expect("enter 2");
    let g3 = budget.try_enter().expect("enter 3");
    assert_eq!(budget.current_depth(), 3);
    // Charging runs *alongside* live guards (only possible because try_enter takes &self).
    budget
        .charge_bytes(0)
        .expect("charge alongside live guards");
    drop(g3);
    assert_eq!(budget.current_depth(), 2);
    drop(g2);
    assert_eq!(budget.current_depth(), 1);
    drop(g1);
    assert_eq!(budget.current_depth(), 0);
}

// ── Bytes / work-steps: refuse at the cumulative limit; counter unchanged on refusal. ────────────────

/// One data-driven table exercising both non-depth resources through a charge function. `at_limit`
/// charges succeed; the charge that would exceed refuses with the right `kind`, `limit`, and cumulative
/// `requested`, and leaves the counter unchanged. A removed charge-guard mutant lets the over-charge
/// succeed — this fails it.
#[test]
fn charge_refuses_at_the_cumulative_limit() {
    struct Case {
        kind: BudgetKind,
        limit: u64,
    }
    let cases = [
        Case {
            kind: BudgetKind::Bytes,
            limit: 100,
        },
        Case {
            kind: BudgetKind::WorkSteps,
            limit: 7,
        },
    ];
    for Case { kind, limit } in cases {
        let budget = RecursionBudget::new(64, limit, limit);
        let charge = |n: u64| match kind {
            BudgetKind::Bytes => budget.charge_bytes(n),
            BudgetKind::WorkSteps => budget.charge_steps(n),
        };
        let current = |b: &RecursionBudget| match kind {
            BudgetKind::Bytes => b.current_bytes(),
            BudgetKind::WorkSteps => b.current_steps(),
        };

        // Charge right up to the ceiling in two steps.
        charge(limit - 1).expect("charge below ceiling");
        charge(1).expect("charge exactly to the ceiling");
        assert_eq!(current(&budget), limit);

        // A charge of 1 more overruns — never-silently, with the cumulative demand.
        match charge(1) {
            Err(BudgetError::OutOfBudget {
                kind: k,
                limit: l,
                requested,
            }) => {
                assert_eq!(k, kind);
                assert_eq!(l, limit);
                assert_eq!(
                    requested,
                    limit + 1,
                    "requested is the cumulative would-be total"
                );
            }
            other => panic!("over-charge must be OutOfBudget, got {other:?}"),
        }
        // The refused charge did not apply.
        assert_eq!(
            current(&budget),
            limit,
            "a refused charge must not mutate the counter"
        );
    }
}

#[test]
fn charge_at_exactly_the_limit_is_accepted() {
    // Mutant witness: `next > limit` vs `next >= limit`. A `>=` mutant would refuse the exact-limit
    // charge below.
    let budget = RecursionBudget::new(64, 10, 10);
    budget
        .charge_bytes(10)
        .expect("charging exactly to the ceiling is allowed");
    budget
        .charge_steps(10)
        .expect("charging exactly to the ceiling is allowed");
    assert_eq!(budget.current_bytes(), 10);
    assert_eq!(budget.current_steps(), 10);
}

#[test]
fn display_is_actionable_and_never_empty() {
    let d = BudgetError::DepthExceeded { limit: 4096 }.to_string();
    assert!(d.contains("4096"), "depth error must name the limit: {d}");
    let o = BudgetError::OutOfBudget {
        kind: BudgetKind::Bytes,
        limit: 10,
        requested: 11,
    }
    .to_string();
    assert!(
        o.contains("10") && o.contains("11"),
        "out-of-budget must name both numbers: {o}"
    );
}
