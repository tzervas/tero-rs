//! `ensure_sufficient_stack` (W2: the deep worker base + the runtime-gated grow backstop, RFC-0041
//! §4.3): it returns `f`'s value, and a genuinely deep recursion that would overflow a default thread
//! stack runs without a crash — so the *budget*, not a host overflow, is what bounds a pathological
//! input. The W2 body swap preserves this W1 guarantee (the worker base is retained); these tests are
//! the regression guard that the swap did not weaken it.

use crate::{ensure_sufficient_stack, RecursionBudget};

#[test]
fn returns_the_closures_value() {
    let budget = RecursionBudget::default();
    assert_eq!(ensure_sufficient_stack(&budget, || 2 + 2), 4);
}

#[test]
fn a_moved_in_budget_charges_on_the_worker_thread() {
    // The budget is `Send`, so a consumer can move it into the worker and charge there. This exercises
    // the intended W1 usage: create/own the budget inside `f`, on the deep stack.
    let outer = RecursionBudget::default();
    let depth_seen = ensure_sufficient_stack(&outer, || {
        let budget = RecursionBudget::new(8, u64::MAX, u64::MAX);
        let _g1 = budget.try_enter().expect("enter 1");
        let _g2 = budget.try_enter().expect("enter 2");
        budget.current_depth()
    });
    assert_eq!(depth_seen, 2);
}

#[test]
fn a_deep_recursion_does_not_overflow_the_host_stack() {
    // Far past a default 2 MiB thread stack at a non-trivial frame size — the deep worker absorbs it.
    fn descend(n: u64, pad: &[u8; 256]) -> u64 {
        if n == 0 {
            u64::from(pad[0])
        } else {
            descend(n - 1, pad).wrapping_add(1)
        }
    }
    let budget = RecursionBudget::default();
    let got = ensure_sufficient_stack(&budget, || descend(150_000, &[3u8; 256]));
    assert_eq!(got, 150_003);
}
