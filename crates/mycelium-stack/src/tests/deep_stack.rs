//! The coarse [`with_deep_stack`] worker: returns `f`'s value, lets `f` borrow from the caller,
//! absorbs a genuinely deep recursion without overflowing, and propagates a panic unchanged.
//! (Extracted from the former inline `#[cfg(test)] mod tests` — CLAUDE.md test layout.)

use crate::with_deep_stack;

#[test]
fn runs_the_closure_and_returns_its_value() {
    assert_eq!(with_deep_stack(|| 2 + 2), 4);
}

#[test]
fn the_closure_may_borrow_from_the_caller() {
    let xs = [1u64, 2, 3];
    let sum: u64 = with_deep_stack(|| xs.iter().sum());
    assert_eq!(sum, 6);
}

#[test]
fn a_genuinely_deep_recursion_does_not_overflow() {
    // Far past any default thread stack (2 MiB) at a non-trivial frame size — the lazily-committed
    // worker stack absorbs it without a crash. This is the regression guard for the "deep input must
    // never overflow the caller's stack" contract.
    fn descend(n: u64, pad: &[u8; 512]) -> u64 {
        if n == 0 {
            u64::from(pad[0])
        } else {
            descend(n - 1, pad).wrapping_add(1)
        }
    }
    let got = with_deep_stack(|| descend(200_000, &[7u8; 512]));
    assert_eq!(got, 200_007);
}

#[test]
#[should_panic(expected = "intentional")]
fn a_panic_in_the_closure_propagates_to_the_caller() {
    with_deep_stack(|| panic!("intentional"));
}
