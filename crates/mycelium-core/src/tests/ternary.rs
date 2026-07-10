//! White-box tests for [`crate::ternary`] beyond the ~40-trit figure the *conversion* utilities
//! (`max_magnitude`/`trits_to_int`/`int_to_trits`) declare — CU-7 recon (mitigation #14: verify
//! against the codebase before implementing).
//!
//! **Finding.** The trx2 kickoff notes describe the runnable fixed-width `trit.add`/`trit.sub`/
//! `trit.mul`/`trit.neg` prims as capped at "~40 trits", attributed to `mycelium_core::ternary`
//! being "`i64`-internal". Reading [`crate::ternary::add`]/[`crate::ternary::mul`] shows this is
//! **not accurate for the arithmetic itself**: both are digit-serial (ripple-carry add,
//! shifted-accumulation multiply) over `&[Trit]`, with no `i64` anywhere in the algorithm —
//! overflow is detected *structurally* (a nonzero final carry / nonzero high digits), never via an
//! integer-range check. The **only** `i64`-capped pieces are the *conversion* utilities
//! (`max_magnitude`'s `3^m` must fit `i64` ⇒ `m ≤ 40`; `int_to_trits`/`trits_to_int` round-trip a
//! **value**, not a width, through `i64`) — used for decimal-literal encoding and this file's own
//! test oracle, a genuinely different concern from RFC-0033 §4.2.2's "arithmetic MUST be
//! arbitrary-width" mandate.
//!
//! These tests pin `add`/`mul`/`neg`'s correctness at `WIDE = 60` trits (well past 40) against the
//! same `int_to_trits`/`trits_to_int` oracle the ≤40-trit tests already use, restricted to small
//! magnitudes (the oracle itself is `i64`-bounded — `3^60` vastly exceeds `i64::MAX`, so this file
//! does not re-litigate `max_magnitude`'s own, already-correct, `i64` ceiling). A final structural
//! test operates at 200 trits — far past any `i64` oracle's reach — to confirm the shape holds
//! with no oracle at all, just direct low/high-digit inspection.
//!
//! The corresponding end-to-end (surface `.myc` → L1-eval ≡ L0-interp ≡ AOT) witness at 80 trits
//! lives in `mycelium-l1/tests/enablement.rs` (`trit_add_beyond_the_claimed_40_trit_cap_three_way`
//! / `trit_mul_beyond_the_claimed_40_trit_cap_three_way`).

use crate::ternary::*;
use crate::value::Trit;

/// Well past the "~40-trit" figure the recon corrected; still an arbitrary, non-special width.
const WIDE: u32 = 60;

#[test]
fn add_matches_the_integer_oracle_at_wide_width() {
    for x in [-1000i64, -500, -1, 0, 1, 500, 1000] {
        for y in [-1000i64, -500, -1, 0, 1, 500, 1000] {
            let a = int_to_trits(x, WIDE).expect("small value fits WIDE trits");
            let b = int_to_trits(y, WIDE).expect("small value fits WIDE trits");
            let got = add(&a, &b).expect("small sums stay well within WIDE trits' range");
            assert_eq!(
                trits_to_int(&got),
                x + y,
                "add({x}, {y}) at width {WIDE} must match the integer oracle"
            );
        }
    }
}

#[test]
fn mul_matches_the_integer_oracle_at_wide_width() {
    for x in -50i64..=50 {
        for y in [-50i64, -10, -1, 0, 1, 10, 50] {
            let a = int_to_trits(x, WIDE).expect("small value fits WIDE trits");
            let b = int_to_trits(y, WIDE).expect("small value fits WIDE trits");
            let got = mul(&a, &b).expect("small products stay well within WIDE trits' range");
            assert_eq!(
                trits_to_int(&got),
                x * y,
                "mul({x}, {y}) at width {WIDE} must match the integer oracle"
            );
        }
    }
}

#[test]
fn neg_matches_the_integer_oracle_at_wide_width() {
    for x in [-1000i64, -500, -1, 0, 1, 500, 1000] {
        let a = int_to_trits(x, WIDE).expect("small value fits WIDE trits");
        let got = neg(&a);
        assert_eq!(
            trits_to_int(&got),
            -x,
            "neg({x}) at width {WIDE} must match the oracle"
        );
    }
}

/// `add` at 200 trits — far past any `i64` oracle's reach (`3^200 ≫ i64::MAX`), so this checks
/// the algorithm's *shape* directly (low digits carry the value, high digits stay zero) rather
/// than via `trits_to_int` on the full width. Confirms nothing in `add` depends on a width ceiling
/// tied to `i64` (there is none in the algorithm — see the module note).
#[test]
fn add_operates_structurally_at_200_trits_far_past_any_i64_oracle() {
    let n = 200usize;
    let a = vec![Trit::Zero; n]; // 0
    let mut b = vec![Trit::Zero; n];
    *b.last_mut().expect("n > 0") = Trit::Pos; // 1
    let sum = add(&a, &b).expect("0 + 1 must be in range at any width");
    assert_eq!(sum.len(), n, "add must preserve width");
    assert_eq!(
        trits_to_int(&sum[(n - 10)..]),
        1,
        "the low 10 digits, read on their own, must equal 1"
    );
    assert!(
        sum[..(n - 10)].iter().all(|&t| t == Trit::Zero),
        "every digit above the low 10 must stay Zero"
    );
}
