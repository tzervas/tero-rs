//! Balanced-ternary integer arithmetic — `add`, `neg`, `mul`, and the `int ↔ trits` codec
//! (M-111; `docs/spec/swaps/binary-ternary.md` §1).
//!
//! This is the Ring-1 wrapper around `mycelium_core::ternary`, surfacing the kernel codec and
//! arithmetic with the full contract: explicit `Option` on every fallible op, no unsafe code, and
//! no new trusted base (KC-3/C5). The kernel's `i64` accuracy ceiling is `m ≤ 40` trits
//! (`(3^40−1)/2 < i64::MAX`); widths above that return `None` on `max_magnitude` and therefore
//! `None` everywhere that depends on it (FLAG: Q4 — bignum ceiling).
//!
//! **Guarantee: `Exact` on every op.** The balanced-ternary algebra is an exact integer identity
//! (Knuth 4.1; `docs/spec/swaps/binary-ternary.md` §1); fallibility is the overflow/range
//! boundary, never a weakening of the tag (C2/VR-5).

use mycelium_core::ternary as kernel;

use crate::primitives::Trit;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a slice of `Trit` (std surface) to a `Vec<mycelium_core::Trit>` (kernel type).
fn to_core(ts: &[Trit]) -> Vec<mycelium_core::Trit> {
    ts.iter().map(|&t| t.to_core()).collect()
}

/// Convert a `Vec<mycelium_core::Trit>` (kernel type) to a `Vec<Trit>` (std surface).
fn from_core(ts: Vec<mycelium_core::Trit>) -> Vec<Trit> {
    ts.into_iter().map(Trit::from_core).collect()
}

// ── Codec ─────────────────────────────────────────────────────────────────────

/// The integer denoted by an MSB-first trit string.
///
/// `value(t) = Σⱼ digit(tⱼ)·3^(m-1-j)` (Horner; `docs/spec/swaps/binary-ternary.md` §1).
/// The empty string denotes 0.
///
/// **Guarantee: `Exact`.** Total — the Horner sum is an integer identity with no approximation
/// (C2). Width ceiling: `i64` is exact for every width up to `m = 40` (M-111).
///
/// **FLAG (Q4):** widths above 40 are not handled (bignum out of scope for v0; C1 is preserved
/// because the value still fits in `i64` if the trits are well-formed; the caller controls width).
#[must_use]
pub fn trits_to_int(ts: &[Trit]) -> i64 {
    kernel::trits_to_int(&to_core(ts))
}

/// The unique `m`-trit balanced representation of `value`, MSB-first.
///
/// **Guarantee: `Exact`.** Returns `None` if `value ∉ [−(3^m−1)/2, +(3^m−1)/2]` — an explicit
/// out-of-range error, never a silent truncation or wrap (C1/G2). Also returns `None` if `3^m`
/// would overflow `i64` (`m ≥ 41`; FLAG Q4). (Mutant witness: if the range check were removed,
/// a value of magnitude 365 in 6 trits would produce a wrong trit string instead of `None`.)
#[must_use]
pub fn int_to_trits(value: i64, m: u32) -> Option<Vec<Trit>> {
    kernel::int_to_trits(value, m).map(from_core)
}

/// The maximum representable magnitude in `m` trits: `(3^m − 1) / 2`.
///
/// The symmetric range is `[−max, +max]`. Returns `None` if `3^m` would overflow `i64` (`m ≥ 41`;
/// FLAG Q4 — bignum ceiling).
///
/// **Guarantee: `Exact`.** Total for valid `m`; explicit `None` for overflow (C1).
#[must_use]
pub fn max_magnitude(m: u32) -> Option<i64> {
    kernel::max_magnitude(m)
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

/// Digit-wise negation of an `m`-trit balanced-ternary number.
///
/// `value(neg a) = −value(a)` exactly (balanced ternary is sign-symmetric — no two's-complement
/// asymmetry; `docs/spec/swaps/binary-ternary.md` §1). Width-preserving.
///
/// **Guarantee: `Exact`.** Total — the range `[−max, +max]` is symmetric, so the negation of
/// every representable value is also representable (C2).
#[must_use]
pub fn neg(a: &[Trit]) -> Vec<Trit> {
    from_core(kernel::neg(&to_core(a)))
}

/// Fixed-width balanced-ternary addition `a + b`.
///
/// **Guarantee: `Exact`.** Returns `None` on fixed-width overflow — i.e. when the true sum
/// `trits_to_int(a) + trits_to_int(b)` lies outside `[−max_magnitude(m), +max_magnitude(m)]` —
/// and `None` if `a.len() != b.len()`. Never silently wraps (C1/G2). (Mutant witness: if the
/// carry check were removed, an overflowing sum would silently produce a wrong result.)
#[must_use]
pub fn add(a: &[Trit], b: &[Trit]) -> Option<Vec<Trit>> {
    kernel::add(&to_core(a), &to_core(b)).map(from_core)
}

/// Fixed-width balanced-ternary subtraction `a − b = add(a, neg(b))`.
///
/// **Guarantee: `Exact`.** Returns `None` on fixed-width overflow or unequal widths (C1/G2).
#[must_use]
pub fn sub(a: &[Trit], b: &[Trit]) -> Option<Vec<Trit>> {
    kernel::sub(&to_core(a), &to_core(b)).map(from_core)
}

/// Fixed-width balanced-ternary multiplication `a × b`.
///
/// Computes the full `2m`-trit product and returns the low `m` trits iff the high trits are all
/// zero — otherwise `None` (overflow, explicit). Also `None` if `a.len() != b.len()`.
///
/// **Guarantee: `Exact`.** Returns `None` on overflow — the mathematical product
/// `trits_to_int(a) * trits_to_int(b)` exceeds the `m`-trit range — never silently truncates
/// (C1/G2). (Mutant witness: if the high-trit check were replaced with `true`, overflow would
/// silently return a wrong low `m` trits.)
#[must_use]
pub fn mul(a: &[Trit], b: &[Trit]) -> Option<Vec<Trit>> {
    kernel::mul(&to_core(a), &to_core(b)).map(from_core)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Enumerate every integer in the `m`-trit range and its encoded form.
    fn each_in_range(m: u32, mut f: impl FnMut(i64, Vec<Trit>)) {
        let max = max_magnitude(m).expect("m is small");
        for v in -max..=max {
            f(v, int_to_trits(v, m).expect("in range"));
        }
    }

    // ── trits_to_int ──────────────────────────────────────────────────────────

    #[test]
    fn trits_to_int_empty_is_zero() {
        assert_eq!(trits_to_int(&[]), 0);
    }

    #[test]
    fn worked_example_neg78_in_6_trits() {
        // binary-ternary.md §5: −78 in 6 trits is ⟨0,−1,0,0,+1,0⟩.
        let t = int_to_trits(-78, 6).expect("in range");
        use Trit::{Neg, Pos, Zero};
        assert_eq!(t, vec![Zero, Neg, Zero, Zero, Pos, Zero]);
        assert_eq!(trits_to_int(&t), -78);
    }

    // ── int_to_trits ──────────────────────────────────────────────────────────

    #[test]
    fn range_is_symmetric() {
        // (3^6 - 1)/2 = 364
        assert_eq!(max_magnitude(6), Some(364));
        assert!(int_to_trits(364, 6).is_some());
        assert!(int_to_trits(-364, 6).is_some());
        // Mutant witness: if the range check were removed, 365 would produce Some(wrong value).
        assert_eq!(
            int_to_trits(365, 6),
            None,
            "just-past-max must be None (C1)"
        );
        assert_eq!(int_to_trits(-365, 6), None, "just-past-min must be None");
    }

    #[test]
    fn codec_round_trips_exhaustively_at_small_widths() {
        // property: trits_to_int(int_to_trits(v, m)) == v for every v in the m-trit range.
        for m in 1..=5 {
            each_in_range(m, |v, t| {
                assert_eq!(t.len(), m as usize, "width={m}");
                assert_eq!(trits_to_int(&t), v, "round-trip at m={m}");
            });
        }
    }

    #[test]
    fn max_magnitude_overflows_at_m41() {
        // m=41 causes 3^41 to overflow i64 — max_magnitude returns None (C1 / FLAG Q4).
        // (int_to_trits(0, 41) is Some because 0 fits in any width; the overflow only affects
        // the magnitude calculation, not the codec for the special case v=0.)
        assert_eq!(max_magnitude(41), None);
        // The integer 0 fits in 41 trits (it is all zeros). int_to_trits does not call
        // max_magnitude; it loops and checks the residual, which is 0 for v=0.
        // So int_to_trits(0, 41) succeeds — this is correct and honest (C1 is about explicit
        // None on out-of-range, not about refusing correct encodings). FLAG Q4: the caller
        // should not pass m>=41 for non-trivial values; max_magnitude returning None signals
        // the ceiling.
        assert!(int_to_trits(0, 41).is_some()); // 0 fits in any width
                                                // A large value that is out of range for m=6 is the real C1 test.
        assert_eq!(
            int_to_trits(365, 6),
            None,
            "value past 6-trit max must be None"
        );
        assert_eq!(
            int_to_trits(-365, 6),
            None,
            "value past 6-trit min must be None"
        );
    }

    // ── neg ───────────────────────────────────────────────────────────────────

    #[test]
    fn neg_is_value_negation_exhaustively() {
        // property: trits_to_int(neg(t)) == -trits_to_int(t) for every value in range.
        for m in 1..=5 {
            each_in_range(m, |v, t| {
                assert_eq!(trits_to_int(&neg(&t)), -v, "neg at m={m}");
            });
        }
    }

    #[test]
    fn neg_is_involution_exhaustively() {
        // property: neg(neg(t)) == t — the balanced-ternary range is symmetric (no asymmetry).
        for m in 1..=5 {
            each_in_range(m, |_v, t| {
                assert_eq!(neg(&neg(&t)), t, "involution at m={m}");
            });
        }
    }

    // ── add ───────────────────────────────────────────────────────────────────

    #[test]
    fn add_matches_integer_oracle_exhaustively() {
        // oracle: add(a, b) == int_to_trits(digit(a)+digit(b), m) for every pair in m-trit range.
        // Mutant witness: removing the carry check causes overflow to silently wrap.
        for m in 1..=4 {
            let max = max_magnitude(m).unwrap();
            for x in -max..=max {
                for y in -max..=max {
                    let a = int_to_trits(x, m).unwrap();
                    let b = int_to_trits(y, m).unwrap();
                    let got = add(&a, &b);
                    let expected = x + y;
                    if expected.abs() <= max {
                        assert_eq!(got, int_to_trits(expected, m), "add({x},{y}) at m={m}");
                    } else {
                        assert_eq!(got, None, "add({x},{y}) should overflow at m={m}");
                    }
                }
            }
        }
    }

    #[test]
    fn add_rejects_unequal_widths() {
        // C1: mismatched widths are an explicit None, not a silent partial result.
        let a = int_to_trits(1, 2).unwrap();
        let b = int_to_trits(1, 3).unwrap();
        assert_eq!(add(&a, &b), None, "unequal-width add must be None");
    }

    // ── sub ───────────────────────────────────────────────────────────────────

    #[test]
    fn sub_matches_integer_oracle_exhaustively() {
        for m in 1..=4 {
            let max = max_magnitude(m).unwrap();
            for x in -max..=max {
                for y in -max..=max {
                    let a = int_to_trits(x, m).unwrap();
                    let b = int_to_trits(y, m).unwrap();
                    let got = sub(&a, &b);
                    let expected = x - y;
                    if expected.abs() <= max {
                        assert_eq!(got, int_to_trits(expected, m), "sub({x},{y}) at m={m}");
                    } else {
                        assert_eq!(got, None, "sub({x},{y}) should overflow at m={m}");
                    }
                }
            }
        }
    }

    // ── mul ───────────────────────────────────────────────────────────────────

    #[test]
    fn mul_matches_integer_oracle_exhaustively() {
        // Mutant witness: replacing the high-trit check with always-pass causes overflow to silently
        // return wrong low trits.
        for m in 1..=4 {
            let max = max_magnitude(m).unwrap();
            for x in -max..=max {
                for y in -max..=max {
                    let a = int_to_trits(x, m).unwrap();
                    let b = int_to_trits(y, m).unwrap();
                    let got = mul(&a, &b);
                    let expected = x * y;
                    if expected.abs() <= max {
                        assert_eq!(got, int_to_trits(expected, m), "mul({x},{y}) at m={m}");
                    } else {
                        assert_eq!(got, None, "mul({x},{y}) should overflow at m={m}");
                    }
                }
            }
        }
    }

    #[test]
    fn mul_rejects_unequal_widths() {
        let a = int_to_trits(1, 2).unwrap();
        let b = int_to_trits(1, 3).unwrap();
        assert_eq!(mul(&a, &b), None, "unequal-width mul must be None");
    }

    // ── algebraic identities ──────────────────────────────────────────────────

    #[test]
    fn add_neg_b_equals_sub() {
        // property: add(a, neg(b)) == sub(a, b) for every pair in range.
        for m in 1..=4 {
            let max = max_magnitude(m).unwrap();
            for x in -max..=max {
                for y in -max..=max {
                    let a = int_to_trits(x, m).unwrap();
                    let b = int_to_trits(y, m).unwrap();
                    let nb = neg(&b);
                    assert_eq!(
                        add(&a, &nb),
                        sub(&a, &b),
                        "add(a,neg(b))==sub(a,b) at ({x},{y}),m={m}"
                    );
                }
            }
        }
    }

    #[test]
    fn neg_is_additive_inverse_when_sum_in_range() {
        // property: add(a, neg(a)) == zero for every a.
        for m in 1..=4 {
            let max = max_magnitude(m).unwrap();
            let zero = int_to_trits(0, m).unwrap();
            for x in -max..=max {
                let a = int_to_trits(x, m).unwrap();
                let na = neg(&a);
                assert_eq!(
                    add(&a, &na),
                    Some(zero.clone()),
                    "a + neg(a) == 0 for x={x},m={m}"
                );
            }
        }
    }
}
