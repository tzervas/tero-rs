//! Exact numeric operations and reified rounding (spec §3 / §4).
//!
//! All functions in this module are `Exact` — they return the mathematically correct result or an
//! explicit `Err`; no approximation, no NaN, no sentinel (C1/C2/G2).
//!
//! # Rounding
//!
//! [`round`] is `Exact` in the sense that the rounding is the *exact image* under the named,
//! reified [`RoundMode`]. The mode IS the specification: it is the inspectable artifact that makes
//! the operation EXPLAIN-able (C3). No ambient rounding mode; no hidden default.
//!
//! # Integer ops
//!
//! `abs`, `neg`, `signum`, `min`/`max`, `gcd`/`lcm`, `checked_div`/`checked_rem` all operate over
//! `i64` for simplicity in this Rust-first implementation. The spec uses `Int` (Mycelium's abstract
//! integer); when the full value model lands, these will be lifted to `Value`-typed surfaces. The
//! tie rule for `min`/`max` is documented: `min` returns the first argument on a tie, `max`
//! returns the first argument on a tie — stable and documented, never silent (C1).

use crate::MathErr;

// ---- Rounding ---------------------------------------------------------------

/// The reified rounding mode for [`round`] (spec §3; C3 / SC-3 / G11).
///
/// The mode is a **required** argument — never a hidden ambient default. Each variant is an
/// inspectable, named specification of the rounding operation. The EXPLAIN artifact for a
/// `round` call is the `RoundMode` value itself plus the resulting integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMode {
    /// Round toward negative infinity (floor): `⌊x⌋`.
    Floor,
    /// Round toward positive infinity (ceiling): `⌈x⌉`.
    Ceil,
    /// Round toward zero (truncation): `trunc(x)`.
    TruncTowardZero,
    /// Round half away from zero: the common "round half up" for positive numbers.
    HalfAwayFromZero,
    /// Round half to even (banker's rounding): the IEEE 754 default.
    HalfToEven,
}

/// Convert an integral-valued finite `f64` to `i64`, refusing values outside the
/// exactly-representable `i64` range (C1 — never a silent saturation/truncation).
///
/// Rust's `f64 as i64` *saturates* (since 1.45): `1e20_f64 as i64 == i64::MAX`. A finite
/// `f64` therefore passes `is_finite()` yet maps to a wrong, silent value. We reject it.
/// `i64::MIN` is exactly representable as `f64` (`-2^63`); `i64::MAX` is **not** (it rounds
/// to `2^63`), so the honest upper bound is "strictly below `2^63`".
fn checked_round_to_i64(r: f64) -> Result<i64, MathErr> {
    const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0; // -2^63, exact
    const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0; // 2^63 == (i64::MAX as f64)
    if !(I64_MIN_F64..TWO_POW_63).contains(&r) {
        return Err(MathErr::Overflow);
    }
    Ok(r as i64)
}

/// `floor(x)` — round toward negative infinity (exact under the `Floor` mode).
///
/// **Guarantee: `Exact`** — the floor of an in-range finite `f64` is representable and exact.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
/// - [`MathErr::Overflow`] when `floor(x)` lies outside the `i64` range (never a silent clamp).
pub fn floor(x: f64) -> Result<i64, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    checked_round_to_i64(x.floor())
}

/// `ceil(x)` — round toward positive infinity (exact under the `Ceil` mode).
///
/// **Guarantee: `Exact`** — the ceiling of a finite `f64` is always representable and exact.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
/// - [`MathErr::Overflow`] when `ceil(x)` lies outside the `i64` range (never a silent clamp).
pub fn ceil(x: f64) -> Result<i64, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    checked_round_to_i64(x.ceil())
}

/// `trunc(x)` — round toward zero (exact under the `TruncTowardZero` mode).
///
/// **Guarantee: `Exact`** — the integer truncation of a finite `f64` is exact.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
/// - [`MathErr::Overflow`] when `trunc(x)` lies outside the `i64` range (never a silent clamp).
pub fn trunc(x: f64) -> Result<i64, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    checked_round_to_i64(x.trunc())
}

/// `round(x, mode)` — round `x` to the nearest integer under the named, reified [`RoundMode`].
///
/// **Guarantee: `Exact`** — the rounded value is the exact image of `x` under `mode`. The mode is
/// the EXPLAIN artifact (C3): it is required, named, and inspectable — never a hidden default.
///
/// Returns the integer result (as `i64`) and the `RoundMode` applied, so callers can EXPLAIN the
/// rounding decision.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
/// - [`MathErr::Overflow`] when the rounded value lies outside the `i64` range (never a clamp).
pub fn round(x: f64, mode: RoundMode) -> Result<(i64, RoundMode), MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    let rounded = match mode {
        RoundMode::Floor => x.floor(),
        RoundMode::Ceil => x.ceil(),
        RoundMode::TruncTowardZero => x.trunc(),
        RoundMode::HalfAwayFromZero => {
            // "round half away from zero": x.round() in Rust does this.
            x.round()
        }
        RoundMode::HalfToEven => {
            // Banker's rounding: if the fractional part is exactly 0.5, round to the nearest even.
            // `frac` and `0.5` are both exact in `f64`, so the half test is exact equality — an
            // epsilon tolerance would misclassify values adjacent to 0.5 (VR-5 honesty).
            let frac = x - x.trunc();
            let is_half = frac.abs() == 0.5;
            if is_half {
                // Round to even: check if the truncated integer is odd.
                let t = x.trunc() as i64;
                if t % 2 == 0 {
                    x.trunc()
                } else if x > 0.0 {
                    x.trunc() + 1.0
                } else {
                    x.trunc() - 1.0
                }
            } else {
                x.round()
            }
        }
    };
    Ok((checked_round_to_i64(rounded)?, mode))
}

// ---- Exact integer operations -----------------------------------------------

/// `abs(x)` — absolute value of a signed integer.
///
/// **Guarantee: `Exact`, total.**
/// Note: `i64::MIN.abs()` would overflow (i64::MIN = -2^63, but i64::MAX = 2^63-1). This returns
/// `i64::MIN` unchanged (two's complement wrapping), which is the Rust `i64::abs` behavior in
/// debug mode panics. To stay never-silent (C1/G2), this function detects overflow and returns
/// `Err(Overflow)`.
///
/// # Errors
///
/// - [`MathErr::Overflow`] when `x == i64::MIN` (the only overflow case for `abs` on `i64`).
pub fn abs(x: i64) -> Result<i64, MathErr> {
    x.checked_abs().ok_or(MathErr::Overflow)
}

/// `neg(x)` — negation of a signed integer.
///
/// **Guarantee: `Exact`; fallible at `i64::MIN`.**
///
/// # Errors
///
/// - [`MathErr::Overflow`] when `x == i64::MIN`.
pub fn neg(x: i64) -> Result<i64, MathErr> {
    x.checked_neg().ok_or(MathErr::Overflow)
}

/// `signum(x)` — signum of a signed integer: -1, 0, or 1.
///
/// **Guarantee: `Exact`, total.**
#[must_use]
pub fn signum(x: i64) -> i64 {
    x.signum()
}

/// `min(a, b)` — minimum of two signed integers.
///
/// **Guarantee: `Exact`, total.**
///
/// Tie rule: when `a == b`, returns `a` (the first argument). This is documented, stable, and
/// never a silent loss of which input won (C1).
#[must_use]
pub fn min(a: i64, b: i64) -> i64 {
    if a <= b {
        a
    } else {
        b
    }
}

/// `max(a, b)` — maximum of two signed integers.
///
/// **Guarantee: `Exact`, total.**
///
/// Tie rule: when `a == b`, returns `a` (the first argument). Documented and stable (C1).
#[must_use]
pub fn max(a: i64, b: i64) -> i64 {
    if a >= b {
        a
    } else {
        b
    }
}

/// `gcd(a, b)` — greatest common divisor (always non-negative).
///
/// **Guarantee: `Exact`; fallible on overflow.** Uses the binary GCD (Stein's algorithm) on the
/// absolute values. `gcd(0, 0) = 0` (conventional).
///
/// # Errors
///
/// - [`MathErr::Overflow`] when the true gcd is `2^63` — i.e. when both inputs are drawn from
///   `{0, i64::MIN}` with at least one `i64::MIN` (e.g. `gcd(0, i64::MIN)`). `|i64::MIN| = 2^63`
///   is not representable in `i64`, so it is refused rather than wrapped to a negative value (C1).
pub fn gcd(a: i64, b: i64) -> Result<i64, MathErr> {
    // Work on absolute values in u64 — `i64::MIN.unsigned_abs() == 2^63` fits u64.
    let mut a = a.unsigned_abs();
    let mut b = b.unsigned_abs();
    let result: u64 = if a == 0 {
        b
    } else if b == 0 {
        a
    } else {
        // Stein's binary GCD.
        let shift = (a | b).trailing_zeros();
        a >>= a.trailing_zeros();
        loop {
            b >>= b.trailing_zeros();
            if a > b {
                core::mem::swap(&mut a, &mut b);
            }
            b -= a;
            if b == 0 {
                break;
            }
        }
        a << shift
    };
    // The result is non-negative; refuse the lone `2^63` case rather than wrap (C1/VR-5).
    i64::try_from(result).map_err(|_| MathErr::Overflow)
}

/// `lcm(a, b)` — least common multiple (always non-negative).
///
/// **Guarantee: `Exact`; fallible on overflow.**
///
/// # Errors
///
/// - [`MathErr::Overflow`] when the result overflows `i64`.
pub fn lcm(a: i64, b: i64) -> Result<i64, MathErr> {
    if a == 0 || b == 0 {
        return Ok(0);
    }
    let g = gcd(a, b)?;
    // Compute |a| / gcd first to minimize overflow risk, then multiply |b|.
    let a_abs = a.unsigned_abs();
    let b_abs = b.unsigned_abs();
    let g_u = g as u64;
    let half = a_abs / g_u;
    // Detect overflow before casting back to i64.
    let product = half.checked_mul(b_abs).ok_or(MathErr::Overflow)?;
    if product > i64::MAX as u64 {
        return Err(MathErr::Overflow);
    }
    Ok(product as i64)
}

/// `checked_div(a, b)` — exact integer division.
///
/// **Guarantee: `Exact`; `Err(DivByZero)` when `b == 0`.**
///
/// Also returns `Err(Overflow)` for `i64::MIN / -1` (the only overflow case for i64 division).
///
/// # Errors
///
/// - [`MathErr::DivByZero`] when `b == 0`.
/// - [`MathErr::Overflow`] when `a == i64::MIN` and `b == -1`.
pub fn checked_div(a: i64, b: i64) -> Result<i64, MathErr> {
    if b == 0 {
        return Err(MathErr::DivByZero);
    }
    a.checked_div(b).ok_or(MathErr::Overflow)
}

/// `checked_rem(a, b)` — exact integer remainder (`a mod b`, truncated toward zero).
///
/// **Guarantee: `Exact`; `Err(DivByZero)` when `b == 0`.**
///
/// # Errors
///
/// - [`MathErr::DivByZero`] when `b == 0`.
/// - [`MathErr::Overflow`] when `a == i64::MIN` and `b == -1`.
pub fn checked_rem(a: i64, b: i64) -> Result<i64, MathErr> {
    if b == 0 {
        return Err(MathErr::DivByZero);
    }
    a.checked_rem(b).ok_or(MathErr::Overflow)
}

/// `ratio(a, b)` — exact rational representation of `a/b`.
///
/// Returns a pair `(numerator, denominator)` in reduced form (the denominator is always positive).
/// When `b == 0`, returns `Err(DivByZero)`.
///
/// **Guarantee: `Exact`; `Err(DivByZero)` when `b == 0`.**
///
/// # Errors
///
/// - [`MathErr::DivByZero`] when `b == 0`.
/// - [`MathErr::Overflow`] when reducing/canonicalizing would leave a non-representable value
///   (e.g. `ratio(i64::MIN, -1) = 2^63`); refused rather than wrapped (C1).
pub fn ratio(a: i64, b: i64) -> Result<(i64, i64), MathErr> {
    if b == 0 {
        return Err(MathErr::DivByZero);
    }
    // `gcd` takes the absolute values of its arguments, so pass `a`/`b` directly — this also
    // avoids an `unsigned_abs() as i64` overflow at `i64::MIN`.
    let g = gcd(a, b)?;
    let (num, den) = if g == 0 { (a, b) } else { (a / g, b / g) };
    // Ensure denominator is always positive (canonical form); negation can overflow at i64::MIN.
    if den < 0 {
        Ok((
            num.checked_neg().ok_or(MathErr::Overflow)?,
            den.checked_neg().ok_or(MathErr::Overflow)?,
        ))
    } else {
        Ok((num, den))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- property: min/max tie rule is documented and stable ----

    #[test]
    fn min_tie_returns_first_arg() {
        // Documented tie rule: min(a, b) returns `a` when a == b (C1: never silent).
        assert_eq!(min(5, 5), 5);
        assert_eq!(min(-3, -3), -3);
        assert_eq!(min(0, 0), 0);
    }

    #[test]
    fn max_tie_returns_first_arg() {
        assert_eq!(max(7, 7), 7);
        assert_eq!(max(-1, -1), -1);
    }

    #[test]
    fn min_selects_smaller() {
        assert_eq!(min(3, 5), 3);
        assert_eq!(min(-5, -3), -5);
        assert_eq!(min(0, 1), 0);
    }

    #[test]
    fn max_selects_larger() {
        assert_eq!(max(3, 5), 5);
        assert_eq!(max(-5, -3), -3);
    }

    // ---- property: abs is exact except at i64::MIN ----

    #[test]
    fn abs_is_exact_for_normal_values() {
        // Property: abs(x) == |x| for all i64 values except i64::MIN.
        for x in [0i64, 1, -1, 100, -100, i64::MAX, i64::MIN + 1] {
            let expected = if x >= 0 { x } else { -x };
            assert_eq!(abs(x).unwrap(), expected, "abs({x})");
        }
    }

    #[test]
    fn abs_min_i64_is_overflow_not_silent() {
        // C1: i64::MIN has no representable abs — must be Err(Overflow), not a silent wrong value.
        assert_eq!(abs(i64::MIN), Err(MathErr::Overflow));
    }

    #[test]
    fn neg_min_i64_is_overflow() {
        assert_eq!(neg(i64::MIN), Err(MathErr::Overflow));
    }

    // ---- property: signum is -1, 0, or +1 for every i64 ----

    #[test]
    fn signum_covers_all_cases() {
        assert_eq!(signum(0), 0);
        assert_eq!(signum(42), 1);
        assert_eq!(signum(-42), -1);
        assert_eq!(signum(i64::MAX), 1);
        assert_eq!(signum(i64::MIN), -1);
    }

    // ---- property: gcd bounds (gcd(a,b) divides both a and b) ----

    #[test]
    fn gcd_divides_both_arguments() {
        // Property: gcd(a, b) divides both a and b.
        let cases = [(12i64, 8i64), (35, 14), (100, 75), (17, 13), (0, 5), (5, 0)];
        for (a, b) in cases {
            let g = gcd(a, b).expect("in-range gcd");
            if a != 0 {
                assert_eq!(a % g, 0, "gcd({a},{b})={g} must divide {a}");
            }
            if b != 0 {
                assert_eq!(b % g, 0, "gcd({a},{b})={g} must divide {b}");
            }
        }
    }

    #[test]
    fn gcd_zero_zero_is_zero() {
        assert_eq!(gcd(0, 0).unwrap(), 0);
    }

    #[test]
    fn gcd_known_values() {
        assert_eq!(gcd(12, 8).unwrap(), 4);
        assert_eq!(gcd(35, 14).unwrap(), 7);
        assert_eq!(gcd(17, 13).unwrap(), 1); // coprime
        assert_eq!(gcd(0, 5).unwrap(), 5);
        assert_eq!(gcd(5, 0).unwrap(), 5);
    }

    #[test]
    fn gcd_i64_min_overflow_is_explicit_error() {
        // |i64::MIN| = 2^63 is not representable in i64 — refused, never wrapped to a negative
        // (C1/VR-5). The only inputs whose true gcd is 2^63 are drawn from {0, i64::MIN}.
        assert_eq!(gcd(0, i64::MIN), Err(MathErr::Overflow));
        assert_eq!(gcd(i64::MIN, 0), Err(MathErr::Overflow));
        assert_eq!(gcd(i64::MIN, i64::MIN), Err(MathErr::Overflow));
        // A mixed case whose gcd fits is still fine: gcd(i64::MIN, 6) = 2.
        assert_eq!(gcd(i64::MIN, 6).unwrap(), 2);
    }

    // ---- property: lcm(a,b) = |a*b| / gcd(a,b) ----

    #[test]
    fn lcm_known_values() {
        assert_eq!(lcm(4, 6).unwrap(), 12);
        assert_eq!(lcm(3, 7).unwrap(), 21);
        assert_eq!(lcm(0, 5).unwrap(), 0);
        assert_eq!(lcm(5, 0).unwrap(), 0);
    }

    #[test]
    fn lcm_overflow_is_explicit_error() {
        // i64::MAX * 2 overflows — must be Err(Overflow).
        assert_eq!(lcm(i64::MAX, 2), Err(MathErr::Overflow));
    }

    // ---- property: checked_div is exact and div-by-zero is explicit ----

    #[test]
    fn checked_div_exact_cases() {
        assert_eq!(checked_div(10, 3).unwrap(), 3); // truncated toward zero
        assert_eq!(checked_div(-10, 3).unwrap(), -3);
        assert_eq!(checked_div(10, -3).unwrap(), -3);
        assert_eq!(checked_div(9, 3).unwrap(), 3);
    }

    #[test]
    fn checked_div_zero_divisor_is_explicit_error() {
        // C1: never silent — division by zero must be Err.
        assert_eq!(checked_div(10, 0), Err(MathErr::DivByZero));
        assert_eq!(checked_div(0, 0), Err(MathErr::DivByZero));
    }

    #[test]
    fn checked_rem_zero_divisor_is_explicit_error() {
        assert_eq!(checked_rem(10, 0), Err(MathErr::DivByZero));
    }

    #[test]
    fn checked_rem_exact_cases() {
        assert_eq!(checked_rem(10, 3).unwrap(), 1);
        assert_eq!(checked_rem(-10, 3).unwrap(), -1);
    }

    // ---- property: ratio is in reduced canonical form ----

    #[test]
    fn ratio_reduced_form() {
        // gcd(6,4)=2 → ratio(6,4) = (3,2).
        assert_eq!(ratio(6, 4).unwrap(), (3, 2));
        // Negative denominator normalised.
        assert_eq!(ratio(3, -4).unwrap(), (-3, 4));
        // Already reduced.
        assert_eq!(ratio(1, 3).unwrap(), (1, 3));
        // Zero numerator.
        assert_eq!(ratio(0, 5).unwrap(), (0, 1));
    }

    #[test]
    fn ratio_div_by_zero_is_explicit_error() {
        assert_eq!(ratio(5, 0), Err(MathErr::DivByZero));
    }

    // ---- property: floor/ceil/trunc are exact on finite f64 ----

    #[test]
    fn floor_is_exact() {
        assert_eq!(floor(2.9).unwrap(), 2);
        assert_eq!(floor(-2.1).unwrap(), -3);
        assert_eq!(floor(3.0).unwrap(), 3);
    }

    #[test]
    fn ceil_is_exact() {
        assert_eq!(ceil(2.1).unwrap(), 3);
        assert_eq!(ceil(-2.9).unwrap(), -2);
        assert_eq!(ceil(3.0).unwrap(), 3);
    }

    #[test]
    fn trunc_is_exact() {
        assert_eq!(trunc(2.9).unwrap(), 2);
        assert_eq!(trunc(-2.9).unwrap(), -2);
        assert_eq!(trunc(3.0).unwrap(), 3);
    }

    #[test]
    fn floor_nan_is_explicit_error() {
        assert_eq!(floor(f64::NAN), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn floor_inf_is_explicit_error() {
        assert_eq!(floor(f64::INFINITY), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn rounding_out_of_i64_range_is_explicit_overflow_never_silent() {
        // A finite f64 far beyond i64 range must NOT silently saturate to i64::MAX (C1/G2):
        // `1e20_f64 as i64` is `i64::MAX` in Rust, so a naive cast would return a wrong value.
        let big = 1e20_f64;
        assert_eq!(floor(big), Err(MathErr::Overflow));
        assert_eq!(ceil(big), Err(MathErr::Overflow));
        assert_eq!(trunc(big), Err(MathErr::Overflow));
        assert_eq!(round(big, RoundMode::HalfToEven), Err(MathErr::Overflow));
        assert_eq!(floor(-1e20_f64), Err(MathErr::Overflow));
        // The exact boundary: 2^63 is one past i64::MAX and must be refused...
        assert_eq!(trunc(9_223_372_036_854_775_808.0), Err(MathErr::Overflow));
        // ...while the largest representable f64 below 2^63 is accepted exactly.
        assert_eq!(
            trunc(9_223_372_036_854_774_784.0).unwrap(),
            9_223_372_036_854_774_784
        );
        // i64::MIN (-2^63) is exactly representable and in range.
        assert_eq!(trunc(-9_223_372_036_854_775_808.0).unwrap(), i64::MIN);
    }

    #[test]
    fn ratio_i64_min_canonicalization_overflow_is_explicit() {
        // ratio(i64::MIN, -1) = 2^63 is not representable — refused, never wrapped (C1).
        assert_eq!(ratio(i64::MIN, -1), Err(MathErr::Overflow));
    }

    // ---- property: round mode is always returned as the EXPLAIN artifact ----

    #[test]
    fn round_mode_is_echoed_back_as_explain_artifact() {
        // C3: the mode is the EXPLAIN artifact — it must always be returned.
        for mode in [
            RoundMode::Floor,
            RoundMode::Ceil,
            RoundMode::TruncTowardZero,
            RoundMode::HalfAwayFromZero,
            RoundMode::HalfToEven,
        ] {
            let (_, returned_mode) = round(2.5, mode).unwrap();
            assert_eq!(returned_mode, mode, "mode must be echoed back");
        }
    }

    #[test]
    fn round_half_even_rounds_to_even() {
        // 0.5 → nearest even: 0 (0 is even).
        assert_eq!(round(0.5, RoundMode::HalfToEven).unwrap().0, 0);
        // 1.5 → nearest even: 2 (2 is even).
        assert_eq!(round(1.5, RoundMode::HalfToEven).unwrap().0, 2);
        // 2.5 → nearest even: 2 (2 is even).
        assert_eq!(round(2.5, RoundMode::HalfToEven).unwrap().0, 2);
        // -0.5 → nearest even: 0 (0 is even).
        assert_eq!(round(-0.5, RoundMode::HalfToEven).unwrap().0, 0);
        // -1.5 → nearest even: -2 (-2 is even).
        assert_eq!(round(-1.5, RoundMode::HalfToEven).unwrap().0, -2);
    }

    #[test]
    fn round_half_away_from_zero() {
        assert_eq!(round(0.5, RoundMode::HalfAwayFromZero).unwrap().0, 1);
        assert_eq!(round(1.5, RoundMode::HalfAwayFromZero).unwrap().0, 2);
        assert_eq!(round(-0.5, RoundMode::HalfAwayFromZero).unwrap().0, -1);
    }

    #[test]
    fn round_nan_is_explicit_error() {
        assert_eq!(round(f64::NAN, RoundMode::Floor), Err(MathErr::OutOfDomain));
    }
}
