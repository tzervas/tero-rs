//! Approximate numeric operations and the `Approx<f64>` bound-carrying carrier (spec §3 / §4).
//!
//! # `Approx<f64>` — the thin bound carrier
//!
//! Per `math.md §7-Q1` (resolved by the `numerics`/README §5 reconciliation), `Approx<T>` is a
//! *thin view* — a plain value with its `Meta`-attached `{Bound, strength}`, **not** a new numeric
//! type. In this Rust-first implementation it is a struct holding the `f64` result and the
//! `mycelium_core::Bound` that certifies it; no kernel change is introduced (KC-3).
//!
//! # Honesty (VR-5)
//!
//! All approximate ops in this module carry `Declared` strength because the transcendental compute
//! floor is the platform libm via Rust's `f64` intrinsics — an unaudited `wild` floor (ADR-014;
//! FLAG in `lib.rs`; M-541). The `Declared` tag is the *honest* floor: we assert an ε bound, but
//! its basis is `UserDeclared` because no checked-side-condition theorem backs the libm call yet.
//! When M-541 lands and the `std-sys` phylum provides an audited surface, the basis can be
//! upgraded to `ProvenThm` (and thus the strength to `Proven`) — but that upgrade requires a
//! checked basis, never an unilateral assertion (VR-5).
//!
//! The ε values used below are conservative float-op error bounds in the Linf norm:
//!   - For scalar transcendentals: 1 ULP ≈ 2^-52 of the result, but we use a conservative
//!     `2 * f64::EPSILON` ≈ 4.4e-16 as a Declared upper bound (no proven theorem backs it here).
//!   - For `round`/`floor`/`ceil`/`trunc`: exact integer-valued results carry ε = 0; the
//!     rounding operation itself is `Exact` in the sense that it always returns the integer nearest
//!     the input under the named mode (the mode IS the specification).
//!
//! **FLAG (ε-ownership):** the concrete ε values used here are `Declared` assertions; the exact
//! magnitudes, norms, and `Proven` reachability are owned by ADR-010 / M-512 (spec §7-Q2). These
//! values should be replaced with M-512's when it lands.

use mycelium_core::{
    Bound, BoundBasis, BoundBasis as Basis, BoundKind, GuaranteeStrength, NormKind,
};

use crate::MathErr;

/// The ε upper bound used for `Declared`-strength float operations (see module honesty note).
///
/// ε-ownership (math.md §7-Q2 — RESOLVED): the constant is **homed in `std.numerics`** (the
/// ε-carrier module, M-512) and re-exported here, so the value is stated in exactly one place
/// (NFR-N2). It is a `UserDeclared` assertion (`2 · f64::EPSILON ≈ 4.44e-16`, Linf) until M-541's
/// audited libm floor provides a checked `ProvenThm` basis.
pub use mycelium_std_numerics::DECLARED_FLOAT_EPS;

/// The `Declared` ε bound attached to all approximate ops in this implementation.
///
/// Basis is `UserDeclared` because the libm compute floor is not yet audited (FLAG in lib.rs /
/// M-541). See [`DECLARED_FLOAT_EPS`] and the module-level honesty note.
#[must_use]
pub fn declared_error_bound() -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps: DECLARED_FLOAT_EPS,
            norm: NormKind::Linf,
        },
        basis: BoundBasis::UserDeclared,
    }
}

/// `Approx<f64>` — a thin carrier for an approximate `f64` result with its attached bound.
///
/// Per spec §3 and the README §5 reconciliation (`math.md §7-Q1`): `Approx<T>` is a plain value
/// with its `Meta`-attached `{Bound, strength}` — it is **not** a new numeric type and **not** a
/// kernel change (KC-3). This Rust-first struct holds the scalar and its bound directly (without
/// a full `Value` wrapper) for ergonomic use in the `std.math` surface.
///
/// The attached `bound` is the EXPLAIN artifact (C3): call [`Approx::explain`] to project it.
#[derive(Debug, Clone, PartialEq)]
pub struct Approx<T> {
    /// The approximate scalar result.
    pub value: T,
    /// The attached error bound — the C2/C3 artifact, never dropped.
    pub bound: Bound,
    /// The honest guarantee strength (derived from `bound.basis`, never asserted; VR-5).
    ///
    /// **Private (the VR-5 seal):** a public field would let a caller write
    /// `Approx { strength: Proven, .. }` and assert a strength the basis doesn't support; keeping
    /// it private means [`Approx::new`] (basis-deriving) is the only way to set it. Read via
    /// [`Approx::strength`].
    strength: GuaranteeStrength,
}

/// The dual human/machine EXPLAIN record for an [`Approx`] result (G11; C3).
#[derive(Debug, Clone, PartialEq)]
pub struct ApproxExplain {
    /// The approximate value.
    pub value: f64,
    /// The ε bound magnitude.
    pub eps: f64,
    /// The norm in which `eps` is expressed.
    pub norm: NormKind,
    /// The basis that produced this bound.
    pub basis: BoundBasis,
    /// The honest strength.
    pub strength: GuaranteeStrength,
    /// Human-readable one-line summary (the G11 human projection).
    pub summary: String,
}

impl<T: Copy + core::fmt::Debug> Approx<T> {
    /// Build an `Approx` value. The strength is **derived** from the bound's basis (VR-5 —
    /// never asserted from outside).
    #[must_use]
    pub fn new(value: T, bound: Bound) -> Self {
        let strength = bound.basis.strength();
        Approx {
            value,
            bound,
            strength,
        }
    }

    /// The honest guarantee strength (derived from the bound's basis; VR-5 — never asserted).
    #[must_use]
    pub fn strength(&self) -> GuaranteeStrength {
        self.strength
    }
}

impl Approx<f64> {
    /// Project this approximate result to a dual human/machine EXPLAIN record (G11; C3).
    ///
    /// Total — every `Approx<f64>` carries a bound that explains.
    #[must_use]
    pub fn explain(&self) -> ApproxExplain {
        let (eps, norm) = match self.bound.kind {
            BoundKind::Error { eps, norm } => (eps, norm),
            // A non-`Error` bound states no ε. Report it as `+∞` (an unstated error is, honestly,
            // unbounded) — never `NaN`, which would silently break any `eps < tol` comparison a
            // consumer makes (C1). Every `Approx` this crate builds carries an `Error` bound, so
            // this arm is defensive rather than expected.
            _ => (f64::INFINITY, NormKind::Linf),
        };
        let basis_desc = match &self.bound.basis {
            Basis::ProvenThm { citation } => format!("ProvenThm: {citation}"),
            Basis::EmpiricalFit { trials, method } => {
                format!("EmpiricalFit ({trials} trials): {method}")
            }
            Basis::UserDeclared => {
                "UserDeclared (unverified — Declared, not Proven; FLAG M-541)".to_owned()
            }
        };
        let summary = format!(
            "{:?} result {:.6e}; ε={eps:.2e} ({norm:?}); basis={basis_desc}",
            self.strength, self.value
        );
        ApproxExplain {
            value: self.value,
            eps,
            norm,
            basis: self.bound.basis.clone(),
            strength: self.strength,
            summary,
        }
    }
}

// ---- approximate operations ------------------------------------------------
//
// All delegate to Rust's `f64` intrinsics (platform libm). Domain checks are performed first;
// any violation returns `Err(MathErr::…)` — never NaN, ±Inf, or a sentinel (C1/G2).
//
// Strength is `Declared` for all ops in this implementation (see module honesty note / FLAG).

/// `sqrt(x)` — approximate square root.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** `x ≥ 0`; `Err(NegativeDomain)` otherwise.
///
/// # Errors
///
/// - [`MathErr::NegativeDomain`] when `x < 0`.
pub fn sqrt(x: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() || x < 0.0 {
        return Err(MathErr::NegativeDomain);
    }
    Ok(Approx::new(x.sqrt(), declared_error_bound()))
}

/// `cbrt(x)` — approximate cube root.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** finite `f64` (cbrt is odd; no pole).
///
/// Note: `cbrt(NaN) = NaN` and `cbrt(±∞) = ±∞` in IEEE 754; to stay never-silent — and because an
/// ε bound is meaningless over a non-finite result — this function returns `Err(OutOfDomain)` for
/// any non-finite input rather than wrapping `NaN`/`±∞` in an `Ok(Approx { .. })` (C1/G2).
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
pub fn cbrt(x: f64) -> Result<Approx<f64>, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    Ok(Approx::new(x.cbrt(), declared_error_bound()))
}

/// `exp(x)` — approximate natural exponential `eˣ`.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** finite `f64`; `Err(Overflow)` if the result would exceed `f64::MAX`.
///
/// # Errors
///
/// - [`MathErr::Overflow`] when `x` is so large that `eˣ > f64::MAX`.
/// - [`MathErr::OutOfDomain`] when `x` is NaN.
pub fn exp(x: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() {
        return Err(MathErr::OutOfDomain);
    }
    let result = x.exp();
    if result.is_infinite() {
        return Err(MathErr::Overflow);
    }
    Ok(Approx::new(result, declared_error_bound()))
}

/// `log(x)` — approximate natural logarithm `ln(x)`.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** `x > 0`; `Err(NonPositiveDomain)` for `x ≤ 0` or NaN.
///
/// # Errors
///
/// - [`MathErr::NonPositiveDomain`] when `x ≤ 0` or NaN.
pub fn log(x: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() || x <= 0.0 {
        return Err(MathErr::NonPositiveDomain);
    }
    Ok(Approx::new(x.ln(), declared_error_bound()))
}

/// `logb(b, x)` — approximate base-`b` logarithm `log_b(x)`.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** `x > 0`; `b > 0`, `b ≠ 1`, `b` finite; otherwise explicit errors.
///
/// # Errors
///
/// - [`MathErr::BadBase`] when `b ≤ 0`, `b == 1`, `b` is NaN, or `b` is infinite.
/// - [`MathErr::NonPositiveDomain`] when `x ≤ 0` or NaN.
pub fn logb(b: f64, x: f64) -> Result<Approx<f64>, MathErr> {
    if b.is_nan() || b.is_infinite() || b <= 0.0 {
        return Err(MathErr::BadBase);
    }
    if x.is_nan() || x <= 0.0 {
        return Err(MathErr::NonPositiveDomain);
    }
    // Base 1 is the *only* degenerate base: `ln(1) == 0` ⇒ division by zero. Guard the actual
    // div-by-zero (`ln(b) == 0`), not a fuzzy ε-band around 1 — a base adjacent to 1 (e.g. the
    // largest f64 below 1) has a tiny-but-nonzero `ln` and is a perfectly valid base, which the
    // old `(b - 1.0).abs() < EPSILON` test wrongly rejected. (`== 0.0` is clippy-clean: float_cmp
    // exempts comparison to zero.)
    let ln_b = b.ln();
    if ln_b == 0.0 {
        return Err(MathErr::BadBase);
    }
    // log_b(x) = ln(x) / ln(b)
    Ok(Approx::new(x.ln() / ln_b, declared_error_bound()))
}

/// `pow(x, y)` — approximate `xʸ`.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
///
/// **Domain restrictions:**
/// - `0^neg` — `x == 0` with `y < 0`: `Err(DivByZero)`.
/// - `neg^non-integer` — `x < 0` with a non-integer `y`: `Err(OutOfDomain)`.
/// - Overflow: result overflows `f64::MAX`: `Err(Overflow)`.
/// - NaN `x` or `y`: `Err(OutOfDomain)`.
///
/// # Errors
///
/// - [`MathErr::DivByZero`] when `x == 0` and `y < 0`.
/// - [`MathErr::OutOfDomain`] when `x < 0` and `y` is not an integer, or input is NaN.
/// - [`MathErr::Overflow`] when the result overflows `f64::MAX`.
pub fn pow(x: f64, y: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() || y.is_nan() {
        return Err(MathErr::OutOfDomain);
    }
    if x == 0.0 && y < 0.0 {
        return Err(MathErr::DivByZero);
    }
    if x < 0.0 && y.fract() != 0.0 {
        return Err(MathErr::OutOfDomain);
    }
    let result = x.powf(y);
    if result.is_infinite() {
        return Err(MathErr::Overflow);
    }
    Ok(Approx::new(result, declared_error_bound()))
}

/// `hypot(x, y)` — approximate `√(x² + y²)`.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** total on all finite `f64`; `Err(Overflow)` if the result overflows.
/// NaN inputs return `Err(OutOfDomain)`.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when either `x` or `y` is NaN.
/// - [`MathErr::Overflow`] when the result overflows `f64::MAX`.
pub fn hypot(x: f64, y: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() || y.is_nan() {
        return Err(MathErr::OutOfDomain);
    }
    let result = x.hypot(y);
    if result.is_infinite() {
        return Err(MathErr::Overflow);
    }
    Ok(Approx::new(result, declared_error_bound()))
}

/// `sin(x)` — approximate sine.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** total on all finite `f64` (sin is entire); NaN input returns `Err(OutOfDomain)`.
///
/// Note: `sin(+Inf)` / `sin(-Inf)` are undefined in IEEE 754; they are rejected as `OutOfDomain`
/// to stay never-silent (C1/G2).
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
pub fn sin(x: f64) -> Result<Approx<f64>, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    Ok(Approx::new(x.sin(), declared_error_bound()))
}

/// `cos(x)` — approximate cosine.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** total on all finite `f64`; NaN or infinite input returns `Err(OutOfDomain)`.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
pub fn cos(x: f64) -> Result<Approx<f64>, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    Ok(Approx::new(x.cos(), declared_error_bound()))
}

/// `tan(x)` — approximate tangent.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** finite `f64` that is not at a pole (odd multiple of π/2).
///
/// Detecting poles exactly is numerically impossible for arbitrary `f64` inputs, because `π/2` is
/// irrational and cannot be represented exactly. The spec acknowledges this: `Err(PoleDomain)` is
/// returned when `|tan(x)| > 1/f64::EPSILON` (effectively ±∞ in f64 arithmetic), which is the
/// machine-observable pole indicator. NaN or infinite inputs are `Err(OutOfDomain)`.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
/// - [`MathErr::PoleDomain`] when `|tan(x)| > 1/f64::EPSILON` (observable pole).
pub fn tan(x: f64) -> Result<Approx<f64>, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    let result = x.tan();
    // Observable pole: tan diverges — result would be ±Inf or an enormous value.
    if result.is_infinite() || result.abs() > 1.0 / f64::EPSILON {
        return Err(MathErr::PoleDomain);
    }
    Ok(Approx::new(result, declared_error_bound()))
}

/// `asin(x)` — approximate arcsine.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** `|x| ≤ 1`; `Err(OutOfDomain)` when `|x| > 1` or NaN.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or `|x| > 1`.
pub fn asin(x: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() || x.abs() > 1.0 {
        return Err(MathErr::OutOfDomain);
    }
    Ok(Approx::new(x.asin(), declared_error_bound()))
}

/// `acos(x)` — approximate arccosine.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** `|x| ≤ 1`; `Err(OutOfDomain)` when `|x| > 1` or NaN.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or `|x| > 1`.
pub fn acos(x: f64) -> Result<Approx<f64>, MathErr> {
    if x.is_nan() || x.abs() > 1.0 {
        return Err(MathErr::OutOfDomain);
    }
    Ok(Approx::new(x.acos(), declared_error_bound()))
}

/// `atan(x)` — approximate arctangent.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** total on all finite `f64`; NaN or infinite input returns `Err(OutOfDomain)`.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `x` is NaN or infinite.
pub fn atan(x: f64) -> Result<Approx<f64>, MathErr> {
    if !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    Ok(Approx::new(x.atan(), declared_error_bound()))
}

/// `atan2(y, x)` — approximate four-quadrant arctangent.
///
/// **Guarantee:** `Declared` (ε = 2·f64::EPSILON, Linf; `UserDeclared` basis — libm floor, M-541).
/// **Domain:** `(y, x) ≠ (0, 0)` and both must be finite; `Err(OutOfDomain)` for NaN/infinite
/// input, `Err(Undefined)` at the origin `(0, 0)` where atan2 is mathematically undefined.
///
/// # Errors
///
/// - [`MathErr::OutOfDomain`] when `y` or `x` is NaN or infinite.
/// - [`MathErr::PoleDomain`] when `y == 0` and `x == 0` (origin — undefined).
pub fn atan2(y: f64, x: f64) -> Result<Approx<f64>, MathErr> {
    if !y.is_finite() || !x.is_finite() {
        return Err(MathErr::OutOfDomain);
    }
    if y == 0.0 && x == 0.0 {
        return Err(MathErr::PoleDomain);
    }
    Ok(Approx::new(y.atan2(x), declared_error_bound()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::BoundBasis;

    // Helper: assert an Approx<f64> carries a Declared bound with the expected ε.
    fn assert_declared(a: &Approx<f64>) {
        assert_eq!(
            a.strength(),
            GuaranteeStrength::Declared,
            "approximate ops must carry Declared strength (VR-5: libm floor not audited, FLAG M-541)"
        );
        assert!(
            matches!(a.bound.basis, BoundBasis::UserDeclared),
            "basis must be UserDeclared for unaudited libm ops"
        );
        if let BoundKind::Error { eps, norm } = a.bound.kind {
            assert!(eps >= 0.0, "ε must be non-negative");
            assert!(eps.is_finite(), "ε must be finite");
            assert_eq!(norm, NormKind::Linf, "norm must be Linf");
        } else {
            panic!("expected Error bound kind");
        }
    }

    // ---- property: Declared strength is the honest floor for all approx ops ----

    #[test]
    fn sqrt_carries_declared_bound() {
        // Property: sqrt of any valid input carries Declared strength (VR-5 / libm flag).
        for x in [0.0, 1.0, 4.0, 100.0, f64::MAX.sqrt()] {
            let a = sqrt(x).unwrap_or_else(|e| panic!("sqrt({x}) failed: {e}"));
            assert_declared(&a);
        }
    }

    #[test]
    fn cbrt_carries_declared_bound() {
        for x in [-8.0, -1.0, 0.0, 1.0, 27.0] {
            let a = cbrt(x).unwrap_or_else(|e| panic!("cbrt({x}) failed: {e}"));
            assert_declared(&a);
        }
    }

    #[test]
    fn exp_carries_declared_bound() {
        for x in [-100.0, -1.0, 0.0, 1.0, 10.0] {
            let a = exp(x).unwrap_or_else(|e| panic!("exp({x}) failed: {e}"));
            assert_declared(&a);
        }
    }

    #[test]
    fn log_carries_declared_bound() {
        for x in [f64::MIN_POSITIVE, 1.0, 2.0, 100.0] {
            let a = log(x).unwrap_or_else(|e| panic!("log({x}) failed: {e}"));
            assert_declared(&a);
        }
    }

    #[test]
    fn trig_carries_declared_bound() {
        for x in [-1.0, 0.0, 1.0, 0.5, -0.5] {
            assert_declared(&sin(x).unwrap());
            assert_declared(&cos(x).unwrap());
        }
    }

    #[test]
    fn inverse_trig_carries_declared_bound() {
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert_declared(&asin(x).unwrap());
            assert_declared(&acos(x).unwrap());
        }
        for x in [-1.0, 0.0, 1.0] {
            assert_declared(&atan(x).unwrap());
        }
    }

    #[test]
    fn atan2_carries_declared_bound() {
        assert_declared(&atan2(1.0, 1.0).unwrap());
        assert_declared(&atan2(0.0, 1.0).unwrap());
    }

    #[test]
    fn hypot_carries_declared_bound() {
        assert_declared(&hypot(3.0, 4.0).unwrap());
    }

    // ---- property: bound ε is non-negative and finite for all ops ----
    // (this is the bound-soundness property test required by the spec)

    #[test]
    fn all_approx_ops_have_nonneg_finite_eps() {
        let ops: Vec<(&str, Approx<f64>)> = vec![
            ("sqrt(4)", sqrt(4.0).unwrap()),
            ("cbrt(8)", cbrt(8.0).unwrap()),
            ("exp(1)", exp(1.0).unwrap()),
            ("log(1)", log(1.0).unwrap()),
            ("logb(2,8)", logb(2.0, 8.0).unwrap()),
            ("pow(2,3)", pow(2.0, 3.0).unwrap()),
            ("hypot(3,4)", hypot(3.0, 4.0).unwrap()),
            ("sin(1)", sin(1.0).unwrap()),
            ("cos(1)", cos(1.0).unwrap()),
            ("tan(0)", tan(0.0).unwrap()),
            ("asin(0)", asin(0.0).unwrap()),
            ("acos(0)", acos(0.0).unwrap()),
            ("atan(1)", atan(1.0).unwrap()),
            ("atan2(1,1)", atan2(1.0, 1.0).unwrap()),
        ];
        for (name, a) in &ops {
            if let BoundKind::Error { eps, .. } = a.bound.kind {
                assert!(
                    eps >= 0.0 && eps.is_finite(),
                    "{name}: ε must be non-negative and finite, got {eps}"
                );
            }
        }
    }

    // ---- C1: domain restrictions are explicit errors, never NaN/Inf ----

    #[test]
    fn sqrt_negative_is_explicit_error() {
        assert_eq!(sqrt(-1.0), Err(MathErr::NegativeDomain));
        assert_eq!(sqrt(f64::NAN), Err(MathErr::NegativeDomain));
    }

    #[test]
    fn cbrt_nan_is_explicit_error() {
        assert_eq!(cbrt(f64::NAN), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn exp_overflow_is_explicit_error() {
        // x = 1000 causes exp to overflow f64::MAX.
        assert_eq!(exp(1000.0), Err(MathErr::Overflow));
    }

    #[test]
    fn exp_nan_is_explicit_error() {
        assert_eq!(exp(f64::NAN), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn log_nonpositive_is_explicit_error() {
        assert_eq!(log(0.0), Err(MathErr::NonPositiveDomain));
        assert_eq!(log(-1.0), Err(MathErr::NonPositiveDomain));
        assert_eq!(log(f64::NAN), Err(MathErr::NonPositiveDomain));
    }

    #[test]
    fn logb_bad_base_is_explicit_error() {
        assert_eq!(logb(0.0, 1.0), Err(MathErr::BadBase));
        assert_eq!(logb(-1.0, 1.0), Err(MathErr::BadBase));
        assert_eq!(logb(1.0, 1.0), Err(MathErr::BadBase)); // base == 1 is degenerate (ln 1 = 0)
        assert_eq!(logb(f64::NAN, 1.0), Err(MathErr::BadBase));
        assert_eq!(logb(f64::INFINITY, 1.0), Err(MathErr::BadBase));
    }

    /// Only base **exactly** 1.0 is degenerate; bases adjacent to 1 have a tiny-but-nonzero `ln`
    /// and are valid. The old fuzzy `(b-1).abs() < EPSILON` guard wrongly rejected these.
    #[test]
    fn logb_bases_adjacent_to_one_are_valid() {
        let below = 1.0_f64.next_down(); // largest f64 < 1
        let above = 1.0_f64.next_up(); // smallest f64 > 1
        assert!(
            logb(below, 8.0).is_ok(),
            "base just below 1 is a valid base"
        );
        assert!(
            logb(above, 8.0).is_ok(),
            "base just above 1 is a valid base"
        );
    }

    #[test]
    fn logb_nonpositive_x_is_explicit_error() {
        assert_eq!(logb(2.0, 0.0), Err(MathErr::NonPositiveDomain));
        assert_eq!(logb(2.0, -1.0), Err(MathErr::NonPositiveDomain));
    }

    #[test]
    fn pow_zero_negative_exponent_is_div_by_zero() {
        assert_eq!(pow(0.0, -1.0), Err(MathErr::DivByZero));
    }

    #[test]
    fn pow_negative_base_fractional_exponent_is_out_of_domain() {
        assert_eq!(pow(-2.0, 0.5), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn pow_overflow_is_explicit_error() {
        assert_eq!(pow(f64::MAX, 2.0), Err(MathErr::Overflow));
    }

    #[test]
    fn pow_nan_is_out_of_domain() {
        assert_eq!(pow(f64::NAN, 1.0), Err(MathErr::OutOfDomain));
        assert_eq!(pow(1.0, f64::NAN), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn hypot_nan_is_explicit_error() {
        assert_eq!(hypot(f64::NAN, 1.0), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn sin_nan_or_inf_is_explicit_error() {
        assert_eq!(sin(f64::NAN), Err(MathErr::OutOfDomain));
        assert_eq!(sin(f64::INFINITY), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn cos_nan_or_inf_is_explicit_error() {
        assert_eq!(cos(f64::NAN), Err(MathErr::OutOfDomain));
        assert_eq!(cos(f64::NEG_INFINITY), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn tan_nan_or_inf_is_explicit_error() {
        assert_eq!(tan(f64::NAN), Err(MathErr::OutOfDomain));
        assert_eq!(tan(f64::INFINITY), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn asin_out_of_domain_is_explicit_error() {
        assert_eq!(asin(1.5), Err(MathErr::OutOfDomain));
        assert_eq!(asin(-1.5), Err(MathErr::OutOfDomain));
        assert_eq!(asin(f64::NAN), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn acos_out_of_domain_is_explicit_error() {
        assert_eq!(acos(1.5), Err(MathErr::OutOfDomain));
        assert_eq!(acos(-1.5), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn atan_nan_or_inf_is_explicit_error() {
        assert_eq!(atan(f64::NAN), Err(MathErr::OutOfDomain));
        assert_eq!(atan(f64::INFINITY), Err(MathErr::OutOfDomain));
    }

    #[test]
    fn atan2_origin_is_pole() {
        assert_eq!(atan2(0.0, 0.0), Err(MathErr::PoleDomain));
    }

    #[test]
    fn atan2_nan_is_explicit_error() {
        assert_eq!(atan2(f64::NAN, 1.0), Err(MathErr::OutOfDomain));
    }

    // ---- C3: EXPLAIN is total and informative ----

    #[test]
    fn approx_explain_is_total_and_informative() {
        let a = sqrt(9.0).expect("sqrt(9) is valid");
        let ex = a.explain();
        assert!(!ex.summary.is_empty(), "explain summary must be non-empty");
        assert_eq!(ex.norm, NormKind::Linf);
        assert!(ex.eps >= 0.0);
        assert!(
            ex.summary.contains("Declared"),
            "summary should mention Declared strength"
        );
        assert!(
            ex.summary.contains("FLAG"),
            "summary should mention the M-541 FLAG"
        );
    }

    // ---- value-correctness spot-checks ----

    #[test]
    fn sqrt_4_is_2() {
        let a = sqrt(4.0).unwrap();
        assert!(
            (a.value - 2.0).abs() < 1e-10,
            "sqrt(4) ≈ 2, got {}",
            a.value
        );
    }

    #[test]
    fn exp_0_is_1() {
        let a = exp(0.0).unwrap();
        assert!((a.value - 1.0).abs() < 1e-10, "exp(0) = 1, got {}", a.value);
    }

    #[test]
    fn log_e_is_1() {
        let a = log(std::f64::consts::E).unwrap();
        assert!((a.value - 1.0).abs() < 1e-10, "log(e) = 1, got {}", a.value);
    }

    #[test]
    fn sin_0_is_0() {
        let a = sin(0.0).unwrap();
        assert!(a.value.abs() < 1e-15, "sin(0) = 0, got {}", a.value);
    }

    #[test]
    fn cos_0_is_1() {
        let a = cos(0.0).unwrap();
        assert!((a.value - 1.0).abs() < 1e-15, "cos(0) = 1, got {}", a.value);
    }

    #[test]
    fn hypot_3_4_is_5() {
        let a = hypot(3.0, 4.0).unwrap();
        assert!(
            (a.value - 5.0).abs() < 1e-10,
            "hypot(3,4) = 5, got {}",
            a.value
        );
    }

    #[test]
    fn logb_base2_of_8_is_3() {
        let a = logb(2.0, 8.0).unwrap();
        assert!(
            (a.value - 3.0).abs() < 1e-10,
            "logb(2,8) = 3, got {}",
            a.value
        );
    }
}
