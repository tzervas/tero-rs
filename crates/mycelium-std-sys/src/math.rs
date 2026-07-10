//! \[Declared\] Transcendental math floor. Thin wrappers over Rust `f64` intrinsics (libm).
//!
//! All functions in this module are tagged `Declared` — the platform libm floor is unaudited;
//! no error-bound theorem backs these calls (VR-5: downgrade to stay honest, never upgrade
//! without a checked basis).
//!
//! RFC-0016 §9: once `std-sys` is the sole entry point for transcendental calls, the pure
//! `std-math` crate earns a `wild`-free badge. Wiring is deferred to a future wave; this
//! module establishes the interface.
//!
//! # Note on `unsafe`
//!
//! These functions are **not** `unsafe` — Rust's `f64` methods wrap the platform libm call
//! internally. The `Declared` tag is about *precision/auditedness*, not memory safety.

/// \[Declared\] `sin(x)`. Delegates to Rust `f64::sin` (platform libm). Precision unaudited (M-541).
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// \[Declared\] `cos(x)`. Delegates to Rust `f64::cos` (platform libm). Precision unaudited (M-541).
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// \[Declared\] `tan(x)`. Delegates to Rust `f64::tan` (platform libm). Precision unaudited (M-541).
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// \[Declared\] `asin(x)`. Delegates to Rust `f64::asin` (platform libm). Precision unaudited (M-541).
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// \[Declared\] `acos(x)`. Delegates to Rust `f64::acos` (platform libm). Precision unaudited (M-541).
pub fn acos(x: f64) -> f64 {
    x.acos()
}

/// \[Declared\] `atan(x)`. Delegates to Rust `f64::atan` (platform libm). Precision unaudited (M-541).
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// \[Declared\] `atan2(y, x)`. Delegates to Rust `f64::atan2` (platform libm). Precision unaudited (M-541).
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// \[Declared\] `exp(x)`. Delegates to Rust `f64::exp` (platform libm). Precision unaudited (M-541).
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// \[Declared\] `exp2(x)`. Delegates to Rust `f64::exp2` (platform libm). Precision unaudited (M-541).
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// \[Declared\] `ln(x)`. Delegates to Rust `f64::ln` (platform libm). Precision unaudited (M-541).
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// \[Declared\] `log2(x)`. Delegates to Rust `f64::log2` (platform libm). Precision unaudited (M-541).
pub fn log2(x: f64) -> f64 {
    x.log2()
}

/// \[Declared\] `log10(x)`. Delegates to Rust `f64::log10` (platform libm). Precision unaudited (M-541).
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// \[Declared\] `sqrt(x)`. Delegates to Rust `f64::sqrt` (platform libm). Precision unaudited (M-541).
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// \[Declared\] `cbrt(x)`. Delegates to Rust `f64::cbrt` (platform libm). Precision unaudited (M-541).
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// \[Empirical\] sin²(x) + cos²(x) ≈ 1.0 for a range of x values.
    /// Tagged `Empirical` — this is a measured property, not a proven theorem (VR-5).
    #[test]
    fn pythagorean_identity() {
        let xs: &[f64] = &[
            0.0,
            0.1,
            0.5,
            1.0,
            1.5,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
            std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            -1.0,
            -std::f64::consts::PI / 3.0,
            100.0,
            -100.0,
        ];
        for &x in xs {
            let s = sin(x);
            let c = cos(x);
            let identity = s * s + c * c;
            assert!(
                (identity - 1.0).abs() < 1e-10,
                "sin²({x}) + cos²({x}) = {identity}, expected 1.0 ± 1e-10"
            );
        }
    }

    #[test]
    fn tan_consistency() {
        // tan(x) = sin(x) / cos(x) for non-singular x
        let xs: &[f64] = &[0.1, 0.5, 1.0, -1.0, 2.5];
        for &x in xs {
            let t = tan(x);
            let expected = sin(x) / cos(x);
            assert!(
                (t - expected).abs() < 1e-12,
                "tan({x}) = {t}, sin/cos = {expected}"
            );
        }
    }

    #[test]
    fn exp_ln_roundtrip() {
        let xs: &[f64] = &[0.1, 1.0, 2.0, 10.0, 0.5];
        for &x in xs {
            let result = ln(exp(x));
            assert!(
                (result - x).abs() < 1e-12,
                "ln(exp({x})) = {result}, expected {x}"
            );
        }
    }

    #[test]
    fn sqrt_squared() {
        let xs: &[f64] = &[0.0, 1.0, 4.0, 9.0, 2.0, 0.25];
        for &x in xs {
            let s = sqrt(x);
            assert!(
                (s * s - x).abs() < 1e-12,
                "sqrt({x})² = {}, expected {x}",
                s * s
            );
        }
    }
}
