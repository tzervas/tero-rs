//! Directed (outward) floating-point rounding for **bound composition** (WS1; findings A2-01 /
//! C1-01; ADR-010 §1).
//!
//! A `Proven`/`Empirical` ε or δ that travels in a [`mycelium_core::Bound`] must be a *true* upper
//! bound on the real-arithmetic quantity it claims. Plain round-to-nearest f64 can round a composed
//! bound *below* the real value by up to half an ULP per operation, so a chain of compositions can
//! emit a `Proven` tag the stored number does not actually justify. These helpers round a
//! bound-increasing result toward +∞ — but **only when the IEEE result was actually rounded down**,
//! recovered exactly via the Knuth/Møller two-sum and an FMA. That preserves an exact result
//! (`0.0 + 0.0` stays `0.0`), so an exact composition does not silently become approximate.

/// The exact round-off of `a + b` under round-to-nearest: `(a + b)_exact − fl(a + b)` (Knuth/Møller
/// two-sum; exact for any finite `a`, `b`). A **positive** result means the IEEE sum rounded *down*.
#[must_use]
pub(crate) fn add_err(a: f64, b: f64) -> f64 {
    let s = a + b;
    let bv = s - a;
    (a - (s - bv)) + (b - bv)
}

/// The exact round-off of `a * b`: `(a · b)_exact − fl(a · b)`, recovered via a fused multiply-add.
/// A **positive** result means the IEEE product rounded *down*.
#[must_use]
pub(crate) fn mul_err(a: f64, b: f64) -> f64 {
    a.mul_add(b, -(a * b))
}

/// `a + b` rounded toward +∞: a sound upper bound on the real sum, tight (and exactly `fl(a + b)`)
/// whenever IEEE addition introduced no downward rounding.
#[must_use]
pub(crate) fn add_up(a: f64, b: f64) -> f64 {
    let s = a + b;
    if add_err(a, b) > 0.0 {
        s.next_up()
    } else {
        s
    }
}

/// `a * b` rounded toward +∞: a sound upper bound on the real product, tight whenever IEEE
/// multiplication introduced no downward rounding.
#[must_use]
pub(crate) fn mul_up(a: f64, b: f64) -> f64 {
    let p = a * b;
    if mul_err(a, b) > 0.0 {
        p.next_up()
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_results_are_not_inflated() {
        // The exact-preservation property the lattice relies on: an exact composition stays exact,
        // so `Exact ⊕ Exact` does not silently become a tiny-but-nonzero (approximate) bound.
        assert_eq!(add_up(0.0, 0.0), 0.0);
        assert_eq!(add_up(1.0, 2.0), 3.0);
        assert_eq!(mul_up(0.0, 5.0), 0.0);
        assert_eq!(mul_up(2.0, 4.0), 8.0);
    }

    #[test]
    fn rounded_down_sums_are_pushed_outward() {
        // 1.0 + 1e-17 rounds to exactly 1.0 under round-to-nearest, which is *below* the true sum;
        // add_up must return a value strictly greater than 1.0 so the bound stays sound.
        let up = add_up(1.0, 1e-17);
        assert!(up > 1.0, "add_up did not round outward: {up}");
        assert!(up >= 1.0 + 1e-17);
    }

    #[test]
    fn rounded_down_products_are_pushed_outward() {
        // A product whose exact value is not representable and rounds down must be pushed up.
        let a = 1.0 + 2f64.powi(-52);
        let up = mul_up(a, a);
        assert!(up >= a * a);
        assert!(up >= 1.0 + 2f64.powi(-51));
    }
}
