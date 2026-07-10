//! MAP-I bundle **capacity bound** — the `Proven` tag via *checked instantiation* (M-131; RFC-0003
//! §5; ADR-010; SC-2; KC-1).
//!
//! The cited theorem (Clarkson-Ubaru-Yang 2023 Thm 6; Thomas-Dasgupta-Rosing 2021) gives, for
//! bundling `m` items, a sufficient dimension
//!
//! ```text
//!   requiredDim(m, δ) = ⌈ (2/μ²) · ln(m/δ) ⌉ ,   μ = 0.1  (the illustrative margin)
//! ```
//!
//! above which a bundle decodes with failure probability `≤ δ`. **The formula is cited, not
//! re-proven** (the M-001 Liquid-Haskell probe axiomatizes the theorem and Z3 discharges only the
//! arithmetic instantiation `d ≥ requiredDim`). Here we replay exactly that checked instantiation in
//! Rust: a `Proven` [`Bound`] is issued **iff** the concrete `d ≥ requiredDim(m, δ)` holds; otherwise
//! the side-condition fails and no `Proven` bound is available (honest downgrade, VR-5 — never stamp
//! `Proven` without the checked basis, M-I2).

use mycelium_core::{Bound, BoundBasis, BoundKind};

/// The illustrative margin `μ` the M-001 probe fixes (so `2/μ² = 200`).
pub const MARGIN_MU: f64 = 0.1;

/// The cited theorem `requiredDim(m, δ) = ⌈(2/μ²)·ln(m/δ)⌉` (RFC-0003 §5). Panics never; for
/// `items = 0` or non-finite `δ` outside `(0,1]` returns `u64::MAX` (no dimension certifies it).
#[must_use]
pub fn required_dim(items: u64, delta: f64, mu: f64) -> u64 {
    if items == 0
        || !delta.is_finite()
        || delta <= 0.0
        || delta > 1.0
        || !mu.is_finite()
        || mu <= 0.0
    {
        return u64::MAX;
    }
    let val = (2.0 / (mu * mu)) * (items as f64 / delta).ln();
    if !val.is_finite() || val < 0.0 {
        return 0;
    }
    val.ceil() as u64
}

/// The citation string for the MAP-I bundle capacity theorem (T0.2).
pub const CAPACITY_CITATION: &str = "Clarkson-Ubaru-Yang 2023 (Thm 6); Thomas-Dasgupta-Rosing 2021";

/// Issue a **`Proven`** capacity [`Bound`] for bundling `items` into `dim`, targeting failure
/// probability `delta` — **iff** the checked side-condition `dim ≥ requiredDim(items, delta)` holds
/// (the M-001 checked-instantiation pattern). Returns `None` when the dimension is insufficient: the
/// theorem does not apply, so a `Proven` tag would be dishonest (M-I2/VR-5).
#[must_use]
pub fn proven_capacity_bound(items: u64, dim: u64, delta: f64) -> Option<Bound> {
    let required = required_dim(items, delta, MARGIN_MU);
    if dim < required {
        return None; // side-condition fails → no Proven basis
    }
    Some(Bound {
        kind: BoundKind::Capacity { items, dim },
        basis: BoundBasis::ProvenThm {
            // Record the assumed margin μ and the checked side-condition in the basis, so EXPLAIN /
            // the serialized bound expose exactly what the Proven tag rests on (A3-03/H6).
            citation: format!(
                "{CAPACITY_CITATION}; μ={MARGIN_MU} (illustrative margin); checked d ≥ requiredDim"
            ),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_dim_matches_the_m001_probe_table() {
        // The four (m, δ) settings the M-001 LH probe checks (proofs/lh-bundle/src/Bundle.hs).
        assert_eq!(required_dim(3, 1e-2, MARGIN_MU), 1141);
        assert_eq!(required_dim(10, 1e-3, MARGIN_MU), 1843);
        assert_eq!(required_dim(50, 1e-3, MARGIN_MU), 2164);
        assert_eq!(required_dim(100, 1e-4, MARGIN_MU), 2764);
    }

    #[test]
    fn proven_bound_only_when_dimension_suffices() {
        // d = 10000 ≥ 1141 → Proven (the probe1 instantiation).
        let b = proven_capacity_bound(3, 10_000, 1e-2).expect("sufficient");
        assert!(matches!(b.basis, BoundBasis::ProvenThm { .. }));
        assert!(matches!(
            b.kind,
            BoundKind::Capacity {
                items: 3,
                dim: 10_000
            }
        ));
        // d = 1000 < 1141 → no Proven bound (side-condition fails).
        assert_eq!(proven_capacity_bound(3, 1000, 1e-2), None);
    }

    #[test]
    fn degenerate_inputs_never_certify() {
        assert_eq!(required_dim(0, 1e-2, MARGIN_MU), u64::MAX);
        assert_eq!(required_dim(3, 0.0, MARGIN_MU), u64::MAX);
        assert_eq!(required_dim(3, 2.0, MARGIN_MU), u64::MAX);
        assert_eq!(proven_capacity_bound(3, 10_000, 0.0), None);
    }
}
