//! Bounds and their basis (RFC-0001 §4.3 r2; ADR-010; ADR-011; `bound.schema.json`).
//!
//! Per **ADR-011**, `basis` is a required companion of *every* [`Bound`], not just capacity bounds:
//! the guarantee strength derives from the basis for all bound kinds.
//!
//! The `serde` wire form is exactly `bound.schema.json` (M-104): a flat object tagged on `kind`
//! (`ErrorBound|ProbabilityBound|CrosstalkBound|CapacityBound`) carrying the payload fields and a
//! sibling `basis` (itself tagged on `kind`: `ProvenThm|EmpiricalFit|UserDeclared`). A `null`/absent
//! `tail` on a `CrosstalkBound` is simply omitted.

use serde::{Deserialize, Serialize};

use crate::GuaranteeStrength;

/// How a bound was obtained — this determines the honest [`crate::GuaranteeStrength`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum BoundBasis {
    /// A cited theorem whose side-conditions are checked (e.g. "Clarkson-Ubaru-Yang 2023").
    ProvenThm {
        /// The citation.
        citation: String,
    },
    /// An empirical fit over `trials` (e.g. method "Frady-Sommer Gaussian").
    EmpiricalFit {
        /// Number of trials.
        trials: u64,
        /// Fitting method.
        method: String,
    },
    /// A user assertion, not yet validated. Tooling must surface a "declared, unverified" marker.
    UserDeclared,
}

impl BoundBasis {
    /// The honest [`GuaranteeStrength`] this basis implies (M-I2/M-I3/M-I4): the basis *is* the
    /// evidence class. This is the core-local equivalent of `mycelium_numerics::basis_strength`,
    /// available to kernel code (e.g. [`crate::recon`]) that must rank a basis against the lattice
    /// without depending on a downstream crate. Expressing rules via the lattice rank (rather than a
    /// `matches!` on a specific variant) keeps them correct if a new, stronger basis is ever added.
    #[must_use]
    pub fn strength(&self) -> GuaranteeStrength {
        match self {
            BoundBasis::ProvenThm { .. } => GuaranteeStrength::Proven,
            BoundBasis::EmpiricalFit { .. } => GuaranteeStrength::Empirical,
            BoundBasis::UserDeclared => GuaranteeStrength::Declared,
        }
    }
}

/// Norm in which an [`BoundKind::Error`] `eps` is expressed (extensible registry; RFC-0001 §4.3 r2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormKind {
    /// ℓ¹.
    L1,
    /// ℓ².
    L2,
    /// ℓ∞.
    Linf,
    /// Relative error.
    Rel,
}

/// The bound payload, per kind (RFC-0001 §4.3). The `serde` tag values match `bound.schema.json`'s
/// `*Bound` discriminants (Rust drops the redundant `Bound` suffix on the variant names).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum BoundKind {
    /// ε-magnitude bound (composes via ADR-010's affine-arithmetic kernel).
    #[serde(rename = "ErrorBound")]
    Error {
        /// Error magnitude (`>= 0`).
        eps: f64,
        /// Norm in which `eps` is measured.
        norm: NormKind,
    },
    /// Failure-probability bound (composes via the union-bound kernel).
    #[serde(rename = "ProbabilityBound")]
    Probability {
        /// Failure probability in `[0, 1]`.
        delta: f64,
    },
    /// Expected crosstalk with an optional tail.
    #[serde(rename = "CrosstalkBound")]
    Crosstalk {
        /// Expected crosstalk (`>= 0`).
        expected: f64,
        /// Optional tail bound (omitted from the wire form when absent).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tail: Option<f64>,
    },
    /// VSA superposition capacity (`items` into `dim`).
    #[serde(rename = "CapacityBound")]
    Capacity {
        /// Number of superposed items (`>= 1`).
        items: u64,
        /// Hypervector dimension (`>= 1`).
        dim: u64,
    },
}

/// A sound bound plus the basis by which it was obtained (ADR-011: `basis` is universal). Serializes
/// as a single flat object: the `kind`-tagged payload with a sibling `basis` (`bound.schema.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bound {
    /// The kind-specific payload.
    #[serde(flatten)]
    pub kind: BoundKind,
    /// How the bound was obtained.
    pub basis: BoundBasis,
}

impl Bound {
    /// Well-formedness per `bound.schema.json`: the payload ranges (magnitudes finite and in range)
    /// **and** the basis constraints. Independent of the guarantee↔basis coupling, which
    /// [`crate::meta::Meta`] enforces; this is re-run on every deserialize, so the schema is a
    /// contract the wire form cannot evade.
    #[must_use]
    pub fn well_formed(&self) -> bool {
        let payload_ok = match self.kind {
            // Magnitudes must be finite, not just `>= 0` — an infinite ε/crosstalk is a vacuous
            // bound that could otherwise ride as `Proven`/`Empirical` (A1-02/B2-03).
            BoundKind::Error { eps, .. } => eps.is_finite() && eps >= 0.0,
            BoundKind::Probability { delta } => (0.0..=1.0).contains(&delta),
            BoundKind::Crosstalk { expected, tail } => {
                expected.is_finite()
                    && expected >= 0.0
                    && tail.is_none_or(|t| t.is_finite() && t >= 0.0)
            }
            BoundKind::Capacity { items, dim } => items >= 1 && dim >= 1,
        };
        payload_ok && self.basis_well_formed()
    }

    /// Basis constraints from `bound.schema.json` (A6-02/B2-03): a cited theorem names its citation;
    /// an empirical fit rests on **at least one** trial with a named method — an `Empirical` tag must
    /// never be evidence-free (`trials: 0`); a user declaration carries nothing.
    #[must_use]
    fn basis_well_formed(&self) -> bool {
        match &self.basis {
            BoundBasis::ProvenThm { citation } => !citation.trim().is_empty(),
            BoundBasis::EmpiricalFit { trials, method } => {
                *trials >= 1 && !method.trim().is_empty()
            }
            BoundBasis::UserDeclared => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proven() -> BoundBasis {
        BoundBasis::ProvenThm {
            citation: "thm".to_owned(),
        }
    }

    #[test]
    fn well_formed_rejects_non_finite_magnitudes() {
        // A1-02: infinite ε / crosstalk are vacuous bounds and must be rejected.
        assert!(!Bound {
            kind: BoundKind::Error {
                eps: f64::INFINITY,
                norm: NormKind::Linf
            },
            basis: proven(),
        }
        .well_formed());
        assert!(!Bound {
            kind: BoundKind::Crosstalk {
                expected: f64::INFINITY,
                tail: None
            },
            basis: proven(),
        }
        .well_formed());
        assert!(Bound {
            kind: BoundKind::Error {
                eps: 0.5,
                norm: NormKind::Linf
            },
            basis: proven(),
        }
        .well_formed());
    }

    // Mutant-witnesses (bound.rs:128:58, 130:62): `&&` → `||` in Bound::well_formed.
    // - bound.rs:128: `tail.is_none_or(|t| t.is_finite() && t >= 0.0)` → with `||`, infinite
    //   positive tail would pass (is_finite()=false but t >= 0.0=true).
    // - bound.rs:130: `items >= 1 && dim >= 1` → with `||`, items=0 or dim=0 alone passes.
    #[test]
    fn well_formed_rejects_zero_capacity_and_infinite_tail() {
        let proven = proven();

        // Capacity bound with items=0 must be rejected (items must be >= 1).
        // Kills: `items >= 1 && dim >= 1` → `items >= 1 || dim >= 1` (items=0, dim=1 → || passes)
        assert!(!Bound {
            kind: BoundKind::Capacity { items: 0, dim: 1 },
            basis: proven.clone(),
        }
        .well_formed());

        // Capacity bound with dim=0 must be rejected.
        // Kills: `&&` → `||` (items=1, dim=0 → || passes with items=1)
        assert!(!Bound {
            kind: BoundKind::Capacity { items: 1, dim: 0 },
            basis: proven.clone(),
        }
        .well_formed());

        // Capacity bound with both >= 1 is valid.
        assert!(Bound {
            kind: BoundKind::Capacity { items: 2, dim: 4 },
            basis: proven.clone(),
        }
        .well_formed());

        // Crosstalk bound with infinite positive tail must be rejected.
        // Kills: `t.is_finite() && t >= 0.0` → `t.is_finite() || t >= 0.0`
        // (t=+inf: is_finite()=false, t >= 0.0=true → || passes, && correctly rejects).
        assert!(!Bound {
            kind: BoundKind::Crosstalk {
                expected: 0.5,
                tail: Some(f64::INFINITY), // infinite tail — not a finite bound
            },
            basis: proven.clone(),
        }
        .well_formed());

        // Crosstalk bound with finite positive tail is valid.
        assert!(Bound {
            kind: BoundKind::Crosstalk {
                expected: 0.5,
                tail: Some(0.1),
            },
            basis: proven,
        }
        .well_formed());
    }

    #[test]
    fn well_formed_rejects_evidence_free_basis() {
        // A6-02/B2-03: an Empirical tag backed by zero trials (or an empty method/citation) is not
        // honest evidence — well_formed (hence deserialize) must reject it.
        let zero_trials = Bound {
            kind: BoundKind::Error {
                eps: 0.5,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::EmpiricalFit {
                trials: 0,
                method: "frady".to_owned(),
            },
        };
        assert!(!zero_trials.well_formed());
        let empty_method = Bound {
            kind: BoundKind::Error {
                eps: 0.5,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::EmpiricalFit {
                trials: 10,
                method: String::new(),
            },
        };
        assert!(!empty_method.well_formed());
        let empty_citation = Bound {
            kind: BoundKind::Error {
                eps: 0.5,
                norm: NormKind::Linf,
            },
            basis: BoundBasis::ProvenThm {
                citation: "  ".to_owned(),
            },
        };
        assert!(!empty_citation.well_formed());
    }
}
