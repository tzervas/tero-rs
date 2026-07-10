//! The **shared `{ε, δ, strength}` certificate**, the **tier-i Rust checker**, the one sanctioned
//! **cross-kernel inference**, and the **bound-composition entry** the interpreter calls (M-203/M-204;
//! ADR-010 §3/§4 + "Trusted base"; RFC-0002 §2).
//!
//! Both kernels reduce to one record — [`Certificate`] `{eps, delta, strength}` — with `strength`
//! composing by `meet` (ADR-010 §3). The **tier-i checker** (ADR-010 "Trusted base") *re-derives* a
//! composition from its recorded inputs and **rejects a claim tighter than the re-derivation**
//! ([`check_error_claim`]/[`check_union_claim`]) — incompleteness is an explicit rejection, never a
//! silent pass (RFC-0002 §2). The single legal cross-kernel rule is accuracy→probability
//! ([`accuracy_to_probability`]); no other mixing exists.

use serde::{Deserialize, Serialize};

use mycelium_core::{Bound, BoundBasis, BoundKind, GuaranteeStrength, NormKind};

use crate::error::ErrorBound;
use crate::prob::ProbBound;

/// Citation for an ε bound obtained by composing proven inputs through affine arithmetic — the
/// composition itself is sound by ADR-010 §1 (Daisy/FloVer), so a `Proven⊕Proven` composition stays
/// `Proven` under this citation (its side-condition — proven operands — is checked at the call site).
const AFFINE_CITATION: &str = "ADR-010 §1 affine-arithmetic ε-composition (Daisy/Rosa; FloVer)";
/// Method tag for an `Empirical` composed ε bound (the weakest contributing basis was a fit).
const COMPOSED_METHOD: &str = "composed (ADR-010 §1 affine ε)";
/// **Relative** slack when comparing a claimed bound to the re-derivation, scaled to the re-derived
/// magnitude (a few ULPs). The re-derivation is itself outward-rounded (A2-01), so this only absorbs
/// a producer's last-ULP rounding — and, being relative, it stays meaningful for tiny bounds where
/// the previous absolute `1e-12` made the check vacuous (A2-02): a claim of `eps = 0` against a
/// re-derived `5e-13` is now correctly rejected.
const CHECK_REL_TOL: f64 = 8.0 * f64::EPSILON;

/// The slack permitted for a claim against a sound re-derivation `recomputed`: a few ULPs of the
/// re-derived magnitude (zero when the re-derivation is exactly `0`, so an exact claim must match).
fn check_slack(recomputed: f64) -> f64 {
    recomputed.abs() * CHECK_REL_TOL
}

/// The error-kernel operation a composition records — re-evaluated by the tier-i checker and used by
/// the interpreter's [`compose_error_bound`] (M-204). Magnitudes for the nonlinear `Mul` are the
/// central operand magnitudes `|x₀|, |y₀|`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorOp {
    /// `x + y` (any arity ≥ 1; magnitudes sum).
    Add,
    /// `x − y` (same magnitude composition as `Add`).
    Sub,
    /// `−x` (unary).
    Neg,
    /// `c·x` (unary; exact scale).
    Scale(f64),
    /// `x·y` (binary; first-order product propagation about `|x₀|, |y₀|`).
    Mul {
        /// `|x₀|` — central magnitude of the first operand.
        x0_mag: f64,
        /// `|y₀|` — central magnitude of the second operand.
        y0_mag: f64,
    },
}

/// Re-derive the composed [`ErrorBound`] of `inputs` under `op` from the kernel — the checker's and
/// the interpreter's single source of composition truth. `None` if the arity is wrong for `op` or the
/// input norms disagree (never a silent norm coercion).
#[must_use]
pub fn recompute_error(inputs: &[ErrorBound], op: ErrorOp) -> Option<ErrorBound> {
    match op {
        ErrorOp::Add | ErrorOp::Sub => {
            let (first, rest) = inputs.split_first()?;
            let mut acc = *first;
            for next in rest {
                acc = acc.add(next)?;
            }
            Some(acc)
        }
        ErrorOp::Neg => match inputs {
            [x] => Some(x.neg()),
            _ => None,
        },
        ErrorOp::Scale(c) => match inputs {
            [x] => Some(x.scale(c)),
            _ => None,
        },
        ErrorOp::Mul { x0_mag, y0_mag } => match inputs {
            [x, y] => x.mul(y, x0_mag, y0_mag),
            _ => None,
        },
    }
}

/// The verdict of a tier-i re-validation (ADR-010 "Trusted base").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckOutcome {
    /// The claimed bound is `≥` the re-derivation — sound (possibly loose, which is allowed).
    Valid,
    /// The claim is **tighter** than the sound re-derivation — rejected (not a silent pass).
    Rejected {
        /// The bound the kernel re-derives.
        recomputed: f64,
        /// The (too-tight) bound that was claimed.
        claimed: f64,
    },
    /// The claim could not be re-derived (bad arity, norm mismatch) — rejected as ill-formed.
    Malformed,
}

/// Re-validate a claimed ε bound for `op` over `inputs`: **Valid** iff the claim is `≥` the
/// re-derivation (sound), **Rejected** if tighter, **Malformed** if it cannot be re-derived. The
/// checker trusts only the kernel arithmetic, not the claimant (ADR-010 tier-i; RFC-0002 §2).
#[must_use]
pub fn check_error_claim(inputs: &[ErrorBound], op: ErrorOp, claimed: ErrorBound) -> CheckOutcome {
    let Some(recomputed) = recompute_error(inputs, op) else {
        return CheckOutcome::Malformed;
    };
    if recomputed.norm != claimed.norm {
        return CheckOutcome::Malformed;
    }
    if claimed.eps + check_slack(recomputed.eps) >= recomputed.eps {
        CheckOutcome::Valid
    } else {
        CheckOutcome::Rejected {
            recomputed: recomputed.eps,
            claimed: claimed.eps,
        }
    }
}

/// Re-validate a claimed δ against the **union bound** of `inputs`: **Valid** iff the claim is `≥`
/// `min(1, Σδ)`, else **Rejected** (ADR-010 §2; RFC-0002 §2).
#[must_use]
pub fn check_union_claim(inputs: &[ProbBound], claimed: ProbBound) -> CheckOutcome {
    let recomputed = ProbBound::union(inputs);
    if claimed.delta + check_slack(recomputed.delta) >= recomputed.delta {
        CheckOutcome::Valid
    } else {
        CheckOutcome::Rejected {
            recomputed: recomputed.delta,
            claimed: claimed.delta,
        }
    }
}

/// The single sanctioned **cross-kernel inference** (ADR-010 §4): an [`ErrorBound`] feeds a
/// [`ProbBound`]. The failure event is "the output deviates from the ideal by more than tolerance
/// `tau`". Given an accuracy bound that itself holds with confidence `1 − acc_delta`, failure
/// probability is `acc_delta` when `eps ≤ tau` (within tolerance whenever the bound holds) and `1.0`
/// otherwise (the bound permits a violation — the honest worst case). `None` if `tau < 0` or
/// `acc_delta ∉ [0, 1]`.
#[must_use]
pub fn accuracy_to_probability(acc: ErrorBound, tau: f64, acc_delta: f64) -> Option<ProbBound> {
    if !(tau.is_finite() && tau >= 0.0) {
        return None;
    }
    let delta = if acc.eps <= tau { acc_delta } else { 1.0 };
    ProbBound::new(delta)
}

/// The shared certificate both kernels reduce to (ADR-010 §3): an ε side, a δ side, and a `strength`
/// tag that composes by `meet`. Serializes as `{ "eps", "delta", "strength" }`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Certificate {
    pub(crate) eps: f64,
    pub(crate) delta: f64,
    pub(crate) strength: GuaranteeStrength,
}

/// Validating `Deserialize`: wire certificates route through [`Certificate::new`], so an
/// out-of-range `eps`/`delta` is rejected on the way in, never silently trusted (A2-05). Fields are
/// private; the only constructors are the validating `new`/`from_*`/`exact`.
impl<'de> Deserialize<'de> for Certificate {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            eps: f64,
            delta: f64,
            strength: GuaranteeStrength,
        }
        let w = Wire::deserialize(deserializer)?;
        Certificate::new(w.eps, w.delta, w.strength)
            .ok_or_else(|| serde::de::Error::custom("certificate eps/delta out of range"))
    }
}

impl Certificate {
    /// The ε-magnitude side (`0` if no error component).
    #[must_use]
    pub fn eps(&self) -> f64 {
        self.eps
    }
    /// The δ failure-probability side (`0` if no probabilistic component).
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.delta
    }
    /// The honest guarantee strength (`meet` of contributors).
    #[must_use]
    pub fn strength(&self) -> GuaranteeStrength {
        self.strength
    }

    /// A well-formed certificate, or `None` if `eps`/`delta` are out of range (never silent).
    #[must_use]
    pub fn new(eps: f64, delta: f64, strength: GuaranteeStrength) -> Option<Self> {
        let ok = eps.is_finite() && eps >= 0.0 && delta.is_finite() && (0.0..=1.0).contains(&delta);
        ok.then_some(Certificate {
            eps,
            delta,
            strength,
        })
    }

    /// The exact certificate `{0, 0, Exact}`.
    #[must_use]
    pub const fn exact() -> Self {
        Certificate {
            eps: 0.0,
            delta: 0.0,
            strength: GuaranteeStrength::Exact,
        }
    }

    /// Lift an [`ErrorBound`] to a certificate at the given `strength` (δ side `0`).
    #[must_use]
    pub fn from_error(error: ErrorBound, strength: GuaranteeStrength) -> Self {
        Certificate {
            eps: error.eps,
            delta: 0.0,
            strength,
        }
    }

    /// Lift a [`ProbBound`] to a certificate at the given `strength` (ε side `0`).
    #[must_use]
    pub fn from_prob(prob: ProbBound, strength: GuaranteeStrength) -> Self {
        Certificate {
            eps: 0.0,
            delta: prob.delta,
            strength,
        }
    }
}

/// A bound composed by the kernel, with the honest `strength` its inputs' bases justify — the
/// interpreter (M-204) sets `Meta.guarantee = strength` and `Meta.bound = Some(bound)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposedBound {
    /// The composed bound (kind `Error`, with a basis matching `strength`).
    pub bound: Bound,
    /// `meet` of the input strengths — never stronger than the weakest input (VR-5).
    pub strength: GuaranteeStrength,
}

/// The strength implied by a bound's basis (M-I2/M-I3/M-I4): the basis *is* the evidence class.
/// Public because certificate consumers (the M-210 translation-validation checker) must derive the
/// honest strength **from** the evidence, never accept an asserted one (RFC-0002 §3; VR-5).
#[must_use]
pub fn basis_strength(basis: &BoundBasis) -> GuaranteeStrength {
    match basis {
        BoundBasis::ProvenThm { .. } => GuaranteeStrength::Proven,
        BoundBasis::EmpiricalFit { .. } => GuaranteeStrength::Empirical,
        BoundBasis::UserDeclared => GuaranteeStrength::Declared,
    }
}

/// The fewest trials among the empirical inputs — the weakest empirical evidence carries forward.
fn min_empirical_trials(bases: &[&BoundBasis]) -> u64 {
    bases
        .iter()
        .filter_map(|b| match b {
            BoundBasis::EmpiricalFit { trials, .. } => Some(*trials),
            _ => None,
        })
        .min()
        .unwrap_or(0)
}

/// The honest basis for a composed bound at the meet `strength`. A `Proven` composition cites the
/// affine-arithmetic soundness (its side-condition — proven operands — holds exactly when the meet is
/// `Proven`); `Empirical` carries the weakest trial count; `Declared` stays declared. Never returns a
/// basis stronger than `strength` (VR-5).
fn composed_basis(strength: GuaranteeStrength, bases: &[&BoundBasis]) -> Option<BoundBasis> {
    match strength {
        GuaranteeStrength::Exact => None,
        GuaranteeStrength::Proven => {
            // Preserve the contributing theorems' provenance rather than collapsing to the bare
            // affine citation (A2-09): cite affine-arithmetic composition *over* the input theorems.
            let inputs: Vec<&str> = bases
                .iter()
                .filter_map(|b| match b {
                    BoundBasis::ProvenThm { citation } => Some(citation.as_str()),
                    _ => None,
                })
                .collect();
            let citation = if inputs.is_empty() {
                AFFINE_CITATION.to_owned()
            } else {
                format!("{AFFINE_CITATION} over [{}]", inputs.join("; "))
            };
            Some(BoundBasis::ProvenThm { citation })
        }
        GuaranteeStrength::Empirical => Some(BoundBasis::EmpiricalFit {
            trials: min_empirical_trials(bases),
            method: COMPOSED_METHOD.to_owned(),
        }),
        GuaranteeStrength::Declared => Some(BoundBasis::UserDeclared),
    }
}

/// Extract a scalar [`ErrorBound`] from a `BoundKind::Error`, else `None`.
fn bound_as_error(bound: &Bound) -> Option<ErrorBound> {
    match bound.kind {
        BoundKind::Error { eps, norm } => ErrorBound::new(eps, norm),
        _ => None,
    }
}

/// Compose the **`Error` bounds of approximate inputs** under `op` into a result bound whose
/// `strength` is the `meet` of the inputs' strengths and whose basis matches that strength (M-204;
/// RFC-0001 §4.7; ADR-010 §1). Returns `None` — so the caller refuses, never fabricates — when any
/// input is not an `Error` bound, norms disagree, the arity is wrong, or `inputs` is empty. The honest
/// upgrade over the Phase-1 refusal: an op over approximate inputs now carries a *checked* composed
/// bound instead of being rejected outright.
#[must_use]
pub fn compose_error_bound(inputs: &[&Bound], op: ErrorOp) -> Option<ComposedBound> {
    if inputs.is_empty() {
        return None;
    }
    let errors: Option<Vec<ErrorBound>> = inputs.iter().map(|b| bound_as_error(b)).collect();
    let errors = errors?;
    let composed = recompute_error(&errors, op)?;
    // Re-validate the composed magnitude: a composition that overflows to non-finite is refused, not
    // emitted as a fabricated bound (A2-04).
    let composed = ErrorBound::new(composed.eps, composed.norm)?;

    let bases: Vec<&BoundBasis> = inputs.iter().map(|b| &b.basis).collect();
    let strength = bases
        .iter()
        .map(|b| basis_strength(b))
        .fold(GuaranteeStrength::TOP, GuaranteeStrength::meet);
    let basis = composed_basis(strength, &bases)?;

    Some(ComposedBound {
        bound: Bound {
            kind: BoundKind::Error {
                eps: composed.eps,
                norm: composed.norm,
            },
            basis,
        },
        strength,
    })
}

/// The norm of a `BoundKind::Error`, for callers assembling [`ErrorOp`]s. `None` for non-`Error`
/// kinds.
#[must_use]
pub fn error_norm(bound: &Bound) -> Option<NormKind> {
    match bound.kind {
        BoundKind::Error { norm, .. } => Some(norm),
        _ => None,
    }
}
