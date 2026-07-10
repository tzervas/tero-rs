//! `mycelium-vsa` — the **VSA submodule**: the `VsaModel` trait and its first model, **MAP-I**
//! (M-130; RFC-0003 §3–§4; ADR-008; T2.6).
//!
//! This is a **dependency-gated submodule** (ADR-008): it depends on `mycelium-core` but the kernel
//! does *not* depend on it. The kernel already type-checks hypervector *mentions* — `Repr::Vsa` and
//! `Payload::Hypervector` live in core — so programs that name VSA values stay well-typed without
//! pulling in this algebra (KC-3: the kernel stays small and auditable; VSA is opt-in).
//!
//! # Honesty of per-operation guarantees (RFC-0003 §4, normative matrix)
//! Each model declares, per operation, an [intrinsic guarantee](VsaModel::intrinsic_guarantee). For
//! **MAP-I**: `bind`/`unbind` are **self-inverse and `Exact`** (algebraic — elementwise product on
//! bipolar vectors), `permute` is **`Exact`** (a cyclic shift), and `bundle` (elementwise
//! superposition) carries a **`Proven`** capacity bound *citing Clarkson/Thomas* — but that bound's
//! derivation, checked instantiation, and ≥1e4-trial validation are **M-131**. So this module ships
//! the `bundle` *algebra* and the Value-level wrappers for the **Exact** ops; the `Proven`
//! Value-level bundle (which must carry the checked `CapacityBound`, M-I2) is added in M-131 — we do
//! not stamp `Proven` on a value without a checked bound here (VR-5).
//!
//! **Trusted-base discipline (ADR-014 / DN-21 §5 F-1):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

pub mod bsc;
pub mod capacity;
pub mod cleanup;
pub mod fhrr;
pub mod hrr;
pub mod mapb;
pub mod mapi;
pub mod matrix;
pub mod recon;
pub mod resonator;
pub mod sbc;
#[cfg(test)]
mod tests;
pub(crate) mod wrap;

pub use bsc::Bsc;
pub use cleanup::{CleanupMemory, Match};
// The RFC-0010 decode-methodology selection surface (`decode_select` + `reconstruct_factors_selected`)
// was relocated to the `mycelium-vsa-decode` crate (M-971): it is the only part of this crate that
// used `mycelium-select`, and hosting it here made `mycelium-vsa` depend on `mycelium-select`, which
// (with `mycelium-select -[dev]-> mycelium-interp`) closed the interp↔vsa↔select cycle DN-68 forbids.
// `mycelium-vsa` now depends only on `mycelium-core`; consumers of the selected decode depend on
// `mycelium-vsa-decode` instead. See DN-68 + xtask/deps-strata.toml.
pub use fhrr::Fhrr;
pub use hrr::Hrr;
pub use mapb::MapB;
pub use mapi::MapI;
pub use matrix::{matrix_tag, RFC0003_MATRIX};
pub use recon::{reconstruct_factors, reconstruct_role};
pub use resonator::{
    factorize, Cleanup, Factorization, Init, IterationRecord, ResonatorParams, ResonatorProfile,
    ResonatorTrace, StopReason, MAPI_RESONATOR_PROFILE,
};
pub use sbc::Sbc;

use mycelium_core::{Bound, BoundBasis, BoundKind, GuaranteeStrength};

/// The VSA operations a model supplies (RFC-0003 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsaOp {
    /// Binding (associate two hypervectors).
    Bind,
    /// Unbinding (recover a factor).
    Unbind,
    /// Bundling / superposition (set-like union).
    Bundle,
    /// Permutation (protect order / quoting).
    Permute,
}

/// Why a VSA operation could not be performed — always explicit, never a silent coercion (G2).
#[derive(Debug, Clone, PartialEq)]
pub enum VsaError {
    /// Operand dimensionalities disagree (or disagree with the model's `dim`).
    DimMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// A bundle was requested over zero items (no superposition is defined).
    EmptyBundle,
    /// A `Proven` bundle was requested but the dimension is below `requiredDim(items, δ)` — the
    /// cited capacity theorem's side-condition fails, so no `Proven` bound can be issued (M-131;
    /// M-I2/VR-5). Raise the dimension or lower the item count / relax `δ`.
    InsufficientCapacity {
        /// Number of items bundled.
        items: u64,
        /// The dimension supplied.
        dim: u64,
        /// The dimension the theorem requires.
        required: u64,
    },
    /// A `Proven` bundle was requested over **non-distinct** items. The cited capacity theorem
    /// assumes distinct codebook atoms; bundling duplicates inflates the apparent capacity, so a
    /// `Proven` tag there would be unbacked (A3-03/H6; M-I2/VR-5). Deduplicate the items.
    DuplicateBundleItems {
        /// Index of the first item that repeats an earlier one.
        index: usize,
    },
    /// Cleanup was requested against a [`CleanupMemory`] that holds no usable codebook atom for
    /// this model/dim — there is nothing to clean up against, which is a distinct condition from an
    /// empty *bundle* of operands ([`EmptyBundle`](VsaError::EmptyBundle)). Surfaced explicitly so a
    /// missing/dim-mismatched codebook is not mistaken for a degenerate bundle (A3-07; G2).
    EmptyCodebook,
    /// A value handed to a Value-level adapter was not a hypervector of the expected model.
    NotThisModel {
        /// The model id the adapter expected.
        expected: &'static str,
    },
    /// A component is outside the model's alphabet (e.g. not `±1` for a bipolar model, not
    /// `0/1` for BSC) — the algebra is undefined there; refused, never coerced (G2).
    NonAlphabetComponent {
        /// Index of the offending component.
        index: usize,
    },
    /// An `Empirical` Value-level op was requested outside the side-conditions its declared
    /// trial-validated profile covers — issuing the tag there would outrun the evidence (VR-5).
    OutsideEmpiricalProfile {
        /// Which side-condition failed.
        detail: String,
    },
    /// A MAP-B bundle input is itself a MAP-B bundle: reliability decays `1/2 + 1/2^r` with
    /// nesting depth `r` (RR-13; RFC-0003 §4), so nesting beyond depth 1 is refused explicitly —
    /// never a silent accuracy loss (M-242).
    NestedBundleUnsupported {
        /// The model whose bundle nesting was refused.
        model: &'static str,
    },
    /// An FHRR bundle component's phasor sum has (near-)zero magnitude — its phase is undefined;
    /// refused, never an arbitrary pick (G2).
    DegenerateBundleComponent {
        /// Index of the offending component.
        index: usize,
    },
    /// The manifest does not support compositional reconstruction with a cleanup decode — the
    /// RFC-0003 §6 indexed-vs-compositional distinction, made operational (M-260).
    NotCompositional,
    /// The requested role is not named in the manifest's recipe — reconstruction outside the
    /// recorded structure is refused, never guessed (G2).
    UnknownRole {
        /// The role that was asked for.
        role: String,
    },
    /// The cleanup confidence fell below the manifest's own threshold — an explicit refusal,
    /// never a silent low-quality retrieval (G2; FR-S4).
    BelowCleanupThreshold {
        /// The achieved confidence.
        confidence: f64,
        /// The manifest's threshold.
        threshold: f64,
    },
    /// A resonator factorization reached its iteration budget without a clean discrete fixed point —
    /// an explicit non-convergence verdict, never a returned factor set (RFC-0009 §6; G2). The
    /// inspectable run trace is attached (boxed to keep the error enum small).
    ResonatorBudgetExhausted {
        /// The full run trace (stop reason, similarity trajectory, final decode).
        trace: Box<crate::resonator::ResonatorTrace>,
    },
    /// A resonator factorization entered a **genuine limit cycle** (period ≥ 2) on the decoded index
    /// tuple `ι` — surfaced explicitly, never run to budget silently (RFC-0009 §3/§6; §8.1 P3).
    ResonatorOscillating {
        /// The full run trace (the `StopReason::Oscillating` records the cycle period).
        trace: Box<crate::resonator::ResonatorTrace>,
    },
    /// A resonator factorization reached a **stationary tuple that plateaued below `τ_lock`**: `ι`
    /// stopped changing and its per-slot similarity stopped climbing before every slot locked — a
    /// stuck fixed point, refused explicitly rather than returned as factors (RFC-0009 §3/§6). This
    /// is distinct from a real cycle (`ResonatorOscillating`); the M-350 fix keeps a *still-climbing*
    /// stationary tuple iterating toward lock instead of aborting it.
    ResonatorStalled {
        /// The full run trace (the `StopReason::Stalled` records the stationary-sweep count).
        trace: Box<crate::resonator::ResonatorTrace>,
    },
    /// A resonator factorization converged, but some slot's confidence fell below the requested
    /// threshold — "converged ≠ correct"; refused, never a silent low-confidence guess (RFC-0009
    /// §5.4; §8.1 P5).
    ResonatorBelowConfidence {
        /// The offending factor slot.
        slot: usize,
        /// The achieved confidence.
        confidence: f64,
        /// The requested threshold.
        threshold: f64,
        /// The full run trace.
        trace: Box<crate::resonator::ResonatorTrace>,
    },
    /// A resonator factorization converged, but some slot's margin (top minus runner-up) fell below
    /// the requested threshold — an explicit ambiguity refusal, never a coin-flip between near-tied
    /// atoms (RFC-0009 §5.4 / §9 Q5).
    ResonatorBelowMargin {
        /// The offending factor slot.
        slot: usize,
        /// The achieved margin.
        margin: f64,
        /// The requested threshold.
        threshold: f64,
        /// The full run trace.
        trace: Box<crate::resonator::ResonatorTrace>,
    },
    /// The decode-method selector (RFC-0010) chose `Refuse`, or a forced arm hit the honesty floor:
    /// the request is too large to enumerate **and** outside the resonator regime (or a forced
    /// `BruteForceExact` exceeds the enumeration budget / a forced `Resonator` is out of regime). An
    /// explicit refusal, never a silent best-effort decode (RFC-0010 §4.4/§4.5; G2).
    DecodeRefused {
        /// Why the decode was refused (which arm, which gate failed).
        detail: String,
    },
    /// A brute-force `Exact` decode (RFC-0010) found the instance **non-identifiable**: the true tuple
    /// is not the *unique* global arg-max, so no `Exact` factor set can be claimed (a coin-flip between
    /// tied tuples is exactly what `Exact` forbids). Refused, never guessed (RFC-0010 §4.4; G2).
    NonIdentifiable {
        /// The runner-up combination's similarity (tied with, or above, the apparent best).
        runner_up_similarity: f64,
    },
    /// A constructed result violated a Core IR invariant.
    Wf(mycelium_core::WfError),
}

impl core::fmt::Display for VsaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VsaError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            VsaError::EmptyBundle => write!(f, "bundle requires at least one item"),
            VsaError::EmptyCodebook => write!(
                f,
                "cleanup memory holds no codebook atom for this model/dim to clean up against"
            ),
            VsaError::InsufficientCapacity {
                items,
                dim,
                required,
            } => write!(
                f,
                "insufficient capacity for a Proven bound: bundling {items} items needs dim ≥ {required}, got {dim}"
            ),
            VsaError::DuplicateBundleItems { index } => write!(
                f,
                "item {index} repeats an earlier one; a Proven capacity bound needs distinct items"
            ),
            VsaError::NotThisModel { expected } => {
                write!(f, "expected a {expected} hypervector value")
            }
            VsaError::NonAlphabetComponent { index } => {
                write!(f, "component {index} is outside the model's alphabet")
            }
            VsaError::OutsideEmpiricalProfile { detail } => {
                write!(f, "outside the trial-validated empirical profile: {detail}")
            }
            VsaError::NestedBundleUnsupported { model } => write!(
                f,
                "{model} bundle nesting beyond depth 1 is refused (reliability decays with depth — RR-13)"
            ),
            VsaError::DegenerateBundleComponent { index } => write!(
                f,
                "bundle component {index} has a vanished phasor sum — its phase is undefined"
            ),
            VsaError::NotCompositional => write!(
                f,
                "manifest does not support compositional reconstruction with a cleanup decode"
            ),
            VsaError::UnknownRole { role } => {
                write!(f, "role {role:?} is not named in the manifest's recipe")
            }
            VsaError::BelowCleanupThreshold {
                confidence,
                threshold,
            } => write!(
                f,
                "cleanup confidence {confidence} is below the manifest threshold {threshold}"
            ),
            VsaError::ResonatorBudgetExhausted { trace } => write!(
                f,
                "resonator did not converge within the iteration budget ({} sweeps)",
                trace.iterations
            ),
            VsaError::ResonatorOscillating { trace } => write!(
                f,
                "resonator oscillated (decoded index tuple entered a limit cycle) after {} sweeps",
                trace.iterations
            ),
            VsaError::ResonatorStalled { trace } => write!(
                f,
                "resonator stalled (decoded index tuple stationary but below τ_lock) after {} sweeps",
                trace.iterations
            ),
            VsaError::ResonatorBelowConfidence {
                slot,
                confidence,
                threshold,
                ..
            } => write!(
                f,
                "resonator slot {slot} confidence {confidence} is below the threshold {threshold}"
            ),
            VsaError::ResonatorBelowMargin {
                slot,
                margin,
                threshold,
                ..
            } => write!(
                f,
                "resonator slot {slot} margin {margin} is below the threshold {threshold} (ambiguous)"
            ),
            VsaError::DecodeRefused { detail } => {
                write!(f, "decode-method selection refused: {detail}")
            }
            VsaError::NonIdentifiable {
                runner_up_similarity,
            } => write!(
                f,
                "brute-force decode found the instance non-identifiable (runner-up similarity {runner_up_similarity}); no Exact factorization"
            ),
            VsaError::Wf(e) => write!(f, "well-formedness violation: {e}"),
        }
    }
}

impl std::error::Error for VsaError {}

/// A composition-style VSA model (RFC-0003 §3): the `bind`/`unbind` (+ self-inverse flag),
/// `bundle`, `permute`, `similarity` algebra over hypervectors (represented as `&[f64]`), plus the
/// honest per-operation guarantee tag. Concrete models (MAP-I, …) implement it; the registry that
/// resolves a `Repr::Vsa { model }` to an implementation is ADR-008 (later).
pub trait VsaModel {
    /// The registry model id (e.g. `"MAP-I"`), matching `Repr::Vsa { model }`.
    fn model_id(&self) -> &'static str;

    /// Whether `unbind` is the same operation as `bind` (true for MAP-I / BSC).
    fn self_inverse(&self) -> bool;

    /// The honest intrinsic guarantee for an operation (RFC-0003 §4). `Proven` here is a *literature*
    /// claim about the operation; a `Proven` **value** still requires a checked bound (M-131, M-I2).
    fn intrinsic_guarantee(&self, op: VsaOp) -> GuaranteeStrength;

    /// Bind two hypervectors (associate). For MAP-I this is the elementwise product.
    fn bind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError>;

    /// Unbind (recover a factor): the (approximate or exact) inverse of [`bind`](Self::bind).
    fn unbind(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError>;

    /// Bundle (superpose) a non-empty set of hypervectors. The retrieval/capacity bound is supplied
    /// by the bound derivation (M-131), not here.
    fn bundle(&self, items: &[&[f64]]) -> Result<Vec<f64>, VsaError>;

    /// Permute (cyclically shift) a hypervector by `shift` positions — protects order/quotes a role.
    fn permute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError>;

    /// The inverse of [`permute`](Self::permute) by the same `shift`.
    fn unpermute(&self, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError>;

    /// Cosine similarity in `[-1, 1]` (`0` if either operand has zero norm).
    fn similarity(&self, a: &[f64], b: &[f64]) -> f64;
}

/// A **trial-validated empirical profile**: the regime over which a crate-declared `Empirical`
/// bound was actually validated, and the bound it backs. The honest counterpart of the M-131
/// checked-instantiation pattern for operations whose corpus basis is trials rather than a cited
/// theorem (RFC-0003 §4 "else `Empirical`"; M-I3/VR-5): the constants below are **exercised by
/// this crate's own trial tests** (`tests/empirical_profiles.rs`) with exactly the declared
/// `trials` count, and a Value-level op refuses — explicitly — outside the profile's
/// side-conditions rather than stretching the tag past its evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalProfile {
    /// Maximum number of operands the trials covered.
    pub max_items: usize,
    /// Whether the trials covered only an odd operand count (majority/sign bundles).
    pub odd_items_only: bool,
    /// Minimum dimensionality the trials covered.
    pub min_dim: u32,
    /// The validated failure probability (the δ the trials stayed at or below).
    pub delta: f64,
    /// Number of trials the validation runs.
    pub trials: u64,
    /// The fitting/validation method recorded in the `EmpiricalFit` basis.
    pub method: &'static str,
}

impl EmpiricalProfile {
    /// Check the profile's side-conditions for an op over `items` operands at `dim`; a violation
    /// is an explicit [`VsaError::OutsideEmpiricalProfile`].
    pub fn check(&self, items: usize, dim: u32) -> Result<(), VsaError> {
        if items == 0 || items > self.max_items {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!("validated for 1..={} items, got {items}", self.max_items),
            });
        }
        if self.odd_items_only && items.is_multiple_of(2) {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!("validated for an odd item count only, got {items}"),
            });
        }
        if dim < self.min_dim {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!("validated for dim ≥ {}, got {dim}", self.min_dim),
            });
        }
        Ok(())
    }

    /// The δ bound this profile backs, with its honest `EmpiricalFit` basis (M-I3).
    #[must_use]
    pub fn bound(&self) -> Bound {
        Bound {
            kind: BoundKind::Probability { delta: self.delta },
            basis: BoundBasis::EmpiricalFit {
                trials: self.trials,
                method: self.method.to_owned(),
            },
        }
    }
}
