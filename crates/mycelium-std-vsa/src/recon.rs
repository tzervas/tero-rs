//! Reconstruction-manifest surface — ergonomic wrappers over `mycelium_vsa::recon`.
//!
//! Provides the two public decode paths the spec (vsa.md §3) names:
//!
//! 1. **[`reconstruct_role`]** — compositional reconstruction: unbind a record by a named role,
//!    then clean up against the codebook.  Requires a `CompositionalReconstruction` manifest with
//!    a `Cleanup` decode.  Fails explicitly on non-compositional manifests, unknown roles, or a
//!    below-threshold cleanup result (C1 / G2).
//!
//! 2. **[`reconstruct_factors`]** — resonator factorization: opt-in, `Empirical` ceiling, MAP-I
//!    first (then BSC per the profile); **never `Proven`** (FR-C2 / RFC-0009 §5).  Non-convergence
//!    verdicts (`BudgetExhausted`, `Oscillating`, `Stalled`, `OutsideEmpiricalProfile`,
//!    `BelowCleanupThreshold`) are explicit errors carrying the inspectable `ResonatorTrace` (C3).
//!
//! # Guarantee tags
//!
//! | Op | Tag | Note |
//! |---|---|---|
//! | `reconstruct_role` | meet of `unbind` + `cleanup` for the model | `Empirical` at most |
//! | `reconstruct_factors` | `Empirical` (MAP-I/BSC); `Declared` (sparse/HRR/FHRR) | FR-C2 |
//!
//! `reconstruct_factors` for sparse/HRR/FHRR is `Declared` (lossy/non-exact self-inverse;
//! RFC-0009 §9 Q6) — FLAG Q4: `std.vsa` v0 routes only the MAP-I/BSC validated path; the
//! `Declared` paths are deferred (RFC-0009 §10.3 deferred coverage).

use mycelium_core::{ReconInfo, Value};
use mycelium_vsa::{CleanupMemory, Factorization, Match, VsaError, VsaModel};

/// Compositional reconstruction: unbind `record` by a named `role`, then clean up against
/// `memory`.
///
/// Requires:
/// - A `CompositionalReconstruction` manifest (`Err(NotCompositional)` otherwise).
/// - The manifest's recipe to name `role` (`Err(UnknownRole)` otherwise).
/// - A `Cleanup` decode procedure with a threshold (`Err(NotCompositional)` otherwise).
/// - The cleanup confidence to meet the manifest's threshold (`Err(BelowCleanupThreshold)`).
///
/// Returns [`Match`] `{ label, index, confidence, margin }` — the recovered filler atom with its
/// confidence/margin so the caller can inspect and judge the quality (C3 / G2 / FR-C2).
///
/// The guarantee tag is the meet of the model's `unbind` tag and `cleanup`'s `Empirical` tag
/// (weakest-wins; RFC-0001 §4.7) — at most `Empirical`, never `Proven`.
///
/// # Errors
/// - [`VsaError::NotThisModel`] — `record` or `role_atom` is not of the manifest's model.
/// - [`VsaError::NotCompositional`] — the manifest does not support compositional reconstruction.
/// - [`VsaError::UnknownRole`] — `role` is not in the manifest's recipe.
/// - [`VsaError::EmptyCodebook`] — `memory` is empty or has the wrong dim.
/// - [`VsaError::BelowCleanupThreshold`] — cleanup confidence is below the manifest threshold.
pub fn reconstruct_role<M: VsaModel>(
    model: &M,
    manifest: &ReconInfo,
    record: &Value,
    role: &str,
    role_atom: &Value,
    memory: &CleanupMemory,
) -> Result<Match, VsaError> {
    mycelium_vsa::recon::reconstruct_role(model, manifest, record, role, role_atom, memory)
}

/// Resonator factorization: recover the unknown factor atoms of a bind product.
///
/// Opt-in, **probabilistic-only** (`Empirical` ceiling for MAP-I/BSC — never `Proven`; FR-C2).
/// For a `Resonator` manifest, reads the iteration budget and decode params, gates on the
/// trial-validated [`MAPI_RESONATOR_PROFILE`](mycelium_vsa::MAPI_RESONATOR_PROFILE), and runs
/// the resonator loop.
///
/// Returns [`Factorization`] **only** on a clean `Converged` verdict clearing every per-slot
/// confidence/margin threshold.  All non-convergence outcomes are explicit errors carrying the
/// inspectable [`ResonatorTrace`](mycelium_vsa::ResonatorTrace) (C3 / G2 / RFC-0009 §5/§6):
/// - [`VsaError::ResonatorBudgetExhausted`]
/// - [`VsaError::ResonatorOscillating`]
/// - [`VsaError::ResonatorStalled`]
/// - [`VsaError::ResonatorBelowConfidence`]
/// - [`VsaError::ResonatorBelowMargin`]
///
/// # FLAG Q4 — `Declared` coverage deferred
///
/// RFC-0009 §9 Q6 caps resonator factorization at `Empirical` for exact-bind models (MAP-I/BSC)
/// and `Declared` for sparse/HRR/FHRR.  This function gates on the MAP-I profile and is the
/// validated path; the `Declared` paths for sparse/HRR/FHRR are deferred to RFC-0009 §10.3
/// (deferred coverage) and not yet exported in `std.vsa` v0 (vsa.md §7-Q4).
///
/// # Errors
/// - [`VsaError::NotThisModel`] — `record` is not of the manifest's model.
/// - [`VsaError::NotCompositional`] — the manifest is not a `Resonator` decode.
/// - [`VsaError::OutsideEmpiricalProfile`] — the request exceeds the validated regime.
/// - [`VsaError::ResonatorBudgetExhausted`] — budget reached without convergence.
/// - [`VsaError::ResonatorOscillating`] — a genuine period-≥2 limit cycle.
/// - [`VsaError::ResonatorStalled`] — stationary tuple plateaued below `τ_lock`.
/// - [`VsaError::ResonatorBelowConfidence`] — a slot's confidence is below the threshold.
/// - [`VsaError::ResonatorBelowMargin`] — a slot's margin is below the threshold.
pub fn reconstruct_factors<M: VsaModel>(
    model: &M,
    manifest: &ReconInfo,
    record: &Value,
    codebooks: &[CleanupMemory],
) -> Result<Factorization, VsaError> {
    mycelium_vsa::recon::reconstruct_factors(model, manifest, record, codebooks)
}
