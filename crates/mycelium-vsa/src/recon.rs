//! **Reconstruction execution** over a [`ReconInfo`] manifest (M-260; RFC-0003 §6; FR-S4).
//!
//! The kernel carries the manifest *data type* (`mycelium_core::ReconInfo`); this module is the
//! submodule-side executor: [`reconstruct_role`] performs **true compositional reconstruction** —
//! unbind the record by a role named in the manifest's recipe, then clean up against the
//! codebook — recovering a *novel combination* never stored as an atom (the §6 exit criterion,
//! VSA's defining capability over a hash table). Everything is inspectable: the recipe names the
//! roles, the codebook is content-addressed, and the returned [`Match`] carries the cleanup
//! confidence/margin, thresholded against the manifest's own `cleanup_threshold` — a
//! below-threshold retrieval is an explicit error, never a silent low-quality answer (G2).

use mycelium_core::{CleanupShape, DecodeProcedure, InitStrategy, ReconInfo, ReconMode, Value};

use crate::resonator::{self, Cleanup, Factorization, Init, ResonatorParams};
use crate::{CleanupMemory, Match, VsaError, VsaModel, MAPI_RESONATOR_PROFILE};

/// Compositionally reconstruct the filler bound under `role` inside `record`, following the
/// manifest: requires a `CompositionalReconstruction` manifest with a `Cleanup` decode whose
/// recipe names `role`; unbinds `record` by the role atom and cleans the noisy result up against
/// `memory` (which must hold the manifest's codebook atoms). Explicit refusals: a non-matching
/// model/dim, a mode/procedure the manifest does not declare, an unknown role, and a retrieval
/// below the manifest's threshold.
pub fn reconstruct_role<M: VsaModel>(
    model: &M,
    manifest: &ReconInfo,
    record: &Value,
    role: &str,
    role_atom: &Value,
    memory: &CleanupMemory,
) -> Result<Match, VsaError> {
    if manifest.model() != model.model_id() {
        return Err(VsaError::NotThisModel {
            expected: model.model_id(),
        });
    }
    let recipe = match (manifest.mode(), manifest.recipe()) {
        (ReconMode::CompositionalReconstruction, Some(r)) => r,
        // An indexed-retrieval manifest cannot reconstruct compositionally — refusing is exactly
        // the §6 distinction made operational.
        _ => return Err(VsaError::NotCompositional),
    };
    if !recipe.roles.iter().any(|r| r == role) {
        return Err(VsaError::UnknownRole {
            role: role.to_owned(),
        });
    }
    let threshold = match (
        manifest.decode().procedure,
        manifest.decode().cleanup_threshold,
    ) {
        (DecodeProcedure::Cleanup, Some(t)) => t,
        // Resonator decoding is Phase-3 exploratory (FR-C2) — explicit, not a fallback.
        _ => return Err(VsaError::NotCompositional),
    };

    let record_hv = hv_payload(model, manifest.dim(), record)?;
    let role_hv = hv_payload(model, manifest.dim(), role_atom)?;
    let noisy = model.unbind(record_hv, role_hv)?;
    // `noisy` has length `dim` (it is an unbind of dim-checked payloads), so a `None` here means the
    // codebook is empty or dim-mismatched — i.e. there is nothing to clean up against. That is a
    // distinct condition from an empty bundle of operands, so surface the right variant (A3-07).
    let hit = memory
        .cleanup(&noisy, model)
        .ok_or(VsaError::EmptyCodebook)?;
    if hit.confidence < threshold {
        return Err(VsaError::BelowCleanupThreshold {
            confidence: hit.confidence,
            threshold,
        });
    }
    Ok(hit)
}

/// Factorize `record` — a bind product `s = x₁ ⊛ … ⊛ x_F` — into one codebook atom per slot, following
/// a **`Resonator`** manifest (RFC-0009; M-350). Mirrors [`reconstruct_role`] for the resonator decode:
/// it checks the model/dim, requires `DecodeProcedure::Resonator`, reads the iteration budget + the
/// optional decode params (`cleanup`/`beta`/`tau_lock`/`init`/`seed`, RFC-0003 §6.1) into a
/// [`ResonatorParams`], **gates on the trial-validated [`MAPI_RESONATOR_PROFILE`]** (an out-of-regime
/// request is an explicit [`VsaError::OutsideEmpiricalProfile`], never a stretched tag — RFC-0009 §5.2),
/// then runs [`resonator::factorize`]. The decoder-side thresholds not carried by the kernel manifest
/// (confidence, ambiguity-margin, oscillation window — RFC-0003 §6.1) take their `ResonatorParams`
/// defaults. Codebook resolution (`ContentHash` → [`CleanupMemory`]) is caller-provided, exactly as
/// `reconstruct_role` takes its `memory` — the `codebooks` are one cleanup memory per factor slot.
///
/// The honesty contract is [`resonator::factorize`]'s: a [`Factorization`] is returned **only** on a
/// clean `Converged` verdict clearing every gate; non-convergence, oscillation, and below-threshold are
/// explicit errors carrying the trace. The guarantee is `Empirical` (the profile's
/// [`bound`](crate::ResonatorProfile::bound)), never `Proven` (schema-enforced, `mycelium-core::recon`).
pub fn reconstruct_factors<M: VsaModel>(
    model: &M,
    manifest: &ReconInfo,
    record: &Value,
    codebooks: &[CleanupMemory],
) -> Result<Factorization, VsaError> {
    if manifest.model() != model.model_id() {
        return Err(VsaError::NotThisModel {
            expected: model.model_id(),
        });
    }
    let params = resonator_params_from_manifest(manifest)?;

    // Gate on the validated regime BEFORE running — out-of-regime is an explicit refusal (§5.2).
    let sizes: Vec<usize> = codebooks.iter().map(CleanupMemory::len).collect();
    MAPI_RESONATOR_PROFILE.check(codebooks.len(), &sizes, manifest.dim())?;

    let s = hv_payload(model, manifest.dim(), record)?;
    resonator::factorize(model, s, codebooks, &params)
}

/// Read the manifest's (additive RFC-0003 §6.1) resonator decode params into a [`ResonatorParams`].
/// A non-`Resonator` procedure (or a missing iteration budget) is the wrong decode for this executor
/// — an explicit [`VsaError::NotCompositional`]. Absent params take the recommended MAP-I defaults;
/// the numeric ranges were already checked by `ReconInfo::new`, so this only translates.
///
/// `pub` for the RFC-0010 selected-decode layer in `mycelium-vsa-decode` (M-971): that crate hosts
/// `reconstruct_factors_selected`, which was relocated out of this module so `mycelium-vsa` no
/// longer depends on `mycelium-select` (breaking the interp↔vsa↔select cycle — DN-68). It reuses
/// this manifest→params translation verbatim rather than duplicating it (DRY).
pub fn resonator_params_from_manifest(manifest: &ReconInfo) -> Result<ResonatorParams, VsaError> {
    let decode = manifest.decode();
    let iteration_budget = match (decode.procedure, decode.iteration_budget) {
        (DecodeProcedure::Resonator, Some(b)) => b,
        _ => return Err(VsaError::NotCompositional),
    };
    let mut params = ResonatorParams::mapi_default(iteration_budget, decode.seed.unwrap_or(0));
    // The kernel `CleanupShape` schema is `ArgMax | Softmax` (additive metadata; not the validated
    // Hebbian cleanup, which lives only in `mycelium-vsa` — no kernel change). So an *unspecified*
    // cleanup keeps the `mapi_default` validated default (`Cleanup::Hebbian`, the §10.3 wall-breach);
    // an explicit shape is honored as the caller's recorded (un-profiled) choice (RFC-0009 §10.3).
    params.cleanup = match decode.cleanup {
        None => params.cleanup, // the validated Hebbian default
        Some(CleanupShape::ArgMax) => Cleanup::ArgMax,
        Some(CleanupShape::Softmax) => Cleanup::Softmax {
            beta: decode.beta.unwrap_or(6.0),
        },
    };
    if let Some(tau) = decode.tau_lock {
        params.tau_lock = tau;
    }
    if let Some(InitStrategy::SeededGuess) = decode.init {
        params.init = Init::SeededGuess;
    }
    Ok(params)
}

/// Resolve a record `Value` to its hypervector payload slice, checking the model id and dimension
/// match (an explicit [`VsaError::NotThisModel`] otherwise, never a silent reinterpretation — G2).
///
/// `pub` for the same reason as [`resonator_params_from_manifest`]: the relocated RFC-0010
/// selected-decode layer (`mycelium-vsa-decode`, M-971) reuses this payload resolver verbatim.
pub fn hv_payload<'a, M: VsaModel>(
    model: &M,
    dim: u32,
    v: &'a Value,
) -> Result<&'a [f64], VsaError> {
    match (v.repr(), v.payload()) {
        (
            mycelium_core::Repr::Vsa {
                model: m, dim: d, ..
            },
            mycelium_core::Payload::Hypervector(h),
        ) if m == model.model_id() && *d == dim => Ok(h),
        _ => Err(VsaError::NotThisModel {
            expected: model.model_id(),
        }),
    }
}
