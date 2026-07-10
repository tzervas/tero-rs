//! `mycelium-vsa-decode` — the **VSA decode-methodology selection** layer (RFC-0010; M-350).
//!
//! This crate hosts the RFC-0005 selector site (site 3) that a factorization decode routes through:
//! for a request `{F, ∏kᵢ, d}` it chooses among `{ BruteForceExact, Resonator, Refuse }` and reads
//! the **honest guarantee tag off the chosen arm** (brute force ⇒ `Exact`, resonator ⇒ `Empirical`,
//! else an explicit refusal — RFC-0010 §4.4/§4.5). The selection machinery ([`decode_select`]) and
//! the manifest-driven entry point [`reconstruct_factors_selected`] were **relocated out of
//! `mycelium-vsa`** (M-971) because they are the *only* VSA code that depends on `mycelium-select`:
//! keeping them in `mycelium-vsa` made that crate depend on `mycelium-select`, and since
//! `mycelium-select -[dev]-> mycelium-interp -[normal]-> mycelium-vsa`, that edge closed the
//! `{interp, select, vsa}` dependency cycle DN-68's acyclic-deps invariant forbids. Extracting this
//! layer up (it legitimately sits above **both** `mycelium-vsa` and `mycelium-select`) leaves
//! `mycelium-vsa` depending only on `mycelium-core`, breaking the cycle structurally — the
//! extraction-not-loosening remedy DN-68 §5 mandates. See DN-68 + `xtask/deps-strata.toml`.
//!
//! **Trusted-base discipline (ADR-014 / DN-21 §5 F-1):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

pub mod decode_select;

pub use decode_select::{
    decode_method_policy, explain_decode_method, reconstruct_factors_auto, DecodeMethod,
    DecodeSelection, Explanation, DEFAULT_ENUM_BUDGET,
};

use mycelium_core::{ReconInfo, Value};
use mycelium_vsa::recon::{hv_payload, resonator_params_from_manifest};
use mycelium_vsa::{CleanupMemory, VsaError, VsaModel};

/// Value-level **auto-selected** factor decode (RFC-0010): like `mycelium_vsa::reconstruct_factors`,
/// but routes the decode **methodology** through the RFC-0005 selector instead of always running the
/// resonator. It reads the same `Resonator` manifest (the resonator arm uses its params), resolves
/// the record's hypervector, then calls [`reconstruct_factors_auto`] — so a request small enough to
/// enumerate is upgraded to a brute-force **`Exact`** decode (even one *outside* the resonator's
/// `{F, ∏kᵢ, d}` regime, e.g. `F=4` — brute force is exact for any factor count), an in-regime
/// request runs the **`Empirical`** resonator, and anything else is an explicit
/// [`VsaError::DecodeRefused`].
///
/// The returned [`DecodeSelection`] carries the chosen method, the mandatory EXPLAIN, the recovered
/// factors, and the **guarantee tag read off the chosen arm** (RFC-0010 §4.4) — only ever `Exact` or
/// `Empirical`, never `Proven` (the recon `≤Empirical` ceiling is untouched; brute force is genuinely
/// `Exact`, a strengthening, not an upgrade past the ceiling). `enum_budget` is the caller's tractable
/// enumeration size (e.g. [`DEFAULT_ENUM_BUDGET`]); `forced` pins an arm but cannot escape the honesty
/// floor (RFC-0010 §4.5). Unlike `reconstruct_factors`, this does **not** pre-gate on the resonator
/// profile — the selector decides, so brute-forceable out-of-regime instances are still recovered
/// (exactly), never refused for being outside the *resonator's* regime.
///
/// Relocated verbatim from `mycelium_vsa::recon` (M-971); it reuses that module's now-`pub`
/// `hv_payload` + `resonator_params_from_manifest` helpers so the manifest→params translation is not
/// duplicated (DRY).
pub fn reconstruct_factors_selected<M: VsaModel>(
    model: &M,
    manifest: &ReconInfo,
    record: &Value,
    codebooks: &[CleanupMemory],
    enum_budget: u128,
    forced: Option<DecodeMethod>,
) -> Result<DecodeSelection, VsaError> {
    if manifest.model() != model.model_id() {
        return Err(VsaError::NotThisModel {
            expected: model.model_id(),
        });
    }
    let params = resonator_params_from_manifest(manifest)?;
    let s = hv_payload(model, manifest.dim(), record)?;
    reconstruct_factors_auto(model, s, codebooks, &params, enum_budget, forced)
}
