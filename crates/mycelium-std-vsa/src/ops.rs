//! Per-model operation vocabulary — the Ring-1 ergonomic surface over `VsaModel`.
//!
//! Each function is a thin, honest wrapper over the corresponding `VsaModel` method.  Its
//! guarantee tag is read from the RFC-0003 §4 matrix (via `VsaModel::intrinsic_guarantee`),
//! never invented here (C2 / VR-5).
//!
//! All errors are explicit (C1 / G2):
//! - `DimMismatch` — operand lengths disagree.
//! - `EmptyBundle` — bundle over zero items.
//! - `NestedBundleUnsupported` — MAP-B nesting beyond depth 1 (RR-13).
//! - `EmptyCodebook` — cleanup against an empty item memory.
//!
//! Approximate ops carry their inspectable artifact (C3):
//! - [`bundle`] — the `VsaModel::intrinsic_guarantee` for `Bundle` (callers may further compose
//!   with a capacity-bound certificate from `mycelium-vsa`'s Value-level adapters).
//! - [`cleanup`] — returns `Match { label, confidence, margin }` and errors on
//!   below-threshold / ambiguous results (the thresholds are caller-supplied, never hidden).

use mycelium_vsa::{CleanupMemory, Match, VsaError, VsaModel};

/// Bind two hypervectors (associate).
///
/// For MAP-I/MAP-B/BSC/HRR/FHRR this is an algebraic `Exact` op; for SBC it is `Proven`
/// (Bloom analysis — `matrix.rs`).  Both operands must have the same length as the model's
/// `dim`; a mismatch is `Err(DimMismatch)`, never a silent coercion (C1 / G2).
///
/// # Errors
/// - [`VsaError::DimMismatch`] — the operands' lengths disagree with each other or the model.
#[inline]
pub fn bind<M: VsaModel>(model: &M, a: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
    model.bind(a, b)
}

/// Unbind (recover a factor from a bind product).
///
/// Exact for self-inverse models (MAP-I/MAP-B/BSC) and SBC (`Proven`); `Empirical` for
/// HRR/FHRR (the approximate inverse is lossy — the residual weak link; RFC-0003 §4.1).  For
/// HRR/FHRR the result should be cleaned up against a codebook ([`cleanup`]) to obtain a
/// usable item — do not treat a raw unbind result as `Exact`.
///
/// # Errors
/// - [`VsaError::DimMismatch`] — the operands' lengths disagree with each other or the model.
#[inline]
pub fn unbind<M: VsaModel>(model: &M, c: &[f64], b: &[f64]) -> Result<Vec<f64>, VsaError> {
    model.unbind(c, b)
}

/// Bundle (superpose) a non-empty set of hypervectors.
///
/// The guarantee tag depends on the model (read from `intrinsic_guarantee(Bundle)`):
/// `Proven` for MAP-I/MAP-B/BSC/SBC; `Empirical` for HRR/FHRR.  A bundle over zero items is
/// `Err(EmptyBundle)`, never a fabricated zero vector (C1 / G2).  MAP-B nested-bundle inputs
/// yield `Err(NestedBundleUnsupported)` — reliability decays `1/2 + 1/2^r` with nesting depth
/// `r`, so nesting beyond depth 1 is refused explicitly rather than silently degrading accuracy
/// (RR-13).
///
/// For a Value-level `Proven` bundle with a checked `CapacityBound` certificate, use the
/// Value-level adapters in `mycelium-vsa` directly (e.g. `MapI::bundle_values_certified`).
///
/// # Errors
/// - [`VsaError::EmptyBundle`] — zero items supplied.
/// - [`VsaError::NestedBundleUnsupported`] — MAP-B input is itself a MAP-B bundle (RR-13).
/// - [`VsaError::DimMismatch`] — items have differing lengths.
#[inline]
pub fn bundle<M: VsaModel>(model: &M, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
    model.bundle(items)
}

/// Permute (cyclically shift) a hypervector by `shift` positions.
///
/// `Exact` for **all** models (§4.1 erratum — a fixed coordinate bijection; RFC-0003 §4.1).
/// The inverse is [`unpermute`] with the same `shift`.
///
/// # Errors
/// - [`VsaError::DimMismatch`] — operand length disagrees with the model's `dim`.
#[inline]
pub fn permute<M: VsaModel>(model: &M, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
    model.permute(a, shift)
}

/// The inverse of [`permute`] by the same `shift` — exactly undoes the cyclic rotation.
///
/// `Exact` for all models (a permutation's inverse is always exact; §4.1 erratum).
///
/// # Errors
/// - [`VsaError::DimMismatch`] — operand length disagrees with the model's `dim`.
#[inline]
pub fn unpermute<M: VsaModel>(model: &M, a: &[f64], shift: i64) -> Result<Vec<f64>, VsaError> {
    model.unpermute(a, shift)
}

/// Cosine similarity of two hypervectors in `[-1, 1]`.
///
/// Deterministic (a fixed score); `Exact` — the primitive *score* is exact even when it is used
/// as a *decision* input (cleanup, resonator per-slot argmax) where the *decision* is
/// bounded-lossy.  See FLAG Q2 in `vsa.md §7` for the open question about a
/// crosstalk-context bound when used decisionally.
///
/// Returns `0.0` if either operand has zero norm (the model's documented convention).
#[inline]
pub fn similarity<M: VsaModel>(model: &M, a: &[f64], b: &[f64]) -> f64 {
    model.similarity(a, b)
}

/// Role–filler binding: `bind(role, filler)`.
///
/// A thin alias for [`bind`] that names the operand roles explicitly.  Same guarantee as
/// `bind` for the model (see [`bind`] for the guarantee-tag breakdown).
///
/// # Errors
/// See [`bind`].
#[inline]
pub fn bind_role<M: VsaModel>(
    model: &M,
    role: &[f64],
    filler: &[f64],
) -> Result<Vec<f64>, VsaError> {
    bind(model, role, filler)
}

/// Cleanup: nearest-atom indexed retrieval against an item memory.
///
/// `Empirical` (bounded-lossy item memory, Clarkson Thm 16; M-132).  Returns
/// `Match { label, index, confidence, margin }` so the caller can inspect and threshold the
/// result — never a silent low-confidence guess (C1/C3 / G2 / FR-C2).
///
/// Explicit refusals (C1 / G2):
/// - `Err(BelowCleanupThreshold)` when `confidence < min_confidence` (the caller-supplied floor).
/// - `Err(BelowCleanupThreshold)` when `margin < min_margin` (near-tied slots). See the FLAG in
///   the body: `mycelium-vsa::VsaError` has **no** distinct `Ambiguous` variant, so the
///   margin-shortfall case is surfaced through `BelowCleanupThreshold` (with `confidence` carrying
///   the margin and `threshold` carrying `min_margin`) — never a silent guess. A dedicated variant
///   is a kernel change (FLAG).
/// - `Err(EmptyCodebook)` when the memory is empty or the query has the wrong length.
///
/// Setting `min_confidence = 0.0` and `min_margin = 0.0` disables the threshold gates,
/// returning the raw nearest-neighbour hit (the confidence/margin are still reported in
/// `Match` for the caller to inspect).
///
/// # Errors
/// - [`VsaError::EmptyCodebook`] — the memory is empty or query length disagrees with `dim`.
/// - [`VsaError::BelowCleanupThreshold`] — `confidence < min_confidence`, **or** the
///   margin-shortfall case `margin < min_margin` (see the body FLAG: `VsaError` has no
///   `Ambiguous` variant, so both refusals share this variant).
pub fn cleanup<M: VsaModel>(
    codebook: &CleanupMemory,
    query: &[f64],
    model: &M,
    min_confidence: f64,
    min_margin: f64,
) -> Result<Match, VsaError> {
    let hit = codebook
        .cleanup(query, model)
        .ok_or(VsaError::EmptyCodebook)?;

    if hit.confidence < min_confidence {
        return Err(VsaError::BelowCleanupThreshold {
            confidence: hit.confidence,
            threshold: min_confidence,
        });
    }
    if hit.margin < min_margin {
        // FLAG: VsaError does not have an Ambiguous variant in the upstream crate.
        // The spec (vsa.md §3) enumerates `Ambiguous` as a distinct error from
        // `BelowCleanupThreshold`.  The kernel's VsaError has `BelowCleanupThreshold` and
        // `ResonatorBelowMargin`, but not a top-level `Ambiguous`.  We surface the margin
        // shortfall through `BelowCleanupThreshold` with the margin as the `confidence` field
        // and the `min_margin` as the `threshold`, with a note so callers can distinguish.
        // Resolving this requires either extending VsaError (a kernel change) or adding a
        // wrapper error type here.  For now we reuse BelowCleanupThreshold rather than
        // fabricating a silent answer (C1 / G2); the actual margin and threshold are visible.
        return Err(VsaError::BelowCleanupThreshold {
            confidence: hit.margin,
            threshold: min_margin,
        });
    }
    Ok(hit)
}
