//! Encoding utilities: compositions of the per-model vocabulary.
//!
//! Tag = meet of constituents (weakest-wins; RFC-0001 §4.7), **never upgraded**.  A sequence
//! built from `Proven` bundle and `Exact` permute is no stronger than its bundle; a set encoding
//! over `Empirical` bundle inputs stays `Empirical`.
//!
//! # Guarantee tags (compositions)
//!
//! | Op | Composition | Tag |
//! |---|---|---|
//! | `encode_seq` | `bundle` over `permute^i(item_i)` | meet(`bundle_tag`, `Exact`) = `bundle_tag` |
//! | `encode_set` | `bundle` of atoms | `bundle_tag` |
//!
//! where `bundle_tag` is `VsaModel::intrinsic_guarantee(Bundle)` for the model — `Proven` for
//! MAP-I/MAP-B/BSC/SBC, `Empirical` for HRR/FHRR.

use mycelium_vsa::{VsaError, VsaModel};

/// Sequence encoding: `bundle( permute^0(items[0]), permute^1(items[1]), … )`.
///
/// Each item `i` is permuted by shift `i` to protect its position (a positional encoding),
/// then the permuted items are bundled.  The guarantee tag is the meet of `bundle`'s tag
/// (model-dependent) and `permute`'s tag (`Exact`), which is just the `bundle` tag — see
/// `vsa.md §4` row `encode_seq`.
///
/// An empty slice is `Err(EmptyBundle)` — the bundle of zero items is undefined (C1 / G2).
///
/// # Errors
/// - [`VsaError::EmptyBundle`] — zero items supplied.
/// - [`VsaError::NestedBundleUnsupported`] — a MAP-B input is itself a bundle (RR-13).
/// - [`VsaError::DimMismatch`] — items have differing lengths.
pub fn encode_seq<M: VsaModel>(model: &M, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
    if items.is_empty() {
        return Err(VsaError::EmptyBundle);
    }
    let permuted: Vec<Vec<f64>> = items
        .iter()
        .enumerate()
        .map(|(i, item)| model.permute(item, i as i64))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&[f64]> = permuted.iter().map(Vec::as_slice).collect();
    model.bundle(&refs)
}

/// Set encoding: `bundle(items[0], items[1], …)` — superpose atoms without positional encoding.
///
/// The guarantee tag equals the model's `bundle` tag (the permute identity is `Exact` and
/// contributes nothing to the meet).  Useful for set/bag membership probes where order does not
/// matter.
///
/// An empty slice is `Err(EmptyBundle)` (C1 / G2).
///
/// # Errors
/// - [`VsaError::EmptyBundle`] — zero items supplied.
/// - [`VsaError::NestedBundleUnsupported`] — MAP-B nesting (RR-13).
/// - [`VsaError::DimMismatch`] — items have differing lengths.
#[inline]
pub fn encode_set<M: VsaModel>(model: &M, items: &[&[f64]]) -> Result<Vec<f64>, VsaError> {
    if items.is_empty() {
        return Err(VsaError::EmptyBundle);
    }
    model.bundle(items)
}
