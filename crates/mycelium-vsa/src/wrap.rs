//! Internal Value-level plumbing shared by the concrete models: extracting a model's hypervector
//! from a [`Value`] (checking model id + dim) and wrapping op results with honest `Meta`.

use mycelium_core::{
    operation_hash, Bound, ContentHash, GuaranteeStrength, Meta, Payload, Provenance, Repr,
    SparsityClass, Value,
};

use crate::VsaError;

/// Extract the hypervector data of a `model_id` value at `dim` (dense sparsity class), refusing
/// anything else explicitly.
pub(crate) fn hv_of<'a>(
    model_id: &'static str,
    dim: u32,
    v: &'a Value,
) -> Result<&'a [f64], VsaError> {
    match (v.repr(), v.payload()) {
        (Repr::Vsa { model, dim: d, .. }, Payload::Hypervector(h))
            if model == model_id && *d == dim =>
        {
            Ok(h)
        }
        _ => Err(VsaError::NotThisModel { expected: model_id }),
    }
}

/// Wrap a result vector into an **`Exact`** `model_id` value with honest `Derived` provenance.
pub(crate) fn wrap_exact(
    model_id: &str,
    dim: u32,
    data: Vec<f64>,
    op: &str,
    inputs: Vec<ContentHash>,
) -> Result<Value, VsaError> {
    wrap(
        model_id,
        dim,
        data,
        op,
        inputs,
        GuaranteeStrength::Exact,
        None,
    )
}

/// Wrap a result vector into a `model_id` value at `guarantee` carrying `bound` (whose basis must
/// match the guarantee — `Meta::new` enforces M-I1…M-I4).
pub(crate) fn wrap(
    model_id: &str,
    dim: u32,
    data: Vec<f64>,
    op: &str,
    inputs: Vec<ContentHash>,
    guarantee: GuaranteeStrength,
    bound: Option<Bound>,
) -> Result<Value, VsaError> {
    let meta = Meta::new(
        Provenance::Derived {
            op: operation_hash(op),
            inputs,
        },
        guarantee,
        bound,
        None,
        None,
        None,
    )
    .map_err(VsaError::Wf)?;
    Value::new(
        Repr::Vsa {
            model: model_id.to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(data),
        meta,
    )
    .map_err(VsaError::Wf)
}

/// Cosine similarity in `[-1, 1]` (`0` if either operand has zero norm) — the default similarity
/// for real-vector models.
pub(crate) fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Cyclic left rotation by `shift` (the `permute` shared by every vector model).
pub(crate) fn rotate(a: &[f64], shift: i64) -> Vec<f64> {
    let d = a.len() as i64;
    (0..a.len())
        .map(|i| a[(i as i64 + shift).rem_euclid(d) as usize])
        .collect()
}
