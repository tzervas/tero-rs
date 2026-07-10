//! The Layer-2 **EXPLAIN** trace (M-1018) — "why this record, at what confidence", the semantic-layer
//! twin of the Layer-1 [`crate::query::Explain`]. A Layer-2 answer is inspectable the same way a
//! Layer-1 answer is (no black boxes — G2): the model + dim + seed that defined the codebook, the
//! query terms, how many records were scanned, the ranked hits with their cosine + margin, the decode
//! method, the profile side-condition check, and the honest guarantee tag on the retrieval.

use serde::Serialize;

/// One ranked Layer-2 candidate in a trace: the recovered Layer-1 `anchor` (the citation key), its
/// cosine similarity to the query probe, and its margin to the next candidate (top hit only; `0.0`
/// for lower ranks where the pairwise margin is not the decision quantity).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Layer2Hit {
    /// The recovered Layer-1 anchor.
    pub anchor: String,
    /// Cosine similarity of the query probe to this record (the confidence).
    pub cosine: f64,
    /// Gap to the next-best candidate (top hit only; `0.0` otherwise).
    pub margin: f64,
}

/// The inspectable trace behind one Layer-2 retrieval.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Layer2Explain {
    /// The VSA model id backing the codebook (`MAP-I`).
    pub model_id: String,
    /// The hypervector dimensionality.
    pub dim: u32,
    /// The committed master seed the codebook atoms were drawn from (reproducibility).
    pub seed: u64,
    /// The tokenized query terms that formed the probe.
    pub query_terms: Vec<String>,
    /// How many records were scanned (the codebook length) — the candidate universe.
    pub candidates_scanned: usize,
    /// The ranked hits considered, best first.
    pub hits: Vec<Layer2Hit>,
    /// The decode methodology used (`"cleanup"` for the primary associative retrieval).
    pub decode_method: String,
    /// The empirical-profile side-condition check result, verbatim (`"ok (m≤max_items, dim≥min_dim)"`
    /// or the explicit failure), so the never-silent guard is visible in the trace.
    pub empirical_profile_check: String,
    /// The honest guarantee tag on this retrieval — `Empirical` for cleanup (measured, not proven).
    pub guarantee_tag: String,
}
