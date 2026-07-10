//! Layer-2 **decoding** (M-1018): rank the codebook against a query probe by cleanup similarity, and
//! the never-silent typed refusals that guard a weak or impossible retrieval.
//!
//! The primary retrieval is **associative cleanup** — nearest record to the query probe by
//! [`mycelium_vsa::VsaModel::similarity`] (cosine). The single-best form is exactly
//! [`mycelium_vsa::CleanupMemory::cleanup`] (used by [`super::Layer2Index::query`]); [`rank_probe`]
//! generalizes it to the top-K list the eval harness needs (correctness@k). Both score the *same*
//! way, so `rank_probe(..)[0]` and `CleanupMemory::cleanup` agree on the top hit.
//!
//! Honesty (VR-5): a recovered anchor is only *useful* with its confidence + margin, so below the
//! declared thresholds the served path **refuses explicitly** ([`Layer2Refusal::LowConfidence`])
//! rather than return a low-quality nearest neighbour (FR-S4/G2). The thresholds are `Declared` knobs
//! (below), **not** tuned to manufacture a gate pass — the eval measures raw recall without them.

use mycelium_vsa::{CleanupMemory, MapI, VsaModel};

/// Minimum cosine confidence for the served [`super::Layer2Index::query`] path to return an answer.
/// A `Declared` knob (a nearest-neighbour below this is refused, not returned). Deliberately modest —
/// it is a floor on "is this even plausibly the right record", not a tuned gate threshold.
pub const L2_MIN_CONFIDENCE: f64 = 0.10;

/// Minimum margin (top minus runner-up) for the served path to return an answer — an ambiguity floor
/// (a near-tie between two records is refused, never a coin-flip). A `Declared` knob.
pub const L2_MIN_MARGIN: f64 = 0.02;

/// One ranked Layer-2 candidate: the recovered Layer-1 `anchor` (the citation key) + its cosine
/// similarity to the query probe.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer2Candidate {
    /// The recovered Layer-1 anchor.
    pub anchor: String,
    /// Cosine similarity of the probe to this record.
    pub cosine: f64,
}

/// A typed, never-silent Layer-2 "no answer" (the semantic-layer twin of [`crate::query::Refusal`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Layer2Refusal {
    /// The query tokenized to zero terms — no probe could be formed (mirrors the Layer-1 empty-text
    /// refusal; consistently matches nothing).
    EmptyQuery {
        /// The raw query string.
        query: String,
    },
    /// The codebook is empty — nothing to clean up against (distinct from an empty query).
    EmptyCodebook,
    /// The best match's confidence/margin fell below the declared floor — refused, never returned as
    /// a low-quality guess.
    LowConfidence {
        /// The best-matching anchor (recorded so the refusal is itself inspectable).
        best_anchor: String,
        /// Its cosine confidence.
        confidence: f64,
        /// Its margin to the runner-up.
        margin: f64,
    },
}

impl std::fmt::Display for Layer2Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer2Refusal::EmptyQuery { query } => write!(
                f,
                "refusing a Layer-2 query {query:?} — it tokenizes to zero terms, so no probe \
                 hypervector can be formed"
            ),
            Layer2Refusal::EmptyCodebook => write!(
                f,
                "refusing a Layer-2 query — the codebook holds no encoded record to clean up against"
            ),
            Layer2Refusal::LowConfidence {
                best_anchor,
                confidence,
                margin,
            } => write!(
                f,
                "refusing a Layer-2 answer — best match {best_anchor:?} confidence {confidence:.4} / \
                 margin {margin:.4} is below the declared floor (conf ≥ {L2_MIN_CONFIDENCE}, \
                 margin ≥ {L2_MIN_MARGIN}); Layer 1 remains the answer"
            ),
        }
    }
}

impl std::error::Error for Layer2Refusal {}

/// Rank the codebook against `probe`, returning the top-`k` candidates by cosine similarity,
/// descending, ties broken by `anchor` ascending (deterministic — two runs rank identically). This is
/// the cleanup decode generalized to a top-K list; `[0]` equals [`CleanupMemory::cleanup`]'s top hit.
#[must_use]
pub fn rank_probe(
    memory: &CleanupMemory,
    model: &MapI,
    probe: &[f64],
    k: usize,
) -> Vec<Layer2Candidate> {
    let mut scored: Vec<Layer2Candidate> = memory
        .atoms()
        .map(|(label, record)| Layer2Candidate {
            anchor: label.to_owned(),
            cosine: model.similarity(probe, record),
        })
        .collect();
    scored.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.anchor.cmp(&b.anchor))
    });
    scored.truncate(k);
    scored
}
