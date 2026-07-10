//! **Layer 2 — the VSA semantic layer** (M-1018 / DN-87 §2.1/§6.1): the Layer-1 corpus rows encoded
//! as hypervector structures on `mycelium-vsa` (role-filler `bind` + `bundle`), retrieved by
//! associative `cleanup`, with an EXPLAIN-able trace — and **every Layer-2 answer names its Layer-1
//! evidence** (a resolvable [`Citation`], the DoD's provenance requirement).
//!
//! This layer is **gated** (DN-87 §6.1, the honestly-gated improved-on-RAG bet): it lands behind the
//! `layer2_enabled` front flag, which stays **`false`** until the eval harness (`crate::eval`) shows
//! Layer 2 measurably beats/complements the Layer-1 baseline. For this ~5k-row structured corpus a
//! **Closed gate is the expected, honest outcome** — cleanup recall over thousands of bundle-encoded
//! records at `d=4096` is modest, which is exactly what the gate exists to measure. A Closed gate ⇒
//! the system keeps serving Layer-1 answers; the "improved-on-RAG" claim stays aspiration (G2/VR-5).
//!
//! Guarantee tags, at their supportable strength (VR-5):
//! - cleanup retrieval → **`Empirical`** (measured; near-orthogonality is not proven here);
//! - the per-record capacity bound → **`Proven` where available** (`dim ≥ requiredDim(m, δ)`, a
//!   checked instantiation — see [`encode`]);
//! - the exact `unbind` structured probe → the op itself is **`Exact`** (MAP-I self-inverse), but
//!   recovering a filler from a *bundle* is **`Empirical`** (crosstalk) — see [`Layer2Index::probe_kind`];
//! - the encode regime profile → **`Declared`** until trials (see [`profile`]).

pub mod atoms;
pub mod decode;
pub mod encode;
pub mod explain;
pub mod profile;

use std::collections::BTreeMap;

use mycelium_vsa::{CleanupMemory, MapI, Match, VsaModel};

use crate::model::TeroIndexReport;
use crate::query::Citation;

pub use atoms::TERO_L2_SEED;
pub use decode::{Layer2Candidate, Layer2Refusal};
pub use encode::{BuildOutput, Layer2EncodeRefusal};
pub use explain::{Layer2Explain, Layer2Hit};
pub use profile::L2_DIM;

/// How many ranked hits a served [`Layer2Index::query`] records in its EXPLAIN trace.
const EXPLAIN_HITS: usize = 5;

/// A Layer-2 answer: the recovered Layer-1 [`Citation`] (its evidence — the DoD requirement), the
/// retrieval confidence + margin, and the inspectable [`Layer2Explain`] trace. Fields are private so
/// an answer **cannot** be constructed without a resolved citation (the never-silent provenance rule,
/// the Layer-2 twin of [`crate::query::Answer`]'s invariant).
#[derive(Debug, Clone, PartialEq)]
pub struct Layer2Answer {
    citation: Citation,
    confidence: f64,
    margin: f64,
    explain: Layer2Explain,
}

impl Layer2Answer {
    /// The Layer-1 evidence this Layer-2 answer names — always resolvable (a Layer-2 answer with no
    /// Layer-1 citation cannot exist).
    #[must_use]
    pub fn citation(&self) -> &Citation {
        &self.citation
    }

    /// The retrieval confidence (top cosine).
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// The margin to the runner-up.
    #[must_use]
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// The EXPLAIN trace for this retrieval.
    #[must_use]
    pub fn explain(&self) -> &Layer2Explain {
        &self.explain
    }
}

/// A concise, inspectable summary of an encode pass — surfaced in every EXPLAIN trace so the
/// never-silent refusals/truncations are always visible (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildSummary {
    /// Records successfully encoded into the codebook.
    pub encoded: usize,
    /// Rows refused at encode time (out-of-regime / unstorable) — never-silent.
    pub refused: usize,
    /// Encoded records that had a term/edge list truncated at the cap.
    pub truncations: usize,
    /// Encoded records carrying an available `Proven` capacity bound.
    pub proven_records: usize,
    /// The largest per-record bundle term count encoded.
    pub max_terms: usize,
}

/// The Layer-2 index facade: a VSA codebook over a Layer-1 report plus the machinery to query it and
/// resolve a recovered anchor back to its Layer-1 citation.
#[derive(Debug, Clone)]
pub struct Layer2Index {
    model: MapI,
    memory: CleanupMemory,
    /// A small codebook of the distinct `kind` filler atoms — for the exact-unbind structured probe.
    kind_codebook: CleanupMemory,
    /// anchor → its resolvable Layer-1 citation (provenance preserved by construction).
    citations: BTreeMap<String, Citation>,
    refused: Vec<Layer2EncodeRefusal>,
    summary: BuildSummary,
}

impl Layer2Index {
    /// Build the Layer-2 index from a Layer-1 report: encode every row into the cleanup codebook
    /// (keyed by `anchor`), build the resolution map, and record the honest build summary + refusals.
    #[must_use]
    pub fn build(report: &TeroIndexReport) -> Self {
        let model = MapI::new(L2_DIM);
        let BuildOutput {
            memory,
            refused,
            truncations,
            proven_records,
            max_terms,
        } = encode::build_codebook(report, &model);

        // Resolution map: anchors are globally unique, so this is 1:1 and lets a recovered anchor
        // resolve to its Layer-1 citation (the DoD provenance link).
        let citations: BTreeMap<String, Citation> = report
            .items
            .iter()
            .map(|it| (it.anchor.clone(), Citation::from(it)))
            .collect();

        // A tiny codebook of the distinct `kind` filler atoms for the structured unbind probe.
        let mut kind_codebook = CleanupMemory::new(L2_DIM);
        let mut seen_kinds: Vec<String> = Vec::new();
        for it in &report.items {
            if !seen_kinds.contains(&it.kind) {
                seen_kinds.push(it.kind.clone());
                // insert cannot dim-mismatch (atom is L2_DIM); ignore the impossible error path.
                let _ = kind_codebook.insert(it.kind.clone(), atoms::atom(&it.kind, L2_DIM));
            }
        }

        let summary = BuildSummary {
            encoded: memory.len(),
            refused: refused.len(),
            truncations,
            proven_records,
            max_terms,
        };

        Layer2Index {
            model,
            memory,
            kind_codebook,
            citations,
            refused,
            summary,
        }
    }

    /// Number of encoded records in the codebook.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Whether the codebook is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// The never-silent encode refusals.
    #[must_use]
    pub fn refused(&self) -> &[Layer2EncodeRefusal] {
        &self.refused
    }

    /// The honest build summary.
    #[must_use]
    pub fn summary(&self) -> &BuildSummary {
        &self.summary
    }

    /// Resolve an `anchor` to its Layer-1 citation, if it is a real row (the provenance check the
    /// eval harness uses: a returned Layer-2 anchor must resolve to a real Layer-1 row).
    #[must_use]
    pub fn resolve(&self, anchor: &str) -> Option<&Citation> {
        self.citations.get(anchor)
    }

    /// Rank the codebook against a free-text query, returning the top-`k` candidates (for the eval
    /// harness's correctness@k). A typed [`Layer2Refusal`] when the query forms no probe / the
    /// codebook is empty — never a silent empty result.
    pub fn rank(&self, text: &str, k: usize) -> Result<Vec<Layer2Candidate>, Layer2Refusal> {
        if self.memory.is_empty() {
            return Err(Layer2Refusal::EmptyCodebook);
        }
        let Some((probe, _terms)) = encode::encode_query(text, &self.model) else {
            return Err(Layer2Refusal::EmptyQuery {
                query: text.to_owned(),
            });
        };
        Ok(decode::rank_probe(&self.memory, &self.model, &probe, k))
    }

    /// The served Layer-2 query path: form the probe, clean up to the best record, and — above the
    /// declared confidence/margin floor — return a [`Layer2Answer`] that **names its Layer-1
    /// citation**. Below the floor (or on an empty query/codebook) a typed [`Layer2Refusal`]; Layer 1
    /// remains the answer (never a silent low-quality retrieval — G2).
    pub fn query(&self, text: &str) -> Result<Layer2Answer, Layer2Refusal> {
        if self.memory.is_empty() {
            return Err(Layer2Refusal::EmptyCodebook);
        }
        let Some((probe, terms)) = encode::encode_query(text, &self.model) else {
            return Err(Layer2Refusal::EmptyQuery {
                query: text.to_owned(),
            });
        };

        // Primary path: the substrate's associative cleanup for the canonical top hit (confidence +
        // margin), plus a short ranked list for the EXPLAIN trace (same cosine scoring).
        let Some(best) = self.memory.cleanup(&probe, &self.model) else {
            return Err(Layer2Refusal::EmptyCodebook);
        };
        let ranked = decode::rank_probe(&self.memory, &self.model, &probe, EXPLAIN_HITS);

        // A non-finite confidence/margin (a NaN from a degenerate probe) must REFUSE, never slip
        // through the `< floor` comparison (NaN < x is false) — never-silent (G2). Unreachable from a
        // random-±1 bundle today (empty-term probes already refuse), but guarded so it stays a refusal.
        if !best.confidence.is_finite()
            || !best.margin.is_finite()
            || best.confidence < decode::L2_MIN_CONFIDENCE
            || best.margin < decode::L2_MIN_MARGIN
        {
            return Err(Layer2Refusal::LowConfidence {
                best_anchor: best.label,
                confidence: best.confidence,
                margin: best.margin,
            });
        }

        // Resolve the recovered anchor to its Layer-1 evidence — a Layer-2 answer with no resolvable
        // Layer-1 citation is a bug, so a missing resolution is a hard refusal, not a fabricated cite.
        let Some(citation) = self.citations.get(&best.label).cloned() else {
            return Err(Layer2Refusal::LowConfidence {
                best_anchor: best.label,
                confidence: best.confidence,
                margin: best.margin,
            });
        };

        let hits = ranked
            .iter()
            .enumerate()
            .map(|(i, c)| Layer2Hit {
                anchor: c.anchor.clone(),
                cosine: c.cosine,
                // Only the top hit's margin is a decision quantity; report it, 0.0 below.
                margin: if i == 0 { best.margin } else { 0.0 },
            })
            .collect();

        let explain = Layer2Explain {
            model_id: self.model.model_id().to_owned(),
            dim: L2_DIM,
            seed: TERO_L2_SEED,
            query_terms: terms,
            candidates_scanned: self.memory.len(),
            hits,
            decode_method: "cleanup (nearest record by MAP-I cosine similarity)".to_owned(),
            empirical_profile_check: format!(
                "ok — {} record(s) encoded within the Declared regime, {} refused (never-silent)",
                self.summary.encoded, self.summary.refused
            ),
            guarantee_tag: "Empirical".to_owned(),
        };

        Ok(Layer2Answer {
            citation,
            confidence: best.confidence,
            margin: best.margin,
            explain,
        })
    }

    /// **Optional structured probe** (secondary path): recover a record's `kind` filler by an
    /// **exact** `unbind` of the `KIND` role, cleaned up against the small kind codebook. The unbind
    /// op is **`Exact`** (MAP-I self-inverse), but recovering a filler from a *bundle* is
    /// **`Empirical`** (bundle crosstalk) — so the returned [`Match`] confidence is the honest,
    /// inspectable quantity, never an `Exact`-stamped guess (VR-5). `None` when the anchor is not
    /// encoded. (The resonator/`reconstruct_factors_auto` factor path is deliberately not used — a
    /// variable-arity role-filler bundle is not an F-factor product in the resonator's regime.)
    #[must_use]
    pub fn probe_kind(&self, anchor: &str) -> Option<Match> {
        let record = self
            .memory
            .atoms()
            .find(|(label, _)| *label == anchor)
            .map(|(_, v)| v.to_vec())?;
        let role = atoms::atom(atoms::ROLE_KIND, L2_DIM);
        // unbind == bind for MAP-I (self-inverse); both operands are L2_DIM, so this cannot error.
        let noisy = self.model.unbind(&record, &role).ok()?;
        self.kind_codebook.cleanup(&noisy, &self.model)
    }
}
