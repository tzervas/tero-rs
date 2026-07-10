//! Layer-2 **encoding** (M-1018): turn each Layer-1 [`TeroIndexItem`] into a role-filler
//! bind+bundle record hypervector, keyed in a [`CleanupMemory`] by the row's **`anchor`** — so the
//! codebook label *is* the Layer-1 citation key and provenance is preserved by construction (a
//! recovered label resolves straight back to its Layer-1 row).
//!
//! A record is `sign( bundle( ROLE ⊗ atom(filler) … ) )` over the row's fields (id/kind/family/
//! status/title-terms/summary-terms/depends_on/doc_refs/epic/guarantee). Every term list is capped
//! top-K ([`super::profile::L2_TERM_CAP`]) with the truncation **recorded, never silent** (G2, the
//! `query.rs::TEXT_RESULT_LIMIT` posture). Before a record is bundled, [`super::profile::L2_PROFILE`]'s
//! side-conditions are checked: a bundle that would exceed the validated `max_items`, or a dimension
//! below `min_dim`, is an **explicit per-record refusal** ([`Layer2EncodeRefusal`]), never a silent
//! over-capacity superposition.
//!
//! Honesty (VR-5): the retrieval this encoding supports is **`Empirical`** (measured by the eval
//! harness). A per-record **`Proven`** capacity bound is *available* where `dim ≥ required_dim(m, δ)`
//! (recorded as [`RecordStats::proven`]) — that is the honest checked-instantiation fact about the
//! bundle's internal decodability, distinct from the cross-codebook cleanup recall the gate measures.

use mycelium_vsa::{capacity, CleanupMemory, MapI, VsaModel};

use crate::model::{Family, TeroIndexItem, TeroIndexReport};

use super::atoms::{
    atom, ROLE_DEPENDS_ON, ROLE_DOC_REF, ROLE_EPIC, ROLE_FAMILY, ROLE_GUARANTEE, ROLE_ID,
    ROLE_KIND, ROLE_STATUS, ROLE_SUMMARY_TERM, ROLE_TITLE_TERM,
};
use super::profile::{L2_DELTA, L2_DIM, L2_PROFILE, L2_TERM_CAP};

/// A never-silent per-record encode refusal (G2): a row whose bundle fell outside the declared
/// [`L2_PROFILE`] regime (or whose vector could not be stored) is refused *explicitly* and recorded,
/// never dropped and never silently over-capacity-bundled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer2EncodeRefusal {
    /// The Layer-1 anchor of the refused row.
    pub anchor: String,
    /// Why it was refused, in author-facing terms.
    pub reason: String,
}

/// Per-record encode statistics — the honest, inspectable facts about one record's bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordStats {
    /// The number of role-filler binds actually bundled (`m`).
    pub n_terms: usize,
    /// Whether any of the row's term/edge lists was capped at [`L2_TERM_CAP`] (recorded, not silent).
    pub truncated: bool,
    /// Whether `dim ≥ required_dim(m, δ)` held — i.e. a `Proven` capacity bound is *available* for
    /// this record's bundle (the checked-instantiation fact; see the module docs).
    pub proven: bool,
}

/// The result of building the whole Layer-2 codebook from a report.
#[derive(Debug, Clone)]
pub struct BuildOutput {
    /// The cleanup memory: one record hypervector per encoded row, keyed by `anchor`.
    pub memory: CleanupMemory,
    /// Rows refused at encode time (out-of-regime / unstorable) — never-silent (G2).
    pub refused: Vec<Layer2EncodeRefusal>,
    /// How many encoded records had at least one list truncated at [`L2_TERM_CAP`].
    pub truncations: usize,
    /// How many encoded records carried an available `Proven` capacity bound (`dim ≥ requiredDim`).
    pub proven_records: usize,
    /// The largest per-record bundle term count `m` actually encoded (0 if none).
    pub max_terms: usize,
}

/// Tokenize `text` into lowercased, de-duplicated alphanumeric terms of length ≥ 2. Splitting on any
/// non-alphanumeric boundary (not just whitespace) is Layer-2's own tokenizer — deliberately a touch
/// richer than the Layer-1 `query.rs` whitespace split, so the semantic layer gets a fair encoding;
/// the two systems are compared as-is by the eval harness (never rigged to converge).
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.len() < 2 {
            continue;
        }
        let t = raw.to_lowercase();
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// Cap a discrete filler list top-K (first-occurrence order); return the kept slice length + whether
/// truncation happened.
fn cap_len(len: usize) -> (usize, bool) {
    if len > L2_TERM_CAP {
        (L2_TERM_CAP, true)
    } else {
        (len, false)
    }
}

/// The ordered role-filler `(role_symbol, filler_symbol)` pairs for one row, with every term/edge
/// list capped top-K. Returns the pairs and whether any list was truncated.
fn record_role_fillers(item: &TeroIndexItem) -> (Vec<(&'static str, String)>, bool) {
    let mut pairs: Vec<(&'static str, String)> = Vec::new();
    let mut truncated = false;

    if let Some(id) = &item.id {
        pairs.push((ROLE_ID, id.clone()));
    }
    pairs.push((ROLE_KIND, item.kind.clone()));
    pairs.push((ROLE_FAMILY, item.family.as_str().to_owned()));
    if let Some(status) = &item.status {
        pairs.push((ROLE_STATUS, status.clone()));
    }

    // Text terms — tokenized then capped.
    let title_terms = tokenize(&item.title);
    let (n, cut) = cap_len(title_terms.len());
    truncated |= cut;
    for t in title_terms.into_iter().take(n) {
        pairs.push((ROLE_TITLE_TERM, t));
    }
    if let Some(summary) = &item.summary {
        let summary_terms = tokenize(summary);
        let (n, cut) = cap_len(summary_terms.len());
        truncated |= cut;
        for t in summary_terms.into_iter().take(n) {
            pairs.push((ROLE_SUMMARY_TERM, t));
        }
    }

    // Edge lists (issues only) — capped as discrete ids/refs (not tokenized).
    let (n, cut) = cap_len(item.depends_on.len());
    truncated |= cut;
    for d in item.depends_on.iter().take(n) {
        pairs.push((ROLE_DEPENDS_ON, d.clone()));
    }
    let (n, cut) = cap_len(item.doc_refs.len());
    truncated |= cut;
    for r in item.doc_refs.iter().take(n) {
        pairs.push((ROLE_DOC_REF, r.clone()));
    }
    if let Some(epic) = &item.epic {
        pairs.push((ROLE_EPIC, epic.clone()));
    }
    if let Some(g) = &item.guarantee_tag {
        pairs.push((ROLE_GUARANTEE, g.clone()));
    }

    (pairs, truncated)
}

/// Elementwise sign of a bundle sum → a clean bipolar (`±1`) record. A tie (`0`, an even split of
/// contributions) is deterministically resolved to `+1` (never a random or context-dependent pick —
/// G2), so encoding stays a pure function of `(corpus, seed)`.
fn sign_bipolar(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|&x| if x < 0.0 { -1.0 } else { 1.0 })
        .collect()
}

/// Encode one row into its record hypervector plus its honest [`RecordStats`], or refuse it
/// explicitly if the bundle falls outside the declared [`L2_PROFILE`] regime.
pub fn encode_record(
    item: &TeroIndexItem,
    model: &MapI,
) -> Result<(Vec<f64>, RecordStats), Layer2EncodeRefusal> {
    let (pairs, truncated) = record_role_fillers(item);
    let m = pairs.len();

    // Never-silent capacity guard: refuse an out-of-regime bundle rather than superpose past the
    // validated profile (VsaError carries the exact failed side-condition).
    if let Err(e) = L2_PROFILE.check(m, L2_DIM) {
        return Err(Layer2EncodeRefusal {
            anchor: item.anchor.clone(),
            reason: format!("outside the Layer-2 empirical profile: {e}"),
        });
    }

    // Bind each role⊗filler, then bundle (superpose) and sign.
    let binds: Vec<Vec<f64>> = pairs
        .iter()
        .map(|(role, filler)| {
            let r = atom(role, L2_DIM);
            let f = atom(filler, L2_DIM);
            // Both operands are exactly L2_DIM bipolar atoms, so bind cannot dim-mismatch; surface any
            // impossible error as a refusal rather than an unwrap (never-silent).
            model.bind(&r, &f)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| Layer2EncodeRefusal {
            anchor: item.anchor.clone(),
            reason: format!("bind failed while encoding record: {e}"),
        })?;

    let refs: Vec<&[f64]> = binds.iter().map(Vec::as_slice).collect();
    let bundled = model.bundle(&refs).map_err(|e| Layer2EncodeRefusal {
        anchor: item.anchor.clone(),
        reason: format!("bundle failed while encoding record: {e}"),
    })?;
    let record = sign_bipolar(&bundled);

    // The checked-instantiation fact: is a Proven capacity bound available for this m at this dim?
    let proven = capacity::proven_capacity_bound(m as u64, u64::from(L2_DIM), L2_DELTA).is_some();

    Ok((
        record,
        RecordStats {
            n_terms: m,
            truncated,
            proven,
        },
    ))
}

/// Encode a free-text query into a probe hypervector: `bundle` over each query term of
/// `TITLE_TERM ⊗ atom(term)` **and** `SUMMARY_TERM ⊗ atom(term)` — so whichever field a record used
/// for that term aligns under cleanup similarity. Returns `None` when the query tokenizes to zero
/// terms (the caller turns that into a typed [`super::decode::Layer2Refusal::EmptyQuery`], mirroring
/// the Layer-1 empty-text refusal — never a silent empty probe).
#[must_use]
pub fn encode_query(text: &str, model: &MapI) -> Option<(Vec<f64>, Vec<String>)> {
    let terms = tokenize(text);
    if terms.is_empty() {
        return None;
    }
    let mut binds: Vec<Vec<f64>> = Vec::with_capacity(terms.len() * 2);
    for t in &terms {
        let ft = atom(t, L2_DIM);
        // bind is infallible for two equal-length atoms; if it ever errored we would rather drop that
        // term than fabricate one — but at L2_DIM it cannot, so an error here is a real invariant break.
        if let Ok(b) = model.bind(&atom(ROLE_TITLE_TERM, L2_DIM), &ft) {
            binds.push(b);
        }
        if let Ok(b) = model.bind(&atom(ROLE_SUMMARY_TERM, L2_DIM), &ft) {
            binds.push(b);
        }
    }
    let refs: Vec<&[f64]> = binds.iter().map(Vec::as_slice).collect();
    let probe = model.bundle(&refs).ok()?;
    Some((probe, terms))
}

/// Build the whole Layer-2 codebook from a Layer-1 report: encode every row, insert the survivors
/// into a [`CleanupMemory`] keyed by `anchor`, and collect the never-silent refusals + honest stats.
#[must_use]
pub fn build_codebook(report: &TeroIndexReport, model: &MapI) -> BuildOutput {
    let mut memory = CleanupMemory::new(L2_DIM);
    let mut refused = Vec::new();
    let mut truncations = 0usize;
    let mut proven_records = 0usize;
    let mut max_terms = 0usize;

    for item in &report.items {
        match encode_record(item, model) {
            Ok((record, stats)) => {
                if let Err(e) = memory.insert(item.anchor.clone(), record) {
                    refused.push(Layer2EncodeRefusal {
                        anchor: item.anchor.clone(),
                        reason: format!("cleanup insert failed: {e}"),
                    });
                    continue;
                }
                if stats.truncated {
                    truncations += 1;
                }
                if stats.proven {
                    proven_records += 1;
                }
                max_terms = max_terms.max(stats.n_terms);
            }
            Err(refusal) => refused.push(refusal),
        }
    }

    // Family is unused past encoding, but keep a debug-only sanity tie so a future refactor that drops
    // a family from the codebook is noticed (all five families should be representable).
    debug_assert!(
        report.items.is_empty()
            || report
                .items
                .iter()
                .any(|i| matches!(i.family, Family::Issue | Family::Doc)),
        "a non-empty corpus should carry at least one issue/doc row"
    );

    BuildOutput {
        memory,
        refused,
        truncations,
        proven_records,
        max_terms,
    }
}
