//! The query engine over the Layer-1 model (M-1016 / DN-87 §4/§6) — structured lookups
//! (`id`/`status`/`kind`), a cross-reference walk over `depends_on`/`doc_refs` edges, and a ranked
//! text search, every one returning either an [`Answer`] that carries ≥1 resolvable [`Citation`] or
//! a typed [`Refusal`] that says *why* nothing citable was found. There is no third outcome: **an
//! answer without a resolvable citation cannot be constructed** — [`Answer`]'s fields are private and
//! the only code paths that build one (the `by_*`/`cross_ref`/`text` functions below) refuse instead
//! of returning an empty one. That is the never-silent rule (G2) applied to retrieval, per DN-87 §6.2:
//! *"an answer without a resolvable citation is a refusal, not an answer."*
//!
//! **EXPLAIN-able (G2/DN-87 §4):** every [`Answer`] carries an [`Explain`] trace — the candidate
//! count, the ordering rule applied, and a per-hit reason — so "why these sources, in what order" is
//! always inspectable, not just for the ranked text search. Ordering is a pure function of the query
//! and the (already-sorted) report: no clock, no rng, no hidden state, so two runs over the same
//! report produce byte-identical `Explain` output (the same determinism contract [`crate::index`]
//! documents for the build itself).
//!
//! **Scope note (a resolved spec ambiguity):** the M-1016 issue body describes the cross-reference
//! walk as following "`depends_on` / `doc_refs` / `supersedes` edges". The M-1015 [`TeroIndexItem`]
//! model does not carry a structured `supersedes` field — an ADR/RFC/DN's superseding relationship is
//! today only prose inside the document body (its `Status` cell says `Superseded`, but not
//! *by what*). Inventing a `supersedes` edge here would mean guessing at unindexed data, which G2
//! forbids; extracting one is M-1015 extractor work, out of this issue's scope (a query layer queries
//! the model M-1015 built, it does not grow it). So [`Query::CrossRef`] walks exactly the two edge
//! kinds the model carries — `depends_on` and `doc_refs` — and this gap is recorded here rather than
//! silently narrowing the DoD.
//!
//! **KC-3 (small, auditable kernel):** everything in this module is `pub` (re-exported from the
//! crate root) except the per-query-kind implementation functions, which stay private — a future
//! front (M-1017) drives the engine through [`QueryEngine::run`] and reads [`Answer`]/[`Refusal`],
//! never the internals.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::model::{Family, TeroIndexItem, TeroIndexReport};

/// Hard cap on [`Query::CrossRef`]'s `depth` — a defence against a pathological request walking the
/// whole graph in one query. A request above the cap is silently clamped **in behavior** but never
/// silently in *report*: the clamp is recorded in the returned [`Explain::query`] string (G2).
const MAX_CROSSREF_DEPTH: usize = 6;

/// Result cap for [`Query::Text`] — the engine answers "the best matches", not "every match".
/// [`Explain::candidates_matched`] always carries the pre-cap match total, so a caller can tell a
/// result set was truncated rather than exhaustive (never silently — G2).
const TEXT_RESULT_LIMIT: usize = 20;

// ── the query engine ────────────────────────────────────────────────────────────────────────────

/// A read-only query engine over a [`TeroIndexReport`] (typically loaded via
/// [`crate::load::load_report`], or produced fresh by [`crate::build_tero_index`]).
///
/// # Invariant
/// The report's `items` must already be in the canonical `(family, file, line, anchor)` order
/// ([`TeroIndexReport::sort`] — every report [`crate::build_tero_index`] or
/// [`crate::load::load_report`] returns already satisfies this). Every structured query's
/// `order_by = "canonical index order"` claim depends on it; a debug build asserts it at
/// construction so a broken invariant fails loudly in tests, not silently in production ranking.
pub struct QueryEngine<'a> {
    report: &'a TeroIndexReport,
}

impl<'a> QueryEngine<'a> {
    /// Build an engine over `report`. See the struct docs for the sorted-report precondition.
    #[must_use]
    pub fn new(report: &'a TeroIndexReport) -> Self {
        debug_assert!(
            is_canonically_sorted(report),
            "QueryEngine::new requires a TeroIndexReport already in canonical (family, file, line, \
             anchor) order (TeroIndexReport::sort) — every `order_by: canonical index order` claim \
             an Explain trace makes depends on this"
        );
        QueryEngine { report }
    }

    /// Run one query, returning either a citation-carrying [`Answer`] or a typed [`Refusal`].
    pub fn run(&self, query: &Query) -> Result<Answer, Refusal> {
        match query {
            Query::Id(id) => by_id(self.report, id),
            Query::Status(status) => by_status(self.report, status),
            Query::Kind(kind) => by_kind(self.report, kind),
            Query::CrossRef { start, depth } => cross_ref(self.report, start, *depth),
            Query::Text(q) => text(self.report, q),
        }
    }
}

// ── the query vocabulary ────────────────────────────────────────────────────────────────────────

/// A structured or free-text query over the Layer-1 model (DN-87 §4: "structured queries (by id,
/// status, kind, cross-reference walk) + text search").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Query {
    /// Exact match on [`TeroIndexItem::id`] (`"RFC-0034"`, `"M-1015"`, `"DN-87"`, an issue id). A
    /// duplicate id (a recorded corpus defect — see `issues.rs`'s union-merge-hazard flag) returns
    /// every matching row; never a silently-deduped one.
    Id(String),
    /// Case-insensitive exact match on [`TeroIndexItem::status`] (`"Accepted"`, `"todo"`, `"done"`,
    /// …).
    Status(String),
    /// Case-insensitive exact match on [`TeroIndexItem::kind`] (`"rfc"`, `"issue"`, `"section"`, …).
    Kind(String),
    /// A breadth-first walk of `depends_on`/`doc_refs` edges starting at the item whose `id` or
    /// `anchor` equals `start`, out to `depth` hops (clamped to a hard internal cap — never
    /// silently: the returned [`Explain::query`] states whether clamping happened). Only issue rows
    /// carry outgoing edges in the M-1015 model, so a walk fans out from an issue and terminates at
    /// the docs/issues it cites (see the module docs' `supersedes` scope note).
    CrossRef {
        /// The starting node's id or anchor.
        start: String,
        /// Hop count to walk (`0` = the start node alone).
        depth: usize,
    },
    /// A free-text search over `id`/`title`/`summary`, ranked by a deterministic term-match score
    /// (the returned [`Answer::explain`]'s [`Explain::order_by`] states the exact weighting), capped
    /// to the top matches.
    Text(String),
}

// ── provenance types ────────────────────────────────────────────────────────────────────────────

/// A resolvable citation to one Layer-1 row — the atomic unit of provenance every [`Answer`] is
/// built from.
///
/// Two honesty fields, deliberately not conflated (VR-5): `item_tag` is the row's uniform
/// *extraction* honesty ([`crate::model::ITEM_TAG`] — how much to trust that the row was captured
/// correctly from source); `guarantee_tag` is the *cited claim's own* declared strength where the
/// source states one (a doc's `| **Guarantee** |` row) — `None` where the source declares none,
/// never invented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Citation {
    /// The row's stable, globally-unique anchor (deep-link/citation key).
    pub anchor: String,
    /// The source's own id where it has one (`RFC-0034`, `M-1015`, …).
    pub id: Option<String>,
    /// The corpus family.
    pub family: Family,
    /// The family-specific kind (`rfc`, `issue`, `section`, …).
    pub kind: String,
    /// Repo-relative source path.
    pub file: String,
    /// 1-based source line.
    pub line: u32,
    /// This row's extraction-honesty tag ([`crate::model::ITEM_TAG`] — uniform across every row).
    pub item_tag: String,
    /// The cited claim's own declared guarantee tag, where the source states one.
    pub guarantee_tag: Option<String>,
}

impl From<&TeroIndexItem> for Citation {
    fn from(it: &TeroIndexItem) -> Self {
        Citation {
            anchor: it.anchor.clone(),
            id: it.id.clone(),
            family: it.family,
            kind: it.kind.clone(),
            file: it.file.clone(),
            line: it.line,
            item_tag: it.tag.clone(),
            guarantee_tag: it.guarantee_tag.clone(),
        }
    }
}

/// One item's place in an [`Explain`] trace: its final rank position is implicit in
/// [`Explain::hits`]'s order; `score`/`why` say *why* it landed there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RankedHit {
    /// The row's anchor (matches the corresponding [`Answer::items`] entry).
    pub anchor: String,
    /// A deterministic ranking score (higher = ranked earlier); structured (non-ranked) queries use
    /// a constant `0` since every hit is an equally-exact match — `order_by` explains their order
    /// instead.
    pub score: i64,
    /// A human-readable reason this row matched / is positioned here.
    pub why: String,
}

/// The EXPLAIN trace for an [`Answer`]: the candidate universe, how many matched, the ordering
/// rule(s) applied (outermost first), any edges that could not be resolved within Layer 1, and a
/// per-hit reason — "why these sources, in what order" (DN-87 §4), for every query kind, not only
/// ranked ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Explain {
    /// A human-readable rendering of the query as executed (including any clamping applied).
    pub query: String,
    /// How many rows (or, for [`Query::CrossRef`], edges) were scanned to build the candidate set.
    pub candidates_scanned: usize,
    /// How many candidates matched before any result-count cap was applied.
    pub candidates_matched: usize,
    /// The ordering rule(s) applied, outermost first.
    pub order_by: Vec<String>,
    /// Per-result explanation, in the answer's final order (parallel to [`Answer::items`]).
    pub hits: Vec<RankedHit>,
    /// [`Query::CrossRef`] only: edges considered but not resolvable within Layer 1 (an `api:`/`src:`
    /// doc_ref, a `depends_on` id with no matching issue, …) — recorded, never silently dropped
    /// (G2). Always empty for the other query kinds.
    pub unresolved_edges: Vec<String>,
}

/// An answer to a [`Query`]: a non-empty, ranked/ordered set of Layer-1 rows plus the [`Explain`]
/// trace for how they were selected and ordered.
///
/// # Invariant
/// **Cannot be constructed with zero items.** `Answer`'s fields are private to this module; the
/// only functions that build one (`by_id`/`by_status`/`by_kind`/`cross_ref`/`text`, all below) check
/// for an empty result set first and return [`Refusal`] instead — so a caller can never observe an
/// `Answer` that carries no citation. This is DN-87 §6.2's "an answer without a resolvable citation
/// is a refusal, not an answer" enforced by the type, not by convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    items: Vec<TeroIndexItem>,
    explain: Explain,
}

impl Answer {
    /// The cited rows, in the answer's final order. Never empty (see the struct-level invariant).
    #[must_use]
    pub fn items(&self) -> &[TeroIndexItem] {
        &self.items
    }

    /// The resolvable citation for each item, same order as [`Answer::items`]. Never empty.
    #[must_use]
    pub fn citations(&self) -> Vec<Citation> {
        self.items.iter().map(Citation::from).collect()
    }

    /// The EXPLAIN trace for this answer's candidate set and ordering.
    #[must_use]
    pub fn explain(&self) -> &Explain {
        &self.explain
    }
}

/// A typed, never-silent "no answer" (DN-87 §6.2). Every variant carries enough to explain *why*
/// nothing citable was found — a refusal is itself EXPLAIN-able, not a bare empty result.
///
/// `Serialize` (M-1017): the API fronts render a refusal as a stable, internally-tagged JSON object
/// `{"variant":"no_match"|"unknown_anchor"|"no_text_match", …}` — the same shape over MCP and HTTP
/// (front parity). The tag key is `variant`; the per-variant fields carry the never-silent detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "variant", rename_all = "snake_case")]
pub enum Refusal {
    /// A structured lookup ([`Query::Id`]/[`Query::Status`]/[`Query::Kind`]) matched zero rows.
    NoMatch {
        /// A human-readable rendering of the query.
        query: String,
        /// How many rows were scanned (the whole report).
        candidates_scanned: usize,
    },
    /// [`Query::CrossRef`]'s `start` does not match any row's `id` or `anchor`.
    UnknownAnchor {
        /// The unresolved start value.
        start: String,
        /// How many rows were scanned looking for it.
        candidates_scanned: usize,
    },
    /// [`Query::Text`] matched zero rows (including an empty/whitespace-only query, which
    /// tokenizes to zero terms and so — consistently — matches nothing).
    NoTextMatch {
        /// The raw query string.
        query: String,
        /// How many rows were scanned.
        candidates_scanned: usize,
    },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NoMatch {
                query,
                candidates_scanned,
            } => write!(
                f,
                "refusing to answer {query} — 0 of {candidates_scanned} row(s) matched, so there is \
                 no resolvable citation to answer with"
            ),
            Refusal::UnknownAnchor {
                start,
                candidates_scanned,
            } => write!(
                f,
                "refusing to walk cross-references from {start:?} — no row with that id or anchor \
                 in the Layer-1 index ({candidates_scanned} row(s) scanned)"
            ),
            Refusal::NoTextMatch {
                query,
                candidates_scanned,
            } => write!(
                f,
                "refusing to answer text query {query:?} — 0 of {candidates_scanned} row(s) matched \
                 any query term in id/title/summary"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

// ── structured queries (id / status / kind) ────────────────────────────────────────────────────

fn by_id(report: &TeroIndexReport, id: &str) -> Result<Answer, Refusal> {
    let items: Vec<&TeroIndexItem> = report
        .items
        .iter()
        .filter(|it| it.id.as_deref() == Some(id))
        .collect();
    finish(
        report,
        items,
        format!("id == {id:?}"),
        "exact id match".to_owned(),
    )
}

fn by_status(report: &TeroIndexReport, status: &str) -> Result<Answer, Refusal> {
    let needle = status.to_lowercase();
    let items: Vec<&TeroIndexItem> = report
        .items
        .iter()
        .filter(|it| {
            it.status
                .as_deref()
                .is_some_and(|s| s.to_lowercase() == needle)
        })
        .collect();
    finish(
        report,
        items,
        format!("status == {status:?} (case-insensitive)"),
        format!("status field == {status:?}"),
    )
}

fn by_kind(report: &TeroIndexReport, kind: &str) -> Result<Answer, Refusal> {
    let needle = kind.to_lowercase();
    let items: Vec<&TeroIndexItem> = report
        .items
        .iter()
        .filter(|it| it.kind.to_lowercase() == needle)
        .collect();
    finish(
        report,
        items,
        format!("kind == {kind:?} (case-insensitive)"),
        format!("kind field == {kind:?}"),
    )
}

/// Shared finisher for the three exact-match structured queries: refuse on an empty match set,
/// otherwise build the `Answer` in the report's existing (already-canonical) order — no re-sort
/// needed, since the filter preserves input order and [`QueryEngine::new`]'s precondition already
/// guarantees that order is canonical.
fn finish(
    report: &TeroIndexReport,
    items: Vec<&TeroIndexItem>,
    query_desc: String,
    why: String,
) -> Result<Answer, Refusal> {
    if items.is_empty() {
        return Err(Refusal::NoMatch {
            query: query_desc,
            candidates_scanned: report.items.len(),
        });
    }
    let hits = items
        .iter()
        .map(|it| RankedHit {
            anchor: it.anchor.clone(),
            score: 0,
            why: why.clone(),
        })
        .collect();
    let explain = Explain {
        query: query_desc,
        candidates_scanned: report.items.len(),
        candidates_matched: items.len(),
        order_by: vec![
            "canonical index order (family, file, line, anchor) — every match is an equally exact \
             hit, so no ranking signal is applied"
                .to_owned(),
        ],
        hits,
        unresolved_edges: Vec::new(),
    };
    Ok(Answer {
        items: items.into_iter().cloned().collect(),
        explain,
    })
}

// ── cross-reference walk ────────────────────────────────────────────────────────────────────────

fn cross_ref(
    report: &TeroIndexReport,
    start: &str,
    requested_depth: usize,
) -> Result<Answer, Refusal> {
    let depth = requested_depth.min(MAX_CROSSREF_DEPTH);
    let Some(start_item) = find_by_id(report, start).or_else(|| find_by_anchor(report, start))
    else {
        return Err(Refusal::UnknownAnchor {
            start: start.to_owned(),
            candidates_scanned: report.items.len(),
        });
    };

    // BFS ⇒ shortest-hop distance; `via` records the edge that first reached each anchor (a
    // deterministic "why" since the frontier is processed in the report's canonical order).
    let mut hop: BTreeMap<String, usize> = BTreeMap::new();
    let mut via: BTreeMap<String, String> = BTreeMap::new();
    hop.insert(start_item.anchor.clone(), 0);
    via.insert(start_item.anchor.clone(), "start node".to_owned());

    let mut edges_considered = 0usize;
    let mut unresolved: Vec<String> = Vec::new();
    let mut frontier = vec![start_item];

    for hop_n in 1..=depth {
        let mut next = Vec::new();
        for item in &frontier {
            for target_id in &item.depends_on {
                edges_considered += 1;
                match find_by_id(report, target_id).filter(|t| t.family == Family::Issue) {
                    Some(target) if !hop.contains_key(&target.anchor) => {
                        hop.insert(target.anchor.clone(), hop_n);
                        via.insert(
                            target.anchor.clone(),
                            format!("depends_on: {} -> {}", item.anchor, target.anchor),
                        );
                        next.push(target);
                    }
                    Some(_) => {} // already reached at an earlier (or equal) hop — shortest kept
                    None => unresolved.push(format!(
                        "{} --depends_on--> {target_id} (no issue with that id in the Layer-1 index)",
                        item.anchor
                    )),
                }
            }
            for doc_ref in &item.doc_refs {
                edges_considered += 1;
                match resolve_doc_ref(report, doc_ref) {
                    Some(target) if !hop.contains_key(&target.anchor) => {
                        hop.insert(target.anchor.clone(), hop_n);
                        via.insert(
                            target.anchor.clone(),
                            format!("doc_refs: {} -> {}", item.anchor, target.anchor),
                        );
                        next.push(target);
                    }
                    Some(_) => {}
                    None => unresolved.push(format!(
                        "{} --doc_refs--> {doc_ref} (unresolved within Layer 1 — an api:/src: \
                         reference, or a corpus: doc/anchor this index does not carry)",
                        item.anchor
                    )),
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    // Result order: hop distance ascending, then the canonical (family, file, line, anchor) key —
    // closest, most-canonical nodes first. The start node itself is always included (hop 0), so
    // this never refuses once `start` resolves: even a start node with zero resolvable outgoing
    // edges is a legitimate, citable answer ("X exists; it has no resolvable further references").
    let mut results: Vec<&TeroIndexItem> = report
        .items
        .iter()
        .filter(|it| hop.contains_key(&it.anchor))
        .collect();
    results.sort_by(|a, b| {
        hop[&a.anchor].cmp(&hop[&b.anchor]).then_with(|| {
            (a.family, &a.file, a.line, &a.anchor).cmp(&(b.family, &b.file, b.line, &b.anchor))
        })
    });

    let hits = results
        .iter()
        .map(|it| RankedHit {
            anchor: it.anchor.clone(),
            score: -(i64::try_from(hop[&it.anchor]).unwrap_or(i64::MAX)),
            why: via[&it.anchor].clone(),
        })
        .collect();

    let query_desc = if depth == requested_depth {
        format!("cross_ref(start={start:?}, depth={depth})")
    } else {
        format!("cross_ref(start={start:?}, depth={requested_depth} -> clamped to {depth})")
    };

    let explain = Explain {
        query: query_desc,
        candidates_scanned: edges_considered,
        candidates_matched: results.len(),
        order_by: vec![
            "hop distance from start, ascending".to_owned(),
            "then canonical index order (family, file, line, anchor)".to_owned(),
        ],
        hits,
        unresolved_edges: unresolved,
    };

    Ok(Answer {
        items: results.into_iter().cloned().collect(),
        explain,
    })
}

/// The first (canonical-order) row whose `id` equals `id`. Duplicate ids are already recorded in
/// the report's own `flagged` list by the M-1015 issues extractor — this lookup does not re-flag
/// them; it simply resolves to the first, deterministic match.
fn find_by_id<'a>(report: &'a TeroIndexReport, id: &str) -> Option<&'a TeroIndexItem> {
    report.items.iter().find(|it| it.id.as_deref() == Some(id))
}

/// The row whose `anchor` equals `anchor`, exactly (anchors are unique in a clean corpus —
/// `tests/anchors.rs`).
fn find_by_anchor<'a>(report: &'a TeroIndexReport, anchor: &str) -> Option<&'a TeroIndexItem> {
    report.items.iter().find(|it| it.anchor == anchor)
}

/// Resolve one `doc_refs` string (the `api:<crate>::<path>` / `corpus:<DOC>[#<anchor>]` /
/// `src:<path>[:<line>]` grammar, CLAUDE.md "`doc_refs:` grammar") to an indexed row, where
/// possible.
///
/// Only `corpus:` refs are resolvable within Layer 1 — `api:` targets a sibling index
/// ([`crate::model::SIBLING_INDICES`]) this report does not carry, and `src:` targets a raw source
/// location, not an indexed row. Both return `None` (recorded as unresolved by the caller, never
/// silently treated as "no edge").
///
/// A bare `corpus:DOC` resolves to the doc/research row whose `id == DOC`. A fragment
/// `corpus:DOC#anchor` first tries the exact composed section anchor tero allocates
/// (`{doc-anchor}--{anchor}`, `mycelium_doc::corpus::AnchorAlloc`'s namespacing), then falls back to
/// [`is_dedup_suffix_of`]'s **exact allocator suffix grammar** — never a bare
/// `starts_with` — so a sibling section whose slug merely *extends* the fragment (e.g.
/// `determinism-details` extending `determinism`) cannot be mistaken for a citation of it. If more
/// than one row satisfies the grammar the target is ambiguous and this refuses (`None`) rather than
/// guessing which one was meant — a wrong-but-confident citation is worse than an unresolved edge
/// (G2).
pub(crate) fn resolve_doc_ref<'a>(
    report: &'a TeroIndexReport,
    doc_ref: &str,
) -> Option<&'a TeroIndexItem> {
    let rest = doc_ref.strip_prefix("corpus:")?;
    let is_doc_family = |it: &&TeroIndexItem| matches!(it.family, Family::Doc | Family::Research);

    match rest.split_once('#') {
        None => report
            .items
            .iter()
            .filter(is_doc_family)
            .find(|it| it.id.as_deref() == Some(rest)),
        Some((doc_id, fragment)) => {
            let doc = report
                .items
                .iter()
                .filter(is_doc_family)
                .find(|it| it.id.as_deref() == Some(doc_id))?;
            let exact = format!("{}--{fragment}", doc.anchor);
            report
                .items
                .iter()
                .find(|it| it.anchor == exact)
                .or_else(|| {
                    let mut candidates = report
                        .items
                        .iter()
                        .filter(|it| is_dedup_suffix_of(&it.anchor, &exact));
                    let first = candidates.next()?;
                    // More than one candidate ⇒ ambiguous — refuse rather than pick arbitrarily.
                    candidates.next().is_none().then_some(first)
                })
        }
    }
}

/// True when `anchor` matches `mycelium_doc::corpus::AnchorAlloc`'s collision-dedup grammar for
/// `prefix`: either `anchor == prefix` exactly, or `anchor == "{prefix}-N"` where `N` is one or
/// more ASCII digits (`AnchorAlloc::alloc`'s `-2`, `-3`, … dedup suffixing on a heading-slug
/// collision — see `mycelium-doc/src/corpus.rs`). Deliberately
/// **not** a bare `starts_with`: an anchor that merely continues past `prefix` with anything other
/// than `-<digits>` (e.g. `{prefix}-details`, a wholly unrelated sibling section whose own slug
/// happens to extend `prefix`) does not match. This was the M-1016 review's false-citation bug — an
/// unrestricted `starts_with` fallback in [`resolve_doc_ref`] silently resolved to the wrong
/// section.
fn is_dedup_suffix_of(anchor: &str, prefix: &str) -> bool {
    match anchor.strip_prefix(prefix) {
        Some("") => true,
        Some(rest) => rest
            .strip_prefix('-')
            .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())),
        None => false,
    }
}

// ── text search ─────────────────────────────────────────────────────────────────────────────────

fn text(report: &TeroIndexReport, query_str: &str) -> Result<Answer, Refusal> {
    let mut terms: Vec<String> = Vec::new();
    for tok in query_str.split_whitespace() {
        let t = tok.to_lowercase();
        if !terms.contains(&t) {
            terms.push(t);
        }
    }
    if terms.is_empty() {
        return Err(Refusal::NoTextMatch {
            query: query_str.to_owned(),
            candidates_scanned: report.items.len(),
        });
    }

    let mut scored: Vec<(i64, String, &TeroIndexItem)> = report
        .items
        .iter()
        .filter_map(|it| {
            let (score, why) = score_text(it, &terms);
            (score > 0).then_some((score, why, it))
        })
        .collect();

    if scored.is_empty() {
        return Err(Refusal::NoTextMatch {
            query: query_str.to_owned(),
            candidates_scanned: report.items.len(),
        });
    }

    // Deterministic order: score descending, tie-broken by the canonical index key — never
    // insertion order or a hash, so two runs over the same report rank identically.
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0).then_with(|| {
            (a.2.family, &a.2.file, a.2.line, &a.2.anchor).cmp(&(
                b.2.family,
                &b.2.file,
                b.2.line,
                &b.2.anchor,
            ))
        })
    });

    let matched = scored.len();
    scored.truncate(TEXT_RESULT_LIMIT);

    let hits = scored
        .iter()
        .map(|(score, why, it)| RankedHit {
            anchor: it.anchor.clone(),
            score: *score,
            why: why.clone(),
        })
        .collect();
    let items = scored.into_iter().map(|(_, _, it)| it.clone()).collect();

    let explain = Explain {
        query: format!("text({query_str:?}) — terms {terms:?}"),
        candidates_scanned: report.items.len(),
        candidates_matched: matched,
        order_by: vec![
            "match score, descending (id match x4 + title match x3 + summary match x1, per \
             matched term)"
                .to_owned(),
            "then canonical index order (family, file, line, anchor)".to_owned(),
        ],
        hits,
        unresolved_edges: Vec::new(),
    };
    Ok(Answer { items, explain })
}

/// Case-insensitive substring scoring of `item` against `terms` — deterministic (a pure function of
/// its inputs; no clock/rng), so [`text`]'s ranking is reproducible. Weighted by field: an id match
/// is the strongest signal (ids are exact corpus identifiers), then title, then summary.
pub(crate) fn score_text(item: &TeroIndexItem, terms: &[String]) -> (i64, String) {
    let title_lc = item.title.to_lowercase();
    let id_lc = item.id.as_deref().map(str::to_lowercase);
    let summary_lc = item.summary.as_deref().map(str::to_lowercase);

    let mut score = 0i64;
    let mut why: Vec<String> = Vec::new();
    for term in terms {
        if id_lc.as_deref().is_some_and(|s| s.contains(term.as_str())) {
            score += 4;
            why.push(format!("id~{term:?}"));
        }
        if title_lc.contains(term.as_str()) {
            score += 3;
            why.push(format!("title~{term:?}"));
        }
        if summary_lc
            .as_deref()
            .is_some_and(|s| s.contains(term.as_str()))
        {
            score += 1;
            why.push(format!("summary~{term:?}"));
        }
    }
    (score, why.join(", "))
}

// ── invariant check ─────────────────────────────────────────────────────────────────────────────

/// Whether `report.items` is already in the canonical `(family, file, line, anchor)` order
/// [`TeroIndexReport::sort`] establishes ([`QueryEngine::new`]'s precondition check).
fn is_canonically_sorted(report: &TeroIndexReport) -> bool {
    report.items.windows(2).all(|w| {
        (w[0].family, &w[0].file, w[0].line, &w[0].anchor)
            <= (w[1].family, &w[1].file, w[1].line, &w[1].anchor)
    })
}
