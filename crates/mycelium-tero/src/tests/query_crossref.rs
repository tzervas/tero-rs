//! `Query::CrossRef` — the `depends_on`/`doc_refs` breadth-first walk: hop ordering, `corpus:` ref
//! resolution, unresolved (`src:`/missing-target) edges recorded rather than dropped, depth
//! clamping, and the `UnknownAnchor` refusal.
//!
//! Fixture wiring this exercises (`fixture::write_corpus`, `defects=false`): the issue `E99-1`
//! `depends_on: [M-0099]`; `M-0099` `depends_on: [M-0001, M-0002]` (neither indexed — unresolved)
//! and `doc_refs: [corpus:RFC-0099, src:crates/mycelium-tero/src/lib.rs]` (the first resolves to the
//! RFC-0099 document row, the second is an unresolvable `src:` reference).

use crate::query::resolve_doc_ref;
use crate::tests::fixture::{temp_dir, write_corpus};
use crate::{
    build_tero_index, Family, Query, QueryEngine, Refusal, TeroIndexItem, TeroIndexReport,
};

#[test]
fn depth_zero_returns_only_the_start_node() {
    let root = temp_dir("q-xref-d0");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine
        .run(&Query::CrossRef {
            start: "E99-1".to_owned(),
            depth: 0,
        })
        .unwrap();
    assert_eq!(answer.items().len(), 1);
    assert_eq!(answer.items()[0].id.as_deref(), Some("E99-1"));
    assert_eq!(answer.explain().hits[0].why, "start node");
}

#[test]
fn depth_one_follows_depends_on_one_hop() {
    let root = temp_dir("q-xref-d1");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine
        .run(&Query::CrossRef {
            start: "E99-1".to_owned(),
            depth: 1,
        })
        .unwrap();
    let ids: Vec<Option<&str>> = answer.items().iter().map(|it| it.id.as_deref()).collect();
    assert!(ids.contains(&Some("E99-1")));
    assert!(ids.contains(&Some("M-0099")));
    assert_eq!(ids.len(), 2);
    // Closest-first ordering: the start node ranks before its one-hop neighbor.
    assert_eq!(answer.items()[0].id.as_deref(), Some("E99-1"));
    assert_eq!(answer.items()[1].id.as_deref(), Some("M-0099"));
}

#[test]
fn depth_two_reaches_the_doc_refs_target_and_records_unresolved_edges() {
    let root = temp_dir("q-xref-d2");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine
        .run(&Query::CrossRef {
            start: "E99-1".to_owned(),
            depth: 2,
        })
        .unwrap();
    let ids: Vec<Option<&str>> = answer.items().iter().map(|it| it.id.as_deref()).collect();
    assert!(ids.contains(&Some("E99-1")));
    assert!(ids.contains(&Some("M-0099")));
    assert!(
        ids.contains(&Some("RFC-0099")),
        "corpus:RFC-0099 doc_refs must resolve: {ids:?}"
    );

    let explain = answer.explain();
    // M-0099's depends_on [M-0001, M-0002] (neither indexed) + its src: doc_ref are all unresolved.
    assert!(
        explain
            .unresolved_edges
            .iter()
            .any(|e| e.contains("M-0001") && e.contains("depends_on")),
        "{:?}",
        explain.unresolved_edges
    );
    assert!(
        explain
            .unresolved_edges
            .iter()
            .any(|e| e.contains("M-0002") && e.contains("depends_on")),
        "{:?}",
        explain.unresolved_edges
    );
    assert!(
        explain
            .unresolved_edges
            .iter()
            .any(|e| e.contains("src:crates/mycelium-tero/src/lib.rs")),
        "{:?}",
        explain.unresolved_edges
    );
    assert_eq!(explain.candidates_matched, 3);
}

#[test]
fn depth_is_clamped_and_the_clamp_is_recorded_in_explain() {
    let root = temp_dir("q-xref-clamp");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine
        .run(&Query::CrossRef {
            start: "E99-1".to_owned(),
            depth: 999,
        })
        .unwrap();
    // Never-silent: the requested depth is visible in the query description even though it was
    // clamped to the hard cap.
    assert!(answer.explain().query.contains("999"));
    assert!(answer.explain().query.contains("clamped"));
}

#[test]
fn an_unknown_start_refuses_rather_than_returning_an_empty_walk() {
    let root = temp_dir("q-xref-unknown");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let err = engine
        .run(&Query::CrossRef {
            start: "NOPE-404".to_owned(),
            depth: 3,
        })
        .unwrap_err();
    match &err {
        Refusal::UnknownAnchor { start, .. } => assert_eq!(start, "NOPE-404"),
        other => panic!("expected UnknownAnchor, got {other:?}"),
    }
}

#[test]
fn start_with_no_resolvable_edges_is_still_a_citable_answer_not_a_refusal() {
    // RFC-0099 (a doc row) carries no depends_on/doc_refs of its own in the M-1015 model — a walk
    // from it is a legitimate one-node answer ("this exists; it has no further modeled edges"), not
    // a refusal (the start node itself is always a resolvable citation).
    let root = temp_dir("q-xref-leaf");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine
        .run(&Query::CrossRef {
            start: "RFC-0099".to_owned(),
            depth: 3,
        })
        .unwrap();
    assert_eq!(answer.items().len(), 1);
    assert_eq!(answer.items()[0].id.as_deref(), Some("RFC-0099"));
    assert!(answer.explain().unresolved_edges.is_empty());
}

// ── direct unit tests of `resolve_doc_ref` (the trickiest pure function here) ──────────────────

fn doc_item(id: &str, anchor: &str) -> TeroIndexItem {
    let mut it = TeroIndexItem::new(anchor, Family::Doc, "rfc", id, "docs/rfcs/x.md", 1);
    it.id = Some(id.to_owned());
    it
}

fn section_item(anchor: &str) -> TeroIndexItem {
    TeroIndexItem::new(
        anchor,
        Family::Doc,
        "section",
        "A Section",
        "docs/rfcs/x.md",
        10,
    )
}

fn issue_item(id: &str, anchor: &str, depends_on: &[&str]) -> TeroIndexItem {
    let mut it = TeroIndexItem::new(
        anchor,
        Family::Issue,
        "issue",
        id,
        "tools/github/issues.yaml",
        1,
    );
    it.id = Some(id.to_owned());
    it.depends_on = depends_on.iter().map(|s| (*s).to_owned()).collect();
    it
}

#[test]
fn resolve_doc_ref_resolves_a_bare_corpus_ref_to_the_document_row() {
    let report = TeroIndexReport {
        items: vec![doc_item("RFC-0034", "rfc-0034")],
        flagged: Vec::new(),
    };
    let found = resolve_doc_ref(&report, "corpus:RFC-0034").unwrap();
    assert_eq!(found.id.as_deref(), Some("RFC-0034"));
}

#[test]
fn resolve_doc_ref_resolves_a_fragment_to_the_namespaced_section_anchor() {
    let report = TeroIndexReport {
        items: vec![
            doc_item("RFC-0034", "rfc-0034"),
            section_item("rfc-0034--a-section"),
        ],
        flagged: Vec::new(),
    };
    let found = resolve_doc_ref(&report, "corpus:RFC-0034#a-section").unwrap();
    assert_eq!(found.anchor, "rfc-0034--a-section");
}

#[test]
fn resolve_doc_ref_falls_back_to_a_prefix_match_for_a_deduped_section_anchor() {
    // `AnchorAlloc` suffixes a heading-slug collision with `-2`; a bare-fragment doc_ref (the
    // `_heading_slug` form `doc_refs_check.py` validates against) should still find it.
    let report = TeroIndexReport {
        items: vec![
            doc_item("RFC-0034", "rfc-0034"),
            section_item("rfc-0034--a-section-2"),
        ],
        flagged: Vec::new(),
    };
    let found = resolve_doc_ref(&report, "corpus:RFC-0034#a-section").unwrap();
    assert_eq!(found.anchor, "rfc-0034--a-section-2");
}

#[test]
fn resolve_doc_ref_returns_none_for_api_and_src_refs_never_a_wrong_guess() {
    let report = TeroIndexReport {
        items: vec![doc_item("RFC-0034", "rfc-0034")],
        flagged: Vec::new(),
    };
    assert!(resolve_doc_ref(&report, "api:mycelium-core::foo::bar").is_none());
    assert!(resolve_doc_ref(&report, "src:crates/mycelium-tero/src/lib.rs").is_none());
    assert!(resolve_doc_ref(&report, "corpus:RFC-9999").is_none());
}

#[test]
fn resolve_doc_ref_refuses_a_sibling_section_that_merely_extends_the_prefix() {
    // Regression (M-1016 review): `rfc-0034--determinism` is absent, but
    // `rfc-0034--determinism-details` — an unrelated SIBLING section whose own slug happens to
    // extend the fragment — is present. `AnchorAlloc::alloc` (mycelium-doc/src/corpus.rs) only ever
    // produces dedup suffixes of the exact form `-<digits>`, never free text, so this must NOT
    // resolve via the old unrestricted `starts_with` fallback — it must refuse (`None`), recorded
    // by the caller as an unresolved edge, not silently cite the wrong section.
    let report = TeroIndexReport {
        items: vec![
            doc_item("RFC-0034", "rfc-0034"),
            section_item("rfc-0034--determinism-details"),
        ],
        flagged: Vec::new(),
    };
    assert!(
        resolve_doc_ref(&report, "corpus:RFC-0034#determinism").is_none(),
        "must refuse rather than mis-cite the sibling `-details` section"
    );
}

#[test]
fn resolve_doc_ref_still_resolves_the_legitimate_exact_dedup_suffix() {
    // The real allocator-produced dedup case (`-<digits>`, not free text) must keep resolving.
    let report = TeroIndexReport {
        items: vec![
            doc_item("RFC-0034", "rfc-0034"),
            section_item("rfc-0034--determinism-2"),
        ],
        flagged: Vec::new(),
    };
    let found = resolve_doc_ref(&report, "corpus:RFC-0034#determinism").unwrap();
    assert_eq!(found.anchor, "rfc-0034--determinism-2");
}

#[test]
fn resolve_doc_ref_refuses_when_more_than_one_dedup_candidate_matches() {
    // Two allocator-shaped dedup suffixes both match the same prefix and neither is the exact
    // (unsuffixed) anchor — genuinely ambiguous which one the ref meant, so this refuses rather
    // than arbitrarily picking the first (G2: never guess at a citation).
    let report = TeroIndexReport {
        items: vec![
            doc_item("RFC-0034", "rfc-0034"),
            section_item("rfc-0034--determinism-2"),
            section_item("rfc-0034--determinism-3"),
        ],
        flagged: Vec::new(),
    };
    assert!(
        resolve_doc_ref(&report, "corpus:RFC-0034#determinism").is_none(),
        "two matching dedup candidates must refuse, not pick arbitrarily"
    );
}

// ── cross-reference BFS cycle safety ────────────────────────────────────────────────────────────

#[test]
fn cross_ref_walk_terminates_on_a_genuine_depends_on_cycle_visiting_each_node_once() {
    // A real back-edge cycle: CYC-A depends_on CYC-B depends_on CYC-A. The BFS's `hop` BTreeMap IS
    // the visited set (query.rs's `cross_ref`) — an edge back to an already-reached node hits the
    // `Some(_) => {}` "already reached" arm rather than re-enqueuing, so the walk must terminate and
    // each node must appear exactly once, never loop or duplicate.
    let mut report = TeroIndexReport {
        items: vec![
            issue_item("CYC-A", "cyc-a", &["CYC-B"]),
            issue_item("CYC-B", "cyc-b", &["CYC-A"]),
        ],
        flagged: Vec::new(),
    };
    report.sort();
    let engine = QueryEngine::new(&report);

    let answer = engine
        .run(&Query::CrossRef {
            start: "CYC-A".to_owned(),
            depth: 6, // MAX_CROSSREF_DEPTH — a cycle must not spin past the cap either
        })
        .unwrap();

    let ids: Vec<Option<&str>> = answer.items().iter().map(|it| it.id.as_deref()).collect();
    assert_eq!(
        ids.len(),
        2,
        "each cycle node must appear exactly once: {ids:?}"
    );
    assert!(ids.contains(&Some("CYC-A")));
    assert!(ids.contains(&Some("CYC-B")));
    assert!(
        answer.explain().unresolved_edges.is_empty(),
        "both depends_on targets exist in the index: {:?}",
        answer.explain().unresolved_edges
    );
}
