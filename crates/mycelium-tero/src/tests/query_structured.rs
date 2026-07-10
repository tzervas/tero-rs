//! Structured queries (`Query::Id`/`Status`/`Kind`) — exact-match lookups, the mandatory-provenance
//! invariant (every `Answer` carries a resolvable citation), the refusal path, and the EXPLAIN
//! trace's "why in what order" for a non-ranked query.

use crate::tests::fixture::{temp_dir, write_corpus};
use crate::{build_tero_index, Query, QueryEngine, Refusal};

#[test]
fn by_id_finds_the_document_row_with_a_resolvable_citation() {
    let root = temp_dir("q-id");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine.run(&Query::Id("RFC-0099".to_owned())).unwrap();
    assert_eq!(answer.items().len(), 1);
    let citation = &answer.citations()[0];
    assert_eq!(citation.id.as_deref(), Some("RFC-0099"));
    assert_eq!(citation.file, "docs/rfcs/RFC-0099-Test.md");
    assert!(citation.line >= 1);
    assert_eq!(citation.item_tag, crate::ITEM_TAG);
    // RFC-0099's fixture body declares `Proven` — the cited claim's own guarantee, distinct from
    // the row's uniform extraction-honesty `item_tag`.
    assert_eq!(citation.guarantee_tag.as_deref(), Some("Proven"));

    let explain = answer.explain();
    assert_eq!(explain.candidates_matched, 1);
    assert!(explain
        .order_by
        .iter()
        .any(|s| s.contains("canonical index order")));
    assert_eq!(explain.hits.len(), 1);
    assert_eq!(explain.hits[0].anchor, citation.anchor);
}

#[test]
fn by_id_returns_every_duplicate_never_a_silently_deduped_one() {
    // The fixture's `defects=true` corpus has a duplicate `M-0099` issue id (the union-merge
    // hazard M-1015 already flags) — the query layer must not silently pick one; both are citable.
    // The fixture's `CHANGELOG.md` also has a `### M-0099 — …` entry (present regardless of
    // `defects`), whose id is independently extracted by `changelog::leading_id` — so `by_id`
    // legitimately returns *three* rows here, spanning two families. That is not a bug: an id is a
    // cross-corpus identifier, and surfacing every family's row for it (the issue *and* the
    // changelog entry that recorded it) is exactly the "what did we decide about X, across which
    // issues/changelog entries" cross-cutting lookup DN-87 §1 asks for.
    let root = temp_dir("q-id-dup");
    write_corpus(&root, true);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine.run(&Query::Id("M-0099".to_owned())).unwrap();
    assert_eq!(answer.items().len(), 3);
    assert_eq!(answer.citations().len(), 3);
    let by_family: Vec<crate::Family> = answer.items().iter().map(|it| it.family).collect();
    assert_eq!(
        by_family
            .iter()
            .filter(|f| **f == crate::Family::Issue)
            .count(),
        2,
        "both duplicate issue rows must survive: {by_family:?}"
    );
    assert_eq!(
        by_family
            .iter()
            .filter(|f| **f == crate::Family::Changelog)
            .count(),
        1,
        "the changelog entry recording M-0099 must also be citable: {by_family:?}"
    );
}

#[test]
fn by_id_with_no_match_refuses_with_the_scanned_count() {
    let root = temp_dir("q-id-miss");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let err = engine.run(&Query::Id("NOPE-0000".to_owned())).unwrap_err();
    match &err {
        Refusal::NoMatch {
            query,
            candidates_scanned,
        } => {
            assert!(query.contains("NOPE-0000"));
            assert_eq!(*candidates_scanned, report.items.len());
        }
        other => panic!("expected NoMatch, got {other:?}"),
    }
    // A Refusal is itself explainable via Display — never a bare/opaque error.
    let msg = err.to_string();
    assert!(msg.contains("refusing") && msg.contains("NOPE-0000"));
}

#[test]
fn by_status_is_case_insensitive_and_finds_every_matching_row() {
    let root = temp_dir("q-status");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine.run(&Query::Status("accepted".to_owned())).unwrap();
    assert!(answer
        .items()
        .iter()
        .any(|it| it.id.as_deref() == Some("RFC-0099")));
    for it in answer.items() {
        assert_eq!(
            it.status.as_deref().map(str::to_lowercase).as_deref(),
            Some("accepted")
        );
    }
}

#[test]
fn by_status_with_no_match_refuses() {
    let root = temp_dir("q-status-miss");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let err = engine
        .run(&Query::Status("nonexistent-status".to_owned()))
        .unwrap_err();
    assert!(matches!(err, Refusal::NoMatch { .. }));
}

#[test]
fn by_kind_is_case_insensitive_and_matches_every_row_of_that_kind() {
    let root = temp_dir("q-kind");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine.run(&Query::Kind("SECTION".to_owned())).unwrap();
    assert!(!answer.items().is_empty());
    for it in answer.items() {
        assert_eq!(it.kind, "section");
    }
    // Every hit is an equally-exact kind match — no ranking signal, all scores 0.
    assert!(answer.explain().hits.iter().all(|h| h.score == 0));
}

#[test]
fn by_kind_with_no_match_refuses() {
    let root = temp_dir("q-kind-miss");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let err = engine
        .run(&Query::Kind("no-such-kind".to_owned()))
        .unwrap_err();
    assert!(matches!(err, Refusal::NoMatch { .. }));
}
