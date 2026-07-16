//! `Query::Text` — deterministic ranked search over `id`/`title`/`summary`, its EXPLAIN trace, and
//! the refusal path (including the empty/whitespace-only query case).

use crate::query::score_text;
use crate::tests::fixture::{temp_dir, write_corpus};
use crate::{build_tero_index, Query, QueryEngine, Refusal, TeroIndexItem};

#[test]
fn a_multi_term_match_ranks_the_stronger_hit_first() {
    let root = temp_dir("q-text-rank");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    // "rfc" matches RFC-0099's id AND title; "test" matches several titles. RFC-0099 must
    // therefore outrank a title-only match like ADR-099.
    let answer = engine.run(&Query::Text("test rfc".to_owned())).unwrap();
    assert!(!answer.items().is_empty());
    assert_eq!(answer.items()[0].id.as_deref(), Some("RFC-0099"));

    let hits = &answer.explain().hits;
    assert_eq!(hits[0].anchor, answer.items()[0].anchor);
    // Scores are non-increasing across the ranked hits (descending sort, ties broken canonically).
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(hits[0].why.contains("rfc"));
}

#[test]
fn candidates_matched_reports_the_pre_cap_total() {
    let root = temp_dir("q-text-cap");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let answer = engine.run(&Query::Text("test".to_owned())).unwrap();
    // candidates_matched is never less than the number of items actually returned.
    assert!(answer.explain().candidates_matched >= answer.items().len());
    assert_eq!(answer.explain().candidates_scanned, report.items.len());
}

#[test]
fn no_match_refuses_with_the_scanned_count() {
    let root = temp_dir("q-text-miss");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let err = engine
        .run(&Query::Text("zzz-no-such-term-anywhere".to_owned()))
        .unwrap_err();
    match &err {
        Refusal::NoTextMatch {
            query,
            candidates_scanned,
        } => {
            assert!(query.contains("zzz-no-such-term-anywhere"));
            assert_eq!(*candidates_scanned, report.items.len());
        }
        other => panic!("expected NoTextMatch, got {other:?}"),
    }
}

#[test]
fn an_empty_or_whitespace_only_query_refuses_rather_than_matching_everything() {
    let root = temp_dir("q-text-empty");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let err = engine.run(&Query::Text("   ".to_owned())).unwrap_err();
    assert!(matches!(err, Refusal::NoTextMatch { .. }));
}

#[test]
fn two_runs_over_the_same_report_produce_byte_identical_explain_output() {
    // The determinism contract (DN-87 §6.3) applied to ranking: no clock/rng in the scorer.
    let root = temp_dir("q-text-det");
    write_corpus(&root, false);
    let report = build_tero_index(&root).unwrap();
    let engine = QueryEngine::new(&report);

    let a = engine.run(&Query::Text("test".to_owned())).unwrap();
    let b = engine.run(&Query::Text("test".to_owned())).unwrap();
    assert_eq!(a.explain().hits, b.explain().hits);
    assert_eq!(a.items(), b.items());
}

// ── direct unit tests of the scorer (the trickiest pure function here) ────────────────────────

fn item(id: Option<&str>, title: &str, summary: Option<&str>) -> TeroIndexItem {
    let mut it = TeroIndexItem::new("anchor", crate::Family::Doc, "other", title, "f.md", 1);
    it.id = id.map(str::to_owned);
    it.summary = summary.map(str::to_owned);
    it
}

#[test]
fn score_text_weights_id_over_title_over_summary() {
    let terms = vec!["rfc".to_owned()];
    let (id_score, _) = score_text(&item(Some("RFC-0034"), "unrelated", None), &terms);
    let (title_score, _) = score_text(&item(None, "an RFC in the title", None), &terms);
    let (summary_score, _) =
        score_text(&item(None, "unrelated", Some("mentions rfc here")), &terms);
    assert!(id_score > title_score);
    assert!(title_score > summary_score);
    assert!(summary_score > 0);
}

#[test]
fn score_text_is_case_insensitive_and_zero_without_any_match() {
    let terms = vec!["proven".to_owned()];
    let (score, why) = score_text(&item(None, "A PROVEN result", None), &terms);
    assert!(score > 0 && why.contains("title"));

    let (score, why) = score_text(&item(None, "no relation", None), &terms);
    assert_eq!(score, 0);
    assert!(why.is_empty());
}
