//! White-box tests for Layer-2 **decoding** (M-1018) over the hermetic fixture corpus: a query
//! recovers the right anchor; below-threshold / empty queries are typed refusals (never-silent); and
//! a recovered anchor resolves to a real Layer-1 [`crate::Citation`] (provenance preserved by
//! construction — the DoD's "a Layer-2 answer always names its Layer-1 evidence").

use crate::vsa2::decode::Layer2Refusal;
use crate::vsa2::Layer2Index;
use crate::Family;

use super::fixture::corpus_report;

/// The most lexically-distinctive fixture row (the research record: "research"/"findings" appear
/// nowhere else) — a self-query over it is an unambiguous recovery target.
fn distinctive_query(report: &crate::TeroIndexReport) -> (String, String) {
    let record = report
        .items
        .iter()
        .find(|it| it.family == Family::Research && it.kind == "record")
        .expect("fixture research record");
    let query = format!(
        "{} {}",
        record.title,
        record.summary.as_deref().unwrap_or_default()
    );
    (query, record.anchor.clone())
}

#[test]
fn a_self_query_recovers_the_right_anchor_top_1() {
    let (_root, report) = corpus_report("l2-decode-recover");
    let index = Layer2Index::build(&report);
    let (query, gold) = distinctive_query(&report);

    let ranked = index.rank(&query, 3).expect("non-empty codebook + query");
    assert_eq!(
        ranked.first().map(|c| c.anchor.as_str()),
        Some(gold.as_str()),
        "cleanup should recover the distinctive record as top-1 (ranked: {ranked:?})"
    );
}

#[test]
fn below_threshold_query_is_a_typed_refusal() {
    let (_root, report) = corpus_report("l2-decode-lowconf");
    let index = Layer2Index::build(&report);
    // A token present in no record ⇒ the probe is ~orthogonal to every record ⇒ confidence below the
    // declared floor ⇒ an explicit LowConfidence refusal, never a low-quality nearest-neighbour.
    match index.query("zzqqxx") {
        Err(Layer2Refusal::LowConfidence { confidence, .. }) => {
            assert!(confidence < crate::vsa2::decode::L2_MIN_CONFIDENCE);
        }
        other => panic!("expected a LowConfidence refusal, got {other:?}"),
    }
}

#[test]
fn empty_query_is_a_typed_refusal() {
    let (_root, report) = corpus_report("l2-decode-empty");
    let index = Layer2Index::build(&report);
    assert!(matches!(
        index.query("   "),
        Err(Layer2Refusal::EmptyQuery { .. })
    ));
    assert!(matches!(
        index.rank("", 3),
        Err(Layer2Refusal::EmptyQuery { .. })
    ));
}

#[test]
fn a_recovered_answer_names_and_resolves_its_layer1_evidence() {
    let (_root, report) = corpus_report("l2-decode-prov");
    let index = Layer2Index::build(&report);
    let (query, gold) = distinctive_query(&report);

    let answer = index.query(&query).expect("a confident recovery");
    // The Layer-2 answer names its Layer-1 evidence, and that citation resolves to a real row.
    assert_eq!(answer.citation().anchor, gold);
    let resolved = index
        .resolve(&answer.citation().anchor)
        .expect("recovered anchor must resolve to a real Layer-1 row (provenance preserved)");
    assert_eq!(resolved.anchor, gold);
    // The citation carries the Layer-1 row's family + the uniform extraction tag (not invented).
    assert_eq!(resolved.family, Family::Research);
    assert!(!resolved.item_tag.is_empty());
    assert!(answer.confidence() >= crate::vsa2::decode::L2_MIN_CONFIDENCE);
}

#[test]
fn structured_unbind_probe_recovers_a_record_kind() {
    // The optional secondary path: an exact `unbind` of the KIND role, cleaned up against the small
    // kind codebook, recovers a record's kind. The unbind op is Exact (MAP-I self-inverse); recovering
    // the filler from a *bundle* is Empirical (crosstalk) — so the Match confidence is the honest
    // quantity, never an Exact-stamped guess.
    let (_root, report) = corpus_report("l2-decode-probe");
    let index = Layer2Index::build(&report);
    let record = report
        .items
        .iter()
        .find(|it| it.family == Family::Research && it.kind == "record")
        .expect("fixture research record");
    let hit = index
        .probe_kind(&record.anchor)
        .expect("a probe over an encoded anchor returns a Match");
    assert_eq!(
        hit.label, record.kind,
        "should recover the record's own kind"
    );
}
