//! White-box tests for the Layer-2 **EXPLAIN** trace shape (M-1018): a Layer-2 answer is inspectable
//! the same way a Layer-1 answer is (no black boxes — G2).

use crate::vsa2::{Layer2Index, TERO_L2_SEED};
use crate::Family;

use super::fixture::corpus_report;

#[test]
fn explain_trace_is_populated_and_honest() {
    let (_root, report) = corpus_report("l2-explain");
    let index = Layer2Index::build(&report);
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

    let answer = index.query(&query).expect("a confident recovery");
    let ex = answer.explain();

    assert_eq!(ex.model_id, "MAP-I");
    assert_eq!(ex.dim, crate::vsa2::L2_DIM);
    assert_eq!(ex.seed, TERO_L2_SEED);
    assert!(!ex.query_terms.is_empty(), "the probe's terms are recorded");
    assert_eq!(
        ex.candidates_scanned,
        index.len(),
        "candidates scanned = the codebook length"
    );
    assert!(!ex.hits.is_empty(), "the ranked hits are recorded");
    assert_eq!(
        ex.hits[0].anchor,
        answer.citation().anchor,
        "the top hit is the answer's cited row"
    );
    assert!(
        (ex.hits[0].margin - answer.margin()).abs() < 1e-12,
        "the top hit carries the decision margin"
    );
    assert!(ex.decode_method.contains("cleanup"));
    assert_eq!(
        ex.guarantee_tag, "Empirical",
        "cleanup retrieval is Empirical"
    );
    assert!(ex.empirical_profile_check.contains("never-silent"));

    // The trace serializes (it is the inspectable, EXPLAIN-able surface).
    let json = serde_json::to_string(ex).expect("Layer2Explain serializes");
    assert!(json.contains("\"model_id\":\"MAP-I\""));
}
