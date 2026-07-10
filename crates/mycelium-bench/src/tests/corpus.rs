use crate::corpus::*;

#[test]
fn every_case_elaborates_to_a_core_term() {
    // A corpus regression — a program that no longer parses/checks/elaborates — must be loud.
    for case in corpus() {
        case.elaborate()
            .unwrap_or_else(|e| panic!("corpus case `{}` failed to elaborate: {e}", case.id));
    }
}

#[test]
fn case_ids_are_unique() {
    let mut ids: Vec<&str> = corpus().iter().map(|c| c.id).collect();
    let n = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), n, "corpus case ids must be unique");
}

#[test]
fn corpus_spans_all_fragments() {
    let frags: std::collections::BTreeSet<_> = corpus().iter().map(|c| c.fragment).collect();
    // The corpus must exercise every fragment so the capability-loss surface is covered.
    assert!(frags.contains(&Fragment::BitSubset));
    assert!(frags.contains(&Fragment::Data));
    assert!(frags.contains(&Fragment::Recursion));
    assert!(frags.contains(&Fragment::Swap));
}
