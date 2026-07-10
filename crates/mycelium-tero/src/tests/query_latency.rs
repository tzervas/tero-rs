//! M-1016 DoD: "latency measured and recorded (Empirical) on the real corpus." Runs a representative
//! query mix over the real, committed `docs/tero-index/index.json` (loaded via
//! [`crate::load::load_report`], not a re-walk of the corpus — this measures the query engine, not
//! the M-1015 build) and prints the wall-clock (`--nocapture` to see it; the number reported in the
//! M-1016 landing message comes from running this locally).
//!
//! **Honesty (VR-5):** the timing below is `Empirical` — one run, on whatever machine executes the
//! test, not a controlled benchmark. The `assert!` is a generous sanity ceiling to catch an
//! accidental O(n²)-style regression, not a performance contract; do not read the printed number as
//! a portable guarantee. Skip-graceful (mirrors `families.rs`'s live-repo tests): a checkout without
//! the committed index (e.g. a stripped fixture repo) yields no assertion, not a failure.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::load::load_report;
use crate::{Query, QueryEngine};

/// The repo root, two levels above this crate's manifest dir (mirrors `families.rs::repo_root`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

#[test]
fn query_latency_over_the_real_committed_corpus_is_measured_and_recorded() {
    let index_path = repo_root().join("docs/tero-index/index.json");
    if !index_path.exists() {
        return; // skip-graceful — a stripped checkout without the committed Layer-1 index
    }
    let report = load_report(&index_path).expect("load the committed docs/tero-index/index.json");
    let engine = QueryEngine::new(&report);

    // A representative mix: one exact id lookup, one status filter, one kind filter, one
    // cross-reference walk, one free-text search — the same five query kinds `Query` exposes.
    let queries = [
        Query::Id("M-1015".to_owned()),
        Query::Status("done".to_owned()),
        Query::Kind("rfc".to_owned()),
        Query::CrossRef {
            start: "M-1016".to_owned(),
            depth: 2,
        },
        Query::Text("transparent memory substrate provenance".to_owned()),
    ];

    const REPS: u32 = 50;
    let start = Instant::now();
    for _ in 0..REPS {
        for q in &queries {
            // Either outcome (Answer or Refusal) is a valid, timed engine response — this measures
            // the query path's latency, not whether every representative query happens to match.
            let _ = engine.run(q);
        }
    }
    let elapsed = start.elapsed();
    let total_queries = REPS as usize * queries.len();
    let avg = elapsed / u32::try_from(total_queries).unwrap();

    println!(
        ">> M-1016 query latency (Empirical, single-machine measurement): {total_queries} queries \
         over {} indexed rows in {elapsed:?} total, {avg:?} average/query",
        report.items.len()
    );

    // A generous sanity ceiling (not a perf contract — machines vary): catches an accidental
    // O(n^2)-per-query regression over the ~5k-row real corpus, not a benchmark gate.
    assert!(
        avg.as_millis() < 500,
        "average query latency {avg:?} exceeds the 500ms sanity ceiling over {} rows — investigate \
         a possible complexity regression",
        report.items.len()
    );
}
