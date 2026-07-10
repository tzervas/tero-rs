//! The 8-core-lib-twin corpus (M-873 follow-on, DN-34 §8 / kickoff `trx`'s "first-class output"):
//! the never-silent invariant, checked directly against the **real** Rust crates backing 6 of the
//! 8 hand-written twins in `lib/std/*.myc` (`crates/mycelium-transpile/fixtures/UNION-BACKLOG.md`
//! is generated from a batch run over this same corpus — see that file's header for how to
//! regenerate it).
//!
//! **Guarantee: `Empirical`.** This is the batch-mode analogue of
//! `src/tests/invariant.rs`'s fixed-corpus check and `src/tests/diff.rs`'s single-crate
//! real-source check, generalized to every crate the union backlog measures — not `Proven` for
//! the same reason (`syn::Item` is `#[non_exhaustive]`; see `src/tests/invariant.rs`'s doc
//! comment).

use crate::batch::{discover_rs_files, transpile_batch};
use std::path::PathBuf;

fn crate_src(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../{name}/src"))
}

/// The 6 Rust crates backing 6 of the 8 core-lib twins (`std.option`/`std.result` are
/// self-hosted directly in Mycelium — M-715/M-649 — with no Rust source to run; see
/// `fixtures/UNION-BACKLOG.md` §Flagged for the grounding).
fn corpus_crates() -> Vec<&'static str> {
    vec![
        "mycelium-std-cmp",
        "mycelium-std-iter",
        "mycelium-std-collections",
        "mycelium-std-text",
        "mycelium-std-fmt",
        "mycelium-std-math",
    ]
}

/// For every crate in the corpus, every file batch-transpiles without a hard parse failure, and
/// the never-silent invariant (`emitted_items.len() + gaps.len() >= total_top_level_items`) holds
/// for every file — the same sum-bound `src/tests/invariant.rs` checks over its fixed corpus,
/// checked here over real, unmodified crate source.
#[test]
fn never_silent_holds_over_the_union_backlog_corpus() {
    for crate_name in corpus_crates() {
        let src = crate_src(crate_name);
        assert!(
            src.is_dir(),
            "expected {crate_name}'s src/ dir at {}",
            src.display()
        );
        let files = discover_rs_files(&src).unwrap_or_else(|e| {
            panic!("failed to discover .rs files under {}: {e}", src.display())
        });
        assert!(
            !files.is_empty(),
            "expected at least one .rs file under {}",
            src.display()
        );

        let (results, failures) = transpile_batch(&files);
        assert!(
            failures.is_empty(),
            "expected every file under {crate_name}/src to parse, got failures={failures:?}"
        );

        for r in &results {
            let covered = r.report.emitted_items.len() + r.report.gaps.len();
            assert!(
                covered >= r.report.total_top_level_items,
                "never-silent invariant violated for {}: {} top-level item(s) but only \
                 {covered} emitted+gap record(s)",
                r.path.display(),
                r.report.total_top_level_items
            );
        }
    }
}

/// A cross-check the union backlog's headline numbers rest on: at least one crate in the corpus
/// has a non-trivial expressible fraction (catches a regression that would silently zero out
/// emission across the whole corpus, e.g. a botched refactor of `dispatch_item`).
#[test]
fn at_least_one_corpus_crate_has_nontrivial_expressible_fraction() {
    let mut any_nontrivial = false;
    for crate_name in corpus_crates() {
        let src = crate_src(crate_name);
        let files = discover_rs_files(&src).expect("discover succeeds");
        let (results, _failures) = transpile_batch(&files);
        let emitted: usize = results.iter().map(|r| r.report.emitted_items.len()).sum();
        let non_test: usize = results.iter().map(|r| r.report.non_test_item_count()).sum();
        if non_test > 0 && emitted as f64 / non_test as f64 > 0.05 {
            any_nontrivial = true;
        }
    }
    assert!(
        any_nontrivial,
        "expected at least one corpus crate with a >5% expressible fraction"
    );
}
