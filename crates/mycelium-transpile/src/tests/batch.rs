//! Unit tests for directory/batch mode (`src/batch.rs`, M-873 follow-on) — no new dev-dependency
//! (e.g. `tempfile`) added for this, per the crate's kickoff-scoped minimal-deps stance (see
//! `Cargo.toml`'s `quote` comment): fixtures are written directly under `std::env::temp_dir()` in
//! a per-test unique subdirectory, cleaned up at the end of each test.

use crate::batch::{discover_rs_files, summarize, transpile_batch};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh, empty temp directory scoped to one test (`tag` disambiguates by test name; the
/// counter disambiguates parallel test threads sharing a `tag`/pid).
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "mycelium-transpile-batch-test-{tag}-{}-{n}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        TempDir(dir)
    }

    fn write(&self, rel: &str, content: &str) {
        let path = self.0.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(&path, content).expect("write fixture file");
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `discover_rs_files` recurses `*.rs` but skips any `tests` directory component (both a
/// crate-root `tests/` dir and the in-crate `src/tests/` layout) and any `tests.rs` file (the
/// older single-file test-module shape, e.g. `mycelium-std-fmt/src/tests.rs`).
#[test]
fn discover_skips_tests_dirs_and_files() {
    let tmp = TempDir::new("discover");
    tmp.write("lib.rs", "fn a(x: bool) -> bool { x }");
    tmp.write("helper.rs", "fn b(x: bool) -> bool { x }");
    tmp.write("tests.rs", "fn only_tests() {}");
    tmp.write("tests/integration.rs", "fn only_tests_2() {}");
    tmp.write("nested/mod_a.rs", "fn c(x: bool) -> bool { x }");
    tmp.write("nested/tests/deep.rs", "fn only_tests_3() {}");

    let found = discover_rs_files(tmp.path()).expect("discover succeeds");
    let names: Vec<String> = found
        .iter()
        .map(|p| {
            p.strip_prefix(tmp.path())
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();

    assert_eq!(
        names,
        vec![
            "helper.rs".to_string(),
            "lib.rs".to_string(),
            "nested/mod_a.rs".to_string(),
        ],
        "expected exactly the non-test .rs files, sorted; got {names:?}"
    );
}

/// `discover_rs_files` over an empty directory returns an empty (not missing/erroring) list —
/// never-silent for the degenerate case.
#[test]
fn discover_over_empty_dir_returns_empty() {
    let tmp = TempDir::new("discover-empty");
    let found = discover_rs_files(tmp.path()).expect("discover succeeds");
    assert!(found.is_empty(), "expected no files, got {found:?}");
}

/// `transpile_batch` + `summarize` over a small multi-file fixture: per-file summaries roll up
/// exactly into the batch totals (sum of counts, union of gaps), and the per-file never-silent
/// invariant (emitted + gaps >= total items) holds for every file in the batch — the batch-mode
/// analogue of `src/tests/invariant.rs`'s single-file check.
#[test]
fn batch_summary_totals_match_per_file_sums() {
    let tmp = TempDir::new("summary");
    // All-expressible file.
    tmp.write(
        "a.rs",
        "enum Ordering { Less, Equal, Greater }\nfn is_lt(o: bool) -> bool { o }",
    );
    // A file with a mix of emitted + gapped items (a known hard gap: named-field struct).
    tmp.write("b.rs", "struct Foo { x: u8 }\nfn ok(x: bool) -> bool { x }");
    // An all-gapped file (macro_rules! def).
    tmp.write("c.rs", "macro_rules! m { () => {}; }");

    let files = discover_rs_files(tmp.path()).expect("discover succeeds");
    assert_eq!(files.len(), 3, "expected all 3 fixture files discovered");

    let (results, failures) = transpile_batch(&files);
    assert!(
        failures.is_empty(),
        "expected every fixture file to parse, got failures={failures:?}"
    );
    assert_eq!(results.len(), 3);

    // Per-crate (per-file, here) never-silent invariant: emitted + gaps >= total items.
    for r in &results {
        let covered = r.report.emitted_items.len() + r.report.gaps.len();
        assert!(
            covered >= r.report.total_top_level_items,
            "never-silent invariant violated for {}: {} items but only {covered} \
             emitted+gap record(s)",
            r.path.display(),
            r.report.total_top_level_items
        );
    }

    let (batch_summary, union) = summarize(&results);
    assert_eq!(batch_summary.files.len(), 3);

    let sum_total_items: usize = batch_summary.files.iter().map(|f| f.total_items).sum();
    let sum_non_test: usize = batch_summary.files.iter().map(|f| f.non_test_items).sum();
    let sum_emitted: usize = batch_summary.files.iter().map(|f| f.emitted).sum();
    let sum_gaps: usize = batch_summary.files.iter().map(|f| f.gaps).sum();

    assert_eq!(batch_summary.totals.total_items, sum_total_items);
    assert_eq!(batch_summary.totals.non_test_items, sum_non_test);
    assert_eq!(batch_summary.totals.emitted, sum_emitted);
    assert_eq!(batch_summary.totals.gaps, sum_gaps);
    assert_eq!(
        union.gaps.len(),
        sum_gaps,
        "union.gap.json must carry every gap from every file, none dropped"
    );

    // At least one item landed (a.rs) and at least one gapped (b.rs's struct, c.rs's macro).
    assert!(
        sum_emitted > 0,
        "expected some emitted items across the batch"
    );
    assert!(sum_gaps > 0, "expected some gaps across the batch");

    // Per-category counts in the union must sum to the same total as `totals.category_counts`
    // (they're built from the same per-file counters) and must equal the raw gap count.
    let union_cat_sum: usize = union.category_counts.values().sum();
    assert_eq!(union_cat_sum, sum_gaps);
    let totals_cat_sum: usize = batch_summary.totals.category_counts.values().sum();
    assert_eq!(totals_cat_sum, sum_gaps);

    // Expressible percentage is a real percentage over the non-test denominator.
    assert!(
        (0.0..=100.0).contains(&batch_summary.totals.expressible_pct),
        "expressible_pct out of [0,100]: {}",
        batch_summary.totals.expressible_pct
    );
}

/// A batch over zero files (e.g. a directory that discovers nothing) yields an honest all-zero
/// summary, not a divide-by-zero panic or a fabricated percentage.
#[test]
fn batch_summary_over_zero_files_is_all_zero_not_a_panic() {
    let (batch_summary, union) = summarize(&[]);
    assert!(batch_summary.files.is_empty());
    assert_eq!(batch_summary.totals.total_items, 0);
    assert_eq!(batch_summary.totals.emitted, 0);
    assert_eq!(batch_summary.totals.gaps, 0);
    assert_eq!(batch_summary.totals.expressible_pct, 0.0);
    assert!(union.gaps.is_empty());
}
