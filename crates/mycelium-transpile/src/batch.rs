//! Directory/batch mode (M-873 follow-on): discover every `*.rs` file under a crate's `src/`,
//! transpile each independently, and summarize the results — the per-file CLI-mode logic pulled
//! out into a reusable, testable module so the CLI (`src/bin/mycelium-transpile.rs`) stays a thin
//! I/O shell.
//!
//! **Guarantee: `Declared`** (same basis as `emit`/`transpile` — see `src/lib.rs`); the
//! aggregation here (sums, percentages, category merges) is exact arithmetic over already-Declared
//! per-file [`crate::gap::GapReport`]s, so it inherits their tag rather than degrading it further.

use crate::gap::{Gap, GapReport};
use crate::transpile::transpile_file;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Recursively discover every `*.rs` file under `root`, skipping test infrastructure: any
/// directory component named `tests` (covers both a crate-root `tests/` integration-test dir and
/// the in-crate `src/tests/` unit-test layout — CLAUDE.md "Test layout") and any file whose stem
/// is exactly `tests` (the older single-file `src/tests.rs` shape, e.g.
/// `mycelium-std-fmt/src/tests.rs`). Both are out of this PoC's transpilation scope (the same
/// scope `emit::is_cfg_test`/`Category::TestItem` already exclude at the item level for
/// `#[cfg(test)] mod`); skipping the *files* here avoids parsing pure test bodies as if they were
/// library surface. Returns files in a deterministic (sorted) order.
pub fn discover_rs_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if path.file_name().and_then(|n| n.to_str()) == Some("tests") {
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file()
                && path.extension().and_then(|e| e.to_str()) == Some("rs")
                && path.file_stem().and_then(|s| s.to_str()) != Some("tests")
            {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// One file's contribution to a [`BatchSummary`].
#[derive(Debug, Clone, Serialize)]
pub struct FileSummary {
    pub file: String,
    pub total_items: usize,
    pub non_test_items: usize,
    pub emitted: usize,
    pub gaps: usize,
    pub expressible_pct: f64,
    pub category_counts: BTreeMap<&'static str, usize>,
}

impl FileSummary {
    fn from_report(file: String, report: &GapReport) -> Self {
        FileSummary {
            file,
            total_items: report.total_top_level_items,
            non_test_items: report.non_test_item_count(),
            emitted: report.emitted_items.len(),
            gaps: report.gaps.len(),
            expressible_pct: report.expressible_fraction() * 100.0,
            category_counts: report.category_counts(),
        }
    }
}

/// The batch-wide aggregate — same shape as [`FileSummary`] minus the per-file `file` name, so a
/// consumer can treat `totals` as "one more row" without a meaningless synthetic filename.
#[derive(Debug, Clone, Serialize)]
pub struct Totals {
    pub total_items: usize,
    pub non_test_items: usize,
    pub emitted: usize,
    pub gaps: usize,
    pub expressible_pct: f64,
    pub category_counts: BTreeMap<&'static str, usize>,
}

/// The combined `summary.json` artifact for a batch/directory transpile run.
#[derive(Debug, Clone, Serialize)]
pub struct BatchSummary {
    pub files: Vec<FileSummary>,
    pub totals: Totals,
}

/// The combined `union.gap.json` artifact: every [`Gap`] from every file in the batch, plus the
/// aggregate per-category counts — never deduplicated or dropped (G2: a gap recorded once per
/// file it occurs in, since each occurrence is a distinct construct at a distinct file/line).
#[derive(Debug, Clone, Serialize)]
pub struct UnionGapReport {
    pub gaps: Vec<Gap>,
    pub category_counts: BTreeMap<&'static str, usize>,
}

/// One file's parse/transpile outcome, kept alongside its report so the CLI can still write the
/// per-file `.myc`/`.gap.json` artifacts after batch summarization.
pub struct FileResult {
    pub path: PathBuf,
    pub myc: String,
    pub report: GapReport,
}

/// Transpile every file in `files` (already-discovered `.rs` paths), collecting a
/// [`FileResult`] per file that parses. A file that fails to parse/read (a hard `syn` failure,
/// distinct from a per-item gap) is **not** silently skipped — its path/error is returned
/// separately so the caller can report it (never-silent, G2).
pub fn transpile_batch(files: &[PathBuf]) -> (Vec<FileResult>, Vec<(PathBuf, String)>) {
    let mut results = Vec::with_capacity(files.len());
    let mut failures = Vec::new();
    for path in files {
        match transpile_file(path) {
            Ok((myc, report)) => results.push(FileResult {
                path: path.clone(),
                myc,
                report,
            }),
            Err(e) => failures.push((path.clone(), e)),
        }
    }
    (results, failures)
}

/// Build the [`BatchSummary`] + [`UnionGapReport`] artifacts from a batch's [`FileResult`]s.
pub fn summarize(results: &[FileResult]) -> (BatchSummary, UnionGapReport) {
    let mut files = Vec::with_capacity(results.len());
    let mut all_gaps: Vec<Gap> = Vec::new();

    let mut total_items = 0usize;
    let mut non_test_items = 0usize;
    let mut emitted = 0usize;
    let mut gaps = 0usize;
    let mut category_counts: BTreeMap<&'static str, usize> = BTreeMap::new();

    for r in results {
        let label = r.path.display().to_string();
        files.push(FileSummary::from_report(label, &r.report));

        total_items += r.report.total_top_level_items;
        non_test_items += r.report.non_test_item_count();
        emitted += r.report.emitted_items.len();
        gaps += r.report.gaps.len();
        for (cat, count) in r.report.category_counts() {
            *category_counts.entry(cat).or_insert(0) += count;
        }
        all_gaps.extend(r.report.gaps.iter().cloned());
    }

    let expressible_pct = if non_test_items == 0 {
        0.0
    } else {
        emitted as f64 / non_test_items as f64 * 100.0
    };

    let totals = Totals {
        total_items,
        non_test_items,
        emitted,
        gaps,
        expressible_pct,
        category_counts: category_counts.clone(),
    };

    (
        BatchSummary { files, totals },
        UnionGapReport {
            gaps: all_gaps,
            category_counts,
        },
    )
}
