//! Unit tests for the transpile → `myc check` vet loop (`src/vet.rs`, M-1000).
//!
//! **Guarantee: `Empirical`** for the live oracle test (it measures the real toolchain);
//! `Declared`/pure for the classification + aggregation tests (they exercise deterministic logic
//! over hand-built inputs, no process spawn — so they run fast and never depend on the toolchain
//! being present). Complex setup stays in fixtures/tables per the house test-layout rule; each test
//! body is `assert over a case`.

use crate::gap::{Category, Gap, GapReport};
use crate::vet::{
    classify_run, vet_batch, MycChecker, VetClass, VetInput, VetRecord, VetReport,
    MAX_DIAGNOSTIC_LEN,
};
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// VetClass::from_exit_code — the exit-contract mapping (data-driven).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Every documented `myc-check` oracle exit code maps to exactly its class; `None` (signal, no code)
/// and every unknown code map to a non-`Clean` class — an unknown outcome is **never** read as
/// clean (G2/VR-5).
#[test]
fn exit_code_maps_to_class() {
    let cases: &[(Option<i32>, VetClass)] = &[
        (Some(0), VetClass::Clean),
        (Some(2), VetClass::ParseError),
        (Some(3), VetClass::CheckError),
        (Some(64), VetClass::Usage),
        (Some(66), VetClass::Io),
        (Some(5), VetClass::Other(5)),
        (Some(101), VetClass::Other(101)),
        (None, VetClass::ToolUnavailable),
    ];
    for (code, expect) in cases {
        assert_eq!(
            VetClass::from_exit_code(*code),
            *expect,
            "exit code {code:?} misclassified"
        );
        // Only exit 0 is ever clean.
        assert_eq!(
            VetClass::from_exit_code(*code).is_clean(),
            *code == Some(0),
            "is_clean wrong for {code:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// classify_run — diagnostic extraction chooses the informative stream per class.
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A clean run carries no diagnostic; a parse/check failure lifts the oracle's stdout
/// `parse-error:`/`check-error:` line; an I/O/unavailable outcome prefers stderr. Never picks an
/// empty line over a non-empty one.
#[test]
fn classify_run_picks_the_right_diagnostic_line() {
    // Clean: no diagnostic.
    let clean = classify_run("f.myc".into(), "f.rs".into(), Some(0), "ok\n", "", 3, 3);
    assert_eq!(clean.class, VetClass::Clean);
    assert!(clean.diagnostic.is_empty(), "clean run has no diagnostic");

    // Check error: the `check-error:` line is on stdout (oracle contract).
    let check = classify_run(
        "f.myc".into(),
        "f.rs".into(),
        Some(3),
        "\ncheck-error: `impl` for unknown trait `Widen`\n",
        "myc-check: 1 finding\n",
        5,
        2,
    );
    assert_eq!(check.class, VetClass::CheckError);
    assert_eq!(
        check.diagnostic,
        "check-error: `impl` for unknown trait `Widen`"
    );

    // Parse error: stdout too.
    let parse = classify_run(
        "f.myc".into(),
        "f.rs".into(),
        Some(2),
        "parse-error: expected a pattern, found Strength(Exact)\n",
        "",
        4,
        1,
    );
    assert_eq!(parse.class, VetClass::ParseError);
    assert!(parse.diagnostic.starts_with("parse-error:"));

    // I/O: prefers stderr.
    let io = classify_run(
        "f.myc".into(),
        "f.rs".into(),
        Some(66),
        "",
        "io-error: nope\n",
        1,
        0,
    );
    assert_eq!(io.class, VetClass::Io);
    assert_eq!(io.diagnostic, "io-error: nope");
}

/// A diagnostic longer than the cap is truncated with a marker, never fully dropped.
#[test]
fn long_diagnostic_is_truncated_not_dropped() {
    let long = format!("check-error: {}", "x".repeat(MAX_DIAGNOSTIC_LEN * 2));
    let rec = classify_run("f.myc".into(), "f.rs".into(), Some(3), &long, "", 1, 1);
    assert!(!rec.diagnostic.is_empty(), "diagnostic not dropped");
    assert!(
        rec.diagnostic.chars().count() <= MAX_DIAGNOSTIC_LEN + 1,
        "diagnostic bounded to the cap (+ the `…` marker)"
    );
    assert!(rec.diagnostic.ends_with('…'), "truncation marker present");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// VetRecord::checked_clean_items — the file-gated bridge.
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn record(class: VetClass, non_test: usize, emitted: usize) -> VetRecord {
    VetRecord {
        myc_file: "x.myc".into(),
        source_file: "x.rs".into(),
        class,
        exit_code: None,
        diagnostic: String::new(),
        non_test_items: non_test,
        emitted_items: emitted,
    }
}

/// A file's emitted items credit the checked numerator iff the whole file is clean; a failing file
/// contributes 0 (all-or-nothing per file — never a guessed partial attribution).
#[test]
fn checked_clean_items_is_file_gated_all_or_nothing() {
    assert_eq!(record(VetClass::Clean, 10, 4).checked_clean_items(), 4);
    assert_eq!(record(VetClass::CheckError, 10, 4).checked_clean_items(), 0);
    assert_eq!(record(VetClass::ParseError, 10, 4).checked_clean_items(), 0);
    assert_eq!(
        record(VetClass::ToolUnavailable, 10, 4).checked_clean_items(),
        0,
        "a tool-unavailable run credits nothing (never counted as clean)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// VetReport aggregation + the two fractions (denominator = non-test items, stated).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Aggregation sums per-file counts, the shared denominator is total non-test items, and
/// `checked_fraction ≤ expressible_fraction` always holds (an item can only be checked-clean if it
/// was emitted). A failing file zeroes its checked credit but still contributes to the denominator.
#[test]
fn vet_report_fractions_over_stated_denominator() {
    // File A: 10 items, 4 emitted, clean → 4 checked-clean.
    // File B: 10 items, 5 emitted, check-error → 0 checked-clean (poisoned).
    // Denominator = 20 non-test items. Emitted = 9. Checked-clean = 4.
    let report = VetReport::from_records(vec![
        record(VetClass::Clean, 10, 4),
        record(VetClass::CheckError, 10, 5),
    ]);
    assert_eq!(report.total_non_test_items, 20);
    assert_eq!(report.total_emitted_items, 9);
    assert_eq!(report.total_checked_clean_items, 4);
    assert!((report.expressible_fraction() - 9.0 / 20.0).abs() < 1e-9);
    assert!((report.checked_fraction() - 4.0 / 20.0).abs() < 1e-9);
    assert!(
        report.checked_fraction() <= report.expressible_fraction(),
        "checked_fraction must never exceed expressible_fraction"
    );

    // Per-class file counts and the clean-file companion metric.
    assert_eq!(report.class_counts.get("Clean"), Some(&1));
    assert_eq!(report.class_counts.get("CheckError"), Some(&1));
    let (clean_files, files_with_emissions) = report.clean_file_fraction();
    assert_eq!((clean_files, files_with_emissions), (1, 2));
}

/// An empty report (zero files/items) yields honest all-zero fractions, never a divide-by-zero
/// panic or a fabricated ratio.
#[test]
fn vet_report_over_zero_items_is_all_zero_not_a_panic() {
    let report = VetReport::from_records(vec![]);
    assert_eq!(report.total_non_test_items, 0);
    assert_eq!(report.checked_fraction(), 0.0);
    assert_eq!(report.expressible_fraction(), 0.0);
    assert_eq!(report.clean_file_fraction(), (0, 0));

    // A file with items but zero emissions: 0/N, and it is not counted as a clean *draft*.
    let report2 = VetReport::from_records(vec![record(VetClass::Clean, 5, 0)]);
    assert_eq!(report2.checked_fraction(), 0.0);
    assert_eq!(report2.expressible_fraction(), 0.0);
    assert_eq!(
        report2.clean_file_fraction(),
        (0, 0),
        "a header-only (zero-emission) clean nodule is not a clean draft"
    );
}

/// `VetInput::from_report` reads the per-file counts straight off a `GapReport`, so the vet
/// denominator matches the report's own `non_test_item_count`.
#[test]
fn vet_input_reads_counts_from_gap_report() {
    let report = GapReport {
        source: "s.rs".into(),
        emitted_items: vec!["A".into(), "B".into()],
        gaps: vec![Gap {
            file: "s.rs".into(),
            line: 1,
            col: 1,
            category: Category::TestItem,
            rust_construct: Category::TestItem.as_str().into(),
            snippet: String::new(),
            reason: String::new(),
            item_name: None,
        }],
        total_top_level_items: 3, // 3 total, 1 test → 2 non-test.
    };
    let input = VetInput::from_report(PathBuf::from("s.myc"), &report);
    assert_eq!(input.non_test_items, 2);
    assert_eq!(input.emitted_items, 2);
    assert_eq!(input.source_file, "s.rs");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// A tool-unavailable checker is recorded, never fatal (never-silent, never a hard stop).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Vetting with a checker that cannot be spawned yields a `ToolUnavailable` record (not a panic /
/// not a silent skip), and that record credits nothing to `checked_fraction`.
#[test]
fn unavailable_checker_records_tool_unavailable() {
    let checker = MycChecker {
        command: vec!["/nonexistent/definitely-not-a-real-binary-xyz".into()],
        cwd: None,
    };
    let inputs = vec![VetInput {
        myc_path: PathBuf::from("/tmp/does-not-need-to-exist.myc"),
        source_file: "x.rs".into(),
        non_test_items: 3,
        emitted_items: 2,
    }];
    let report = vet_batch(&checker, &inputs);
    assert_eq!(report.records.len(), 1);
    assert_eq!(report.records[0].class, VetClass::ToolUnavailable);
    assert_eq!(report.total_checked_clean_items, 0);
    assert_eq!(report.checked_fraction(), 0.0);
    assert!(
        report.records[0].diagnostic.contains("could not run"),
        "the unavailable-tool diagnostic names the failure, never silent"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Live end-to-end witness against the REAL `myc check` — skip-gracefully when it isn't built.
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Locate a runnable `myc-check`: `MYC_CHECK_CMD` (first whitespace token), else the workspace
/// `target/debug/myc-check`. Returns `None` (→ graceful skip) when neither is present — this test
/// must not *build* the checker (it isn't a dep of this crate), only exercise it if already built.
///
/// `pub(in crate::tests)` (not private/`pub`): the `binop_operand_gated_forms_check_clean` live
/// oracle in `src/tests/emit.rs` and the forward-map oracle tests in `src/tests/prim_map.rs` reuse
/// this exact helper (DRY, CLAUDE.md house rule 5) instead of each keeping a drifting copy — scoped
/// to `crate::tests` since it is test-only infrastructure, never part of the crate's real API.
pub(in crate::tests) fn find_myc_check() -> Option<PathBuf> {
    if let Ok(cmd) = std::env::var("MYC_CHECK_CMD") {
        if let Some(first) = cmd.split_whitespace().next() {
            let p = PathBuf::from(first);
            if p.exists() {
                return Some(p);
            }
        }
    }
    let built = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/myc-check");
    if built.exists() {
        Some(built)
    } else {
        None
    }
}

/// End-to-end: a hand-written, known-clean `.myc` classifies `Clean`; a known-broken one (an
/// unresolved `use`) classifies `CheckError`. Skips (never fails) when `myc-check` is not built.
#[test]
fn live_myc_check_classifies_clean_and_broken() {
    let Some(bin) = find_myc_check() else {
        eprintln!(
            "vet: live oracle test skipped — no runnable myc-check (set MYC_CHECK_CMD or build \
             `cargo build -p mycelium-check --bin myc-check`). Pure vet tests still cover the logic."
        );
        return;
    };
    let checker = MycChecker {
        command: vec![bin.display().to_string()],
        cwd: None,
    };

    let dir = std::env::temp_dir().join(format!(
        "mycelium-transpile-vet-live-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // Known-clean: a nullary sum type + a total projection (both confirmed to check by the profile).
    let clean = dir.join("clean.myc");
    std::fs::write(
        &clean,
        "// nodule: p\nnodule p;\n\ntype Ordering = Lt | Eq | Gt;\n",
    )
    .expect("write clean.myc");
    let clean_rec = checker.vet_file(&clean, "clean.rs", 1, 1);
    assert_eq!(
        clean_rec.class,
        VetClass::Clean,
        "known-clean .myc must classify Clean; diagnostic={:?}",
        clean_rec.diagnostic
    );
    assert_eq!(clean_rec.checked_clean_items(), 1);

    // Known-broken: an unresolved external `use` (the dominant real-toolchain check poison).
    let broken = dir.join("broken.myc");
    std::fs::write(
        &broken,
        "// nodule: p\nnodule p;\n\nuse mycelium_core.GuaranteeStrength;\ntype X = A | B;\n",
    )
    .expect("write broken.myc");
    let broken_rec = checker.vet_file(&broken, "broken.rs", 1, 1);
    assert_eq!(
        broken_rec.class,
        VetClass::CheckError,
        "an unresolved `use` must classify CheckError; diagnostic={:?}",
        broken_rec.diagnostic
    );
    assert_eq!(
        broken_rec.checked_clean_items(),
        0,
        "a check-failing file credits nothing"
    );
    assert!(
        !broken_rec.diagnostic.is_empty(),
        "the failure diagnostic is captured, never silent"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
