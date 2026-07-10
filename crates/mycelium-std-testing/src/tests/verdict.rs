//! White-box unit tests for `verdict.rs` (Verdict, FailRecord, SkipReason, UndetReason,
//! Summary).
//!
//! Extracted as-touched per the test-layout rule (CLAUDE.md Â§Test layout).

use crate::verdict::*;

/// Verify `SkipReason` variants are distinct (no accidental merging).
#[test]
fn skip_reasons_are_distinct() {
    use SkipReason::*;
    let reasons = [
        Ignored,
        UnmetPrecondition,
        NeedsRecord,
        BackendUnavailable,
        ToolMissing,
    ];
    for i in 0..reasons.len() {
        for j in 0..reasons.len() {
            if i != j {
                assert_ne!(
                    reasons[i], reasons[j],
                    "SkipReason variants must be distinct"
                );
            }
        }
    }
}

/// Verify `UndetReason` variants are distinct.
#[test]
fn undet_reasons_are_distinct() {
    use UndetReason::*;
    let reasons = [
        OracleUnavailable,
        BudgetExhaustedInconclusive,
        NonDeterministicInput,
    ];
    for i in 0..reasons.len() {
        for j in 0..reasons.len() {
            if i != j {
                assert_ne!(
                    reasons[i], reasons[j],
                    "UndetReason variants must be distinct"
                );
            }
        }
    }
}

/// `Summary::total()` equals the sum of all counts.
#[test]
fn summary_total_equals_sum() {
    let s = Summary {
        passed: 3,
        failed: 1,
        skipped: 2,
        undetermined: 1,
    };
    assert_eq!(s.total(), 7);
}

/// Default `Summary` has all-zero counts.
#[test]
fn summary_default_is_zero() {
    let s = Summary::default();
    assert_eq!(s.total(), 0);
}

/// `Verdict::Pass` is not equal to `Verdict::Skipped` (the honesty crux).
#[test]
fn verdict_pass_ne_skipped_all_reasons() {
    for reason in [
        SkipReason::Ignored,
        SkipReason::UnmetPrecondition,
        SkipReason::NeedsRecord,
        SkipReason::BackendUnavailable,
        SkipReason::ToolMissing,
    ] {
        assert_ne!(
            Verdict::Pass,
            Verdict::Skipped { reason },
            "Verdict::Pass must never equal Skipped{{reason={reason:?}}} (honesty crux)"
        );
    }
}

/// `Verdict::Pass` is not equal to `Verdict::Undetermined` (the honesty crux).
#[test]
fn verdict_pass_ne_undetermined_all_reasons() {
    for reason in [
        UndetReason::OracleUnavailable,
        UndetReason::BudgetExhaustedInconclusive,
        UndetReason::NonDeterministicInput,
    ] {
        assert_ne!(
            Verdict::Pass,
            Verdict::Undetermined { reason },
            "Verdict::Pass must never equal Undetermined{{reason={reason:?}}} (honesty crux)"
        );
    }
}

/// `Verdict::Fail` carries its record fields accurately.
#[test]
fn verdict_fail_record_fields() {
    let record = FailRecord {
        description: "test failure".to_owned(),
        seed: 42,
        trial: 3,
        context: "for_all".to_owned(),
    };
    let v = Verdict::Fail {
        record: record.clone(),
    };
    if let Verdict::Fail { record: r } = v {
        assert_eq!(r.description, "test failure");
        assert_eq!(r.seed, 42);
        assert_eq!(r.trial, 3);
        assert_eq!(r.context, "for_all");
    } else {
        panic!("expected Fail");
    }
}

/// `FailRecord::to_diag` delegates to the canonical `mycelium_diag::Diag` (testing↔diag seam,
/// spec §7-Q2): the description is the message; context/seed/trial ride along as notes; the
/// severity is `Error` — never an opaque red/green bit (C1/C3).
#[test]
fn fail_record_projects_to_diag() {
    let record = FailRecord {
        description: "shrunk counterexample: n=7".to_owned(),
        seed: 42,
        trial: 3,
        context: "for_all".to_owned(),
    };
    let d = record.to_diag();
    assert_eq!(d.severity(), mycelium_diag::Severity::Error);
    assert_eq!(d.message, "shrunk counterexample: n=7");
    // The reproduction metadata survives in the diagnostic's EXPLAIN notes.
    assert!(d.notes.iter().any(|n| n == "seed=42"));
    assert!(d.notes.iter().any(|n| n == "trial=3"));
    assert!(d.notes.iter().any(|n| n == "context=for_all"));
}
