//! Tests for [`crate::guarantee_matrix`] — extracted from the logic file per the as-touched
//! test-layout rule (M-797), and extended with the M-963 rows and the M-964 RT2/determinism
//! honesty guard.

use crate::guarantee_matrix::MATRIX;
use mycelium_core::GuaranteeStrength;

#[test]
fn matrix_non_empty() {
    assert!(!MATRIX.is_empty(), "guarantee matrix must have entries");
}

#[test]
fn task_purity_is_declared_not_higher() {
    let row = MATRIX
        .iter()
        .find(|r| r.operation == "Task purity contract")
        .expect("Task purity row must exist");
    assert_eq!(
        row.strength,
        GuaranteeStrength::Declared,
        "Task purity must not be upgraded beyond Declared without a checked basis (VR-5)"
    );
    // Mutant witness: changing strength to Empirical would make this test fail,
    // correctly catching an ungrounded tag upgrade.
}

#[test]
fn kahn_determinism_is_empirical_not_proven() {
    for row in MATRIX {
        if row.operation.contains("Kahn") {
            assert_ne!(
                row.strength,
                GuaranteeStrength::Proven,
                "Kahn-determinism must not be Proven without a checked theorem (VR-5): op={}",
                row.operation
            );
            assert_ne!(
                row.strength,
                GuaranteeStrength::Exact,
                "Kahn-determinism must not be Exact — it is Empirical (ADR-020 §4): op={}",
                row.operation
            );
        }
    }
    // Mutant witness: setting any Kahn row to Proven would make this test fail.
}

#[test]
fn no_reserved_vocabulary_in_operation_names() {
    // RFC-0008 §4.5 reserved vocabulary — must not appear in v0 public API.
    let reserved = [
        "hypha", "fuse", "xloc", "cyst", "graft", "forage", "backbone", "mesh", "tier", "reclaim",
    ];
    for row in MATRIX {
        for word in &reserved {
            assert!(
                !row.operation.contains(word),
                "Reserved vocabulary '{}' must not appear in v0 guarantee matrix (ADR-020 §5): op={}",
                word,
                row.operation
            );
        }
    }
}

#[test]
fn test_new_channel_ops_are_exact() {
    // The four new bounded-channel operation rows are all Exact (deterministic by
    // construction). We match them by their known operation name prefixes, which are
    // distinct from the pre-existing Empirical rows (Kahn, Deadlock, Colony, etc.).
    // Mutant witness: changing any of these four rows to Empirical would make this test
    // fail, correctly catching an ungrounded tag downgrade for a deterministic operation.
    let exact_channel_op_prefixes = [
        "Network::channel (construction)",
        "Network::channel zero-capacity check",
        "Sender::try_send FIFO",
        "Receiver::try_recv FIFO",
    ];
    for prefix in &exact_channel_op_prefixes {
        let row = MATRIX
            .iter()
            .find(|r| r.operation.starts_with(prefix))
            .unwrap_or_else(|| panic!("guarantee matrix missing row starting with '{prefix}'"));
        assert_eq!(
            row.strength,
            GuaranteeStrength::Exact,
            "bounded-channel op '{}' must be Exact (ADR-020 §4)",
            row.operation
        );
    }
}

// ── M-963: the capture/set + residual-ledger rows (DN-78 §3) ──

/// Fixture: the five M-963 rows with their expected strengths (data-driven — the test body is
/// an assert over the case).
const M963_ROWS: [(&str, GuaranteeStrength); 5] = [
    (
        "PolicySlot::set transition record (reified setter)",
        GuaranteeStrength::Exact,
    ),
    (
        "PolicySlot::select without an active policy (explicit refusal)",
        GuaranteeStrength::Exact,
    ),
    (
        "Policy capture resolution (unknown ref is an explicit error)",
        GuaranteeStrength::Exact,
    ),
    (
        "Policy capture replay reaches the recorded decision",
        GuaranteeStrength::Empirical,
    ),
    (
        "Deferred-construct refusal (R2 residual ledger)",
        GuaranteeStrength::Exact,
    ),
];

#[test]
fn m963_rows_present_at_expected_strength() {
    for (op, strength) in M963_ROWS {
        let row = MATRIX
            .iter()
            .find(|r| r.operation == op)
            .unwrap_or_else(|| panic!("guarantee matrix missing M-963 row '{op}'"));
        assert_eq!(
            row.strength, strength,
            "M-963 op '{op}' must stay at its audited strength (VR-5; DN-78)"
        );
    }
    // Mutant witness: upgrading the replay row to Exact/Proven (or downgrading a fail-closed
    // Exact row) makes this fail, catching an unaudited tag move.
}

// ── M-964: the RT2/determinism honesty guard (VR-5; DN-78 appendix) ──

/// The only rows allowed to claim a determinism-flavored guarantee at `Exact`: pure,
/// single-threaded functions of their inputs, deterministic **by construction** (no ambient
/// input, no cross-thread schedule dependence). Audited M-964 (DN-78 appendix); adding a row
/// here requires re-running that audit, not just editing the list.
const EXACT_DETERMINISM_WHITELIST: [&str; 2] = [
    // A pure function of the queue state (task.rs — single-threaded sweep).
    "SweepOrder determinism",
    // StealPolicy::select_victim is a total pure function; the *decision* is deterministic
    // (the cross-thread *execution* claims are the separate Empirical differential rows).
    "Steal-victim-selection policy determinism (RT3 EXPLAIN)",
];

#[test]
fn determinism_claims_stay_honest() {
    // M-964 rule: a determinism/differential claim stays Empirical unless a machine-checked
    // side-condition upgrades it; no Proven without a checked basis — and there is no
    // mechanized concurrency theorem in-repo, so NO runtime row may be Proven at all.
    for row in MATRIX {
        assert_ne!(
            row.strength,
            GuaranteeStrength::Proven,
            "no std.runtime row may be Proven without an in-repo checked theorem (VR-5/M-964): op={}",
            row.operation
        );
        let determinism_flavored = row.operation.contains("determinism")
            || row.operation.contains("differential")
            || row.operation.contains("liveness");
        if determinism_flavored && row.strength == GuaranteeStrength::Exact {
            assert!(
                EXACT_DETERMINISM_WHITELIST.contains(&row.operation),
                "determinism claim '{}' is Exact but not in the audited by-construction \
                 whitelist — re-run the M-964 audit before upgrading (VR-5)",
                row.operation
            );
        }
    }
    // Mutant witness: tagging any differential row Exact/Proven, or removing a whitelist
    // entry without downgrading its row, makes this fail.
}
