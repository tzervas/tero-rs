//! Tests for [`crate::r2_residual`] (M-963; DN-78 §3 B-3 / §4) — the regression guard that
//! makes the deferral ledger *checked*, not prose (G2).

use crate::r2_residual::{require, residual_for, DeferredR2, RESIDUALS};

#[test]
fn ledger_is_complete_one_row_per_item_in_order() {
    assert_eq!(
        RESIDUALS.len(),
        DeferredR2::ALL.len(),
        "exactly one ledger row per deferred item (DN-78 §4) — a deferral without a row is a \
         silent gap (G2)"
    );
    for (i, item) in DeferredR2::ALL.iter().enumerate() {
        assert_eq!(
            RESIDUALS[i].item, *item,
            "ledger row {i} must correspond to DeferredR2::ALL[{i}]"
        );
        assert_eq!(
            residual_for(*item).item,
            *item,
            "residual_for must round-trip to the item's own row"
        );
    }
    // Mutant witness: removing a row, reordering, or mispointing residual_for fails here.
}

#[test]
fn every_row_names_its_construct_reason_tracker_and_basis() {
    for row in RESIDUALS {
        assert!(
            !row.construct.is_empty(),
            "a refusal must name what it refuses (G2): {:?}",
            row.item
        );
        assert!(
            !row.why_deferred.is_empty(),
            "a deferral must state its unmet prerequisite (VR-5): {:?}",
            row.item
        );
        assert!(
            row.tracker.contains("M-"),
            "a residual must be tracked by a task id, never dropped: {:?} → {}",
            row.item,
            row.tracker
        );
        assert!(
            row.basis.contains("DN-78"),
            "each row cites the split that decided it (grounding): {:?} → {}",
            row.item,
            row.basis
        );
    }
}

#[test]
fn require_refuses_every_deferred_item_explicitly() {
    for item in DeferredR2::ALL {
        let err = require(item)
            .expect_err("in Phase I every deferred item must refuse — never a silent no-op (G2)");
        assert_eq!(err.item, item);
        let msg = err.to_string();
        assert!(
            msg.contains(err.row.construct) && msg.contains(err.row.tracker),
            "the refusal must teach (construct + tracker): got '{msg}'"
        );
    }
    // Mutant witness: an arm returning Ok(()) before its construct's vehicle lands fails here.
}

#[test]
fn refusal_is_deterministic() {
    for item in DeferredR2::ALL {
        assert_eq!(
            require(item),
            require(item),
            "the refusal is a pure function of the item (Exact by construction)"
        );
    }
}
