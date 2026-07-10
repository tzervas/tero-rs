//! Tests for `crate::region` — DN-32 §2.3 / RFC-0027 §10.3 / MEM-3.
//!
//! M-797 in-crate test layout: all tests live here, not in `region.rs`.
//!
//! # DoD coverage
//!
//! 1. **Batched ScopeExit emission:** a region with N deferred entries emits exactly N
//!    `ReclamationRecord(ScopeExit)` records at close, in order (never-silent G2).
//! 2. **ScopeExit is the trigger:** every emitted record has `trigger = ScopeExit` (the SECOND
//!    live trigger after `RcZero` from MEM-2).
//! 3. **Canonical ID types:** `ScopeNodeId::as_scope_id()` and `RegionEpoch::as_sweep_epoch()`
//!    thread the canonical types through `ReclamationRecord` fields correctly.
//! 4. **Parent–child ordering TOTAL:** the parent's `RegionEpoch` is numerically greater than
//!    every child's epoch (children close before the parent → lower epoch by construction).
//!    Property test over arbitrary child counts.
//! 5. **Sibling ordering CONCURRENT (OQ-1 resolved):** two sibling regions close without any
//!    ordering constraint; their epochs are independent (no assertion that one < the other).
//!    Property test verifying no cross-sibling dependency.
//! 6. **Empty region:** closing a region with no deferred entries emits 0 records and still
//!    advances the epoch (the close event itself is observable).
//! 7. **`ScopeNodeId` uniqueness:** each `Region::new()` allocates a distinct `ScopeNodeId`
//!    (monotonically increasing). Property test.
//! 8. **`RegionEpoch` monotonicity:** each `Region::close()` allocates a strictly increasing
//!    `RegionEpoch`. Property test.
//! 9. **`ScopeTree::close_ordered` enforces child-before-parent and emits all records.**

use proptest::prelude::*;

use crate::reclamation::{CollectingSink, ReclamationTrigger};
use crate::region::{ClosedRegion, Region, RegionEpoch, ScopeNodeId, ScopeTree};
use mycelium_core::ContentHash;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_hash(n: u64) -> ContentHash {
    let digest = format!("{:064}", n);
    ContentHash::from_parts("blake3", &digest).expect("test hash must be well-formed")
}

/// Create a region, defer `n` values, close it, and return the collected records.
fn close_region_with_n_deferred(n: usize) -> (ClosedRegion, CollectingSink) {
    let mut region = Region::new();
    for i in 0..n {
        region.defer(make_hash(i as u64));
    }
    let mut sink = CollectingSink::new();
    let closed = region.close(&mut sink);
    (closed, sink)
}

// ── 1. Batched ScopeExit emission ─────────────────────────────────────────────

#[test]
fn empty_region_emits_zero_records_at_close() {
    // DoD item 6: no deferred entries → no records. Close still succeeds.
    let (closed, sink) = close_region_with_n_deferred(0);
    assert!(sink.is_empty(), "empty region must emit 0 records at close");
    assert_eq!(closed.reclaimed_count, 0, "reclaimed_count must be 0");
    assert!(
        closed.epoch.as_u64() > 0,
        "epoch must be allocated even for empty region"
    );
}

#[test]
fn region_with_one_deferred_emits_one_scope_exit_record() {
    // DoD items 1 + 2: exactly one record, trigger = ScopeExit (the SECOND live trigger, MEM-3).
    let (closed, sink) = close_region_with_n_deferred(1);
    assert_eq!(
        sink.len(),
        1,
        "exactly one record must be emitted for one deferred entry (G2)"
    );
    assert_eq!(
        sink.records[0].trigger,
        ReclamationTrigger::ScopeExit,
        "trigger must be ScopeExit (the MEM-3 live trigger)"
    );
    assert_eq!(
        closed.reclaimed_count, 1,
        "reclaimed_count must match deferred count"
    );
}

#[test]
fn region_with_three_deferred_emits_three_scope_exit_records() {
    // DoD item 1: N deferred → exactly N records at close; all ScopeExit.
    let (closed, sink) = close_region_with_n_deferred(3);
    assert_eq!(sink.len(), 3, "three deferred → three records (G2)");
    for (i, record) in sink.records.iter().enumerate() {
        assert_eq!(
            record.trigger,
            ReclamationTrigger::ScopeExit,
            "record {i} must have trigger ScopeExit"
        );
    }
    assert_eq!(closed.reclaimed_count, 3);
}

#[test]
fn scope_exit_records_carry_correct_scope_id_and_epoch() {
    // DoD item 2 + 3: the emitted records carry the region's ScopeNodeId (as ScopeId)
    // and the allocated RegionEpoch (as SweepEpoch) — canonical ID threading.
    let mut region = Region::new();
    let scope_node_id = region.id;

    region.defer(make_hash(1));
    region.defer(make_hash(2));

    let mut sink = CollectingSink::new();
    let closed = region.close(&mut sink);

    // Both records share the same scope_id and sweep_epoch.
    let expected_scope_id = scope_node_id.as_scope_id();
    let expected_epoch = closed.epoch.as_sweep_epoch();

    for (i, record) in sink.records.iter().enumerate() {
        assert_eq!(
            record.scope_id, expected_scope_id,
            "record {i} scope_id must match the region's ScopeNodeId"
        );
        assert_eq!(
            record.sweep_epoch, expected_epoch,
            "record {i} sweep_epoch must match the close epoch"
        );
        assert_eq!(
            record.trigger,
            ReclamationTrigger::ScopeExit,
            "record {i} trigger must be ScopeExit"
        );
    }
}

#[test]
fn deferred_value_meta_hash_round_trips_into_record() {
    // DoD item 1: the deferred value's ContentHash must appear in the emitted record unchanged.
    let hash = make_hash(77777);
    let mut region = Region::new();
    region.defer(hash.clone());
    let mut sink = CollectingSink::new();
    let _ = region.close(&mut sink);
    assert_eq!(
        sink.records[0].value_meta_hash, hash,
        "value_meta_hash must round-trip into the ScopeExit record"
    );
    assert_eq!(
        sink.records[0].channel_id, None,
        "ScopeExit has no channel_id"
    );
}

#[test]
fn deferred_entries_emit_in_order() {
    // The deferred list is drained in FIFO order (drain(..) traverses front-to-back).
    // This is an implementation-level property; it is not normative in RFC-0027 (the batch is
    // unordered by spec), but we assert it here to catch future regressions.
    let mut region = Region::new();
    let hashes: Vec<ContentHash> = (0..5).map(|i| make_hash(i as u64)).collect();
    for h in &hashes {
        region.defer(h.clone());
    }
    let mut sink = CollectingSink::new();
    let _ = region.close(&mut sink);
    assert_eq!(sink.len(), 5);
    for (i, (record, expected_hash)) in sink.records.iter().zip(hashes.iter()).enumerate() {
        assert_eq!(
            record.value_meta_hash, *expected_hash,
            "record {i} value_meta_hash must match the deferred entry (FIFO drain order)"
        );
    }
}

// ── 3. Canonical ID type threading ────────────────────────────────────────────

#[test]
fn scope_node_id_as_scope_id_is_lossless() {
    // DoD item 3: ScopeNodeId::as_scope_id() must preserve the underlying u64 exactly.
    let id = ScopeNodeId(12345);
    let scope_id = id.as_scope_id();
    assert_eq!(scope_id.0, 12345, "as_scope_id() must be lossless");
}

#[test]
fn region_epoch_as_sweep_epoch_is_lossless() {
    // DoD item 3: RegionEpoch::as_sweep_epoch() must preserve the underlying u64 exactly.
    let epoch = RegionEpoch(99999);
    let sweep_epoch = epoch.as_sweep_epoch();
    assert_eq!(sweep_epoch.0, 99999, "as_sweep_epoch() must be lossless");
}

// ── 4. Parent–child ordering TOTAL ───────────────────────────────────────────

#[test]
fn child_epoch_less_than_parent_epoch_single_child() {
    // DoD item 4: child closes before parent → child epoch < parent epoch.
    // This is the child→root total order encoded as a number line.
    // Guarantee: Exact — monotonic counter allocates epoch at close time.
    let mut tree = ScopeTree::new(1);
    let mut sink = CollectingSink::new();
    let results = tree.close_ordered(&mut sink);

    assert_eq!(
        results.len(),
        2,
        "one child + one parent = 2 closed regions"
    );
    let child_epoch = results[0].epoch;
    let parent_epoch = results[1].epoch;
    assert!(
        child_epoch < parent_epoch,
        "child epoch ({}) must be less than parent epoch ({}) — total parent–child order",
        child_epoch.as_u64(),
        parent_epoch.as_u64()
    );
}

#[test]
fn all_child_epochs_less_than_parent_epoch_three_children() {
    // DoD item 4 with 3 siblings: all children close before the parent.
    let mut tree = ScopeTree::new(3);
    let mut sink = CollectingSink::new();
    let results = tree.close_ordered(&mut sink);

    assert_eq!(results.len(), 4, "three children + parent = 4 results");
    let parent_epoch = results[3].epoch;
    for (i, child_result) in results[..3].iter().enumerate() {
        let child_epoch = child_result.epoch;
        assert!(
            child_epoch < parent_epoch,
            "child[{i}] epoch ({}) must be less than parent epoch ({}) — total order",
            child_epoch.as_u64(),
            parent_epoch.as_u64()
        );
    }
}

#[test]
fn scope_tree_close_ordered_emits_deferred_records_for_all_regions() {
    // DoD item 9: ScopeTree::close_ordered emits all deferred records for children and parent.
    let mut tree = ScopeTree::new(2);

    // Defer one entry per child.
    tree.children[0].defer(make_hash(10));
    tree.children[1].defer(make_hash(20));
    // Defer one entry in the parent.
    tree.parent.defer(make_hash(30));

    let mut sink = CollectingSink::new();
    let results = tree.close_ordered(&mut sink);

    // Total: 1 + 1 + 1 = 3 records.
    assert_eq!(
        sink.len(),
        3,
        "all three deferred entries (2 children + 1 parent) must emit records"
    );
    // All records are ScopeExit.
    for (i, record) in sink.records.iter().enumerate() {
        assert_eq!(
            record.trigger,
            ReclamationTrigger::ScopeExit,
            "record {i} must be ScopeExit"
        );
    }
    // 3 closed regions returned.
    assert_eq!(results.len(), 3, "two children + parent = 3 closed regions");
}

// ── 5. Sibling ordering CONCURRENT (OQ-1 resolved) ───────────────────────────

#[test]
fn two_sibling_regions_close_independently_no_ordering_constraint() {
    // DoD item 5: siblings are order-independent (OQ-1 weak coupling).
    // We close them in arbitrary order — the test asserts NO cross-sibling ordering constraint.
    // Both must close successfully without violating any invariant.
    // Guarantee: Proven-modulo-LR-9 (DN-32 §3 argument; no in-repo mechanized proof).

    let mut sibling_a = Region::new();
    let mut sibling_b = Region::new();

    sibling_a.defer(make_hash(1));
    sibling_b.defer(make_hash(2));

    let mut sink_a = CollectingSink::new();
    let mut sink_b = CollectingSink::new();

    // Close A before B (one valid sibling order).
    let closed_a = sibling_a.close(&mut sink_a);
    let closed_b = sibling_b.close(&mut sink_b);

    // Both closed successfully.
    assert!(sibling_a.is_closed());
    assert!(sibling_b.is_closed());
    assert_eq!(sink_a.len(), 1, "sibling A must emit its record");
    assert_eq!(sink_b.len(), 1, "sibling B must emit its record");

    // No ordering is required between sibling epochs — either could be larger.
    // We only assert that they are DISTINCT (different allocations).
    assert_ne!(
        closed_a.epoch, closed_b.epoch,
        "sibling epochs must be distinct (independent allocations)"
    );

    // The records belong to the correct regions (by scope_id).
    assert_eq!(
        sink_a.records[0].scope_id,
        sibling_a.id.as_scope_id(),
        "A's record must carry A's scope_id"
    );
    assert_eq!(
        sink_b.records[0].scope_id,
        sibling_b.id.as_scope_id(),
        "B's record must carry B's scope_id"
    );
}

#[test]
fn sibling_regions_in_reversed_order_also_valid() {
    // DoD item 5: closing B before A is equally valid — no ordering constraint.
    let mut sibling_a = Region::new();
    let mut sibling_b = Region::new();

    sibling_a.defer(make_hash(100));
    sibling_b.defer(make_hash(200));

    let mut sink = CollectingSink::new();

    // Close B first, then A — the reverse of the previous test.
    let closed_b = sibling_b.close(&mut sink);
    let closed_a = sibling_a.close(&mut sink);

    assert_eq!(sink.len(), 2, "both siblings must emit their records");
    assert_ne!(
        closed_a.epoch, closed_b.epoch,
        "sibling epochs must be distinct even in reversed order"
    );
    // In this order, B's epoch < A's epoch — the opposite of the previous test.
    // This confirms there is no required ordering: either can come first.
    assert!(
        closed_b.epoch < closed_a.epoch,
        "B closed first → B epoch < A epoch (no required sibling ordering — either is valid)"
    );
}

// ── 6. Empty region close ─────────────────────────────────────────────────────

#[test]
fn closing_region_without_deferred_entries_is_valid() {
    // DoD item 6: empty region closes without error; epoch is still allocated.
    let mut region = Region::new();
    assert_eq!(region.deferred_count(), 0);
    assert!(!region.is_closed());
    assert_eq!(region.closed_epoch(), None);

    let mut sink = CollectingSink::new();
    let closed = region.close(&mut sink);

    assert!(region.is_closed());
    assert!(sink.is_empty(), "empty region emits 0 records");
    assert_eq!(closed.reclaimed_count, 0);
    assert!(
        closed.epoch.as_u64() > 0,
        "epoch must be allocated even for an empty close"
    );
    assert_eq!(region.closed_epoch(), Some(closed.epoch));
}

// ── 7. ScopeNodeId uniqueness ─────────────────────────────────────────────────

#[test]
fn consecutive_regions_have_distinct_scope_node_ids() {
    // DoD item 7: each Region::new() allocates a distinct ScopeNodeId.
    // Check 10 consecutive allocations.
    let regions: Vec<Region> = (0..10).map(|_| Region::new()).collect();
    let ids: Vec<ScopeNodeId> = regions.iter().map(|r| r.id).collect();

    // All IDs must be distinct.
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(
                ids[i], ids[j],
                "ScopeNodeId[{i}] and ScopeNodeId[{j}] must be distinct"
            );
        }
    }

    // IDs must be strictly increasing (monotonic counter).
    for i in 1..ids.len() {
        assert!(
            ids[i] > ids[i - 1],
            "ScopeNodeId must be strictly increasing: id[{i}]={} must be > id[{}]={}",
            ids[i].as_u64(),
            i - 1,
            ids[i - 1].as_u64()
        );
    }

    // Close all to avoid the debug-build drop-with-deferred panic.
    let mut sink = CollectingSink::new();
    for mut r in regions {
        r.close(&mut sink);
    }
}

// ── 8. RegionEpoch monotonicity ───────────────────────────────────────────────

#[test]
fn consecutive_region_closes_have_strictly_increasing_epochs() {
    // DoD item 8: each close allocates a strictly increasing RegionEpoch.
    let mut regions: Vec<Region> = (0..10).map(|_| Region::new()).collect();
    let mut sink = CollectingSink::new();
    let closed: Vec<ClosedRegion> = regions.iter_mut().map(|r| r.close(&mut sink)).collect();

    for i in 1..closed.len() {
        assert!(
            closed[i].epoch > closed[i - 1].epoch,
            "epoch[{i}]={} must be strictly greater than epoch[{}]={} (monotonic counter)",
            closed[i].epoch.as_u64(),
            i - 1,
            closed[i - 1].epoch.as_u64()
        );
    }
}

// ── Property tests ────────────────────────────────────────────────────────────

// Property: for any N deferred entries, a region emits exactly N ScopeExit records at close.
//
// This is the primary G2 never-silent property for MEM-3.
//
// Guarantee: `Empirical` — property tested over 0..=20 deferred entries.
proptest! {
    #[test]
    fn property_region_emits_exactly_n_scope_exit_records(n in 0usize..=20) {
        let (closed, sink) = close_region_with_n_deferred(n);

        prop_assert_eq!(
            sink.len(),
            n,
            "region must emit exactly N ScopeExit records for N deferred entries (G2)"
        );
        prop_assert_eq!(
            closed.reclaimed_count,
            n,
            "ClosedRegion.reclaimed_count must equal N"
        );

        // All emitted records have trigger = ScopeExit.
        // Note: proptest macros expand via concat! and cannot capture loop variables in
        // the format string; use a static message and verify the collection as a whole.
        let all_scope_exit = sink
            .records
            .iter()
            .all(|r| r.trigger == ReclamationTrigger::ScopeExit);
        prop_assert!(
            all_scope_exit,
            "all emitted records must have trigger ScopeExit — never any other trigger in MEM-3"
        );
    }

    // Property: in a ScopeTree with N children, every child's epoch is strictly less than the
    // parent's epoch (total parent–child ordering; child→root monotone).
    //
    // This encodes the RFC-0027 §10.3 sweep-order-derives-from-scope-tree property:
    // children always reclaim before the parent.
    //
    // Guarantee: `Exact` — by monotonic counter; property tested over 0..=8 children.
    #[test]
    fn property_all_child_epochs_less_than_parent_epoch(n_children in 0usize..=8) {
        let mut tree = ScopeTree::new(n_children);
        let mut sink = CollectingSink::new();
        let results = tree.close_ordered(&mut sink);

        prop_assert_eq!(
            results.len(),
            n_children + 1,
            "total closed regions = n_children + 1 (children + parent)"
        );

        let parent_epoch = results.last().expect("parent is always present").epoch;

        // All child epochs must be strictly less than the parent epoch.
        // Note: proptest macros use concat! and cannot capture loop vars in format strings;
        // we check the invariant as a whole and surface the violation separately.
        let all_children_before_parent = results[..n_children]
            .iter()
            .all(|cr| cr.epoch < parent_epoch);
        prop_assert!(
            all_children_before_parent,
            "all child epochs must be < parent epoch — total parent–child order (RFC-0027 §10.3)"
        );
    }

    // Property: two sibling regions are order-independent — no constraint between their epochs.
    //
    // This is the OQ-1 weak/concurrent sibling coupling property (DN-32 §3):
    // sibling scopes reclaim concurrently; neither is required to finish before the other.
    //
    // Guarantee: `Proven`-modulo-LR-9 (DN-32 §3 argument); property tested for structural
    // independence.
    #[test]
    fn property_sibling_regions_are_order_independent(
        n_entries_a in 0usize..=5,
        n_entries_b in 0usize..=5,
    ) {
        // Sibling A closed before B.
        let mut a = Region::new();
        let mut b = Region::new();
        for i in 0..n_entries_a {
            a.defer(make_hash(i as u64));
        }
        for i in 0..n_entries_b {
            b.defer(make_hash((i + 100) as u64));
        }

        let mut sink_a = CollectingSink::new();
        let mut sink_b = CollectingSink::new();
        let closed_a = a.close(&mut sink_a);
        let closed_b = b.close(&mut sink_b);

        // Both must close successfully.
        prop_assert!(a.is_closed(), "sibling A must be closed");
        prop_assert!(b.is_closed(), "sibling B must be closed");

        // Each emits exactly the right number of records.
        prop_assert_eq!(sink_a.len(), n_entries_a, "sibling A record count");
        prop_assert_eq!(sink_b.len(), n_entries_b, "sibling B record count");

        // Epochs are distinct (independent allocations).
        prop_assert_ne!(
            closed_a.epoch,
            closed_b.epoch,
            "sibling epochs must be distinct"
        );

        // No ordering is REQUIRED between sibling epochs — either may be larger.
        // We do NOT assert `closed_a.epoch < closed_b.epoch` or vice versa.
        // The unit test `sibling_regions_in_reversed_order_also_valid` confirms both orderings.
    }

    // Property: ScopeNodeId values are strictly increasing across consecutive Region::new() calls.
    //
    // Guarantee: `Exact` — atomic monotonic counter.
    #[test]
    fn property_scope_node_ids_are_strictly_increasing(n in 2usize..=10) {
        let regions: Vec<Region> = (0..n).map(|_| Region::new()).collect();
        let ids: Vec<ScopeNodeId> = regions.iter().map(|r| r.id).collect();

        let strictly_increasing = ids.windows(2).all(|w| w[1] > w[0]);
        prop_assert!(
            strictly_increasing,
            "ScopeNodeId values must be strictly increasing across consecutive allocations"
        );

        let mut sink = CollectingSink::new();
        for mut r in regions {
            r.close(&mut sink);
        }
    }
}
