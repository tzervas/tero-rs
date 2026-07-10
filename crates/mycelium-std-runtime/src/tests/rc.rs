//! Tests for `crate::rc` — DN-32 §2.2 / RFC-0027 §10.1 / MEM-2.
//!
//! M-797 in-crate test layout: all tests live here, not in `rc.rs`.
//!
//! # DoD coverage
//!
//! 1. **Refcount correctness:** `new` starts at 1; `clone_ref` increments; `drop_ref`
//!    decrements. Property test over inc/dec sequences.
//! 2. **`rc → 0` (UniqueOwner) emits exactly one `ReclamationRecord(RcZero)` via sink (G2).**
//!    Never-silent contract — the first live trigger into MEM-1's `ReclamationSink`.
//! 3. **`rc == 1` is detectable as the reuse opportunity (`UniqueOwner` probe).**
//! 4. **`rc > 1` returns `Shared` (no record emitted).**
//! 5. **Property test:** over arbitrary inc/dec sequences, the sink receives exactly one
//!    `RcZero` record at the end (the last `drop_ref`), and zero records for all intermediate
//!    drops.

use proptest::prelude::*;

use crate::rc::{RcCell, RcProbe};
use crate::reclamation::{
    CollectingSink, ReclamationSink, ReclamationTrigger, ScopeId, SweepEpoch,
};
use mycelium_core::ContentHash;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_hash(n: u64) -> ContentHash {
    // Build a deterministic well-formed test hash (blake3 digest = 64 zero-padded hex chars).
    let digest = format!("{:064}", n);
    ContentHash::from_parts("blake3", &digest).expect("test hash must be well-formed")
}

/// Drop one handle via `drop_ref`, routing through the given sink.
/// Returns the probe outcome.
fn do_drop<T>(cell: RcCell<T>, sink: &mut dyn ReclamationSink) -> RcProbe<T> {
    cell.drop_ref(sink, ScopeId(1), SweepEpoch(1), make_hash(42))
}

// ── 1. Refcount correctness ───────────────────────────────────────────────────

#[test]
fn new_starts_at_refcount_one() {
    // Guarantee: Exact — Rc::new sets strong_count to 1.
    let cell: RcCell<u32> = RcCell::new(99);
    assert_eq!(cell.refcount(), 1, "new cell must start at refcount 1");
    // Clean up (drain into a noop sink to avoid leaking the audit record).
    let mut sink = CollectingSink::new();
    let probe = do_drop(cell, &mut sink);
    assert!(matches!(probe, RcProbe::UniqueOwner(_)));
}

#[test]
fn clone_ref_increments_refcount() {
    // Guarantee: Exact — each clone_ref adds 1.
    let cell: RcCell<u32> = RcCell::new(7);
    assert_eq!(cell.refcount(), 1);

    let c2 = cell.clone_ref();
    assert_eq!(cell.refcount(), 2);
    assert_eq!(c2.refcount(), 2);

    let c3 = cell.clone_ref();
    assert_eq!(cell.refcount(), 3);
    assert_eq!(c2.refcount(), 3);
    assert_eq!(c3.refcount(), 3);

    // Clean up three handles — each via a collecting sink so no G2 violation.
    let mut sink = CollectingSink::new();
    let p1 = do_drop(cell, &mut sink);
    assert!(
        matches!(p1, RcProbe::Shared),
        "still 2 live handles after first drop"
    );
    assert!(sink.is_empty(), "no record on Shared");

    let p2 = do_drop(c2, &mut sink);
    assert!(
        matches!(p2, RcProbe::Shared),
        "still 1 live handle (c3) after second drop"
    );
    assert!(sink.is_empty(), "no record on Shared");

    let p3 = do_drop(c3, &mut sink);
    assert!(
        matches!(p3, RcProbe::UniqueOwner(_)),
        "last handle → UniqueOwner"
    );
    assert_eq!(sink.len(), 1, "exactly one RcZero record on last drop (G2)");
}

#[test]
fn drop_ref_decrements_refcount() {
    // Guarantee: Exact — Shared decrements by 1.
    let cell: RcCell<i32> = RcCell::new(-5);
    let c2 = cell.clone_ref();
    // Now refcount == 2.
    assert_eq!(cell.refcount(), 2);

    let mut sink = CollectingSink::new();
    let probe = do_drop(c2, &mut sink);
    // c2 dropped — refcount should now be 1.
    assert!(matches!(probe, RcProbe::Shared));
    assert_eq!(cell.refcount(), 1, "after one drop, refcount must be 1");

    // Drop the last one.
    let probe2 = do_drop(cell, &mut sink);
    assert!(matches!(probe2, RcProbe::UniqueOwner(_)));
}

// ── 2. rc → 0 emits exactly one ReclamationRecord(RcZero) ────────────────────

#[test]
fn last_drop_emits_exactly_one_rczero_record() {
    // DoD item 2: never-silent G2 contract — first live trigger into ReclamationSink.
    // Guarantee: Exact (enforced-by-construction).
    let cell: RcCell<&str> = RcCell::new("hello");
    let mut sink = CollectingSink::new();

    let probe = cell.drop_ref(&mut sink, ScopeId(10), SweepEpoch(20), make_hash(99));

    // Exactly one record in the sink.
    assert_eq!(
        sink.len(),
        1,
        "exactly one RcZero record must be emitted on last-ref drop (G2)"
    );

    // The record has the correct trigger.
    assert_eq!(
        sink.records[0].trigger.clone(),
        ReclamationTrigger::RcZero,
        "trigger must be RcZero"
    );

    // The record has the correct scope_id and sweep_epoch.
    assert_eq!(sink.records[0].scope_id, ScopeId(10));
    assert_eq!(sink.records[0].sweep_epoch, SweepEpoch(20));

    // The record has the correct value_meta_hash.
    assert_eq!(sink.records[0].value_meta_hash, make_hash(99));

    // No channel_id (RcZero events have no channel).
    assert_eq!(sink.records[0].channel_id, None);

    // The probe is UniqueOwner.
    assert!(matches!(probe, RcProbe::UniqueOwner(_)));
}

#[test]
fn multiple_clones_only_last_drop_emits_record() {
    // DoD item 2 + 4: intermediate drops do NOT emit records; only the last does.
    let cell: RcCell<u64> = RcCell::new(42);
    let c2 = cell.clone_ref();
    let c3 = cell.clone_ref();
    // refcount == 3

    let mut sink = CollectingSink::new();

    // Drop two non-last handles — no records expected.
    let p1 = do_drop(cell, &mut sink);
    assert!(matches!(p1, RcProbe::Shared));
    assert!(sink.is_empty(), "no record on first (non-last) drop");

    let p2 = do_drop(c2, &mut sink);
    assert!(matches!(p2, RcProbe::Shared));
    assert!(sink.is_empty(), "no record on second (non-last) drop");

    // Drop the last handle — exactly one record expected.
    let p3 = do_drop(c3, &mut sink);
    assert!(matches!(p3, RcProbe::UniqueOwner(_)));
    assert_eq!(sink.len(), 1, "exactly one record on the last drop (G2)");
    assert_eq!(sink.records[0].trigger.clone(), ReclamationTrigger::RcZero);
}

// ── 3. rc == 1 is the UniqueOwner reuse opportunity ──────────────────────────

#[test]
fn sole_owner_returns_unique_owner_probe() {
    // DoD item 3: rc==1 before drop → UniqueOwner.
    // A fresh cell has refcount==1 — it is immediately the unique owner.
    let cell: RcCell<String> = RcCell::new("mycelium".to_string());
    assert_eq!(cell.refcount(), 1);

    let mut sink = CollectingSink::new();
    let probe = do_drop(cell, &mut sink);

    // Probe indicates unique owner (reuse opportunity).
    match probe {
        RcProbe::UniqueOwner(value) => {
            assert_eq!(value, "mycelium", "UniqueOwner must carry the owned value");
        }
        RcProbe::Shared => {
            panic!("expected UniqueOwner for sole-owner probe, got Shared");
        }
    }
}

#[test]
fn unique_owner_value_is_returned_correctly() {
    // DoD item 3: the caller can inspect the value in UniqueOwner for FBIP reuse.
    let cell: RcCell<Vec<i32>> = RcCell::new(vec![1, 2, 3]);
    let mut sink = CollectingSink::new();
    let probe = do_drop(cell, &mut sink);
    match probe {
        RcProbe::UniqueOwner(v) => assert_eq!(v, vec![1, 2, 3]),
        _ => panic!("expected UniqueOwner"),
    }
}

// ── 4. rc > 1 returns Shared (no record emitted) ──────────────────────────────

#[test]
fn shared_owner_returns_shared_probe_no_record() {
    // DoD item 4.
    let cell: RcCell<u8> = RcCell::new(1);
    let _keep = cell.clone_ref(); // rc == 2 now

    let mut sink = CollectingSink::new();
    let probe = do_drop(cell, &mut sink);

    assert!(matches!(probe, RcProbe::Shared), "rc>1 must return Shared");
    assert!(
        sink.is_empty(),
        "no record on Shared — only last-ref emits (G2)"
    );

    // Clean up the remaining handle.
    let mut sink2 = CollectingSink::new();
    let p2 = do_drop(_keep, &mut sink2);
    assert!(matches!(p2, RcProbe::UniqueOwner(_)));
    assert_eq!(sink2.len(), 1);
}

// ── 5. Property test — arbitrary inc/dec sequences ───────────────────────────

// Property: for any N clones (N >= 1), dropping N-1 handles returns `Shared` and emits no
// records, then dropping the Nth (last) handle returns `UniqueOwner` and emits exactly one
// `ReclamationRecord(RcZero)`. The sink accumulates exactly one record total.
//
// Guarantee: `Empirical` — property tested over 0..=9 extra clones (total 1..=10 handles).
proptest! {
    #[test]
    fn property_last_drop_emits_exactly_one_rczero_record(
        n_clones in 0usize..=9,
    ) {
        // n_clones: number of EXTRA handles beyond the original (so total handles = n_clones + 1).
        let cell: RcCell<u64> = RcCell::new(0);
        let mut handles: Vec<RcCell<u64>> = (0..n_clones).map(|_| cell.clone_ref()).collect();
        handles.push(cell); // push original last so we drain in LIFO order

        let expected_total = n_clones + 1;
        prop_assert_eq!(
            handles[0].refcount(),
            expected_total,
            "total refcount must equal number of live handles"
        );

        let mut sink = CollectingSink::new();

        // Drop all handles except the last — each must return Shared and emit no record.
        while handles.len() > 1 {
            let handle = handles.remove(0);
            let probe = do_drop(handle, &mut sink);
            prop_assert!(
                matches!(probe, RcProbe::Shared),
                "non-last drop must return Shared"
            );
            prop_assert!(sink.is_empty(), "no record emitted on non-last drop (G2)");
        }

        // Drop the last handle — must return UniqueOwner and emit exactly one RcZero record.
        let last = handles.remove(0);
        let probe = do_drop(last, &mut sink);
        prop_assert!(
            matches!(probe, RcProbe::UniqueOwner(_)),
            "last drop must return UniqueOwner"
        );
        prop_assert_eq!(
            sink.len(),
            1,
            "exactly one RcZero record must be emitted on last drop (G2 / never-silent)"
        );
        prop_assert_eq!(
            sink.records[0].trigger.clone(),
            ReclamationTrigger::RcZero,
            "trigger must be RcZero"
        );
    }

    // Property: refcount is always the number of live handles.
    #[test]
    fn property_refcount_equals_live_handle_count(
        n_clones in 0usize..=8,
    ) {
        let cell: RcCell<u32> = RcCell::new(0);
        let mut handles: Vec<RcCell<u32>> = (0..n_clones).map(|_| cell.clone_ref()).collect();
        handles.push(cell);

        let initial_count = handles.len(); // == n_clones + 1
        prop_assert_eq!(handles[0].refcount(), initial_count);

        let mut sink = CollectingSink::new();

        // Drop handles one by one, checking refcount at each step.
        let mut remaining = initial_count;
        while handles.len() > 1 {
            let handle = handles.remove(0);
            let prev_count = handle.refcount();
            prop_assert_eq!(prev_count, remaining, "refcount must equal remaining handles");
            do_drop(handle, &mut sink);
            remaining -= 1;
            prop_assert_eq!(handles[0].refcount(), remaining, "refcount must decrement by 1");
        }

        // Last handle.
        prop_assert_eq!(handles[0].refcount(), 1);
        let last = handles.remove(0);
        do_drop(last, &mut sink);
    }
}

// ── 6. Value accessor ─────────────────────────────────────────────────────────

#[test]
fn value_accessor_returns_correct_immutable_reference() {
    let cell: RcCell<u32> = RcCell::new(42);
    assert_eq!(*cell.value(), 42);

    // Shared clones see the same value.
    let c2 = cell.clone_ref();
    assert_eq!(*c2.value(), 42);
    assert!(
        std::ptr::eq(cell.value(), c2.value()),
        "both handles must share the same allocation"
    );

    let mut sink = CollectingSink::new();
    do_drop(c2, &mut sink);
    do_drop(cell, &mut sink);
    assert_eq!(sink.len(), 1, "exactly one record for one shared value");
}

// ── 7. ReclamationRecord fields in the emitted record ────────────────────────

#[test]
fn emitted_record_carries_caller_supplied_fields() {
    // The caller supplies scope_id, sweep_epoch, and value_meta_hash —
    // verify these round-trip into the emitted record unchanged.
    let scope_id = ScopeId(77);
    let sweep_epoch = SweepEpoch(999);
    let hash = make_hash(12345);

    let cell: RcCell<u8> = RcCell::new(0);
    let mut sink = CollectingSink::new();
    let _ = cell.drop_ref(&mut sink, scope_id, sweep_epoch, hash.clone());

    assert_eq!(sink.len(), 1);
    let record = &sink.records[0];
    assert_eq!(record.scope_id, scope_id, "scope_id must round-trip");
    assert_eq!(
        record.sweep_epoch, sweep_epoch,
        "sweep_epoch must round-trip"
    );
    assert_eq!(
        record.value_meta_hash, hash,
        "value_meta_hash must round-trip"
    );
    assert_eq!(record.trigger, ReclamationTrigger::RcZero);
    assert_eq!(record.channel_id, None, "RcZero events have no channel_id");
}
