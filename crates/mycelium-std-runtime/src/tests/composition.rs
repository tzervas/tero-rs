//! End-to-end composition test — L1/L2/L3 memory model + all three live triggers (E12).
//!
//! MEM-1/2/3 + the live wiring landed as separate modules:
//! - L2 ([`crate::rc`]) — `RcCell` sharing; the last-handle drop emits `RcZero`.
//! - L3 ([`crate::region`] / [`crate::scope_region`]) — a scope-region batches `ScopeExit`.
//! - network ([`crate::network`]) — channel teardown reclaims in-transit values as `ChannelClose`.
//!
//! This module proves they **compose**: one [`CollectingSink`] observes a single scope in which
//! shared values are RC-reclaimed (L2), a channel of in-transit values is torn down
//! (`ChannelClose`), the uniquely-owned survivors are deferred to the scope-region, and the
//! region closes — batching `ScopeExit`. The contract verified here is the never-silent one
//! (G2 / RFC-0027 §9): **every** reclamation event — regardless of trigger — yields exactly one
//! audit record in the sink, and the three triggers interleave without loss.
//!
//! These are integration-style assertions over the public composition surface; the per-module
//! unit tests live in their own `src/tests/<module>.rs`. The L1 (affine/uniqueness) layer is the
//! *static* analysis leg (MEM-4, deferred — DN-33); here L1 is represented by the runtime
//! `UniqueOwner` probe (the fallback the static analysis would elide), so the composition is the
//! full *runtime* model. (RFC-0027 §10.1: `UniqueOwner` IS the `rc → 0` event.)

use crate::network::Network;
use crate::rc::{RcCell, RcProbe};
use crate::reclamation::{CollectingSink, ReclamationTrigger, SweepEpoch};
use crate::scope_region::RegionScope;
use mycelium_core::ContentHash;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// A well-shaped deterministic test hash (mirrors the per-module test helpers).
fn make_hash(n: u64) -> ContentHash {
    let digest = format!("{n:064}");
    ContentHash::from_parts("blake3", &digest).expect("test hash must be well-formed")
}

/// `hash_of` mapper for `i32` channel payloads.
fn hash_of_i32(v: &i32) -> ContentHash {
    make_hash(*v as u64)
}

/// Count records in `sink` carrying `trigger`.
fn count_trigger(sink: &CollectingSink, trigger: &ReclamationTrigger) -> usize {
    sink.records
        .iter()
        .filter(|r| &r.trigger == trigger)
        .count()
}

// ── 1. All three triggers compose through one never-silent sink ──────────────

#[test]
fn three_triggers_compose_through_one_sink_never_silent() {
    // A single scope-region, a single sink. We drive every layer and assert the sink saw
    // exactly one record per reclamation event — never-silent across all three triggers (G2).
    let mut sink = CollectingSink::new();
    let mut scope = RegionScope::enter();
    let scope_id = scope.id().as_scope_id();
    // Pre-close epoch marker for the eager RC-zero events (caller-supplied; KC-3 — no global
    // state in `drop_ref`). The batched ScopeExit events get the region's real close epoch.
    let pre_close_epoch = SweepEpoch(0);

    // ── L2: explicit reference counting ──────────────────────────────────────
    // A shared value: two handles. Dropping the first is `Shared` (no record); dropping the
    // last is `UniqueOwner` + exactly one `RcZero` record.
    let shared = RcCell::new("shared-value");
    let shared_2 = shared.clone_ref(); // rc == 2
    assert_eq!(shared.refcount(), 2, "two handles share the allocation");

    match shared.drop_ref(&mut sink, scope_id, pre_close_epoch, make_hash(10)) {
        RcProbe::Shared => {} // expected: another handle remains, no record
        RcProbe::UniqueOwner(_) => panic!("rc was 2 — must be Shared, not UniqueOwner"),
    }
    assert_eq!(
        sink.len(),
        0,
        "dropping a non-last handle emits no record (L2)"
    );

    match shared_2.drop_ref(&mut sink, scope_id, pre_close_epoch, make_hash(10)) {
        RcProbe::UniqueOwner(v) => {
            assert_eq!(
                v, "shared-value",
                "the sole owner recovers the value (FBIP)"
            );
            // L2 → L3 handoff: a value that became uniquely-owned inside the scope is deferred
            // to the region for *batched* scope-exit reclamation.
            scope.defer(make_hash(10));
        }
        RcProbe::Shared => panic!("rc was 1 — must be UniqueOwner"),
    }
    assert_eq!(
        count_trigger(&sink, &ReclamationTrigger::RcZero),
        1,
        "the last-handle drop emits exactly one RcZero record (L2 / RFC-0027 §10.1)"
    );

    // A second, never-shared value: one handle, dropped → one more RcZero.
    let solo = RcCell::new(7u64);
    match solo.drop_ref(&mut sink, scope_id, pre_close_epoch, make_hash(11)) {
        RcProbe::UniqueOwner(v) => {
            assert_eq!(v, 7);
            scope.defer(make_hash(11));
        }
        RcProbe::Shared => panic!("a single handle must be UniqueOwner"),
    }
    assert_eq!(
        count_trigger(&sink, &ReclamationTrigger::RcZero),
        2,
        "two distinct sole-owner drops → two RcZero records"
    );

    // ── network: ChannelClose — reclaim in-transit values on teardown ─────────
    // A channel with 3 values still buffered (never received) is torn down: each in-transit
    // value is reclaimed as a `ChannelClose` event (never silently dropped — G2 / §7.3).
    const IN_TRANSIT: usize = 3;
    let (tx, rx) = Network::channel::<i32>(IN_TRANSIT + 1).expect("channel must construct");
    for i in 0..IN_TRANSIT as i32 {
        assert_eq!(tx.try_send(i), crate::network::TrySend::Sent);
    }
    let reclaimed = rx.close_with_reclaim(&mut sink, scope_id, pre_close_epoch, hash_of_i32);
    assert_eq!(reclaimed, IN_TRANSIT, "every in-transit value is reclaimed");
    assert_eq!(
        count_trigger(&sink, &ReclamationTrigger::ChannelClose),
        IN_TRANSIT,
        "channel teardown emits one ChannelClose record per in-transit value"
    );

    // ── L3: scope-exit batched reclamation ────────────────────────────────────
    // Close the scope: the two deferred (uniquely-owned) values are batch-reclaimed as
    // `ScopeExit`. This is the second live trigger, fired from the structured scope.
    let records_before_close = sink.len();
    let closed = scope.close(&mut sink);
    assert_eq!(
        closed.reclaimed_count, 2,
        "two deferred entries batched at close"
    );
    assert_eq!(
        count_trigger(&sink, &ReclamationTrigger::ScopeExit),
        2,
        "scope-exit emits one ScopeExit record per deferred entry (L3)"
    );
    assert_eq!(
        sink.len() - records_before_close,
        2,
        "close added exactly the batch — nothing silently dropped or duplicated"
    );

    // ── Never-silent accounting: every event produced exactly one record ─────
    // 2 RcZero + 3 ChannelClose + 2 ScopeExit = 7. No reclamation was silent (G2).
    assert_eq!(
        sink.len(),
        2 + IN_TRANSIT + 2,
        "total records == sum of every reclamation event across all three triggers (G2)"
    );
    // The scope-exit batch closes the audit: the closed region's epoch is the real sweep epoch.
    assert!(
        closed.epoch.as_u64() > 0,
        "the region close allocates a real monotonic sweep epoch"
    );
}

// ── 2. Nested scopes compose with RC reclamation; child batch precedes parent ─

#[test]
fn nested_scopes_compose_with_rc_child_batch_precedes_parent() {
    // A parent scope holds a child scope. RC-reclaimed values are deferred to whichever scope
    // owns them; the child closes first (lower epoch), then the parent. This is the L2-feeds-L3
    // story across the scope tree (RFC-0027 §10.3): child→parent total order.
    let mut sink = CollectingSink::new();

    let mut parent = RegionScope::enter();
    let parent_id = parent.id().as_scope_id();

    // A value reclaimed in the parent scope.
    let p_val = RcCell::new(1u8);
    if let RcProbe::UniqueOwner(_) =
        p_val.drop_ref(&mut sink, parent_id, SweepEpoch(0), make_hash(1))
    {
        parent.defer(make_hash(1));
    }

    // Child scope: a value reclaimed inside it, deferred to the child.
    let mut child = RegionScope::enter();
    let child_id = child.id().as_scope_id();
    let c_val = RcCell::new(2u8);
    if let RcProbe::UniqueOwner(_) =
        c_val.drop_ref(&mut sink, child_id, SweepEpoch(0), make_hash(2))
    {
        child.defer(make_hash(2));
    }

    // Child closes BEFORE parent → numerically lower epoch (child→parent order).
    let child_closed = child.close(&mut sink);
    let parent_closed = parent.close(&mut sink);

    assert!(
        child_closed.epoch < parent_closed.epoch,
        "child scope ({}) must close before parent ({}) — child→parent total order",
        child_closed.epoch.as_u64(),
        parent_closed.epoch.as_u64()
    );
    assert_eq!(
        count_trigger(&sink, &ReclamationTrigger::RcZero),
        2,
        "both values were sole-owner reclaimed (2 RcZero)"
    );
    assert_eq!(
        count_trigger(&sink, &ReclamationTrigger::ScopeExit),
        2,
        "both scopes batched their single deferred entry (2 ScopeExit)"
    );
}
