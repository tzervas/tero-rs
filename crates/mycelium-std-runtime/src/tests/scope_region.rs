//! Tests for `crate::scope_region` — live-executor scope/region wiring (MEM-3).
//!
//! M-797 in-crate test layout: all tests live here, not in `scope_region.rs`.
//!
//! # DoD coverage
//!
//! 1. **`with_region` N deferrals → N `ScopeExit` records:** `with_region` deferring N values
//!    emits exactly N `ScopeExit` records through `CollectingSink`; `ClosedRegion.reclaimed_count
//!    == N`; the body return value is propagated.
//! 2. **`with_region` zero deferrals:** emits zero records — empty scope close is silent-correct.
//! 3. **Nested `with_region`:** inner closes before outer → `inner_closed.epoch < outer_closed.epoch`
//!    (child→parent total order by construction); records appear inner-before-outer.
//! 4. **`RegionScope` explicit close:** `enter` → `defer` × K → `close` returns a `ClosedRegion`
//!    with `reclaimed_count == K`; emits K `ScopeExit` records; `deferred_count` tracks correctly.

use crate::reclamation::{CollectingSink, ReclamationTrigger};
use crate::scope_region::{with_region, RegionScope};
use mycelium_core::ContentHash;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a test `ContentHash` from an integer discriminator.
///
/// Uses `blake3:<64-digit decimal>` which satisfies `ContentHash::parse`'s shape rules.
/// Guarantee: `Exact` — deterministic from the input `u64`.
fn make_hash(n: u64) -> ContentHash {
    let digest = format!("{:064}", n);
    ContentHash::from_parts("blake3", &digest).expect("test hash must be well-formed")
}

/// Assert that every record in `sink` has trigger `ScopeExit`.
fn assert_all_scope_exit(sink: &CollectingSink) {
    for (i, record) in sink.records.iter().enumerate() {
        assert_eq!(
            record.trigger,
            ReclamationTrigger::ScopeExit,
            "record {i} must have trigger ScopeExit"
        );
    }
}

// ── 1. with_region — N deferrals → N ScopeExit records ───────────────────────

#[test]
fn with_region_one_deferral_emits_one_scope_exit_record() {
    // DoD item 1: one deferred value → exactly one ScopeExit record; body result propagated.
    let mut sink = CollectingSink::new();
    let (body_result, closed) = with_region(&mut sink, |region| {
        region.defer(make_hash(1));
        42usize
    });

    assert_eq!(body_result, 42, "body return value must be propagated");
    assert_eq!(sink.len(), 1, "one deferred entry → one record (G2)");
    assert_eq!(closed.reclaimed_count, 1, "reclaimed_count must be 1");
    assert_all_scope_exit(&sink);
}

#[test]
fn with_region_five_deferrals_emits_five_scope_exit_records() {
    // DoD item 1: N deferred values → exactly N records.
    let mut sink = CollectingSink::new();
    let (body_result, closed) = with_region(&mut sink, |region| {
        for i in 0..5 {
            region.defer(make_hash(i as u64));
        }
        "done"
    });

    assert_eq!(body_result, "done", "body return value must be propagated");
    assert_eq!(sink.len(), 5, "five deferred entries → five records (G2)");
    assert_eq!(closed.reclaimed_count, 5, "reclaimed_count must be 5");
    assert_all_scope_exit(&sink);
}

#[test]
fn with_region_reclaimed_count_matches_sink_len() {
    // DoD item 1: ClosedRegion.reclaimed_count == sink.len() for any N.
    let n = 7;
    let mut sink = CollectingSink::new();
    let (_result, closed) = with_region(&mut sink, |region| {
        for i in 0..n {
            region.defer(make_hash(i as u64));
        }
    });

    assert_eq!(
        closed.reclaimed_count, n,
        "reclaimed_count must equal the number of deferred entries"
    );
    assert_eq!(sink.len(), n, "sink len must equal reclaimed_count");
}

#[test]
fn with_region_body_return_value_is_propagated() {
    // DoD item 1: the tuple fst is the body's return value.
    let mut sink = CollectingSink::new();
    let (v, _closed) = with_region(&mut sink, |_region| vec![1u8, 2, 3]);
    assert_eq!(
        v,
        vec![1u8, 2, 3],
        "complex body return value must be propagated"
    );
}

// ── 2. with_region — zero deferrals ──────────────────────────────────────────

#[test]
fn with_region_zero_deferrals_emits_zero_records() {
    // DoD item 2: empty scope close is silent-correct — no records, no noise.
    let mut sink = CollectingSink::new();
    let ((), closed) = with_region(&mut sink, |_region| {
        // no deferrals
    });

    assert!(
        sink.is_empty(),
        "empty scope must emit 0 records (silent-correct)"
    );
    assert_eq!(closed.reclaimed_count, 0, "reclaimed_count must be 0");
    // The epoch is still allocated — the close event is observable even when nothing is reclaimed.
    assert!(
        closed.epoch.as_u64() > 0,
        "epoch must be allocated even for empty with_region"
    );
}

// ── 3. Nested with_region ─────────────────────────────────────────────────────
//
// Nesting model: an inner scope (child) closes before the outer scope (parent). In the
// structured-concurrency model each scope carries its own sink; the parent sink aggregates
// records after the child closes. This matches the per-hypha model (RFC-0027 §10.3): each
// scope has its own `ReclamationSink` routed to the supervision policy's observability sink.
//
// Rust's borrow checker prevents two `&mut` borrows of the same `CollectingSink` at the same
// time (one for the outer closure, one for the inner call inside it). The correct test uses
// either (a) a separate sink for the inner scope, or (b) `RegionScope` for the inner scope
// so closures are not involved. Both are demonstrated below.
//
// The epoch-ordering invariant (inner.epoch < outer.epoch) holds regardless of which sink
// strategy is used, because it derives from the global monotonic counter in `region.rs`.

#[test]
fn nested_with_region_inner_epoch_less_than_outer_epoch() {
    // DoD item 3: inner closes before outer → inner_closed.epoch < outer_closed.epoch.
    // This is the child→parent total order by construction (monotonic counter, Exact).
    //
    // Implementation note: the inner and outer scopes use separate sinks — each scope in
    // the structured-concurrency model has its own reclamation sink (RFC-0027 §10.3). The
    // epoch-ordering invariant is independent of which sink is used.
    let mut outer_sink = CollectingSink::new();
    let mut inner_sink = CollectingSink::new();

    // Open the outer scope guard explicitly so we can hold both it and the inner call.
    let mut outer_scope = RegionScope::enter();
    outer_scope.defer(make_hash(100));

    // Inner scope — closes first (allocates a lower epoch).
    let (inner_result, inner_closed) = with_region(&mut inner_sink, |inner_region| {
        inner_region.defer(make_hash(200));
        "inner"
    });
    assert_eq!(inner_result, "inner");

    // Outer scope closes after the inner scope — gets a higher epoch.
    let outer_closed = outer_scope.close(&mut outer_sink);

    assert!(
        inner_closed.epoch < outer_closed.epoch,
        "inner epoch ({}) must be less than outer epoch ({}) — child closes before parent (Exact)",
        inner_closed.epoch.as_u64(),
        outer_closed.epoch.as_u64()
    );

    // Each sink has exactly the records for its own scope.
    assert_eq!(inner_sink.len(), 1, "inner scope emits its own record");
    assert_eq!(outer_sink.len(), 1, "outer scope emits its own record");
    assert_all_scope_exit(&inner_sink);
    assert_all_scope_exit(&outer_sink);
}

#[test]
fn nested_with_region_inner_records_precede_outer_records_in_shared_sink() {
    // DoD item 3: when a shared collecting sink is used (via RegionScope for the outer scope),
    // records appear in inner-before-outer order because the inner scope closes first.
    //
    // We pass the same sink to both the inner call and the outer close.
    // Rust allows this because RegionScope is not a closure — the borrow ends before close().
    let mut sink = CollectingSink::new();

    let inner_hash = make_hash(10);
    let outer_hash = make_hash(20);

    // Open outer scope first (no closure borrow).
    let mut outer_scope = RegionScope::enter();
    outer_scope.defer(outer_hash.clone());

    // Inner scope closes into the shared sink first.
    with_region(&mut sink, |inner_region| {
        inner_region.defer(inner_hash.clone());
    });

    // Outer scope closes into the same sink second.
    outer_scope.close(&mut sink);

    // Inner closed first → inner's record (hash=10) appears before outer's (hash=20).
    assert_eq!(sink.len(), 2, "both scopes must emit their records");
    assert_eq!(
        sink.records[0].value_meta_hash, inner_hash,
        "inner record must appear first in the sink"
    );
    assert_eq!(
        sink.records[1].value_meta_hash, outer_hash,
        "outer record must appear second in the sink"
    );
}

// ── 4. RegionScope explicit close ─────────────────────────────────────────────

#[test]
fn region_scope_enter_defer_close_emits_records_and_returns_closed_region() {
    // DoD item 4: enter → defer × K → close returns ClosedRegion with reclaimed_count == K
    // and emits K ScopeExit records.
    let k = 4;
    let mut sink = CollectingSink::new();
    let mut scope = RegionScope::enter();

    for i in 0..k {
        scope.defer(make_hash(i as u64));
    }

    // deferred_count tracks correctly before close.
    assert_eq!(
        scope.deferred_count(),
        k,
        "deferred_count must equal K before close"
    );

    let closed = scope.close(&mut sink);

    assert_eq!(
        closed.reclaimed_count, k,
        "reclaimed_count must equal K after close"
    );
    assert_eq!(sink.len(), k, "sink must have K records after close");
    assert_all_scope_exit(&sink);
}

#[test]
fn region_scope_deferred_count_tracks_each_defer() {
    // DoD item 4: deferred_count increments correctly with each defer call.
    let mut scope = RegionScope::enter();

    assert_eq!(scope.deferred_count(), 0, "starts at 0");
    scope.defer(make_hash(1));
    assert_eq!(scope.deferred_count(), 1, "after 1 defer");
    scope.defer(make_hash(2));
    assert_eq!(scope.deferred_count(), 2, "after 2 defers");
    scope.defer(make_hash(3));
    assert_eq!(scope.deferred_count(), 3, "after 3 defers");

    let mut sink = CollectingSink::new();
    let closed = scope.close(&mut sink);
    assert_eq!(closed.reclaimed_count, 3);
}

#[test]
fn region_scope_id_is_stable_across_defer_calls() {
    // The ScopeNodeId assigned at enter() does not change after defer calls.
    let mut scope = RegionScope::enter();
    let id_at_enter = scope.id();

    scope.defer(make_hash(42));
    scope.defer(make_hash(43));

    assert_eq!(
        scope.id(),
        id_at_enter,
        "scope id must be stable across defer calls"
    );

    let mut sink = CollectingSink::new();
    let closed = scope.close(&mut sink);

    // The ClosedRegion carries the same ScopeNodeId.
    assert_eq!(
        closed.id, id_at_enter,
        "ClosedRegion.id must match the id at enter"
    );
}

#[test]
fn region_scope_zero_deferrals_emits_zero_records() {
    // Edge case: empty RegionScope is valid — closes without emitting records.
    let mut sink = CollectingSink::new();
    let scope = RegionScope::enter();
    let closed = scope.close(&mut sink);

    assert!(sink.is_empty(), "empty RegionScope must emit 0 records");
    assert_eq!(closed.reclaimed_count, 0);
}

#[test]
fn region_scope_records_carry_correct_scope_id() {
    // Records emitted by close must carry the scope's id (as ScopeId).
    let mut sink = CollectingSink::new();
    let mut scope = RegionScope::enter();
    let scope_id = scope.id();

    scope.defer(make_hash(99));
    let _closed = scope.close(&mut sink);

    assert_eq!(sink.len(), 1);
    assert_eq!(
        sink.records[0].scope_id,
        scope_id.as_scope_id(),
        "emitted record must carry the scope's ScopeNodeId as ScopeId"
    );
}
