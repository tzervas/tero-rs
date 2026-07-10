//! Tests for `crate::network` — ADR-020 v0 R1 / RFC-0027 §9 / §7.3.
//!
//! M-797 in-crate test layout: all tests live here, not inline in `network.rs`.
//! Uses `use crate::network::*;` and `use crate::reclamation::*;` for white-box access.
//!
//! # Coverage
//!
//! 1. **Zero-capacity fail-closed (G2):** `Network::channel(0)` returns `Err(ZeroCapacity)`.
//! 2. **FIFO send/recv roundtrip (`Exact`):** 3 sends → 3 receives in order.
//! 3. **Buffer-full returns `Full(v)` (G2):** capacity=1; second send returns the value.
//! 4. **Closed channel returns `Closed` after drain (`Exact`):** `Sender::close` then drain.
//! 5. **FIFO is exact — property:** ∀i, send[i] == recv[i].
//! 6. **`close_with_reclaim` emits N `ChannelClose` records for N buffered values (`Exact`).**
//! 7. **`close_with_reclaim` on empty channel emits 0 records and returns 0.**
//! 8. **Two channels get distinct `ChannelNodeId`s (monotonic uniqueness, `Exact`).**
//! 9. **`value_meta_hash` in emitted records matches `hash_of` applied to sent values in FIFO order.**
//! 10. **`channel_id()` accessor is consistent between `Sender` and `Receiver`.**

use crate::network::{ChannelError, Network, TryRecv, TrySend};
use crate::reclamation::{CollectingSink, ReclamationTrigger, ScopeId, SweepEpoch};
use mycelium_core::ContentHash;

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Build a deterministic, well-shaped BLAKE3-shaped test hash from a counter.
/// The real BLAKE3 hex is 64 chars; we use a padded representation here to satisfy
/// the `ContentHash::from_parts` shape validator (algo=[a-z0-9]+, digest=[A-Za-z0-9_-]+).
fn make_hash(n: u64) -> ContentHash {
    let digest = format!("{:064}", n);
    ContentHash::from_parts("blake3", &digest)
        .expect("test hash must be well-formed (64-digit zero-padded u64)")
}

/// `hash_of` closure for `i32` values — maps value to a deterministic `ContentHash`.
fn hash_of_i32(v: &i32) -> ContentHash {
    make_hash(*v as u64)
}

// ── 1. Zero-capacity fail-closed ──────────────────────────────────────────────

#[test]
fn test_channel_zero_capacity_fails() {
    // Mutant witness: removing the zero-capacity check would make this return Ok(_),
    // causing unwrap_err() to panic with a success value.
    let err = Network::channel::<i32>(0).unwrap_err();
    assert_eq!(
        err,
        ChannelError::ZeroCapacity,
        "zero-capacity channel must fail closed with ZeroCapacity (G2)"
    );
}

// ── 2. FIFO send/recv roundtrip ───────────────────────────────────────────────

#[test]
fn test_channel_send_recv_roundtrip() {
    // Send 3 values; receive them in FIFO order.
    // Mutant witness: if VecDeque were replaced with a LIFO stack, order would reverse.
    let (tx, rx) = Network::channel::<i32>(8).expect("channel creation must succeed");
    assert_eq!(tx.try_send(10), TrySend::Sent);
    assert_eq!(tx.try_send(20), TrySend::Sent);
    assert_eq!(tx.try_send(30), TrySend::Sent);
    assert_eq!(rx.try_recv(), TryRecv::Received(10));
    assert_eq!(rx.try_recv(), TryRecv::Received(20));
    assert_eq!(rx.try_recv(), TryRecv::Received(30));
    assert_eq!(
        rx.try_recv(),
        TryRecv::Empty,
        "buffer must be empty after draining"
    );
}

// ── 3. Buffer-full returns Full(v) ────────────────────────────────────────────

#[test]
fn test_channel_full_returns_full() {
    // capacity=1; second send must return Full with the value.
    // Mutant witness: if capacity check were removed, both sends would succeed and
    // the second try_send would return Sent instead of Full.
    let (tx, rx) = Network::channel::<i32>(1).expect("channel creation must succeed");
    assert_eq!(tx.try_send(42), TrySend::Sent);
    assert_eq!(
        tx.try_send(99),
        TrySend::Full(99),
        "try_send must return Full(value) when buffer is at capacity"
    );
    // The original value is still in the buffer.
    assert_eq!(rx.try_recv(), TryRecv::Received(42));
}

// ── 4. Closed channel returns Closed after drain ──────────────────────────────

#[test]
fn test_channel_closed_receiver_returns_closed() {
    // Close sender, drain buffer, then try_recv must return Closed.
    // Mutant witness: if Sender::close did not set closed=true, try_recv would return Empty.
    let (tx, rx) = Network::channel::<i32>(4).expect("channel creation must succeed");
    assert_eq!(tx.try_send(1), TrySend::Sent);
    tx.close();
    // Drain the one buffered value.
    assert_eq!(rx.try_recv(), TryRecv::Received(1));
    // Now buffer is empty and channel is closed.
    assert_eq!(
        rx.try_recv(),
        TryRecv::Closed,
        "drained + closed channel must return Closed, not Empty"
    );
}

// ── 5. FIFO is exact (property) ───────────────────────────────────────────────

#[test]
fn test_channel_fifo_is_exact() {
    // 5 sends followed by 5 receives; result order must match send order.
    // Property: ∀i ∈ 0..5, send[i] == recv[i].
    // Mutant witness: if the channel used a random/priority queue, this test would fail.
    let sends: Vec<i32> = (0..5).collect();
    let (tx, rx) = Network::channel::<i32>(8).expect("channel creation must succeed");
    for &v in &sends {
        assert_eq!(tx.try_send(v), TrySend::Sent);
    }
    for (i, &expected) in sends.iter().enumerate() {
        match rx.try_recv() {
            TryRecv::Received(got) => assert_eq!(
                got, expected,
                "FIFO violation at index {i}: expected {expected}, got {got}"
            ),
            other => panic!("expected Received at index {i}, got {other:?}"),
        }
    }
}

// ── 6. close_with_reclaim emits N ChannelClose records ───────────────────────

#[test]
fn close_with_reclaim_emits_n_channel_close_records() {
    // DoD: N buffered values → N `ChannelClose` records emitted, count returned == N.
    // Guarantee: `Exact` — enforced-by-construction (drain + emit per value).
    // Mutant witness: if drain loop were skipped, sink.records would be empty.
    const N: usize = 5;
    let (tx, rx) = Network::channel::<i32>(N + 1).expect("channel creation must succeed");

    // Record which channel id we expect in the records.
    let expected_channel_id = rx.channel_id().as_channel_id();

    for i in 0..N {
        assert_eq!(tx.try_send(i as i32), TrySend::Sent);
    }

    let mut sink = CollectingSink::new();
    let count = rx.close_with_reclaim(&mut sink, ScopeId(1), SweepEpoch(1), hash_of_i32);

    // Returned count == N.
    assert_eq!(
        count, N,
        "close_with_reclaim must return the reclaimed count"
    );

    // Exactly N records emitted.
    assert_eq!(
        sink.records.len(),
        N,
        "exactly N ChannelClose records must be emitted (G2 / never-silent)"
    );

    // Every record has trigger == ChannelClose and the correct channel_id.
    for (i, record) in sink.records.iter().enumerate() {
        assert_eq!(
            record.trigger,
            ReclamationTrigger::ChannelClose,
            "record {i} must have trigger ChannelClose"
        );
        assert_eq!(
            record.channel_id,
            Some(expected_channel_id),
            "record {i} must carry the channel's ChannelId"
        );
    }
}

// ── 7. close_with_reclaim on empty channel emits 0 records ───────────────────

#[test]
fn close_with_reclaim_on_empty_channel_emits_zero_records() {
    // No buffered values → no records, count == 0.
    // Mutant witness: if the emit loop ran unconditionally, count would be non-zero.
    let (_tx, rx) = Network::channel::<i32>(4).expect("channel creation must succeed");
    let mut sink = CollectingSink::new();
    let count = rx.close_with_reclaim(&mut sink, ScopeId(0), SweepEpoch(0), hash_of_i32);
    assert_eq!(count, 0, "empty channel: reclaimed count must be 0");
    assert!(sink.is_empty(), "empty channel: no records must be emitted");
}

// ── 8. Two channels get distinct ChannelNodeIds ───────────────────────────────

#[test]
fn two_channels_get_distinct_channel_node_ids() {
    // Monotonic uniqueness: each `Network::channel` call allocates a fresh ChannelNodeId.
    // Guarantee: `Exact` — atomic counter, strictly increasing.
    // Mutant witness: if allocate() returned a constant, ids would be equal.
    let (tx1, rx1) = Network::channel::<i32>(4).expect("channel 1 must succeed");
    let (tx2, rx2) = Network::channel::<i32>(4).expect("channel 2 must succeed");

    let id1 = rx1.channel_id();
    let id2 = rx2.channel_id();

    assert_ne!(id1, id2, "two channels must have distinct ChannelNodeIds");
    assert!(
        id2.as_u64() > id1.as_u64(),
        "ChannelNodeId must be monotonically increasing: id2={} must be > id1={}",
        id2.as_u64(),
        id1.as_u64()
    );

    // Sender and receiver for the same channel share the same id.
    assert_eq!(
        tx1.channel_id(),
        rx1.channel_id(),
        "Sender and Receiver for the same channel must have the same ChannelNodeId"
    );
    assert_eq!(
        tx2.channel_id(),
        rx2.channel_id(),
        "Sender and Receiver for the same channel must have the same ChannelNodeId"
    );

    // Suppress unused-variable warnings (tx1/tx2 dropped here intentionally).
    drop(tx1);
    drop(tx2);
}

// ── 9. value_meta_hash matches hash_of in FIFO order ─────────────────────────

#[test]
fn close_with_reclaim_value_meta_hashes_match_in_fifo_order() {
    // The `value_meta_hash` in each emitted record must equal `hash_of` applied to the
    // originally-sent value, in FIFO order.
    // Guarantee: `Exact` — values are drained in FIFO order; `hash_of` is applied to each.
    // Mutant witness: reversing the drain order would cause a mismatch at index 0.
    let values: Vec<i32> = vec![10, 20, 30, 40];
    let (tx, rx) = Network::channel::<i32>(values.len() + 1).expect("channel must succeed");

    for &v in &values {
        assert_eq!(tx.try_send(v), TrySend::Sent);
    }

    let mut sink = CollectingSink::new();
    let count = rx.close_with_reclaim(&mut sink, ScopeId(7), SweepEpoch(3), hash_of_i32);

    assert_eq!(count, values.len());
    assert_eq!(sink.records.len(), values.len());

    for (i, (record, &original)) in sink.records.iter().zip(values.iter()).enumerate() {
        let expected_hash = hash_of_i32(&original);
        assert_eq!(
            record.value_meta_hash, expected_hash,
            "record {i}: value_meta_hash must match hash_of applied to the sent value in FIFO order"
        );
    }
}

// ── 10. channel_id() accessor consistency ────────────────────────────────────

#[test]
fn channel_id_accessor_consistent_across_sender_and_receiver() {
    // Both Sender::channel_id() and Receiver::channel_id() must return the same ChannelNodeId.
    // Guarantee: `Exact` — both share the same Arc<Mutex<ChannelInner>> with one channel_id field.
    let (tx, rx) = Network::channel::<u64>(8).expect("channel creation must succeed");
    assert_eq!(
        tx.channel_id(),
        rx.channel_id(),
        "Sender and Receiver must report the same ChannelNodeId"
    );
    // channel_id is non-zero (0 is reserved as "no channel").
    assert_ne!(
        tx.channel_id().as_u64(),
        0,
        "ChannelNodeId must be non-zero (0 is reserved)"
    );
}

// ── 11. ChannelNodeId type properties ────────────────────────────────────────

#[test]
fn channel_node_id_as_channel_id_roundtrips_u64() {
    // as_channel_id() is a lossless bridge: the underlying u64 is preserved.
    // Guarantee: `Exact` (lossless conversion, by construction).
    let (tx, rx) = Network::channel::<i32>(2).expect("channel creation must succeed");
    let node_id = rx.channel_id();
    let channel_id = node_id.as_channel_id();
    assert_eq!(
        channel_id.0,
        node_id.as_u64(),
        "as_channel_id must preserve the u64 value exactly"
    );
    drop(tx);
}
