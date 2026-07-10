//! Network, Sender, Receiver, TrySend, TryRecv — channel surface (ADR-020 v0 R1).
//!
//! This module also carries the **third live reclamation trigger** (`ChannelClose` — RFC-0027 §9 /
//! §7.3 / G2) and the **canonical `ChannelNodeId`** that resolves the MEM-1 `ChannelId` placeholder
//! FLAG. When a `Receiver` is torn down via [`Receiver::close_with_reclaim`] while values are still
//! buffered, each in-transit value emits one `ReclamationRecord(ChannelClose)` through the supplied
//! [`ReclamationSink`] — never silently dropped (G2).
//!
//! # Guarantee (Empirical — Kahn-determinism of channel-mediated communication)
//!
//! Message ordering within a single channel is **Exact** (FIFO).
//! Cross-channel Kahn-determinism is **Empirical**: grounded in the RT2 differentials
//! (ADR-020 §4) but not yet Proven with a formal theorem.
//!
//! # Fail-closed on invalid input (G2)
//!
//! `Network::channel(0)` returns `Err(ChannelError::ZeroCapacity)` — zero-capacity channels
//! are nonsensical and are rejected at construction time, not silently converted to a
//! placeholder (ADR-020 §4 / G2: never-silent principle).
//!
//! # ChannelClose semantics (RFC-0027 §9 / §7.3 / G2)
//!
//! When a channel is torn down while values are still buffered (sent but never to be received),
//! those in-transit values' ownership is released and MUST be reclaimed. Use
//! [`Receiver::close_with_reclaim`] as the teardown path to emit one
//! `ReclamationRecord(ChannelClose)` per buffered value. **Normal drain** (via [`Receiver::try_recv`])
//! delivers values to the receiver — no reclamation needed; `close_with_reclaim` is the teardown
//! path for **undelivered** in-transit values only.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mycelium_core::{ContentHash, GuaranteeStrength};

use crate::reclamation::{ChannelId, ReclamationRecord, ReclamationSink, ScopeId, SweepEpoch};

/// Guarantee strength for single-channel FIFO ordering.
pub const CHANNEL_FIFO_STRENGTH: GuaranteeStrength = GuaranteeStrength::Exact;

/// Guarantee strength for cross-channel Kahn-determinism.
pub const KAHN_DETERMINISM_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

// ── Global monotonic counter for ChannelNodeId allocation ────────────────────
//
// Mirrors the `GLOBAL_SCOPE_ID` pattern in `region.rs` (ScopeNodeId). One counter, allocated
// per `Network::channel` call. Uses `Relaxed` ordering — uniqueness, not cross-thread
// happens-before, is the requirement. Guarantee: `Exact` — `fetch_add(1, Relaxed)` is strictly
// monotonic and unique within a process lifetime. (Counter wrap at u64::MAX is `Declared` as a
// known corner-case; 585 years at 10^9/s — same argument as `region.rs`.)

static GLOBAL_CHANNEL_ID: AtomicU64 = AtomicU64::new(1); // start at 1; 0 reserved as "no channel"

// ── ChannelNodeId — canonical channel-tier identity (resolves MEM-1 FLAG) ────

/// The canonical channel-tier identity type (resolves the MEM-1 `ChannelId` placeholder FLAG).
///
/// Allocated per [`Network::channel`] call via a monotonic global counter. Unique within a
/// process lifetime. Mirrors [`crate::region::ScopeNodeId`] — same pattern, same semantics,
/// different tier. Use [`ChannelNodeId::as_channel_id`] to convert to [`ChannelId`] for use in
/// [`ReclamationRecord`] fields (the stable RFC-0027 §9 type).
///
/// Guarantee: `Exact` — monotonic, unique per allocation (atomic counter, strictly increasing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelNodeId(pub(crate) u64);

impl ChannelNodeId {
    /// Allocate a fresh, unique `ChannelNodeId`.
    ///
    /// Guarantee: `Exact` — strictly greater than all previously allocated ids.
    fn allocate() -> Self {
        ChannelNodeId(GLOBAL_CHANNEL_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Convert to a [`ChannelId`] for use in [`ReclamationRecord`] fields (RFC-0027 §9).
    ///
    /// This is the canonical bridge between the channel-tier allocation identity and the
    /// MEM-1 record field type. The underlying `u64` is preserved losslessly.
    ///
    /// Guarantee: `Exact` — lossless conversion.
    #[must_use]
    pub fn as_channel_id(self) -> ChannelId {
        ChannelId(self.0)
    }

    /// Return the underlying `u64` value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ── Channel inner state ───────────────────────────────────────────────────────

/// Shared mutable state behind a `Sender`/`Receiver` pair.
struct ChannelInner<V> {
    buf: VecDeque<V>,
    capacity: usize,
    closed: bool,
    /// The canonical channel identity, allocated once at construction.
    channel_id: ChannelNodeId,
}

impl<V> ChannelInner<V> {
    fn new(capacity: usize, channel_id: ChannelNodeId) -> Self {
        ChannelInner {
            buf: VecDeque::with_capacity(capacity),
            capacity,
            closed: false,
            channel_id,
        }
    }
}

// ── Error types ───────────────────────────────────────────────────────────────

/// Errors returned by `Network` construction operations.
#[derive(Debug, PartialEq, Eq)]
pub enum ChannelError {
    /// A zero-capacity channel is nonsensical; rejected at construction (G2: fail-closed).
    ZeroCapacity,
}

// ── Network ───────────────────────────────────────────────────────────────────

/// A named network of typed channels within a `Colony`.
///
/// Guarantee: **Empirical** (Kahn-determinism, ADR-020 §4).
#[derive(Debug)]
pub struct Network {
    _priv: (),
}

impl Network {
    /// Create a new network.
    ///
    /// Guarantee: **Exact** (constructor, trivially correct).
    pub fn new() -> Self {
        Network { _priv: () }
    }

    /// Create a bounded FIFO channel with the given capacity.
    ///
    /// Allocates a fresh [`ChannelNodeId`] for the channel pair (monotonic, unique). Both
    /// `Sender` and `Receiver` share the same id, accessible via `channel_id()`.
    ///
    /// Returns `Err(ChannelError::ZeroCapacity)` if `capacity == 0` (fail-closed, G2:
    /// invalid input is never silently accepted — a zero-capacity channel cannot store any
    /// value and would make every `try_send` return `Full`, which is nonsensical).
    ///
    /// Guarantee: **Exact** (construction is deterministic; the zero-capacity check is
    /// deterministic — mutant witness: removing the check makes `test_channel_zero_capacity_fails`
    /// fail).
    pub fn channel<V>(capacity: usize) -> Result<(Sender<V>, Receiver<V>), ChannelError> {
        if capacity == 0 {
            // Fail-closed: zero-capacity is an explicit error, not a silent stub (G2).
            return Err(ChannelError::ZeroCapacity);
        }
        let channel_id = ChannelNodeId::allocate();
        let inner = Arc::new(Mutex::new(ChannelInner::new(capacity, channel_id)));
        Ok((
            Sender {
                inner: Arc::clone(&inner),
            },
            Receiver { inner },
        ))
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sender ────────────────────────────────────────────────────────────────────

/// Sending end of a typed channel.
///
/// SPSC by design: `Sender<V>` is not `Clone` (ADR-020 §4 / RFC-0008 §4.3 RT1).
///
/// Guarantee: **Exact** (FIFO ordering within this channel; backed by `VecDeque`).
pub struct Sender<V> {
    inner: Arc<Mutex<ChannelInner<V>>>,
}

impl<V> std::fmt::Debug for Sender<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("Sender")
            .field("buf_len", &inner.buf.len())
            .field("capacity", &inner.capacity)
            .field("closed", &inner.closed)
            .finish()
    }
}

impl<V> Sender<V> {
    /// Non-blocking send. Returns `TrySend::Sent` on success, `TrySend::Full(v)` if the
    /// buffer is at capacity, or `TrySend::Closed(v)` if the channel is closed.
    ///
    /// The value is **always returned on failure** — never dropped silently (G2 / ADR-020 §4).
    ///
    /// Guarantee: **Exact** (FIFO push into `VecDeque`; deterministic given the buffer state).
    pub fn try_send(&self, value: V) -> TrySend<V> {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return TrySend::Closed(value);
        }
        if inner.buf.len() >= inner.capacity {
            return TrySend::Full(value);
        }
        inner.buf.push_back(value);
        TrySend::Sent
    }

    /// Close the channel from the sender side.
    ///
    /// After this, `try_recv` on a drained buffer returns `TryRecv::Closed`. Values already
    /// buffered are still deliverable via normal `try_recv` drain — the receiver can drain them
    /// normally (no reclamation needed: the values are delivered, not lost). Only use
    /// [`Receiver::close_with_reclaim`] for the teardown path where **undelivered** in-transit
    /// values must be reclaimed (RFC-0027 §9 / §7.3 / G2).
    ///
    /// Guarantee: **Exact** (sets a boolean flag, deterministic).
    pub fn close(self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
    }

    /// The canonical channel identity shared by this `Sender` and its paired `Receiver`.
    ///
    /// Guarantee: **Exact** — allocated once at channel construction; the same id is returned
    /// by both the `Sender` and `Receiver` ends.
    #[must_use]
    pub fn channel_id(&self) -> ChannelNodeId {
        self.inner.lock().unwrap().channel_id
    }
}

// ── Receiver ──────────────────────────────────────────────────────────────────

/// Receiving end of a typed channel.
///
/// Guarantee: **Exact** (FIFO ordering within this channel; backed by `VecDeque`).
pub struct Receiver<V> {
    inner: Arc<Mutex<ChannelInner<V>>>,
}

impl<V> std::fmt::Debug for Receiver<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("Receiver")
            .field("buf_len", &inner.buf.len())
            .field("capacity", &inner.capacity)
            .field("closed", &inner.closed)
            .finish()
    }
}

impl<V> Receiver<V> {
    /// Non-blocking receive. Returns `TryRecv::Received(v)` if a message is buffered,
    /// `TryRecv::Empty` if the buffer is empty and the channel is still open, or
    /// `TryRecv::Closed` if the channel is closed and the buffer is drained.
    ///
    /// This is the **normal drain path** — values are delivered to the receiver, no
    /// reclamation is needed. For the teardown path (values in transit that will never be
    /// delivered), use [`Receiver::close_with_reclaim`].
    ///
    /// Guarantee: **Exact** (FIFO pop from `VecDeque`; deterministic given the buffer state).
    pub fn try_recv(&self) -> TryRecv<V> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(v) = inner.buf.pop_front() {
            return TryRecv::Received(v);
        }
        if inner.closed {
            TryRecv::Closed
        } else {
            TryRecv::Empty
        }
    }

    /// The canonical channel identity shared by this `Receiver` and its paired `Sender`.
    ///
    /// Guarantee: **Exact** — allocated once at channel construction; the same id is returned
    /// by both the `Sender` and `Receiver` ends.
    #[must_use]
    pub fn channel_id(&self) -> ChannelNodeId {
        self.inner.lock().unwrap().channel_id
    }

    /// Teardown path: drain all buffered values, emit one `ReclamationRecord(ChannelClose)`
    /// per value through `sink`, then mark the channel closed.
    ///
    /// This is the **third live reclamation trigger** (RFC-0027 §9 / §7.3 / G2): when a
    /// receiver disconnects while values are still in transit (sent but never to be received),
    /// those values' ownership is released and MUST be reclaimed — one record per value, never
    /// silently dropped (G2 / never-silent principle).
    ///
    /// ## Distinction from normal drain
    ///
    /// - **Normal drain (`try_recv`):** values are delivered to the receiver (no loss, no
    ///   reclamation needed). The receiver is the intended consumer; ownership transfers normally.
    /// - **`close_with_reclaim` (this method):** the receiver is tearing down and will NEVER
    ///   deliver the buffered values. The values are undeliverable in-transit; this method
    ///   releases their ownership and emits a `ChannelClose` reclamation record for each.
    ///
    /// The `hash_of` closure supplies the `value_meta_hash` for each value (RFC-0027 §9 field)
    /// because `V` is an arbitrary type with no inherent [`ContentHash`] (KISS — never
    /// over-constrain `V`). The closure is called on each value before reclamation, in FIFO order.
    ///
    /// ## Returns
    ///
    /// The number of values reclaimed (= the number of `ReclamationRecord`s emitted).
    ///
    /// ## Guarantee
    ///
    /// `Exact` — one `ReclamationRecord(ChannelClose)` is emitted per buffered value,
    /// enforced-by-construction: the buffer is drained with `drain(..)` (exhaustive iteration),
    /// and `sink.emit` is called exactly once per element. The channel is marked closed after
    /// the drain. No value in the buffer at call time can escape without emitting a record.
    pub fn close_with_reclaim(
        self,
        sink: &mut dyn ReclamationSink,
        scope_id: ScopeId,
        sweep_epoch: SweepEpoch,
        hash_of: impl Fn(&V) -> ContentHash,
    ) -> usize {
        let mut inner = self.inner.lock().unwrap();

        let channel_id = inner.channel_id.as_channel_id();

        // Drain all buffered values in FIFO order. For each, emit one ChannelClose record.
        // `drain(..)` is exhaustive — no value is silently dropped (G2 / never-silent).
        let values: Vec<V> = inner.buf.drain(..).collect();
        let count = values.len();

        for value in values {
            let record = ReclamationRecord::for_channel_close(
                scope_id,
                sweep_epoch,
                hash_of(&value),
                channel_id,
            );
            sink.emit(record);
        }

        // Mark the channel closed so any concurrent try_recv (via a live Sender's Arc)
        // will see Closed after the drain.
        inner.closed = true;

        count
    }
}

// ── TrySend / TryRecv enums ───────────────────────────────────────────────────

/// Result of a non-blocking send attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum TrySend<V> {
    /// Message accepted into the channel buffer.
    Sent,
    /// Channel buffer full; value returned to caller (G2: never silently dropped).
    Full(V),
    /// Channel closed; value returned to caller (G2: never silently dropped).
    Closed(V),
}

/// Result of a non-blocking receive attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum TryRecv<V> {
    /// Message received.
    Received(V),
    /// Channel buffer empty; no message available (sender still live).
    Empty,
    /// Channel closed (sender dropped or `close()` called) and buffer drained.
    Closed,
}

// Tests for this module live in `src/tests/network.rs` per the M-797 in-crate test layout
// (`#[cfg(test)] mod tests;` in `lib.rs` → `src/tests/mod.rs` → `src/tests/network.rs`).
