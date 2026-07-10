//! Layer-3 region-based batched scope reclamation — DN-32 §2.3 / RFC-0027 §10.3 / MEM-3.
//!
//! **Region = Scope.** A [`Region`] is the runtime embodiment of one node in the RT7 scope tree
//! (RFC-0008 RT7). It accumulates **deferred reclamation entries** — values whose RC hit zero
//! within this scope, plus scope-local allocations that should be freed at scope exit — and emits
//! one [`ReclamationRecord`] per entry (trigger = `ScopeExit`) through a [`ReclamationSink`] when
//! the region is **closed** (`Region::close`). Reclamation is **batched** (bulk, amortized) rather
//! than per-value.
//!
//! # Canonical identity types (`ScopeNodeId` / `RegionEpoch`)
//!
//! MEM-1 (`reclamation.rs`) left [`ScopeId`] and [`SweepEpoch`] as `u64` placeholders with a FLAG
//! to canonicalize in MEM-3. This module provides the canonical forms:
//!
//! - [`ScopeNodeId`] — the region-tier scope identity (a monotonic `u64` allocated per
//!   [`Region::new`] call, unique within a process lifetime). [`ScopeId`] is bridged via
//!   [`ScopeNodeId::as_scope_id`] so MEM-1/MEM-2 records carry the same field type.
//! - [`RegionEpoch`] — a monotonic counter that advances once per [`Region::close`] call.
//!   [`SweepEpoch`] is bridged via [`RegionEpoch::as_sweep_epoch`].
//!
//! Both are newtypes over `u64` — the same backing integer — but now grounded in the region
//! model rather than being bare placeholders. The MEM-1 types (`ScopeId`, `SweepEpoch`) are
//! retained as the record-field surface (they are part of the stable RFC-0027 §9 field set);
//! `ScopeNodeId` / `RegionEpoch` are the region-tier ALLOCATION types that produce them.
//!
//! # Sweep order (RFC-0027 §10.3 / DN-32 §3)
//!
//! The scope tree encodes the reclamation order directly:
//!
//! - **Parent–child: TOTAL.** A child region MUST be closed before its parent closes. The
//!   parent's [`Region::close`] does NOT close children; the caller is responsible for closing
//!   all child regions first (structured-concurrency RT7 contract: a scope does not exit until
//!   every child completes). This is encoded in [`ScopeTree::close_ordered`] and proven in the
//!   property test (`child epoch < parent epoch`). Live executor wiring is downstream (FLAG).
//! - **Siblings: CONCURRENT (OQ-1 resolved).** Sibling regions are independent — there is no
//!   ordering constraint between them. Each sibling closes (and batches its reclamation) without
//!   synchronizing with any other sibling. This is safe because LR-9 rules out cross-sibling
//!   aliases (RFC-0027 §11 OQ-1 / DN-32 §3).
//!
//! # Guarantee tags (per-op)
//!
//! | Operation | Tag | Basis |
//! |---|---|---|
//! | Batched `ScopeExit` emit (one record per deferred entry) | `Exact` | Every entry in `deferred` is drained; `sink.emit` called once per entry; loop is exhaustive; enforced-by-construction |
//! | `ScopeNodeId` uniqueness within process | `Exact` | Atomic monotonic counter, `fetch_add(1)`, strictly increasing |
//! | `RegionEpoch` monotonicity | `Exact` | `GLOBAL_EPOCH.fetch_add(1, Relaxed)` — strictly increasing per close call |
//! | Child epoch < parent epoch (total parent–child order) | `Exact` | Children close (allocate epoch) before parent → lower epoch by construction |
//! | Sibling reclamation is CONCURRENT (safe) | `Proven`-modulo-LR-9 | LR-9 rules out cross-sibling aliases; argument per DN-32 §3 / RFC-0027 §11 OQ-1; no in-repo mechanized proof |
//! | Batching is a perf win over per-value drops | `Declared` | Expected from Tofte-Talpin + DN-32 §2.3 / §6a; no Mycelium measurement |
//!
//! # FLAGs — downstream work
//!
//! - **FLAG (live-executor wiring):** The actual firing of `Region::close` at scope-exit in the
//!   running scheduler/MLIR runtime is downstream (live-executor wiring). This module provides the
//!   data structure and the `close()` call; the executor must call it at the right time.
//! - **FLAG (cross-hypha atomic RC):** `Region` is single-threaded (intra-hypha). Cross-hypha
//!   sharing and the atomic RC path are DN-32 §7 / RFC-0027 §12 — a named sub-question for the
//!   follow-on, not handled here.
//! - **FLAG (strong/total sibling coupling — opt-in):** DN-32 §3 mentions a "strong coupling
//!   opt-in for high-assurance subsets". That option is not implemented here; deferred to the
//!   follow-on RFC/implementation.
//! - **FLAG (`ChannelId` canonicalization):** `ChannelId` in `reclamation.rs` remains a `u64`
//!   placeholder — it belongs to the network/channel tier, which is out of scope for MEM-3.
//!
//! Tests: `src/tests/region.rs` (M-797 in-crate layout).

use std::sync::atomic::{AtomicU64, Ordering};

use mycelium_core::ContentHash;

use crate::reclamation::{
    ReclamationRecord, ReclamationSink, ReclamationTrigger, ScopeId, SweepEpoch,
};

// ── Global monotonic counters ─────────────────────────────────────────────────
//
// Two independent atomic counters:
// - GLOBAL_SCOPE_ID: allocates a unique `ScopeNodeId` per `Region::new` call.
// - GLOBAL_EPOCH:    allocates a monotonic `RegionEpoch` per `Region::close` call.
//
// Both use `Relaxed` ordering: the values are unique across calls and monotonically
// increasing within a process lifetime. No cross-thread happens-before is needed for
// the IDs themselves — the region is owned by one hypha at a time (intra-hypha only);
// the atomics are used to guarantee uniqueness across hypha boundaries for id allocation,
// not for memory synchronization.
//
// Guarantee: `Exact` — `fetch_add(1, Relaxed)` is strictly monotonic and unique within
// a process lifetime on any platform where u64 arithmetic does not overflow in practice.
// (Counter wrap at u64::MAX is `Declared` as a known corner-case; 585 years at 10^9/s.)

static GLOBAL_SCOPE_ID: AtomicU64 = AtomicU64::new(1); // start at 1; 0 reserved as "no scope"
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(1); // start at 1; 0 reserved as "pre-close"

// ── ScopeNodeId — canonical scope-tree identity (resolves MEM-1 FLAG) ─────────

/// The canonical scope-tree identity type for the region tier (MEM-3).
///
/// Allocated per [`Region::new`] via a monotonic global counter. Unique within a process
/// lifetime. This is the resolved form of the `u64` placeholder that MEM-1 left in
/// [`ScopeId`] with a FLAG to canonicalize here. Use [`ScopeNodeId::as_scope_id`] to convert
/// to [`ScopeId`] for use in [`ReclamationRecord`] fields (the stable RFC-0027 §9 type).
///
/// Guarantee: `Exact` — monotonic, unique per allocation (atomic counter, strictly increasing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScopeNodeId(pub(crate) u64);

impl ScopeNodeId {
    /// Allocate a fresh, unique `ScopeNodeId`.
    ///
    /// Guarantee: `Exact` — strictly greater than all previously allocated ids.
    fn allocate() -> Self {
        ScopeNodeId(GLOBAL_SCOPE_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Convert to a [`ScopeId`] for use in [`ReclamationRecord`] fields (RFC-0027 §9).
    ///
    /// This is the canonical bridge between the region-tier allocation identity and the
    /// MEM-1 record field type. The underlying `u64` is preserved losslessly.
    ///
    /// Guarantee: `Exact` — lossless conversion.
    #[must_use]
    pub fn as_scope_id(self) -> ScopeId {
        ScopeId(self.0)
    }

    /// Return the underlying `u64` value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ── RegionEpoch — canonical sweep epoch (resolves MEM-1 FLAG) ────────────────

/// A monotonic epoch counter — one allocated per [`Region::close`] call.
///
/// Ties a reclamation batch to the scheduling model's deterministic audit anchor
/// (RFC-0008 §4.3). Children close before their parent and therefore receive numerically
/// lower epoch values, encoding the child→root total order as a number line.
///
/// This is the resolved form of the `u64` placeholder that MEM-1 left in [`SweepEpoch`]
/// with a FLAG to canonicalize here. Use [`RegionEpoch::as_sweep_epoch`] to convert to
/// [`SweepEpoch`] for use in [`ReclamationRecord`] fields (the stable RFC-0027 §9 type).
///
/// Guarantee: `Exact` — monotonic, unique per `close` call (atomic counter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionEpoch(pub(crate) u64);

impl RegionEpoch {
    /// Allocate a fresh, monotonically-increasing `RegionEpoch`.
    ///
    /// Guarantee: `Exact` — strictly greater than all previously allocated epochs.
    fn allocate() -> Self {
        RegionEpoch(GLOBAL_EPOCH.fetch_add(1, Ordering::Relaxed))
    }

    /// Convert to a [`SweepEpoch`] for use in [`ReclamationRecord`] fields (RFC-0027 §9).
    ///
    /// Guarantee: `Exact` — lossless conversion.
    #[must_use]
    pub fn as_sweep_epoch(self) -> SweepEpoch {
        SweepEpoch(self.0)
    }

    /// Return the underlying `u64` value.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ── DeferredEntry — one value deferred for scope-exit reclamation ─────────────

/// One value deferred for scope-exit reclamation within a [`Region`].
///
/// Holds the content identity of the value to be reclaimed. The `ScopeExit` trigger and
/// the `scope_id`/`sweep_epoch` are supplied at close time (from the region itself), not
/// at deferral time — deferral is cheap (a single `ContentHash` push, no bookkeeping).
///
/// Guarantee: `Exact` — the hash is stored as-is; no approximation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredEntry {
    /// Content identity of the value deferred for reclamation (RFC-0027 §9 / RFC-0001 §4.6).
    pub value_meta_hash: ContentHash,
}

impl DeferredEntry {
    /// Create a new deferred entry for `value_meta_hash`.
    #[must_use]
    pub fn new(value_meta_hash: ContentHash) -> Self {
        DeferredEntry { value_meta_hash }
    }
}

// ── Region — the batched scope-exit reclamation structure ─────────────────────

/// A scope-exit reclamation region (DN-32 §2.3 / RFC-0027 §10.3 / MEM-3).
///
/// One `Region` corresponds to one RT7 scope-tree node. It accumulates
/// [`DeferredEntry`] values — values whose RC hit zero within this scope, or scope-local
/// allocations — and at [`Region::close`] emits one [`ReclamationRecord`] per entry
/// (trigger = `ScopeExit`) through a [`ReclamationSink`].
///
/// ## Sweep order
///
/// - **Parent–child TOTAL:** Close all child `Region`s before closing the parent.
///   The child's epoch will be numerically lower than the parent's (it closed first),
///   encoding the child→root ordering as a number line. The caller (the executor) is
///   responsible for this invariant; [`ScopeTree::close_ordered`] enforces it in tests.
///
/// - **Siblings CONCURRENT (OQ-1 resolved, `Proven`-modulo-LR-9):** Sibling regions
///   close independently — no ordering constraint. Safe because LR-9 rules out
///   cross-sibling aliases (DN-32 §3).
///
/// ## Never-silent contract (G2)
///
/// Every [`DeferredEntry`] pushed via [`Region::defer`] emits exactly one
/// `ReclamationRecord(ScopeExit)` when [`Region::close`] is called. A `Region` dropped
/// without calling `close` while still holding deferred entries is a G2 violation; in
/// debug builds [`Region::drop`] panics to surface this.
///
/// Guarantee: `Exact` — one record per deferred entry at close (enforced-by-construction).
#[derive(Debug)]
pub struct Region {
    /// The unique identity of this region in the scope tree.
    pub id: ScopeNodeId,

    /// Deferred reclamation entries accumulated during this scope's lifetime.
    deferred: Vec<DeferredEntry>,

    /// The epoch assigned at close time. `None` until `close()` is called.
    closed_epoch: Option<RegionEpoch>,
}

impl Region {
    /// Create a new, open region for a scope-tree node.
    ///
    /// Allocates a fresh [`ScopeNodeId`]. The region starts open (no epoch yet).
    ///
    /// Guarantee: `Exact` — `ScopeNodeId` is unique and monotonically increasing.
    #[must_use]
    pub fn new() -> Self {
        Region {
            id: ScopeNodeId::allocate(),
            deferred: Vec::new(),
            closed_epoch: None,
        }
    }

    /// Defer a value for scope-exit reclamation.
    ///
    /// The value's content identity is pushed to the deferred list. At `close()`,
    /// a `ReclamationRecord(ScopeExit)` is emitted for it.
    ///
    /// Calling `defer` after `close()` is a logic error; panics in debug builds.
    ///
    /// Guarantee: `Exact` — the entry is pushed exactly once; no approximation.
    pub fn defer(&mut self, value_meta_hash: ContentHash) {
        debug_assert!(
            self.closed_epoch.is_none(),
            "Region::defer called on a closed region (id={:?})",
            self.id
        );
        self.deferred.push(DeferredEntry::new(value_meta_hash));
    }

    /// The number of values currently deferred in this region.
    ///
    /// Guarantee: `Exact` — returns the current length of the deferred list.
    #[must_use]
    pub fn deferred_count(&self) -> usize {
        self.deferred.len()
    }

    /// Whether this region has been closed.
    ///
    /// Guarantee: `Exact` — `closed_epoch.is_some()` is the canonical indicator.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed_epoch.is_some()
    }

    /// The epoch assigned at close time, if this region has been closed.
    ///
    /// Returns `None` if the region is still open.
    ///
    /// Guarantee: `Exact` — the epoch is the monotonic counter value at close time.
    #[must_use]
    pub fn closed_epoch(&self) -> Option<RegionEpoch> {
        self.closed_epoch
    }

    /// Close this region: **batch-emit all deferred reclamations**, advance the monotonic
    /// epoch, and return the [`ClosedRegion`] summary.
    ///
    /// ## What happens at close (`Exact` — enforced-by-construction)
    ///
    /// 1. A fresh [`RegionEpoch`] is allocated (monotonically increasing).
    /// 2. For each [`DeferredEntry`] in `self.deferred` (drained in order), exactly one
    ///    [`ReclamationRecord`] is constructed with:
    ///    - `scope_id = self.id.as_scope_id()`
    ///    - `sweep_epoch = epoch.as_sweep_epoch()`
    ///    - `trigger = ReclamationTrigger::ScopeExit`  ← **the SECOND live trigger (MEM-3)**
    ///    - `value_meta_hash = entry.value_meta_hash`
    ///
    ///    Each record is emitted through `sink.emit(record)` (never-silent G2).
    /// 3. The deferred list is drained (emptied).
    /// 4. `self.closed_epoch` is set to the allocated epoch.
    /// 5. A [`ClosedRegion`] summary is returned.
    ///
    /// ## Sweep-order contract (`Declared`)
    ///
    /// The caller MUST close all child regions before calling `close()` on the parent.
    /// A child closed before its parent will have a numerically lower `RegionEpoch`
    /// (it allocated its epoch first), encoding the child→root total order.
    /// `Region::close` does NOT enforce this by type — it is an RT7 contract (live-executor
    /// wiring is downstream, FLAG).
    ///
    /// ## Calling `close` more than once panics in debug builds.
    ///
    /// Bulk-efficiency-as-perf-win: `Declared` (DN-32 §6a — no Mycelium measurement yet).
    pub fn close(&mut self, sink: &mut dyn ReclamationSink) -> ClosedRegion {
        debug_assert!(
            self.closed_epoch.is_none(),
            "Region::close called on an already-closed region (id={:?})",
            self.id
        );

        // Allocate a monotonic epoch for this close event.
        let epoch = RegionEpoch::allocate();

        let scope_id = self.id.as_scope_id();
        let sweep_epoch = epoch.as_sweep_epoch();
        let count = self.deferred.len();

        // Batch-emit one ScopeExit record per deferred entry (never-silent G2).
        // `drain(..)` empties the deferred list, processing every entry exactly once.
        for entry in self.deferred.drain(..) {
            let record = ReclamationRecord::new(
                scope_id,
                sweep_epoch,
                ReclamationTrigger::ScopeExit,
                entry.value_meta_hash,
            );
            sink.emit(record);
        }

        self.closed_epoch = Some(epoch);

        ClosedRegion {
            id: self.id,
            epoch,
            reclaimed_count: count,
        }
    }
}

impl Default for Region {
    fn default() -> Self {
        Region::new()
    }
}

impl Drop for Region {
    /// G2 guard: panic in debug builds if a region is dropped while still holding
    /// deferred entries without having been closed — that would silently lose audit records.
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if self.closed_epoch.is_none() && !self.deferred.is_empty() {
            panic!(
                "Region (id={:?}) dropped with {} deferred entries without calling close() — \
                 G2 violation: audit records would be silently lost",
                self.id,
                self.deferred.len()
            );
        }
    }
}

// ── ClosedRegion — the summary returned by Region::close ─────────────────────

/// Summary returned by [`Region::close`] — the audit-visible outcome of a scope-exit
/// reclamation batch.
///
/// Carries the region's identity, the epoch assigned at close time, and the count of
/// deferred entries reclaimed. Useful for supervision observability, property tests, and
/// the sweep-order assertion (child epoch < parent epoch).
///
/// Guarantee: `Exact` — all fields are deterministic functions of the close operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosedRegion {
    /// The identity of the closed region.
    pub id: ScopeNodeId,
    /// The monotonic epoch allocated at close time.
    pub epoch: RegionEpoch,
    /// The number of deferred entries reclaimed (one `ScopeExit` record emitted per entry).
    pub reclaimed_count: usize,
}

// ── ScopeTree — a minimal scope-tree for sweep-order tests ───────────────────

/// A minimal parent–child scope-tree for encoding and testing the sweep-order model
/// (RFC-0027 §10.3 / DN-32 §3).
///
/// `ScopeTree` holds one parent [`Region`] and zero or more child [`Region`]s. The correct
/// close order (RT7) is: close all children before the parent. [`ScopeTree::close_ordered`]
/// enforces this order. Siblings may close in any order — they are concurrent (OQ-1).
///
/// This is NOT a full runtime scope tree; it is the minimal structure needed to encode and
/// property-test the parent–child TOTAL and sibling-CONCURRENT ordering guarantees. Live
/// executor integration is downstream (FLAG).
///
/// Guarantee: `Declared` — the tree structure encodes the sweep order; epoch-ordering
/// property (child < parent) is `Exact` by monotonic counter.
pub struct ScopeTree {
    /// Child regions — independent siblings (closed before the parent, any order).
    pub children: Vec<Region>,
    /// The parent region — closed last.
    pub parent: Region,
}

impl ScopeTree {
    /// Create a `ScopeTree` with `n_children` child regions under one parent.
    ///
    /// All regions start open. Siblings are independent (no ordering constraint).
    ///
    /// Guarantee: `Exact` — `n_children + 1` unique `ScopeNodeId`s allocated.
    #[must_use]
    pub fn new(n_children: usize) -> Self {
        ScopeTree {
            children: (0..n_children).map(|_| Region::new()).collect(),
            parent: Region::new(),
        }
    }

    /// Close all children first (in any order — siblings are concurrent), then close the parent.
    ///
    /// Returns all [`ClosedRegion`] summaries in close order: children first, parent last.
    ///
    /// The sweep-order invariant — child epoch < parent epoch — is guaranteed by the monotonic
    /// counter: children allocate their epochs before the parent does.
    ///
    /// Guarantee: `Exact` — close order is enforced; child epoch < parent epoch by construction.
    pub fn close_ordered(&mut self, sink: &mut dyn ReclamationSink) -> Vec<ClosedRegion> {
        let mut results = Vec::with_capacity(self.children.len() + 1);

        // Close all children first (any order among siblings — they are concurrent; OQ-1 resolved).
        for child in &mut self.children {
            results.push(child.close(sink));
        }

        // Close the parent last — total order (all child epochs < parent epoch).
        results.push(self.parent.close(sink));

        results
    }
}
