//! Layer-2 explicit reference-counting core — DN-32 §2.2 / RFC-0027 §10.1 / MEM-2.
//!
//! Implements the **non-atomic intra-hypha RC cell** and the `rc`-probe decision tree:
//!
//! ```text
//! rc(v) == 1  ⟹  UniqueOwner: sole-owner probe — allocation may be reused (FBIP)
//!                               + emit ReclamationRecord(RcZero) — MEM-1 never-silent wiring
//! rc(v) > 1   ⟹  Shared:      decrement only; other owners retain the value
//! ```
//!
//! The probe is evaluated **before** the logical decrement. `rc == 1` before `drop_ref` means
//! this IS the last live handle — we emit a `ReclamationRecord(RcZero)` and return
//! `RcProbe::UniqueOwner(T)`. `rc > 1` means other handles remain; we decrement and return
//! `RcProbe::Shared`.
//!
//! This is the **first live trigger wiring** into the MEM-1 `ReclamationSink` contract: the
//! `RcZero` emission on last-reference drop is G2-enforced (RFC-0027 §9).
//!
//! # Implementation note — `std::rc::Rc<T>` backing
//!
//! `RcCell<T>` wraps `std::rc::Rc<T>` to leverage Rust's existing non-atomic reference counting,
//! rather than managing raw pointers. This is correct and zero-cost (no double-allocation):
//! - `Rc<T>` is non-atomic (single-threaded), so `RcCell<T>` is `!Send + !Sync` by construction.
//! - `Rc::strong_count` gives the current refcount for the probe.
//! - `Rc::try_unwrap` extracts the owned `T` when count reaches 1 — the unique-owner fast path.
//!
//! This approach respects `#![forbid(unsafe_code)]` (the crate-wide policy — ADR-014 / `wild`-free)
//! while delivering the correct DN-32 Layer-2 semantics.
//!
//! # Design decisions
//!
//! ## Explicit probe discipline (no `Drop` impl)
//!
//! `RcCell<T>` does **not** implement `std::ops::Drop` in a way that emits a `ReclamationRecord`.
//! Dropping an `RcCell<T>` without calling `drop_ref` is a resource leak of the audit record —
//! the caller must explicitly call `drop_ref` to honour the G2 never-silent contract. This is by
//! design: a `ReclamationSink` cannot be passed through `Drop` without violating KC-3.
//!
//! ## `rc == 1` reuse probe (FBIP / Perceus)
//!
//! When `drop_ref` returns `RcProbe::UniqueOwner(T)`, the caller holds the owned `T` and may
//! reuse the value's storage for the next allocation of compatible shape (FBIP). Surface visibility
//! of the reuse-vs-copy choice is EXPLAIN-record-only by default (RFC-0027 §10.2 / OQ-4 resolved
//! by DN-32). The probe is `Declared` as a perf optimization — no measurement yet (DN-32 §6a).
//!
//! # Guarantee tags (per-op)
//!
//! | Operation | Tag | Basis |
//! |---|---|---|
//! | `clone_ref` increments count correctly | `Exact` | `Rc::clone` increments by 1 |
//! | `drop_ref` emits exactly one record on last-ref | `Exact` | Single `sink.emit` call; enforced-by-construction |
//! | `drop_ref` → `UniqueOwner` when rc==1 | `Exact` | `Rc::strong_count == 1` before decrement |
//! | `drop_ref` → `Shared` when rc>1 | `Exact` | `Rc::strong_count > 1` before decrement |
//! | `rc==1` reuse as perf win | `Declared` | Expected from Perceus/FBIP (lane-F F-4); no Mycelium measurement (DN-32 §6a) |
//!
//! # FLAGs — downstream work
//!
//! - **FLAG (MEM-3 / atomic cross-hypha RC):** `RcCell<T>` is `!Send + !Sync` (from `Rc<T>`).
//!   Cross-hypha transfer requires an atomic path (e.g. `Arc<T>`) — the DN-32 §7 reconciliation
//!   sub-question. Do not make `RcCell<T>: Send`; introduce a new type in MEM-3.
//!
//! - **FLAG (MEM-3 / region integration):** Deferred-drop accumulation per scope and bulk flush
//!   at scope-exit is MEM-3's responsibility. This module provides the per-cell probe primitive.
//!
//! - **FLAG (MEM-4 / static RC elision):** The Perceus compile-time uniqueness analysis
//!   eliminating RC ops is MEM-4 (deferred). The `UniqueOwner` probe here is the runtime fallback.
//!
//! Tests: `src/tests/rc.rs` (M-797 in-crate layout).

use std::rc::Rc;

use mycelium_core::ContentHash;

use crate::reclamation::{
    ReclamationRecord, ReclamationSink, ReclamationTrigger, ScopeId, SweepEpoch,
};

// ── RcCell<T> — the handle ────────────────────────────────────────────────────

/// A non-atomic intra-hypha reference-counted handle to a shared immutable value.
///
/// Backed by [`std::rc::Rc<T>`] for correct non-atomic reference counting within a single hypha.
///
/// ## Sharing model
///
/// [`clone_ref`](Self::clone_ref) produces a second handle, incrementing the refcount.
/// [`drop_ref`](Self::drop_ref) decrements the count, probes the outcome, and returns
/// an [`RcProbe`]. The caller **must** call `drop_ref` — `RcCell<T>` has no `Drop` impl
/// that emits a `ReclamationRecord`; dropping without `drop_ref` silently leaks the audit
/// record (G2 violation).
///
/// ## `!Send + !Sync`
///
/// Inherited from `Rc<T>`: the non-atomic refcount is safe only within one hypha.
///
/// ## Immutability
///
/// `T` is held immutably (LR-8 / DN-32 §2.2). There is no interior-mutation surface.
///
/// Guarantee: `Exact` — sharing and probe logic are deterministic by construction.
#[derive(Debug)]
pub struct RcCell<T> {
    inner: Rc<T>,
}

impl<T> RcCell<T> {
    /// Allocate a new `RcCell<T>` containing `value` with an initial strong refcount of 1.
    ///
    /// Guarantee: `Exact` — `Rc::new` allocates exactly once.
    #[must_use]
    pub fn new(value: T) -> Self {
        RcCell {
            inner: Rc::new(value),
        }
    }

    /// Return the current strong refcount (snapshot, non-atomic).
    ///
    /// Use [`drop_ref`](Self::drop_ref) for the reclamation probe — the probe is atomic with
    /// the decrement. This accessor is for introspection and testing.
    ///
    /// Guarantee: `Exact` — reads `Rc::strong_count` at call time (intra-hypha only).
    #[must_use]
    pub fn refcount(&self) -> usize {
        Rc::strong_count(&self.inner)
    }

    /// Get a reference to the shared value.
    ///
    /// Guarantee: `Exact` — shared reference valid for the lifetime of `&self`.
    #[must_use]
    pub fn value(&self) -> &T {
        &self.inner
    }

    /// Clone a handle — increment the refcount and return a new `RcCell<T>` pointing to the
    /// same allocation.
    ///
    /// Guarantee: `Exact` — `Rc::clone` increments the strong count by exactly 1.
    #[must_use]
    pub fn clone_ref(&self) -> RcCell<T> {
        RcCell {
            inner: Rc::clone(&self.inner),
        }
    }

    /// Decrement the refcount and return the [`RcProbe`] outcome, consuming this handle.
    ///
    /// ## Probe logic (evaluated on strong count BEFORE decrement)
    ///
    /// - **`strong_count == 1`** → `RcProbe::UniqueOwner(T)`: this IS the last handle.
    ///   Exactly one [`ReclamationRecord`] with `trigger = RcZero` is emitted via `sink`
    ///   (never-silent G2 / RFC-0027 §9). The owned `T` is extracted and returned.
    ///   `Rc::try_unwrap` is used to extract `T` from the last `Rc` without copying.
    ///
    /// - **`strong_count > 1`** → `RcProbe::Shared`: other handles remain. The `Rc` is dropped
    ///   (decrementing the count). No record is emitted.
    ///
    /// ## Reclamation context
    ///
    /// `scope_id`, `sweep_epoch`, and `value_meta_hash` anchor the `ReclamationRecord` in the
    /// RT7 scope tree and the RFC-0008 §4.3 sweep model. These are caller-supplied (KC-3 — no
    /// global state here); the caller is responsible for correct values.
    ///
    /// ## Never-silent contract (`Exact` within this module)
    ///
    /// When `strong_count == 1`, `sink.emit` is called exactly once. No code path through
    /// `UniqueOwner` can avoid this call — enforced-by-construction.
    ///
    /// Guarantee: `Exact` — one record emitted on last-ref; none otherwise.
    pub fn drop_ref(
        self,
        sink: &mut dyn ReclamationSink,
        scope_id: ScopeId,
        sweep_epoch: SweepEpoch,
        value_meta_hash: ContentHash,
    ) -> RcProbe<T> {
        let count = Rc::strong_count(&self.inner);

        if count == 1 {
            // Last handle — emit the reclamation record, then extract the value.
            let record = ReclamationRecord::new(
                scope_id,
                sweep_epoch,
                ReclamationTrigger::RcZero,
                value_meta_hash,
            );
            sink.emit(record);

            // `Rc::try_unwrap` extracts `T` from the last `Rc` without copying.
            // It returns `Ok(T)` iff strong_count == 1, which we just verified.
            // The `Err` branch is unreachable: we read count == 1 and this is the only
            // handle (single-threaded; no concurrent clone possible within one hypha).
            match Rc::try_unwrap(self.inner) {
                Ok(value) => RcProbe::UniqueOwner(value),
                Err(_rc) => {
                    // Unreachable: strong_count was 1, we are the last handle, and this is
                    // intra-hypha (no concurrent clone). Treat as a programming error (G2).
                    panic!(
                        "RcCell::drop_ref: Rc::try_unwrap failed despite strong_count==1 \
                         (intra-hypha invariant violated — G2)"
                    );
                }
            }
        } else {
            // count > 1: shared. Drop the Rc (decrements count). No record emitted.
            drop(self.inner);
            RcProbe::Shared
        }
    }
}

// ── RcProbe — the three-way probe outcome ─────────────────────────────────────

/// The outcome of an [`RcCell::drop_ref`] call — the rc-probe decision (RFC-0027 §10.1).
///
/// ## Variant semantics
///
/// | Variant | Condition (strong_count before decrement) | Record emitted? |
/// |---|---|---|
/// | `UniqueOwner(T)` | `strong_count == 1` | Yes — `ReclamationRecord(trigger = RcZero)` |
/// | `Shared` | `strong_count > 1` | No |
///
/// ## `UniqueOwner` is the `rc → 0` case
///
/// RFC-0027 §10.1 lists `rc→0` / `rc==1` / `rc>1`. In a drop context, `rc→0` happens exactly
/// when `rc` was 1 before the drop — so `UniqueOwner` IS the `rc→0` / `RcZero` event.
/// The `ReclamationRecord(trigger = RcZero)` is emitted inside `drop_ref`; `UniqueOwner(T)`
/// signals to the caller that reclamation occurred and the owned value is available for reuse.
///
/// ## FBIP reuse
///
/// When `UniqueOwner(T)` is returned, the caller may reuse the `T` value's allocation for the
/// next value of compatible shape (FBIP / Perceus `rc==1` reuse — RFC-0027 §10.2). This is a
/// `Declared` optimization: no Mycelium perf measurement yet (DN-32 §6a).
///
/// Guarantee: `Exact` — variant is a deterministic function of the refcount before decrement.
#[derive(Debug)]
pub enum RcProbe<T> {
    /// This handle was the **sole owner** (`rc == 1` before decrement / `rc → 0`).
    ///
    /// A `ReclamationRecord(trigger = RcZero)` has been emitted to the sink exactly once.
    /// The owned value `T` is returned; the caller may inspect it, recurse-reclaim its
    /// sub-values, or reuse its allocation (FBIP).
    ///
    /// Guarantee: `Exact` — exactly one `ReclamationRecord(RcZero)` was emitted.
    UniqueOwner(T),

    /// The refcount was > 1 before decrement; other handles remain.
    ///
    /// The `Rc` was dropped (count decremented). No record was emitted.
    ///
    /// Guarantee: `Exact` — strong_count >= 1 after decrement.
    Shared,
}
