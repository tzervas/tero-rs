//! Live-executor scope/region wiring — DN-32 §2.3 / RFC-0027 §10.3 / §9 / MEM-3.
//!
//! **Purpose:** Ties a [`Region`]'s lifecycle to a single-hypha structured-concurrency scope so
//! that scope-exit reclamation fires from a *running* scope, not just from a bare data structure.
//! The bare [`Region`] in `region.rs` provides the data model and the `close()` call; this module
//! provides the **structured closure entry points** that guarantee `close()` is always called —
//! never silently skipped (G2 / VR-5).
//!
//! # Design placement
//!
//! `Region` is single-threaded (`!Sync`) — all wiring here is **intra-hypha**. Cross-hypha
//! atomic-RC sharing is a named sub-question (DN-32 §7 / RFC-0027 §12); see the FLAG below.
//! Nothing in this module is `Send` or `Sync`.
//!
//! # API surface
//!
//! Two entry points, one for each calling style:
//!
//! 1. **[`with_region`]** — closure form. Opens a `Region`, runs a body closure, then
//!    unconditionally closes the region after the body returns. The entry point for structured
//!    scope-exit reclamation in one hypha's body. This is also the per-hypha scope-exit hook
//!    described in RFC-0027 §10.3: "run the hypha's body within its scope-region; reclaim at
//!    hypha-exit."
//!
//! 2. **[`RegionScope`]** — explicit-close scope guard. For callers who need to interleave
//!    deferrals with other work between `enter` and `close`. The caller is responsible for
//!    calling [`RegionScope::close`]; a dropped `RegionScope` with pending entries triggers
//!    the underlying [`Region`]'s debug-drop-panic guard (G2 — silent audit loss is
//!    impossible in debug builds).
//!
//! # Nested scopes (free, by construction)
//!
//! [`with_region`] calls nest: an inner `with_region` closes (allocating a lower [`RegionEpoch`])
//! before the outer one, so `inner_closed.epoch < outer_closed.epoch` holds by construction.
//! No extra code is needed; the monotonic epoch counter encodes child→root ordering as a number
//! line (RFC-0027 §10.3 / region.rs §"Sweep order"). Verified in `src/tests/scope_region.rs`.
//!
//! # Point 4 choice — `run_in_region` alias
//!
//! **FLAG (KC-3 / DRY):** `run_in_region` would be a pure alias over `with_region` with no
//! added logic. Rather than add a useless wrapper (YAGNI / KC-3), the per-hypha scope-exit
//! entry-point role is **documented on `with_region` directly** via the doc-comment above.
//! No `run_in_region` function is exported.
//!
//! # Guarantee tags (per-op)
//!
//! | Operation | Tag | Basis |
//! |---|---|---|
//! | `close` always called after `body` returns (normal path) | `Exact` | `with_region` calls `region.close` unconditionally after `body(…)`; enforced-by-construction |
//! | `close` called on `RegionScope::close` | `Exact` | consuming `close` delegates to `region.close`; enforced-by-construction |
//! | Panic path: `close` NOT called if `body` panics | `Exact` | stack-unwind drops `Region`; debug-build guard surfaces lost records (G2 never-silent signal is preserved) |
//! | `inner_closed.epoch < outer_closed.epoch` for nested `with_region` | `Exact` | inner closes first → lower monotonic epoch; inherited from `region.rs` |
//! | Batching as perf-win (amortised vs per-value) | `Declared` | DN-32 §6a; no Mycelium measurement |
//!
//! # FLAGs
//!
//! - **FLAG (cross-hypha atomic RC / `Send`):** `Region` is `!Sync` (intra-hypha). Cross-hypha
//!   sharing and the atomic-RC path are DN-32 §7 / RFC-0027 §12 — a named sub-question for a
//!   follow-on RFC/implementation. Every hypha owns its own single-threaded region; no `Send`
//!   bound is introduced here.
//! - **FLAG (`lib.rs` + `tests/mod.rs` registration):** This module must be declared as
//!   `pub mod scope_region;` in `lib.rs` and `pub mod scope_region;` in `src/tests/mod.rs`.
//!   Both files are orchestrator-owned; this leaf does NOT edit them.
//!
//! Tests: `src/tests/scope_region.rs` (M-797 in-crate layout).

use mycelium_core::ContentHash;

use crate::reclamation::ReclamationSink;
use crate::region::{ClosedRegion, Region, ScopeNodeId};

// ── with_region — closure-form structured scope-exit reclamation ──────────────

/// Run `body` within a freshly-opened scope region and close the region after the body returns.
///
/// # Per-hypha scope-exit entry point (RFC-0027 §10.3)
///
/// This is the structured-concurrency entry point for scope-exit reclamation in one hypha's body:
/// "run the hypha body; reclaim at hypha-exit." Each hypha owns exactly one single-threaded
/// intra-hypha region (see the cross-hypha FLAG in this module's doc-comment).
///
/// # What happens
///
/// 1. A fresh [`Region`] is opened (`Region::new()`).
/// 2. `body(&mut region)` is called. The body may call `region.defer(hash)` for every value
///    whose RC hit zero within this scope (or any scope-local allocation to be freed at exit).
/// 3. **After `body` returns** (normal path), `region.close(sink)` is called unconditionally,
///    emitting one `ScopeExit` [`crate::reclamation::ReclamationRecord`] per deferred entry
///    through `sink` and returning the [`ClosedRegion`] summary.
/// 4. The function returns `(body_result, closed_region)`.
///
/// # Panic behaviour
///
/// If `body` panics, the region is dropped as the stack unwinds. A panic is an abnormal scope
/// abort; in debug builds the underlying [`Region`]'s drop-guard panics again if there are
/// deferred entries, surfacing the number of lost audit records (G2 never-silent signal). This
/// is intentional — do not catch the panic to silence the guard. The debug double-panic is
/// the price of a G2-clean design: silent audit loss is impossible in debug builds.
///
/// # Nested scopes
///
/// Call `with_region` inside a `body` to nest an inner scope under an outer one. The inner scope
/// closes (allocating its [`crate::region::RegionEpoch`]) before the outer body returns, so
/// `inner_closed.epoch < outer_closed.epoch` by construction (monotonic counter — `Exact`).
///
/// # Guarantee tags
///
/// - **`close` always called (normal path):** `Exact` — unconditional call after `body`; no
///   conditional branch can skip it (enforced-by-construction).
/// - **Panic path (abnormal abort):** `Exact` — drop fires; debug guard surfaces lost records.
/// - **Nested epoch ordering:** `Exact` — inherited from `Region::close`'s monotonic counter.
/// - **Batching perf:** `Declared` — DN-32 §6a; no Mycelium measurement.
///
/// # Example
///
/// ```rust,ignore
/// use mycelium_std_runtime::scope_region::with_region;
/// use mycelium_std_runtime::reclamation::CollectingSink;
///
/// let mut sink = CollectingSink::new();
/// let (result, closed) = with_region(&mut sink, |region| {
///     region.defer(some_hash);
///     42
/// });
/// assert_eq!(result, 42);
/// assert_eq!(closed.reclaimed_count, 1);
/// ```
pub fn with_region<R>(
    sink: &mut dyn ReclamationSink,
    body: impl FnOnce(&mut Region) -> R,
) -> (R, ClosedRegion) {
    let mut region = Region::new();
    // Run the body — may call region.defer(hash) zero or more times.
    let result = body(&mut region);
    // Unconditionally close after body returns (normal path). This is the guarantee:
    // `close` is always called after `body` on the normal return path (Exact).
    let closed = region.close(sink);
    (result, closed)
}

// ── RegionScope — explicit-close scope guard ──────────────────────────────────

/// An explicit-close scope guard wrapping a [`Region`].
///
/// For callers who need to interleave deferrals with other logic between scope entry and exit
/// and cannot use the closure form ([`with_region`]). Holds an open [`Region`] and exposes
/// `defer`, `deferred_count`, `id`, and a **consuming** `close` that guarantees exactly one
/// `Region::close` call.
///
/// # Never-silent contract (G2)
///
/// The caller **MUST** call [`RegionScope::close`] before the guard goes out of scope. There is
/// no `Drop` impl that accepts a [`ReclamationSink`] (a sink cannot be threaded through `Drop`
/// without violating KC-3 / the never-silent contract); dropping a `RegionScope` that still holds
/// deferred entries causes the underlying [`Region`]'s debug-drop-panic to fire, surfacing the
/// number of lost audit records. This is the same design discipline as `rc.rs`'s `RcCell` —
/// explicit-close is the correct pattern; sink-bearing `Drop` is not.
///
/// # Example
///
/// ```rust,ignore
/// use mycelium_std_runtime::scope_region::RegionScope;
/// use mycelium_std_runtime::reclamation::CollectingSink;
///
/// let mut sink = CollectingSink::new();
/// let mut scope = RegionScope::enter();
/// scope.defer(hash_a);
/// // … interleaved work …
/// scope.defer(hash_b);
/// let closed = scope.close(&mut sink);
/// assert_eq!(closed.reclaimed_count, 2);
/// ```
///
/// # Guarantee tags
///
/// - **`close` called exactly once (on `RegionScope::close`):** `Exact` — consuming `close`
///   destructs `self`, so double-close is a compile-time type error.
/// - **Dropped without `close` (debug builds):** `Exact` — underlying `Region` drop-guard panics
///   if deferred entries remain (G2 never-silent signal).
/// - **`defer` / `deferred_count` / `id` accuracy:** `Exact` — delegates to `Region`; all
///   fields are exact functions of the region state.
#[derive(Debug)]
pub struct RegionScope {
    /// The open region backing this scope guard.
    region: Region,
}

impl RegionScope {
    /// Open a new scope guard, allocating a fresh [`Region`].
    ///
    /// Guarantee: `Exact` — allocates a unique [`ScopeNodeId`] via the monotonic counter.
    #[must_use]
    pub fn enter() -> Self {
        RegionScope {
            region: Region::new(),
        }
    }

    /// Defer a value for scope-exit reclamation.
    ///
    /// Delegates to [`Region::defer`]. The value's content identity is pushed to the deferred
    /// list; at [`RegionScope::close`] a `ScopeExit` record is emitted for it.
    ///
    /// Calling `defer` after `close` is impossible by type (consuming close destructs `self`).
    ///
    /// Guarantee: `Exact` — delegates to `Region::defer`; the entry is pushed exactly once.
    pub fn defer(&mut self, value_meta_hash: ContentHash) {
        self.region.defer(value_meta_hash);
    }

    /// Number of values currently deferred in this scope.
    ///
    /// Guarantee: `Exact` — delegates to `Region::deferred_count`.
    #[must_use]
    pub fn deferred_count(&self) -> usize {
        self.region.deferred_count()
    }

    /// The unique identity of this scope's underlying [`Region`].
    ///
    /// Guarantee: `Exact` — delegates to `region.id`; unique monotonic allocation.
    #[must_use]
    pub fn id(&self) -> ScopeNodeId {
        self.region.id
    }

    /// Close this scope guard, emitting all deferred reclamation records and returning the
    /// [`ClosedRegion`] summary.
    ///
    /// This is a **consuming** method — `self` is destructed, making double-close a compile-time
    /// type error. After `close` returns, the sink has received one `ScopeExit` record per
    /// deferred entry.
    ///
    /// Guarantee: `Exact` — delegates to [`Region::close`]; one `ScopeExit` record per deferred
    /// entry, emitted unconditionally (enforced-by-construction).
    pub fn close(mut self, sink: &mut dyn ReclamationSink) -> ClosedRegion {
        self.region.close(sink)
    }
}
