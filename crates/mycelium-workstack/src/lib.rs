//! **The shared recursion-budget + guarded-stack leaf for RFC-0041 (Wave-1).**
//!
//! This crate is the *canonical home* of the never-silent recursion budget every Mycelium execution
//! machine charges against — the L1 evaluator, the L0 reference interpreter, and the AOT env-machine
//! (RFC-0041 §4.1). It deliberately extracts **only** the shared *budget + guarded-stack helper*, not a
//! universal `WorkStack<Frame>`: each machine keeps its own bespoke frame/loop shape (a substitution
//! machine, a CEK env machine, a frame machine — §4.6), and only the *counters, limits, and the
//! never-silent over-budget surface* live here.
//!
//! # What is (and is not) here
//!
//! - **[`RecursionBudget`]** — the per-invocation budget: a depth ceiling on the §4.0 metric
//!   (default [`RecursionBudget::DEFAULT_DEPTH_LIMIT`] = `4096`), a memory ceiling, and a work-step
//!   (CPU) ceiling, each **tunable per-invocation** with **deterministic defaults**.
//! - **[`BudgetError`]** — the canonical never-silent over-budget surface. [`BudgetError::DepthExceeded`]
//!   is the *canonical over-budget variant* (the orchestrator decision — RFC-0041 §5.1): the interp/AOT
//!   `EvalError::DepthLimit` reconcile to it in W4/W3½.
//! - **[`DepthGuard`]** / [`RecursionBudget::charge_bytes`] / [`RecursionBudget::charge_steps`] — the
//!   consumer-side charging. The charge happens *at each machine's frame-push/env-insert site*, never in
//!   this leaf (the §4.1 deps-cycle fix: the leaf exposes only counters/limits and never depends on
//!   `interp`/`core`/`l1`).
//! - **[`ProcessArena`]** — the process-wide memory ceiling (§4.2): a shared atomic byte counter every
//!   pass charges against, so the *sum* over concurrent passes cannot exceed a per-process ceiling.
//! - **[`ensure_sufficient_stack`]** — the host-stack guard helper. **W2 (RFC-0041 §4.3):** its body
//!   now routes through the fine-grained **runtime-gated grow** ([`mycelium_stack::grow`]) layered on
//!   the deep worker base, so the stack is *growable* (not capped) and callers whose internal recursion
//!   points are not yet grow-wired stay non-regressing. The signature is unchanged (W1 consumers intact).
//! - **[`MAX_FRAME_BYTES`]** — the §4.2 per-machine frame-size baseline: the pinned maximum `size_of` of
//!   the three machines' value/frame structs, so a toolchain frame-size bump fails CI, not production
//!   (the ADR-041 lesson; pinned by the `tests/frame_size_baseline.rs` gate).
//! - **[`assert_mem_ceiling_honors_floor`]** — the §4.2 determinism invariant as a checked function
//!   (`mem_ceiling >= depth_floor * max_frame_bytes`). **W2** wires it at startup via [`check_startup`],
//!   together with [`mycelium_stack::growable_ceiling_honors_floor`] (the §4.3 host-stack floor check).
//!
//! # Architecture (DN-68: acyclic, downward-only)
//!
//! `mycelium-workstack` is a **leaf**. It depends on `std` and the `mycelium-stack` host-stack adapter
//! **only** — never on `mycelium-interp`/`mycelium-core`/`mycelium-l1` (those are *upward*). This is the
//! §4.1 deps-cycle fix: the memory ceiling *is* RFC-0014's `Alloc` `EffectBudget` conceptually, but the
//! leaf exposes only the counters/limits and the *charge happens consumer-side*, so no cycle forms.
//!
//! # House rules
//!
//! `#![forbid(unsafe_code)]`. **Never-silent (G2):** every over-budget path returns a [`BudgetError`] —
//! never a panic, `abort`, or silent truncation. **Honesty (VR-5):** the budgets are `Declared`
//! (asserted config) and their sufficiency is `Empirical` (validated by trials/fixtures), **never**
//! `Proven` — there is no machine-checked theorem here, only checked runtime guards.
#![forbid(unsafe_code)]

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The never-silent over-budget surface (§5.1 canonical variant).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// Which non-depth resource a [`BudgetError::OutOfBudget`] refused on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetKind {
    /// Live/allocated **bytes** — the memory ceiling (§4.2).
    Bytes,
    /// Node-visit **work steps** — the CPU/work-step budget (§4.2, guards the `O(N²)` re-walks).
    WorkSteps,
}

impl BudgetKind {
    /// A short, stable label for diagnostics/`EXPLAIN` (house rule #2 — inspectable, never opaque).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            BudgetKind::Bytes => "bytes",
            BudgetKind::WorkSteps => "work-steps",
        }
    }
}

/// The canonical, never-silent over-budget error (RFC-0041 §5.1). Every over-budget path in this crate
/// — and, by reconciliation, in the interp (W4) and AOT (W3½) — surfaces one of these, **never** a
/// panic/abort. Each variant carries the *actionable number that was hit*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// The **canonical over-budget variant** (orchestrator decision, on the §4.0 depth metric). This is
    /// *the* type the interp/AOT `EvalError::DepthLimit { .. }` reconcile to in W4/W3½. `limit` is the
    /// depth ceiling that was reached — the maximum permitted source-call/β nesting.
    DepthExceeded {
        /// The depth ceiling (max permitted concurrent [`DepthGuard`]s) that refused the enter.
        limit: u32,
    },
    /// A non-depth resource (bytes or work-steps) was exhausted. `requested` is the **cumulative total
    /// that would result** from the refused charge (directly comparable to `limit`), so a consumer can
    /// report "needed `requested`, ceiling is `limit`".
    OutOfBudget {
        /// Which resource refused.
        kind: BudgetKind,
        /// The ceiling for `kind`.
        limit: u64,
        /// The cumulative total the refused charge would have reached (`> limit`).
        requested: u64,
    },
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::DepthExceeded { limit } => {
                write!(
                    f,
                    "recursion depth budget exceeded (limit {limit} source-call frames)"
                )
            }
            BudgetError::OutOfBudget {
                kind,
                limit,
                requested,
            } => write!(
                f,
                "{} budget exhausted (needed {requested}, ceiling {limit})",
                kind.label()
            ),
        }
    }
}

impl std::error::Error for BudgetError {}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The per-invocation recursion budget.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// A per-invocation recursion budget: a depth ceiling on the §4.0 metric plus memory and work-step
/// ceilings. **Tunable per-invocation, deterministic defaults.** The runtime charge state uses interior
/// mutability ([`std::cell::Cell`]) so that [`DepthGuard`]s **nest** and [`charge_bytes`](Self::charge_bytes)
/// can be called *while guards are live* — a `&mut self`-borrowing RAII guard could do neither (the
/// outer guard's exclusive borrow would lock the budget for its whole scope).
///
/// Single-threaded per-invocation by construction (each pass owns its budget). It is [`Send`] (so a
/// consumer may *move* it into an [`ensure_sufficient_stack`] worker) but **not** [`Sync`] — cross-pass
/// sharing goes through the [`ProcessArena`], not a shared `&RecursionBudget`.
#[derive(Debug)]
pub struct RecursionBudget {
    depth_limit: u32,
    mem_limit_bytes: u64,
    work_step_limit: u64,
    depth: std::cell::Cell<u32>,
    mem_charged: std::cell::Cell<u64>,
    steps_charged: std::cell::Cell<u64>,
}

impl RecursionBudget {
    /// The global default depth ceiling on the §4.0 metric (RFC-0041 §4.2). The parser's 256 and eval's
    /// 64 are raised *to* this (eval's raise is held to W5, §7).
    pub const DEFAULT_DEPTH_LIMIT: u32 = 4096;

    /// A budget with explicit ceilings. `mem_limit_bytes`/`work_step_limit` of [`u64::MAX`] mean
    /// "effectively unbounded" (the real memory ceiling is wired in W2 after the per-machine frame
    /// census; §4.2).
    #[must_use]
    pub const fn new(depth_limit: u32, mem_limit_bytes: u64, work_step_limit: u64) -> Self {
        Self {
            depth_limit,
            mem_limit_bytes,
            work_step_limit,
            depth: std::cell::Cell::new(0),
            mem_charged: std::cell::Cell::new(0),
            steps_charged: std::cell::Cell::new(0),
        }
    }

    /// A budget with the default depth ceiling and the given memory/work-step ceilings.
    #[must_use]
    pub const fn with_depth_default(mem_limit_bytes: u64, work_step_limit: u64) -> Self {
        Self::new(Self::DEFAULT_DEPTH_LIMIT, mem_limit_bytes, work_step_limit)
    }

    /// Try to enter one source-call/β frame (the §4.0 metric unit). Increments the live depth and
    /// returns a [`DepthGuard`] that **decrements on `Drop`** (so a caller cannot forget to release).
    /// Refuses never-silently with [`BudgetError::DepthExceeded`] when the enter would push the live
    /// depth *past* [`depth_limit`](Self::depth_limit) — so at most `depth_limit` guards are live at once.
    ///
    /// Takes `&self` (not `&mut self`) precisely so nested enters compose and charging can run alongside
    /// live guards.
    ///
    /// # Errors
    /// [`BudgetError::DepthExceeded`] if the resulting depth would exceed the ceiling.
    pub fn try_enter(&self) -> Result<DepthGuard<'_>, BudgetError> {
        let next = self.depth.get().saturating_add(1);
        if next > self.depth_limit {
            return Err(BudgetError::DepthExceeded {
                limit: self.depth_limit,
            });
        }
        self.depth.set(next);
        Ok(DepthGuard { depth: &self.depth })
    }

    /// Charge `n` bytes against the memory ceiling. Never-silent: refuses with
    /// [`BudgetError::OutOfBudget`] (`kind = Bytes`) when the cumulative charge would exceed the ceiling,
    /// and does **not** apply the charge in that case (the counter is unchanged on refusal).
    ///
    /// # Errors
    /// [`BudgetError::OutOfBudget`] when the cumulative byte charge would exceed the ceiling.
    pub fn charge_bytes(&self, n: u64) -> Result<(), BudgetError> {
        Self::charge_cell(
            &self.mem_charged,
            self.mem_limit_bytes,
            n,
            BudgetKind::Bytes,
        )
    }

    /// Charge `n` node-visit work steps against the work-step (CPU) ceiling. Never-silent: refuses with
    /// [`BudgetError::OutOfBudget`] (`kind = WorkSteps`) when the cumulative charge would exceed the
    /// ceiling; the counter is unchanged on refusal.
    ///
    /// # Errors
    /// [`BudgetError::OutOfBudget`] when the cumulative step charge would exceed the ceiling.
    pub fn charge_steps(&self, n: u64) -> Result<(), BudgetError> {
        Self::charge_cell(
            &self.steps_charged,
            self.work_step_limit,
            n,
            BudgetKind::WorkSteps,
        )
    }

    /// The one charge primitive (DRY) both [`charge_bytes`](Self::charge_bytes) and
    /// [`charge_steps`](Self::charge_steps) route through.
    fn charge_cell(
        cell: &std::cell::Cell<u64>,
        limit: u64,
        n: u64,
        kind: BudgetKind,
    ) -> Result<(), BudgetError> {
        let next = cell.get().saturating_add(n);
        if next > limit {
            return Err(BudgetError::OutOfBudget {
                kind,
                limit,
                requested: next,
            });
        }
        cell.set(next);
        Ok(())
    }

    /// The configured depth ceiling.
    #[must_use]
    pub const fn depth_limit(&self) -> u32 {
        self.depth_limit
    }
    /// The configured memory ceiling in bytes.
    #[must_use]
    pub const fn mem_limit_bytes(&self) -> u64 {
        self.mem_limit_bytes
    }
    /// The configured work-step ceiling.
    #[must_use]
    pub const fn work_step_limit(&self) -> u64 {
        self.work_step_limit
    }
    /// The current live depth (inspectable — house rule #2 / `EXPLAIN`).
    #[must_use]
    pub fn current_depth(&self) -> u32 {
        self.depth.get()
    }
    /// The cumulative bytes charged so far.
    #[must_use]
    pub fn current_bytes(&self) -> u64 {
        self.mem_charged.get()
    }
    /// The cumulative work steps charged so far.
    #[must_use]
    pub fn current_steps(&self) -> u64 {
        self.steps_charged.get()
    }
}

impl Default for RecursionBudget {
    /// The default budget: the [`DEFAULT_DEPTH_LIMIT`](Self::DEFAULT_DEPTH_LIMIT) depth ceiling and
    /// **unbounded** memory/work-step ceilings ([`u64::MAX`]) — the real memory ceiling is wired in W2.
    fn default() -> Self {
        Self::with_depth_default(u64::MAX, u64::MAX)
    }
}

/// The RAII depth reservation returned by [`RecursionBudget::try_enter`]. Holds a shared borrow of the
/// budget's depth counter and **decrements it on `Drop`** — so a frame's depth charge is released
/// exactly when the frame's scope ends, and a caller cannot leak it. Because it borrows *shared*
/// (`&`), multiple guards nest freely and charging runs alongside.
#[derive(Debug)]
pub struct DepthGuard<'a> {
    depth: &'a std::cell::Cell<u32>,
}

impl Drop for DepthGuard<'_> {
    fn drop(&mut self) {
        // Cannot underflow given construction (each guard corresponds to exactly one increment), but
        // saturate defensively — never panic in `Drop` (G2).
        self.depth.set(self.depth.get().saturating_sub(1));
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The process-wide memory arena (§4.2): a shared ceiling across concurrent passes.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The process-wide charged-byte counter. The per-invocation [`RecursionBudget`] caps *one* pass; but
/// LSP re-analyses, parallel eval workers, and spore batches run **many** passes at once, so the per-pass
/// cap alone multiplies under concurrency (§4.2). Every [`ProcessArena::reserve`] charges this shared
/// counter, so the *sum* across concurrent passes is what the ceiling bounds.
static PROCESS_BYTES_CHARGED: AtomicU64 = AtomicU64::new(0);

/// The current process-wide reserved-byte total (inspectable — house rule #2 / `EXPLAIN`).
#[must_use]
pub fn current_process_bytes() -> u64 {
    PROCESS_BYTES_CHARGED.load(Ordering::Relaxed)
}

/// The process-wide memory arena (§4.2). Its ceiling bounds the **sum** of live reservations across all
/// concurrent passes (LSP re-analyses / parallel eval workers / spore batch), refusing never-silently
/// when a reservation would push the process-wide total over the ceiling. Reservations release on drop.
///
/// All instances charge the same process-global counter ([`current_process_bytes`]); construct the arena
/// once per process with the intended ceiling.
#[derive(Debug, Clone)]
pub struct ProcessArena {
    ceiling_bytes: u64,
}

impl ProcessArena {
    /// A process arena with the given per-process byte ceiling.
    #[must_use]
    pub const fn new(ceiling_bytes: u64) -> Self {
        Self { ceiling_bytes }
    }

    /// The configured per-process byte ceiling.
    #[must_use]
    pub const fn ceiling_bytes(&self) -> u64 {
        self.ceiling_bytes
    }

    /// Reserve `bytes` against the process-wide ceiling. Returns an RAII [`ArenaReservation`] that
    /// **releases the bytes on `Drop`**. Never-silent: refuses with [`BudgetError::OutOfBudget`]
    /// (`kind = Bytes`) — atomically, so two concurrent reservations that would *jointly* exceed the
    /// ceiling cannot both succeed (a compare-exchange loop, not a racy load-then-add).
    ///
    /// # Errors
    /// [`BudgetError::OutOfBudget`] when the process-wide total would exceed [`ceiling_bytes`](Self::ceiling_bytes).
    pub fn reserve(&self, bytes: u64) -> Result<ArenaReservation<'_>, BudgetError> {
        let mut current = PROCESS_BYTES_CHARGED.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_add(bytes);
            if next > self.ceiling_bytes {
                return Err(BudgetError::OutOfBudget {
                    kind: BudgetKind::Bytes,
                    limit: self.ceiling_bytes,
                    requested: next,
                });
            }
            match PROCESS_BYTES_CHARGED.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(ArenaReservation {
                        bytes,
                        _arena: PhantomData,
                    })
                }
                Err(observed) => current = observed,
            }
        }
    }
}

/// An RAII reservation of process-wide bytes, from [`ProcessArena::reserve`]. Releases its bytes back to
/// the process-wide counter on `Drop`. Borrows the arena, so the arena outlives its reservations.
#[derive(Debug)]
pub struct ArenaReservation<'a> {
    bytes: u64,
    _arena: PhantomData<&'a ProcessArena>,
}

impl ArenaReservation<'_> {
    /// The number of bytes this reservation holds against the process ceiling.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for ArenaReservation<'_> {
    fn drop(&mut self) {
        // Release exactly what we reserved. `fetch_sub` cannot underflow: every live reservation was
        // added by a successful `reserve`, and each is subtracted exactly once.
        PROCESS_BYTES_CHARGED.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The host-stack guard helper + the memory-ceiling determinism invariant.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// Run `f` on a host stack large enough that the [`RecursionBudget`] — not a host-stack overflow —
/// always bounds a pathological input.
///
/// **W2 body swap (RFC-0041 §4.3).** The body now routes `f` through the fine-grained **runtime-gated
/// grow** ([`mycelium_stack::grow`], a `#![forbid(unsafe_code)]` wrapper of `stacker::maybe_grow`)
/// layered on the deep worker base ([`mycelium_stack::with_deep_stack`]). The signature is unchanged, so
/// every W1 consumer is untouched. What changes is the *contract*: the guarded stack is now **growable**
/// rather than capped at the 256 MiB worker — if a pass ever recurses past even that base (deep input at
/// a large frame size), the runtime-gated grow enlarges the stack on demand instead of a `SIGABRT`.
///
/// **Why keep the worker base (non-regression, honest scope).** The material "pay for the depth you use"
/// benefit of the fine-grained grow requires the *recursion points inside `f`* to call
/// [`mycelium_stack::ensure_sufficient_stack`] stride-1 (RFC-0041 §4.7 / W4 / W3½ — consumer wiring, not
/// this leaf). Until those are wired, the deep worker base is what keeps a caller whose internal
/// recursion is *not* yet grow-wired (e.g. `mir-passes::emit`, `lsp::render_node`) safe on deep input —
/// so dropping it would regress them. The top-level `grow` here backstops the base; the per-point stride
/// growth lands with the consumers. `budget` is threaded for that future wiring (it will size per-point
/// growth); the growable base needs no sizing, so it is not read yet.
///
/// The growth is **bounded, not a memory-DoS vector**: `budget`'s depth ceiling (default 4096) refuses
/// recursion before the stack can grow without bound.
pub fn ensure_sufficient_stack<R, F>(budget: &RecursionBudget, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    // `budget` sizes the future per-recursion-point stride growth (§4.7); the growable base is generous
    // and needs no sizing input, so it is threaded but not yet read.
    let _ = budget;
    // Deep lazily-committed worker base (non-regressing) + runtime-gated grow backstop (§4.3).
    mycelium_stack::with_deep_stack(move || mycelium_stack::grow(f))
}

/// The **depth floor** every execution machine honors (RFC-0041 §4.4): the global default depth ceiling
/// on the §4.0 metric. Both startup invariants ([`check_startup`]) are stated against this floor — the
/// memory ceiling and the host stack must each be able to hold it.
pub const DEPTH_FLOOR: u32 = RecursionBudget::DEFAULT_DEPTH_LIMIT;

/// The §4.2 **frame-size baseline**: the pinned maximum `size_of` (bytes) of the three execution
/// machines' value/frame structs — the interp/L0 `CoreValue`/`Node`, the L1 `L1Value`, and the AOT
/// env-machine `Frame` — used as `max_frame_bytes` in the determinism invariant
/// ([`assert_mem_ceiling_honors_floor`]). Pinning it keeps the (frame-size-dependent, machine-dependent)
/// memory ceiling a fixed distance from the accept/reject boundary, and — via the
/// `tests/frame_size_baseline.rs` gate — makes a **toolchain/IR frame-size bump fail CI, not production**
/// (the ADR-041 lesson).
///
/// **Current measured max (64-bit): 328 bytes** — the AOT `Frame` (the value structs are 240). The
/// baseline carries a small documented headroom over that so cross-target padding jitter does not
/// false-trip, while a genuine field addition past it does. **On an intended frame-size change,
/// re-measure all three machines and bump this** (the baseline test's failure message says so).
pub const MAX_FRAME_BYTES: u64 = 384;

/// The conservative **per-recursion-frame host-stack cost** (bytes) used for the §4.3 host-stack floor
/// check ([`mycelium_stack::growable_ceiling_honors_floor`]). This is the *call-stack* cost of one guarded
/// recursion level — **distinct from [`MAX_FRAME_BYTES`]**, which is the *heap* footprint of a value/frame
/// struct. Set generously above the measured worst case (the L1 checker's ~10.9 KiB/frame in debug) so the
/// no-grow floor refusal is conservative (it can only *over*-estimate the stack a floor needs, never
/// under-estimate it into a silent overflow).
pub const HOST_STACK_BYTES_PER_FRAME: u64 = 16 * 1024;

/// Check the §4.2 **determinism invariant**: the memory ceiling must be at least `depth_floor`
/// frames of the largest frame, `mem_ceiling >= depth_floor * max_frame_bytes`. This keeps the
/// (frame-size-dependent, hence machine-dependent) memory limit **off** the observable accept/reject
/// boundary within the depth floor — so the boundary stays deterministic on the one §4.0 metric.
///
/// **W2 wires it at startup** via [`check_startup`], using the [`MAX_FRAME_BYTES`] census against the
/// [`DEPTH_FLOOR`]. The function remains callable standalone for tests/tools.
///
/// # Errors
/// [`BudgetError::OutOfBudget`] (`kind = Bytes`) when `mem_limit_bytes < depth_floor * max_frame_bytes`;
/// `requested` is the required product (saturating, so an overflowing product reports [`u64::MAX`]).
pub fn assert_mem_ceiling_honors_floor(
    mem_limit_bytes: u64,
    depth_floor: u32,
    max_frame_bytes: u64,
) -> Result<(), BudgetError> {
    let required = u64::from(depth_floor).saturating_mul(max_frame_bytes);
    if mem_limit_bytes < required {
        return Err(BudgetError::OutOfBudget {
            kind: BudgetKind::Bytes,
            limit: mem_limit_bytes,
            requested: required,
        });
    }
    Ok(())
}

/// The never-silent **startup refusal** surface (RFC-0041 §4.2/§4.3): the one error [`check_startup`]
/// returns when a machine must **not** begin because a determinism/safety precondition fails. Refusing at
/// startup with an actionable diagnostic is the explicit alternative to a mid-run surprise — an
/// accept/reject boundary that silently depends on the memory ceiling, or a `SIGABRT` below the depth
/// floor (G2). `Display` + `std::error::Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupError {
    /// The §4.2 determinism invariant failed: the configured memory ceiling is below
    /// `DEPTH_FLOOR × MAX_FRAME_BYTES`, so the (machine-dependent) memory limit could bind *within* the
    /// depth floor and make the accept/reject boundary non-deterministic.
    MemCeiling(BudgetError),
    /// The §4.3 host-stack floor is unsatisfiable: on-demand growth is unavailable on this target and
    /// the fixed stack cannot hold the depth floor — so the machine would overflow below the floor.
    HostStack(mycelium_stack::StackError),
}

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupError::MemCeiling(e) => write!(
                f,
                "startup refused (RFC-0041 §4.2 determinism invariant): memory ceiling cannot honor the \
                 depth floor — {e}"
            ),
            StartupError::HostStack(e) => write!(
                f,
                "startup refused (RFC-0041 §4.3 host-stack floor): {e}"
            ),
        }
    }
}

impl std::error::Error for StartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StartupError::MemCeiling(e) => Some(e),
            StartupError::HostStack(e) => Some(e),
        }
    }
}

/// The **startup gate** (RFC-0041 §4.2/§4.3): before any execution machine runs deep input, assert both
/// preconditions against the workspace [`DEPTH_FLOOR`] and refuse never-silently if either fails.
///
/// 1. **Memory-ceiling determinism (§4.2):** `budget`'s memory ceiling must be at least
///    `DEPTH_FLOOR × MAX_FRAME_BYTES` (via [`assert_mem_ceiling_honors_floor`]) — so the memory limit can
///    never bind at or below the floor and perturb the accept/reject boundary.
/// 2. **Host-stack floor (§4.3):** the host stack must be able to hold the floor — either because
///    on-demand growth is available, or because the fixed no-grow ceiling is large enough (via
///    [`mycelium_stack::growable_ceiling_honors_floor`], using [`HOST_STACK_BYTES_PER_FRAME`]). On a
///    no-grow target (`wasm32`) that cannot hold the floor, this refuses rather than risk a `SIGABRT`.
///
/// Call once per machine at startup. It is a pure read of the passed budget plus the runtime
/// growth-availability probe — idempotent, side-effect-free.
///
/// # Errors
/// [`StartupError::MemCeiling`] if precondition 1 fails; [`StartupError::HostStack`] if precondition 2
/// fails. Precondition 1 is checked first.
pub fn check_startup(budget: &RecursionBudget) -> Result<(), StartupError> {
    assert_mem_ceiling_honors_floor(budget.mem_limit_bytes(), DEPTH_FLOOR, MAX_FRAME_BYTES)
        .map_err(StartupError::MemCeiling)?;
    mycelium_stack::growable_ceiling_honors_floor(DEPTH_FLOOR, HOST_STACK_BYTES_PER_FRAME)
        .map_err(StartupError::HostStack)?;
    Ok(())
}

#[cfg(test)]
mod tests;
