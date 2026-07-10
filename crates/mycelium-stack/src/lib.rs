//! **Host-stack management for the L1 frontend's recursive passes — isolated outside the trusted
//! kernel.**
//!
//! The L1 checker ([`mycelium_l1::checkty`]) and elaborator ([`mycelium_l1::elab`]) recurse over the
//! expression AST. They must never overflow the *caller's* thread stack on deep input — but the
//! caller's stack size is a host resource, not a semantic limit, and it varies (a 2 MiB worker thread
//! vs the larger `main` stack), and it interacts with frame size (which grows as the IR evolves).
//! Coupling correctness to that resource is fragile. This crate breaks the coupling **without putting
//! any `unsafe` in the trusted kernel**.
//!
//! ## Why a separate crate (the architecture)
//!
//! - **The kernel stays `unsafe`-free and *machine-proven*.** `mycelium-l1` is `#![forbid(unsafe_code)]`
//!   (ADR-014: "trusted-base crates should stay unsafe-free; may re-pin `unsafe_code = forbid`
//!   per-crate"). All host-stack machinery lives **here**, behind a safe API the kernel calls.
//! - **The semantic budgets stay in the kernel.** The parser caps surface nesting
//!   (`mycelium_l1::parse::MAX_EXPR_DEPTH`), the checker carries `MAX_CHECK_DEPTH`, and the evaluator a
//!   per-node depth clock — each an **explicit, reified budget** that refuses past it with a clean
//!   error, never a crash (the "banked guard 4" discipline). *Those* are the bound on a pathological
//!   input. This crate only ensures the host stack is large enough that the **budget**, not an
//!   overflow, is always what stops it.
//! - **Self-hosting:** the explicit budgets are the **portable primitive** — they carry directly to the
//!   future Mycelium-native frontend, whose value-semantic, fuel/clock-bounded model has *no host call
//!   stack to grow* (RFC-0007 §4.5/§4.6). **This crate is the transitional Rust-host adapter** and is
//!   expected to disappear when the frontend self-hosts (the budgets stay; the stack sizing does not).
//!
//! ## The coarse worker (`with_deep_stack`) — the lazily-committed base (`unsafe`-free)
//!
//! [`with_deep_stack`] runs a recursive pass on a dedicated worker thread with a large explicit stack.
//! The address space is *reserved* up front (cheap) and physical pages are touched only as recursion
//! actually deepens — so a shallow program pays ~nothing (the same "pay for the depth you use" benefit
//! as a segmented stack), with **zero `unsafe`**: it is pure `std::thread`. It is the generous,
//! non-regressing **base** the fine-grained grow layers on.
//!
//! ## The fine-grained grow (`ensure_sufficient_stack`) — runtime-gated, `unsafe`-free (RFC-0041 §4.3)
//!
//! [`ensure_sufficient_stack`] wraps [`stacker::maybe_grow`] — rustc's own deep-recursion pattern: at a
//! recursion point, if less than `red_zone` stack remains, allocate a fresh `stack_size` segment and
//! run the callback on it; otherwise run it inline (near-zero cost). `stacker::maybe_grow` is a **SAFE
//! API** — the stack-switching `unsafe` (and its inline asm, via `psm`) is internal to the `stacker`
//! leaf crate. So **this crate stays `#![forbid(unsafe_code)]`**: we author no `unsafe`; it is contained
//! in a single audited upstream leaf, the furthest possible point from the kernel (ADR-014/KC-3).
//!
//! **Runtime-gated, not feature-gated.** Growth is enabled by *probing the platform at runtime*
//! ([`stack_growth_available`] via [`stacker::remaining_stack`]), never by a cargo feature. On a
//! **no-grow target** (`wasm32` — `psm`'s stack switch is a silent no-op) the probe *reveals* that
//! growth is unavailable, and [`growable_ceiling_honors_floor`] **refuses to start** when the fixed
//! physical ceiling cannot hold the depth floor — an explicit [`StackError`], **never** a silent
//! `SIGABRT` below the floor (G2). The no-op case is detected and surfaced, never a silent degrade.
//!
//! **Bounded, not unbounded.** Growth is not a memory-DoS vector: the caller's explicit
//! `RecursionBudget` depth ceiling (`mycelium-workstack`, default 4096) refuses recursion *before* the
//! stack can grow without bound, so total growth is bounded by `floor × stack-bytes-per-frame`.
#![forbid(unsafe_code)]

use core::fmt;

/// A generous worker-thread stack for the coarse [`with_deep_stack`] base. Reserved virtually;
/// committed lazily, so a shallow pass touches only a handful of pages. Comfortably exceeds what the
/// kernel's explicit depth budgets admit at any plausible frame size, so those budgets — not a
/// host-stack overflow — always bound a pathological input. (Measured: the L1 checker uses ~10.9 KiB
/// per frame in debug, so 256 MiB physically supports ~24,600 levels — ~6× the checker's 4096 budget.)
const DEEP_STACK_BYTES: usize = 256 * 1024 * 1024;

/// The default red zone for the fine-grained grow: the minimum remaining stack (bytes) below which
/// [`ensure_sufficient_stack`] grows before running the callback. Generous relative to any single
/// recursion frame (the L1 checker's ~10.9 KiB), so a stride-1 check has ample headroom and can never
/// overrun a frame between checks (the RFC-0041 §4.3 "stride S is overrun-safe only if
/// `red_zone ≥ S × max_frame`" condition, met with `S = 1` and a large margin). rustc uses 100 KiB; we
/// take a rounder, slightly larger 128 KiB.
pub const DEFAULT_RED_ZONE: usize = 128 * 1024;

/// The default new-segment size the fine-grained grow allocates on each grow event. rustc uses 1 MiB;
/// a larger 16 MiB amortises the (real, committed) allocation over many recursion levels so growth is
/// infrequent, while staying small next to the 256 MiB coarse base.
pub const DEFAULT_GROW_SEGMENT: usize = 16 * 1024 * 1024;

/// The conservative physical usable-stack ceiling **assumed on a no-grow target** (e.g. `wasm32`, where
/// `psm`'s stack switch is a silent no-op — RFC-0041 §4.3). On such a target the host thread stack
/// cannot be enlarged, so this is the ceiling [`growable_ceiling_honors_floor`] tests the depth floor
/// against. Deliberately small (1 MiB — a typical `wasm32` linker default) so that a floor the fixed
/// stack genuinely cannot hold is **refused at startup**, never discovered as a `SIGABRT` mid-run. A
/// deployment that provisions *less* than this on a no-grow target must lower the depth floor to match;
/// one that provisions more is only ever refused conservatively (never a false "OK"). Growth-available
/// targets ignore this constant — their ceiling is bounded by the depth budget, not a fixed stack.
pub const NO_GROW_CEILING_BYTES: u64 = 1024 * 1024;

/// Run `f` on a worker thread with a large explicit stack ([`DEEP_STACK_BYTES`]) and return its value.
///
/// A panic inside `f` is propagated to the caller unchanged (via [`std::panic::resume_unwind`]), so
/// assertions and `#[should_panic]` behave exactly as if `f` had run inline. The closure may borrow
/// from the caller (it runs on a *scoped* thread), so the recursive passes keep taking `&Nodule` /
/// `&Env` by reference. `unsafe`-free: pure `std::thread`.
///
/// Cost: one worker-thread spawn per call (tens of microseconds) — negligible for a compiler pass and
/// paid once at the top of `check`/`elaborate`, not per recursion level. The large stack is virtual
/// address space, committed lazily, so a shallow pass uses only a few pages.
pub fn with_deep_stack<T, F>(f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("mycelium-deep-stack".to_owned())
            .stack_size(DEEP_STACK_BYTES)
            .spawn_scoped(scope, f)
            .expect("spawn the deep-stack worker thread")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

/// Fine-grained runtime-gated stack grow (RFC-0041 §4.3): ensure at least `red_zone` bytes of stack
/// remain before running `f`, allocating a fresh `stack_size` segment to run it on if not. This is the
/// primitive the still-recursive guarded passes call **at each genuine recursion point** (stride-1,
/// rustc's `ensure_sufficient_stack` pattern) so a deep input grows on demand instead of overflowing.
///
/// A thin, `unsafe`-free wrapper over [`stacker::maybe_grow`] (its stack-switching `unsafe` is internal
/// to `stacker`/`psm`). On a **no-grow target** `maybe_grow` runs `f` inline without growing — which is
/// why the depth floor must be validated up front via [`growable_ceiling_honors_floor`], so growth's
/// absence there is a startup refusal, not a silent overflow.
///
/// Prefer [`grow`] for the fixed generous defaults; use this when a caller must tune the stride.
pub fn ensure_sufficient_stack<R>(red_zone: usize, stack_size: usize, f: impl FnOnce() -> R) -> R {
    stacker::maybe_grow(red_zone, stack_size, f)
}

/// Fine-grained grow with the fixed generous defaults ([`DEFAULT_RED_ZONE`] / [`DEFAULT_GROW_SEGMENT`]).
/// The convenience most recursion-point call-sites should use (rustc's `ensure_sufficient_stack`).
pub fn grow<R>(f: impl FnOnce() -> R) -> R {
    ensure_sufficient_stack(DEFAULT_RED_ZONE, DEFAULT_GROW_SEGMENT, f)
}

/// Whether on-demand stack growth is available on the current target **at runtime** (RFC-0041 §4.3).
///
/// Growth is gated on this probe, **not** on a cargo feature: [`stacker::remaining_stack`] returns
/// `Some` on a platform where `stacker`/`psm` can measure and switch the stack, and `None` where it
/// cannot (e.g. `wasm32`, where the stack switch is a silent no-op). We report the latter honestly so a
/// caller can refuse rather than silently degrade (G2). This is the honest signal — it reflects what
/// `stacker` itself can actually do on this target, not a compile-time assumption.
#[must_use]
pub fn stack_growth_available() -> bool {
    stacker::remaining_stack().is_some()
}

/// The never-silent host-stack refusal surface (RFC-0041 §4.3). Returned when the machine **cannot
/// safely start** because the host stack cannot hold the depth floor and cannot be grown to. Refusing
/// here — with the actionable numbers — is the explicit alternative to a silent `SIGABRT` below the
/// floor (G2). `Display` + `std::error::Error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StackError {
    /// On-demand growth is **unavailable** on this target (e.g. `wasm32`, where `psm` is a no-op) and
    /// the fixed physical stack ceiling ([`NO_GROW_CEILING_BYTES`]) is **smaller** than the depth floor
    /// requires (`floor × stack_bytes_per_frame`). The machine refuses to start rather than risk an
    /// overflow past the floor. Carries the numbers so the diagnostic is actionable (house rule #2).
    FloorUnsatisfiableOnNoGrowTarget {
        /// The depth floor (frames) the machine must be able to reach without overflow.
        floor: u32,
        /// The assumed per-frame host-stack cost in bytes used for the check.
        stack_bytes_per_frame: u64,
        /// The required stack, `floor × stack_bytes_per_frame` (saturating).
        required: u64,
        /// The fixed physical ceiling available on this no-grow target ([`NO_GROW_CEILING_BYTES`]).
        ceiling: u64,
    },
}

impl fmt::Display for StackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StackError::FloorUnsatisfiableOnNoGrowTarget {
                floor,
                stack_bytes_per_frame,
                required,
                ceiling,
            } => write!(
                f,
                "host-stack floor unsatisfiable: on-demand stack growth is unavailable on this target \
                 (a no-grow target such as wasm32), and the fixed usable stack ({ceiling} bytes) is \
                 smaller than the depth floor requires (floor {floor} frames × {stack_bytes_per_frame} \
                 bytes/frame = {required} bytes). Refusing to start rather than overflow below the \
                 floor; lower the depth floor or provision a larger stack on this target."
            ),
        }
    }
}

impl std::error::Error for StackError {}

/// Check that the host stack can hold the depth floor — refusing never-silently when it cannot and
/// cannot be grown (RFC-0041 §4.3). `stack_bytes_per_frame` is the (conservative) per-recursion-frame
/// host-stack cost; `floor` is the depth floor every machine must reach without overflow.
///
/// - **Growth available** (the common native case): the fine-grained grow ([`ensure_sufficient_stack`])
///   can enlarge the stack on demand, and that growth is *bounded* by the caller's depth budget — so the
///   floor is always satisfiable. Returns `Ok(())`.
/// - **Growth unavailable** (a no-grow target — `wasm32`, `psm` no-op): the physical ceiling is the fixed
///   [`NO_GROW_CEILING_BYTES`]. If `floor × stack_bytes_per_frame` exceeds it, the machine **would**
///   overflow past the floor, so this refuses with [`StackError::FloorUnsatisfiableOnNoGrowTarget`]
///   rather than let it start toward a silent `SIGABRT`.
///
/// This is deliberately conservative: it can only *refuse* on the no-grow path (never a false OK), and
/// on the grow path it trusts the depth budget to bound growth (the caller must actually charge depth —
/// that is the never-silent budget's job, exercised elsewhere).
///
/// # Errors
/// [`StackError::FloorUnsatisfiableOnNoGrowTarget`] when growth is unavailable **and**
/// `NO_GROW_CEILING_BYTES < floor × stack_bytes_per_frame`.
pub fn growable_ceiling_honors_floor(
    floor: u32,
    stack_bytes_per_frame: u64,
) -> Result<(), StackError> {
    check_floor(
        stack_growth_available(),
        NO_GROW_CEILING_BYTES,
        floor,
        stack_bytes_per_frame,
    )
}

/// The pure decision the public [`growable_ceiling_honors_floor`] wraps (DRY + deterministically
/// testable): given whether growth is available and the fixed no-grow `ceiling`, decide whether the
/// depth floor is satisfiable. Extracted so the **no-grow refusal path can be exercised on any host**
/// (a native test host always has growth *available*, so the refusal branch is unreachable through the
/// public probe) — the test simulates `growth_available = false` with a tiny `ceiling`.
fn check_floor(
    growth_available: bool,
    ceiling: u64,
    floor: u32,
    stack_bytes_per_frame: u64,
) -> Result<(), StackError> {
    if growth_available {
        // Growth covers the floor; total growth is bounded above by the depth budget, not this stack.
        return Ok(());
    }
    let required = u64::from(floor).saturating_mul(stack_bytes_per_frame);
    if ceiling < required {
        return Err(StackError::FloorUnsatisfiableOnNoGrowTarget {
            floor,
            stack_bytes_per_frame,
            required,
            ceiling,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
