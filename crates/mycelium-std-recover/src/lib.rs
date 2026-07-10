//! `std.recover` — the declarative recovery bridge (M-520, issue #156; **Rust-first half only**).
//!
//! # Scope (honesty)
//!
//! This crate is the **Rust-first library surface** of `std.recover`. The **self-hosting migration**
//! half of M-520 (recover authored in Mycelium-lang) is **Batch P5-C, M-502-gated** and is
//! *not* in this wave (`docs/spec/stdlib/recover.md` §Status).
//!
//! # Architecture (KC-3 / NFR-7)
//!
//! Recovery elaborates to an L0 `Match` over the error sum — **no new kernel node** (RFC-0014
//! §4.3 / KC-3).  The driver and the policy are ordinary functions/values; budget enforcement
//! lives in `mycelium-interp` (M-353), not in the kernel calculus.  The self-hosted form will be
//! held against this Rust origin by a differential (NFR-7) in Batch P5-C.
//!
//! # Invariants (RFC-0014 §4.2/§4.5; RFC-0016 C1–C6)
//!
//! - **(I1) Never-silent.** [`handle_classified`] always yields [`Resolution::Recovered`] or
//!   [`Resolution::Propagated`] — there is no `Dropped` variant.  Every error is either
//!   explicitly recovered or explicitly re-propagated.
//! - **(I2) Honest tags.** The recovered tag is **at most `Declared`** for a `Fallback`;
//!   **inherited** for a `Retry` success; **`Exact`** for a clean `Ok` pass-through.  Recovery
//!   never launders a tag upward (VR-5).  This fixes the P5-B exact-tag bug (FR-R3).
//! - **(I3) Effects are declared.** [`effect::check_effects`] catches any performed-but-undeclared
//!   effect as an explicit [`effect::UndeclaredEffect`] (no unknown side effects).
//! - **(I4) Budgets overrun gracefully.** [`effect::Budgets::consume`] returns an explicit
//!   [`effect::EffectBudgetExhausted`] on overrun — never a hang, OOM, or panic.
//! - **(I5) Tightly scoped by default.** An effect with no declared budget cannot run.
//!
//! # Module layout
//!
//! - [`action`] — the closed v0 recovery-action set (`Fallback` / `Retry` / `Escalate` /
//!   `CleanupThenPropagate`; RFC-0014 §4.4/§8).
//! - [`registry`] — a minimal error-class registry (`ClassRegistry`, `ClassName`, `UnknownClass`);
//!   the Rust-first stand-in for `std.diag`'s shared registry (M-510).
//! - [`policy`] — the reified, content-addressed `RecoveryPolicy<T>` (RFC-0005 pattern; ADR-006)
//!   and `PolicyRef = ContentHash` (RFC-0001 §4.6).
//! - [`outcome`] — the `Outcome<T,E>` input sum and the `Resolution<T,E>` output sum (no
//!   `Dropped` variant — I1).
//! - [`effect`] — re-exports from `mycelium_interp::budget` + `check_effects`/`UndeclaredEffect`.
//! - [`handle`] — the never-silent driver: `handle_classified`, `recover_classified`.
//! - [`guarantee_matrix`] — the §4.5 matrix as checked data (RFC-0016 §4.5).
//!
//! # Design spec
//!
//! `docs/spec/stdlib/recover.md`; RFC-0014; task M-520, issue #156.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A):** Recovery operations are representation-aware: a recover
//! over a `Dense` value does not silently re-interpret the payload as `Ternary` on fallback.
//! The recovered value's `Repr` matches the source; a `Fallback` that substitutes a different
//! representation requires an explicit swap (and yields a `Declared` tag — I2/VR-5).
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/recover.md` (spec status:
//! Accepted, Rust-first half (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod action;
pub mod effect;
pub mod guarantee_matrix;
pub mod handle;
pub mod outcome;
pub mod policy;
pub mod registry;

#[cfg(test)]
mod tests;

// ---- top-level re-exports (the primary public surface) ----------------------

pub use action::RecoveryAction;
pub use effect::{
    check_effects, Budgets, EffectBudget, EffectBudgetExhausted, EffectKind, EffectSet,
    UndeclaredEffect,
};
pub use handle::{handle_classified, recover_classified};
pub use outcome::{DiagError, Outcome, Resolution};
pub use policy::{policy_effects, PolicyHashError, PolicyRef, RecoveryPolicy};
pub use registry::{ClassName, ClassRegistry, UnknownClass};

/// `RecoverOutcome<T, E>` is `Resolution<T, E>` — the concrete shape that resolves
/// `error.md` §7-Q1 (the `std.error` bridge signature owned here).
///
/// The bridge is [`handle_classified`] itself; `RecoverOutcome = Recovered | Propagated`, no
/// drop variant (I1), honest inherited tag (I2).  `error`'s contract holds verbatim.
pub type RecoverOutcome<T, E> = Resolution<T, E>;
