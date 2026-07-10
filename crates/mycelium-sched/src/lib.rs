//! `mycelium-sched` — the foundational work-stealing OS-thread [`scheduler::Scheduler`]
//! (M-709 / M-861 / RFC-0008 RT1·RT2·RT3, relocated per the E25/M-862 dependency-cycle fix).
//!
//! # Why this crate exists (relocation rationale)
//!
//! The Scheduler originally lived in `mycelium-std-runtime`, which also depends on
//! `mycelium-interp` (to reuse the M-356 supervision kernel: `CancelToken`/`TaskOutcome`/
//! `Supervisor`). That made `mycelium-interp -> mycelium-sched -> mycelium-interp` a cycle the
//! moment the interpreter wanted to use the Scheduler for its own parallel evaluation (M-862) —
//! Cargo rejects circular package dependencies outright, and they are not permitted here on
//! principle either.
//!
//! The Scheduler itself needs none of `mycelium-std-runtime`'s surface (`Colony`/`Task`/
//! supervision/etc.) — only `mycelium_core::GuaranteeStrength` for its honesty tags — so it moves
//! down to this new, foundational crate:
//!
//! ```text
//! mycelium-core
//!       ^
//!       |
//! mycelium-sched  <---------------------+
//!       ^                               |
//!       |                               |
//! mycelium-interp  <---  mycelium-std-runtime
//! ```
//!
//! `mycelium-std-runtime` re-exports [`scheduler::Scheduler`] at the same path
//! (`mycelium_std_runtime::scheduler::Scheduler`) so existing consumers (the M-859 bench)
//! compile unchanged. `mycelium-interp` now depends directly on this crate so its own
//! parallel-evaluation work (M-862) can use the Scheduler without reintroducing the cycle.
//!
//! **Trusted-base discipline (ADR-014):** zero `unsafe` — compiler-enforced.
//!
//! # M-864 — the persistent work-stealing pool
//!
//! [`scheduler::Scheduler::run_indexed`] no longer spawns fresh OS threads per call — it dispatches
//! onto a process-wide, persistent, bounded [`pool`], sized once to `available_parallelism()` and
//! reused for the life of the process, including across **nested** `run_indexed` calls (a worker
//! calling `run_indexed` again from inside a job). See `pool`'s module docs for the help-stealing
//! design and its deadlock-freedom argument, and `docs/notes/DN-67-Persistent-Work-Stealing-Pool.md`
//! for the ratified `'static` job-closure contract change this requires.
#![forbid(unsafe_code)]

mod pool;
pub mod scheduler;

#[cfg(test)]
mod tests;
