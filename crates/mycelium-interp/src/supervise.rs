//! **Concurrency composition primitives** (RFC-0008 §4.7; RFC-0014 §8 concurrency deferral lifted) —
//! per-task budgets, cooperative cancellation, cross-task failure propagation, and `reclaim`
//! bounded-cascade supervision.
//!
//! These are **runtime-orchestration types, not kernel calculus** — they introduce **no L0 node** and
//! do not change the trusted interpreter's sequential evaluation (RT2: "concurrency adds scheduling
//! *outside* the kernel, never new meaning inside it"; KC-3). They live here, beside the shared
//! [`Budgets`] ledger (M-353), because both the recovery driver (`mycelium-lsp`) and
//! the forthcoming RT2 deterministic-fragment runtime (`mycelium-mlir`, M-357) consume them and both
//! crates depend on `mycelium-interp` — the same no-cycle placement the budget primitive used. There
//! is **no task scheduler here**: that is M-357 (RFC-0008 R1). This module is the scheduler-independent
//! composition kernel those tasks will run under.
//!
//! Everything here is **additive over the explicit error** (RFC-0014 I1) and **declared + bounded**
//! (I3/I4): a task failure, a per-task budget overrun, a cancellation, and a supervised restart storm
//! are each an **explicit value**, never a silent stop or an unbounded cascade.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::budget::{Budgets, EffectBudgetExhausted, EffectKind};

/// A **cooperative** cancellation token (RFC-0008 §4.7; structured-concurrency cancellation, RT7).
///
/// Cancellation is **never preemptive**: a task observes the token at its own budget-check points (the
/// same cadence it checks fuel/depth), so it can never be stopped mid-step in a way that drops an
/// in-flight explicit outcome. Observing a cancelled token yields an explicit [`Cancelled`] — an
/// *additive* outcome (RFC-0014 I1), never a silent termination. `Arc<AtomicBool>` so the token can be
/// shared with tasks placed on other threads by the future runtime (M-357) without a redesign.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh, un-cancelled token.
    #[must_use]
    pub fn new() -> Self {
        CancelToken {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation. Idempotent; cooperative — it sets a flag the task observes at its next
    /// checkpoint, it does **not** interrupt a running step.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Observe the token at a checkpoint: an explicit [`Cancelled`] if cancellation was requested, else
    /// `Ok`. A task calls this where it already checks its budgets — cooperative, never preemptive.
    ///
    /// # Errors
    /// Returns [`Cancelled`] when [`cancel`](CancelToken::cancel) has been called.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

/// A task observed its [`CancelToken`] cancelled — an **explicit, additive** outcome (RFC-0014 I1),
/// never a silent stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("task cancelled (cooperative, RFC-0008 §4.7) — an explicit, additive outcome")
    }
}

impl std::error::Error for Cancelled {}

/// The **explicit, additive result of running a task** (RFC-0014 I1 lifted across the task boundary,
/// RFC-0008 §4.7). A task always resolves to exactly one of these — there is deliberately **no silent /
/// dropped variant**, so cross-task propagation cannot lose a failure:
/// - [`Done`](TaskOutcome::Done) — completed with a value;
/// - [`Failed`](TaskOutcome::Failed) — an explicit error (propagates to the parent scope, additive);
/// - [`BudgetExhausted`](TaskOutcome::BudgetExhausted) — a **per-task** effect-budget overrun (the
///   in-that-task [`EffectBudgetExhausted`], on the same channel as the single-task case — M-353);
/// - [`Cancelled`](TaskOutcome::Cancelled) — cooperatively cancelled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOutcome<T, E> {
    /// Completed with a value.
    Done(T),
    /// Failed with an explicit error — propagates additively to the owning scope (RT4/I1).
    Failed(E),
    /// A per-task effect budget overran — an in-that-task, graceful refusal (M-353; I4).
    BudgetExhausted(EffectBudgetExhausted),
    /// Cooperatively cancelled (RT7).
    Cancelled,
}

impl<T, E> TaskOutcome<T, E> {
    /// Whether this outcome is a **failure** the parent scope must observe (failure, budget overrun, or
    /// cancellation) — i.e. anything other than a clean [`Done`](TaskOutcome::Done). Cross-task
    /// propagation is "the parent acts on this explicitly"; it can never silently treat a failure as
    /// success (I1).
    #[must_use]
    pub fn is_failure(&self) -> bool {
        !matches!(self, TaskOutcome::Done(_))
    }
}

/// **Max-restart-intensity** for `reclaim` supervision (RFC-0008 §4.7; Erlang/OTP, Research Record 05
/// T5.3): at most `max_restarts` within a window of `window_ticks` **logical** ticks.
///
/// v0 uses a **logical clock** — a deterministic, monotonic counter the supervisor advances — not the
/// wall clock; physical/hybrid clocks for real time are RFC-0008 R8-Q3, deferred. This is the *rate*
/// bound; it composes with a `cascade` effect **budget** (the *total* restart cap) so a restart storm
/// is bounded on **both** axes (the combined disposition): exceeding *either* is an explicit
/// [`Escalation`], a **declared, bounded cascade** (RT4/RT7), never an unbounded storm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartIntensity {
    /// The maximum restarts permitted within the window.
    pub max_restarts: u32,
    /// The window length, in logical-clock ticks.
    pub window_ticks: u64,
}

/// A supervisor escalated: a restart cascade hit a bound and the supervisor itself fails (its own
/// explicit outcome — never an unbounded restart storm; RT4/RT7). Carries *why* (rate vs. total).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Escalation {
    /// The windowed [`RestartIntensity`] rate was exceeded.
    IntensityExceeded {
        /// The restarts observed within the window (including the one that tripped it).
        restarts_in_window: u32,
        /// The configured ceiling.
        max_restarts: u32,
        /// The window length (logical ticks).
        window_ticks: u64,
    },
    /// The total restart **cascade budget** was exhausted (the M-353 effect-budget channel).
    CascadeBudgetExhausted(EffectBudgetExhausted),
}

impl fmt::Display for Escalation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Escalation::IntensityExceeded {
                restarts_in_window,
                max_restarts,
                window_ticks,
            } => write!(
                f,
                "supervisor escalated: {restarts_in_window} restarts within {window_ticks} ticks \
                 exceeds the max-restart-intensity {max_restarts} (RFC-0008 §4.7) — a bounded cascade, \
                 not an unbounded storm"
            ),
            Escalation::CascadeBudgetExhausted(e) => {
                write!(f, "supervisor escalated: {e}")
            }
        }
    }
}

impl std::error::Error for Escalation {}

/// A `reclaim` **supervisor** (RFC-0008 §4.7; RT4/RT7): it restarts a failed child under a *bounded*
/// cascade, escalating (its own explicit failure) when the cascade hits **either** bound — the total
/// `cascade` budget (M-353) **or** the windowed [`RestartIntensity`] rate. Both axes are honest and
/// declared; neither lets a restart storm run away.
///
/// The logical clock is the supervisor's own monotonic counter ([`tick`](Supervisor::tick)); a driver
/// advances it deterministically (one tick per supervised step, say). No wall clock is read (R8-Q3).
#[derive(Debug, Clone)]
pub struct Supervisor {
    intensity: RestartIntensity,
    budgets: Budgets,
    /// The logical tick of each restart still inside the current window (oldest at the front).
    restarts: VecDeque<u64>,
    clock: u64,
}

impl Supervisor {
    /// A supervisor with a windowed [`RestartIntensity`] (the *rate* bound) and a total `cascade`
    /// budget of `max_total` restarts (the *cascade* bound). Both must hold for a restart to proceed.
    #[must_use]
    pub fn new(intensity: RestartIntensity, max_total: u64) -> Self {
        Supervisor {
            intensity,
            budgets: Budgets::new().with(crate::budget::EffectBudget::Depth(max_total)),
            restarts: VecDeque::new(),
            clock: 0,
        }
    }

    /// The current logical tick.
    #[must_use]
    pub fn now(&self) -> u64 {
        self.clock
    }

    /// Advance the logical clock by one tick and return the new value. A deterministic, monotonic
    /// counter — *not* wall-clock time (R8-Q3).
    pub fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The total restart budget remaining (the `cascade` cap).
    #[must_use]
    pub fn restarts_remaining(&self) -> u64 {
        self.budgets.remaining(&EffectKind::Cascade).unwrap_or(0)
    }

    /// Record a restart at the current logical tick. Succeeds iff **both** bounds hold; otherwise the
    /// supervisor **escalates** with an explicit [`Escalation`] (never an unbounded storm).
    ///
    /// Enforcement order (both are checked, both are honest): first the **total** `cascade` budget
    /// (M-353 channel) — a [`EffectBudgetExhausted`] becomes [`Escalation::CascadeBudgetExhausted`];
    /// then the **windowed** intensity *rate* — too many restarts within `window_ticks` becomes
    /// [`Escalation::IntensityExceeded`].
    ///
    /// # Errors
    /// Returns [`Escalation`] when the total cascade budget is exhausted or the windowed restart
    /// intensity is exceeded.
    pub fn record_restart(&mut self) -> Result<(), Escalation> {
        // Total cap (the bounded cascade): consume one from the `cascade` budget.
        self.budgets
            .consume(EffectKind::Cascade, 1)
            .map_err(Escalation::CascadeBudgetExhausted)?;

        // Rate cap (the windowed intensity): drop restarts that have aged out of the window, then count.
        let cutoff = self.clock.saturating_sub(self.intensity.window_ticks);
        while self.restarts.front().is_some_and(|&t| t <= cutoff) {
            self.restarts.pop_front();
        }
        self.restarts.push_back(self.clock);
        let in_window = u32::try_from(self.restarts.len()).unwrap_or(u32::MAX);
        if in_window > self.intensity.max_restarts {
            return Err(Escalation::IntensityExceeded {
                restarts_in_window: in_window,
                max_restarts: self.intensity.max_restarts,
                window_ticks: self.intensity.window_ticks,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_cooperative_and_explicit() {
        let tok = CancelToken::new();
        assert!(tok.check().is_ok());
        tok.cancel();
        assert!(tok.is_cancelled());
        assert_eq!(tok.check().unwrap_err(), Cancelled);
        // A shared clone observes the same cancellation (the future runtime shares it across tasks).
        let other = tok.clone();
        assert!(other.is_cancelled());
    }

    #[test]
    fn a_task_outcome_never_silently_succeeds_on_failure() {
        // I1 across the task boundary: every non-Done outcome is a failure the parent must observe.
        let done: TaskOutcome<u8, &str> = TaskOutcome::Done(1);
        assert!(!done.is_failure());
        assert!(TaskOutcome::<u8, &str>::Failed("boom").is_failure());
        assert!(TaskOutcome::<u8, &str>::Cancelled.is_failure());
        assert!(
            TaskOutcome::<u8, &str>::BudgetExhausted(EffectBudgetExhausted {
                kind: EffectKind::Retry,
                requested: 1,
                remaining: 0,
            })
            .is_failure()
        );
    }

    #[test]
    fn supervision_bounds_a_cascade_by_total_count() {
        // The cascade budget caps the TOTAL restarts: 2 allowed, the 3rd escalates (bounded cascade).
        let mut sup = Supervisor::new(
            RestartIntensity {
                max_restarts: 100,
                window_ticks: 1_000,
            },
            2,
        );
        sup.tick();
        assert!(sup.record_restart().is_ok());
        assert!(sup.record_restart().is_ok());
        assert_eq!(sup.restarts_remaining(), 0);
        match sup.record_restart() {
            Err(Escalation::CascadeBudgetExhausted(e)) => assert_eq!(e.kind, EffectKind::Cascade),
            other => panic!("expected a cascade-budget escalation, got {other:?}"),
        }
    }

    #[test]
    fn supervision_bounds_a_cascade_by_windowed_rate() {
        // The windowed intensity caps the RATE: at most 2 restarts within 5 ticks. A 3rd inside the
        // window escalates; once restarts age out of the window, restarting is allowed again.
        let mut sup = Supervisor::new(
            RestartIntensity {
                max_restarts: 2,
                window_ticks: 5,
            },
            1_000, // ample total, so the *rate* bound is what bites
        );
        sup.tick(); // t=1
        assert!(sup.record_restart().is_ok());
        sup.tick(); // t=2
        assert!(sup.record_restart().is_ok());
        sup.tick(); // t=3
        match sup.record_restart() {
            Err(Escalation::IntensityExceeded {
                restarts_in_window,
                max_restarts,
                ..
            }) => {
                assert_eq!(restarts_in_window, 3);
                assert_eq!(max_restarts, 2);
            }
            other => panic!("expected an intensity escalation, got {other:?}"),
        }
        // Advance well past the window so the earlier restarts age out, then a restart is fine again.
        for _ in 0..10 {
            sup.tick();
        }
        assert!(
            sup.record_restart().is_ok(),
            "after the window clears, restarting is bounded-OK again"
        );
    }

    // ---- supervise.rs:79 — Cancelled Display → Ok(Default::default()) ----
    // Mutant: fmt body is a no-op; the formatted string is empty.
    // Kill: assert the Cancelled Display output is non-empty and mentions "cancel".
    #[test]
    fn cancelled_display_is_non_empty_and_descriptive() {
        // Mutant-witness: supervise.rs:79 replace fmt → Ok(Default::default()).
        let msg = Cancelled.to_string();
        assert!(
            !msg.is_empty(),
            "Cancelled Display must not be empty (got empty string)"
        );
        assert!(
            msg.to_lowercase().contains("cancel"),
            "Cancelled Display must mention 'cancel'; got: {msg:?}"
        );
    }

    // ---- supervise.rs:151 — Escalation Display → Ok(Default::default()) ----
    // Mutant: fmt body is a no-op; the formatted string is empty for all variants.
    // Kill: assert the Escalation Display contains variant-specific content.
    #[test]
    fn escalation_display_is_non_empty_and_variant_specific() {
        // Mutant-witness: supervise.rs:151 replace fmt → Ok(Default::default()).
        let e = Escalation::IntensityExceeded {
            restarts_in_window: 5,
            max_restarts: 3,
            window_ticks: 10,
        };
        let msg = e.to_string();
        assert!(
            !msg.is_empty(),
            "Escalation::IntensityExceeded Display must not be empty"
        );
        assert!(
            msg.contains('5') || msg.contains('3'),
            "IntensityExceeded Display must contain the restart counts; got: {msg:?}"
        );

        // CascadeBudgetExhausted variant.
        let cascade_e = Escalation::CascadeBudgetExhausted(EffectBudgetExhausted {
            kind: EffectKind::Cascade,
            requested: 1,
            remaining: 0,
        });
        let cascade_msg = cascade_e.to_string();
        assert!(
            !cascade_msg.is_empty(),
            "Escalation::CascadeBudgetExhausted Display must not be empty"
        );
    }

    // ---- supervise.rs:203 — Supervisor::now → 0 or 1 ----
    // Mutant A (→ 0): now() always returns 0, regardless of actual tick count.
    // Mutant B (→ 1): now() always returns 1.
    // Kill: tick twice and assert now() reflects the actual counter value.
    #[test]
    fn supervisor_now_reflects_actual_tick_count() {
        // Mutant-witness: supervise.rs:203 replace now() → 0 or 1.
        let mut sup = Supervisor::new(
            RestartIntensity {
                max_restarts: 100,
                window_ticks: 1_000,
            },
            1_000,
        );
        assert_eq!(sup.now(), 0, "initial tick is 0");
        sup.tick();
        assert_eq!(
            sup.now(),
            1,
            "after 1 tick, now() must be 1 (not 0 or some constant)"
        );
        sup.tick();
        assert_eq!(sup.now(), 2, "after 2 ticks, now() must be 2 (not 1 or 0)");
        for _ in 0..8 {
            sup.tick();
        }
        assert_eq!(sup.now(), 10, "after 10 ticks total, now() must be 10");
    }

    // ---- supervise.rs:216 — Supervisor::restarts_remaining → 0 ----
    // Mutant: restarts_remaining always returns 0, even when budget remains.
    // Kill: after creating a supervisor with max_total=5 and recording 2 restarts, remaining must be 3.
    #[test]
    fn supervisor_restarts_remaining_decrements_correctly() {
        // Mutant-witness: supervise.rs:216 replace restarts_remaining → 0.
        let mut sup = Supervisor::new(
            RestartIntensity {
                max_restarts: 100,
                window_ticks: 1_000,
            },
            5, // max_total = 5
        );
        assert_eq!(
            sup.restarts_remaining(),
            5,
            "before any restarts, remaining must be 5"
        );
        sup.tick();
        sup.record_restart().unwrap();
        assert_eq!(
            sup.restarts_remaining(),
            4,
            "after 1 restart, remaining must be 4 (not 0)"
        );
        sup.tick();
        sup.record_restart().unwrap();
        assert_eq!(
            sup.restarts_remaining(),
            3,
            "after 2 restarts, remaining must be 3 (not 0)"
        );
    }
}
