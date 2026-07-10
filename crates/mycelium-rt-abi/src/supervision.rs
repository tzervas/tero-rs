//! Structured-concurrency supervision + cancellation (M-713 / RFC-0008 RT4·RT7 / E12-1).
//!
//! RFC-0008 RT7: a scope does not exit until every child completes or is cancelled — "an orphan
//! hypha is not expressible". The scheduler-independent composition kernel (M-356) already provides
//! [`CancelToken`], the explicit [`TaskOutcome`], and the bounded-cascade [`Supervisor`]; this
//! module makes them **execute end-to-end on the OS-thread pool** (M-709):
//!
//! - [`CancelTree`] — a cancellation **tree**: cancelling a node cascades to every descendant
//!   (parent→child, never child→parent), so a cancelled colony propagates to all its children (RT7).
//! - [`run_supervised`] — runs a task set on the [`Scheduler`], collects **every** child's explicit
//!   [`TaskOutcome`] (never a dropped/silent variant — RT4/I1), and on the first failure cancels the
//!   remaining siblings (never-silent propagation, G2).
//! - [`supervise_with_restart`] — a live restart policy ([`Supervisor`] bounds) that is
//!   **EXPLAIN-able**: each decision is a reified [`SupervisionRecord`] (no black boxes; ADR-006).
//!
//! # Honesty (VR-5)
//!
//! - Cancellation propagation + outcome collection are **`Empirical`** (property-tested; cooperative
//!   observation, not preemption — a task sees cancellation at its own checkpoints).
//! - The restart bound (rate + total cascade) is **`Exact`** — inherited from [`Supervisor`], whose
//!   bounds are enforced structurally (M-356).

pub use mycelium_interp::{
    CancelToken, Cancelled, Escalation, RestartIntensity, Supervisor, TaskOutcome,
};

use mycelium_core::GuaranteeStrength;

use mycelium_sched::scheduler::Scheduler;

/// Guarantee strength for cancellation propagation + explicit outcome collection.
pub const SUPERVISION_PROPAGATION_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

/// Guarantee strength for the restart bound (inherited from the M-356 [`Supervisor`]).
pub const SUPERVISION_RESTART_BOUND_STRENGTH: GuaranteeStrength = GuaranteeStrength::Exact;

/// A cancellation **tree** (RFC-0008 RT7): a node with its own [`CancelToken`] and child tokens.
///
/// Cancelling a node cascades to **every descendant** (parent→child), so cancelling a colony
/// propagates failure to all its children — never-silent (G2). Cancellation never flows the other
/// way: a child cancel leaves the parent live (structured-concurrency direction).
///
/// Deliberately **not `Clone`**: cloning would deep-copy the child subtree, so attaching a child to
/// one clone after the split would not cascade from the other — silently violating the
/// "cancels every descendant" contract. Share a node's cancellation via [`token`](CancelTree::token)
/// (the [`CancelToken`] *is* `Clone`, sharing one flag) instead of cloning the tree.
#[derive(Debug, Default)]
pub struct CancelTree {
    token: CancelToken,
    children: Vec<CancelTree>,
}

impl CancelTree {
    /// A fresh, un-cancelled root.
    #[must_use]
    pub fn new() -> Self {
        CancelTree {
            token: CancelToken::new(),
            children: Vec::new(),
        }
    }

    /// This node's cooperative cancel token (clones share the same flag).
    #[must_use]
    pub fn token(&self) -> CancelToken {
        self.token.clone()
    }

    /// Attach a fresh child scope and return a mutable handle to it, so callers can build a genuine
    /// multi-level tree (attach grandchildren via the returned node's own [`child`](CancelTree::child)).
    /// The child — and everything attached under it — is cancelled if **this** node is later
    /// cancelled (the cascade), but cancelling the child does not cancel this node (RT7). Use
    /// [`token`](CancelTree::token) on the returned handle to get its cooperative cancel token.
    pub fn child(&mut self) -> &mut CancelTree {
        self.children.push(CancelTree::new());
        self.children
            .last_mut()
            .expect("a child was just pushed, so last_mut is Some")
    }

    /// Whether this node has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Cancel this node **and every descendant** — the never-silent cascade (G2/RT7). Idempotent.
    pub fn cancel(&self) {
        self.token.cancel();
        for c in &self.children {
            c.cancel();
        }
    }

    /// The number of direct child scopes (for inspection/tests).
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

/// Run `tasks` on the OS-thread pool under a shared [`CancelToken`], collecting **every** child's
/// explicit [`TaskOutcome`] in spawn order (RT4/I1 — no outcome is ever silently dropped).
///
/// Each task is a closure `FnOnce(&CancelToken) -> TaskOutcome<T, E>`: it observes the token
/// cooperatively (at its own checkpoints). The **never-silent failure-propagation** contract: if a
/// task returns a failure outcome ([`TaskOutcome::is_failure`]), it cancels the shared token, so
/// still-running siblings that next check the token resolve to [`TaskOutcome::Cancelled`] — a
/// cancelled scope never silently leaks a sibling (G2/RT7). Every task's outcome is still reported.
///
/// **M-864 note:** `Scheduler::run_indexed` now requires `'static` job closures, so `token` can no
/// longer be *borrowed* from the caller's stack frame the way the pre-M-864 `thread::scope`-backed
/// scheduler allowed — each job instead takes its own [`CancelToken::clone`]. [`CancelToken`] is a
/// thin, `Clone`-able handle onto one shared flag (see its own docs), so every clone still observes
/// and cancels the *same* underlying cancellation state — the sharing semantics this function's
/// contract depends on are unchanged, only the ownership shape is.
///
/// Guarantee: **`Empirical`** ([`SUPERVISION_PROPAGATION_STRENGTH`]).
#[must_use]
pub fn run_supervised<T, E, F>(
    scheduler: &Scheduler,
    token: &CancelToken,
    tasks: Vec<F>,
) -> Vec<TaskOutcome<T, E>>
where
    F: FnOnce(&CancelToken) -> TaskOutcome<T, E> + Send + 'static,
    T: Send + 'static,
    E: Send + 'static,
{
    let jobs: Vec<_> = tasks
        .into_iter()
        .map(|task| {
            let token = token.clone();
            move || {
                let outcome = task(&token);
                if outcome.is_failure() {
                    // Never-silent propagation: a failure cancels the scope so siblings observe it
                    // at their next checkpoint and resolve to Cancelled (RT7/G2), never leak. `token`
                    // is this job's own clone, but `cancel` flips the one shared flag every clone
                    // reads (see the doc above), so the propagation is identical to before M-864.
                    token.cancel();
                }
                outcome
            }
        })
        .collect();
    scheduler.run_indexed(jobs, None, None)
}

/// What a supervisor did about one child failure — a reified, inspectable action (EXPLAIN; no black
/// boxes — ADR-006).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisionAction {
    /// The child was restarted (under the bounded cascade).
    Restarted,
    /// The restart cascade hit a bound; the supervisor escalated (its own explicit failure).
    Escalated(Escalation),
}

/// One reified supervision decision — the EXPLAIN record (RFC-0008 §4.7; ADR-006: selections are
/// inspectable, never silent). A driver emits one of these per child failure it handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisionRecord {
    /// The supervisor's logical tick at which the decision was made (RestartIntensity clock).
    pub logical_tick: u64,
    /// Restarts already consumed from the total `cascade` budget before this decision.
    pub restarts_before: u64,
    /// What the supervisor did.
    pub action: SupervisionAction,
}

/// The result of supervising a restartable child to resolution or escalation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisedRun<T> {
    /// The final outcome: `Ok(value)` if the child eventually succeeded, `Err` if the supervisor
    /// escalated (a bounded cascade hit a bound — an explicit failure, never an unbounded storm).
    pub result: Result<T, SupervisedFailure>,
    /// The EXPLAIN trace: every restart/escalation decision, in order (no black boxes; ADR-006).
    pub trace: Vec<SupervisionRecord>,
}

/// Why a supervised child run ended in failure — always explicit (G2).
///
/// The child's *transient* per-attempt errors are not surfaced here: each one is **handled** by a
/// restart and recorded in the EXPLAIN trace as a [`SupervisionAction::Restarted`] decision (so no
/// failure is silently dropped — RT4/I1), and the terminal outcome is exactly one of these two
/// explicit cases. (The supervisor's contract is "restart on failure, escalate when the bound is
/// hit"; a non-restartable error is not a case `supervise_with_restart` itself produces.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisedFailure {
    /// The supervisor escalated: the restart cascade hit a bound (rate or total).
    Escalated(Escalation),
    /// The child was cancelled (cooperative; RT7).
    Cancelled,
}

/// Run a restartable child under a live [`Supervisor`] (M-356) until it succeeds or the supervisor
/// escalates, recording an EXPLAIN [`SupervisionRecord`] for each decision (RFC-0008 RT7/RT4).
///
/// `attempt` runs the child once and returns its [`TaskOutcome`]. On a failure outcome the
/// supervisor tries to restart (consuming the bounded cascade); on success it returns the value with
/// the full trace; on a bound hit it escalates explicitly (never an unbounded restart storm). The
/// child's `E` is the per-attempt error type; transient failures are *handled* by restart (traced),
/// so the terminal [`SupervisedRun`] does not itself carry an `E`.
///
/// Guarantee: the restart **bound** is **`Exact`** ([`SUPERVISION_RESTART_BOUND_STRENGTH`], inherited
/// from [`Supervisor`]); the trace is exact (every decision is recorded).
pub fn supervise_with_restart<T, E>(
    supervisor: &mut Supervisor,
    mut attempt: impl FnMut() -> TaskOutcome<T, E>,
) -> SupervisedRun<T> {
    let mut trace = Vec::new();
    let mut restarts_before = 0u64; // honest local count of restarts already granted this run
    loop {
        match attempt() {
            TaskOutcome::Done(v) => {
                return SupervisedRun {
                    result: Ok(v),
                    trace,
                };
            }
            TaskOutcome::Cancelled => {
                return SupervisedRun {
                    result: Err(SupervisedFailure::Cancelled),
                    trace,
                };
            }
            TaskOutcome::Failed(_) | TaskOutcome::BudgetExhausted(_) => {
                // A failure: try to restart under the bounded cascade. Record the decision (EXPLAIN).
                let tick = supervisor.tick();
                match supervisor.record_restart() {
                    Ok(()) => {
                        trace.push(SupervisionRecord {
                            logical_tick: tick,
                            restarts_before,
                            action: SupervisionAction::Restarted,
                        });
                        restarts_before += 1;
                        // Loop: re-attempt the child (the restart).
                    }
                    Err(escalation) => {
                        trace.push(SupervisionRecord {
                            logical_tick: tick,
                            restarts_before,
                            action: SupervisionAction::Escalated(escalation.clone()),
                        });
                        return SupervisedRun {
                            result: Err(SupervisedFailure::Escalated(escalation)),
                            trace,
                        };
                    }
                }
            }
        }
    }
}
