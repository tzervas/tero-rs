//! Task, TaskCtx, Poll, SweepOrder, Deadlock — task surface (ADR-020 v0 R1).
//!
//! # Guarantee (Declared — Task purity contract)
//!
//! `Task` purity is **Declared**: the type system cannot enforce that a task body
//! has no side effects, so this is an assertion-level guarantee (VR-5: not upgraded
//! to Empirical/Proven without a checked basis).

use mycelium_core::GuaranteeStrength;

/// Guarantee strength for the `Task` purity contract.
pub const TASK_PURITY_STRENGTH: GuaranteeStrength = GuaranteeStrength::Declared;

/// A computation that can be spawned into a `Scope`.
///
/// Holds a `Box<dyn FnOnce() + Send>` closure. The caller asserts purity
/// (RT1 contract, `Declared` — the type system cannot enforce no-side-effects).
///
/// Guarantee: **Declared** — purity contract is asserted, not enforced by the type system.
/// Out-of-scope effects are a `wild`-level concern (ADR-014).
pub struct Task {
    inner: Box<dyn FnOnce() + Send + 'static>,
}

impl Task {
    /// Construct a task from a closure. The caller asserts purity (Declared).
    pub fn new<F: FnOnce() + Send + 'static>(f: F) -> Self {
        Task { inner: Box::new(f) }
    }

    /// Run the task closure exactly once.
    ///
    /// Guarantee: **Declared** (inherited from the Task purity contract).
    pub fn run(self) {
        (self.inner)();
    }
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task").finish_non_exhaustive()
    }
}

/// Context passed to a running task — carries cancellation signal and scope ref.
#[derive(Debug)]
pub struct TaskCtx {
    cancelled: bool,
}

impl TaskCtx {
    pub fn new() -> Self {
        TaskCtx { cancelled: false }
    }

    /// Returns `true` if this task's scope has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Cancel this context (used by `Scope` when cancellation is requested).
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
}

impl Default for TaskCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Poll result for an async task step.
#[derive(Debug, PartialEq, Eq)]
pub enum Poll<T> {
    Ready(T),
    Pending,
}

/// Order in which tasks are swept from a scope's run queue.
///
/// Guarantee: **Exact** — the sweep order is deterministic given the same queue state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SweepOrder {
    /// FIFO (default): tasks are completed in the order they were spawned.
    #[default]
    Fifo,
    /// Priority: tasks are swept highest-priority first.
    Priority,
}

/// Deadlock descriptor: returned when a scope cannot make progress.
///
/// Guarantee: **Empirical** — detection is complete for the supported channel graph
/// (DAG channels); cyclic graphs are an open follow-up (FLAG: ADR-020 §7).
#[derive(Debug, PartialEq, Eq)]
pub struct Deadlock {
    pub task_count: usize,
}

impl Deadlock {
    pub fn new(task_count: usize) -> Self {
        Deadlock { task_count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_task_run_executes_closure() {
        // Mutant witness: if Task::run did not call the closure, the flag would stay false.
        let flag = Arc::new(Mutex::new(false));
        let flag_clone = Arc::clone(&flag);
        let task = Task::new(move || {
            *flag_clone.lock().unwrap() = true;
        });
        task.run();
        assert!(*flag.lock().unwrap(), "Task::run must execute the closure");
    }

    #[test]
    fn test_taskctx_not_cancelled_by_default() {
        let ctx = TaskCtx::new();
        assert!(
            !ctx.is_cancelled(),
            "a freshly constructed TaskCtx must not be cancelled"
        );
    }

    #[test]
    fn test_poll_ready() {
        let poll = Poll::Ready(42i32);
        assert_eq!(poll, Poll::Ready(42), "Poll::Ready must contain the value");
    }

    #[test]
    fn test_deadlock_has_count() {
        let dl = Deadlock::new(3);
        assert_eq!(dl.task_count, 3, "Deadlock must record the task count");
    }

    #[test]
    fn test_sweep_order_default_is_fifo() {
        // Mutant witness: if default changed to Priority, this assertion fails.
        assert_eq!(
            SweepOrder::default(),
            SweepOrder::Fifo,
            "SweepOrder default must be Fifo (ADR-020 §4)"
        );
    }
}
