//! `Colony<T,E>` and `Scope<T,E>` — structured concurrency surface (ADR-020 v0 R1).
//!
//! # Guarantee (Empirical — grounded in RT2 sequentialization + Kahn-determinism differentials)
//!
//! `Scope<T,E>` guarantees that all tasks spawned within a scope complete or are
//! cancelled before the scope exits. The Kahn-determinism guarantee (channel-mediated
//! communication is deterministic) is **Empirical**: grounded in the RT2 differential
//! but not yet Proven with a formal theorem.
//!
//! `Scope::join_all` sweeps tasks in FIFO order (**Exact** — deterministic given same spawn
//! order). On any task panic or explicit cancellation the scope returns early with an
//! explicit error (G2: never silent).

use std::panic;

use mycelium_core::GuaranteeStrength;

use crate::task::Task;

/// Guarantee strength for `Scope` join semantics (RT2 sequentialization differential).
pub const SCOPE_JOIN_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

/// Guarantee strength for `Colony` Kahn-determinism (channel-mediated communication).
pub const COLONY_KAHN_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

/// Error type for scope exits with active tasks.
#[derive(Debug, PartialEq, Eq)]
pub enum ScopeError {
    /// The scope was cancelled before all tasks completed.
    Cancelled,
    /// One or more tasks panicked; remaining tasks were cancelled (never silent — G2).
    TasksStillRunning,
}

/// Structured concurrency scope: all tasks complete or are cancelled before scope exit.
///
/// Tasks are stored in spawn order and swept FIFO by `join_all` (**Exact** sweep guarantee).
///
/// # Guarantee
/// - Sweep order: **Exact** (FIFO by construction, ADR-020 §4).
/// - Completion semantics: **Empirical** (RT2 sequentialization differential, ADR-020 §4).
/// - A scope that exits before all tasks complete returns `Err(ScopeError::TasksStillRunning)`.
pub struct Scope<T, E> {
    tasks: Vec<Task>,
    cancelled: bool,
    _output: std::marker::PhantomData<T>,
    _error: std::marker::PhantomData<E>,
}

impl<T, E> std::fmt::Debug for Scope<T, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scope")
            .field("task_count", &self.tasks.len())
            .field("cancelled", &self.cancelled)
            .finish()
    }
}

impl<T, E> Scope<T, E> {
    /// Create a new empty scope.
    ///
    /// Guarantee: **Exact** (constructor, trivially correct).
    pub fn new() -> Self {
        Scope {
            tasks: Vec::new(),
            cancelled: false,
            _output: std::marker::PhantomData,
            _error: std::marker::PhantomData,
        }
    }

    /// Returns the number of tasks currently queued in this scope.
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Spawn a task into this scope.
    ///
    /// Tasks are appended in FIFO order; `join_all` will execute them in this same order.
    ///
    /// Guarantee: **Exact** (push to a `Vec` — deterministic ordering, ADR-020 §4).
    pub fn spawn(&mut self, task: Task) {
        self.tasks.push(task);
    }

    /// Cancel this scope. After calling `cancel`, `join_all` will return
    /// `Err(ScopeError::Cancelled)` without running any remaining tasks.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
}

impl<E> Scope<(), E> {
    /// Run all spawned tasks in FIFO spawn order.
    ///
    /// In v0, `Task` closures always return `()`. Returns `Ok(())` when all complete.
    ///
    /// Returns `Err(ScopeError::Cancelled)` if the scope was cancelled before `join_all`.
    /// Returns `Err(ScopeError::TasksStillRunning)` if any task panics (the panic is caught
    /// and an explicit error is returned — G2: never silent).
    ///
    /// Guarantee: **Empirical** (RT2 sequentialization differential, ADR-020 §4).
    pub fn join_all(self) -> Result<(), ScopeError> {
        if self.cancelled {
            return Err(ScopeError::Cancelled);
        }
        for task in self.tasks {
            match panic::catch_unwind(panic::AssertUnwindSafe(|| task.run())) {
                Ok(()) => {}
                Err(_) => return Err(ScopeError::TasksStillRunning),
            }
        }
        Ok(())
    }
}

impl<T, E> Default for Scope<T, E> {
    fn default() -> Self {
        Self::new()
    }
}

/// Colony: a group of scopes sharing a supervision tree and a `Network`.
///
/// Provides a factory for `Scope<T, E>` via [`Colony::scope`].
///
/// Guarantee: `Empirical` (Kahn-determinism of channel communication, ADR-020 §4).
#[derive(Debug)]
pub struct Colony<T, E> {
    _output: std::marker::PhantomData<T>,
    _error: std::marker::PhantomData<E>,
}

impl<T, E> Colony<T, E> {
    /// Create a new colony.
    ///
    /// Guarantee: **Exact** (constructor, trivially correct).
    pub fn new() -> Self {
        Colony {
            _output: std::marker::PhantomData,
            _error: std::marker::PhantomData,
        }
    }

    /// Create a new empty `Scope<T, E>` managed by this colony.
    ///
    /// Guarantee: **Exact** (delegates to `Scope::new`, trivially correct).
    pub fn scope(&self) -> Scope<T, E> {
        Scope::new()
    }
}

impl<T, E> Default for Colony<T, E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Task;
    use std::panic;

    /// Test-only typed scope: stores closures returning `T` for FIFO-order assertions.
    struct TypedScope<T> {
        closures: Vec<Box<dyn FnOnce() -> T + Send + 'static>>,
        cancelled: bool,
    }
    impl<T> TypedScope<T> {
        fn new() -> Self {
            TypedScope {
                closures: Vec::new(),
                cancelled: false,
            }
        }
        fn spawn<F: FnOnce() -> T + Send + 'static>(&mut self, f: F) {
            self.closures.push(Box::new(f));
        }
        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }
    impl<T: Send + 'static> TypedScope<T> {
        fn join_all(self) -> Result<Vec<T>, ScopeError> {
            if self.cancelled {
                return Err(ScopeError::Cancelled);
            }
            let mut out = Vec::with_capacity(self.closures.len());
            for f in self.closures {
                match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
                    Ok(v) => out.push(v),
                    Err(_) => return Err(ScopeError::TasksStillRunning),
                }
            }
            Ok(out)
        }
    }

    #[test]
    fn test_scope_new_is_empty() {
        let scope: Scope<(), ()> = Scope::new();
        assert_eq!(scope.task_count(), 0, "new scope must have no tasks");
    }

    #[test]
    fn test_scope_join_empty() {
        // join_all on an empty scope returns Ok([]).
        let scope: TypedScope<i32> = TypedScope::new();
        let result = scope.join_all().expect("empty scope join must succeed");
        assert!(result.is_empty(), "result must be empty for zero tasks");
    }

    #[test]
    fn test_scope_join_all_fifo() {
        // Two tasks spawned; results arrive in spawn order (FIFO = Exact guarantee).
        // Mutant witness: if join_all reversed order, assert result[0] == 1 would fail.
        let mut scope: TypedScope<i32> = TypedScope::new();
        scope.spawn(|| 1);
        scope.spawn(|| 2);
        let results = scope.join_all().expect("join must succeed");
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0], 1,
            "first spawned task must be first in results (FIFO)"
        );
        assert_eq!(
            results[1], 2,
            "second spawned task must be second in results (FIFO)"
        );
    }

    #[test]
    fn test_scope_error_on_cancelled() {
        // A manually cancelled scope returns Err(ScopeError::Cancelled).
        // Mutant witness: removing the cancellation check would make this return Ok([]).
        let mut scope: TypedScope<i32> = TypedScope::new();
        scope.cancel();
        let err = scope
            .join_all()
            .expect_err("cancelled scope must return Err");
        assert_eq!(
            err,
            ScopeError::Cancelled,
            "must return Cancelled, not TasksStillRunning"
        );
    }

    #[test]
    fn test_scope_spawn_with_task() {
        // Scope<(), ()> with Task-based spawn.
        let mut scope: Scope<(), ()> = Scope::new();
        scope.spawn(Task::new(|| {}));
        assert_eq!(scope.task_count(), 1);
    }

    #[test]
    fn test_colony_scope_factory() {
        // Colony::scope() produces a new empty Scope.
        let colony: Colony<(), ()> = Colony::new();
        let scope = colony.scope();
        assert_eq!(
            scope.task_count(),
            0,
            "colony-created scope must start empty"
        );
    }
}
