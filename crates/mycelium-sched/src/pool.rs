//! The persistent, bounded, **help-stealing** OS-thread pool underlying
//! [`crate::scheduler::Scheduler::run_indexed`] (M-864).
//!
//! # Why this exists
//!
//! Before M-864, `run_indexed` used [`std::thread::scope`], spawning **fresh OS threads on every
//! call**. That is deadlock-safe under nesting (a nested `run_indexed` just spawns its own fresh
//! scope) but costs `O(depth × fan-out)` threads for recursive submission — a resource blowup that
//! forced M-860 (parallel AOT codegen) and M-862 (parallel interpreter eval) to bound their own
//! parallelism to a single, non-nested, top-level batch (see their module docs).
//!
//! This module is the fix: **one process-wide pool of `available_parallelism()` OS threads, created
//! once, lazily, and reused forever** ([`get`]). No `run_indexed` call — however deeply nested —
//! ever spawns a new OS thread. Nested submission instead grows the number of **helpers** (see
//! below), never the number of **threads**.
//!
//! # The help-stealing pattern (the hard part)
//!
//! A persistent, *bounded* pool creates an obvious hazard: if a worker thread submits a nested batch
//! and then just *blocks* waiting for it, and every other pool thread is doing the same thing
//! (waiting on *its own* nested batch), the whole pool can wedge with no thread actually running
//! anything — classic thread-pool deadlock (the "everyone is waiting for someone else to make
//! progress, but nobody is free to do it" hazard).
//!
//! [`Pool::help_while`] is the fix, straight out of the Cilk/TBB/Rayon playbook: a thread that would
//! otherwise block instead **executes any pending task from the shared queue** — its own batch's, or
//! anyone else's, at any nesting depth — until its own wait condition is satisfied. A "blocked"
//! thread is therefore never idle-and-unproductive; it is an *additional* worker for as long as it
//! waits. This is what keeps a **fixed**-size pool **deadlock-free** at any nesting depth: the
//! resource that scales with nesting is *helpers-that-happen-to-be-waiting*, not OS threads.
//!
//! **Bounded-progress, NOT bounded-stack (M-864 — never-silent, VR-5).** The induction below proves
//! logical *progress* (no deadlock), but **not** a bounded call *stack*. Because `help_while` pops
//! the shared queue **indiscriminately** (any batch's lane-loop, not only tasks descending from the
//! waiter's own subtree), a nested pop → nested `help_while` stacks a call frame, and under a
//! **deep-AND-wide** fully-fanned tree at low `P` a single OS thread can accumulate help-steal frames
//! from many sibling/cousin batches — worst case ~`O(w^(d-1))` frames for width `w`, depth `d`. That
//! is a **stack overflow (a crash), not a hang**. So this pool is deadlock-free / panic-safe /
//! deterministic at *any* depth, but only *stack*-safe for **moderate** depth×width (measured region:
//! see `deep_and_wide_low_p_completes_within_a_normal_stack_moderate_region` in [`crate::tests`] and
//! DN-67 §3.4). Current consumers (M-860/M-862) submit a **single, non-nested** batch, so they are
//! trivially inside the safe region. The `O(depth)`-stack fix — Cilk-style **leapfrogging**, where
//! `help_while` runs only tasks descending from its own batch — is the tracked follow-up **M-868**.
//!
//! ## Deadlock-freedom argument (`Empirical` — checked by [`crate::tests`]'s nested-recursion
//! stress tests, not mechanically proven)
//!
//! Claim: for any finite tree of (possibly deeply nested) `run_indexed` calls, where every
//! submitted job terminates in finite time (the pre-existing "pure task" contract — see
//! [`crate::scheduler`] module docs), the whole tree completes in finite time, for **any** pool size
//! `P ≥ 1`.
//!
//! Sketch: model the live system as a forest of "batches" (one per in-flight `run_indexed` call),
//! each with a finite, positive number of outstanding lane-loop tasks. A *helper* is any thread
//! (one of the `P` persistent workers, or any external/ancestor caller currently inside
//! [`Pool::help_while`]) not currently running a task. Every helper is, by construction, actively
//! trying to pop and run a pending task ([`Pool::help_while`]'s loop body) — never *just* parked
//! without also checking the shared queue. So as long as the shared queue is non-empty, *some*
//! helper (there is always at least the `P ≥ 1` persistent workers, plus the original caller, which
//! is never itself one of the `P` and so is always a *net addition*) will dequeue and run a task in
//! bounded time (the queue is `Mutex`-guarded FIFO; no task is ever skipped or starved indefinitely
//! by construction — see `submit`/`blocking_pop`). Running a task either (a) completes a lane-loop
//! permanently (decrementing its batch's `remaining` counter — see
//! [`crate::scheduler::Scheduler::run_indexed`]), or (b) runs a user job, which (by the pure-task
//! contract) itself terminates in finite time, *possibly* after submitting and waiting
//! (help-shell recursion) on a **strictly nested, strictly smaller** sub-batch. Because the nesting
//! is a finite call tree (finite program, finite fuel per RFC-0007 §4.5's cooperative-stepping
//! budget bounds every task's own runtime), induction on tree depth gives: every leaf batch (no
//! further nesting) completes because its lane-loops are plain jobs with no further waiting: they
//! terminate outright once popped. Given every leaf batch completes, every batch one level up
//! completes (its lane-loops' jobs each wait on leaf batches, which complete), and so on to the
//! root. No step in this induction assumes `P` is large enough for any particular *concurrency*
//! width — only that `P ≥ 1` so the queue is never permanently unattended. Hence: **no deadlock,
//! for any fixed `P ≥ 1`, at any nesting depth.** (This is a *progress* result only — it says
//! nothing about the call-stack depth, which is the separate bounded-stack caveat above.)
//!
//! This is the informal (but load-bearing) argument; [`crate::tests`] checks it empirically with
//! deep-chain and wide-fan-out nested stress tests, incl. **forced-low-worker-count** ones
//! (`P ∈ {1,2,3,4}`) that hang on the pre-fix code and pass here, under a wall-clock timeout (a real
//! hang would time the test out rather than the assertion merely failing) — `Empirical`, not
//! `Proven` (no mechanized model-checked proof is in-repo), per VR-5.
//!
//! # Determinism is untouched
//!
//! This module knows nothing about job *order* — it is a bag of `'static` closures. Spawn-order
//! result determinism (RT2) is entirely [`crate::scheduler::Scheduler::run_indexed`]'s
//! responsibility (indexed writes into a pre-sized results slot, exactly as before M-864); this
//! module only changes **how many OS threads execute the work and how they're recruited**, never
//! **which job produces which output slot**.
//!
//! # `#![forbid(unsafe_code)]` — unchanged
//! The whole pool is built from `Arc`/`Mutex`/`Condvar`/`VecDeque`/`thread::spawn` — ordinary safe
//! std, zero `unsafe`, zero external dependencies (mirrors the crate-level M-861 rationale: a
//! lock-free Chase-Lev deque would need `unsafe` or an external crate; both stay out of scope).

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

/// One unit of work submitted to the shared pool queue: a boxed, type-erased, `'static` closure.
/// Type erasure (rather than threading a generic through the pool) is what lets one shared queue
/// hold heterogeneous job kinds — batch lane-loops from any number of concurrently active
/// `run_indexed` calls, at any nesting depth, from any caller — behind a single `Mutex`/`Condvar`.
type PoolTask = Box<dyn FnOnce() + Send + 'static>;

/// The bounded pool poll interval [`Pool::help_while`] falls back to when the shared queue is
/// momentarily empty but its own `done` condition isn't satisfied yet. This is a defensive bound
/// against a missed-wakeup race between checking `done` and re-parking on the queue's condvar (the
/// queue's condvar and a batch's own completion condvar are deliberately different, so a `submit`
/// alone does not guarantee an immediate wake for a helper waiting on completion) — never a
/// correctness dependency, only a worst-case added latency (`Exact` bound: a helper can be stalled
/// no HELP_POLL_INTERVAL` interval).
const HELP_POLL_INTERVAL: Duration = Duration::from_micros(200);

/// The persistent, process-wide, bounded work-stealing pool (M-864).
///
/// See the module docs for the help-stealing deadlock-freedom argument.
pub(crate) struct Pool {
    queue: Mutex<VecDeque<PoolTask>>,
    cv: Condvar,
}

impl Pool {
    /// Spawn exactly `workers` persistent OS threads, each running [`Pool::worker_loop`] forever.
    /// These threads are **never joined** — the pool is a process-lifetime singleton (see [`get`]);
    /// this mirrors the standard "global thread pool" pattern (e.g. Rayon's default pool) where
    /// worker threads simply end when the process exits.
    fn new(workers: usize) -> Arc<Self> {
        let pool = Arc::new(Pool {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        });
        for n in 0..workers {
            let pool = Arc::clone(&pool);
            thread::Builder::new()
                .name(format!("mycelium-sched-pool-{n}"))
                .spawn(move || pool.worker_loop())
                .expect("mycelium-sched: spawning a persistent pool worker thread must succeed");
        }
        pool
    }

    /// A **fresh, non-global** pool with exactly `workers` OS threads — **test-only**. The forced,
    /// small worker count is the only way to reproduce the nested-submission deadlock the M-864
    /// rewrite fixes on a machine with many cores (the global [`get`] pool is sized to
    /// `available_parallelism()`, so a wide-fan-out shape that hangs at `P = 4` completes fine at
    /// `P = 28`). Paired with [`crate::scheduler::Scheduler::run_indexed_on`], which lets a batch and
    /// its nested sub-batches all run on this same forced-count pool.
    #[cfg(test)]
    pub(crate) fn with_workers_for_test(workers: usize) -> Arc<Self> {
        Pool::new(workers)
    }

    /// A persistent worker thread's body: forever pop-and-run, parking on `cv` when the queue is
    /// empty. Never returns (the pool's threads live for the process's duration).
    ///
    /// Each task runs under [`std::panic::catch_unwind`] so a panicking task can **never** kill this
    /// persistent worker (M-864 Defect-2 safety: a dead worker would shrink the fixed pool toward
    /// zero with no replenishment). Job-panic *propagation* to the batch caller is handled one level
    /// up, in the lane-loop ([`crate::scheduler::Scheduler::run_indexed_on`]); this is the last-line
    /// guard against any *other* unwind (e.g. a poisoned lock) reaching the worker's stack.
    fn worker_loop(self: Arc<Self>) {
        loop {
            let task = self.blocking_pop();
            run_caught(task);
        }
    }

    /// Block until a task is available, then remove and return it.
    fn blocking_pop(&self) -> PoolTask {
        let mut q = self
            .queue
            .lock()
            .expect("mycelium-sched: pool queue mutex poisoned");
        loop {
            if let Some(task) = q.pop_front() {
                return task;
            }
            q = self
                .cv
                .wait(q)
                .expect("mycelium-sched: pool queue mutex poisoned");
        }
    }

    /// Submit `task` to the shared queue and wake one waiter — a persistent worker's
    /// [`Pool::blocking_pop`], or a helping caller's [`Pool::help_while`].
    pub(crate) fn submit(&self, task: PoolTask) {
        let mut q = self
            .queue
            .lock()
            .expect("mycelium-sched: pool queue mutex poisoned");
        q.push_back(task);
        self.cv.notify_one();
    }

    /// Wake every waiter parked on the pool's queue condvar. Called when a batch's lane-loop
    /// finishes, so a helper parked in [`Pool::help_while`] with nothing to steal re-checks its
    /// `done` condition immediately rather than waiting out [`HELP_POLL_INTERVAL`] — a latency
    /// optimization, never a correctness dependency (the poll bound alone is already sufficient for
    /// liveness).
    pub(crate) fn notify_all(&self) {
        // Acquire the lock first so this cannot race a `wait`/`wait_timeout` call that has not yet
        // re-parked (the standard condvar-notify discipline: notifications are not queued, so a
        // notify with no one yet waiting is a no-op — acquiring first ensures we are not notifying
        // into a gap right as a waiter is about to park).
        let _q = self
            .queue
            .lock()
            .expect("mycelium-sched: pool queue mutex poisoned");
        self.cv.notify_all();
    }

    /// **The nested-join wait loop (M-864's help-steal pattern).** Instead of blocking, run pending
    /// tasks from the shared queue — from this caller's own batch, or from any other batch at any
    /// nesting depth — until `done()` reports the caller's own wait condition is satisfied.
    ///
    /// Called by:
    /// - the thread that invoked [`crate::scheduler::Scheduler::run_indexed`] (which may be an
    ///   ordinary external caller, or itself a persistent pool worker executing a job that
    ///   recursively calls `run_indexed` — a **nested** submission);
    /// - transitively, therefore, at any nesting depth, with no additional OS threads ever spawned.
    ///
    /// See the module docs for the deadlock-freedom argument.
    pub(crate) fn help_while(&self, mut done: impl FnMut() -> bool) {
        loop {
            if done() {
                return;
            }
            let mut q = self
                .queue
                .lock()
                .expect("mycelium-sched: pool queue mutex poisoned");
            match q.pop_front() {
                Some(task) => {
                    drop(q);
                    // Catch, same as `worker_loop`: a helping caller must not unwind out of
                    // `help_while` because some *other* batch's task panicked. Job-panic propagation
                    // to the OWNING batch's caller is handled in that batch's own lane-loop, not here.
                    run_caught(task);
                }
                None => {
                    // Nothing to help with right now. Park briefly — woken promptly by a fresh
                    // `submit`/`notify_all` in the common case, with `HELP_POLL_INTERVAL` as a
                    // defensive re-check bound (never a correctness dependency; see its doc).
                    let (guard, _timeout) = self
                        .cv
                        .wait_timeout(q, HELP_POLL_INTERVAL)
                        .expect("mycelium-sched: pool queue mutex poisoned");
                    drop(guard);
                }
            }
        }
    }
}

/// Run one pool task, swallowing any panic so it cannot unwind out of a persistent worker thread or
/// a helping caller (M-864 Defect-2 last-line safety — see [`Pool::worker_loop`]). Job-panic
/// *propagation* to the owning batch's caller is done deliberately in the lane-loop (which captures
/// the payload and re-raises it at the join); this function only prevents an unwind from *escaping*
/// into the pool's own control loops. Any payload reaching here is therefore a non-job unwind (e.g. a
/// poisoned lock); it is dropped, and the affected batch's result slot stays `None` → surfaced at the
/// caller's own liveness unwrap rather than by killing the worker.
fn run_caught(task: PoolTask) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(task));
}

/// The process-wide persistent pool, sized once (lazily, on first use) to
/// [`thread::available_parallelism`] (fallback: 1 — never zero, so the pool is never unable to make
/// progress). Never resized, never recreated: every [`crate::scheduler::Scheduler`] value, however
/// many are constructed, and every `run_indexed` call, however deeply nested, shares this one pool.
/// This is the M-864 property that makes nested submission cheap — a nested call never spawns a new
/// OS thread, it only ever [`Pool::submit`]s more work to this same pool and [`Pool::help_while`]s.
pub(crate) fn get() -> Arc<Pool> {
    static POOL: OnceLock<Arc<Pool>> = OnceLock::new();
    Arc::clone(POOL.get_or_init(|| {
        let workers = thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        Pool::new(workers)
    }))
}
