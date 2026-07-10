//! Tests for `crate::dataflow` (M-711 sweep-based deadlock-freedom; M-864 `run_dataflow_scheduled`
//! ownership-swap adjustment).
//!
//! M-797 in-crate test layout: extracted from the former inline `#[cfg(test)] mod tests` in
//! `dataflow.rs` (as-touched, per the M-864 change to `run_dataflow_scheduled`).

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::dataflow::*;
use crate::scheduler::Scheduler;

/// A task that resolves after `steps` polls, bumping shared progress each step (a stand-in for a
/// successful channel op). Models a producer/consumer that makes forward progress.
struct Countdown {
    steps: usize,
    progress: Rc<Cell<u64>>,
}
impl PollTask for Countdown {
    fn poll(&mut self) -> Step {
        if self.steps == 0 {
            return Step::Done;
        }
        self.steps -= 1;
        self.progress.set(self.progress.get() + 1);
        if self.steps == 0 {
            Step::Done
        } else {
            Step::Pending
        }
    }
}

/// A task that never makes progress — always `Pending`, never bumps progress. A pair of these
/// models a true deadlock (two tasks each waiting on the other).
struct Stuck;
impl PollTask for Stuck {
    fn poll(&mut self) -> Step {
        Step::Pending
    }
}

#[test]
fn satisfiable_network_completes() {
    let prog = Rc::new(Cell::new(0u64));
    let mut tasks: Vec<Box<dyn PollTask>> = vec![
        Box::new(Countdown {
            steps: 3,
            progress: Rc::clone(&prog),
        }),
        Box::new(Countdown {
            steps: 5,
            progress: Rc::clone(&prog),
        }),
    ];
    let p = Rc::clone(&prog);
    let r = run_dataflow(&mut tasks, SweepDir::Ascending, move || p.get());
    assert!(
        r.is_ok(),
        "a network that makes progress must complete, got {r:?}"
    );
}

#[test]
fn stalled_network_is_explicit_deadlock_never_hangs() {
    // Mutant witness: removing the no-progress check would loop forever (test would hang).
    let prog = Rc::new(Cell::new(0u64));
    let mut tasks: Vec<Box<dyn PollTask>> = vec![Box::new(Stuck), Box::new(Stuck)];
    let p = Rc::clone(&prog);
    let err = run_dataflow(&mut tasks, SweepDir::Ascending, move || p.get())
        .expect_err("a fully stalled network must return Deadlock, never hang (G2)");
    assert_eq!(
        err.task_count, 2,
        "Deadlock must report the parked task count"
    );
}

#[test]
fn sweep_direction_is_determinism_invariant() {
    // Kahn-determinism (RFC-0008 §4.3): ascending and descending sweeps complete the same
    // satisfiable network (both Ok). The schedule differs; the outcome does not.
    for dir in [SweepDir::Ascending, SweepDir::Descending] {
        let prog = Rc::new(Cell::new(0u64));
        let mut tasks: Vec<Box<dyn PollTask>> = (1..=4)
            .map(|s| {
                Box::new(Countdown {
                    steps: s,
                    progress: Rc::clone(&prog),
                }) as Box<dyn PollTask>
            })
            .collect();
        let p = Rc::clone(&prog);
        assert!(
            run_dataflow(&mut tasks, dir, move || p.get()).is_ok(),
            "{dir:?} sweep must complete the satisfiable network"
        );
    }
}

// ── Scheduled (OS-thread) path: the deadlock decision holds across real threads ──

/// A `Send` countdown using atomics, for the scheduled driver. Idempotent `Done`.
struct AtomicCountdown {
    steps: AtomicUsize,
    progress: Arc<AtomicU64>,
}
impl PollTask for AtomicCountdown {
    fn poll(&mut self) -> Step {
        let cur = self.steps.load(Ordering::SeqCst);
        if cur == 0 {
            return Step::Done;
        }
        self.steps.store(cur - 1, Ordering::SeqCst);
        self.progress.fetch_add(1, Ordering::SeqCst);
        if cur - 1 == 0 {
            Step::Done
        } else {
            Step::Pending
        }
    }
}

struct AtomicStuck;
impl PollTask for AtomicStuck {
    fn poll(&mut self) -> Step {
        Step::Pending
    }
}

#[test]
fn scheduled_satisfiable_network_completes_on_os_threads() {
    let sched = Scheduler::with_workers(4, 8).unwrap();
    let prog = Arc::new(AtomicU64::new(0));
    let mut tasks: Vec<Box<dyn PollTask + Send>> = (1..=6)
        .map(|s| {
            Box::new(AtomicCountdown {
                steps: AtomicUsize::new(s),
                progress: Arc::clone(&prog),
            }) as Box<dyn PollTask + Send>
        })
        .collect();
    let p = Arc::clone(&prog);
    let r = run_dataflow_scheduled(&sched, &mut tasks, move || p.load(Ordering::SeqCst));
    assert!(
        r.is_ok(),
        "scheduled satisfiable network must complete, got {r:?}"
    );
}

#[test]
fn scheduled_stalled_network_is_explicit_deadlock_never_hangs() {
    // The never-silent guarantee under the real scheduler: a stall is Deadlock, not a hung pool.
    let sched = Scheduler::with_workers(4, 8).unwrap();
    let prog = Arc::new(AtomicU64::new(0));
    let mut tasks: Vec<Box<dyn PollTask + Send>> = vec![
        Box::new(AtomicStuck),
        Box::new(AtomicStuck),
        Box::new(AtomicStuck),
    ];
    let p = Arc::clone(&prog);
    let err = run_dataflow_scheduled(&sched, &mut tasks, move || p.load(Ordering::SeqCst))
        .expect_err("a stalled network must return Deadlock under the scheduler, never hang");
    assert_eq!(
        err.task_count, 3,
        "Deadlock must report the parked task count"
    );
}

#[test]
fn scheduled_sweep_restores_tasks_to_their_original_slots_across_multiple_sweeps() {
    // M-864 regression witness (strengthened per the semantic review): `run_dataflow_scheduled` now
    // takes OWNERSHIP of each still-pending task for a sweep (swapping a transient `AlreadyDone`
    // placeholder into its slot) and must restore the REAL task to the SAME index afterward.
    //
    // Why the old "only assert Ok()" was too weak: `done[]` is tracked by INDEX. If a sweep restored
    // a task to the wrong slot, a still-pending task could land at an index already marked done and
    // be skipped forever — which *usually* deadlocks, but can instead spuriously "complete" if a
    // finished task double-counts at another slot (remaining hits 0 while a real task was stranded).
    // So `Ok()` alone does NOT catch a mis-restore.
    //
    // The robust check: each task bumps a SHARED `progress` counter once per real step. The total
    // number of real steps is fixed (sum of the per-task step counts). A correct run performs EXACTLY
    // that many steps; a mis-restore that strands any task performs FEWER. Assert the exact total —
    // slot-sensitive, so a shuffle bug is caught whether it deadlocks or "completes early".
    let sched = Scheduler::with_workers(3, 8).unwrap();
    let step_counts: Vec<usize> = (1..=9).collect();
    let expected_total_steps: u64 = step_counts.iter().map(|&s| s as u64).sum(); // 45
    let prog = Arc::new(AtomicU64::new(0));
    let mut tasks: Vec<Box<dyn PollTask + Send>> = step_counts
        .iter()
        .map(|&s| {
            Box::new(AtomicCountdown {
                steps: AtomicUsize::new(s),
                progress: Arc::clone(&prog),
            }) as Box<dyn PollTask + Send>
        })
        .collect();
    let p = Arc::clone(&prog);
    let r = run_dataflow_scheduled(&sched, &mut tasks, move || p.load(Ordering::SeqCst));
    assert!(
        r.is_ok(),
        "a satisfiable network with varied per-task step counts must complete across many sweeps, \
         got {r:?}"
    );
    assert_eq!(
        prog.load(Ordering::SeqCst),
        expected_total_steps,
        "every task must run its full step count exactly once — a mis-restore (wrong slot) would \
         strand a task and leave the total short (M-864 ownership-swap correctness)"
    );
}
