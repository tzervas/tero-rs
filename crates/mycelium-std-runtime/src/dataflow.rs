//! `run_dataflow` — deadlock-freedom for communicating tasks (M-711 / RFC-0008 §4.3 / E12-1).
//!
//! Communicating tasks (producers/consumers over the [`crate::network`] channels) cannot be run to
//! completion one-at-a-time: a consumer spawned before its producer would block forever. They are
//! **swept** instead — one non-blocking poll-step per still-pending task per sweep — and a sweep
//! that makes **no progress** while tasks remain pending is a [`Deadlock`](crate::task::Deadlock):
//! an **explicit error, never a silent hang** (G2). This is the std-runtime sibling of the proven
//! `mycelium-mlir::runtime::Scope::run_dataflow` model.
//!
//! # Why it can never hang (G2)
//!
//! Every step is a **non-blocking poll** ([`PollTask::poll`] returns [`Step::Pending`] rather than
//! blocking on an empty/full channel — the channel surface is `try_send`/`try_recv` only). "Progress
//! this sweep" = *some task resolved* **or** the caller-supplied monotone `progress` counter advanced
//! (a successful channel op). A full sweep with neither, while pending tasks remain, is a genuine
//! stall — surfaced as `Deadlock`, not a parked thread.
//!
//! # Honesty (VR-5)
//!
//! - **Deadlock detection — `Empirical`.** Detection is *complete* for an acyclic (DAG) channel
//!   graph: a stalled DAG network always trips the no-progress sweep. Cyclic channel graphs are an
//!   open follow-up (FLAG: ADR-020 §7) — flagged, never silently mis-reported. Property-tested, not
//!   `Proven`.
//! - **Sweep determinism — `Exact`** (cooperative path): the schedule is a fixed function of the
//!   sweep direction, so the cooperative run is reproducible. The [`run_dataflow_scheduled`] path
//!   runs each sweep's independent polls across the OS-thread pool (M-709); it shares the *same*
//!   never-silent deadlock decision (a stall is order-independent), tagged `Empirical`.

use std::sync::atomic::{AtomicU64, Ordering};

use mycelium_core::GuaranteeStrength;

use crate::scheduler::Scheduler;
use crate::task::Deadlock;

/// Guarantee strength for deadlock detection (complete for DAG channel graphs; cyclic = open).
pub const DEADLOCK_DETECTION_STRENGTH: GuaranteeStrength = GuaranteeStrength::Empirical;

/// Guarantee strength for the cooperative sweep schedule (a fixed function of the sweep direction).
pub const SWEEP_DETERMINISM_STRENGTH: GuaranteeStrength = GuaranteeStrength::Exact;

/// One non-blocking step of a communicating task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// The task resolved (it will not be polled again).
    Done,
    /// The task made no terminal progress this step — typically parked on an empty/full channel.
    /// **Never blocks**: parking is an explicit value, so the sweep can detect a stall (G2).
    Pending,
}

/// A communicating task that is **swept** (polled non-blockingly) rather than run to completion.
///
/// `poll` must never block: park (return [`Step::Pending`]) instead of waiting on a channel, so a
/// global stall is detectable as an explicit [`Deadlock`] rather than a hung thread (G2).
pub trait PollTask {
    /// Take one non-blocking step. Returns [`Step::Done`] when the task has resolved.
    fn poll(&mut self) -> Step;
}

/// The order a sweep visits still-pending tasks. Two directions over the same DAG network yield the
/// same completion (Kahn-determinism, RFC-0008 §4.3) — the differential a test checks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SweepDir {
    /// Ascending task index (default).
    #[default]
    Ascending,
    /// Descending task index.
    Descending,
}

/// A monotone progress counter for channel work — bump it on every successful `try_send`/`try_recv`
/// so the sweep can tell "a channel op advanced the network" from "nothing moved" (G2).
///
/// The driver reads it once per sweep; the caller increments it from inside task polls.
#[derive(Debug, Default)]
pub struct Progress {
    epoch: AtomicU64,
}

impl Progress {
    /// A fresh counter at epoch 0.
    #[must_use]
    pub fn new() -> Self {
        Progress {
            epoch: AtomicU64::new(0),
        }
    }

    /// Record one unit of channel progress (a successful send or recv).
    pub fn bump(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// The current epoch (monotone).
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }
}

fn sweep_indices(n: usize, dir: SweepDir) -> Vec<usize> {
    match dir {
        SweepDir::Ascending => (0..n).collect(),
        SweepDir::Descending => (0..n).rev().collect(),
    }
}

/// Sweep `tasks` cooperatively until all resolve, or return [`Deadlock`] on a no-progress sweep.
///
/// `progress()` reports the monotone count of channel work done *outside* the tasks' own resolution.
/// A sweep counts as progress if **either** a task resolved **or** `progress()` advanced; a full
/// sweep with neither, while tasks remain pending, is an explicit [`Deadlock`] — never a hang (G2).
///
/// Guarantee: schedule is **`Exact`** (a fixed function of `dir`); deadlock detection is
/// **`Empirical`** (complete for DAG channel graphs — see module docs).
///
/// # Errors
/// Returns [`Deadlock`] (carrying the count of still-parked tasks) when a full sweep makes no
/// progress while tasks remain pending.
pub fn run_dataflow(
    tasks: &mut [Box<dyn PollTask>],
    dir: SweepDir,
    progress: impl Fn() -> u64,
) -> Result<(), Deadlock> {
    let n = tasks.len();
    let mut done = vec![false; n];
    let mut remaining = n;
    // The sweep order is a fixed function of `n` and `dir`, so compute it once and reuse it across
    // sweeps (no per-sweep allocation on long-running networks).
    let order = sweep_indices(n, dir);
    while remaining > 0 {
        let before = progress();
        let mut advanced = false;
        for &i in &order {
            if done[i] {
                continue;
            }
            if tasks[i].poll() == Step::Done {
                done[i] = true;
                remaining -= 1;
                advanced = true;
            }
        }
        // Progress = a task resolved OR a channel op advanced the epoch. Neither, with tasks still
        // pending, is a genuine deadlock — explicit, never a silent hang (G2 / RFC-0008 §4.3).
        if !advanced && progress() == before && remaining > 0 {
            return Err(Deadlock::new(remaining));
        }
    }
    Ok(())
}

/// A transient placeholder [`PollTask`] swapped into `tasks[i]` while the real task at that index is
/// on loan to a parallel sweep (see [`run_dataflow_scheduled`]) — never actually polled itself (the
/// real task is always swapped back before the next sweep), but must type-check as a valid
/// [`PollTask`] to occupy the slot.
struct AlreadyDone;
impl PollTask for AlreadyDone {
    fn poll(&mut self) -> Step {
        Step::Done
    }
}

/// As [`run_dataflow`], but each sweep's independent polls run **across the OS-thread pool**
/// (M-709/M-864) — the "checked across OS threads" path (RFC-0008 §4.3 under the real scheduler).
///
/// Each still-pending task is polled **at most once per sweep** on a worker thread; the tasks are
/// disjoint objects (no shared mutable state but the channels, RT1), so the parallel polls are
/// data-race-free. The deadlock decision is identical and order-independent: a stall is a stall in
/// any sweep order, so it is surfaced as the same explicit [`Deadlock`] (G2), never a hung worker.
///
/// **M-864 note:** `Scheduler::run_indexed` now requires `'static` job closures (the persistent
/// pool's worker threads outlive any single call), so this sweep can no longer *borrow* `tasks[i]`
/// by `&mut` across the call the way the pre-M-864 `thread::scope`-backed scheduler allowed. Each
/// still-pending task is instead swapped **out** of `tasks` (taking ownership, `'static` by
/// construction — only the borrow was ever the obstacle, not the boxed task itself) for the
/// duration of the parallel poll, and swapped back the moment its result is in hand — the sweep's
/// own book-keeping (`done`/`remaining`) and the deadlock decision are otherwise unchanged.
///
/// Guarantee: deadlock detection **`Empirical`** (same basis as [`run_dataflow`], now exercised on
/// real OS threads).
///
/// # Errors
/// Returns [`Deadlock`] when a full parallel sweep makes no progress while tasks remain pending.
pub fn run_dataflow_scheduled(
    scheduler: &Scheduler,
    tasks: &mut [Box<dyn PollTask + Send>],
    progress: impl Fn() -> u64,
) -> Result<(), Deadlock> {
    let n = tasks.len();
    let mut done = vec![false; n];
    let mut remaining = n;
    while remaining > 0 {
        let before = progress();
        // Take OWNERSHIP of each still-pending task for this sweep's parallel poll (M-864 — see the
        // doc above): a placeholder occupies the slot only for the duration of the poll and is never
        // itself polled.
        let pending: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
        let taken: Vec<Box<dyn PollTask + Send>> = pending
            .iter()
            .map(|&i| std::mem::replace(&mut tasks[i], Box::new(AlreadyDone)))
            .collect();
        let jobs: Vec<_> = taken
            .into_iter()
            .map(|mut t| {
                move || {
                    let became_done = matches!(t.poll(), Step::Done);
                    (t, became_done)
                }
            })
            .collect();
        let results = scheduler.run_indexed(jobs, None, None);
        let mut advanced = false;
        for (&i, (t, became_done)) in pending.iter().zip(results) {
            tasks[i] = t; // restore the (possibly now-resolved) task to its original slot
            if became_done {
                done[i] = true;
                remaining -= 1;
                advanced = true;
            }
        }
        // Identical never-silent deadlock decision, now over a parallel sweep (order-independent).
        if !advanced && progress() == before && remaining > 0 {
            return Err(Deadlock::new(remaining));
        }
    }
    Ok(())
}
