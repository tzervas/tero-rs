//! **Typed single-producer/single-consumer channels** (RFC-0008 §4.3) — the Kahn-deterministic
//! *communicating* half of the RT2 fragment, the slice the [`runtime`](crate::runtime) fork/join
//! executor named as next.
//!
//! ## What lands here (the chosen scope, §4.3)
//! - **Typed, value-carrying SPSC channels.** [`Network::channel`] returns an affine
//!   [`Sender`]/[`Receiver`] pair — *neither is [`Clone`]*, so single-producer/single-consumer holds
//!   **by construction** (RT1: the only thing that crosses a channel is a moved value; the queue is
//!   the channel's own state, not shared mutable task state).
//! - **Bounded buffer + demand-signalled backpressure.** Capacity is an explicit, finite
//!   [`NonZeroUsize`] — an unbounded silent buffer is a hidden resource leak and is *excluded by
//!   construction* (RT7's spirit on queues, §4.3). A [`Sender::try_send`] on a full buffer returns
//!   [`TrySend::Full`] **carrying the value back** (never dropped); the producer task yields and is
//!   re-polled when the consumer drains a slot — the consumer draining *is* the demand signal.
//! - **Explicit close / end-of-stream.** Dropping the [`Sender`] closes the channel; the
//!   [`Receiver`] drains what is buffered, then observes [`TryRecv::Closed`] — never a silent hang. A
//!   [`Sender::try_send`] to a hung-up receiver is [`TrySend::Disconnected`] (value returned,
//!   explicit — G2), never a silent drop.
//! - **Deadlock is explicit.** The single-threaded cooperative scheduler cannot block, so a stuck
//!   network is surfaced as a [`Deadlock`](crate::runtime::Deadlock) by
//!   [`Scope::run_dataflow`](crate::runtime::Scope::run_dataflow), never a hang (G2).
//!
//! ## Determinism (the RT2 obligation for communicating tasks)
//! A [`Network`] is a **Kahn process network**: deterministic processes (pure tasks) communicating
//! over blocking single-reader channels. By Kahn's theorem (T4.1) its observable behaviour is
//! independent of the (fair) schedule. We *verify* this with a differential — the same network run
//! under two distinct fair schedules ([`SweepOrder`](crate::runtime::SweepOrder) ascending vs
//! descending) yields identical per-task outcomes and channel transcripts. The honest tag on that
//! determinism is therefore **`Empirical`** (the differential is the evidence), with Kahn T4.1 cited
//! as the basis; it is **not** `Proven` — no mechanized proof ships in-repo (VR-5: never upgrade
//! without a checked basis).
//!
//! ## What does **not** land here (honest boundary)
//! Multi-source `select`/`merge` stay **RT3** constructs (the arbitration is a named policy);
//! session/protocol typing beyond the §4.3 *hook* is deferred (dynamic check first, static
//! discipline later); zero-capacity rendezvous channels are excluded by [`NonZeroUsize`];
//! cross-node `xloc`/`mesh` are **R2** (distribution). No kernel change (KC-3); the scheduler and
//! channels live wholly outside the trusted evaluator. Single-threaded cooperative scheduling means
//! every [`RefCell`] borrow is taken and released within one `poll`, before the task yields, so
//! borrows never overlap — no `unsafe`, no atomics.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::rc::Rc;

/// The shared bounded buffer behind one channel. Owned jointly by the channel's single [`Sender`]
/// and single [`Receiver`]; the cooperative scheduler serializes access, so the `RefCell` is never
/// contended across a yield point.
struct Shared<V> {
    /// The buffered, in-flight values (FIFO; bounded by `cap`).
    buf: VecDeque<V>,
    /// The explicit, finite capacity — no unbounded buffer (RT7 spirit).
    cap: usize,
    /// Live producer count: `1` while the `Sender` exists, `0` once it is dropped (closed).
    senders: u8,
    /// Live consumer count: `1` while the `Receiver` exists, `0` once it is dropped (hung up).
    receivers: u8,
    /// The network-wide **progress clock** (see [`Network::epoch`]): bumped on every successful
    /// send/recv so the dataflow scheduler can tell "a task advanced the dataflow" from "every task
    /// is parked" — the basis of deadlock detection.
    epoch: Rc<Cell<u64>>,
}

/// A **Kahn process network** (RFC-0008 §4.3): the grouping whose typed SPSC channels form a
/// deterministic dataflow graph. It owns the shared **progress clock** — the running count of
/// successful channel operations across the network — which
/// [`Scope::run_dataflow`](crate::runtime::Scope::run_dataflow) watches to detect a stalled network.
/// The clock is a plain, inspectable count (no black box — SC-3).
#[derive(Clone, Default)]
pub struct Network {
    epoch: Rc<Cell<u64>>,
}

impl Network {
    /// A fresh network with its progress clock at zero.
    #[must_use]
    pub fn new() -> Self {
        Network {
            epoch: Rc::new(Cell::new(0)),
        }
    }

    /// The number of successful channel sends + recvs across this network so far — monotone,
    /// inspectable, and the signal the dataflow scheduler uses for progress/deadlock.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch.get()
    }

    /// Create a typed SPSC channel on this network with explicit, finite capacity `cap` (no
    /// unbounded buffer — RT7). Returns the affine [`Sender`]/[`Receiver`] pair.
    #[must_use]
    pub fn channel<V>(&self, cap: NonZeroUsize) -> (Sender<V>, Receiver<V>) {
        let shared = Rc::new(RefCell::new(Shared {
            buf: VecDeque::new(),
            cap: cap.get(),
            senders: 1,
            receivers: 1,
            epoch: Rc::clone(&self.epoch),
        }));
        (
            Sender {
                shared: Rc::clone(&shared),
            },
            Receiver { shared },
        )
    }
}

/// The **single producer** end of a channel. Not [`Clone`] — single-producer by construction.
pub struct Sender<V> {
    shared: Rc<RefCell<Shared<V>>>,
}

/// The **single consumer** end of a channel. Not [`Clone`] — single-consumer by construction.
pub struct Receiver<V> {
    shared: Rc<RefCell<Shared<V>>>,
}

/// Why a [`Sender::try_send`] could not complete *right now*. Both variants **carry the value back**
/// — a send that does not happen never drops the value (G2: never silent).
#[derive(Debug, PartialEq, Eq)]
pub enum TrySend<V> {
    /// The buffer is at capacity — **backpressure**. The producer should yield and retry after the
    /// consumer drains a slot. The value is returned to retry with.
    Full(V),
    /// The receiver has hung up — the value can never be delivered. Explicit, not a silent drop.
    Disconnected(V),
}

/// Why a [`Receiver::try_recv`] yielded no value.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TryRecv {
    /// No value buffered right now, but the producer is still connected — the consumer **parks**
    /// (yields) until a send or a close.
    Empty,
    /// The producer hung up **and** the buffer is drained — **end of stream**, explicit (never a
    /// hang).
    Closed,
}

impl<V> Sender<V> {
    /// Non-blocking send. `Ok(())` on success (and the network epoch advances); `Err(Full(v))` for
    /// backpressure; `Err(Disconnected(v))` if the receiver is gone. On any failure the value is
    /// returned, never dropped.
    ///
    /// # Errors
    /// [`TrySend::Full`] when the buffer is at capacity; [`TrySend::Disconnected`] when the receiver
    /// has been dropped.
    pub fn try_send(&self, v: V) -> Result<(), TrySend<V>> {
        let mut s = self.shared.borrow_mut();
        if s.receivers == 0 {
            return Err(TrySend::Disconnected(v));
        }
        if s.buf.len() >= s.cap {
            return Err(TrySend::Full(v));
        }
        s.buf.push_back(v);
        s.epoch.set(s.epoch.get() + 1);
        Ok(())
    }

    /// Whether the receiver is still connected — an inspectable predicate (no hidden state).
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.shared.borrow().receivers > 0
    }
}

impl<V> Receiver<V> {
    /// Non-blocking receive. `Ok(v)` on success (the network epoch advances); `Err(Empty)` to park;
    /// `Err(Closed)` at end of stream (producer gone and buffer drained).
    ///
    /// # Errors
    /// [`TryRecv::Empty`] when nothing is buffered but the producer is still live;
    /// [`TryRecv::Closed`] when the producer has hung up and the buffer is drained.
    pub fn try_recv(&self) -> Result<V, TryRecv> {
        let mut s = self.shared.borrow_mut();
        match s.buf.pop_front() {
            Some(v) => {
                s.epoch.set(s.epoch.get() + 1);
                Ok(v)
            }
            None => {
                if s.senders == 0 {
                    Err(TryRecv::Closed)
                } else {
                    Err(TryRecv::Empty)
                }
            }
        }
    }

    /// Whether the producer is still connected (buffered items may remain even after it hangs up).
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.shared.borrow().senders > 0
    }
}

impl<V> Drop for Sender<V> {
    fn drop(&mut self) {
        // Closing the producer end: the receiver will drain, then see `Closed` (never a hang).
        self.shared.borrow_mut().senders -= 1;
    }
}

impl<V> Drop for Receiver<V> {
    fn drop(&mut self) {
        // Hanging up the consumer end: a subsequent `try_send` is an explicit `Disconnected`.
        self.shared.borrow_mut().receivers -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Deadlock, Poll, Scope, SweepOrder, Task, TaskCtx};
    use mycelium_interp::{Budgets, TaskOutcome};

    fn cap(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("non-zero capacity")
    }

    // --- direct channel mechanics (backpressure, close, disconnect — all explicit) ---

    #[test]
    fn backpressure_full_returns_the_value_then_drains() {
        let net = Network::new();
        let (tx, rx) = net.channel::<i64>(cap(2));
        assert!(tx.try_send(1).is_ok());
        assert!(tx.try_send(2).is_ok());
        // Buffer full → backpressure, the value comes back (never dropped).
        assert_eq!(tx.try_send(3), Err(TrySend::Full(3)));
        assert_eq!(net.epoch(), 2, "two successful sends");
        // Consumer drains one → a slot frees → the send now succeeds.
        assert_eq!(rx.try_recv(), Ok(1));
        assert!(tx.try_send(3).is_ok());
        assert_eq!(rx.try_recv(), Ok(2));
        assert_eq!(rx.try_recv(), Ok(3));
        assert_eq!(
            rx.try_recv(),
            Err(TryRecv::Empty),
            "drained, producer still live"
        );
    }

    #[test]
    fn close_drains_then_reports_end_of_stream() {
        let net = Network::new();
        let (tx, rx) = net.channel::<i64>(cap(4));
        tx.try_send(10).unwrap();
        tx.try_send(20).unwrap();
        drop(tx); // producer hangs up
        assert_eq!(rx.try_recv(), Ok(10), "buffered items drain first");
        assert_eq!(rx.try_recv(), Ok(20));
        assert_eq!(
            rx.try_recv(),
            Err(TryRecv::Closed),
            "then explicit end-of-stream"
        );
    }

    #[test]
    fn send_to_a_gone_receiver_is_explicit_not_a_silent_drop() {
        let net = Network::new();
        let (tx, rx) = net.channel::<i64>(cap(1));
        drop(rx);
        assert_eq!(
            tx.try_send(7),
            Err(TrySend::Disconnected(7)),
            "value returned, not dropped"
        );
        assert!(!tx.is_connected());
    }

    // --- the dataflow scheduler: producer → consumer over a bounded channel ---

    /// A producer that sends `items` (in order) into `tx`, honouring backpressure, then **closes**
    /// (drops its sender) so the consumer sees end-of-stream. Pure: owns its state (RT1).
    struct Producer {
        tx: Option<Sender<i64>>,
        items: VecDeque<i64>,
    }

    impl Task for Producer {
        type Output = Vec<i64>;
        type Error = String;
        fn poll(&mut self, _cx: &mut TaskCtx) -> Poll<Vec<i64>, String> {
            let tx = match &self.tx {
                Some(tx) => tx,
                None => return Poll::Ready(TaskOutcome::Done(vec![])),
            };
            match self.items.front().copied() {
                None => {
                    self.tx = None; // close: the consumer will now drain → Closed
                    Poll::Ready(TaskOutcome::Done(vec![]))
                }
                Some(v) => match tx.try_send(v) {
                    Ok(()) => {
                        self.items.pop_front();
                        Poll::Pending
                    }
                    Err(TrySend::Full(_)) => Poll::Pending, // backpressure — yield and retry
                    Err(TrySend::Disconnected(_)) => {
                        self.tx = None;
                        Poll::Ready(TaskOutcome::Failed("receiver gone".into()))
                    }
                },
            }
        }
    }

    /// A consumer that accumulates everything it receives, in order, until end-of-stream. Its
    /// collected vector is the channel **transcript** the determinism differential compares.
    struct Consumer {
        rx: Receiver<i64>,
        got: Vec<i64>,
    }

    impl Task for Consumer {
        type Output = Vec<i64>;
        type Error = String;
        fn poll(&mut self, _cx: &mut TaskCtx) -> Poll<Vec<i64>, String> {
            match self.rx.try_recv() {
                Ok(v) => {
                    self.got.push(v);
                    Poll::Pending
                }
                Err(TryRecv::Empty) => Poll::Pending, // park until a send or close
                Err(TryRecv::Closed) => {
                    Poll::Ready(TaskOutcome::Done(std::mem::take(&mut self.got)))
                }
            }
        }
    }

    /// Build a fresh producer→consumer scope over a `cap`-bounded channel carrying `items`. Returns
    /// the scope and the network (whose epoch the dataflow scheduler watches).
    fn pipe(items: &[i64], capacity: usize) -> (Scope<'static, Vec<i64>, String>, Network) {
        let net = Network::new();
        let (tx, rx) = net.channel::<i64>(cap(capacity));
        let mut scope = Scope::new();
        scope.spawn(
            Box::new(Producer {
                tx: Some(tx),
                items: items.iter().copied().collect(),
            }),
            Budgets::new(),
        );
        scope.spawn(Box::new(Consumer { rx, got: vec![] }), Budgets::new());
        (scope, net)
    }

    #[test]
    fn pipeline_delivers_in_order_under_backpressure() {
        let items: Vec<i64> = (0..7).collect();
        let (scope, net) = pipe(&items, 2); // cap 2 forces repeated backpressure
        let out = scope
            .run_dataflow(SweepOrder::Ascending, || net.epoch())
            .expect("no deadlock");
        assert_eq!(out[0], TaskOutcome::Done(vec![]), "producer completes");
        assert_eq!(
            out[1],
            TaskOutcome::Done(items),
            "consumer receives every item, in order"
        );
    }

    #[test]
    fn kahn_determinism_two_fair_schedules_agree() {
        // The RT2 obligation for communicating tasks: the same network under two DISTINCT fair
        // schedules produces identical outcomes + transcripts (Kahn T4.1; verified, not assumed).
        let items: Vec<i64> = (0..10).collect();
        let (asc_scope, asc_net) = pipe(&items, 3);
        let (desc_scope, desc_net) = pipe(&items, 3);
        let asc = asc_scope
            .run_dataflow(SweepOrder::Ascending, || asc_net.epoch())
            .unwrap();
        let desc = desc_scope
            .run_dataflow(SweepOrder::Descending, || desc_net.epoch())
            .unwrap();
        assert_eq!(
            asc, desc,
            "channel network is schedule-independent (Kahn determinism)"
        );
        assert_eq!(
            asc[1],
            TaskOutcome::Done(items),
            "and it is the right transcript"
        );
    }

    // --- deadlock is an explicit error, never a hang (G2) ---

    /// A task that only ever tries to receive from `rx` and never completes on its own.
    struct WaitForever {
        rx: Receiver<i64>,
    }

    impl Task for WaitForever {
        type Output = i64;
        type Error = String;
        fn poll(&mut self, _cx: &mut TaskCtx) -> Poll<i64, String> {
            match self.rx.try_recv() {
                Ok(v) => Poll::Ready(TaskOutcome::Done(v)),
                Err(TryRecv::Empty) => Poll::Pending, // parked: producer alive but silent
                Err(TryRecv::Closed) => Poll::Ready(TaskOutcome::Failed("closed".into())),
            }
        }
    }

    #[test]
    fn a_stalled_network_is_an_explicit_deadlock_not_a_hang() {
        let net = Network::new();
        let (tx, rx) = net.channel::<i64>(cap(1));
        // `tx` is held here and never used: the channel stays OPEN (so the receiver parks on `Empty`,
        // not `Closed`) but no value ever arrives — a genuine deadlock.
        let mut scope: Scope<i64, String> = Scope::new();
        scope.spawn(Box::new(WaitForever { rx }), Budgets::new());
        let res = scope.run_dataflow(SweepOrder::Ascending, || net.epoch());
        assert_eq!(
            res,
            Err(Deadlock { parked: vec![0] }),
            "explicit deadlock, never a silent hang"
        );
        drop(tx);
    }
}
