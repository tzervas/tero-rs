//! The process-wide arena (§4.2): concurrent reservations sum against one shared ceiling, refuse when
//! the sum would exceed it, and release on drop.
//!
//! `PROCESS_BYTES_CHARGED` is a process-global static, and cargo runs tests in parallel threads within
//! one process — so these tests **serialize** on a local `Mutex` and each fully drops its reservations
//! before releasing the lock, returning the counter to zero between cases (no test-only reset backdoor).

use crate::{current_process_bytes, BudgetError, BudgetKind, ProcessArena};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};

/// Serializes every arena test (the counter is process-global). A poisoned lock is fine to recover —
/// we only need mutual exclusion.
static ARENA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn arena_guard() -> std::sync::MutexGuard<'static, ()> {
    let g = ARENA_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        current_process_bytes(),
        0,
        "each arena test starts from a zeroed process counter"
    );
    g
}

#[test]
fn reservations_sum_and_release_on_drop() {
    let _lock = arena_guard();
    let arena = ProcessArena::new(1000);

    let a = arena.reserve(400).expect("first reservation fits");
    assert_eq!(current_process_bytes(), 400);
    let b = arena
        .reserve(500)
        .expect("second reservation still fits (sum 900 <= 1000)");
    assert_eq!(current_process_bytes(), 900);

    // The sum — not the per-reservation size — is what the ceiling bounds.
    match arena.reserve(200) {
        Err(BudgetError::OutOfBudget {
            kind: BudgetKind::Bytes,
            limit,
            requested,
        }) => {
            assert_eq!(limit, 1000);
            assert_eq!(
                requested, 1100,
                "requested is the process-wide would-be total"
            );
        }
        other => panic!("a reservation past the ceiling must refuse, got {other:?}"),
    }
    assert_eq!(
        current_process_bytes(),
        900,
        "a refused reservation charges nothing"
    );

    drop(b);
    assert_eq!(
        current_process_bytes(),
        400,
        "releasing frees exactly its bytes"
    );
    // With `b` released there is room again.
    let c = arena.reserve(200).expect("room after release");
    assert_eq!(current_process_bytes(), 600);
    drop(a);
    drop(c);
    assert_eq!(
        current_process_bytes(),
        0,
        "all reservations released → counter back to zero"
    );
}

#[test]
fn reservation_at_exactly_the_ceiling_is_accepted() {
    // Mutant witness for the arena's `next > ceiling` boundary.
    let _lock = arena_guard();
    let arena = ProcessArena::new(512);
    let r = arena
        .reserve(512)
        .expect("reserving exactly the ceiling is allowed");
    assert_eq!(current_process_bytes(), 512);
    assert!(
        arena.reserve(1).is_err(),
        "one byte past the ceiling refuses"
    );
    drop(r);
    assert_eq!(current_process_bytes(), 0);
}

/// Concurrency: `THREADS` workers each try to reserve one `CHUNK`, and **all hold** at a barrier before
/// any releases — so the reserve phase runs with zero releases and the peak is observed with every
/// grant live. With a ceiling admitting exactly 10 chunks, exactly 10 grants succeed and the peak equals
/// the ceiling: the compare-exchange loop makes the joint check atomic, so grants can never *jointly*
/// exceed the ceiling (a non-atomic load-then-add would let concurrent threads both "see room" and
/// overrun — this test would catch that).
#[test]
fn concurrent_reservations_never_jointly_exceed_the_ceiling() {
    let _lock = arena_guard();

    const THREADS: usize = 16;
    const CHUNK: u64 = 100;
    const FITS: u64 = 10; // ceiling admits exactly 10 of the 16 chunks
    let arena = ProcessArena::new(CHUNK * FITS);
    let granted = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(THREADS));

    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            let arena = arena.clone();
            let granted = Arc::clone(&granted);
            let peak = Arc::clone(&peak);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                // Reserve (or be refused) — no reservation is released until the barrier below, so the
                // whole reserve phase runs with zero releases.
                let reservation = arena.reserve(CHUNK).ok();
                if reservation.is_some() {
                    granted.fetch_add(1, Ordering::AcqRel);
                }
                // Every thread — granted or not — meets here, so all grants are simultaneously live.
                barrier.wait();
                // With all grants live, this reads the peak process-wide total.
                peak.fetch_max(current_process_bytes(), Ordering::AcqRel);
                // `reservation` drops at the end of the closure, after the peak read.
                drop(reservation);
            });
        }
    });

    // Deterministic: with no releases during the reserve phase, exactly `FITS` grants succeed and the
    // peak equals the ceiling — never above it.
    assert_eq!(
        granted.load(Ordering::Acquire),
        FITS,
        "exactly the admitted number of grants succeed"
    );
    assert_eq!(
        peak.load(Ordering::Acquire),
        CHUNK * FITS,
        "the peak equals the ceiling, never above"
    );
    assert_eq!(current_process_bytes(), 0, "all threads released on join");
}
