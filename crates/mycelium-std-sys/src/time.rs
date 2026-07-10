//! \[Declared\] OS clock floor. Thin wrappers over Rust `std::time`.
//!
//! Declared — no audit of platform clock resolution or monotonicity guarantees beyond
//! what Rust's standard library provides. Wiring from `std-time` deferred to a future wave.
//!
//! RFC-0016 §9: once `std-time`'s `SystemClock` routes exclusively through this module,
//! the pure `std-time` crate earns a `wild`-free badge.
//!
//! # Never-silent contract (G2)
//!
//! - `wall_nanos()` returns `Err(String)` if the system time is before the Unix epoch.
//! - `mono_nanos()` is infallible within a single process run (Rust's `Instant` guarantee).
//! - `sleep_nanos()` is infallible but imprecise — callers must not assume precision
//!   (use `mono_nanos()` before and after to measure actual elapsed time).

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// \[Declared\] Returns nanoseconds since the Unix epoch from the wall clock.
///
/// Returns `Err` if the system time is before the epoch — never-silent (G2).
///
/// # Guarantee
///
/// `Declared` — backed by `std::time::SystemTime`. No audit of platform clock resolution,
/// drift, or leap-second handling.
pub fn wall_nanos() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .map_err(|e| e.to_string())
}

/// \[Declared\] Returns monotonic nanoseconds since an unspecified process-local epoch.
///
/// Guaranteed non-decreasing within a single process run (Rust `Instant` guarantee).
/// Not comparable across processes or restarts.
///
/// # Guarantee
///
/// `Declared` — backed by `std::time::Instant`. OS monotonic clock; no precision audit.
pub fn mono_nanos() -> u64 {
    static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let origin = ORIGIN.get_or_init(Instant::now);
    // Saturate at u64::MAX (~584 years) rather than wrapping; u128→u64 explicit saturating cast.
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// \[Declared\] Pause the current thread for approximately `nanos` nanoseconds.
///
/// Actual sleep duration may exceed `nanos` due to OS scheduling. This is never-silent on that
/// imprecision: callers must not assume precision. Use `mono_nanos()` before and after to
/// measure actual elapsed duration.
///
/// # Guarantee
///
/// `Declared` — backed by `std::thread::sleep(Duration::from_nanos(nanos))`. No precision audit.
pub fn sleep_nanos(nanos: u64) {
    std::thread::sleep(Duration::from_nanos(nanos));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_nanos_ok_and_nonzero() {
        let result = wall_nanos();
        assert!(result.is_ok(), "wall_nanos() returned Err: {result:?}");
        let nanos = result.unwrap();
        // The Unix epoch was 1970-01-01; any reasonable system clock returns a large value.
        assert!(
            nanos > 0,
            "wall_nanos() returned 0, expected a positive timestamp"
        );
    }

    #[test]
    fn mono_nanos_nondecreasing() {
        let t0 = mono_nanos();
        // Do a tiny bit of work so the clock actually ticks.
        std::thread::sleep(Duration::from_millis(1));
        let t1 = mono_nanos();
        assert!(
            t1 >= t0,
            "mono_nanos() decreased: t0={t0}, t1={t1} (monotonicity violation)"
        );
    }

    #[test]
    fn sleep_nanos_does_not_panic() {
        // Sleep 1 ms — just verify it doesn't panic or return an error.
        sleep_nanos(1_000_000);
    }

    #[test]
    fn mono_nanos_advances_after_sleep() {
        let t0 = mono_nanos();
        sleep_nanos(5_000_000); // 5 ms
        let t1 = mono_nanos();
        // After 5 ms, t1 must be strictly greater than t0.
        assert!(
            t1 > t0,
            "mono_nanos() did not advance after 5 ms sleep: t0={t0}, t1={t1}"
        );
    }
}
