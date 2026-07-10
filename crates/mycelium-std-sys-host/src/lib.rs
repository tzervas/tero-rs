// nodule: mycelium-std-sys-host — production host wiring (RFC-0028 §4.5; M-722/M-723)
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use mycelium_std_rand::{EntropyEffect, EntropySource, RandErr};
use mycelium_std_time::{
    ClockSource, DeclaredTime, DeclaredTimeEntropy, LogicalInstant, MonoInstant, TimeErr,
    WallInstant,
};

/// The production [`EntropySource`] — fills entropy from the audited `std-sys` OS floor
/// (`/dev/urandom` via `std::fs`; M-723). This is the seam `std-rand` left injectable: the pure
/// `std-rand` crate never touches the OS, and this adapter supplies the real source.
///
/// **Guarantee: `Declared`** (RFC-0016 §4.1 C2 / VR-5). The source is the kernel CSPRNG, a genuine
/// OS entropy source, but no statistical-quality theorem or corpus is checked here — so `Declared`,
/// never `Empirical`/`Proven`. **Never-silent (G2):** an unavailable source is
/// `Err(RandErr::EntropyUnavailable)`, never a silent zero-fill (the `std-sys` floor guarantees a
/// full fill or an explicit error; this adapter only maps the error type).
#[derive(Debug, Default, Clone, Copy)]
pub struct OsEntropy;

impl EntropySource for OsEntropy {
    fn fill_bytes(&mut self, buf: &mut [u8]) -> Result<EntropyEffect, RandErr> {
        // Map the floor's explicit `Err(EntropyError::Unavailable)` onto `std-rand`'s never-silent
        // `Err(RandErr::EntropyUnavailable)`. No fallback fill (G2): the caller is told it failed.
        mycelium_std_sys::rand::fill_bytes(buf)
            .map(|()| EntropyEffect)
            .map_err(|_| RandErr::EntropyUnavailable)
    }
}

/// The production [`ClockSource`] — reads the OS clock through the audited `std-sys` time floor
/// (`std::time::{Instant, SystemTime}`; M-723). The seam `std-time` left injectable: `std-time`
/// owns the typed/effect-declaring reading surface, this adapter supplies the real clock.
///
/// **Guarantee: `Declared`** for every read (RFC-0016 §4.1 C2 / VR-5): an ambient, nondeterministic
/// host read with no checked bound. Effects are carried by the return types (`DeclaredTime` for the
/// monotonic/logical reads, `DeclaredTimeEntropy` for the wall read — civil time is an RT3 entropy
/// source). **Never-silent (G2):** a wall-clock read before the Unix epoch or an out-of-range tick
/// is an explicit `Err(TimeErr::…)`, never a wrap/clamp.
#[derive(Debug, Default, Clone, Copy)]
pub struct OsClock;

impl ClockSource for OsClock {
    fn mono_now(&self) -> DeclaredTime<Result<MonoInstant, TimeErr>> {
        // The monotonic floor read is total (`-> u64`): it never fails, so always `Ok`.
        DeclaredTime::new(Ok(MonoInstant::from_nanos(
            mycelium_std_sys::time::mono_nanos(),
        )))
    }

    fn wall_now(&self) -> DeclaredTimeEntropy<Result<WallInstant, TimeErr>> {
        let r = match mycelium_std_sys::time::wall_nanos() {
            // Never-silent: a count that does not fit `i128` is an explicit overflow, not a wrap.
            Ok(ns) => match i128::try_from(ns) {
                Ok(signed) => Ok(WallInstant::from_nanos_since_epoch(signed)),
                Err(_) => Err(TimeErr::Overflow),
            },
            // `std-sys::time::wall_nanos` errors only when `SystemTime::now() < UNIX_EPOCH`
            // (`duration_since` fails), so name that exact failure mode — not a generic
            // "unavailable" (the wall read itself is always reachable).
            Err(_) => Err(TimeErr::ClockUnavailable {
                reason: "OS wall clock read a time before the Unix epoch",
            }),
        };
        DeclaredTimeEntropy::new(r)
    }

    fn logical_now(&self) -> DeclaredTime<LogicalInstant> {
        // FLAG (M-356 / std-time §7-Q1): the LOGICAL tick is runtime-owned (RFC-0008 §4.7); `OsClock`
        // is not the runtime, so this is a monotonic-derived placeholder, exactly as `std-time`'s own
        // `SystemClock` stand-in. The real deterministic tick arrives with the `std.runtime`/`colony`
        // scheduler (M-356); until then a deterministic fragment should read the runtime's tick, not
        // this host clock.
        DeclaredTime::new(LogicalInstant::from_tick(
            mycelium_std_sys::time::mono_nanos(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{OsClock, OsEntropy};
    use mycelium_std_rand::{EntropyRng, EntropySource};
    use mycelium_std_time::ClockSource;

    /// `OsEntropy` fills from the OS floor, or — on a platform without `/dev/urandom` — fails
    /// explicitly (never a silent zero-fill, G2). Skips gracefully when the source is unavailable.
    #[test]
    fn os_entropy_fills_or_is_explicitly_unavailable() {
        let mut src = OsEntropy;
        let mut buf = [0u8; 32];
        match src.fill_bytes(&mut buf) {
            Ok(_) => assert!(
                buf.iter().any(|&b| b != 0),
                "entropy fill returned all-zero (smoke)"
            ),
            Err(_) => { /* no /dev/urandom on this platform — explicit, never silent. */ }
        }
    }

    /// `std-rand`'s `EntropyRng` seeds end-to-end from the OS floor through this adapter — the seam
    /// is wired. Skips gracefully when entropy is unavailable.
    #[test]
    fn entropy_rng_seeds_from_os_floor() {
        if let Ok((mut rng, _eff)) = EntropyRng::new(OsEntropy) {
            let (a, _) = rng.next_entropy();
            let (b, _) = rng.next_entropy();
            // Two draws from a real CSPRNG are astronomically unlikely to collide; a light smoke only.
            assert!(a != 0 || b != 0, "two entropy draws were both zero (smoke)");
        }
    }

    /// The monotonic clock read is total and non-decreasing across two reads (M-723).
    #[test]
    fn os_clock_mono_is_total_and_monotonic() {
        let clk = OsClock;
        let t1 = clk
            .mono_now()
            .into_inner()
            .expect("mono read is total (Ok)");
        let t2 = clk
            .mono_now()
            .into_inner()
            .expect("mono read is total (Ok)");
        assert!(
            t2.as_nanos() >= t1.as_nanos(),
            "monotonic clock went backwards"
        );
    }

    /// The wall clock executes through the floor and yields a post-epoch instant (it is `> 0` for any
    /// real system time after 1970). Never-silent: a pre-epoch/overflow read would be `Err`.
    #[test]
    fn os_clock_wall_now_executes() {
        let clk = OsClock;
        if let Ok(w) = clk.wall_now().into_inner() {
            assert!(
                w.as_nanos_since_epoch() > 0,
                "wall clock is not after the Unix epoch"
            );
        }
    }

    /// The logical read is total (the placeholder tick is always readable).
    #[test]
    fn os_clock_logical_is_total() {
        let clk = OsClock;
        let _ = clk.logical_now().into_inner();
    }
}
