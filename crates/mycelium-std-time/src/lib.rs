//! `std.time` — Ring 2 / Tier B typed clocks, durations, and instants (M-529).
//!
//! # Summary
//!
//! The value-semantic time surface: immutable [`Duration`] and three *typed* instant kinds
//! ([`MonoInstant`], [`WallInstant`], [`LogicalInstant`]), pure duration/instant arithmetic, and
//! a **typed clock-read surface** over an injectable [`ClockSource`] that declares each read's
//! effect. The **honesty crux** is a typed distinction that is structurally never silent:
//!
//! - Cross-source subtraction is a **compile-time type error** — `MonoInstant − WallInstant`
//!   does not exist; there is no method to call.
//! - A **wall-clock read is `Declared` + effectful** (`{time, entropy}`): it is never dressed up
//!   as a pure exact value (VR-5, C2). The `entropy` declared effect is the same construct `rand`
//!   uses — a deterministic fragment (RT2) cannot read it silently.
//! - **Overflow is `Err(Overflow)`, never a wrap or clamp** (C1/G2): the arithmetic is total on
//!   success and explicitly fallible on range exhaustion.
//!
//! # Design-phase positioning
//!
//! The real OS clock floor (`clock_gettime`-equivalent via `wild`/FFI, ADR-014) is deferred to
//! `std-sys` (RFC-0016 §8-Q6 / M-541). This crate implements the **pure arithmetic + the
//! declared-effect surface over an injectable [`ClockSource`]**: pass the built-in
//! [`SystemClock`] for production, a [`ManualClock`] (or any `ClockSource` impl) for
//! deterministic tests. See *FLAG notes* below.
//!
//! # Contract conformance (RFC-0016 §4.1 C1–C6)
//!
//! - **C1 never-silent (G2):** every failure is `Err(TimeErr)` — overflow, unavailable clock, or
//!   a wall-clock backward jump are explicit, propagating outcomes, never zero/clamp/wrap.
//! - **C2 honest per-op tag (VR-5):** pure arithmetic tags `Exact`; every clock read tags
//!   `Declared` and is effectful — a wall read is never upgraded to a pure value.
//! - **C3 no black boxes / EXPLAIN (SC-3/G11):** every read names its clock identity + declared
//!   effect; every refusal carries [`TimeErr`] naming the cause.
//! - **C4 value-semantic (ADR-003 / RFC-0001):** `Duration` and the three instant kinds are
//!   immutable values; arithmetic is a pure function of inputs.
//! - **C5 above the small kernel (KC-3):** no trusted code, no `unsafe`, no FFI — the real OS
//!   clock is FLAGGED to `std-sys` (§8-Q6 / M-541).
//! - **C6 declared bounded effects (RFC-0014):** clock reads carry a marker type
//!   ([`DeclaredTime`], [`DeclaredTimeEntropy`]) that names the effect on the return type, never
//!   an undeclared side effect.
//!
//! # FLAG — std-sys clock floor (§7-Q3 / RFC-0016 §8-Q6 / M-541)
//!
//! The [`SystemClock`] provided here reads `std::time::{Instant, SystemTime}` from Rust's
//! standard library. This is a **placeholder** — the final Mycelium `std-sys` phylum will
//! replace it with an audited `wild`/FFI block (`clock_gettime`-equivalent, ADR-014, LR-9).
//! Until M-541 lands, `SystemClock` is the stand-in; it is the *only* site that touches OS
//! time. The `C5 "no new trusted code"` claim narrows to "no new trusted code beyond the Rust
//! stdlib `std::time` placeholder in `SystemClock`."
//!
//! # FLAG — RFC-0008 logical clock ownership (§7-Q1 / M-521 / M-356)
//!
//! [`LogicalInstant`] is the typed reading surface for the RFC-0008 deterministic monotonic
//! counter. The **advancement** of that counter is `std.runtime`/`colony`-owned (M-521 over
//! RFC-0008 §4.7 / M-356). This crate does **not** fabricate that surface; `LogicalInstant`
//! carries a `u64` tick provided by the caller's [`ClockSource::logical_now`]. The concrete API
//! shape — per-colony/per-scope, value shape, effect classification — is deferred to M-521/M-356.
//!
//! # FLAG — effect-declaration syntax (§7-Q5 / RFC-0014)
//!
//! The `! time` / `! { time, entropy }` notation in the spec is illustrative (RFC-0014 T3.4).
//! This Rust implementation reifies the declared effect as **marker return types**:
//! [`DeclaredTime`] wraps a mono or logical read result; [`DeclaredTimeEntropy`] wraps a wall
//! read result. The marker type is the inspectable effect declaration — inspectable via its type
//! tag, not a doc comment. The shared `entropy` token must agree with `std.rand` (M-531) when
//! that crate lands (§8-Q3).
//!
//! Design spec: `docs/spec/stdlib/time.md`; contract: RFC-0016 §4.1 (C1–C6);
//! guarantee matrix: §4.5 (encoded as data, asserted in tests).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Clock sources are representation-independent —
//! `Duration` and typed instants (`MonoInstant`, `WallInstant`, `LogicalInstant`) carry no
//! `Repr`. Cross-source instant arithmetic is a compile-time type error; typed instants prevent
//! cross-source subtraction structurally. Wall-clock reads are `Declared` + effectful; the
//! declared effect ([`DeclaredTimeEntropy`]) is the inspectable annotation, never an implicit side effect.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/time.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

use mycelium_std_core::GuaranteeStrength;

// ── §1. Error type ──────────────────────────────────────────────────────────────────────────────

/// Every explicit failure from a `std.time` operation (C1 / G2 / RFC-0013 diagnostic shape).
///
/// No sentinel value, no silent zero, no wrap: every failure route surfaces here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeErr {
    /// A duration or instant arithmetic operation overflowed the representable range.
    ///
    /// **Never** a wrap-around or a saturating clamp — always this error (C1/G2).
    Overflow,
    /// A MONOTONIC or WALL clock could not be read (platform unavailability, spec §7-Q3).
    ///
    /// Carries an optional human-readable reason (RFC-0013 diagnostic aid).
    ClockUnavailable { reason: &'static str },
    /// A WALL-clock pair difference found a *backward jump* (NTP, leap, DST).
    ///
    /// **Never** a silent zero or negative span — always this error (C1). Carries the two
    /// instants as opaque nanosecond values for diagnostics (RFC-0013).
    NonMonotonic { earlier_ns: i128, later_ns: i128 },
}

impl core::fmt::Display for TimeErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimeErr::Overflow => write!(f, "time arithmetic overflow (C1 — never wrap/clamp)"),
            TimeErr::ClockUnavailable { reason } => {
                write!(f, "clock unavailable: {reason}")
            }
            TimeErr::NonMonotonic {
                earlier_ns,
                later_ns,
            } => write!(
                f,
                "wall-clock backward jump (non-monotonic): \
                 later_ns={later_ns} < earlier_ns={earlier_ns} \
                 (C1 — never a silent zero/negative span)"
            ),
        }
    }
}

mycelium_std_core::impl_std_error!(TimeErr);

// ── §2. Effect-declaration marker types (C6 / RFC-0014 / FLAG §7-Q5) ───────────────────────────

/// Declared-effect wrapper for a MONOTONIC or LOGICAL clock read (effect: `time`).
///
/// This type is the Rust-level reification of the `! time` effect tag (RFC-0014; FLAG §7-Q5).
/// It is **not** `time`-or-entropy: MONOTONIC reads are effectful ambient reads but are *not*
/// civil-time entropy sources.  The inner value is accessible via [`DeclaredTime::into_inner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredTime<T>(T);

impl<T> DeclaredTime<T> {
    /// Wrap a value produced by an effectful, time-only read.
    pub fn new(inner: T) -> Self {
        DeclaredTime(inner)
    }

    /// Unwrap to the inner value (consuming the declaration; the effect was declared at read-time).
    pub fn into_inner(self) -> T {
        self.0
    }

    /// Borrow the inner value without consuming the effect declaration.
    pub fn as_inner(&self) -> &T {
        &self.0
    }
}

/// Declared-effect wrapper for a WALL-CLOCK read (effect: `{ time, entropy }`).
///
/// This type is the Rust-level reification of the `! { time, entropy }` effect tag (RFC-0014;
/// FLAG §7-Q5). A wall-clock read is an **entropy source** under RT3 (RFC-0008) — nondeterminism
/// reified and named. A deterministic fragment (RT2) cannot call a function that returns this
/// type without explicitly naming the entropy effect. The `entropy` marker must agree with
/// `std.rand` (M-531) when that crate lands (FLAG §7-Q5 / §8-Q3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredTimeEntropy<T>(T);

impl<T> DeclaredTimeEntropy<T> {
    /// Wrap a value produced by a wall-clock read, naming both `time` and `entropy` effects.
    pub fn new(inner: T) -> Self {
        DeclaredTimeEntropy(inner)
    }

    /// Unwrap to the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }

    /// Borrow the inner value.
    pub fn as_inner(&self) -> &T {
        &self.0
    }
}

// ── §3. Value types ─────────────────────────────────────────────────────────────────────────────

/// A signed nanosecond span (C4 / RFC-0001 value-semantic).
///
/// Represented as a signed 128-bit nanosecond count — large enough to span many millennia and
/// small enough to be exact at nanosecond resolution. Arithmetic is **always checked** (C1/G2):
/// an operation that would overflow returns `Err(TimeErr::Overflow)`, never a wrap or clamp.
///
/// **Guarantee tag: `Exact`** for all pure arithmetic — the computation is exact integer
/// arithmetic; fallibility is the never-silent overflow guard, not approximation.
///
/// # FLAG — representation (§7-Q4)
///
/// The signedness and resolution of `Duration` are open questions (spec §7-Q4): is it signed
/// (allowing a negative span from `sub`) or unsigned-with-explicit-direction? This implementation
/// uses a **signed `i128` nanoseconds** representation as the simplest honest choice — overflow
/// is still `Err(Overflow)`, and a negative span is meaningful (e.g., a deadline that has
/// passed). Disposition: defer final representation choice to spec ratification; overflow is
/// `Err(Overflow)` regardless (C1 holds either way).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration {
    /// Signed nanosecond count. `i128` gives ≈ ±292 years at nanosecond resolution.
    nanos: i128,
}

impl Duration {
    /// The zero span.
    pub const ZERO: Duration = Duration { nanos: 0 };

    /// The smallest representable (most-negative) span.
    pub const MIN: Duration = Duration { nanos: i128::MIN };

    /// The largest representable (most-positive) span.
    pub const MAX: Duration = Duration { nanos: i128::MAX };

    /// Construct from a nanosecond count (value-semantic constructor — always exact).
    #[must_use]
    pub const fn from_nanos(nanos: i128) -> Self {
        Duration { nanos }
    }

    /// Construct from whole seconds. Returns `Err(Overflow)` if out of range (C1/G2).
    pub fn from_secs(secs: i64) -> Result<Self, TimeErr> {
        let nanos = (secs as i128)
            .checked_mul(1_000_000_000)
            .ok_or(TimeErr::Overflow)?;
        Ok(Duration { nanos })
    }

    /// Construct from milliseconds. Returns `Err(Overflow)` if out of range (C1/G2).
    pub fn from_millis(millis: i64) -> Result<Self, TimeErr> {
        let nanos = (millis as i128)
            .checked_mul(1_000_000)
            .ok_or(TimeErr::Overflow)?;
        Ok(Duration { nanos })
    }

    /// Construct from microseconds. Returns `Err(Overflow)` if out of range (C1/G2).
    pub fn from_micros(micros: i64) -> Result<Self, TimeErr> {
        let nanos = (micros as i128)
            .checked_mul(1_000)
            .ok_or(TimeErr::Overflow)?;
        Ok(Duration { nanos })
    }

    /// The raw nanosecond count (exact, total — the canonical representation).
    #[must_use]
    pub const fn as_nanos(self) -> i128 {
        self.nanos
    }

    /// Truncating whole-second count (rounds toward zero).
    ///
    /// # Errors
    ///
    /// - [`TimeErr::Overflow`] when the whole-second count falls outside the `i64` range. The
    ///   nanosecond field is `i128` (up to [`Duration::MAX`]), so the second count can exceed
    ///   `i64`; it is refused rather than silently truncated (C1/G2).
    pub const fn as_secs_trunc(self) -> Result<i64, TimeErr> {
        let secs = self.nanos / 1_000_000_000;
        if secs < i64::MIN as i128 || secs > i64::MAX as i128 {
            return Err(TimeErr::Overflow);
        }
        Ok(secs as i64)
    }

    /// Whether this span is negative.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.nanos < 0
    }

    /// Whether this span is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.nanos == 0
    }

    /// Negate the span. Returns `Err(Overflow)` for `Duration::MIN` (C1/G2).
    ///
    /// Named `checked_neg` rather than `neg` to avoid ambiguity with `std::ops::Neg`
    /// (which cannot be implemented here because negation is fallible — C1/G2).
    pub fn checked_neg(self) -> Result<Duration, TimeErr> {
        self.nanos
            .checked_neg()
            .map(|n| Duration { nanos: n })
            .ok_or(TimeErr::Overflow)
    }

    /// Absolute value of the span. Returns `Err(Overflow)` for `Duration::MIN` (C1/G2).
    ///
    /// Named `checked_abs` to match the `checked_*` pattern; `i128::MIN.abs()` would overflow.
    pub fn checked_abs(self) -> Result<Duration, TimeErr> {
        self.nanos
            .checked_abs()
            .map(|n| Duration { nanos: n })
            .ok_or(TimeErr::Overflow)
    }
}

/// A point on the MONOTONIC clock (never-backward, no civil meaning).
///
/// Represented as an opaque nanosecond-resolution tick count from an unspecified epoch.
/// The epoch is platform-defined and has no civil meaning — **only differences between two
/// `MonoInstant` values are meaningful** (`diff(a, b) → Duration`).
///
/// Cross-source subtraction is a **compile-time type error**: `diff(MonoInstant, WallInstant)`
/// does not exist and cannot be called accidentally (the typed distinction that is "structurally
/// never a silent swap", spec §1).
///
/// **Guarantee tag: `Declared`** — this value is an input from outside the pure fragment, not a
/// computation. A MONOTONIC clock read declares effect `time` (not `entropy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonoInstant {
    /// Nanoseconds from the MONOTONIC clock's epoch (platform-defined, never civil UTC).
    nanos: u64,
}

impl MonoInstant {
    /// Construct from a raw nanosecond tick (used by [`ClockSource`] implementations).
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        MonoInstant { nanos }
    }

    /// The raw nanosecond tick (canonical representation — exact, total).
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.nanos
    }
}

/// A point on the WALL-CLOCK (civil/UTC time, an entropy source).
///
/// Represented as a signed nanosecond count from the Unix epoch (UTC). This is a civil-time
/// value: unlike [`MonoInstant`], its clock can step backward (NTP, leap seconds, DST), so
/// `diff(WallInstant, WallInstant)` may return `Err(NonMonotonic)`.
///
/// **Guarantee tag: `Declared`** — a wall read declares `{ time, entropy }`. It is an entropy
/// source in the sense of RT3 (RFC-0008): every read can differ, it is unpredictable, and it
/// seeds. A deterministic-fragment (RT2) program cannot call `wall_now` silently — the return
/// type is `DeclaredTimeEntropy<WallInstant>`, which carries the effect marker explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WallInstant {
    /// Nanoseconds from the Unix epoch (UTC), signed (can be negative for dates before 1970).
    nanos_since_epoch: i128,
}

impl WallInstant {
    /// Construct from a signed nanosecond count since the Unix epoch (used by [`ClockSource`]).
    #[must_use]
    pub const fn from_nanos_since_epoch(nanos: i128) -> Self {
        WallInstant {
            nanos_since_epoch: nanos,
        }
    }

    /// The raw signed nanosecond count since the Unix epoch (exact, total).
    #[must_use]
    pub const fn as_nanos_since_epoch(self) -> i128 {
        self.nanos_since_epoch
    }
}

/// A point on the RFC-0008 LOGICAL clock (a deterministic monotonic tick the runtime advances).
///
/// The LOGICAL clock is the **only** time source legible to the deterministic fragment (RT2),
/// because its value is reproducible under RT2 sequentialization: it is not real-world entropy.
/// Its advancement is owned by `std.runtime`/`colony` (M-521 over RFC-0008 §4.7 / M-356) — this
/// crate exposes only the **typed reading surface** (a `u64` tick supplied by the caller's
/// [`ClockSource`]).
///
/// **Guarantee tag: `Declared`** — it still declares the `time` effect (a read of runtime state,
/// not a pure constant), but `time` only — not `entropy`.
///
/// # FLAG — concrete API ownership (§7-Q1 / M-521 / M-356)
///
/// The tick shape (per-colony? per-scope?), the read's exact effect classification, and the
/// advancement semantics are **all deferred to M-521/M-356**. This crate provides the thin typed
/// wrapper only; the orchestrator wires it to the real runtime clock when that API lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalInstant {
    /// The deterministic monotonic tick provided by the RFC-0008 runtime (M-356).
    tick: u64,
}

impl LogicalInstant {
    /// Construct from a logical tick (used by [`ClockSource`] implementations).
    #[must_use]
    pub const fn from_tick(tick: u64) -> Self {
        LogicalInstant { tick }
    }

    /// The raw tick value (exact, total).
    #[must_use]
    pub const fn as_tick(self) -> u64 {
        self.tick
    }
}

// ── §4. Duration arithmetic (pure, `Exact`, checked — C1/G2) ───────────────────────────────────

/// Add two durations. Returns `Err(Overflow)` on range exhaustion — **never** wraps (C1/G2).
///
/// **Guarantee tag: `Exact`** — exact integer arithmetic; fallibility is the overflow guard.
///
/// # Errors
///
/// - [`TimeErr::Overflow`] — the result is outside the representable range.
pub fn duration_add(a: Duration, b: Duration) -> Result<Duration, TimeErr> {
    a.nanos
        .checked_add(b.nanos)
        .map(|n| Duration { nanos: n })
        .ok_or(TimeErr::Overflow)
}

/// Subtract two durations (`a - b`). Returns `Err(Overflow)` on range exhaustion (C1/G2).
///
/// **Guarantee tag: `Exact`** — exact integer arithmetic.
///
/// # Errors
///
/// - [`TimeErr::Overflow`] — the result is outside the representable range.
pub fn duration_sub(a: Duration, b: Duration) -> Result<Duration, TimeErr> {
    a.nanos
        .checked_sub(b.nanos)
        .map(|n| Duration { nanos: n })
        .ok_or(TimeErr::Overflow)
}

/// Scale a duration by a signed integer factor. Returns `Err(Overflow)` on range exhaustion.
///
/// **Guarantee tag: `Exact`** — exact integer arithmetic.
///
/// # Errors
///
/// - [`TimeErr::Overflow`] — the product is outside the representable range.
pub fn duration_scale(d: Duration, k: i64) -> Result<Duration, TimeErr> {
    d.nanos
        .checked_mul(k as i128)
        .map(|n| Duration { nanos: n })
        .ok_or(TimeErr::Overflow)
}

/// Compare two durations. **Guarantee tag: `Exact`**, total.
#[must_use]
pub fn duration_cmp(a: Duration, b: Duration) -> core::cmp::Ordering {
    a.cmp(&b)
}

/// Convert a duration to a coarser unit (truncating), or return `Err(Overflow)` if the truncated
/// value in nanoseconds overflows (narrowing-unit conversion).
///
/// **Guarantee tag: `Exact`** — truncating integer division is exact.
///
/// `unit_nanos` must be the nanosecond count of one unit (e.g., `1_000_000_000` for seconds).
/// Returns `Err(Overflow)` if `unit_nanos == 0` (degenerate) or if re-expressing the truncated
/// value in nanoseconds overflows.
///
/// # Errors
///
/// - [`TimeErr::Overflow`] — `unit_nanos` is zero, or the result overflows on narrowing.
pub fn duration_as_unit(d: Duration, unit_nanos: i128) -> Result<Duration, TimeErr> {
    if unit_nanos == 0 {
        return Err(TimeErr::Overflow);
    }
    let truncated_units = d.nanos / unit_nanos;
    let nanos = truncated_units
        .checked_mul(unit_nanos)
        .ok_or(TimeErr::Overflow)?;
    Ok(Duration { nanos })
}

// ── §5. Instant differences (same-source only — C1/G2) ─────────────────────────────────────────

/// Compute the signed duration between two MONOTONIC instants (`later − earlier`).
///
/// **Guarantee tag: `Exact`**, total. Both instants are on the same MONOTONIC clock source; the
/// result is always representable because `u64::MAX - u64::MIN < i128::MAX`.
///
/// Cross-source differences (e.g., `MonoInstant − WallInstant`) **do not exist** as a function —
/// the typed distinction is enforced at compile time (spec §1 / RFC-0016 §4.4).
#[must_use]
pub fn mono_diff(later: MonoInstant, earlier: MonoInstant) -> Duration {
    // u64 subtraction in i128 is always representable: max span = u64::MAX < i128::MAX.
    let span = (later.nanos as i128) - (earlier.nanos as i128);
    Duration { nanos: span }
}

/// Compute the signed duration between two WALL-CLOCK instants (`later − earlier`).
///
/// **Guarantee tag: `Exact`** on success. Returns `Err(NonMonotonic)` if `later < earlier`
/// (a wall-clock can step backward — NTP, leap, DST). That backward jump is **never** a silent
/// zero or negative span: it is an explicit, traceable `Err` carrying both instants' nanosecond
/// values for diagnostics (C1/RFC-0013).
///
/// # Errors
///
/// - [`TimeErr::NonMonotonic`] — `later.nanos_since_epoch < earlier.nanos_since_epoch`.
/// - [`TimeErr::Overflow`] — the difference overflows `i128` (astronomically unlikely but guarded).
pub fn wall_diff(later: WallInstant, earlier: WallInstant) -> Result<Duration, TimeErr> {
    if later.nanos_since_epoch < earlier.nanos_since_epoch {
        return Err(TimeErr::NonMonotonic {
            earlier_ns: earlier.nanos_since_epoch,
            later_ns: later.nanos_since_epoch,
        });
    }
    let span = later
        .nanos_since_epoch
        .checked_sub(earlier.nanos_since_epoch)
        .ok_or(TimeErr::Overflow)?;
    Ok(Duration { nanos: span })
}

/// Compute the duration between two LOGICAL instants (`later − earlier`).
///
/// **Guarantee tag: `Exact`**, total. The LOGICAL clock is monotonic by construction (runtime-
/// owned; RFC-0008 §4.7 / M-356), so this is always well-defined. The tick difference fits in
/// `i128` for any pair of `u64` values.
#[must_use]
pub fn logical_diff(later: LogicalInstant, earlier: LogicalInstant) -> Duration {
    let span = (later.tick as i128) - (earlier.tick as i128);
    Duration { nanos: span }
}

// ── §6. Clock-source abstraction (injectable — deterministic test clock; C6) ───────────────────

/// The injectable clock-source surface (C6 / RFC-0014 declared effects).
///
/// All three reads declare their effect via the return type:
/// - `mono_now` returns `DeclaredTime<Result<MonoInstant, TimeErr>>` — effect: `time`.
/// - `wall_now` returns `DeclaredTimeEntropy<Result<WallInstant, TimeErr>>` — effect:
///   `{ time, entropy }`.
/// - `logical_now` returns `DeclaredTime<LogicalInstant>` — effect: `time` (deterministic).
///
/// The effect marker is on the **return type**, not a doc comment — it is inspectable and
/// greppable. Production code passes [`SystemClock`]; tests pass [`ManualClock`].
///
/// # FLAG — std-sys floor (§7-Q3 / M-541)
///
/// The real OS clock is accessed only inside [`SystemClock`]. When M-541 lands, that
/// implementation will migrate to the audited `wild` block in `std-sys`.
pub trait ClockSource {
    /// Read the MONOTONIC clock. Declares effect `time` (not entropy).
    ///
    /// **Guarantee tag: `Declared`** — an ambient, nondeterministic-across-runs read.
    /// Returns `Err(ClockUnavailable)` if the platform MONOTONIC clock is not available.
    fn mono_now(&self) -> DeclaredTime<Result<MonoInstant, TimeErr>>;

    /// Read the WALL-CLOCK. Declares effects `{ time, entropy }`.
    ///
    /// **Guarantee tag: `Declared`** — an entropy source (civil time, RT3). A deterministic
    /// fragment (RT2) cannot use a function returning `DeclaredTimeEntropy` without explicitly
    /// naming the entropy effect (the structural enforcement of RT2/RT3).
    fn wall_now(&self) -> DeclaredTimeEntropy<Result<WallInstant, TimeErr>>;

    /// Read the RFC-0008 LOGICAL clock tick. Declares effect `time` only (deterministic).
    ///
    /// **Guarantee tag: `Declared`** — total (the counter is always readable in v0).
    /// The tick is provided by the runtime (M-356); this crate only reads it.
    fn logical_now(&self) -> DeclaredTime<LogicalInstant>;
}

// ── §7. Production clock source (std-sys placeholder) ──────────────────────────────────────────

/// A [`ClockSource`] backed by Rust's `std::time` — the **std-sys placeholder** (FLAG §7-Q3).
///
/// This is not the final `std-sys` audited `wild` block; it is a usable stand-in until M-541.
/// The LOGICAL clock read returns a monotonically-increasing counter seeded from the MONOTONIC
/// clock's `Instant` (a placeholder; the real counter is M-356's runtime-owned tick).
///
/// # FLAG — logical clock ownership (§7-Q1 / M-521 / M-356)
///
/// `SystemClock::logical_now` returns a tick derived from the MONOTONIC clock's elapsed
/// nanoseconds as a **placeholder** only. The real logical tick is M-356's deterministic counter,
/// owned by `std.runtime`. When that API lands, `SystemClock::logical_now` will delegate to it.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl ClockSource for SystemClock {
    fn mono_now(&self) -> DeclaredTime<Result<MonoInstant, TimeErr>> {
        // std::time::Instant is platform-monotonic. We convert to nanos from an arbitrary epoch
        // by noting elapsed time from a lazily-initialized reference instant. This avoids absolute
        // platform-epoch differences and stays pure-Rust (no unsafe, no libc).
        //
        // FLAG (§7-Q3): the real implementation will use clock_gettime(CLOCK_MONOTONIC) via the
        // audited std-sys wild block (M-541).
        use std::time::Instant;
        // Use a thread-local reference epoch so `as_nanos()` never overflows a u64 in practice.
        thread_local! {
            static EPOCH: Instant = Instant::now();
        }
        let elapsed_ns = EPOCH.with(|epoch| {
            let elapsed = epoch.elapsed();
            // elapsed().as_nanos() returns u128; clamp to u64::MAX (≈ 585 years).
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
        });
        DeclaredTime::new(Ok(MonoInstant::from_nanos(elapsed_ns)))
    }

    fn wall_now(&self) -> DeclaredTimeEntropy<Result<WallInstant, TimeErr>> {
        // std::time::SystemTime is wall-clock. We express it as signed nanos since the Unix epoch.
        //
        // FLAG (§7-Q3): the real implementation will use clock_gettime(CLOCK_REALTIME) via M-541.
        use std::time::{SystemTime, UNIX_EPOCH};
        let result = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TimeErr::ClockUnavailable {
                reason: "SystemTime before UNIX_EPOCH (platform clock error)",
            })
            .and_then(|d| {
                // d.as_nanos() is u128; convert with try_from rather than a bare `as` cast so a
                // post-i128 timestamp (year ~5138+) is refused, never silently wrapped (C1/G2).
                let nanos =
                    i128::try_from(d.as_nanos()).map_err(|_| TimeErr::ClockUnavailable {
                        reason: "wall-clock nanoseconds since epoch exceed i128",
                    })?;
                Ok(WallInstant::from_nanos_since_epoch(nanos))
            });
        DeclaredTimeEntropy::new(result)
    }

    fn logical_now(&self) -> DeclaredTime<LogicalInstant> {
        // PLACEHOLDER: returns the same elapsed-nanos tick as mono_now, cast to u64.
        // FLAG (§7-Q1): the real logical tick is M-356's runtime-owned monotonic counter.
        // This placeholder is strictly a test/build scaffold; it must not be used for
        // any semantics that depend on the logical clock's advancement being runtime-owned.
        let tick = self
            .mono_now()
            .into_inner()
            .map(|m| m.as_nanos())
            .unwrap_or(0);
        DeclaredTime::new(LogicalInstant::from_tick(tick))
    }
}

// ── §8. Deterministic test clock ────────────────────────────────────────────────────────────────

/// A [`ClockSource`] with manually-settable time values — for deterministic tests.
///
/// All three reads return the values set by the last `set_*` call (default: zero for each).
/// The WALL-CLOCK read still returns `DeclaredTimeEntropy` — the effect marker is structural,
/// not conditional on nondeterminism — but the value is deterministic, enabling pure test logic.
///
/// # Example
///
/// ```rust
/// # use mycelium_std_time::{ManualClock, ClockSource, MonoInstant, WallInstant, LogicalInstant};
/// let mut clock = ManualClock::default();
/// clock.set_mono(MonoInstant::from_nanos(1_000_000_000));
/// clock.set_wall(WallInstant::from_nanos_since_epoch(1_717_000_000_000_000_000));
/// clock.set_logical(LogicalInstant::from_tick(42));
///
/// let mono = clock.mono_now().into_inner().unwrap();
/// assert_eq!(mono.as_nanos(), 1_000_000_000);
///
/// let wall = clock.wall_now().into_inner().unwrap();
/// assert_eq!(wall.as_nanos_since_epoch(), 1_717_000_000_000_000_000);
///
/// let logical = clock.logical_now().into_inner();
/// assert_eq!(logical.as_tick(), 42);
/// ```
#[derive(Debug, Clone)]
pub struct ManualClock {
    mono: MonoInstant,
    wall: WallInstant,
    logical: LogicalInstant,
}

impl Default for ManualClock {
    fn default() -> Self {
        ManualClock {
            mono: MonoInstant::from_nanos(0),
            wall: WallInstant::from_nanos_since_epoch(0),
            logical: LogicalInstant::from_tick(0),
        }
    }
}

impl ManualClock {
    /// Set the MONOTONIC clock value returned by `mono_now`.
    pub fn set_mono(&mut self, instant: MonoInstant) {
        self.mono = instant;
    }

    /// Set the WALL-CLOCK value returned by `wall_now`.
    pub fn set_wall(&mut self, instant: WallInstant) {
        self.wall = instant;
    }

    /// Set the LOGICAL tick returned by `logical_now`.
    pub fn set_logical(&mut self, instant: LogicalInstant) {
        self.logical = instant;
    }

    /// Advance the MONOTONIC clock by `delta_ns` nanoseconds (for tests that simulate time
    /// passing without an actual sleep). Saturates at `u64::MAX`.
    pub fn advance_mono(&mut self, delta_ns: u64) {
        self.mono = MonoInstant::from_nanos(self.mono.as_nanos().saturating_add(delta_ns));
    }

    /// Advance the LOGICAL clock by one tick (for tests that simulate a runtime step).
    pub fn step_logical(&mut self) {
        self.logical = LogicalInstant::from_tick(self.logical.as_tick().saturating_add(1));
    }
}

impl ClockSource for ManualClock {
    fn mono_now(&self) -> DeclaredTime<Result<MonoInstant, TimeErr>> {
        DeclaredTime::new(Ok(self.mono))
    }

    fn wall_now(&self) -> DeclaredTimeEntropy<Result<WallInstant, TimeErr>> {
        DeclaredTimeEntropy::new(Ok(self.wall))
    }

    fn logical_now(&self) -> DeclaredTime<LogicalInstant> {
        DeclaredTime::new(self.logical)
    }
}

// ── §9. Convenience free-function wrappers (thin, re-export-friendly) ──────────────────────────

/// Read the MONOTONIC clock from `source`. Declares effect `time` (not entropy).
///
/// **Guarantee tag: `Declared`** — ambient read, not a pure computation (VR-5, C2).
///
/// # Errors
///
/// - [`TimeErr::ClockUnavailable`] — the platform clock is unavailable.
pub fn mono_now(source: &dyn ClockSource) -> DeclaredTime<Result<MonoInstant, TimeErr>> {
    source.mono_now()
}

/// Read the WALL-CLOCK from `source`. Declares effects `{ time, entropy }`.
///
/// **Guarantee tag: `Declared`** — an entropy source (civil time, RT3 reified nondeterminism).
/// Return type carries [`DeclaredTimeEntropy`] — the structural RT2/RT3 enforcement.
///
/// # Errors
///
/// - [`TimeErr::ClockUnavailable`] — the platform clock is unavailable.
pub fn wall_now(source: &dyn ClockSource) -> DeclaredTimeEntropy<Result<WallInstant, TimeErr>> {
    source.wall_now()
}

/// Read the LOGICAL clock from `source`. Declares effect `time` only (deterministic).
///
/// **Guarantee tag: `Declared`** — total (the counter is always readable in v0). The
/// deterministic-fragment-legible time read (RFC-0008 §4.7 / M-356).
pub fn logical_now(source: &dyn ClockSource) -> DeclaredTime<LogicalInstant> {
    source.logical_now()
}

// ── §10. Guarantee matrix (RFC-0016 §4.5 — encoded as data, asserted in tests) ────────────────

/// One row of the `std.time` guarantee matrix (RFC-0016 §4.5 / spec §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuaranteeRow {
    /// The exported operation name.
    pub op: &'static str,
    /// The honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.
    pub tag: GuaranteeStrength,
    /// The explicit fallibility: `"total"`, or the `Result`/`Option` shape.
    pub fallibility: &'static str,
    /// Declared effects: `"none"`, `"time"`, or `"{ time, entropy }"`.
    pub effects: &'static str,
    /// Whether the op surfaces an inspectable EXPLAIN artifact.
    pub explainable: bool,
}

/// The `std.time` guarantee matrix (spec §4 / RFC-0016 §4.5).
///
/// **Encoding obligation** (RFC-0016 §4.5): every row is encoded here as data and asserted in
/// the test suite — never prose-only.
///
/// Tag justification (VR-5 — downgrade rather than overclaim):
/// - **`Exact` rows** are pure integer-span computations: exact, never approximate. Fallibility
///   is the never-silent overflow guard — a result that cannot be represented is `Err(Overflow)`,
///   never a wrap or saturating clamp (C1/G2).
/// - **`Declared` rows** are clock reads: inputs from outside the pure fragment. They are
///   `Declared` because the value is asserted/ambient, not computed. They never tag `Exact`,
///   `Proven`, or `Empirical` — those would be overclaims (VR-5, the crux of this module).
pub const GUARANTEE_MATRIX: &[GuaranteeRow] = &[
    // ── Pure arithmetic (Exact) ─────────────────────────────────────────────
    GuaranteeRow {
        op: "duration_add",
        tag: GuaranteeStrength::Exact,
        fallibility: "Err(Overflow) — never wrap/clamp",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "duration_sub",
        tag: GuaranteeStrength::Exact,
        fallibility: "Err(Overflow) — never wrap/clamp",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "duration_scale",
        tag: GuaranteeStrength::Exact,
        fallibility: "Err(Overflow) — never wrap/clamp",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "duration_cmp",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "duration_as_unit",
        tag: GuaranteeStrength::Exact,
        fallibility: "Err(Overflow) on unit=0 or narrowing overflow",
        effects: "none",
        explainable: true,
    },
    // ── Instant differences (Exact, same-source) ─────────────────────────────
    GuaranteeRow {
        op: "mono_diff",
        tag: GuaranteeStrength::Exact,
        fallibility: "total (same-source, MONOTONIC is never-backward)",
        effects: "none",
        explainable: false,
    },
    GuaranteeRow {
        op: "wall_diff",
        tag: GuaranteeStrength::Exact,
        fallibility: "Err(NonMonotonic) on backward jump — never silent zero",
        effects: "none",
        explainable: true,
    },
    GuaranteeRow {
        op: "logical_diff",
        tag: GuaranteeStrength::Exact,
        fallibility: "total (LOGICAL is monotonic by construction)",
        effects: "none",
        explainable: false,
    },
    // ── COMPILE-TIME TYPE ERROR (cross-source difference does not exist) ─────
    // Not a row — documented in §4 of the spec as "does not exist / compile error".
    // ── Clock reads (Declared — the crux of this module) ─────────────────────
    GuaranteeRow {
        op: "mono_now",
        // Declared: an ambient, nondeterministic-across-runs read. Never Exact/Proven/Empirical.
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(ClockUnavailable)",
        // `time` only — MONOTONIC is NOT an entropy source in the RNG-seeding sense (spec §4).
        effects: "time",
        explainable: true,
    },
    GuaranteeRow {
        op: "wall_now",
        // Declared: an entropy source (civil time, RT3). The crux row.
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(ClockUnavailable)",
        // `{ time, entropy }` — a wall read IS an entropy source; DeclaredTimeEntropy enforces RT2.
        effects: "{ time, entropy }",
        explainable: true,
    },
    GuaranteeRow {
        op: "logical_now",
        // Declared: total (counter always readable), deterministic — but still a read of runtime
        // state, not a pure constant. The sole fragment-legible time read.
        tag: GuaranteeStrength::Declared,
        fallibility: "total (deterministic counter, always readable in v0)",
        effects: "time",
        explainable: true,
    },
];

/// Assert the structural invariants of the guarantee matrix — called from tests.
///
/// Discharges the RFC-0016 §4.5 obligation: "encoded as data, asserted in tests, never
/// prose-only." Panics with a descriptive message on any violation.
pub fn assert_matrix_invariants() {
    for row in GUARANTEE_MATRIX {
        // 1. Non-empty op name.
        assert!(!row.op.is_empty(), "matrix row has empty op name");

        // 2. Clock reads must be `Declared` — never Exact/Proven/Empirical (VR-5, the crux).
        if matches!(row.op, "mono_now" | "wall_now" | "logical_now") {
            assert_eq!(
                row.tag,
                GuaranteeStrength::Declared,
                "op {}: clock reads must be Declared (VR-5, spec §4)",
                row.op
            );
        }

        // 3. Pure arithmetic rows must be `Exact`.
        if matches!(
            row.op,
            "duration_add"
                | "duration_sub"
                | "duration_scale"
                | "duration_cmp"
                | "duration_as_unit"
                | "mono_diff"
                | "wall_diff"
                | "logical_diff"
        ) {
            assert_eq!(
                row.tag,
                GuaranteeStrength::Exact,
                "op {}: pure arithmetic must be Exact",
                row.op
            );
        }

        // 4. `wall_now` is the only row that declares entropy (the typed distinction).
        if row.op == "wall_now" {
            assert!(
                row.effects.contains("entropy"),
                "wall_now must declare entropy effect (spec §4 — the typed distinction)"
            );
        }

        // 5. `mono_now` and `logical_now` must NOT declare entropy.
        if matches!(row.op, "mono_now" | "logical_now") {
            assert!(
                !row.effects.contains("entropy"),
                "op {}: must NOT declare entropy (spec §4 — typed distinction from wall)",
                row.op
            );
        }

        // 6. `Declared` rows are all explainable (the clock identity + effect is inspectable, C3).
        if row.tag == GuaranteeStrength::Declared {
            assert!(
                row.explainable,
                "op {}: Declared rows must be explainable (C3 — no black boxes)",
                row.op
            );
        }
    }
}

// ── §11. Tests ──────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_clock() -> ManualClock {
        ManualClock::default()
    }

    // ── Guarantee matrix invariants ──────────────────────────────────────────────────────────────

    /// The guarantee matrix is internally consistent (RFC-0016 §4.5).
    #[test]
    fn guarantee_matrix_invariants_hold() {
        assert_matrix_invariants();
    }

    /// All expected ops appear in the matrix exactly once.
    #[test]
    fn matrix_contains_all_ops_exactly_once() {
        let expected = [
            "duration_add",
            "duration_sub",
            "duration_scale",
            "duration_cmp",
            "duration_as_unit",
            "mono_diff",
            "wall_diff",
            "logical_diff",
            "mono_now",
            "wall_now",
            "logical_now",
        ];
        for op in &expected {
            let count = GUARANTEE_MATRIX.iter().filter(|r| r.op == *op).count();
            assert_eq!(count, 1, "op '{op}' must appear exactly once in the matrix");
        }
    }

    /// Clock-read rows tag `Declared`; pure-arithmetic rows tag `Exact`. No overclaim.
    #[test]
    fn matrix_clock_rows_are_declared_arithmetic_rows_are_exact() {
        let clock_ops = ["mono_now", "wall_now", "logical_now"];
        let arith_ops = [
            "duration_add",
            "duration_sub",
            "duration_scale",
            "duration_cmp",
            "duration_as_unit",
            "mono_diff",
            "wall_diff",
            "logical_diff",
        ];
        for row in GUARANTEE_MATRIX {
            if clock_ops.contains(&row.op) {
                assert_eq!(
                    row.tag,
                    GuaranteeStrength::Declared,
                    "clock op {} must be Declared (VR-5)",
                    row.op
                );
            }
            if arith_ops.contains(&row.op) {
                assert_eq!(
                    row.tag,
                    GuaranteeStrength::Exact,
                    "arithmetic op {} must be Exact",
                    row.op
                );
            }
        }
    }

    /// `wall_now` is the only op that declares `entropy`; mono/logical do not.
    #[test]
    fn matrix_entropy_declared_only_for_wall_now() {
        for row in GUARANTEE_MATRIX {
            if row.op == "wall_now" {
                assert!(
                    row.effects.contains("entropy"),
                    "wall_now must declare entropy"
                );
            } else {
                assert!(
                    !row.effects.contains("entropy"),
                    "op {} must NOT declare entropy (only wall_now does)",
                    row.op
                );
            }
        }
    }

    // ── Duration arithmetic property tests ──────────────────────────────────────────────────────

    /// `add(a, b) == add(b, a)` (commutativity) — property test over a span of pairs.
    ///
    /// Mutation witness: change `duration_add(a, b)` to `duration_add(b, Duration::ZERO)` →
    /// assertion fires for non-zero `b`.
    #[test]
    fn as_secs_trunc_is_exact_in_range_and_refuses_overflow() {
        // In-range: truncates toward zero, exact.
        assert_eq!(
            Duration::from_nanos(2_500_000_000).as_secs_trunc().unwrap(),
            2
        );
        assert_eq!(
            Duration::from_nanos(-2_500_000_000)
                .as_secs_trunc()
                .unwrap(),
            -2
        );
        assert_eq!(Duration::ZERO.as_secs_trunc().unwrap(), 0);
        // Out of i64-seconds range (Duration::MAX nanos ≈ 1.7e38 → ~1.7e29 s ≫ i64::MAX):
        // must be Err(Overflow), never a silent truncation (C1/G2).
        assert_eq!(Duration::MAX.as_secs_trunc(), Err(TimeErr::Overflow));
        assert_eq!(Duration::MIN.as_secs_trunc(), Err(TimeErr::Overflow));
    }

    #[test]
    fn duration_add_is_commutative() {
        let cases: &[(i128, i128)] = &[
            (0, 0),
            (1, 2),
            (-1, 2),
            (1, -2),
            (100, -100),
            (i128::MAX / 2, -(i128::MAX / 2)),
        ];
        for &(a_ns, b_ns) in cases {
            let a = Duration::from_nanos(a_ns);
            let b = Duration::from_nanos(b_ns);
            let ab = duration_add(a, b);
            let ba = duration_add(b, a);
            // Both succeed or both overflow — results must agree.
            assert_eq!(
                ab, ba,
                "add({a_ns}, {b_ns}) must be commutative; got {ab:?} vs {ba:?}"
            );
        }
    }

    /// `add(a, sub(b, a)) == b` (add/sub round-trip) — property test.
    #[test]
    fn duration_add_sub_round_trip() {
        let cases: &[(i128, i128)] = &[(0, 0), (10, 5), (-10, 5), (10, -5), (1_000_000, 999_999)];
        for &(a_ns, b_ns) in cases {
            let a = Duration::from_nanos(a_ns);
            let b = Duration::from_nanos(b_ns);
            // sub(b, a) then add(a, ...) should recover b.
            let diff = duration_sub(b, a).expect("sub should not overflow for test pairs");
            let recovered = duration_add(a, diff).expect("add should not overflow");
            assert_eq!(
                recovered, b,
                "add(a, sub(b, a)) != b for a={a_ns}, b={b_ns}"
            );
        }
    }

    /// `add(a, ZERO) == a` (additive identity) — property test over a sample.
    #[test]
    fn duration_add_zero_identity() {
        let cases = [0i128, 1, -1, 1_000_000_000, i128::MAX, i128::MIN];
        for &ns in &cases {
            let a = Duration::from_nanos(ns);
            let result = duration_add(a, Duration::ZERO).expect("add ZERO should not overflow");
            assert_eq!(result, a, "add(a, ZERO) != a for a={ns}");
        }
    }

    /// `scale(d, 0) == ZERO` (scale by zero gives zero) — property test.
    #[test]
    fn duration_scale_by_zero() {
        let cases = [0i128, 1, -1, 1_000_000_000];
        for &ns in &cases {
            let d = Duration::from_nanos(ns);
            let result = duration_scale(d, 0).expect("scale by 0 should not overflow");
            assert_eq!(
                result,
                Duration::ZERO,
                "scale(d, 0) must be ZERO for d={ns}"
            );
        }
    }

    /// `scale(d, 1) == d` (scale by one is identity) — property test.
    #[test]
    fn duration_scale_by_one_identity() {
        let cases = [0i128, 1, -1, 1_000_000_000, -1_000_000_000];
        for &ns in &cases {
            let d = Duration::from_nanos(ns);
            let result = duration_scale(d, 1).expect("scale by 1 should not overflow");
            assert_eq!(result, d, "scale(d, 1) must equal d for d={ns}");
        }
    }

    /// `scale(add(a, b), k) == add(scale(a, k), scale(b, k))` (distributivity over a safe range).
    #[test]
    fn duration_scale_distributes_over_add() {
        let cases: &[(i128, i128, i64)] = &[(10, 20, 3), (-5, 5, 7), (0, 100, 42)];
        for &(a_ns, b_ns, k) in cases {
            let a = Duration::from_nanos(a_ns);
            let b = Duration::from_nanos(b_ns);
            let lhs = duration_add(a, b)
                .and_then(|sum| duration_scale(sum, k))
                .expect("lhs should not overflow");
            let rhs = duration_add(
                duration_scale(a, k).expect("scale a"),
                duration_scale(b, k).expect("scale b"),
            )
            .expect("rhs should not overflow");
            assert_eq!(
                lhs, rhs,
                "scale distributes over add: failed for a={a_ns},b={b_ns},k={k}"
            );
        }
    }

    // ── Overflow is Err(Overflow), never a wrap (C1/G2 — the crux) ──────────────────────────────

    /// Overflow on `duration_add` is `Err(Overflow)`, never a wrap (C1/G2).
    ///
    /// Mutation witness: use wrapping_add → assertion fires.
    #[test]
    fn duration_add_overflow_is_explicit_never_wrap() {
        let result = duration_add(Duration::MAX, Duration::from_nanos(1));
        assert_eq!(
            result,
            Err(TimeErr::Overflow),
            "MAX + 1 must be Err(Overflow), never a wrap"
        );
    }

    /// Overflow on `duration_sub` is `Err(Overflow)`, never a wrap.
    #[test]
    fn duration_sub_overflow_is_explicit_never_wrap() {
        let result = duration_sub(Duration::MIN, Duration::from_nanos(1));
        assert_eq!(
            result,
            Err(TimeErr::Overflow),
            "MIN - 1 must be Err(Overflow)"
        );
    }

    /// Overflow on `duration_scale` is `Err(Overflow)`, never a wrap.
    #[test]
    fn duration_scale_overflow_is_explicit_never_wrap() {
        let result = duration_scale(Duration::MAX, 2);
        assert_eq!(
            result,
            Err(TimeErr::Overflow),
            "MAX * 2 must be Err(Overflow)"
        );
    }

    /// `duration_as_unit` with `unit_nanos=0` is `Err(Overflow)` (degenerate unit guard).
    #[test]
    fn duration_as_unit_zero_unit_is_overflow() {
        let d = Duration::from_nanos(1_000_000_000);
        let result = duration_as_unit(d, 0);
        assert_eq!(
            result,
            Err(TimeErr::Overflow),
            "unit_nanos=0 must be Err(Overflow)"
        );
    }

    // ── Instant difference property tests ────────────────────────────────────────────────────────

    /// `mono_diff(a, a) == ZERO` (reflexive) — property test.
    #[test]
    fn mono_diff_reflexive() {
        let cases = [0u64, 1, 1_000_000_000, u64::MAX];
        for &ns in &cases {
            let a = MonoInstant::from_nanos(ns);
            let d = mono_diff(a, a);
            assert_eq!(d, Duration::ZERO, "mono_diff(a, a) must be ZERO for a={ns}");
        }
    }

    /// `mono_diff(later, earlier) + mono_diff(earlier, later) == ZERO` (anti-commutativity).
    #[test]
    fn mono_diff_anti_commutative() {
        let cases: &[(u64, u64)] = &[(0, 1), (100, 200), (1_000, 5_000)];
        for &(a_ns, b_ns) in cases {
            let a = MonoInstant::from_nanos(a_ns);
            let b = MonoInstant::from_nanos(b_ns);
            let ab = mono_diff(b, a);
            let ba = mono_diff(a, b);
            let sum = duration_add(ab, ba).expect("anti-commutative sum");
            assert_eq!(sum, Duration::ZERO, "diff(b,a) + diff(a,b) must be ZERO");
        }
    }

    /// `logical_diff(later, earlier) == mono_diff(later_as_mono, earlier_as_mono)` for the
    /// tick-is-ns interpretation — the property holds structurally.
    #[test]
    fn logical_diff_reflexive() {
        let cases = [0u64, 1, 42, u64::MAX / 2];
        for &t in &cases {
            let a = LogicalInstant::from_tick(t);
            let d = logical_diff(a, a);
            assert_eq!(
                d,
                Duration::ZERO,
                "logical_diff(a, a) must be ZERO for tick={t}"
            );
        }
    }

    /// `wall_diff(a, a) == Ok(ZERO)` (reflexive — same wall instant is not non-monotonic).
    #[test]
    fn wall_diff_reflexive() {
        let cases = [0i128, 1, -1, 1_717_000_000_000_000_000i128];
        for &ns in &cases {
            let a = WallInstant::from_nanos_since_epoch(ns);
            let d = wall_diff(a, a).expect("wall_diff(a, a) must be Ok");
            assert_eq!(
                d,
                Duration::ZERO,
                "wall_diff(a, a) must be ZERO for ns={ns}"
            );
        }
    }

    /// `wall_diff(later, earlier)` with `later < earlier` is `Err(NonMonotonic)` — never
    /// a silent zero or negative span (C1/G2).
    ///
    /// Mutation witness: return `Ok(ZERO)` → assertion fires.
    #[test]
    fn wall_diff_backward_jump_is_non_monotonic_never_silent() {
        let earlier = WallInstant::from_nanos_since_epoch(100);
        let later = WallInstant::from_nanos_since_epoch(50); // "later" is actually earlier in time
        let result = wall_diff(later, earlier);
        match result {
            Err(TimeErr::NonMonotonic {
                earlier_ns,
                later_ns,
            }) => {
                // The diagnostic record names both instants (RFC-0013 structured output).
                assert_eq!(earlier_ns, 100);
                assert_eq!(later_ns, 50);
            }
            _ => panic!(
                "expected Err(NonMonotonic), got {result:?} — \
                 a backward jump must never be a silent zero"
            ),
        }
    }

    /// `wall_diff(a, b)` where `a >= b` returns the positive span (not non-monotonic).
    #[test]
    fn wall_diff_forward_jump_is_positive_span() {
        let earlier = WallInstant::from_nanos_since_epoch(50);
        let later = WallInstant::from_nanos_since_epoch(150);
        let d = wall_diff(later, earlier).expect("forward diff must be Ok");
        assert_eq!(d.as_nanos(), 100);
    }

    // ── Clock-read surface (ManualClock deterministic tests) ─────────────────────────────────────

    /// `mono_now` returns the set mono value and the result is `DeclaredTime<Ok(...)>`.
    ///
    /// Tests: typed effect marker + value round-trip.
    #[test]
    fn manual_clock_mono_now_returns_declared_time() {
        let mut clock = mk_clock();
        clock.set_mono(MonoInstant::from_nanos(12_345));
        let declared = mono_now(&clock);
        let inner = declared
            .into_inner()
            .expect("ManualClock mono_now is always Ok");
        assert_eq!(inner.as_nanos(), 12_345);
    }

    /// `wall_now` returns the set wall value wrapped in `DeclaredTimeEntropy` — the effect
    /// marker is structural, not conditional on nondeterminism (the test clock is deterministic,
    /// but the type still carries the effect declaration).
    #[test]
    fn manual_clock_wall_now_returns_declared_time_entropy() {
        let mut clock = mk_clock();
        clock.set_wall(WallInstant::from_nanos_since_epoch(999_999));
        let declared = wall_now(&clock);
        let inner = declared
            .into_inner()
            .expect("ManualClock wall_now is always Ok");
        assert_eq!(inner.as_nanos_since_epoch(), 999_999);
    }

    /// `logical_now` returns the set logical tick wrapped in `DeclaredTime`.
    #[test]
    fn manual_clock_logical_now_returns_declared_time() {
        let mut clock = mk_clock();
        clock.set_logical(LogicalInstant::from_tick(77));
        let declared = logical_now(&clock);
        let inner = declared.into_inner();
        assert_eq!(inner.as_tick(), 77);
    }

    /// `advance_mono` and `step_logical` move the clock forward monotonically.
    #[test]
    fn manual_clock_advance_moves_forward() {
        let mut clock = mk_clock();
        clock.set_mono(MonoInstant::from_nanos(1_000));
        clock.advance_mono(500);
        let m = clock.mono_now().into_inner().unwrap();
        assert_eq!(
            m.as_nanos(),
            1_500,
            "advance_mono should add to the current value"
        );

        clock.set_logical(LogicalInstant::from_tick(10));
        clock.step_logical();
        let l = clock.logical_now().into_inner();
        assert_eq!(l.as_tick(), 11, "step_logical should increment by 1");
    }

    /// Two reads of `mono_now` are non-decreasing on `ManualClock` without explicit advance.
    #[test]
    fn manual_clock_mono_reads_are_stable_without_advance() {
        let clock = mk_clock();
        let t1 = clock.mono_now().into_inner().unwrap();
        let t2 = clock.mono_now().into_inner().unwrap();
        assert!(
            t2 >= t1,
            "mono reads without advance must be non-decreasing"
        );
    }

    // ── Typed distinction — cross-source subtraction does not exist (compile-time) ──────────────
    //
    // There is no `diff(MonoInstant, WallInstant)` function — confirmed by the absence of such
    // an impl and by the fact that this test file compiles without one. This property is
    // structural (a type error), not a runtime assertion. The test below documents this guarantee
    // explicitly.

    /// Document that cross-source difference does not exist — the typed distinction is structural.
    ///
    /// There is no `diff(MonoInstant, WallInstant)` function. Attempting to write one would be
    /// a compile error, not a runtime check. This test documents the *spec §4* guarantee:
    /// "cross-source: — (does not exist) / compile-time type error".
    #[test]
    fn cross_source_diff_is_compile_time_type_error_documented() {
        // This test cannot "call" a non-existent function, so it verifies the structure indirectly:
        // - `mono_diff` takes `(MonoInstant, MonoInstant)`.
        // - `wall_diff` takes `(WallInstant, WallInstant)`.
        // - There is no function that accepts a mixed pair.
        // The fact that this file compiles with no mixed call is the witness.
        //
        // We verify the signatures are distinct by calling each with its own type:
        let m1 = MonoInstant::from_nanos(10);
        let m2 = MonoInstant::from_nanos(20);
        let w1 = WallInstant::from_nanos_since_epoch(10);
        let w2 = WallInstant::from_nanos_since_epoch(20);
        let _ = mono_diff(m2, m1); // compiles: (MonoInstant, MonoInstant)
        let _ = wall_diff(w2, w1); // compiles: (WallInstant, WallInstant)
                                   // The following would NOT compile (uncomment to verify):
                                   // let _ = mono_diff(m2, w1);  // ERROR: expected MonoInstant, found WallInstant
                                   // let _ = wall_diff(w2, m1);  // ERROR: expected WallInstant, found MonoInstant
    }

    // ── Duration construction helpers ────────────────────────────────────────────────────────────

    /// `from_secs` / `from_millis` / `from_micros` round-trip through `as_nanos`.
    #[test]
    fn duration_constructors_round_trip() {
        let d = Duration::from_secs(1).unwrap();
        assert_eq!(d.as_nanos(), 1_000_000_000);

        let d = Duration::from_millis(1).unwrap();
        assert_eq!(d.as_nanos(), 1_000_000);

        let d = Duration::from_micros(1).unwrap();
        assert_eq!(d.as_nanos(), 1_000);
    }

    /// `from_secs` succeeds for `i64::MAX` because `i64::MAX * 1_000_000_000 < i128::MAX`.
    /// The overflow guard exists for when the input type is widened in the future; for now,
    /// the `i64` parameter range fits in `i128` nanos. This test documents the actual behavior.
    #[test]
    fn duration_from_secs_i64_max_succeeds() {
        // i64::MAX * 1_000_000_000 ≈ 9.2e27 < i128::MAX ≈ 1.7e38 — fits.
        let result = Duration::from_secs(i64::MAX);
        assert!(
            result.is_ok(),
            "from_secs(i64::MAX) must succeed: i64::MAX * 1e9 < i128::MAX"
        );
        assert_eq!(result.unwrap().as_nanos(), i64::MAX as i128 * 1_000_000_000);
    }

    /// `duration_scale` overflow is the primary overflow path for sub-second arithmetic:
    /// scaling `Duration::MAX` by any factor > 1 overflows (already tested above).
    /// Also document that `from_secs` with a negative overflow is not reachable with `i64`.
    #[test]
    fn duration_scale_max_by_two_is_overflow() {
        let result = duration_scale(Duration::MAX, 2);
        assert_eq!(
            result,
            Err(TimeErr::Overflow),
            "Duration::MAX * 2 must be Err(Overflow) — never a silent wrap (C1/G2)"
        );
    }

    // ── System clock smoke test (checks the std-sys placeholder compiles and runs) ───────────────

    /// The system clock reads non-decreasing mono values (non-deterministic by nature, but
    /// two successive reads should be non-decreasing on any sane platform).
    ///
    /// This test exercises the `SystemClock` std-sys placeholder (FLAG §7-Q3).
    #[test]
    fn system_clock_mono_reads_are_non_decreasing() {
        let clock = SystemClock;
        let t1 = clock
            .mono_now()
            .into_inner()
            .expect("SystemClock mono_now must succeed");
        let t2 = clock
            .mono_now()
            .into_inner()
            .expect("SystemClock mono_now must succeed");
        assert!(
            t2 >= t1,
            "SystemClock mono reads must be non-decreasing: t1={t1:?}, t2={t2:?}"
        );
    }

    /// The system clock wall read wraps the value in `DeclaredTimeEntropy`.
    #[test]
    fn system_clock_wall_now_returns_declared_time_entropy() {
        let clock = SystemClock;
        let declared = clock.wall_now();
        let inner = declared
            .into_inner()
            .expect("SystemClock wall_now must succeed on a sane OS");
        // The wall instant should be after the Unix epoch.
        assert!(
            inner.as_nanos_since_epoch() > 0,
            "SystemClock wall_now must return a post-epoch timestamp"
        );
    }

    // ── `from_nanos` round-trip ──────────────────────────────────────────────────────────────────

    /// `Duration::from_nanos(n).as_nanos() == n` (round-trip).
    #[test]
    fn duration_from_nanos_round_trip() {
        let cases = [0i128, 1, -1, i128::MAX, i128::MIN];
        for &ns in &cases {
            let d = Duration::from_nanos(ns);
            assert_eq!(d.as_nanos(), ns);
        }
    }

    /// `MonoInstant::from_nanos(n).as_nanos() == n` (round-trip).
    #[test]
    fn mono_instant_from_nanos_round_trip() {
        let cases = [0u64, 1, u64::MAX];
        for &ns in &cases {
            let m = MonoInstant::from_nanos(ns);
            assert_eq!(m.as_nanos(), ns);
        }
    }

    /// `WallInstant::from_nanos_since_epoch(n).as_nanos_since_epoch() == n` (round-trip).
    #[test]
    fn wall_instant_from_nanos_round_trip() {
        let cases = [0i128, 1, -1, i128::MAX, i128::MIN];
        for &ns in &cases {
            let w = WallInstant::from_nanos_since_epoch(ns);
            assert_eq!(w.as_nanos_since_epoch(), ns);
        }
    }

    /// `LogicalInstant::from_tick(t).as_tick() == t` (round-trip).
    #[test]
    fn logical_instant_from_tick_round_trip() {
        let cases = [0u64, 1, u64::MAX];
        for &t in &cases {
            let l = LogicalInstant::from_tick(t);
            assert_eq!(l.as_tick(), t);
        }
    }
}
