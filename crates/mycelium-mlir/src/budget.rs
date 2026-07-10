//! Dynamic **depth budget** for the AOT env-machine (DN-05 §2.4 / DN05-Q5).
//!
//! The M-347 trampoline put object-level recursion on an explicit **heap** control stack
//! ([`crate::aot`]'s `Vec<Frame>`), so the *host* stack is O(1) and the residual resource a runaway
//! recursion consumes is **memory** (the control stack plus each frame's captured environment). The
//! depth ceiling that bounds it must therefore be a *policy over memory headroom*, set
//! **dynamically** — a fixed constant is either too timid on a large machine (rejecting valid
//! recursion) or too bold on a small one (risking an OOM). DN-05 §2.4 anticipated this ("*with #2 the
//! control stack is on the heap, so the budget becomes a policy over heap/work*"); this module is the
//! resolution of **DN05-Q5**.
//!
//! ## Guardrails (DN-05 §2.4 — verbatim constraints)
//! - **A small trait.** [`DepthBudget`] resolves a [`DepthResolution`]; that is the whole surface.
//! - **A conservative static fallback, never a guess.** When detection is unavailable or uncertain,
//!   [`AutoDepthBudget`] returns the prior fixed default ([`STATIC_FALLBACK_DEPTH`]).
//! - **`EXPLAIN`-able basis, no black box.** [`DepthBasis`] records *how* the number was derived
//!   (source, headroom, per-frame estimate, margin, whether it was clamped) and `Display`s as one
//!   honest line.
//! - **An explicit error, never an abort/hang.** The resolved `max_depth` is fed to the env-machine,
//!   which refuses past it with [`mycelium_interp::EvalError::DepthLimit`] — graceful, never silent.
//! - **Minimal `unsafe` (ADR-014) — here, none.** Detection reads the kernel's own accounting via
//!   pure-`std` `/proc` files (Linux): no FFI, no stack-pointer reading. On any other platform, or on
//!   any read/parse failure, it falls back. (Native-path *stack* detection for the libMLIR backend,
//!   DN05-Q4 / M-348, reuses this trait but will introduce its own — minimal — platform probe.)
//!
//! ## Honesty (VR-5)
//! The per-frame heap cost is a **conservative `Declared` estimate** (documented below,
//! caller-overridable), not a `Proven` figure: it deliberately *over*-counts so the derived depth
//! *under*-shoots affordable memory (biasing toward an early, graceful refusal rather than an OOM).
//! The detected headroom itself is `Empirical` (the kernel's live `MemAvailable` / `RLIMIT_AS`).
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use std::fmt;

/// Conservative per-frame heap estimate (bytes). Each control-stack `Frame` is a few machine words
/// plus a captured environment (a `HashMap` whose size depends on the program); 1 KiB
/// **over**-counts the common case on purpose — under-shooting the affordable depth is the safe
/// direction (refuse early, never OOM). `Declared`, caller-overridable. Public so the env-machine can
/// charge a declared `alloc` effect budget at the same per-frame rate (RFC-0014 §4.8 — the effect
/// budget is the opt-in sibling of this depth ceiling, both reasoning in per-frame bytes).
pub const DEFAULT_PER_FRAME_BYTES: u64 = 1024;
/// Fraction of detected headroom to actually spend, as a percent. 70 % leaves a generous reserve for
/// everything else the process holds (the env maps the frames point at, the result, allocator slack).
const DEFAULT_MARGIN_PCT: u8 = 70;
/// Never resolve below this — a budget too small to run useful recursion is its own footgun. The
/// floor stays well under the static fallback so a *constrained* host still gets a sane, bounded
/// ceiling rather than the (larger) fixed default.
const DEFAULT_FLOOR: usize = 10_000;
/// Never resolve above this — a hard cap so a very large machine does not get an effectively
/// unbounded ceiling (the per-frame estimate is conservative, but envs can still grow; a finite cap
/// keeps the never-silent limit *meaningful*).
const DEFAULT_CEIL: usize = 2_000_000;

/// The conservative static fallback ceiling: the prior fixed default (M-347's `AOT_MAX_DEPTH`),
/// preserved as the honest floor used when memory detection is unavailable. Public so callers and the
/// EXPLAIN surface can name it.
pub const STATIC_FALLBACK_DEPTH: usize = 200_000;

/// Resolves a control-stack **depth ceiling** for the AOT env-machine, with an inspectable basis.
///
/// The whole point of the trait (KC-3 / no bloat): the *mechanism* (the trampoline) and the *policy*
/// (how deep is safe) are separated, so the policy can be detected dynamically, fixed for tests, or —
/// later — specialised for the native path, without touching the machine.
pub trait DepthBudget {
    /// Resolve the ceiling **and** the basis it was derived from (never just a bare number — the
    /// basis is what makes it `EXPLAIN`-able / not a black box, G2).
    fn resolve(&self) -> DepthResolution;
}

/// A resolved depth ceiling plus the [`DepthBasis`] explaining how it was chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthResolution {
    /// The control-stack depth ceiling the env-machine will refuse past ([`EvalError::DepthLimit`]).
    ///
    /// [`EvalError::DepthLimit`]: mycelium_interp::EvalError::DepthLimit
    pub max_depth: usize,
    /// How `max_depth` was derived — the inspectable, `EXPLAIN`-able basis.
    pub basis: DepthBasis,
}

/// Why a [`DepthBasis::Static`] budget was used (detection not attempted, failed, or explicit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticReason {
    /// No `/proc` memory detection on this platform (only Linux is wired) — honestly deferred.
    UnsupportedPlatform,
    /// `/proc` was unreadable or unparsable — never guess; fall back.
    DetectionFailed,
    /// The caller asked for a fixed budget explicitly ([`StaticDepthBudget`]).
    Explicit,
}

/// Which kernel accounting figure the detected headroom came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemSource {
    /// `/proc/meminfo` `MemAvailable` — the kernel's estimate of allocatable memory.
    MemAvailable,
    /// `/proc/self/limits` `Max address space` (RLIMIT_AS), when finite and the tighter bound.
    AddressSpaceRlimit,
}

/// The inspectable derivation of a [`DepthResolution`] — the no-black-box record (G2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepthBasis {
    /// A fixed, conservative ceiling: detection was not attempted, failed, or was explicit.
    Static {
        /// Why the static value was used.
        reason: StaticReason,
    },
    /// Derived from detected memory headroom:
    /// `clamp(floor(headroom_bytes * margin_pct/100 / per_frame_bytes), floor, ceil)`.
    DetectedMemory {
        /// Which `/proc` figure bounded the headroom.
        source: MemSource,
        /// The detected headroom (bytes) that drove the calculation.
        headroom_bytes: u64,
        /// The per-frame heap estimate (bytes) used as the divisor.
        per_frame_bytes: u64,
        /// The safety margin (percent of headroom) applied.
        margin_pct: u8,
        /// Whether the raw derived value was clamped to `[floor, ceil]`.
        clamped: bool,
    },
}

impl fmt::Display for DepthResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.basis {
            DepthBasis::Static { reason } => {
                let why = match reason {
                    StaticReason::UnsupportedPlatform => {
                        "no /proc memory detection on this platform"
                    }
                    StaticReason::DetectionFailed => "/proc headroom unreadable/unparsable",
                    StaticReason::Explicit => "caller-specified fixed budget",
                };
                write!(f, "max_depth={} (static fallback: {why})", self.max_depth)
            }
            DepthBasis::DetectedMemory {
                source,
                headroom_bytes,
                per_frame_bytes,
                margin_pct,
                clamped,
            } => {
                let src = match source {
                    MemSource::MemAvailable => "MemAvailable",
                    MemSource::AddressSpaceRlimit => "RLIMIT_AS",
                };
                write!(
                    f,
                    "max_depth={} (detected: {src} headroom {headroom_bytes} B, \
                     ~{per_frame_bytes} B/frame, {margin_pct}% margin{})",
                    self.max_depth,
                    if *clamped { ", clamped" } else { "" },
                )
            }
        }
    }
}

/// The default policy: derive the ceiling from **detected memory headroom**, conservative fallback
/// otherwise. Fields are public so a caller may tune the estimate/margin/clamps without a new type
/// (e.g. a memory-tight embedding lowers `ceil`, a measured per-frame cost replaces the estimate).
#[derive(Clone, Copy, Debug)]
pub struct AutoDepthBudget {
    /// Per-frame heap estimate (bytes) — the divisor; over-counting is the safe direction.
    pub per_frame_bytes: u64,
    /// Fraction of detected headroom to spend (percent).
    pub margin_pct: u8,
    /// Never resolve below this.
    pub floor: usize,
    /// Never resolve above this.
    pub ceil: usize,
    /// The ceiling used when detection is unavailable/uncertain.
    pub fallback: usize,
}

impl Default for AutoDepthBudget {
    fn default() -> Self {
        Self {
            per_frame_bytes: DEFAULT_PER_FRAME_BYTES,
            margin_pct: DEFAULT_MARGIN_PCT,
            floor: DEFAULT_FLOOR,
            ceil: DEFAULT_CEIL,
            fallback: STATIC_FALLBACK_DEPTH,
        }
    }
}

impl AutoDepthBudget {
    /// The pure derivation, factored out for property-testing without touching `/proc`: spend
    /// `margin_pct`% of `headroom_bytes`, divide by the per-frame estimate, clamp to `[floor, ceil]`.
    /// Returns `(max_depth, clamped)`. Saturating throughout — no overflow, no panic.
    fn derive_from_headroom(&self, headroom_bytes: u64) -> (usize, bool) {
        let usable = (headroom_bytes / 100).saturating_mul(u64::from(self.margin_pct));
        let raw64 = usable / self.per_frame_bytes.max(1);
        let raw = usize::try_from(raw64).unwrap_or(usize::MAX);
        let clamped = raw.clamp(self.floor, self.ceil);
        (clamped, clamped != raw)
    }
}

impl DepthBudget for AutoDepthBudget {
    fn resolve(&self) -> DepthResolution {
        match detect_headroom() {
            Some((source, headroom_bytes)) => {
                let (max_depth, clamped) = self.derive_from_headroom(headroom_bytes);
                DepthResolution {
                    max_depth,
                    basis: DepthBasis::DetectedMemory {
                        source,
                        headroom_bytes,
                        per_frame_bytes: self.per_frame_bytes,
                        margin_pct: self.margin_pct,
                        clamped,
                    },
                }
            }
            None => DepthResolution {
                max_depth: self.fallback,
                basis: DepthBasis::Static {
                    reason: if cfg!(target_os = "linux") {
                        StaticReason::DetectionFailed
                    } else {
                        StaticReason::UnsupportedPlatform
                    },
                },
            },
        }
    }
}

/// An explicit, fixed depth ceiling — for tests and callers that want a deterministic budget. The
/// basis is honestly recorded as `Explicit` (not dressed up as a detection).
#[derive(Clone, Copy, Debug)]
pub struct StaticDepthBudget(pub usize);

impl DepthBudget for StaticDepthBudget {
    fn resolve(&self) -> DepthResolution {
        DepthResolution {
            max_depth: self.0,
            basis: DepthBasis::Static {
                reason: StaticReason::Explicit,
            },
        }
    }
}

/// Detect allocatable memory headroom from the kernel's own accounting (Linux `/proc`, zero
/// `unsafe`): `MemAvailable`, capped by a finite `RLIMIT_AS`. `None` ⇒ caller falls back.
#[cfg(target_os = "linux")]
fn detect_headroom() -> Option<(MemSource, u64)> {
    let avail = read_meminfo_available_bytes()?;
    match read_address_space_rlimit_bytes() {
        // A finite, tighter address-space cap (e.g. a ulimit/container constraint) wins — it is the
        // real bound on how much this process may allocate.
        Some(rlimit) if rlimit < avail => Some((MemSource::AddressSpaceRlimit, rlimit)),
        _ => Some((MemSource::MemAvailable, avail)),
    }
}

/// Non-Linux: no `/proc` probe wired — honestly fall back (never guess).
#[cfg(not(target_os = "linux"))]
fn detect_headroom() -> Option<(MemSource, u64)> {
    None
}

/// Parse `MemAvailable` (kB) from `/proc/meminfo` → bytes. Any read/parse failure ⇒ `None`.
#[cfg(target_os = "linux")]
fn read_meminfo_available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // "MemAvailable:   12345678 kB"
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Parse the soft `Max address space` (bytes) from `/proc/self/limits`; `unlimited` or any failure
/// ⇒ `None` (treated as "no finite cap").
#[cfg(target_os = "linux")]
fn read_address_space_rlimit_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/limits").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Max address space") {
            // "Max address space         <soft>   <hard>   bytes"
            let soft = rest.split_whitespace().next()?;
            if soft == "unlimited" {
                return None;
            }
            return soft.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_budget_is_honestly_labelled() {
        let r = StaticDepthBudget(64).resolve();
        assert_eq!(r.max_depth, 64);
        assert_eq!(
            r.basis,
            DepthBasis::Static {
                reason: StaticReason::Explicit
            }
        );
    }

    #[test]
    fn auto_resolves_to_a_sane_bounded_ceiling() {
        // On this host either detection succeeds (a clamped, in-range ceiling) or it falls back to the
        // conservative static default — both honest, both bounded, never zero.
        let b = AutoDepthBudget::default();
        let r = b.resolve();
        assert!(r.max_depth > 0, "a depth budget is never zero");
        match r.basis {
            DepthBasis::DetectedMemory { .. } => {
                assert!(
                    r.max_depth >= b.floor && r.max_depth <= b.ceil,
                    "detected depth {} must lie in [{}, {}]",
                    r.max_depth,
                    b.floor,
                    b.ceil
                );
            }
            DepthBasis::Static { reason } => {
                assert_eq!(r.max_depth, b.fallback);
                assert_ne!(reason, StaticReason::Explicit, "auto never claims Explicit");
            }
        }
    }

    #[test]
    fn derivation_is_bounded_and_monotone_for_all_headroom() {
        // The property of the bound (CLAUDE.md: a property test for every bound): for *any* headroom,
        // the derived depth stays within [floor, ceil] and never decreases as headroom grows. Swept
        // across decades of headroom plus the saturation extreme.
        let b = AutoDepthBudget::default();
        let mut prev = 0usize;
        let mut h = 0u64;
        for _ in 0..64 {
            let (d, clamped) = b.derive_from_headroom(h);
            assert!(
                d >= b.floor && d <= b.ceil,
                "{d} out of [{}, {}]",
                b.floor,
                b.ceil
            );
            assert!(
                d >= prev,
                "depth must be monotone non-decreasing in headroom"
            );
            if d == b.ceil {
                assert!(
                    clamped || h == 0,
                    "hitting the ceil from a positive raw means a clamp"
                );
            }
            prev = d;
            h = (h.saturating_mul(2)).max(1);
        }
        // The saturation extreme clamps to the ceiling, never overflows/panics.
        let (d, clamped) = b.derive_from_headroom(u64::MAX);
        assert_eq!(d, b.ceil);
        assert!(clamped);
    }

    #[test]
    fn a_constrained_host_scales_down_below_the_static_fallback() {
        // 256 MiB of headroom at 70% / 1 KiB per frame ≈ 183k frames — smaller than the 200k static
        // fallback, i.e. the dynamic budget *tightens* on a constrained host (the §1.1 motivation).
        let b = AutoDepthBudget::default();
        let (d, _) = b.derive_from_headroom(256 * 1024 * 1024);
        assert!(
            d < STATIC_FALLBACK_DEPTH,
            "constrained host should tighten below fallback, got {d}"
        );
        assert!(d >= b.floor);
    }

    #[test]
    fn display_explains_the_basis() {
        let detected = DepthResolution {
            max_depth: 123_456,
            basis: DepthBasis::DetectedMemory {
                source: MemSource::MemAvailable,
                headroom_bytes: 8_000_000_000,
                per_frame_bytes: 1024,
                margin_pct: 70,
                clamped: false,
            },
        };
        let s = detected.to_string();
        assert!(s.contains("max_depth=123456"));
        assert!(s.contains("MemAvailable"));
        assert!(s.contains("70% margin"));

        let fell_back = DepthResolution {
            max_depth: STATIC_FALLBACK_DEPTH,
            basis: DepthBasis::Static {
                reason: StaticReason::UnsupportedPlatform,
            },
        };
        assert!(fell_back.to_string().contains("static fallback"));
    }
}
