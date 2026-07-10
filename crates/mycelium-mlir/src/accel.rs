//! **BitNet-class packed-ternary acceleration — the explicit capability gate** (M-728; E15-1;
//! FR-C3; RFC-0029 §7.4; ADR-009/ADR-014; G2/VR-5).
//!
//! The M-360 packed-ternary dot kernels ([`crate::bitnet`], [`crate::simd`]) already exist and are
//! differential-checked against [`crate::bitnet::ternary_dot_ref`]. What M-728 adds is the
//! **formalization of the capability contract** RFC-0029 §7.4 requires:
//!
//! 1. **Explicit capability flag.** The acceleration is engaged **iff** BOTH
//!    - the compile-time feature `bitnet-accel` is ON (the build *opts in*), AND
//!    - the runtime capability is present ([`BitnetCapability::detect`] — today: a working `clang`
//!      JIT toolchain, the same dependency the M-360 kernels probe),
//!
//!    so it is **never silently engaged** on a build that didn't opt in or a host that can't run it.
//! 2. **Correctness.** When the accelerated path runs, its result **equals** the reference ternary
//!    result (`Empirical`, `tests/bitnet_accel.rs`) — the kernel is never trusted over the oracle, it
//!    is *checked against* it.
//! 3. **Never-silent graceful degradation (G2).** When the capability is absent (feature off, or
//!    runtime tool missing), the **reference** path runs and the fallback is **recorded** in the
//!    returned [`AccelOutcome`] (which [`Path`] ran + an `EXPLAIN`-able reason) — never a silent slow
//!    path, never an error. The caller always *knows* which path produced the number.
//!
//! **Honesty (VR-5).** This module does not claim a speedup. The "acceleration" is the existing
//! packed-ternary JIT kernel; the measured throughput is whatever `xtask e1` reports (no pre-written
//! number). The guarantee on the *value* is `Exact` (integer dot product, both paths), and on the
//! *equivalence* `Empirical` (the differential). The capability flag is the never-silent contract,
//! not a performance claim.

use mycelium_core::{PackScheme, Trit};

use crate::bitnet::{ternary_dot_ref, KERNEL_SCHEME};
use crate::llvm::AotError;

/// Whether this build was compiled with the `bitnet-accel` capability opted in (the compile-time half
/// of the gate). `true` only under `--features bitnet-accel`. A `const` so the gate is visible at
/// compile time and the dead-path is statically obvious — no hidden runtime branch decides whether the
/// build *could* accelerate.
pub const ACCEL_FEATURE_ENABLED: bool = cfg!(feature = "bitnet-accel");

/// The **runtime capability** to run the BitNet packed-ternary kernel (M-728; RFC-0029 §7.4). Today
/// the only requirement beyond the compile-time feature is a working JIT toolchain (`clang`) — the
/// same dependency [`crate::bitnet::compile_bitnet_dot`] probes — since the "accelerator" is the
/// in-process compiled kernel. Detected explicitly and reported with a reason, so the gate is
/// inspectable, never a silent capability sniff (NFR-1; G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitnetCapability {
    /// The build opted in (`--features bitnet-accel`).
    feature_enabled: bool,
    /// The runtime toolchain to JIT-compile the kernel is present.
    runtime_available: bool,
}

impl BitnetCapability {
    /// **Detect** the BitNet acceleration capability — the explicit runtime query (RFC-0029 §7.4:
    /// "compile-time feature + runtime query"). Combines the compile-time feature with a runtime probe
    /// of the JIT toolchain. The probe is a real, cheap compile of the I2_S kernel (the M-360 path),
    /// so "available" means "we actually compiled a kernel just now", not an asserted guess (VR-5).
    #[must_use]
    pub fn detect() -> Self {
        Self {
            feature_enabled: ACCEL_FEATURE_ENABLED,
            runtime_available: ACCEL_FEATURE_ENABLED && probe_runtime(),
        }
    }

    /// Whether the **accelerated** path will be taken: the build opted in **and** the runtime
    /// capability is present. When `false`, [`accelerated_ternary_dot`] degrades to the reference
    /// path — explicitly and recorded (never silent, G2).
    #[must_use]
    pub fn is_accelerated(&self) -> bool {
        self.feature_enabled && self.runtime_available
    }

    /// An `EXPLAIN` of the capability state — *why* the accelerated path will or won't run, so the
    /// gate is auditable and never a black box (NFR-1/NFR-4; DN-01). Names the compile-time feature
    /// and the runtime probe result.
    #[must_use]
    pub fn explain(&self) -> String {
        let feat = if self.feature_enabled {
            "feature `bitnet-accel` ON"
        } else {
            "feature `bitnet-accel` OFF (build did not opt in)"
        };
        let rt = if self.feature_enabled {
            if self.runtime_available {
                "runtime JIT toolchain present"
            } else {
                "runtime JIT toolchain ABSENT (clang missing)"
            }
        } else {
            "runtime probe skipped (feature off)"
        };
        let verdict = if self.is_accelerated() {
            "→ ACCELERATED path (packed-ternary JIT kernel)"
        } else {
            "→ REFERENCE path (explicit graceful degradation — never silent, G2)"
        };
        format!("BitnetCapability: {feat}; {rt} {verdict}")
    }
}

/// Probe whether the JIT toolchain can actually compile a kernel **right now** (a real I2_S compile,
/// not a guess). Returns `false` on `ToolchainMissing` (the house "skip gracefully" idiom) — never a
/// panic, never an assertion. Any *other* compile error is treated as "not available" too (we degrade
/// rather than fail), but is a genuine environment anomaly; the reference path still produces the
/// correct number, so degradation is safe. `pub(crate)` for white-box testing (`src/tests/accel.rs`).
pub(crate) fn probe_runtime() -> bool {
    crate::bitnet::compile_bitnet_dot().is_ok()
}

/// Which execution path produced an [`AccelOutcome`]'s value — recorded so a degradation is **never
/// silent** (G2). The caller can always see whether the accelerated kernel or the reference oracle
/// ran, and (for the reference path) why it was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Path {
    /// The BitNet packed-ternary JIT kernel ran (the capability was present).
    Accelerated {
        /// The packing scheme the kernel decoded.
        scheme: PackScheme,
    },
    /// The reference ternary dot ran, by **explicit graceful degradation** — with the recorded reason.
    Reference(DegradeReason),
}

/// Why the reference path was chosen instead of the accelerator — the recorded, inspectable basis for
/// a graceful degradation (G2: a fallback is never silent, it carries its reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeReason {
    /// The build was not compiled with `--features bitnet-accel` (the compile-time gate is closed).
    FeatureDisabled,
    /// The build opted in, but the runtime JIT toolchain (`clang`) is absent on this host.
    RuntimeUnavailable,
}

impl DegradeReason {
    /// A human-readable `EXPLAIN` of the degradation reason.
    #[must_use]
    pub fn explain(self) -> &'static str {
        match self {
            DegradeReason::FeatureDisabled => {
                "feature `bitnet-accel` OFF — build did not opt into acceleration"
            }
            DegradeReason::RuntimeUnavailable => {
                "runtime JIT toolchain (clang) absent — accelerator cannot be compiled on this host"
            }
        }
    }
}

/// The result of [`accelerated_ternary_dot`]: the dot-product **value** plus the **recorded path** that
/// produced it (G2 — a degradation is never silent). The value is `Exact` on either path (integer dot
/// product); the *equivalence* of the two paths is `Empirical` (the differential).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccelOutcome {
    /// The ternary dot-product value `Σ digit(wᵢ)·xᵢ`.
    pub value: i64,
    /// Which path produced it — recorded, inspectable.
    pub path: Path,
}

impl AccelOutcome {
    /// Whether the accelerated kernel produced this value (vs. a reference-path degradation).
    #[must_use]
    pub fn was_accelerated(&self) -> bool {
        matches!(self.path, Path::Accelerated { .. })
    }

    /// An `EXPLAIN` of which path ran and (for a degradation) why — so the capability decision is
    /// always auditable (NFR-1; G2).
    #[must_use]
    pub fn explain(&self) -> String {
        match &self.path {
            Path::Accelerated { scheme } => {
                format!(
                    "value={} via ACCELERATED packed-ternary kernel ({scheme:?})",
                    self.value
                )
            }
            Path::Reference(reason) => format!(
                "value={} via REFERENCE ternary dot (graceful degradation: {})",
                self.value,
                reason.explain()
            ),
        }
    }
}

/// **Run the ternary dot product `Σ digit(wᵢ)·xᵢ` through the BitNet capability gate** (M-728;
/// RFC-0029 §7.4).
///
/// The path is chosen by [`BitnetCapability::detect`]:
/// - **Capability present** (feature ON + runtime toolchain) → the packed-ternary JIT kernel
///   ([`crate::bitnet::jit_ternary_dot`], scheme [`KERNEL_SCHEME`]). If the kernel *unexpectedly*
///   reports the toolchain vanished between probe and run, that is surfaced as an **explicit
///   degradation** (recorded `RuntimeUnavailable`), never a silent failure.
/// - **Capability absent** (feature OFF, or runtime toolchain missing) → the reference ternary dot
///   ([`ternary_dot_ref`]), with the reason **recorded** in the returned [`AccelOutcome`].
///
/// Either way the returned value is the correct ternary dot product, and the [`AccelOutcome`] records
/// **which** path ran — so a degradation is never silent (G2). The accelerated and reference values
/// are equal (`Empirical`, `tests/bitnet_accel.rs`).
///
/// # Errors
/// Returns [`AotError`] only for a genuine *kernel* failure on the accelerated path that is **not** a
/// toolchain-missing degradation (e.g. an FFI/IO error, or a length/overflow refusal from the kernel)
/// — an explicit error, never a silent wrong answer. A toolchain-missing condition is **not** an
/// error: it degrades to the reference path and is recorded (G2).
pub fn accelerated_ternary_dot(
    weights: &[Trit],
    activations: &[i32],
) -> Result<AccelOutcome, AotError> {
    let cap = BitnetCapability::detect();
    if !cap.is_accelerated() {
        // Explicit, recorded graceful degradation — never a silent slow path (G2).
        let reason = if cap.feature_enabled {
            DegradeReason::RuntimeUnavailable
        } else {
            DegradeReason::FeatureDisabled
        };
        return Ok(AccelOutcome {
            value: ternary_dot_ref(weights, activations),
            path: Path::Reference(reason),
        });
    }

    // Capability present: run the accelerated packed-ternary kernel.
    match crate::bitnet::jit_ternary_dot(weights, activations) {
        Ok(value) => Ok(AccelOutcome {
            value,
            path: Path::Accelerated {
                scheme: KERNEL_SCHEME,
            },
        }),
        // The toolchain vanished between probe and run (a TOCTOU race / removed binary): degrade
        // explicitly and record it — never a silent failure (G2).
        Err(AotError::ToolchainMissing(_)) => Ok(AccelOutcome {
            value: ternary_dot_ref(weights, activations),
            path: Path::Reference(DegradeReason::RuntimeUnavailable),
        }),
        // A genuine kernel error (FFI/IO/length/overflow) — surfaced explicitly, never a silent
        // wrong answer. The caller decides; we do not paper over a real failure with the reference
        // value (that would hide a kernel bug — G2/VR-5).
        Err(e) => Err(e),
    }
}
