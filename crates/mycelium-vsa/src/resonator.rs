//! **Resonator-network factorization** (M-350; RFC-0009, Accepted; FR-C2 / G2 / G4 / RR-5).
//!
//! Recover the unknown factors `(x₁ … x_F)` of a bind product `s = x₁ ⊛ … ⊛ x_F`, where each `xᵢ`
//! is an atom of a known per-slot codebook `Cᵢ`, when a brute-force search over `∏ᵢ |Cᵢ|` is
//! intractable. The network holds a superposition estimate per slot and updates them **in parallel**
//! against the others' current estimates, letting the per-slot cleanup pull the estimates toward a
//! self-consistent set of clean atoms — a fixed point of the coupled update (Frady, Kent, Olshausen,
//! Sommer, *Neural Computation* **32**(12), 2020). It is iterative, **bounded**, and **best-effort**.
//!
//! # Honesty contract (RFC-0009 §5/§6 — non-negotiable)
//! - **Probabilistic-only, never `Proven`.** Convergence to the correct factors is not guaranteed; it
//!   almost always succeeds within an operational-capacity regime `{F, ∏kᵢ, d}` and fails outside it.
//!   The guarantee is capped at `Empirical` (for a model with an *exact* self-inverse bind, MAP-I);
//!   the `mycelium-core::recon` schema enforces the ceiling.
//! - **Never silent.** A non-converging run **cannot hang** (the iteration budget caps work) and is
//!   **never** returned as an answer: [`factorize`] yields [`Factorization`] **only** on a clean
//!   [`StopReason::Converged`] verdict that also clears the per-slot confidence and margin thresholds.
//!   `BudgetExhausted`, `Oscillating`, `Stalled`, below-confidence, and below-margin are explicit
//!   errors that carry the inspectable [`ResonatorTrace`] (so `EXPLAIN` works on failure too).
//!   "Converged ≠ correct" — a resonator can reach a *wrong* fixed point, which the brute-force oracle
//!   (`tests/resonator_oracle.rs`) and the trial-validated [`ResonatorProfile`] are what keep honest.
//! - **Deterministic.** Given the params + seed, a run is reproducible (no `rand`; a tiny in-crate
//!   LCG seeds initialisation) — the §8.1 P1 correction over the prior art.
//!
//! # The loop (RFC-0009 §3)
//! Per sweep, against a **snapshot** of the previous estimates (Jacobi, *not* in-place Gauss-Seidel —
//! §8.1 P6): `rᵢ = unbind(s, ⊛_{j≠i} x̂ⱼ)`; `x̂ᵢ ← cleanup_i(rᵢ)`. Convergence and oscillation are
//! decided on the **discrete per-slot top-atom index tuple `ι`** (§8.1 P3), never the real-valued
//! vector: converged iff `ι` is unchanged for a full sweep AND every slot's similarity ≥ `τ_lock`;
//! oscillating iff a *distinct* previously-seen `ι` recurs (a period-≥2 limit cycle). A **stationary**
//! tuple (`ι` == the previous sweep's) is not, on its own, a cycle — its real-valued estimate may
//! still be sharpening, so the loop keeps iterating while the lock bottleneck (min per-slot
//! similarity) climbs and only refuses (`Stalled`) once that climb plateaus below `τ_lock` for
//! `STALL_PATIENCE` sweeps. This is the M-350 premature-abort fix: the prior code conflated a
//! stationary-but-still-improving tuple with a genuine limit cycle and aborted it as `Oscillating`.

use std::collections::VecDeque;

use mycelium_core::{Bound, BoundBasis, BoundKind};

use crate::{CleanupMemory, Match, VsaError, VsaModel};

/// Minimum gain in the **min** per-slot top-similarity (the lock bottleneck) for a stationary sweep
/// to count as *still improving toward `τ_lock`* rather than plateaued (RFC-0009 §3/§6; M-350). A
/// genuinely converging tuple climbs by far more than this per sweep; once the climb falls below it
/// for [`STALL_PATIENCE`] consecutive sweeps the tuple is declared [`StopReason::Stalled`].
const STALL_MIN_GAIN: f64 = 1e-3;

/// Consecutive stationary sweeps with no [`STALL_MIN_GAIN`] improvement before a stationary tuple is
/// refused as [`StopReason::Stalled`]. Small (a stationary bipolar fixed point is frozen, so the
/// plateau is detected within a couple of sweeps), but > 0 so a one-sweep pause in the climb does not
/// abort a tuple that then resumes rising (the M-350 premature-abort fix; RFC-0009 §3/§6).
const STALL_PATIENCE: u64 = 4;

/// Per-slot cleanup projection (RFC-0009 §3 / §9 Q2 / §10.3 ablation).
///
/// The `*Sign` / `Hebbian` variants **bipolarize** the estimate (`x̂ ← sign(·)`) so the explain-away
/// product `g_{j≠i} x̂ⱼ` stays on the `±1` alphabet — making the unbind *exact* (MAP-I self-inverse on
/// `±1`) instead of compounding real-valued crosstalk through the product of `F−1` noisy real vectors.
/// This is the §10.3-measured fix that breaches the operational-capacity wall (Frady et al. 2020 apply
/// a sign nonlinearity to the bipolar resonator estimate; `Softmax`/`ArgMax` alone collapse as `∏k → d`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cleanup {
    /// Softmax-weighted superposition `Σⱼ softmax(β·simⱼ)·cᵢ,ⱼ` over the codebook (smoother dynamics).
    /// Real-valued output — the original prototype default; collapses past the wall (`∏k → d`).
    Softmax {
        /// Inverse temperature `β > 0` — larger sharpens toward the arg-max atom.
        beta: f64,
    },
    /// Winner-take-all: the single arg-max atom ([`CleanupMemory::cleanup`]).
    ArgMax,
    /// Bipolarized softmax: `sign(Σⱼ softmax(β·simⱼ)·cᵢ,ⱼ)`. Keeps the smooth softmax weighting but
    /// projects the estimate back to `±1` before re-binding (§10.3 variant 1).
    SoftmaxSign {
        /// Inverse temperature `β > 0` for the softmax weights, sharpening toward the arg-max atom.
        beta: f64,
    },
    /// Frady's standard bipolar resonator cleanup: `sign(Σⱼ simⱼ·cᵢ,ⱼ)` — the similarity-weighted
    /// (Hebbian / `Cᵀ C`) superposition projected to `±1`. No temperature (§10.3 variant 3).
    Hebbian,
}

/// Initialisation strategy (RFC-0009 §9 Q1). Default is the uniform codebook superposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    /// Equal-weight superposition of all codebook atoms per slot (the Frady "uniform" start). The
    /// explain-away against the specific product `s` breaks the symmetry from the first sweep.
    UniformSuperposition,
    /// A single seeded codebook atom per slot (the §10.3 ablation alternative).
    SeededGuess,
}

/// All resonator run parameters. A run is a pure function of these plus the codebooks and `s`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonatorParams {
    /// Per-slot cleanup projection.
    pub cleanup: Cleanup,
    /// Initialisation strategy.
    pub init: Init,
    /// Per-slot top-similarity lock threshold for the `Converged` verdict (RFC-0009 §3).
    pub tau_lock: f64,
    /// Minimum final per-slot confidence; below it is an explicit refusal (RFC-0009 §5.4).
    pub confidence_threshold: f64,
    /// Minimum final per-slot margin (top minus runner-up); below it is an ambiguity refusal (§5.4).
    pub margin_threshold: f64,
    /// Iteration budget (≥ 1): caps work so a non-converging run cannot hang (RFC-0009 §6).
    pub iteration_budget: u64,
    /// Oscillation-detection history window; `0` ⇒ remember every tuple this run (the §9 Q3 default).
    pub history_window: usize,
    /// Initialisation seed — determinism (RFC-0009 §6). Consumed by `SeededGuess`; reserved otherwise.
    pub seed: u64,
}

impl ResonatorParams {
    /// The recommended MAP-I defaults (Hebbian bipolar cleanup, uniform superposition init, τ_lock=0.9,
    /// conf≥0.3, margin≥0.1) — the knob values the [`MAPI_RESONATOR_PROFILE`] trials validate and record.
    /// [`Cleanup::Hebbian`] (`sign(Σⱼ simⱼ·cⱼ)`) is the §10.3-measured wall-breach: it keeps the
    /// explain-away on `±1`, so the MAP-I unbind stays exact and the F=3,k=16 (∏k=4096) regime recovers
    /// where the original softmax cleanup collapsed (RFC-0009 §10.3 ablation; changelog footer).
    #[must_use]
    pub fn mapi_default(iteration_budget: u64, seed: u64) -> Self {
        ResonatorParams {
            cleanup: Cleanup::Hebbian,
            init: Init::UniformSuperposition,
            tau_lock: 0.9,
            confidence_threshold: 0.3,
            margin_threshold: 0.1,
            iteration_budget,
            history_window: 0,
            seed,
        }
    }
}

/// The terminal verdict of a run (RFC-0009 §3/§6). Exactly one is reached; a run never hangs.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// `ι` unchanged for a full sweep AND every slot's top-similarity ≥ `τ_lock` (a discrete fixed
    /// point). The only verdict that can yield factors.
    Converged,
    /// A **genuine limit cycle** (`period ≥ 2`): a previously-seen `ι` *other than the immediately
    /// preceding one* recurred, so the network is alternating between distinct tuples (`period` =
    /// shortest distance, in sweeps, back to the matching earlier tuple). A *stationary* tuple
    /// (`ι` == the previous sweep's `ι`) is **not** an oscillation — it is handled separately: it
    /// keeps iterating while its confidence is still climbing toward `τ_lock`, and only refuses as
    /// [`StopReason::Stalled`] once that climb plateaus (RFC-0009 §3/§6; §8.1 P3).
    Oscillating {
        /// Shortest distance (in sweeps, `≥ 2`) back to the matching earlier tuple.
        period: usize,
    },
    /// A **stationary tuple that plateaued below `τ_lock`**: `ι` stopped changing but its per-slot
    /// top-similarity stopped improving (no min-confidence gain ≥ `STALL_MIN_GAIN` for
    /// `STALL_PATIENCE` consecutive sweeps) before every slot reached `τ_lock`. This is a genuine
    /// stuck fixed point — an explicit refusal, never returned as factors (RFC-0009 §6; never-silent).
    /// Distinguished from [`StopReason::Oscillating`] (a real cycle) and from a *still-improving*
    /// stationary tuple (which keeps iterating toward lock — the M-350 premature-abort fix).
    Stalled {
        /// Consecutive stationary sweeps without a confidence gain before the refusal.
        sweeps: u64,
    },
    /// The iteration budget was reached with `ι` still changing every sweep.
    BudgetExhausted,
}

/// One sweep's decoded snapshot, for `EXPLAIN` (RFC-0009 §4 run trace / similarity trajectory).
#[derive(Debug, Clone, PartialEq)]
pub struct IterationRecord {
    /// The sweep index (number of updates already applied; 0 = the initial estimate).
    pub iter: u64,
    /// `ι`: the per-slot arg-max codebook atom index.
    pub indices: Vec<usize>,
    /// Per-slot top similarity (the confidence the `τ_lock`/threshold gates read).
    pub confidences: Vec<f64>,
    /// Per-slot margin (top minus runner-up).
    pub margins: Vec<f64>,
}

/// The full inspectable trace + verdict — returned on **any** stop (success or error), so `EXPLAIN`
/// can render a failed run (RFC-0009 §4/§6).
#[derive(Debug, Clone, PartialEq)]
pub struct ResonatorTrace {
    /// Which terminal verdict was reached.
    pub stop: StopReason,
    /// Number of update sweeps performed before terminating.
    pub iterations: u64,
    /// Per-sweep decoded snapshots (the similarity trajectory).
    pub trajectory: Vec<IterationRecord>,
    /// The per-slot decoded [`Match`] at termination (label, index, confidence, margin).
    pub final_decode: Vec<Match>,
}

/// A clean, gate-passing factorization: the per-slot recovered atom plus its confidence/margin, and
/// the trace that produced it. Only ever constructed on a [`StopReason::Converged`] run that clears
/// the confidence and margin thresholds (RFC-0009 §6).
#[derive(Debug, Clone, PartialEq)]
pub struct Factorization {
    /// The recovered factor per slot (in slot order).
    pub factors: Vec<Match>,
    /// The run trace.
    pub trace: ResonatorTrace,
}

/// Trial-validated operational regime for resonator factorization (RFC-0009 §5.2 / §9 Q4). Distinct
/// from the bundle [`EmpiricalProfile`](crate::EmpiricalProfile) (§8.1 P4): its axes are the factor
/// count `F`, the operational capacity `∏ᵢ kᵢ`, and the dimension `d`. The constants are exercised by
/// `tests/resonator_profile.rs` at exactly `trials`, scoring **exact-tuple recovery against a
/// brute-force oracle** (not self-reported convergence — §8.1 P5), so the `Empirical` tag is earned.
#[derive(Debug, Clone, PartialEq)]
pub struct ResonatorProfile {
    /// Factor count `F ≤ this`.
    pub max_factors: usize,
    /// Each codebook size `kᵢ ≤ this`.
    pub max_codebook: usize,
    /// Operational capacity `∏ᵢ kᵢ ≤ this`.
    pub max_capacity: u128,
    /// Dimension `d ≥ this`.
    pub min_dim: u32,
    /// The validated (oracle-measured) failure probability the trials stayed at or below.
    pub delta: f64,
    /// Number of trials the validation runs.
    pub trials: u64,
    /// The validation method, recording the knob values (β, τ_lock, init) used.
    pub method: &'static str,
}

impl ResonatorProfile {
    /// Check the regime side-conditions for a concrete request; a violation is an explicit
    /// [`VsaError::OutsideEmpiricalProfile`], never a stretched tag (RFC-0009 §5.2).
    pub fn check(
        &self,
        factors: usize,
        codebook_sizes: &[usize],
        dim: u32,
    ) -> Result<(), VsaError> {
        if factors == 0 || factors > self.max_factors {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!(
                    "validated for 1..={} factors, got {factors}",
                    self.max_factors
                ),
            });
        }
        if codebook_sizes.len() != factors {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!(
                    "{factors} factors but {} codebook sizes",
                    codebook_sizes.len()
                ),
            });
        }
        let mut capacity: u128 = 1;
        for &k in codebook_sizes {
            if k == 0 || k > self.max_codebook {
                return Err(VsaError::OutsideEmpiricalProfile {
                    detail: format!(
                        "validated for 1..={} atoms per slot, got {k}",
                        self.max_codebook
                    ),
                });
            }
            capacity = capacity.saturating_mul(k as u128);
        }
        if capacity > self.max_capacity {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!(
                    "validated for operational capacity ∏kᵢ ≤ {}, got {capacity}",
                    self.max_capacity
                ),
            });
        }
        if dim < self.min_dim {
            return Err(VsaError::OutsideEmpiricalProfile {
                detail: format!("validated for dim ≥ {}, got {dim}", self.min_dim),
            });
        }
        Ok(())
    }

    /// The δ bound this profile backs, with its honest `EmpiricalFit` basis (RFC-0009 §5.2).
    #[must_use]
    pub fn bound(&self) -> Bound {
        Bound {
            kind: BoundKind::Probability { delta: self.delta },
            basis: BoundBasis::EmpiricalFit {
                trials: self.trials,
                method: self.method.to_owned(),
            },
        }
    }
}

/// The trial-validated MAP-I regime (RFC-0009 §9 Q4 / §10.2 / §10.3). The regime is fixed here; `delta`
/// is **set from the measured trial rate** in `tests/resonator_profile.rs`, never asserted ahead of
/// the run (VR-5). Conservative ceiling: 0/`trials` failures ⇒ `δ ≤ 0.01` is an honest bound.
///
/// **Wall breached (§10.3 cleanup ablation).** The original softmax cleanup collapsed at `∏k → d`
/// (F=3,k=16 ≈100% failure even at d=8192). Switching the per-slot cleanup to the **Hebbian bipolar**
/// projection `sign(Σⱼ simⱼ·cⱼ)` ([`Cleanup::Hebbian`]) keeps the explain-away on the `±1` alphabet, so
/// the MAP-I unbind stays exact — and F=3,k=16 (∏k=4096) now recovers with **0/300** failures at
/// d=4096. The validated envelope widens from `k≤8, ∏k≤512` to **`k≤16, ∏k≤4096`** at this measured δ.
pub const MAPI_RESONATOR_PROFILE: ResonatorProfile = ResonatorProfile {
    max_factors: 3,
    max_codebook: 16,
    max_capacity: 4096, // 16³ — the operational capacity ∏ᵢ kᵢ
    min_dim: 4096,
    delta: 0.02,
    trials: 1_000,
    method: "Monte-Carlo exact-tuple recovery vs brute-force oracle (MAP-I bipolar; Hebbian bipolar \
             cleanup sign(Σⱼ simⱼ·cⱼ), τ_lock=0.9, uniform superposition init, budget 50; F≤3, k≤16, \
             ∏k≤4096, d≥4096; worst corner F=3,k=16,d=4096 measured 0/1000 ⇒ δ=0.02 conservative \
             ceiling. §10.3 wall-breach: the prior softmax-β=6 cleanup collapsed here (≈100% fail), \
             fixed by bipolarizing the estimate so the explain-away stays on ±1. Boundary data beyond \
             the envelope: F=3,k=32 (∏=32768) needs d≥16384 (0.005) — not validated at d=8192 (0.085); \
             SoftmaxSign does NOT breach the wall, ArgMax only partially (RFC-0009 §9 Q4/Q6, §10.3)",
};

/// Factorize `s` into one atom per slot of `codebooks`, running the RFC-0009 §3 loop with `params`.
///
/// Returns a [`Factorization`] **only** on a clean [`StopReason::Converged`] verdict whose every slot
/// clears `τ_lock` (during the loop) and the `confidence_threshold`/`margin_threshold` (at the end).
/// Every other outcome is an explicit error carrying the [`ResonatorTrace`]:
/// [`VsaError::ResonatorBudgetExhausted`], [`VsaError::ResonatorOscillating`] (a genuine period-≥2
/// limit cycle), [`VsaError::ResonatorStalled`] (a stationary tuple that plateaued below `τ_lock`),
/// [`VsaError::ResonatorBelowConfidence`], [`VsaError::ResonatorBelowMargin`]. Input problems (empty
/// codebook list / empty codebook / dim mismatch / zero budget) are the usual explicit errors.
pub fn factorize<M: VsaModel>(
    model: &M,
    s: &[f64],
    codebooks: &[CleanupMemory],
    params: &ResonatorParams,
) -> Result<Factorization, VsaError> {
    let trace = run_loop(model, s, codebooks, params)?;
    match trace.stop {
        StopReason::Converged => {
            // §5.4 gate: a clean convergence still refuses on a low-confidence or ambiguous slot —
            // "converged ≠ correct"; never a silent low-quality / coin-flip guess (§8.1 P5).
            for (slot, m) in trace.final_decode.iter().enumerate() {
                if m.confidence < params.confidence_threshold {
                    return Err(VsaError::ResonatorBelowConfidence {
                        slot,
                        confidence: m.confidence,
                        threshold: params.confidence_threshold,
                        trace: Box::new(trace),
                    });
                }
                if m.margin < params.margin_threshold {
                    return Err(VsaError::ResonatorBelowMargin {
                        slot,
                        margin: m.margin,
                        threshold: params.margin_threshold,
                        trace: Box::new(trace),
                    });
                }
            }
            let factors = trace.final_decode.clone();
            Ok(Factorization { factors, trace })
        }
        StopReason::Oscillating { .. } => Err(VsaError::ResonatorOscillating {
            trace: Box::new(trace),
        }),
        StopReason::Stalled { .. } => Err(VsaError::ResonatorStalled {
            trace: Box::new(trace),
        }),
        StopReason::BudgetExhausted => Err(VsaError::ResonatorBudgetExhausted {
            trace: Box::new(trace),
        }),
    }
}

/// The raw §3 loop without the §5.4 confidence/margin gate: returns the trace + [`StopReason`] for
/// any outcome (`Err` only on input validation). Exposed for the oracle/ablation tests, which inspect
/// convergence independent of the gate.
pub(crate) fn run_loop<M: VsaModel>(
    model: &M,
    s: &[f64],
    codebooks: &[CleanupMemory],
    params: &ResonatorParams,
) -> Result<ResonatorTrace, VsaError> {
    // --- input validation (explicit, never silent) ---
    if codebooks.is_empty() {
        return Err(VsaError::EmptyCodebook);
    }
    if params.iteration_budget == 0 {
        // The schema enforces ≥ 1 on the manifest; guard here too so the loop cannot no-op silently.
        return Err(VsaError::OutsideEmpiricalProfile {
            detail: "iteration_budget must be ≥ 1".to_owned(),
        });
    }
    let dim = s.len();
    for cb in codebooks {
        if cb.is_empty() {
            return Err(VsaError::EmptyCodebook);
        }
        if cb.dim() as usize != dim {
            return Err(VsaError::DimMismatch {
                expected: dim,
                got: cb.dim() as usize,
            });
        }
    }

    let f = codebooks.len();
    let mut est = init_estimates(model, codebooks, params)?;

    let window = if params.history_window == 0 {
        params.iteration_budget as usize
    } else {
        params.history_window
    };
    let mut history: VecDeque<Vec<usize>> = VecDeque::new();
    let mut trajectory: Vec<IterationRecord> = Vec::new();
    let mut prev_indices: Option<Vec<usize>> = None;
    let mut updates_done: u64 = 0;
    // Stall tracking for a *stationary* tuple (ι == prev): the best min-confidence seen during the
    // current stationary streak (a monotone ratchet) and how many consecutive sweeps it has failed to
    // beat by STALL_MIN_GAIN. A stationary tuple is only refused (Stalled) once the climb plateaus —
    // never on the first recurrence, so a still-improving correct tuple is no longer aborted (M-350).
    let mut stall_best = f64::NEG_INFINITY;
    let mut stall_streak: u64 = 0;

    loop {
        // Decode the current estimates: ι + per-slot confidence/margin (§3 discrete decode).
        let decode = decode_estimates(model, codebooks, &est)?;
        let indices: Vec<usize> = decode.iter().map(|m| m.index).collect();
        let confidences: Vec<f64> = decode.iter().map(|m| m.confidence).collect();
        let margins: Vec<f64> = decode.iter().map(|m| m.margin).collect();
        trajectory.push(IterationRecord {
            iter: updates_done,
            indices: indices.clone(),
            confidences: confidences.clone(),
            margins: margins.clone(),
        });

        let locked = confidences.iter().all(|&c| c >= params.tau_lock);
        let stationary = prev_indices.as_ref() == Some(&indices);
        if stationary {
            // Converged: ι unchanged for a full sweep AND every slot locked (§3 / §8.1 P3).
            if locked {
                return Ok(ResonatorTrace {
                    stop: StopReason::Converged,
                    iterations: updates_done,
                    trajectory,
                    final_decode: decode,
                });
            }
            // Stationary but not yet locked. The tuple stopped *moving*, but its real-valued estimate
            // may still be sharpening — so this is NOT a limit cycle. Keep iterating while the lock
            // bottleneck (min per-slot top-similarity) is still climbing; only refuse once that climb
            // plateaus for STALL_PATIENCE sweeps (genuine stuck fixed point — never-silent, §6).
            let min_conf = confidences.iter().copied().fold(f64::INFINITY, f64::min);
            if min_conf > stall_best + STALL_MIN_GAIN {
                stall_best = min_conf;
                stall_streak = 0;
            } else {
                stall_streak += 1;
                if stall_streak >= STALL_PATIENCE {
                    return Ok(ResonatorTrace {
                        stop: StopReason::Stalled {
                            sweeps: stall_streak,
                        },
                        iterations: updates_done,
                        trajectory,
                        final_decode: decode,
                    });
                }
            }
        } else {
            // ι changed this sweep. A recurrence of any *earlier* tuple (the immediately preceding one
            // is excluded — it differs by construction) is a genuine limit cycle of period ≥ 2; the
            // shortest such distance is the nearest (rposition) match (§3/§8.1 P3/§9 Q3).
            if let Some(pos) = history.iter().rposition(|h| h == &indices) {
                let period = history.len() - pos;
                return Ok(ResonatorTrace {
                    stop: StopReason::Oscillating { period },
                    iterations: updates_done,
                    trajectory,
                    final_decode: decode,
                });
            }
            // A fresh tuple resets the stall ratchet — the plateau is measured per stationary streak.
            stall_best = f64::NEG_INFINITY;
            stall_streak = 0;
        }
        history.push_back(indices.clone());
        if history.len() > window {
            history.pop_front();
        }
        prev_indices = Some(indices);

        // Bounded: stop at the budget with ι still moving (§6).
        if updates_done >= params.iteration_budget {
            return Ok(ResonatorTrace {
                stop: StopReason::BudgetExhausted,
                iterations: updates_done,
                trajectory,
                final_decode: decode,
            });
        }

        // Parallel / Jacobi update: build the whole next sweep from the snapshot `est` (§8.1 P6).
        let mut next: Vec<Vec<f64>> = Vec::with_capacity(f);
        for (i, cb) in codebooks.iter().enumerate() {
            let others = bind_others(model, &est, i, dim)?;
            let r = model.unbind(s, &others)?;
            next.push(cleanup_slot(model, cb, &r, params.cleanup)?);
        }
        est = next;
        updates_done += 1;
    }
}

/// Initial estimate per slot (RFC-0009 §9 Q1).
fn init_estimates<M: VsaModel>(
    model: &M,
    codebooks: &[CleanupMemory],
    params: &ResonatorParams,
) -> Result<Vec<Vec<f64>>, VsaError> {
    codebooks
        .iter()
        .enumerate()
        .map(|(i, cb)| match params.init {
            Init::UniformSuperposition => {
                let atoms: Vec<&[f64]> = cb.atoms().map(|(_, a)| a).collect();
                model.bundle(&atoms)
            }
            Init::SeededGuess => {
                // Deterministic per-slot pick; the seed makes the run reproducible (§6).
                let mut lcg =
                    Lcg::new(params.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let pick = (lcg.next_u64() % cb.len() as u64) as usize;
                let atom = cb
                    .atoms()
                    .nth(pick)
                    .map(|(_, a)| a.to_vec())
                    .ok_or(VsaError::EmptyCodebook)?;
                Ok(atom)
            }
        })
        .collect()
}

/// Bind every estimate except slot `i` (the "explain away" product). For `F == 1` there are no other
/// factors, so the result is the multiplicative identity (all-ones — MAP-I/BSC bind identity).
fn bind_others<M: VsaModel>(
    model: &M,
    est: &[Vec<f64>],
    i: usize,
    dim: usize,
) -> Result<Vec<f64>, VsaError> {
    let mut acc: Option<Vec<f64>> = None;
    for (j, e) in est.iter().enumerate() {
        if j == i {
            continue;
        }
        acc = Some(match acc {
            None => e.clone(),
            Some(a) => model.bind(&a, e)?,
        });
    }
    Ok(acc.unwrap_or_else(|| vec![1.0; dim]))
}

/// Project `r` onto slot `i`'s codebook (RFC-0009 §3 / §9 Q2).
fn cleanup_slot<M: VsaModel>(
    model: &M,
    cb: &CleanupMemory,
    r: &[f64],
    cleanup: Cleanup,
) -> Result<Vec<f64>, VsaError> {
    match cleanup {
        Cleanup::ArgMax => {
            let hit = cb.cleanup(r, model).ok_or(VsaError::EmptyCodebook)?;
            cb.atoms()
                .nth(hit.index)
                .map(|(_, a)| a.to_vec())
                .ok_or(VsaError::EmptyCodebook)
        }
        Cleanup::Softmax { beta } => Ok(softmax_superposition(model, cb, r, beta)),
        Cleanup::SoftmaxSign { beta } => {
            // §10.3 variant 1: bipolarize the softmax estimate so the explain-away stays on ±1.
            Ok(bipolarize(&softmax_superposition(model, cb, r, beta)))
        }
        Cleanup::Hebbian => {
            // §10.3 variant 3: Frady's bipolar cleanup `sign(Σⱼ simⱼ·cᵢ,ⱼ)` — the similarity-weighted
            // (Cᵀ C) superposition, projected to ±1. Linear weights (no softmax temperature).
            let mut acc = vec![0.0_f64; cb.dim() as usize];
            for (_, atom) in cb.atoms() {
                let w = model.similarity(r, atom);
                for (a, &x) in acc.iter_mut().zip(atom) {
                    *a += w * x;
                }
            }
            Ok(bipolarize(&acc))
        }
    }
}

/// The softmax-weighted codebook superposition `Σⱼ softmax(β·simⱼ)·cᵢ,ⱼ` (numerically stable).
fn softmax_superposition<M: VsaModel>(
    model: &M,
    cb: &CleanupMemory,
    r: &[f64],
    beta: f64,
) -> Vec<f64> {
    let sims: Vec<f64> = cb.atoms().map(|(_, a)| model.similarity(r, a)).collect();
    let max = sims.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = sims.iter().map(|&s| ((s - max) * beta).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut acc = vec![0.0_f64; cb.dim() as usize];
    // sum is ≥ 1 (the max term contributes exp(0) = 1), so the division is safe.
    for ((_, atom), e) in cb.atoms().zip(exps.iter()) {
        let w = e / sum;
        for (a, &x) in acc.iter_mut().zip(atom) {
            *a += w * x;
        }
    }
    acc
}

/// Project a real-valued estimate onto the bipolar `±1` alphabet (`sign`, with `sign(0) = +1` for a
/// deterministic tie-break). Keeping the estimate bipolar makes the MAP-I explain-away exact (§10.3).
fn bipolarize(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
        .collect()
}

/// Decode each slot's current estimate to its top codebook atom (`ι` + confidence + margin).
fn decode_estimates<M: VsaModel>(
    model: &M,
    codebooks: &[CleanupMemory],
    est: &[Vec<f64>],
) -> Result<Vec<Match>, VsaError> {
    codebooks
        .iter()
        .zip(est)
        .map(|(cb, e)| cb.cleanup(e, model).ok_or(VsaError::EmptyCodebook))
        .collect()
}

/// A tiny deterministic LCG (no `rand` dependency — house rule). Same constants as the model tests;
/// used for seeded initialisation and by this crate's resonator trials.
pub(crate) struct Lcg(u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// A deterministic bipolar (`±1`) hypervector of length `dim` (used by the inline tests).
    #[cfg(test)]
    pub(crate) fn bipolar(&mut self, dim: u32) -> Vec<f64> {
        (0..dim)
            .map(|_| {
                if (self.next_u64() >> 63) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MapI;

    const D: u32 = 4096;

    /// A codebook of `k` deterministic bipolar atoms seeded from `base`.
    fn codebook(k: usize, base: u64) -> (CleanupMemory, Vec<Vec<f64>>) {
        let mut mem = CleanupMemory::new(D);
        let mut atoms = Vec::with_capacity(k);
        let mut lcg = Lcg::new(base);
        for j in 0..k {
            let atom = lcg.bipolar(D);
            mem.insert(format!("{base}:{j}"), atom.clone()).unwrap();
            atoms.push(atom);
        }
        (mem, atoms)
    }

    /// Bind a chosen atom from each codebook into the product `s` (MAP-I elementwise product).
    fn product(model: &MapI, atoms: &[&Vec<f64>]) -> Vec<f64> {
        let mut acc = atoms[0].clone();
        for a in &atoms[1..] {
            acc = model.bind(&acc, a).unwrap();
        }
        acc
    }

    const RES0_SEED: u64 = 0x5350_0001;

    fn params() -> ResonatorParams {
        ResonatorParams::mapi_default(50, RES0_SEED)
    }

    #[test]
    fn two_factor_converges_and_recovers() {
        let model = MapI::new(D);
        let (c0, a0) = codebook(8, 10);
        let (c1, a1) = codebook(8, 20);
        // True tuple = (3, 5).
        let s = product(&model, &[&a0[3], &a1[5]]);
        let out = factorize(&model, &s, &[c0, c1], &params()).expect("converges");
        assert_eq!(out.trace.stop, StopReason::Converged);
        assert_eq!(out.factors[0].index, 3);
        assert_eq!(out.factors[1].index, 5);
    }

    #[test]
    fn budget_exhausted_is_an_error_not_an_answer() {
        // A 1-iteration budget on a fresh uniform init cannot lock ⇒ explicit error, never factors.
        let model = MapI::new(D);
        let (c0, a0) = codebook(8, 30);
        let (c1, a1) = codebook(8, 40);
        let s = product(&model, &[&a0[1], &a1[2]]);
        let mut p = params();
        p.iteration_budget = 1;
        match factorize(&model, &s, &[c0, c1], &p) {
            Err(VsaError::ResonatorBudgetExhausted { trace }) => {
                assert_eq!(trace.stop, StopReason::BudgetExhausted);
            }
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
    }

    #[test]
    fn below_margin_refuses_even_when_converged() {
        // Set the margin threshold absurdly high so a genuine convergence still refuses (§5.4).
        let model = MapI::new(D);
        let (c0, a0) = codebook(8, 50);
        let (c1, a1) = codebook(8, 60);
        let s = product(&model, &[&a0[0], &a1[0]]);
        let mut p = params();
        p.margin_threshold = 1.9; // margin (top − runner-up) cannot reach this on a cosine in [−1,1]
        match factorize(&model, &s, &[c0, c1], &p) {
            Err(VsaError::ResonatorBelowMargin { slot, .. }) => assert!(slot < 2),
            other => panic!("expected ResonatorBelowMargin, got {other:?}"),
        }
    }

    #[test]
    fn determinism_same_seed_same_trace() {
        let model = MapI::new(D);
        let (c0, _) = codebook(8, 70);
        let (c1, a1) = codebook(8, 80);
        let (c0b, a0) = codebook(8, 70);
        let (c1b, _) = codebook(8, 80);
        let s = product(&model, &[&a0[4], &a1[6]]);
        let r1 = run_loop(&model, &s, &[c0, c1], &params()).unwrap();
        let r2 = run_loop(&model, &s, &[c0b, c1b], &params()).unwrap();
        assert_eq!(r1, r2, "identical params + seed ⇒ identical trace");
    }

    #[test]
    fn argmax_cleanup_also_recovers() {
        let model = MapI::new(D);
        let (c0, a0) = codebook(8, 90);
        let (c1, a1) = codebook(8, 100);
        let s = product(&model, &[&a0[2], &a1[7]]);
        let mut p = params();
        p.cleanup = Cleanup::ArgMax;
        let out = factorize(&model, &s, &[c0, c1], &p).expect("argmax converges");
        assert_eq!((out.factors[0].index, out.factors[1].index), (2, 7));
    }

    #[test]
    fn stall_below_lock_is_stalled_not_an_answer() {
        // With an unreachable τ_lock (cosine ≤ 1 < 1.5) the decode stabilises (ι == prev) and its
        // confidence climbs to the genuine fixed point, then plateaus there — below the impossible
        // lock. The loop keeps iterating while the min-confidence is still rising (the M-350 fix: a
        // stationary-but-improving tuple is NOT aborted), and only once the climb plateaus for
        // STALL_PATIENCE sweeps does it refuse with the explicit StopReason::Stalled verdict —
        // carrying the trace, never a returned factor set, and distinct from a genuine period-≥2
        // limit cycle (§3/§6/§8.1 P3). Updated from the prior Oscillating{period:1} semantics.
        let model = MapI::new(D);
        let (c0, a0) = codebook(8, 110);
        let (c1, a1) = codebook(8, 120);
        let s = product(&model, &[&a0[1], &a1[1]]);
        let mut p = params();
        p.tau_lock = 1.5; // impossible for a cosine
        match factorize(&model, &s, &[c0, c1], &p) {
            Err(VsaError::ResonatorStalled { trace }) => {
                assert!(matches!(trace.stop, StopReason::Stalled { sweeps } if sweeps >= 1));
            }
            other => panic!("expected ResonatorStalled, got {other:?}"),
        }
    }

    #[test]
    fn stationary_but_improving_tuple_converges_not_aborted() {
        // M-350 regression: the §3 loop must NOT abort a tuple that has gone stationary on ι while its
        // per-slot confidence is still climbing toward τ_lock. Before the fix, the second sweep at the
        // correct tuple recurred in the history and was mislabelled Oscillating{period:1}; now the loop
        // keeps iterating and locks. We reproduce the observed premature-abort instance (F=3, k=16,
        // d=4096, Hebbian) with the sequential generator the profile/recon tests use (one LCG seeded
        // once, slot by slot) at the seed that exhibited it — slot 2 sat at ~0.72 ↗ when the old code
        // aborted. The fix must yield a clean Converged that recovers the exact true tuple.
        const KSEED: u64 = 0x1234 ^ 104;
        let model = MapI::new(D);
        let mut lcg = Lcg::new(KSEED);
        let (f, k) = (3usize, 16usize);
        let mut mems = Vec::with_capacity(f);
        let mut atoms: Vec<Vec<Vec<f64>>> = Vec::with_capacity(f);
        for i in 0..f {
            let mut mem = CleanupMemory::new(D);
            let mut slot = Vec::with_capacity(k);
            for j in 0..k {
                let a = lcg.bipolar(D);
                mem.insert(format!("{i}:{j}"), a.clone()).unwrap();
                slot.push(a);
            }
            mems.push(mem);
            atoms.push(slot);
        }
        let truth: Vec<usize> = (0..f)
            .map(|_| (lcg.next_u64() % k as u64) as usize)
            .collect();
        let mut s = atoms[0][truth[0]].clone();
        for slot in 1..f {
            s = model.bind(&s, &atoms[slot][truth[slot]]).unwrap();
        }
        let mut p = ResonatorParams::mapi_default(50, lcg.next_u64());
        p.cleanup = Cleanup::Hebbian;
        let out = factorize(&model, &s, &mems, &p).expect("rising stationary tuple now converges");
        assert_eq!(out.trace.stop, StopReason::Converged);
        let got: Vec<usize> = out.factors.iter().map(|m| m.index).collect();
        assert_eq!(
            got, truth,
            "recovers the exact true tuple it was aborting on"
        );
    }

    #[test]
    fn empty_codebook_and_dim_mismatch_are_explicit() {
        let model = MapI::new(D);
        let s = vec![1.0; D as usize];
        assert!(matches!(
            run_loop(&model, &s, &[], &params()),
            Err(VsaError::EmptyCodebook)
        ));
        let empty = CleanupMemory::new(D);
        assert!(matches!(
            run_loop(&model, &s, &[empty], &params()),
            Err(VsaError::EmptyCodebook)
        ));
        let (wrong, _) = codebook(4, 1);
        let short_s = vec![1.0; 8];
        assert!(matches!(
            run_loop(&model, &short_s, &[wrong], &params()),
            Err(VsaError::DimMismatch { .. })
        ));
    }

    #[test]
    fn profile_check_refuses_outside_regime() {
        let p = &MAPI_RESONATOR_PROFILE;
        // In regime: F≤3, k≤16, ∏k≤4096, d≥4096 (the §10.3 widened envelope).
        assert!(p.check(2, &[8, 8], 4096).is_ok());
        assert!(p.check(3, &[8, 8, 8], 4096).is_ok());
        assert!(p.check(3, &[16, 8, 8], 4096).is_ok()); // k=16 is now in regime (∏=1024)
        assert!(p.check(3, &[16, 16, 16], 4096).is_ok()); // the new edge: ∏=4096 = max_capacity
                                                          // Too many factors / too-large codebook / too-small dimension all refuse explicitly.
        assert!(matches!(
            p.check(4, &[16, 16, 16, 16], 4096),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
        assert!(matches!(
            p.check(3, &[32, 8, 8], 4096), // k=32 is past the validated edge
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
        assert!(matches!(
            p.check(3, &[16, 16, 16], 1024),
            Err(VsaError::OutsideEmpiricalProfile { .. })
        ));
        // The bound it backs is Empirical, never stronger.
        assert_eq!(
            p.bound().basis.strength(),
            mycelium_core::GuaranteeStrength::Empirical
        );
    }
}
