//! The **WIN / LOSS / REGRESSION classifier** — the honest core of the harness. For each non-baseline
//! backend on each case, it compares against the trusted interpreter and emits an explicit verdict.
//! This module is pure (no I/O, no timing of its own), so the classification is exhaustively unit-
//! testable and deterministic.
//!
//! The honesty discipline, made mechanical:
//! - A backend whose result **diverges** from the interpreter's is a **correctness LOSS** — recorded,
//!   never hidden. (A divergence is the worst kind of loss: a wrong answer faster is still wrong.)
//! - A backend that **cannot lower** the program is a **capability LOSS**, with the reason (G2).
//! - When both agree, the speed comparison yields **WIN** / **LOSS** / **NEUTRAL** vs the interpreter
//!   — tagged `Empirical`, with no pre-written target (VR-5). The "loss" here means *slower than the
//!   trusted in-process interpreter*, which for a trivial kernel is the expected, honest finding for
//!   a process-spawn-bound compiled path (M-602/E1) — surfaced with that reason, not buried.
//! - A backend that was only **skipped** (toolchain absent) yields no verdict — the harness could not
//!   measure it here; that is neither a win nor a loss.
//!
//! ## Regression gates (M-859) — this-run-vs-committed-baseline
//! The classifier above compares a backend **against the interpreter, same run**. The
//! [`RegressionGate`]/[`regression_classify`] pair below is a *second*, orthogonal comparison: this
//! run's `ns_per_call` for a (case, backend) **against a committed baseline JSON** captured on a
//! specific host — the day-to-day "did we get faster or slower since the baseline was taken"
//! question. Both comparisons coexist; neither replaces the other (the interpreter differential is
//! still how a divergence/capability loss is caught; the regression gate is purely a speed trend).
//!
//! **Honesty:** a regression verdict is `Empirical`, exactly like a speed verdict — measured, with
//! the [`RegressionGate::REGRESSION_BAND`] threshold reified (no black box), and it is **only ever
//! compared against a baseline captured on the *same host tag*** — cross-host comparison is refused
//! (a different host's numbers are not portable; see [`RegressionOutcome::HostMismatch`]).

use crate::backend::{observable_eq, Backend, Outcome};
use crate::timing::Timing;

/// The speed comparison band of a backend vs the interpreter, once both produced an *equal* value.
/// `Empirical`, with the threshold reified (no black box). A ratio is `interp_ns / backend_ns`:
/// `> 1` means the backend is faster than the interpreter (a win); `< 1` means slower (a loss).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Speed {
    /// Faster than the interpreter beyond the neutral band — a measured WIN.
    Win,
    /// Within the neutral band of the interpreter — neither a clear win nor loss.
    Neutral,
    /// Slower than the interpreter beyond the neutral band — a measured LOSS.
    Loss,
}

/// The full classification of one (backend, case) pair vs the trusted interpreter baseline.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Verdict {
    /// Both produced equal values; the backend was faster than the interpreter.
    SpeedWin {
        /// `interp_ns / backend_ns` (`> 1`).
        ratio_x1000: u64,
    },
    /// Both produced equal values; within the neutral band.
    SpeedNeutral {
        /// `interp_ns / backend_ns`.
        ratio_x1000: u64,
    },
    /// Both produced equal values; the backend was slower than the interpreter (a speed LOSS).
    SpeedLoss {
        /// `interp_ns / backend_ns` (`< 1`).
        ratio_x1000: u64,
        /// A derivable reason, where one applies (e.g. process-spawn-bound) — honest, not buried.
        reason: String,
    },
    /// The backend produced a value that **diverges** from the interpreter's — a correctness LOSS.
    CorrectnessLoss {
        /// A short description of the divergence.
        detail: String,
    },
    /// The backend cannot lower this program — a capability LOSS, with the reason.
    CapabilityLoss {
        /// The backend's own explanation (the unlowerable-node reason).
        reason: String,
    },
    /// The backend errored at run time (overflow, depth limit, compile/exec failure).
    RuntimeError {
        /// The error message.
        message: String,
    },
    /// The backend was skipped (toolchain absent / feature off) — not measured, not a verdict.
    Skipped {
        /// Why it was skipped.
        reason: String,
    },
    /// The interpreter baseline itself failed on this case — no comparison is possible (the harness
    /// records it loudly; the trusted base should not fail, so this is a corpus/engine red flag).
    BaselineFailed {
        /// What the interpreter reported.
        message: String,
    },
}

impl Verdict {
    /// A short status word for the report table.
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            Verdict::SpeedWin { .. } => "WIN",
            Verdict::SpeedNeutral { .. } => "neutral",
            Verdict::SpeedLoss { .. } => "LOSS (speed)",
            Verdict::CorrectnessLoss { .. } => "LOSS (correctness)",
            Verdict::CapabilityLoss { .. } => "LOSS (capability)",
            Verdict::RuntimeError { .. } => "error",
            Verdict::Skipped { .. } => "skipped",
            Verdict::BaselineFailed { .. } => "baseline-failed",
        }
    }

    /// Whether this verdict counts as a LOSS (any of the three loss kinds) — for the "where we're
    /// losing" rollup. A skip / error / baseline-failure is not counted as a loss (it is its own
    /// category), and neutral/win are not losses.
    #[must_use]
    pub fn is_loss(&self) -> bool {
        matches!(
            self,
            Verdict::SpeedLoss { .. }
                | Verdict::CorrectnessLoss { .. }
                | Verdict::CapabilityLoss { .. }
        )
    }

    /// Whether this verdict counts as a WIN (a measured speed win).
    #[must_use]
    pub fn is_win(&self) -> bool {
        matches!(self, Verdict::SpeedWin { .. })
    }

    /// The honest guarantee tag for this verdict. A measured speed band is `Empirical`; a capability
    /// loss / runtime error / skip is `Declared` (an observed fact about the run, not a trial mean).
    #[must_use]
    pub fn guarantee_tag(&self) -> &'static str {
        match self {
            Verdict::SpeedWin { .. }
            | Verdict::SpeedNeutral { .. }
            | Verdict::SpeedLoss { .. }
            | Verdict::CorrectnessLoss { .. } => "Empirical",
            Verdict::CapabilityLoss { .. }
            | Verdict::RuntimeError { .. }
            | Verdict::Skipped { .. }
            | Verdict::BaselineFailed { .. } => "Declared",
        }
    }
}

/// The neutral band half-width: a backend within `[1/(1+NEUTRAL), 1+NEUTRAL]` of the interpreter's
/// time is `Neutral` (neither a clear win nor loss). 0.10 ⇒ within ±10%. Reified here (no black box);
/// a different study can pick a different band and say so.
pub const NEUTRAL_BAND: f64 = 0.10;

/// Encode a ratio as fixed-point parts-per-thousand (so the verdict is exactly serializable and
/// comparable without float noise in the report). `2.5x` ⇒ `2500`.
#[must_use]
fn ratio_x1000(interp_ns: f64, backend_ns: f64) -> u64 {
    if backend_ns <= 0.0 {
        return 0;
    }
    let r = interp_ns / backend_ns;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (r * 1000.0).round().max(0.0) as u64
    }
}

/// Classify one (backend, case) pair. `interp` is the trusted-baseline outcome+timing; `other` is the
/// backend under test. The baseline must be the interpreter (asserted in debug).
///
/// `backend` is the identity of `other` (used only to derive an honest speed-loss *reason*, e.g.
/// process-spawn-bound — never to change the verdict itself).
#[must_use]
pub fn classify(
    backend: Backend,
    interp: (&Outcome, Option<Timing>),
    other: (&Outcome, Option<Timing>),
) -> Verdict {
    debug_assert!(
        !backend.is_baseline(),
        "classify compares a NON-baseline backend"
    );
    let (interp_outcome, interp_timing) = interp;
    let (other_outcome, other_timing) = other;

    // 1. The trusted base must have produced a value to compare against.
    let interp_val = match interp_outcome {
        Outcome::Value(v) => v,
        Outcome::Skipped(m) | Outcome::Unlowerable(m) | Outcome::Error(m) => {
            return Verdict::BaselineFailed {
                message: format!("interpreter did not produce a value: {m}"),
            };
        }
    };

    // 2. The backend's outcome decides the verdict category (never-silent).
    let other_val = match other_outcome {
        Outcome::Value(v) => v,
        Outcome::Skipped(reason) => {
            return Verdict::Skipped {
                reason: reason.clone(),
            }
        }
        Outcome::Unlowerable(reason) => {
            return Verdict::CapabilityLoss {
                reason: reason.clone(),
            }
        }
        Outcome::Error(message) => {
            return Verdict::RuntimeError {
                message: message.clone(),
            }
        }
    };

    // 3. Differential correctness: a divergence from the trusted base is a correctness LOSS — the
    //    worst loss, recorded plainly (a wrong answer, however fast, is wrong). We compare on the
    //    OBSERVABLE (repr+payload+guarantee / content-identity), excluding dynamic Meta provenance —
    //    the same equivalence the M-210 checker + the three-way differential test use. (A full `==`
    //    would flag a spurious loss when a compiled backend stamps `Provenance::Root` on a read-back
    //    value vs the interpreter's `Derived` chain, though the result is identical.)
    if !observable_eq(interp_val, other_val) {
        return Verdict::CorrectnessLoss {
            detail: format!(
                "backend result diverges from the interpreter on the observable \
                 (interp={interp_val:?}, backend={other_val:?})"
            ),
        };
    }

    // 4. Both agree — compare speed. Without both timings we cannot band the speed (Neutral, honest).
    let (Some(it), Some(ot)) = (interp_timing, other_timing) else {
        return Verdict::SpeedNeutral { ratio_x1000: 1000 };
    };
    let r = it.ns_per_call / ot.ns_per_call;
    let x1000 = ratio_x1000(it.ns_per_call, ot.ns_per_call);

    if r > 1.0 + NEUTRAL_BAND {
        Verdict::SpeedWin { ratio_x1000: x1000 }
    } else if r < 1.0 / (1.0 + NEUTRAL_BAND) {
        // A speed loss — attach the honest derivable reason where one applies.
        let reason = if backend.is_process_spawn_bound() {
            "process-spawn-bound: the per-invocation time is dominated by spawning a fresh native \
             process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the \
             in-process interpreter"
                .to_string()
        } else {
            "slower than the in-process interpreter on this case (measured; no target — VR-5)"
                .to_string()
        };
        Verdict::SpeedLoss {
            ratio_x1000: x1000,
            reason,
        }
    } else {
        Verdict::SpeedNeutral { ratio_x1000: x1000 }
    }
}

// ─────────────────────────────── Regression gates (M-859) ────────────────────────────────────────

/// One committed baseline timing: a (case, backend) `ns_per_call` snapshot from a prior run, plus the
/// host tag it was captured on (regression gates never compare across hosts — see
/// [`RegressionOutcome::HostMismatch`]).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BaselineEntry {
    /// The case id (matches [`crate::corpus::Case::id`]).
    pub case_id: String,
    /// The backend, by its stable label (matches [`Backend::label`] — kept as a string so the
    /// baseline JSON is self-describing without re-deriving the `Backend` serde mapping).
    pub backend: String,
    /// The committed `ns_per_call` this entry was captured at.
    pub ns_per_call: f64,
}

/// A committed regression baseline: a host tag (provenance — see [`crate::host_note_for_scaling`]'s
/// sibling convention) plus every `(case, backend)` timing captured on that host at commit time.
/// This is the artifact `regression_classify` compares a fresh run against — **not** a target to hit
/// (VR-5): a REGRESSION row means "slower than this baseline was, on this host", nothing more.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegressionBaseline {
    /// The host tag this baseline was captured on (e.g. `"x86_64-linux, 4 hw threads"`) — the
    /// portability guard: a baseline is only ever compared against a run tagged with the same host.
    pub host_tag: String,
    /// When the baseline was captured (a free-form provenance note — a date/commit, not parsed).
    pub captured: String,
    /// The trial count each entry's `ns_per_call` represents (documents the baseline's own honesty —
    /// it was itself an `Empirical` measurement with this many iterations, matching
    /// [`crate::timing::Timing::iters`]'s convention).
    pub trial_iters: u32,
    /// Every captured `(case, backend)` timing.
    pub entries: Vec<BaselineEntry>,
}

impl RegressionBaseline {
    /// Look up this baseline's `ns_per_call` for `(case_id, backend)`, if captured.
    #[must_use]
    pub fn lookup(&self, case_id: &str, backend: Backend) -> Option<f64> {
        self.entries
            .iter()
            .find(|e| e.case_id == case_id && e.backend == backend.label())
            .map(|e| e.ns_per_call)
    }

    /// Parse a baseline from its committed JSON text. Never-silent: a malformed baseline is an
    /// explicit `Err`, never treated as "no baseline" (which would silently disable regression
    /// gating).
    ///
    /// # Errors
    /// The `serde_json` parse error, verbatim.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Serialize this baseline to pretty JSON (the committed-artifact format).
    ///
    /// # Errors
    /// A `serde_json` serialization error (only possible on an unrepresentable float, e.g. NaN/∞ —
    /// never expected from a real measurement, but never silently substituted either).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// The regression-gate half-width: a fresh `ns_per_call` within `[1/(1+REGRESSION_BAND),
/// 1+REGRESSION_BAND]` of the baseline is a `Hold`; slower beyond that is a `Regression`, faster
/// beyond that is an `Improvement`. Reified (no black box) — a wider/narrower band is a different,
/// nameable study. Set wider than [`NEUTRAL_BAND`] (±10%) because host-to-host-run noise on a shared
/// CI-like box compounds two `Empirical` measurements (the baseline capture and this run), not one.
pub const REGRESSION_BAND: f64 = 0.20;

/// The regression-gate verdict for one `(case, backend)` pair — this run vs the committed baseline.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RegressionOutcome {
    /// Faster than the baseline beyond [`REGRESSION_BAND`] — a measured improvement.
    Improvement {
        /// `baseline_ns / this_run_ns` (`> 1`).
        ratio_x1000: u64,
    },
    /// Within [`REGRESSION_BAND`] of the baseline — holding steady.
    Hold {
        /// `baseline_ns / this_run_ns`.
        ratio_x1000: u64,
    },
    /// Slower than the baseline beyond [`REGRESSION_BAND`] — a measured regression, flagged.
    Regression {
        /// `baseline_ns / this_run_ns` (`< 1`).
        ratio_x1000: u64,
    },
    /// This run has no timing to compare (skip / capability loss / error / untimed) — not gated.
    NotTimed,
    /// The baseline has no entry for this `(case, backend)` — nothing to compare against yet (e.g. a
    /// newly added corpus case); never fabricated as a `Hold`.
    NoBaseline,
    /// The baseline's host tag does not match this run's host tag — refused rather than silently
    /// compared (a different host's numbers are not portable, VR-5).
    HostMismatch {
        /// The baseline's recorded host tag.
        baseline_host: String,
        /// This run's host tag.
        this_host: String,
    },
}

impl RegressionOutcome {
    /// A short status word for the report table.
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            RegressionOutcome::Improvement { .. } => "WIN (vs baseline)",
            RegressionOutcome::Hold { .. } => "hold",
            RegressionOutcome::Regression { .. } => "REGRESSION",
            RegressionOutcome::NotTimed => "not-timed",
            RegressionOutcome::NoBaseline => "no-baseline",
            RegressionOutcome::HostMismatch { .. } => "host-mismatch",
        }
    }

    /// Whether this is a flagged regression (the row the "regression gate" name refers to).
    #[must_use]
    pub fn is_regression(&self) -> bool {
        matches!(self, RegressionOutcome::Regression { .. })
    }
}

/// Classify one `(case, backend)` pair's fresh timing against the committed `baseline`, gated on the
/// host tag matching. `this_host` is the running host's tag (the caller's provenance note, e.g.
/// [`crate::host_note_for_scaling`]); `this_ns` is `None` when this run did not time the pair (a
/// skip / capability loss / error) — recorded as [`RegressionOutcome::NotTimed`], never compared as
/// if it were zero.
#[must_use]
pub fn regression_classify(
    baseline: &RegressionBaseline,
    this_host: &str,
    case_id: &str,
    backend: Backend,
    this_ns: Option<f64>,
) -> RegressionOutcome {
    if baseline.host_tag != this_host {
        return RegressionOutcome::HostMismatch {
            baseline_host: baseline.host_tag.clone(),
            this_host: this_host.to_string(),
        };
    }
    let Some(this_ns) = this_ns else {
        return RegressionOutcome::NotTimed;
    };
    let Some(baseline_ns) = baseline.lookup(case_id, backend) else {
        return RegressionOutcome::NoBaseline;
    };
    if this_ns <= 0.0 || baseline_ns <= 0.0 {
        // A non-positive timing is a measurement anomaly, never divided against — treated as
        // not-timed rather than fabricating an infinite/NaN ratio (G2).
        return RegressionOutcome::NotTimed;
    }
    let r = baseline_ns / this_ns; // > 1 means this run is FASTER than the baseline.
    let x1000 = ratio_x1000(baseline_ns, this_ns);
    if r > 1.0 + REGRESSION_BAND {
        RegressionOutcome::Improvement { ratio_x1000: x1000 }
    } else if r < 1.0 / (1.0 + REGRESSION_BAND) {
        RegressionOutcome::Regression { ratio_x1000: x1000 }
    } else {
        RegressionOutcome::Hold { ratio_x1000: x1000 }
    }
}
