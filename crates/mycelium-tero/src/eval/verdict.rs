//! The Layer-2 **gate verdict** + the **latency regression classifier** (M-1018) — an in-crate mirror
//! of `mycelium-bench::verdict`'s shape (we do **not** depend on `mycelium-bench`: it drags in a large
//! backend/timing subtree this crate has no use for). Pure (no I/O, no timing of its own), so both the
//! gate decision and the latency classification are deterministic and exhaustively unit-testable.
//!
//! **THE GATE (DN-87 §6.1, G2/VR-5):** until Layer 2 *measurably* beats/complements the Layer-1
//! baseline, the improved-on-RAG claim stays aspiration and the system serves Layer-1 answers. The
//! gate is **Closed by default** and opens **iff** Layer 2 beats Layer-1 correctness@1 beyond the
//! reified band **and** keeps provenance fidelity at 1.0 **and** stays within the latency band. A
//! **Closed gate is a first-class, honest, successful outcome** — never a failure to explain away.

use serde::{Deserialize, Serialize};

/// The regression band half-width, reified (no black box) — matches `mycelium-bench`'s
/// `REGRESSION_BAND`. A value within `[1/(1+band), 1+band]` of its comparator is holding steady;
/// beyond that it is a measured improvement/regression. `0.20` (±20%) absorbs the compounded noise of
/// two `Empirical` single-machine measurements.
pub const REGRESSION_BAND: f64 = 0.20;

/// Encode a ratio as fixed-point parts-per-thousand (exact, comparable without float noise in the
/// serialized artifact). `2.5x` ⇒ `2500`.
#[must_use]
fn ratio_x1000(numer: f64, denom: f64) -> u64 {
    if denom <= 0.0 {
        return 0;
    }
    let r = numer / denom;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (r * 1000.0).round().max(0.0) as u64
    }
}

// ─────────────────────────────── latency regression baseline ─────────────────────────────────────

/// One committed latency snapshot: a `(case, system)` `ns_per_call` from a prior run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyEntry {
    /// The case id (here: the eval question set id / `"aggregate"`).
    pub case_id: String,
    /// Which system produced it (`"layer1"` / `"layer2"`).
    pub system: String,
    /// The committed `ns_per_call` this entry was captured at.
    pub ns_per_call: f64,
}

/// A committed latency baseline: a host tag (the portability guard) plus every captured `(case,
/// system)` timing. **Not** a target to hit (VR-5) — a regression row means "slower than this baseline
/// was, on this host", nothing more.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyBaseline {
    /// The host tag this baseline was captured on — a baseline is only ever compared against a run
    /// tagged with the same host ([`LatencyOutcome::HostMismatch`] otherwise).
    pub host_tag: String,
    /// When it was captured (a free-form provenance note — a date/commit, not parsed).
    pub captured: String,
    /// The trial iteration count each entry's `ns_per_call` represents (the baseline's own honesty:
    /// it was itself an `Empirical` measurement with this many iterations).
    pub trial_iters: u32,
    /// Every captured `(case, system)` timing.
    pub entries: Vec<LatencyEntry>,
}

impl LatencyBaseline {
    /// Look up this baseline's `ns_per_call` for `(case_id, system)`, if captured.
    #[must_use]
    pub fn lookup(&self, case_id: &str, system: &str) -> Option<f64> {
        self.entries
            .iter()
            .find(|e| e.case_id == case_id && e.system == system)
            .map(|e| e.ns_per_call)
    }

    /// Parse a baseline from its committed JSON text. Never-silent: a malformed baseline is an
    /// explicit `Err`, never treated as "no baseline" (which would silently disable the gate).
    ///
    /// # Errors
    /// The `serde_json` parse error, verbatim.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Serialize this baseline to pretty JSON (the committed-artifact format).
    ///
    /// # Errors
    /// A `serde_json` serialization error (only on an unrepresentable float — never expected from a
    /// real measurement, but never silently substituted either).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// The latency-gate verdict for one `(case, system)` pair — this run vs the committed baseline.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LatencyOutcome {
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
    /// This run has no timing to compare — not gated.
    NotTimed,
    /// The baseline has no entry for this `(case, system)` — nothing to compare against yet.
    NoBaseline,
    /// The baseline's host tag does not match this run's — refused rather than silently compared (a
    /// different host's numbers are not portable, VR-5).
    HostMismatch {
        /// The baseline's recorded host tag.
        baseline_host: String,
        /// This run's host tag.
        this_host: String,
    },
}

impl LatencyOutcome {
    /// A short status word for the report.
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            LatencyOutcome::Improvement { .. } => "improvement",
            LatencyOutcome::Hold { .. } => "hold",
            LatencyOutcome::Regression { .. } => "REGRESSION",
            LatencyOutcome::NotTimed => "not-timed",
            LatencyOutcome::NoBaseline => "no-baseline",
            LatencyOutcome::HostMismatch { .. } => "host-mismatch",
        }
    }

    /// Whether this is a flagged regression.
    #[must_use]
    pub fn is_regression(&self) -> bool {
        matches!(self, LatencyOutcome::Regression { .. })
    }
}

/// Classify one `(case, system)` pair's fresh timing against the committed `baseline`, gated on the
/// host tag matching. `this_ns` is `None` when this run did not time the pair.
#[must_use]
pub fn latency_classify(
    baseline: &LatencyBaseline,
    this_host: &str,
    case_id: &str,
    system: &str,
    this_ns: Option<f64>,
) -> LatencyOutcome {
    if baseline.host_tag != this_host {
        return LatencyOutcome::HostMismatch {
            baseline_host: baseline.host_tag.clone(),
            this_host: this_host.to_string(),
        };
    }
    let Some(this_ns) = this_ns else {
        return LatencyOutcome::NotTimed;
    };
    let Some(baseline_ns) = baseline.lookup(case_id, system) else {
        return LatencyOutcome::NoBaseline;
    };
    if this_ns <= 0.0 || baseline_ns <= 0.0 {
        return LatencyOutcome::NotTimed;
    }
    let r = baseline_ns / this_ns; // > 1 means this run is FASTER than the baseline.
    let x1000 = ratio_x1000(baseline_ns, this_ns);
    if r > 1.0 + REGRESSION_BAND {
        LatencyOutcome::Improvement { ratio_x1000: x1000 }
    } else if r < 1.0 / (1.0 + REGRESSION_BAND) {
        LatencyOutcome::Regression { ratio_x1000: x1000 }
    } else {
        LatencyOutcome::Hold { ratio_x1000: x1000 }
    }
}

// ─────────────────────────────────────── the gate ────────────────────────────────────────────────

/// The measured evidence a gate verdict rests on — every number, with its denominator, recorded so
/// the verdict is fully re-derivable (never a bare Open/Closed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateEvidence {
    /// Number of questions graded (the common denominator for the correctness rates).
    pub questions: usize,
    /// The `k` used for correctness@k.
    pub k: usize,
    /// Layer-1 baseline correctness@1 (fraction of questions whose gold anchor is top-1).
    pub l1_correct_at_1: f64,
    /// Layer-2 correctness@1.
    pub l2_correct_at_1: f64,
    /// Layer-1 baseline correctness@k.
    pub l1_correct_at_k: f64,
    /// Layer-2 correctness@k.
    pub l2_correct_at_k: f64,
    /// Layer-2 provenance fidelity (fraction of returned citations resolving to a real Layer-1 row —
    /// must be 1.0 for the gate to open).
    pub l2_provenance: f64,
    /// Layer-1 baseline latency (ns/query, Empirical, single-machine).
    pub l1_ns_per_query: f64,
    /// Layer-2 latency (ns/query).
    pub l2_ns_per_query: f64,
    /// The reified decision band (`REGRESSION_BAND`).
    pub band: f64,
}

/// The append-only gate verdict. **Closed by default**; opens only on a real, measured Layer-2 win
/// that keeps provenance and latency honest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "kebab-case")]
pub enum GateVerdict {
    /// Layer 2 has **not** (yet) beaten/complemented the baseline on the gate criteria — the system
    /// keeps serving Layer-1 answers; the improved-on-RAG claim stays aspiration. A first-class,
    /// honest, expected outcome for this corpus (G2/VR-5).
    Closed {
        /// Which criteria were not met, in author-facing terms.
        reason: String,
        /// The measured evidence.
        evidence: GateEvidence,
    },
    /// Layer 2 measurably beat the baseline on correctness beyond the band **and** kept provenance at
    /// 1.0 **and** stayed within the latency band — the gate opens (Layer 2 may be served).
    Open {
        /// The measured evidence.
        evidence: GateEvidence,
    },
}

impl GateVerdict {
    /// Whether the gate opened.
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(self, GateVerdict::Open { .. })
    }

    /// A short status word.
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            GateVerdict::Closed { .. } => "CLOSED (serving Layer-1)",
            GateVerdict::Open { .. } => "OPEN (Layer-2 eligible)",
        }
    }

    /// The measured evidence, whichever arm.
    #[must_use]
    pub fn evidence(&self) -> &GateEvidence {
        match self {
            GateVerdict::Closed { evidence, .. } | GateVerdict::Open { evidence } => evidence,
        }
    }
}

/// Decide the gate from the measured `evidence` (pure — the sole gate decision point). Opens **iff**
/// Layer-2 correctness@1 beats Layer-1's beyond the band, provenance is exactly 1.0, and Layer-2
/// latency is within the band of Layer-1's. Otherwise Closed, with the failing criteria named.
#[must_use]
pub fn decide_gate(evidence: GateEvidence) -> GateVerdict {
    let band = evidence.band;
    let correctness_beats = evidence.l2_correct_at_1 > evidence.l1_correct_at_1 * (1.0 + band);
    let provenance_ok = evidence.l2_provenance >= 1.0 - f64::EPSILON;
    // Latency "within band": Layer-2 not a large regression vs Layer-1 (complement, don't regress).
    let latency_ok = evidence.l2_ns_per_query <= evidence.l1_ns_per_query * (1.0 + band);

    if correctness_beats && provenance_ok && latency_ok {
        return GateVerdict::Open { evidence };
    }

    let mut reasons: Vec<String> = Vec::new();
    if !correctness_beats {
        reasons.push(format!(
            "Layer-2 correctness@1 {:.4} did not beat Layer-1 {:.4} beyond the {:.0}% band",
            evidence.l2_correct_at_1,
            evidence.l1_correct_at_1,
            band * 100.0
        ));
    }
    if !provenance_ok {
        reasons.push(format!(
            "Layer-2 provenance fidelity {:.4} is below the required 1.0",
            evidence.l2_provenance
        ));
    }
    if !latency_ok {
        reasons.push(format!(
            "Layer-2 latency {:.0}ns/query exceeds Layer-1 {:.0}ns/query beyond the {:.0}% band",
            evidence.l2_ns_per_query,
            evidence.l1_ns_per_query,
            band * 100.0
        ));
    }
    GateVerdict::Closed {
        reason: reasons.join("; "),
        evidence,
    }
}
