//! Deterministic **report emission** — markdown (human) + JSON (machine), the dual projection (G11).
//! The report carries: run metadata + caveats, the per-backend/per-case numbers, the explicit
//! WIN/LOSS/REGRESSION table, an explicit **"where we're losing"** section (capability + correctness +
//! speed losses), and the ingested LLM-harness section.
//!
//! **Honesty stamped in:** the harness *plumbing* is `Empirical` (test-verified). Every measured
//! number is `Empirical` (with its trial accounting); a capability loss / skip is `Declared`. There
//! is **no pre-written performance target** (VR-5): the verdicts are whatever was measured. The
//! microbench caveats (warmup, process-spawn cost, debug-vs-release) are stated, not buried.

use std::fmt::Write as _;

use crate::backend::Backend;
use crate::llm::LlmReport;
use crate::measure::RunRecord;
use crate::scaling::{ScalingOutcome, ScalingRun};
use crate::verdict::{
    regression_classify, RegressionBaseline, RegressionOutcome, Verdict, NEUTRAL_BAND,
    REGRESSION_BAND,
};

/// Everything the report needs: the run record, optional ingested LLM-harness report, and run
/// metadata. Serializable verbatim to the JSON projection.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Report {
    /// Schema/tool identity.
    pub tool: &'static str,
    /// The build profile the numbers were taken under (`release` is the only honest one).
    pub profile: &'static str,
    /// Whether the `mlir-dialect` feature was compiled in (affects which backends could run).
    pub mlir_dialect_feature: bool,
    /// A short note on the host (best-effort; for provenance only).
    pub host_note: String,
    /// The honesty posture: the lattice + the rule that governs the verdict tags.
    pub honesty: Honesty,
    /// The neutral band half-width used to classify speed (reified — no black box).
    pub neutral_band: f64,
    /// The execution-backend run.
    pub run: RunRecord,
    /// The ingested LLM-harness section (provenance + per-validation rows), if a report was found.
    pub llm: Option<LlmSection>,
    /// The multicore scaling run (M-859), if one was taken alongside the single-core measurements.
    /// `None` for a single-core-only report (e.g. the fast pre-commit / test-only reports) — the
    /// scaling section is additive, never required for the single-core WIN/LOSS table to be valid.
    pub scaling: Option<ScalingRun>,
    /// The regression-gate section (M-859): this run's `ns_per_call` vs the committed baseline,
    /// `None` when no baseline was supplied (regression gating is opt-in per invocation).
    pub regression: Option<RegressionSection>,
}

/// The regression-gate section: the committed baseline this run was compared against, plus every
/// `(case, backend)` outcome. Built by [`Report::with_regression_gate`] — never fabricated when no
/// baseline is on hand (the `regression` field on [`Report`] just stays `None`, G2).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RegressionSection {
    /// The baseline's own host tag + capture provenance (carried through so the report is
    /// self-contained — a reader does not need the baseline file to see what was compared).
    pub baseline_host_tag: String,
    /// The baseline's capture provenance note.
    pub baseline_captured: String,
    /// The regression-gate band half-width used (reified — no black box).
    pub regression_band: f64,
    /// Every `(case, backend)` regression-gate row.
    pub rows: Vec<RegressionRow>,
}

/// One `(case, backend)` regression-gate row.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RegressionRow {
    /// The case id.
    pub case_id: String,
    /// The backend's stable label.
    pub backend: &'static str,
    /// This run's `ns_per_call`, if timed.
    pub this_ns: Option<f64>,
    /// The baseline's `ns_per_call`, if this pair was in the baseline.
    pub baseline_ns: Option<f64>,
    /// The classified outcome.
    pub outcome: RegressionOutcome,
}

/// The honesty posture block stamped into the report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Honesty {
    /// The guarantee lattice, strongest-first.
    pub lattice: [&'static str; 4],
    /// The rule the verdict tags obey.
    pub rule: &'static str,
}

impl Default for Honesty {
    fn default() -> Self {
        Self {
            lattice: ["Exact", "Proven", "Empirical", "Declared"],
            rule: "Every measured number is Empirical (a trial mean with its trial count + spread); a \
                   capability loss / skip / runtime error is Declared. No verdict is Proven or Exact, \
                   and no performance target is pre-written (VR-5). A differential divergence from the \
                   trusted interpreter is a recorded correctness LOSS; an unlowerable node is a \
                   recorded capability LOSS (G2 — never omitted).",
        }
    }
}

/// The LLM-harness section of the unified report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LlmSection {
    /// The source file the report was read from.
    pub source_path: String,
    /// Whether the source was the committed SYNTHETIC sample (vs a real run found in the reports dir).
    pub is_synthetic: bool,
    /// The provenance one-liner.
    pub provenance: String,
    /// Per-validation rows (id, status, tag, latency, tokens, message).
    pub validations: Vec<LlmValidationRow>,
}

/// One per-validation row in the LLM section.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LlmValidationRow {
    /// Validation id.
    pub id: String,
    /// Status (PASS / mock-PASS / SKIP / FAIL).
    pub status: String,
    /// Honest guarantee tag.
    pub guarantee_tag: Option<String>,
    /// Wall-clock latency (seconds), if recorded.
    pub wall_seconds: Option<f64>,
    /// (prompt, generated) token counts, if recorded.
    pub prompt_tokens: Option<u64>,
    /// Generated token count, if recorded.
    pub generated_tokens: Option<u64>,
    /// The one-line message.
    pub message: String,
}

impl LlmSection {
    /// Build the section from a parsed report + its source path / synthetic flag.
    #[must_use]
    pub fn from_report(report: &LlmReport, source_path: String, is_synthetic: bool) -> Self {
        let validations = report
            .results
            .iter()
            .map(|v| {
                let (p, g) = match v.token_counts() {
                    Some((p, g)) => (p, g),
                    None => (None, None),
                };
                LlmValidationRow {
                    id: v.id.clone(),
                    status: v.status.clone(),
                    guarantee_tag: v.guarantee_tag.clone(),
                    wall_seconds: v.wall_seconds(),
                    prompt_tokens: p,
                    generated_tokens: g,
                    message: v.message.clone(),
                }
            })
            .collect();
        Self {
            source_path,
            is_synthetic,
            provenance: report.provenance(),
            validations,
        }
    }

    /// Build the section from the schema-agnostic [`crate::llm::ParsedLlmSection`] produced by
    /// [`crate::llm::parse_any_llm_json`]. This is the preferred entry-point when the caller
    /// does not know in advance whether the JSON is a bench-harness or a Grok-harness report.
    #[must_use]
    pub fn from_parsed(parsed: crate::llm::ParsedLlmSection) -> Self {
        Self {
            source_path: parsed.source_path,
            is_synthetic: parsed.is_synthetic,
            provenance: parsed.provenance,
            validations: parsed.validations,
        }
    }
}

/// A roll-up of losses for the "where we're losing" section.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LossRollup {
    /// (case, backend, reason) capability losses.
    pub capability: Vec<(String, &'static str, String)>,
    /// (case, backend, detail) correctness losses (divergences) — the most serious.
    pub correctness: Vec<(String, &'static str, String)>,
    /// (case, backend, ratio_x1000, reason) speed losses.
    pub speed: Vec<(String, &'static str, u64, String)>,
}

impl Report {
    /// Build the [`RegressionSection`] for this run against `baseline`, and attach it (`self.regression
    /// = Some(...)`). Every `(case, backend)` pair in `self.run` gets a row — including pairs the
    /// baseline has no entry for ([`RegressionOutcome::NoBaseline`]) and pairs this run did not time
    /// ([`RegressionOutcome::NotTimed`]) — never a silently-dropped row (G2). Consumes and returns
    /// `self` for convenient chaining at the call site (`bin/bench.rs`).
    /// **Host tag note:** the comparison uses the caller-supplied `this_host_tag` — the *canonical,
    /// bare* host tag ([`crate::host_tag`], e.g. `"x86_64-linux, 4 hw threads"`), **not**
    /// [`Report::host_note`] (which wraps the same information in report-header prose, `"host: ...
    /// (provenance only)"`). Passing the prose form here would make every gate row a spurious
    /// [`RegressionOutcome::HostMismatch`] even on the exact host the baseline was captured on — a
    /// real bug caught while dogfooding this module (M-859) before this doc note existed.
    #[must_use]
    pub fn with_regression_gate(
        mut self,
        this_host_tag: &str,
        baseline: &RegressionBaseline,
    ) -> Self {
        let mut rows = Vec::new();
        for case in &self.run.cases {
            for b in &case.backends {
                let this_ns = b.timing.map(|t| t.ns_per_call);
                let baseline_ns = baseline.lookup(&case.id, b.backend);
                let outcome =
                    regression_classify(baseline, this_host_tag, &case.id, b.backend, this_ns);
                rows.push(RegressionRow {
                    case_id: case.id.clone(),
                    backend: b.backend.label(),
                    this_ns,
                    baseline_ns,
                    outcome,
                });
            }
        }
        self.regression = Some(RegressionSection {
            baseline_host_tag: baseline.host_tag.clone(),
            baseline_captured: baseline.captured.clone(),
            regression_band: REGRESSION_BAND,
            rows,
        });
        self
    }

    /// Roll up every loss across the run for the "where we're losing" section.
    #[must_use]
    pub fn loss_rollup(&self) -> LossRollup {
        let mut roll = LossRollup::default();
        for case in &self.run.cases {
            for b in &case.backends {
                match &b.verdict {
                    Verdict::CapabilityLoss { reason } => {
                        roll.capability
                            .push((case.id.clone(), b.backend.label(), reason.clone()));
                    }
                    Verdict::CorrectnessLoss { detail } => {
                        roll.correctness
                            .push((case.id.clone(), b.backend.label(), detail.clone()));
                    }
                    Verdict::SpeedLoss {
                        ratio_x1000,
                        reason,
                    } => {
                        roll.speed.push((
                            case.id.clone(),
                            b.backend.label(),
                            *ratio_x1000,
                            reason.clone(),
                        ));
                    }
                    _ => {}
                }
            }
        }
        roll
    }

    /// Count (wins, speed-losses, correctness-losses, capability-losses, skips) across the run.
    #[must_use]
    pub fn tallies(&self) -> Tallies {
        let mut t = Tallies::default();
        for case in &self.run.cases {
            for b in &case.backends {
                match &b.verdict {
                    Verdict::SpeedWin { .. } => t.wins += 1,
                    Verdict::SpeedNeutral { .. } => t.neutral += 1,
                    Verdict::SpeedLoss { .. } => t.speed_losses += 1,
                    Verdict::CorrectnessLoss { .. } => t.correctness_losses += 1,
                    Verdict::CapabilityLoss { .. } => t.capability_losses += 1,
                    Verdict::RuntimeError { .. } => t.errors += 1,
                    Verdict::Skipped { .. } => t.skips += 1,
                    Verdict::BaselineFailed { .. } => t.baseline_failures += 1,
                }
            }
        }
        t
    }

    /// The machine-readable JSON projection (pretty-printed, deterministic).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// The human-readable markdown projection (deterministic — same run ⇒ same bytes, modulo the
    /// measured numbers themselves).
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        self.write_header(&mut s);
        self.write_winloss_table(&mut s);
        self.write_per_backend_numbers(&mut s);
        self.write_losses(&mut s);
        self.write_scaling(&mut s);
        self.write_regression(&mut s);
        self.write_llm(&mut s);
        // Normalize the trailer to exactly one newline so the emitted markdown is lint-clean
        // (markdownlint MD012 — no multiple consecutive blank lines at EOF) regardless of which
        // section wrote the last line. Deterministic: idempotent on already-clean output.
        let trimmed = s.trim_end();
        s.truncate(trimmed.len());
        s.push('\n');
        s
    }

    fn write_header(&self, s: &mut String) {
        let t = self.tallies();
        let _ = writeln!(s, "# Mycelium honest benchmark report\n");
        let _ = writeln!(
            s,
            "> Tool `{}` — profile `{}` — `mlir-dialect` feature: {} — {}\n",
            self.tool,
            self.profile,
            if self.mlir_dialect_feature {
                "ON"
            } else {
                "OFF"
            },
            self.host_note
        );
        let _ = writeln!(
            s,
            "Guarantee lattice: `{}`.\n\n**Honesty:** {}\n",
            self.honesty.lattice.join(" ⊐ "),
            self.honesty.rule
        );
        let _ = writeln!(
            s,
            "Speed band: a backend within ±{:.0}% of the interpreter is *neutral*; faster is a \
             **WIN**, slower a **LOSS (speed)**. Trusted baseline: the **interpreter** (in-process; \
             NFR-7/ADR-007).\n",
            self.neutral_band * 100.0
        );
        let _ = writeln!(
            s,
            "Tally across the run: **{} win(s)**, {} neutral, **{} speed-loss(es)**, **{} \
             correctness-loss(es)**, **{} capability-loss(es)**, {} runtime-error(s), {} skip(s){}.\n",
            t.wins,
            t.neutral,
            t.speed_losses,
            t.correctness_losses,
            t.capability_losses,
            t.errors,
            t.skips,
            if t.baseline_failures > 0 {
                format!(", **{} BASELINE FAILURE(S) — investigate**", t.baseline_failures)
            } else {
                String::new()
            }
        );
        let _ = writeln!(
            s,
            "**Microbench caveats (honest):** numbers are warmup + min-mean over batches via \
             `std::time::Instant` (no `criterion`). The compiled native paths (`direct-llvm`, \
             `mlir-dialect`) are **process-spawn-bound**: each invocation execs a fresh native \
             artifact, so for a trivial kernel the per-invocation figure is spawn-dominated, **not** \
             kernel compute (the honest M-602/E1 finding — surfaced, not buried). `jit` runs \
             in-process (`dlopen`) so it is not spawn-bound. A debug build is refused for perf \
             numbers.\n"
        );
    }

    fn write_winloss_table(&self, s: &mut String) {
        let _ = writeln!(s, "## WIN / LOSS / regression table\n");
        let _ = writeln!(
            s,
            "Each non-baseline backend vs the interpreter, per case. `ratio` is `interp / backend` \
             (>1 ⇒ backend faster). Tag is per-row.\n"
        );
        let _ = writeln!(
            s,
            "| case | fragment | backend | verdict | ratio | tag | reason / detail |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|");
        for case in &self.run.cases {
            for b in &case.backends {
                let (ratio, reason) = verdict_ratio_reason(&b.verdict);
                let _ = writeln!(
                    s,
                    "| `{}` | {} | `{}` | {} | {} | {} | {} |",
                    case.id,
                    case.fragment.label(),
                    b.backend.label(),
                    b.verdict.status(),
                    ratio,
                    b.verdict.guarantee_tag(),
                    md_escape(&reason),
                );
            }
        }
        let _ = writeln!(s);
    }

    fn write_per_backend_numbers(&self, s: &mut String) {
        let _ = writeln!(s, "## Per-case timings (ns/call, Empirical)\n");
        let _ = writeln!(
            s,
            "Interpreter baseline + each backend that produced a timed value. The best ns/call is \
             shown; the worst/best spread (a noise flag) is in the JSON projection \
             (`ns_per_call_worst`), omitted from this compact table. `—` = not timed (skip / \
             capability loss / error).\n"
        );
        let _ = writeln!(
            s,
            "| case | interp ns | aot-env ns | jit ns | direct-llvm ns | mlir-dialect ns |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|");
        for case in &self.run.cases {
            let base = case.baseline_ns.map_or_else(|| "—".to_string(), fmt_ns);
            let cell = |backend: Backend| -> String {
                case.backends
                    .iter()
                    .find(|b| b.backend == backend)
                    .and_then(|b| b.timing)
                    .map_or_else(|| "—".to_string(), |t| fmt_ns(t.ns_per_call))
            };
            let _ = writeln!(
                s,
                "| `{}` | {} | {} | {} | {} | {} |",
                case.id,
                base,
                cell(Backend::AotEnv),
                cell(Backend::Jit),
                cell(Backend::DirectLlvm),
                cell(Backend::MlirDialect),
            );
        }
        let _ = writeln!(s);
        // The compiled backends' one-time compile cost is reported separately so the per-run figures
        // above stay honest (compile cost is amortized over many runs, not charged per invocation).
        let mut any_compile = false;
        for case in &self.run.cases {
            for b in &case.backends {
                if let Some(c) = b.compile_ns {
                    if !any_compile {
                        let _ = writeln!(
                            s,
                            "One-time compile cost (emit IR → toolchain → native, NOT in the per-run \
                             figures above):\n"
                        );
                        any_compile = true;
                    }
                    let _ = writeln!(
                        s,
                        "- `{}` / `{}`: {} (one-time)",
                        case.id,
                        b.backend.label(),
                        fmt_ns(c)
                    );
                }
            }
        }
        if any_compile {
            let _ = writeln!(s);
        }
    }

    fn write_losses(&self, s: &mut String) {
        let roll = self.loss_rollup();
        let _ = writeln!(s, "## Where we're losing (explicit)\n");
        if roll.capability.is_empty() && roll.correctness.is_empty() && roll.speed.is_empty() {
            let _ = writeln!(
                s,
                "No losses recorded in this run. (That is itself a measurement, not a target — \
                 VR-5.)\n"
            );
            return;
        }

        if !roll.correctness.is_empty() {
            let _ = writeln!(
                s,
                "### Correctness losses (divergence from the trusted interpreter — most serious)\n"
            );
            let _ = writeln!(s, "| case | backend | divergence |");
            let _ = writeln!(s, "|---|---|---|");
            for (case, backend, detail) in &roll.correctness {
                let _ = writeln!(s, "| `{}` | `{}` | {} |", case, backend, md_escape(detail));
            }
            let _ = writeln!(s);
        }

        if !roll.capability.is_empty() {
            let _ = writeln!(
                s,
                "### Capability losses (a backend cannot lower the program — the reason, never \
                 omitted, G2)\n"
            );
            let _ = writeln!(s, "| case | backend | reason |");
            let _ = writeln!(s, "|---|---|---|");
            for (case, backend, reason) in &roll.capability {
                let _ = writeln!(s, "| `{}` | `{}` | {} |", case, backend, md_escape(reason));
            }
            let _ = writeln!(s);
        }

        if !roll.speed.is_empty() {
            let _ = writeln!(
                s,
                "### Speed losses (slower than the in-process interpreter — measured, with the \
                 derivable reason)\n"
            );
            let _ = writeln!(s, "| case | backend | ratio (interp/backend) | reason |");
            let _ = writeln!(s, "|---|---|---|---|");
            for (case, backend, ratio_x1000, reason) in &roll.speed {
                let _ = writeln!(
                    s,
                    "| `{}` | `{}` | {} | {} |",
                    case,
                    backend,
                    fmt_ratio(*ratio_x1000),
                    md_escape(reason),
                );
            }
            let _ = writeln!(s);
        }
    }

    fn write_scaling(&self, s: &mut String) {
        let _ = writeln!(s, "## Multicore scaling (M-859)\n");
        let Some(run) = &self.scaling else {
            let _ = writeln!(
                s,
                "No scaling run attached to this report (scaling is opt-in per invocation — the \
                 single-core WIN/LOSS numbers above stand on their own). This section is empty, not \
                 synthesized.\n"
            );
            return;
        };
        let _ = writeln!(
            s,
            "> {} — worker counts exercised: {:?}. Speedup is `t(1 worker) / t(N workers)` per job \
             (min-of-batches); *ideal* is linear (`speedup == N`). Every figure is **Empirical** \
             (measured, trial-counted — VR-5); no scaling target is pre-written. The **Amdahl serial \
             fraction** column is a coarse two-point fit (1-worker and max-worker samples only), also \
             **Empirical** — a derived statistic, not a proof.\n",
            run.host_note, run.worker_counts,
        );
        let _ = writeln!(
            s,
            "| case | backend | spawn-bound | speedup @ max workers | ideal (linear) | Amdahl serial \
             fraction | note |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|");
        for point in &run.points {
            let spawn_bound = if point.backend.is_process_spawn_bound() {
                "yes"
            } else {
                "no"
            };
            let (speedup_cell, ideal_cell, amdahl_cell, note) = match &point.outcome {
                ScalingOutcome::Measured(_) => {
                    let sp = point.speedups();
                    let max = sp
                        .as_ref()
                        .and_then(|v| v.iter().max_by_key(|(w, _, _)| *w));
                    let (speedup, ideal) = max
                        .map_or(("—".to_string(), "—".to_string()), |(_, s, i)| {
                            (format!("{s:.2}x"), format!("{i:.2}x"))
                        });
                    let amdahl = point
                        .amdahl_serial_fraction()
                        .map_or_else(|| "—".to_string(), |f| format!("{f:.3}"));
                    (
                        speedup,
                        ideal,
                        amdahl,
                        if point.backend.is_process_spawn_bound() {
                            "spawn-dominated: contention here is mostly OS process creation, not \
                             kernel compute (M-602/E1 carried into the scaling curve)"
                                .to_string()
                        } else {
                            String::new()
                        },
                    )
                }
                ScalingOutcome::Skipped(reason) => (
                    "—".to_string(),
                    "—".to_string(),
                    "—".to_string(),
                    reason.clone(),
                ),
                ScalingOutcome::Unmeasurable(reason) => (
                    "—".to_string(),
                    "—".to_string(),
                    "—".to_string(),
                    reason.clone(),
                ),
            };
            let _ = writeln!(
                s,
                "| `{}` | `{}` | {} | {} | {} | {} | {} |",
                point.case_id,
                point.backend.label(),
                spawn_bound,
                speedup_cell,
                ideal_cell,
                amdahl_cell,
                md_escape(&note),
            );
        }
        let _ = writeln!(s);
    }

    fn write_regression(&self, s: &mut String) {
        let _ = writeln!(s, "## Regression gate vs committed baseline (M-859)\n");
        let Some(sec) = &self.regression else {
            let _ = writeln!(
                s,
                "No baseline supplied for this report — the regression gate is opt-in \
                 (`Report::with_regression_gate`). This section is empty, not synthesized.\n"
            );
            return;
        };
        let _ = writeln!(
            s,
            "> Baseline captured on `{}` ({}). Band: ±{:.0}% (wider than the single-run neutral band \
             — two independent Empirical measurements compound noise). `REGRESSION` rows are the \
             ones to look at; `host-mismatch` means this run's host does not match the baseline's, so \
             the comparison was refused rather than silently taken (VR-5: a different host's numbers \
             are not portable).\n",
            sec.baseline_host_tag,
            sec.baseline_captured,
            sec.regression_band * 100.0,
        );
        let regressions: Vec<_> = sec
            .rows
            .iter()
            .filter(|r| r.outcome.is_regression())
            .collect();
        let _ = writeln!(
            s,
            "**{} regression(s)** flagged out of {} gated row(s).\n",
            regressions.len(),
            sec.rows.len()
        );
        let _ = writeln!(
            s,
            "| case | backend | baseline ns | this run ns | verdict |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|");
        for row in &sec.rows {
            let baseline_cell = row.baseline_ns.map_or_else(|| "—".to_string(), fmt_ns);
            let this_cell = row.this_ns.map_or_else(|| "—".to_string(), fmt_ns);
            let _ = writeln!(
                s,
                "| `{}` | `{}` | {} | {} | {} |",
                row.case_id,
                row.backend,
                baseline_cell,
                this_cell,
                row.outcome.status(),
            );
        }
        let _ = writeln!(s);
    }

    fn write_llm(&self, s: &mut String) {
        let _ = writeln!(s, "## LLM-harness leverage (KC-2 / SC-5b)\n");
        match &self.llm {
            None => {
                let _ = writeln!(
                    s,
                    "No LLM-harness report found to ingest (no `*-report.json` in the reports dir, \
                     and the committed synthetic sample was not reachable). This section is empty — \
                     not synthesized.\n"
                );
            }
            Some(sec) => {
                let label = if sec.is_synthetic {
                    "**SYNTHETIC sample** (a fixture run — NOT real model quality; never treated as \
                     evidence, per the harness's own VR-5/V-03 rule)"
                } else {
                    "real model run"
                };
                let _ = writeln!(s, "Source: `{}` — {}.\n", sec.source_path, label);
                let _ = writeln!(s, "> {}\n", sec.provenance);
                let _ = writeln!(
                    s,
                    "| validation | status | tag | latency (s) | prompt tok | gen tok | message |"
                );
                let _ = writeln!(s, "|---|---|---|---|---|---|---|");
                for v in &sec.validations {
                    let _ = writeln!(
                        s,
                        "| `{}` | {} | {} | {} | {} | {} | {} |",
                        v.id,
                        v.status,
                        v.guarantee_tag.as_deref().unwrap_or("—"),
                        v.wall_seconds
                            .map_or_else(|| "—".to_string(), |w| format!("{w:.4}")),
                        v.prompt_tokens
                            .map_or_else(|| "—".to_string(), |t| t.to_string()),
                        v.generated_tokens
                            .map_or_else(|| "—".to_string(), |t| t.to_string()),
                        md_escape(&v.message),
                    );
                }
                let _ = writeln!(s);
            }
        }
    }
}

/// Loss/win tallies across a run.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Tallies {
    /// Measured speed wins.
    pub wins: u32,
    /// Near-parity (neutral band).
    pub neutral: u32,
    /// Measured speed losses.
    pub speed_losses: u32,
    /// Differential divergences (correctness losses).
    pub correctness_losses: u32,
    /// Unlowerable-node capability losses.
    pub capability_losses: u32,
    /// Runtime errors.
    pub errors: u32,
    /// Environmental skips (toolchain absent / feature off).
    pub skips: u32,
    /// Baseline (interpreter) failures — should be zero; loud if not.
    pub baseline_failures: u32,
}

/// Extract a `(ratio_str, reason)` for a verdict's table row.
fn verdict_ratio_reason(v: &Verdict) -> (String, String) {
    match v {
        Verdict::SpeedWin { ratio_x1000 } | Verdict::SpeedNeutral { ratio_x1000 } => {
            (fmt_ratio(*ratio_x1000), String::new())
        }
        Verdict::SpeedLoss {
            ratio_x1000,
            reason,
        } => (fmt_ratio(*ratio_x1000), reason.clone()),
        Verdict::CorrectnessLoss { detail } => ("—".to_string(), detail.clone()),
        Verdict::CapabilityLoss { reason } => ("—".to_string(), reason.clone()),
        Verdict::RuntimeError { message } => ("—".to_string(), message.clone()),
        Verdict::Skipped { reason } => ("—".to_string(), reason.clone()),
        Verdict::BaselineFailed { message } => ("—".to_string(), message.clone()),
    }
}

/// Format a ns figure compactly.
fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2}M", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.1}k", ns / 1_000.0)
    } else {
        format!("{ns:.1}")
    }
}

/// Format a parts-per-thousand ratio as `N.NNx`.
fn fmt_ratio(x1000: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let r = x1000 as f64 / 1000.0;
    format!("{r:.2}x")
}

/// Escape `|` and newlines so a reason cannot break a markdown table row. `pub(crate)` (not
/// private) so the in-crate `src/tests/report.rs` white-box test module can exercise it directly
/// (the CLAUDE.md test-layout convention: tests live in a sibling module, not `mod tests` nested
/// inside this file, so plain private visibility would not reach them).
pub(crate) fn md_escape(s: &str) -> String {
    s.replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// The neutral-band constant, re-exported for the binary to stamp into the report metadata.
#[must_use]
pub fn neutral_band() -> f64 {
    NEUTRAL_BAND
}
