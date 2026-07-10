//! Ingestion of the **LLM-harness** report (`tools/llm-harness/`) — read, never run. The language-
//! leverage data (KC-2 / SC-5b: per-validation quality + latency + token cost) sits in the unified
//! report alongside the execution-backend data, so both halves of "what are we getting out of this
//! lang" are in one place.
//!
//! **Honesty:** the harness marks a fixture run `mode: "mock"` (and each result `status: "mock-PASS"`);
//! a real model run is `mode: "real"`/`"server"`. This module *preserves* that label — a synthetic
//! sample is surfaced as SYNTHETIC and never presented as evidence of real model quality (VR-5, and
//! the harness's own V-03 rule). We bind to the harness's documented schema
//! (`tools/llm-harness/harness.py::_write_json_report`); unknown `detail` fields are kept opaque.
//!
//! ## Grok/xAI harness bridge
//!
//! The Grok co-author harness (`tools/llm-harness/`) emits a different schema than the bench
//! harness above. Both are ingested via the schema-dispatching [`parse_any_llm_json`] function:
//!
//! - **Bench schema** (`mycelium-llm-validation`): `{harness, version, run_id, mode, honesty_posture,
//!   summary, results}` — ingested via [`LlmReport::from_json`] (original path; unchanged).
//! - **Grok schema** (`mycelium-grok-coauthor`): `{metadata, honesty_posture, quality, performance,
//!   outcomes, ablation}` — ingested via [`GrokLlmReport::from_json`].
//!
//! Tagged dispatch in [`parse_any_llm_json`]: a top-level `"metadata"` key (an object) unambiguously
//! marks the Grok schema; its absence selects the bench schema. The two root shapes are structurally
//! distinct — no magic-string check on `harness` is needed.
//!
//! **Honesty (VR-5):** numeric metrics in [`GrokOutcome`] carry their own `guarantee_tag`
//! (`"Declared"` in the synthetic sample). [`parse_any_llm_json`] preserves that tag — never
//! upgrades. The `synthetic` flag from `honesty_posture` propagates so `is_synthetic` / the
//! "SYNTHETIC" label remain correct in the unified report.

use std::path::Path;

use serde::Deserialize;

/// A parsed LLM-harness report (the subset of the schema the unified report needs; extra fields are
/// ignored, and `detail` is kept as opaque JSON so the binding does not over-fit the harness).
#[derive(Debug, Clone, Deserialize)]
pub struct LlmReport {
    /// Harness identifier (expected `"mycelium-llm-validation"`).
    pub harness: String,
    /// Harness schema version (e.g. `"0.1.0"`).
    pub version: String,
    /// The run id (an ISO-8601-ish `YYYYMMDDTHHMMSSZ` stamp).
    pub run_id: String,
    /// The run mode: `"mock"` (synthetic fixtures), `"real"` (llama-cli), `"server"` (llama.cpp
    /// HTTP), or `"skip"` (model unavailable). This is the primary synthetic/real discriminator.
    pub mode: String,
    /// The honesty posture block (lattice, allowed model tags, VR-5 rule text).
    pub honesty_posture: HonestyPosture,
    /// The roll-up summary.
    pub summary: Summary,
    /// Per-validation results (V-01..V-04 and any further checks).
    pub results: Vec<ValidationResult>,
}

/// The honesty posture the harness stamps into every report.
#[derive(Debug, Clone, Deserialize)]
pub struct HonestyPosture {
    /// Always true for the harness (it is never-silent).
    pub never_silent: bool,
    /// The guarantee lattice, strongest-first (`["Exact","Proven","Empirical","Declared"]`).
    pub guarantee_lattice: Vec<String>,
    /// The tags a model-derived claim is allowed to carry (`["Declared","Empirical"]`).
    pub model_allowed_tags: Vec<String>,
    /// The VR-5 rule text.
    pub vr5_rule: String,
}

/// The report roll-up.
#[derive(Debug, Clone, Deserialize)]
pub struct Summary {
    /// `"PASS"` | `"FAIL"` | `"MOCK"` | `"INCONCLUSIVE"`.
    pub overall: String,
    /// Total validations.
    pub total: u32,
    /// Count of real PASS.
    pub pass: u32,
    /// Count of fixture (mock) PASS — never counted as real-quality evidence.
    pub mock_pass: u32,
    /// Count of SKIP.
    pub skip: u32,
    /// Count of FAIL.
    pub fail: u32,
    /// Process exit code (0 if no FAILs).
    pub exit_code: i32,
    /// The run mode (duplicated here by the harness).
    pub mode: String,
    /// The model path / server URL, or null.
    pub model: Option<String>,
}

/// One validation result. `detail` is kept opaque (its shape varies per validation); the latency /
/// token fields the unified report shows are pulled from it defensively in [`ValidationResult::wall_seconds`]
/// and [`ValidationResult::token_counts`].
#[derive(Debug, Clone, Deserialize)]
pub struct ValidationResult {
    /// The validation id (e.g. `"V-04-latency-tokens"`).
    pub id: String,
    /// `"PASS"` | `"FAIL"` | `"SKIP"` | `"mock-PASS"`.
    pub status: String,
    /// The honest guarantee tag (`"Empirical"` | `"Declared"` | null).
    pub guarantee_tag: Option<String>,
    /// A human-readable one-line summary.
    pub message: String,
    /// The validation-specific detail object (opaque).
    #[serde(default)]
    pub detail: serde_json::Value,
}

impl ValidationResult {
    /// The wall-clock latency this validation recorded, in seconds, if present in `detail`
    /// (`detail.wall_seconds`, or the V-01 `run_a_wall_seconds`). `None` when the validation carries
    /// no latency (or it is the mock sentinel `0.0`, which we surface as `Some(0.0)` so the report can
    /// label it synthetic rather than silently dropping it).
    #[must_use]
    pub fn wall_seconds(&self) -> Option<f64> {
        self.detail
            .get("wall_seconds")
            .and_then(serde_json::Value::as_f64)
            .or_else(|| {
                self.detail
                    .get("run_a_wall_seconds")
                    .and_then(serde_json::Value::as_f64)
            })
    }

    /// The (prompt, generated) token counts this validation recorded, if present
    /// (`detail.token_counts.{prompt,generated}`). Either side may be `None`.
    #[must_use]
    pub fn token_counts(&self) -> Option<(Option<u64>, Option<u64>)> {
        let tc = self.detail.get("token_counts")?;
        let prompt = tc.get("prompt").and_then(serde_json::Value::as_u64);
        let generated = tc.get("generated").and_then(serde_json::Value::as_u64);
        Some((prompt, generated))
    }

    /// Whether this result is a fixture (mock) result, not real-model evidence.
    #[must_use]
    pub fn is_mock(&self) -> bool {
        self.status == "mock-PASS"
            || self.detail.get("mode").and_then(serde_json::Value::as_str) == Some("mock")
    }
}

impl LlmReport {
    /// `true` when this report is a SYNTHETIC fixture run (no real model) — the primary honesty gate.
    /// Driven by `mode == "mock"` (and corroborated by `summary.overall == "MOCK"`). The unified
    /// report MUST label such a report synthetic.
    #[must_use]
    pub fn is_synthetic(&self) -> bool {
        self.mode == "mock" || self.summary.overall == "MOCK"
    }

    /// A one-line provenance string for the unified report header.
    #[must_use]
    pub fn provenance(&self) -> String {
        let kind = if self.is_synthetic() {
            "SYNTHETIC (fixture; not real model quality)"
        } else {
            "real model run"
        };
        format!(
            "{} v{} — run {} — mode={} — {} ({} validations: {} pass / {} mock-pass / {} skip / {} fail)",
            self.harness,
            self.version,
            self.run_id,
            self.mode,
            kind,
            self.summary.total,
            self.summary.pass,
            self.summary.mock_pass,
            self.summary.skip,
            self.summary.fail,
        )
    }

    /// Parse a report from JSON text. Errors are explicit (a malformed report is loud, not skipped).
    pub fn from_json(text: &str) -> Result<Self, LlmIngestError> {
        serde_json::from_str(text).map_err(|e| LlmIngestError::Parse(e.to_string()))
    }

    /// Read + parse a report from a file path.
    pub fn from_path(path: &Path) -> Result<Self, LlmIngestError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| LlmIngestError::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Find the **newest** report under a harness reports directory (`*-report.json`, lexicographic
    /// max — the timestamped names sort chronologically), if any. Returns `Ok(None)` when the
    /// directory has no report (the caller then falls back to the committed synthetic sample).
    pub fn newest_in_dir(dir: &Path) -> Result<Option<std::path::PathBuf>, LlmIngestError> {
        if !dir.is_dir() {
            return Ok(None);
        }
        let mut newest: Option<std::path::PathBuf> = None;
        for entry in std::fs::read_dir(dir).map_err(|e| LlmIngestError::Io(e.to_string()))? {
            let entry = entry.map_err(|e| LlmIngestError::Io(e.to_string()))?;
            let path = entry.path();
            let is_report = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("-report.json"));
            if is_report {
                match &newest {
                    Some(cur) if path <= *cur => {}
                    _ => newest = Some(path),
                }
            }
        }
        Ok(newest)
    }
}

/// A never-silent ingestion error.
#[derive(Debug)]
pub enum LlmIngestError {
    /// The report file could not be read.
    Io(String),
    /// The report JSON could not be parsed against the schema.
    Parse(String),
}

impl std::fmt::Display for LlmIngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmIngestError::Io(m) => write!(f, "llm-report I/O error: {m}"),
            LlmIngestError::Parse(m) => write!(f, "llm-report parse error: {m}"),
        }
    }
}

impl std::error::Error for LlmIngestError {}

// ─── Grok/xAI harness schema ──────────────────────────────────────────────────────────────────

/// The top-level Grok/xAI harness report (`mycelium-grok-coauthor`).
///
/// `#[serde(deny_unknown_fields)]` pins the contract: any unknown key in a Grok fixture is a loud
/// parse error, not a silent drop (G2 — never silent). This catches schema drift at the boundary.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokLlmReport {
    /// Harness metadata (name, version, model, mode, …).
    pub metadata: GrokMetadata,
    /// Honesty posture block (shared lattice; Grok adds `synthetic` + `synthetic_note`).
    pub honesty_posture: GrokHonestyPosture,
    /// Aggregate quality metrics (syntactic validity rate, type-check pass rate, …).
    pub quality: GrokQuality,
    /// Aggregate token / latency / cost metrics.
    pub performance: GrokPerformance,
    /// Per-task outcomes — the primary per-validation rows.
    pub outcomes: Vec<GrokOutcome>,
    /// Ablation experiment block (arms + retention result). Optional: the harness emits `null`
    /// when `--ablation` was not run (and may omit it on some versions) — Copilot #308.
    #[serde(default)]
    pub ablation: Option<GrokAblation>,
}

/// Harness metadata block in the Grok report.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokMetadata {
    /// Harness identifier (e.g. `"mycelium-grok-coauthor"`).
    pub harness: String,
    /// Schema version (e.g. `"0.1.0"`).
    pub version: String,
    /// Model identifier (e.g. `"grok-4.3"`).
    pub model: String,
    /// Run mode — the Grok harness emits `"live"`, `"batch"`, or `"self-test"`
    /// (`tools/llm-harness/grok/report.py`). Corroborates `synthetic` in `is_synthetic`.
    pub mode: String,
    /// Endpoint description (e.g. `"mock (offline)"`, or a real URL).
    pub endpoint: String,
    /// Task-set identifier.
    pub task_set_id: String,
    /// Random seed used.
    pub seed: u64,
    /// Maximum rounds per task.
    pub max_rounds: u32,
    /// UTC timestamp string (or `"SAMPLE-DETERMINISTIC"` for fixtures).
    pub timestamp_utc: String,
}

/// Honesty posture block in the Grok report. Extends the bench posture with `synthetic` flag.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokHonestyPosture {
    /// Always true in the Grok harness.
    pub never_silent: bool,
    /// Guarantee lattice, strongest-first.
    pub guarantee_lattice: Vec<String>,
    /// Tags a model-derived claim is allowed to carry.
    pub model_allowed_tags: Vec<String>,
    /// VR-5 rule text (Grok variant is longer than the bench variant).
    pub vr5_rule: String,
    /// `true` when this is a synthetic self-test (no real model call). The authoritative
    /// synthetic discriminant for the Grok schema.
    pub synthetic: bool,
    /// Human-readable note explaining the synthetic status.
    pub synthetic_note: String,
}

/// Aggregate quality metrics from the Grok report.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokQuality {
    /// Total tasks in the run.
    pub total: u32,
    /// Tasks actually scored (not skipped).
    pub scored: u32,
    /// Tasks skipped.
    pub skipped: u32,
    /// Count of syntactically valid outputs.
    pub syntactic_valid: u32,
    /// Count of type-check passes.
    pub typecheck_pass: u32,
    /// Syntactic validity rate (0.0–1.0).
    pub syntactic_validity_rate: f64,
    /// Type-check pass rate (0.0–1.0).
    pub typecheck_pass_rate: f64,
    /// Edit-to-fix iteration counts (one per task).
    pub edit_to_fix_iterations: Vec<u32>,
    /// Mean edit-to-fix iterations across tasks.
    pub mean_edit_to_fix: f64,
}

/// Aggregate performance metrics from the Grok report.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokPerformance {
    /// Total prompt tokens across all rounds.
    pub prompt_tokens: u64,
    /// Total completion tokens across all rounds.
    pub completion_tokens: u64,
    /// Sum of prompt + completion tokens.
    pub total_tokens: u64,
    /// Total cost in USD (may be 0.0 for mock runs).
    pub total_cost_usd: f64,
    /// Number of API requests made.
    pub request_count: u32,
    /// Number of batch requests (0 for non-batch runs).
    pub batch_count: u32,
    /// Mean per-request latency in seconds. Optional: `null` for batch runs with no per-request
    /// latencies (Copilot #308).
    #[serde(default)]
    pub mean_latency_s: Option<f64>,
    /// Total latency in seconds.
    pub total_latency_s: f64,
}

/// Per-task outcome in the Grok report. Maps to one `LlmValidationRow` in the unified report.
///
/// `guarantee_tag` is preserved verbatim — never upgraded (VR-5 / honesty rule). The Grok
/// synthetic sample carries `"Declared"` for all outcomes; a real run may carry `"Empirical"`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokOutcome {
    /// Task identifier (e.g. `"g01"`).
    pub task_id: String,
    /// Task specification name (e.g. `"identity"`).
    pub spec: String,
    /// Model that ran this task.
    pub model: String,
    /// Outcome status: `"PASS"` | `"PARTIAL_PASS"` | `"FAIL"` | `"SKIP"`.
    pub status: String,
    /// Honest guarantee tag — preserved verbatim, never upgraded (VR-5).
    pub guarantee_tag: String,
    /// Number of edit–fix iterations until clean (or until max_rounds exhausted).
    pub iterations_to_clean: u32,
    /// Per-round detail (kept as opaque JSON — round shape may evolve between harness versions).
    pub rounds: Vec<serde_json::Value>,
    /// Total prompt tokens for this task (sum across rounds).
    pub total_prompt_tokens: u64,
    /// Total completion tokens for this task (sum across rounds).
    pub total_completion_tokens: u64,
    /// Total latency in seconds for this task.
    pub total_latency_s: f64,
    /// Total cost in USD for this task.
    pub total_cost_usd: f64,
    /// Human-readable one-line message.
    pub message: String,
}

/// The ablation block in the Grok report. Scalars are typed; `arms` and `retention` are kept
/// opaque (`serde_json::Value`) because their inner schemas evolve independently.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrokAblation {
    /// Experiment description.
    pub experiment: String,
    /// Model that ran the ablation.
    pub model: String,
    /// Task-set identifier.
    pub task_set_id: String,
    /// Seeds used.
    pub seeds: Vec<u64>,
    /// Ablation arms (opaque — inner shape may evolve).
    pub arms: Vec<serde_json::Value>,
    /// Retention result block (opaque — inner shape may evolve).
    pub retention: serde_json::Value,
}

impl GrokLlmReport {
    /// Parse a Grok harness report from JSON text. Errors are explicit — never silent (G2).
    pub fn from_json(text: &str) -> Result<Self, LlmIngestError> {
        serde_json::from_str(text).map_err(|e| LlmIngestError::Parse(e.to_string()))
    }

    /// Whether this Grok report represents a synthetic (offline mock / self-test) run.
    ///
    /// The primary honesty gate: a synthetic report MUST be labeled synthetic in the unified
    /// report (VR-5 / the harness's V-03 rule). Driven by `honesty_posture.synthetic` (the
    /// authoritative flag); corroborated by `metadata.mode == "self-test"` (the Grok harness
    /// never emits `"mock"` — its modes are `"live"`/`"batch"`/`"self-test"`).
    #[must_use]
    pub fn is_synthetic(&self) -> bool {
        self.honesty_posture.synthetic || self.metadata.mode == "self-test"
    }

    /// A one-line provenance string for the unified report, analogous to [`LlmReport::provenance`].
    #[must_use]
    pub fn provenance(&self) -> String {
        let kind = if self.is_synthetic() {
            "SYNTHETIC (self-test / offline mock; NOT real model quality)"
        } else {
            "real model run"
        };
        format!(
            "{} v{} — model={} — mode={} — {} ({} outcomes: {} scored / {} skipped)",
            self.metadata.harness,
            self.metadata.version,
            self.metadata.model,
            self.metadata.mode,
            kind,
            self.quality.total,
            self.quality.scored,
            self.quality.skipped,
        )
    }
}

// ─── Schema-dispatching ingestion ─────────────────────────────────────────────────────────────

/// Peek at a raw JSON value to decide which harness schema it belongs to.
///
/// - `"metadata"` key present at root → Grok schema (`mycelium-grok-coauthor`)
/// - absent → bench schema (`mycelium-llm-validation`)
///
/// Structural discriminant: the Grok schema has a nested `metadata` object at root; the bench
/// schema has flat `harness`/`version` strings at root. The two shapes are unambiguous.
#[must_use]
fn is_grok_schema(raw: &serde_json::Value) -> bool {
    raw.get("metadata")
        .is_some_and(serde_json::Value::is_object)
}

/// Intermediate form produced by [`parse_any_llm_json`]; consumed by
/// [`crate::report::LlmSection::from_parsed`].
#[derive(Debug)]
pub struct ParsedLlmSection {
    /// Source file path (for provenance, not reachability).
    pub source_path: String,
    /// Whether the report is synthetic — the honesty gate (VR-5).
    pub is_synthetic: bool,
    /// One-line provenance string.
    pub provenance: String,
    /// Per-validation/per-outcome rows; guarantee tags preserved verbatim.
    pub validations: Vec<crate::report::LlmValidationRow>,
}

/// Parse raw JSON text as *either* the Grok or the bench harness schema, returning a
/// [`ParsedLlmSection`] ready for the unified report.
///
/// **Dispatch:** peeks at the root JSON value (one `serde_json::from_str` parse); dispatches on
/// the presence of a `"metadata"` object key. Both paths are explicit parse errors, never silent
/// (G2).
///
/// **Honesty:** per-outcome `guarantee_tag` values are preserved verbatim from their source
/// schema — never upgraded. `is_synthetic` propagates from the authoritative flag in each schema.
///
/// # Errors
/// Returns [`LlmIngestError::Parse`] if the text is not valid JSON or does not match the
/// detected schema (including `deny_unknown_fields` violations in the Grok path).
pub fn parse_any_llm_json(
    text: &str,
    source_path: String,
) -> Result<ParsedLlmSection, LlmIngestError> {
    // One parse to peek the discriminant; the schema-specific parse repeats the work but
    // avoids an unsafe transmute or complex deserialization dance. Both parses are fast — the
    // reports are small JSON blobs.
    let raw: serde_json::Value =
        serde_json::from_str(text).map_err(|e| LlmIngestError::Parse(e.to_string()))?;

    if is_grok_schema(&raw) {
        let grok = GrokLlmReport::from_json(text)?;
        let is_synthetic = grok.is_synthetic();
        let provenance = grok.provenance();
        let validations = grok
            .outcomes
            .iter()
            .map(|o| crate::report::LlmValidationRow {
                // Compose a stable id: "<task_id>/<spec>" mirrors the V-xx style used by
                // the bench harness while being self-documenting for the Grok outcomes.
                id: format!("{}/{}", o.task_id, o.spec),
                status: o.status.clone(),
                // Preserve the per-outcome guarantee tag verbatim — never upgrade (VR-5).
                guarantee_tag: Some(o.guarantee_tag.clone()),
                wall_seconds: Some(o.total_latency_s),
                prompt_tokens: Some(o.total_prompt_tokens),
                generated_tokens: Some(o.total_completion_tokens),
                message: o.message.clone(),
            })
            .collect();
        Ok(ParsedLlmSection {
            source_path,
            is_synthetic,
            provenance,
            validations,
        })
    } else {
        let bench = LlmReport::from_json(text)?;
        let is_synthetic = bench.is_synthetic();
        let provenance = bench.provenance();
        let validations = bench
            .results
            .iter()
            .map(|v| {
                let (p, g) = match v.token_counts() {
                    Some((p, g)) => (p, g),
                    None => (None, None),
                };
                crate::report::LlmValidationRow {
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
        Ok(ParsedLlmSection {
            source_path,
            is_synthetic,
            provenance,
            validations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal but faithful synthetic sample matching the harness schema (the committed
    /// `tools/llm-harness/reports/*-report.json` shape). Used so the unit test is fully offline +
    /// deterministic and does not depend on a path outside the crate.
    const SAMPLE: &str = r#"{
      "harness": "mycelium-llm-validation",
      "version": "0.1.0",
      "run_id": "20260617T182214Z",
      "mode": "mock",
      "timestamp_utc": "20260617T182214Z",
      "honesty_posture": {
        "never_silent": true,
        "guarantee_lattice": ["Exact", "Proven", "Empirical", "Declared"],
        "model_allowed_tags": ["Declared", "Empirical"],
        "vr5_rule": "Model-derived claims are Empirical or Declared — NEVER Proven or Exact."
      },
      "summary": {
        "overall": "MOCK", "total": 4, "pass": 1, "mock_pass": 3,
        "skip": 0, "fail": 0, "exit_code": 0, "mode": "mock", "model": null
      },
      "results": [
        {
          "id": "V-01-determinism", "status": "mock-PASS", "guarantee_tag": "Declared",
          "message": "[MOCK] Determinism simulated with fixture.",
          "detail": {"mode": "mock", "matched": true}
        },
        {
          "id": "V-04-latency-tokens", "status": "mock-PASS", "guarantee_tag": "Declared",
          "message": "[MOCK] latency simulated.",
          "detail": {"mode": "mock", "wall_seconds": 0.0,
                     "token_counts": {"prompt": 12, "generated": 7, "note": "Declared"}}
        }
      ]
    }"#;

    #[test]
    fn parses_the_synthetic_sample_and_marks_it_synthetic() {
        let r = LlmReport::from_json(SAMPLE).expect("parses");
        assert_eq!(r.harness, "mycelium-llm-validation");
        assert_eq!(r.version, "0.1.0");
        assert_eq!(r.mode, "mock");
        assert!(
            r.is_synthetic(),
            "a mock-mode report must be flagged synthetic"
        );
        assert!(
            r.provenance().contains("SYNTHETIC"),
            "provenance must surface the synthetic label: {}",
            r.provenance()
        );
        assert_eq!(r.results.len(), 2);
        assert!(r.honesty_posture.never_silent);
    }

    #[test]
    fn pulls_latency_and_token_counts_from_detail() {
        let r = LlmReport::from_json(SAMPLE).unwrap();
        let v4 = r
            .results
            .iter()
            .find(|v| v.id == "V-04-latency-tokens")
            .unwrap();
        assert_eq!(
            v4.wall_seconds(),
            Some(0.0),
            "the mock sentinel 0.0 is surfaced, not dropped"
        );
        assert_eq!(v4.token_counts(), Some((Some(12), Some(7))));
        assert!(v4.is_mock(), "a mock-PASS result must be flagged mock");
    }

    #[test]
    fn a_real_report_is_not_flagged_synthetic() {
        let real = SAMPLE
            .replace("\"mode\": \"mock\"", "\"mode\": \"real\"")
            .replace("\"overall\": \"MOCK\"", "\"overall\": \"PASS\"");
        let r = LlmReport::from_json(&real).unwrap();
        assert!(
            !r.is_synthetic(),
            "a real-mode report must NOT be flagged synthetic"
        );
        assert!(r.provenance().contains("real model run"));
    }

    #[test]
    fn malformed_json_is_an_explicit_error_not_a_silent_skip() {
        let err = LlmReport::from_json("{ not json").unwrap_err();
        assert!(matches!(err, LlmIngestError::Parse(_)));
    }

    #[test]
    fn the_committed_harness_sample_if_present_parses() {
        // If the real committed sample is reachable from the workspace, it must parse against the
        // schema we bind to (a guard that our binding stays faithful to the harness). Skips silently
        // (Ok) when the path is not reachable from the test's CWD — it is not crate-local.
        let candidates = [
            "../../tools/llm-harness/reports/20260617T182214Z-report.json",
            "tools/llm-harness/reports/20260617T182214Z-report.json",
        ];
        for c in candidates {
            let p = Path::new(c);
            if p.is_file() {
                let r = LlmReport::from_path(p)
                    .unwrap_or_else(|e| panic!("the committed harness sample must parse: {e}"));
                assert_eq!(r.harness, "mycelium-llm-validation");
                assert!(r.is_synthetic(), "the committed sample is the mock fixture");
                return;
            }
        }
        // Not reachable from CWD — fine; the SAMPLE-based tests above cover the binding.
    }

    // ─── Grok schema tests ────────────────────────────────────────────────────────────────────

    /// A minimal but complete Grok synthetic sample matching the committed fixture
    /// (`crates/mycelium-bench/tests/SYNTHETIC-SAMPLE-grok-4.3-self-test.json`).
    /// Kept inline so the unit tests are fully offline and deterministic.
    const GROK_SAMPLE: &str = r#"{
      "metadata": {
        "harness": "mycelium-grok-coauthor",
        "version": "0.1.0",
        "model": "grok-4.3",
        "mode": "self-test",
        "endpoint": "mock (offline)",
        "task_set_id": "gold-compose-v1",
        "seed": 42,
        "max_rounds": 3,
        "timestamp_utc": "SAMPLE-DETERMINISTIC"
      },
      "honesty_posture": {
        "never_silent": true,
        "guarantee_lattice": ["Exact", "Proven", "Empirical", "Declared"],
        "model_allowed_tags": ["Empirical", "Declared"],
        "vr5_rule": "Model-derived claims are Empirical or Declared — NEVER Proven or Exact.",
        "synthetic": true,
        "synthetic_note": "SYNTHETIC (self-test): offline mock. NOT real model quality."
      },
      "quality": {
        "total": 2,
        "scored": 2,
        "skipped": 0,
        "syntactic_valid": 2,
        "typecheck_pass": 2,
        "syntactic_validity_rate": 1.0,
        "typecheck_pass_rate": 1.0,
        "edit_to_fix_iterations": [1, 1],
        "mean_edit_to_fix": 1.0
      },
      "performance": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "total_cost_usd": 0.0002,
        "request_count": 2,
        "batch_count": 0,
        "mean_latency_s": 0.0,
        "total_latency_s": 0.0
      },
      "outcomes": [
        {
          "task_id": "g01",
          "spec": "identity",
          "model": "grok-4.3",
          "status": "PASS",
          "guarantee_tag": "Declared",
          "iterations_to_clean": 1,
          "rounds": [
            {"attempt": 1, "is_correction": false, "verdict": "clean",
             "syntactic_valid": true, "typecheck_pass": true,
             "prompt_tokens": 60, "completion_tokens": 25, "latency_s": 0.0,
             "cost_usd": 0.00014, "chat_ok": true, "chat_error": "", "diagnostics": []}
          ],
          "total_prompt_tokens": 60,
          "total_completion_tokens": 25,
          "total_latency_s": 0.0,
          "total_cost_usd": 0.00014,
          "message": "clean on first attempt"
        },
        {
          "task_id": "g04",
          "spec": "widen",
          "model": "grok-4.3",
          "status": "PASS",
          "guarantee_tag": "Declared",
          "iterations_to_clean": 1,
          "rounds": [
            {"attempt": 1, "is_correction": false, "verdict": "clean",
             "syntactic_valid": true, "typecheck_pass": true,
             "prompt_tokens": 40, "completion_tokens": 25, "latency_s": 0.0,
             "cost_usd": 0.00006, "chat_ok": true, "chat_error": "", "diagnostics": []}
          ],
          "total_prompt_tokens": 40,
          "total_completion_tokens": 25,
          "total_latency_s": 0.0,
          "total_cost_usd": 0.00006,
          "message": "clean on first attempt"
        }
      ],
      "ablation": {
        "experiment": "test ablation",
        "model": "grok-4.3",
        "task_set_id": "gold-compose-v1",
        "seeds": [1, 2],
        "arms": [],
        "retention": {"determinate": false}
      }
    }"#;

    #[test]
    fn grok_schema_parses_and_marks_synthetic() {
        let g = GrokLlmReport::from_json(GROK_SAMPLE).expect("grok sample parses");
        assert_eq!(g.metadata.harness, "mycelium-grok-coauthor");
        assert_eq!(g.metadata.model, "grok-4.3");
        assert_eq!(g.metadata.mode, "self-test");
        assert!(g.honesty_posture.never_silent);
        assert!(g.honesty_posture.synthetic, "synthetic flag must be set");
        assert!(
            g.is_synthetic(),
            "a self-test Grok report must be flagged synthetic"
        );
        assert!(
            g.provenance().contains("SYNTHETIC"),
            "provenance must surface the synthetic label: {}",
            g.provenance()
        );
        assert_eq!(g.outcomes.len(), 2);
    }

    #[test]
    fn grok_deny_unknown_fields_catches_drift() {
        // Copilot #308: start from a VALID sample and inject ONE unknown root key, so the failure
        // is specifically the unknown field (deny_unknown_fields) — not a missing required field on
        // an empty object. An extra root key must be a loud error, never a silent drop (G2).
        let mut v: serde_json::Value =
            serde_json::from_str(GROK_SAMPLE).expect("base Grok sample is valid");
        v.as_object_mut()
            .expect("root is an object")
            .insert("extra_key".into(), serde_json::json!("drift"));
        let drifted = serde_json::to_string(&v).expect("re-serialize");
        let err = GrokLlmReport::from_json(&drifted).unwrap_err();
        assert!(
            matches!(err, LlmIngestError::Parse(_)),
            "unknown root field must be a parse error, not silent: {err}"
        );
        // Control: the un-drifted base parses — proves the error is the extra key, not the base.
        GrokLlmReport::from_json(GROK_SAMPLE).expect("the un-drifted base sample parses");
    }

    #[test]
    fn grok_accepts_null_ablation_and_null_mean_latency() {
        // Copilot #308: real harness shapes — `ablation: null` (when `--ablation` was not run) and
        // `performance.mean_latency_s: null` (batch runs with no per-request latencies). Both must
        // ingest cleanly into `Option` fields, not fail deserialization.
        let mut v: serde_json::Value =
            serde_json::from_str(GROK_SAMPLE).expect("base Grok sample is valid");
        let root = v.as_object_mut().expect("root is an object");
        root.insert("ablation".into(), serde_json::Value::Null);
        root.get_mut("performance")
            .and_then(serde_json::Value::as_object_mut)
            .expect("performance is an object")
            .insert("mean_latency_s".into(), serde_json::Value::Null);
        let text = serde_json::to_string(&v).expect("re-serialize");
        let g =
            GrokLlmReport::from_json(&text).expect("null ablation + null mean_latency must ingest");
        assert!(g.ablation.is_none(), "null ablation -> None");
        assert!(
            g.performance.mean_latency_s.is_none(),
            "null mean_latency_s -> None"
        );
        // And it still dispatches + marks synthetic (the sample is a self-test).
        let parsed =
            parse_any_llm_json(&text, "grok-null.json".into()).expect("null-field Grok dispatches");
        assert!(parsed.is_synthetic);
    }

    #[test]
    fn parse_any_dispatches_bench_schema_without_metadata_key() {
        let parsed = parse_any_llm_json(SAMPLE, "sample.json".into())
            .expect("bench sample dispatches correctly");
        assert!(!parsed.validations.is_empty());
        assert!(parsed.is_synthetic, "mock bench report is synthetic");
        assert!(parsed.provenance.contains("SYNTHETIC"));
    }

    #[test]
    fn parse_any_dispatches_grok_schema_with_metadata_key() {
        let parsed = parse_any_llm_json(GROK_SAMPLE, "grok-sample.json".into())
            .expect("grok sample dispatches correctly");
        assert_eq!(parsed.validations.len(), 2, "two outcomes → two rows");
        assert!(parsed.is_synthetic, "self-test Grok report is synthetic");
        assert!(parsed.provenance.contains("SYNTHETIC"));
        // Guarantee tags must be preserved verbatim — the Grok sample carries "Declared".
        for row in &parsed.validations {
            assert_eq!(
                row.guarantee_tag.as_deref(),
                Some("Declared"),
                "guarantee_tag must be preserved from the Grok outcome: {}",
                row.id
            );
        }
    }

    #[test]
    fn parse_any_malformed_json_is_loud() {
        let err = parse_any_llm_json("{ not json", "x.json".into()).unwrap_err();
        assert!(matches!(err, LlmIngestError::Parse(_)));
    }
}
