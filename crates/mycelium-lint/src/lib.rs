//! `mycelium-lint` — **`myc-lint`**, lint + auto-fix (M-366).
//!
//! Surfaces the M-141 invariant lints (`mycelium_lsp::lint`) and the M-358/M-359 header lints as
//! **actionable** findings under one rule — **no silent rewrite; every fix is reified + opt-in** (G2) —
//! with a bright [`FixTier`] boundary:
//!
//! - **suggest** — a printed, reviewable edit; never auto-applied.
//! - **apply** — a mechanical, behaviour-preserving edit; applied only on `--fix`.
//! - **scaffold** — an incomplete skeleton the author must finish (e.g. an explicit `swap`, or an
//!   RFC-0014 recovery handler); **never** auto-applied (A2/I1/I5 — a control-flow change is the author's
//!   declared, bounded, opt-in choice, never the tool's).
//!
//! **First-implementation finding (M-366 §8.1 confirmation).** No M-141 lint has a *behaviour-preserving*
//! auto-fix that is not already `mycfmt`'s header canonicalization — so v0 maps every lint fix to
//! **suggest** or **scaffold**, and `--fix` applies **nothing** (header canonicalization is delegated to
//! `mycfmt`, M-364). This keeps the never-silent guarantee absolute: `myc-lint` v0 cannot rewrite your code.
//!
//! The §4.1 documentation quality-bar lint (M-363 §6, 8 checks) is now **ACTIVE** (Phase 9 Wave B): the
//! check-name set ([`DOC_QUALITY_CHECKS`]) is the single source of truth in `mycelium-doc`, where the
//! checks run over the M-363 doc-IR (see `mycelium_doc::doc_lint`). It has its own gate (`myc-doc`) and
//! still does **not** block the M-366 lint gate. KC-3: above the kernel.

use mycelium_l1::{check_and_resolve, elaborate, parse};
use mycelium_lsp::baseline::RecoveryProfile;
use mycelium_lsp::{lint, lint_structured_header, Severity};

/// How a fix may be applied — the opt-in boundary (the crux).
///
/// `#[non_exhaustive]`: a future tier may be added without a breaking change — an external exhaustive
/// `match` must carry a `_` arm (M-644; additive — no variant removed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FixTier {
    /// A reviewable edit; never auto-applied.
    Suggest,
    /// A behaviour-preserving edit; applied only on `--fix`.
    Apply,
    /// An incomplete skeleton the author completes; never auto-applied (control-flow stays the author's).
    Scaffold,
}

impl FixTier {
    /// The canonical label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            FixTier::Suggest => "suggest",
            FixTier::Apply => "apply",
            FixTier::Scaffold => "scaffold",
        }
    }
}

/// A reified fix offer for a finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fix {
    /// The tier (the opt-in boundary).
    pub tier: FixTier,
    /// What the fix does, in author-facing terms.
    pub description: String,
    /// For a scaffold: the skeleton the author completes (never applied automatically).
    pub scaffold: Option<String>,
}

impl Default for Fix {
    /// The neutral fix offer (M-644 ergonomics): tier [`FixTier::Suggest`] — the **never-auto-applied**
    /// tier, so a defaulted `Fix` can never silently rewrite code (G2). Empty description, no scaffold.
    fn default() -> Self {
        Self {
            tier: FixTier::Suggest,
            description: String::new(),
            scaffold: None,
        }
    }
}

/// One lint finding with its (optional) reified fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    /// The file.
    pub file: String,
    /// The lint code (`implicit-swap`, `unverified-bound`, …).
    pub code: String,
    /// Severity (from the M-141 lint).
    pub severity: Severity,
    /// The breadcrumb / location within the file.
    pub at: String,
    /// The message.
    pub message: String,
    /// The fix offer, if any.
    pub fix: Option<Fix>,
}

impl LintFinding {
    /// Attach a reified fix offer, fluently (M-644 ergonomics). Additive builder; sets `fix`.
    #[must_use]
    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.fix = Some(fix);
        self
    }
}

/// The aggregated lint result.
#[derive(Debug, Clone, Default)]
pub struct LintReport {
    /// Every finding, file-ordered.
    pub findings: Vec<LintFinding>,
    /// Files linted.
    pub files: usize,
}

impl LintReport {
    /// Push a finding, fluently (M-644 ergonomics). Additive builder; appends to `findings` (does
    /// **not** touch `files` — set that explicitly with [`LintReport::with_files`]).
    #[must_use]
    pub fn with_finding(mut self, finding: LintFinding) -> Self {
        self.findings.push(finding);
        self
    }

    /// Set the linted-file count, fluently (M-644 ergonomics). Additive builder.
    #[must_use]
    pub fn with_files(mut self, files: usize) -> Self {
        self.files = files;
        self
    }

    /// Whether any finding is an error-severity house-rule violation.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    /// Counts by tier: (apply, suggest, scaffold).
    #[must_use]
    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let mut c = (0, 0, 0);
        for f in &self.findings {
            match f.fix.as_ref().map(|x| x.tier) {
                Some(FixTier::Apply) => c.0 += 1,
                Some(FixTier::Suggest) => c.1 += 1,
                Some(FixTier::Scaffold) => c.2 += 1,
                None => {}
            }
        }
        c
    }
}

/// Map a lint code to its reified fix offer (the contract §3 table). Control-flow-changing fixes are
/// **scaffolds** (never auto-applied); the rest are **suggest** (a value is never fabricated — VR-5). No
/// lint maps to **apply** in v0 (the first-impl finding above).
fn fix_for(code: &str) -> Option<Fix> {
    match code {
        "implicit-swap" => Some(Fix {
            tier: FixTier::Scaffold,
            description: "wrap the mixed-paradigm op in an explicit `swap` — you choose the target \
                          representation and policy (a conversion is never inserted implicitly; A2/I5)"
                .to_owned(),
            scaffold: Some(
                "swap(<value>, to: <TargetRepr>, policy: <policy_ref>)  // fill the target + a real PolicyRef"
                    .to_owned(),
            ),
        }),
        "placeholder-policy" => Some(Fix {
            tier: FixTier::Suggest,
            description: "replace the stub policy digest with a real reified PolicyRef (G2 — a swap's \
                          selection must be reified, not faked)"
                .to_owned(),
            scaffold: None,
        }),
        "unverified-bound" => Some(Fix {
            tier: FixTier::Suggest,
            description: "annotate why this `Declared` bound is asserted, or verify it to upgrade the tag \
                          (never upgraded silently — VR-5)"
                .to_owned(),
            scaffold: None,
        }),
        "free-variable" => Some(Fix {
            tier: FixTier::Suggest,
            description: "bind the variable with an enclosing `let`, or fix the typo — this is a real \
                          open-term error, not a mechanical fix"
                .to_owned(),
            scaffold: None,
        }),
        "nodule-header" => Some(Fix {
            tier: FixTier::Suggest,
            description: "fix the header per the diagnostic; `mycfmt` canonicalizes a well-formed header, \
                          but a malformed value is never fabricated (VR-5)"
                .to_owned(),
            scaffold: None,
        }),
        _ => None,
    }
}

/// Generate an RFC-0014 **recovery scaffold** for an error `class` under a named, bounded [`RecoveryProfile`]
/// — an explicit handler skeleton the author completes. This is the actionable form of the RFC-0015 §9
/// "this class is only logged — add a handler?" advisory. It is **always** a scaffold (never applied): the
/// decision to recover is declared, bounded, and opt-in (A2/I4/I5). The L1 surface syntax for handlers is
/// RFC-0014's (not yet in the parser), so this is an illustrative, clearly-marked skeleton.
#[must_use]
pub fn recovery_scaffold(class: &str, profile: RecoveryProfile) -> String {
    let body = match profile {
        RecoveryProfile::Strict => {
            "    // strict: propagate (no recovery) — the honest default".to_owned()
        }
        RecoveryProfile::Resilient => format!(
            "    // resilient: bounded retry(<={}) on `{class}`; if every attempt fails, propagate (I1/I4)\n\
             \x20   retry(max = {}) {{ /* todo: the recoverable action */ }}",
            mycelium_lsp::baseline::RESILIENT_MAX_ATTEMPTS,
            mycelium_lsp::baseline::RESILIENT_MAX_ATTEMPTS
        ),
    };
    format!(
        "// recovery scaffold (opt-in, bounded — RFC-0014 I1/I5; complete or delete):\n\
         handle {class} with {} {{\n{body}\n}}",
        profile.as_str()
    )
}

/// The §4.1 documentation quality-bar lint check set (M-363 §6) — now **active** (Phase 9 Wave B). The
/// canonical names live in `mycelium-doc` (the crate that runs the checks over the doc-IR); re-exported
/// here under the historical name so existing callers keep working (DRY — one source of truth).
pub use mycelium_doc::CHECK_NAMES as DOC_QUALITY_CHECKS;

/// The status line for the §4.1 doc lint — now **active** (it runs over the M-363 doc-IR via `myc-doc`,
/// `mycelium_doc::doc_lint`). Has its own gate; does not block the M-366 lint gate.
#[must_use]
pub fn doc_lint_status() -> String {
    format!(
        "doc quality-bar lint (§4.1): active — {} checks run over the M-363 doc-IR (myc-doc gate); \
         does not block the M-366 gate",
        DOC_QUALITY_CHECKS.len()
    )
}

/// Lint one source, appending findings. Header lints always run (text); the M-141 Core-IR lints run over
/// each **elaborable** definition (parse → check → elaborate → lint); a definition that does not elaborate
/// is not lintable at the IR level yet and is skipped (honest — not a silent pass, just out of IR scope).
pub fn lint_source(file: &str, src: &str, out: &mut Vec<LintFinding>) {
    // Header lints (M-358/M-359) — raw text.
    for d in lint_structured_header(src) {
        out.push(to_finding(file, d.code, d.severity, &d.at, &d.message));
    }
    // Core-IR invariant lints (M-141) over the elaborable fragment.
    let Ok(nodule) = parse(src) else {
        return; // parse errors are myc-check's domain; the linter does not double-report them
    };
    let Ok((env, _twin)) = check_and_resolve(&nodule) else {
        return; // unchecked code is not IR-lintable; myc-check reports the refusal
    };
    let mut names: Vec<&String> = env.fns.keys().collect();
    names.sort();
    for name in names {
        if let Ok(node) = elaborate(&env, name) {
            for d in lint(&node) {
                out.push(to_finding(file, d.code, d.severity, &d.at, &d.message));
            }
        }
    }
}

fn to_finding(file: &str, code: &str, severity: Severity, at: &str, message: &str) -> LintFinding {
    LintFinding {
        file: file.to_owned(),
        code: code.to_owned(),
        severity,
        at: at.to_owned(),
        message: message.to_owned(),
        fix: fix_for(code),
    }
}

/// Lint an explicit set of `(file, contents)` sources, deterministically.
#[must_use]
pub fn lint_sources(sources: &[(String, String)]) -> LintReport {
    let mut findings = Vec::new();
    for (file, src) in sources {
        lint_source(file, src, &mut findings);
    }
    findings
        .sort_by(|a, b| (a.file.as_str(), a.at.as_str()).cmp(&(b.file.as_str(), b.at.as_str())));
    LintReport {
        findings,
        files: sources.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fix_table_maps_each_lint_to_the_right_tier() {
        // The fix model (contract §3): a stub policy / unverified bound / free var / header is `suggest`
        // (a value is never fabricated); a mixed-paradigm op is `scaffold` (control-flow change).
        for code in [
            "placeholder-policy",
            "unverified-bound",
            "free-variable",
            "nodule-header",
        ] {
            assert_eq!(fix_for(code).unwrap().tier, FixTier::Suggest, "{code}");
        }
        assert_eq!(fix_for("implicit-swap").unwrap().tier, FixTier::Scaffold);
        assert!(fix_for("not-a-lint").is_none());
    }

    #[test]
    fn an_implicit_swap_offers_a_scaffold_never_an_apply() {
        // The fix model: a control-flow change (inserting a swap) is a scaffold, never auto-applied.
        let fix = fix_for("implicit-swap").unwrap();
        assert_eq!(fix.tier, FixTier::Scaffold);
        assert!(fix.scaffold.is_some());
    }

    #[test]
    fn no_lint_maps_to_an_auto_apply_fix_in_v0() {
        // First-impl finding: v0 never auto-applies a lint fix (suggest/scaffold only).
        for code in [
            "implicit-swap",
            "placeholder-policy",
            "unverified-bound",
            "free-variable",
            "nodule-header",
        ] {
            if let Some(fix) = fix_for(code) {
                assert_ne!(
                    fix.tier,
                    FixTier::Apply,
                    "{code} must not be auto-apply in v0"
                );
            }
        }
    }

    #[test]
    fn a_malformed_header_is_surfaced_with_a_suggest_fix() {
        let r = lint_sources(&[("h.myc".to_owned(), "// nodule: 9bad\nnodule d\n".to_owned())]);
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "nodule-header")
            .expect("header lint fires");
        assert_eq!(f.fix.as_ref().unwrap().tier, FixTier::Suggest);
    }

    #[test]
    fn the_recovery_scaffold_is_bounded_and_marked() {
        let s = recovery_scaffold("SwapOutOfRange", RecoveryProfile::Resilient);
        assert!(s.contains("retry(max = 3)"), "{s}");
        assert!(s.contains("opt-in, bounded"), "{s}");
        // strict propagates (no recovery body).
        assert!(recovery_scaffold("SwapOutOfRange", RecoveryProfile::Strict).contains("propagate"));
    }

    #[test]
    fn the_doc_lint_is_now_active() {
        // Wave B: the §4.1 lint is no longer dormant — it runs over the M-363 doc-IR (myc-doc).
        assert_eq!(DOC_QUALITY_CHECKS.len(), 8);
        assert!(doc_lint_status().contains("active"));
        // The check-name set is the single source of truth re-exported from mycelium-doc.
        assert_eq!(DOC_QUALITY_CHECKS, mycelium_doc::CHECK_NAMES);
    }

    #[test]
    fn fix_default_is_the_never_auto_applied_tier_and_builders_compose() {
        // M-644: a defaulted Fix is Suggest — never auto-applied (G2); builders compose fluently.
        assert_eq!(Fix::default().tier, FixTier::Suggest);
        let finding = LintFinding {
            file: "a.myc".into(),
            code: "implicit-swap".into(),
            severity: Severity::Warning,
            at: "f".into(),
            message: "m".into(),
            fix: None,
        }
        .with_fix(Fix::default());
        assert_eq!(finding.fix.as_ref().map(|x| x.tier), Some(FixTier::Suggest));
        let report = LintReport::default().with_finding(finding).with_files(1);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.files, 1);
    }
}
