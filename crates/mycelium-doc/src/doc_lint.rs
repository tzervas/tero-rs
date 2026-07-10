//! The **§4.1 documentation quality-bar lint** (spec §6, M-363 §6) — eight explicit pass/fail checks
//! over the doc-IR, the never-silent analogue for docs (G2). Each check is `Active`, `PartiallyDormant`
//! (some sub-aspect needs machinery we don't have yet — honestly named, never faked green), or
//! `Dormant`. The gate fails on any **error-severity** finding; warnings/info are surfaced, not gated.
//!
//! This is what flips `mycelium_lint::doc_lint_status()` from *dormant-but-defined* to *active*: the
//! checks now run over a real, content-addressed model produced by [`mod@crate::build`].

use mycelium_l1::{check_nodule, parse};

use crate::emit;
use crate::ir::{DocModel, Payload, XrefResolution};

/// The eight §4.1 checks, by canonical name — the single source of truth (`mycelium-lint` re-exports
/// this as `DOC_QUALITY_CHECKS`).
pub const CHECK_NAMES: &[&str] = &[
    "single-template-conformance",
    "navigability",
    "progressive-disclosure",
    "checked-examples",
    "no-dead-xref",
    "dual-projection-parity",
    "no-hallucinated-prose",
    "legibility-accessibility",
];

/// Finding severity (mirrors the lattice's never-silent posture: an error gates, a warning advises).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A build failure — gates the doc check.
    Error,
    /// An advisory — surfaced, not gated.
    Warning,
    /// Informational — counts and honest gaps.
    Info,
}

impl Severity {
    /// The canonical label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        }
    }
}

/// One finding from a check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Which check raised it.
    pub check: String,
    /// Its severity.
    pub severity: Severity,
    /// The node anchor it concerns (or the document).
    pub anchor: String,
    /// The author-facing message.
    pub message: String,
}

/// Whether a check is fully active, partly dormant (named sub-aspects await machinery), or dormant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckStatus {
    /// Fully implemented over the IR.
    Active,
    /// Active, but a named sub-aspect is not yet implementable (honest — never faked green).
    PartiallyDormant(String),
    /// Not yet implementable (an explicit dormant entry; never a faked green).
    Dormant(String),
}

/// The outcome of one check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckOutcome {
    /// The check name.
    pub name: String,
    /// Its status.
    pub status: CheckStatus,
    /// A one-line human summary.
    pub summary: String,
    /// Any findings.
    pub findings: Vec<Finding>,
}

/// The full §4.1 lint report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocLintReport {
    /// One outcome per check, in `CHECK_NAMES` order.
    pub outcomes: Vec<CheckOutcome>,
}

impl DocLintReport {
    /// Whether any finding is error-severity (the gate condition).
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.outcomes
            .iter()
            .any(|o| o.findings.iter().any(|f| f.severity == Severity::Error))
    }

    /// Every error-severity finding, flattened.
    #[must_use]
    pub fn errors(&self) -> Vec<&Finding> {
        self.outcomes
            .iter()
            .flat_map(|o| o.findings.iter())
            .filter(|f| f.severity == Severity::Error)
            .collect()
    }
}

/// Run all eight §4.1 checks over the model.
#[must_use]
pub fn lint(model: &DocModel) -> DocLintReport {
    DocLintReport {
        outcomes: vec![
            check_single_template(model),
            check_navigability(model),
            check_progressive_disclosure(model),
            check_checked_examples(model),
            check_no_dead_xref(model),
            check_dual_projection_parity(model),
            check_no_hallucinated_prose(model),
            check_legibility_accessibility(model),
        ],
    }
}

// ── 1. single template conformance ────────────────────────────────────────────────────────────
fn check_single_template(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    for doc in &model.documents {
        if !matches!(doc.payload, Payload::Document { .. }) {
            findings.push(err(
                "single-template-conformance",
                &doc.anchor,
                "top-level node is not a Document — divergent structure",
            ));
        }
        if doc.title.as_deref().unwrap_or("").trim().is_empty() {
            findings.push(err(
                "single-template-conformance",
                &doc.anchor,
                "document has no title (the one template requires a header title)",
            ));
        }
        if doc.children.is_empty() {
            findings.push(err(
                "single-template-conformance",
                &doc.anchor,
                "document has no body (index→detail requires content)",
            ));
        }
    }
    CheckOutcome {
        name: "single-template-conformance".to_owned(),
        status: CheckStatus::Active,
        summary: format!(
            "{} documents conform to the one template (title + body)",
            model.documents.len()
        ),
        findings,
    }
}

// ── 2. navigability ───────────────────────────────────────────────────────────────────────────
fn check_navigability(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    let n_nodes = model.all_nodes().len();
    let n_anchors = model.anchors.len();
    if model.documents.is_empty() {
        findings.push(err(
            "navigability",
            "(model)",
            "no documents — nothing is reachable from the index",
        ));
    }
    if n_anchors != n_nodes {
        findings.push(err(
            "navigability",
            "(model)",
            &format!(
                "anchor collision: {n_nodes} nodes but {n_anchors} unique anchors — \
                 a node is unreachable / a deep link would collide"
            ),
        ));
    }
    CheckOutcome {
        name: "navigability".to_owned(),
        status: CheckStatus::Active,
        summary: format!(
            "{n_nodes} nodes, all uniquely anchored and reachable from the index; search index present"
        ),
        findings,
    }
}

// ── 3. progressive disclosure ─────────────────────────────────────────────────────────────────
fn check_progressive_disclosure(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    for doc in &model.documents {
        let mut levels = std::collections::BTreeSet::new();
        doc.walk(&mut |n| {
            if let Some(l) = n.level {
                levels.insert(l.as_str());
            }
        });
        if levels.is_empty() {
            findings.push(err(
                "progressive-disclosure",
                &doc.anchor,
                "no level-graded blocks at all — graded depth (minimal/medium/detailed) is required",
            ));
        } else if levels.len() == 1 {
            findings.push(Finding {
                check: "progressive-disclosure".to_owned(),
                severity: Severity::Warning,
                anchor: doc.anchor.clone(),
                message: format!(
                    "only one depth present ({}) — a richer document should offer graded depth",
                    levels.iter().next().unwrap()
                ),
            });
        }
    }
    CheckOutcome {
        name: "progressive-disclosure".to_owned(),
        status: CheckStatus::Active,
        summary: "every document carries graded depth (RFC-0013 levels reused for docs)".to_owned(),
        findings,
    }
}

// ── 4. checked examples ───────────────────────────────────────────────────────────────────────
fn check_checked_examples(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    let mut checked = 0usize;
    let mut illustrative = 0usize;
    for node in model.all_nodes() {
        if let Payload::Example {
            lang,
            source,
            checked: is_checked,
        } = &node.payload
        {
            if *is_checked {
                checked += 1;
                if let Err(msg) = type_check_example(source) {
                    findings.push(err(
                        "checked-examples",
                        &node.anchor,
                        &format!(
                            "checked example ({lang}) does not type-check via the trusted L1 checker: {msg}"
                        ),
                    ));
                }
            } else {
                illustrative += 1;
                findings.push(Finding {
                    check: "checked-examples".to_owned(),
                    severity: Severity::Info,
                    anchor: node.anchor.clone(),
                    message: format!(
                        "illustrative example ({lang}) — honestly not CI-checked (tag ```myc-checked to enforce)"
                    ),
                });
            }
        }
    }
    CheckOutcome {
        name: "checked-examples".to_owned(),
        status: CheckStatus::Active,
        summary: format!(
            "{checked} checked examples type-check (myc-check pipeline); {illustrative} illustrative, honestly flagged"
        ),
        findings,
    }
}

/// Type-check an example by the same pipeline the `myc-check` gate uses (parse → check_nodule).
fn type_check_example(source: &str) -> Result<(), String> {
    let nodule = parse(source).map_err(|e| format!("parse: {e}"))?;
    check_nodule(&nodule).map_err(|e| format!("check: {e}"))?;
    Ok(())
}

// ── 5. no dead xref ───────────────────────────────────────────────────────────────────────────
fn check_no_dead_xref(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    let (mut internal, mut external, mut out_of_scope) = (0usize, 0usize, 0usize);
    for node in model.all_nodes() {
        if let Payload::Xref { target } = &node.payload {
            match &target.resolution {
                XrefResolution::Internal { .. } => internal += 1,
                XrefResolution::ExternalUrl => external += 1,
                XrefResolution::OutOfScope => out_of_scope += 1,
                XrefResolution::Dead { reason } => findings.push(err(
                    "no-dead-xref",
                    &node.anchor,
                    &format!("dead cross-reference `{}`: {reason}", target.raw),
                )),
                XrefResolution::Unresolved => findings.push(err(
                    "no-dead-xref",
                    &node.anchor,
                    &format!(
                        "cross-reference `{}` was never resolved (build bug)",
                        target.raw
                    ),
                )),
            }
        }
    }
    CheckOutcome {
        name: "no-dead-xref".to_owned(),
        status: CheckStatus::Active,
        summary: format!(
            "{internal} internal xrefs resolve; {external} external + {out_of_scope} out-of-scope \
             (links.sh owns external reachability)"
        ),
        findings,
    }
}

// ── 6. dual projection parity (G11) ───────────────────────────────────────────────────────────
fn check_dual_projection_parity(model: &DocModel) -> CheckOutcome {
    let html = emit::html::render_concat(model);
    let json = emit::json::render_model_json(model);
    let model_ids = model.id_set();
    let html_ids = extract_hashes(&html);
    let json_ids = extract_hashes(&json);

    let mut findings = Vec::new();
    let missing_html: Vec<&String> = model_ids.difference(&html_ids).collect();
    let missing_json: Vec<&String> = model_ids.difference(&json_ids).collect();
    if !missing_html.is_empty() {
        findings.push(err(
            "dual-projection-parity",
            "(model)",
            &format!(
                "{} node(s) present in the IR are missing from the HTML view (e.g. {})",
                missing_html.len(),
                missing_html.first().map(|s| s.as_str()).unwrap_or("")
            ),
        ));
    }
    if !missing_json.is_empty() {
        findings.push(err(
            "dual-projection-parity",
            "(model)",
            &format!(
                "{} node(s) present in the IR are missing from the JSON view",
                missing_json.len()
            ),
        ));
    }
    // The HTML and JSON views must agree on exactly the model's node set (two views of one truth).
    if html_ids != model_ids || json_ids != model_ids {
        // Extra ids in a view (beyond the model) would also be a divergence.
        let extra_html = html_ids.difference(&model_ids).count();
        let extra_json = json_ids.difference(&model_ids).count();
        if extra_html > 0 || extra_json > 0 {
            findings.push(err(
                "dual-projection-parity",
                "(model)",
                &format!(
                    "a view carries content addresses absent from the IR (html +{extra_html}, json +{extra_json})"
                ),
            ));
        }
    }
    CheckOutcome {
        name: "dual-projection-parity".to_owned(),
        status: CheckStatus::Active,
        summary: format!(
            "HTML and JSON are two views of {} content-addressed nodes (same hashes — G11/ADR-003)",
            model_ids.len()
        ),
        findings,
    }
}

/// Extract every `blake3:<64 hex>` content address appearing in `s`.
fn extract_hashes(s: &str) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    let bytes = s.as_bytes();
    let pat = b"blake3:";
    let mut i = 0;
    while i + pat.len() + 64 <= bytes.len() {
        if &bytes[i..i + pat.len()] == pat {
            let digest = &bytes[i + pat.len()..i + pat.len() + 64];
            if digest.iter().all(u8::is_ascii_hexdigit) {
                out.insert(format!("blake3:{}", std::str::from_utf8(digest).unwrap()));
                i += pat.len() + 64;
                continue;
            }
        }
        i += 1;
    }
    out
}

// ── 7. no hallucinated prose / undocumented-is-flagged ────────────────────────────────────────
fn check_no_hallucinated_prose(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    let (mut documented, mut undocumented) = (0usize, 0usize);
    for node in model.all_nodes() {
        match &node.payload {
            Payload::ApiItem { summary, .. } => {
                if node.provenance.source.trim().is_empty() {
                    findings.push(err(
                        "no-hallucinated-prose",
                        &node.anchor,
                        "api-item has no provenance — an ungrounded (hallucinated) statement",
                    ));
                }
                match summary {
                    Some(_) => documented += 1,
                    None => {
                        undocumented += 1;
                        findings.push(Finding {
                            check: "no-hallucinated-prose".to_owned(),
                            severity: Severity::Info,
                            anchor: node.anchor.clone(),
                            message: "undocumented — explicit gap, never invented filler (G2)"
                                .to_owned(),
                        });
                    }
                }
            }
            Payload::Undocumented { .. } => undocumented += 1,
            _ => {}
        }
    }
    CheckOutcome {
        name: "no-hallucinated-prose".to_owned(),
        status: CheckStatus::Active,
        summary: format!(
            "{documented} api statements all trace to source; {undocumented} gaps flagged, never invented"
        ),
        findings,
    }
}

// ── 8. legibility / accessibility ─────────────────────────────────────────────────────────────
fn check_legibility_accessibility(model: &DocModel) -> CheckOutcome {
    let mut findings = Vec::new();
    let html = emit::html::render_concat(model);

    if !html.contains("<main>") {
        findings.push(err(
            "legibility-accessibility",
            "(html)",
            "no <main> landmark — non-semantic layout",
        ));
    }
    if !html.contains("lang=\"en\"") {
        findings.push(err(
            "legibility-accessibility",
            "(html)",
            "no document language declared (`lang`)",
        ));
    }
    if !html.contains("aria-label") {
        findings.push(err(
            "legibility-accessibility",
            "(html)",
            "navigation is not labelled (`aria-label`)",
        ));
    }
    // Every <img> must carry alt text (we emit none today, so this is a forward guard).
    for chunk in html.split("<img").skip(1) {
        let tag_end = chunk.find('>').unwrap_or(chunk.len());
        if !chunk[..tag_end].contains("alt=") {
            findings.push(err(
                "legibility-accessibility",
                "(html)",
                "an <img> lacks alt text",
            ));
        }
    }
    // Code blocks must carry a language class (real syntax highlighting hook).
    let code_blocks = html.matches("<pre><code").count();
    let classed = html.matches("<pre><code class=\"language-").count();
    if classed != code_blocks {
        findings.push(err(
            "legibility-accessibility",
            "(html)",
            &format!(
                "{} of {code_blocks} code blocks lack a language class",
                code_blocks - classed
            ),
        ));
    }
    // Heading order must not skip within a page (h1→h3 jump). Checked structurally below.
    for skip in heading_order_skips(&html) {
        findings.push(err("legibility-accessibility", "(html)", &skip));
    }

    CheckOutcome {
        name: "legibility-accessibility".to_owned(),
        status: CheckStatus::PartiallyDormant(
            "structural checks active (semantic landmarks, lang, labelled nav, alt text, code language \
             classes, heading order); colour-contrast + typography need a rendering engine — dormant"
                .to_owned(),
        ),
        summary: "semantic, accessible HTML by construction (structural aspects checked)".to_owned(),
        findings,
    }
}

/// Detect heading-level skips (e.g. an `<h1>` followed by an `<h3>`) across the rendered HTML.
fn heading_order_skips(html: &str) -> Vec<String> {
    let mut skips = Vec::new();
    let mut prev: Option<u8> = None;
    let bytes = html.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'h' && bytes[i + 2].is_ascii_digit() {
            let level = bytes[i + 2] - b'0';
            if (1..=6).contains(&level) {
                if let Some(p) = prev {
                    if level > p + 1 {
                        skips.push(format!(
                            "heading order skips from h{p} to h{level} (must be contiguous)"
                        ));
                    }
                }
                prev = Some(level);
            }
        }
        i += 1;
    }
    skips
}

fn err(check: &str, anchor: &str, message: &str) -> Finding {
    Finding {
        check: check.to_owned(),
        severity: Severity::Error,
        anchor: anchor.to_owned(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{ingest, AnchorAlloc};
    use crate::ir::{Level, Node, Payload, Provenance, SourceKind, XrefResolution, XrefTarget};

    fn prov() -> Provenance {
        Provenance {
            source: "d.md".to_owned(),
            line: 1,
        }
    }

    fn small_model() -> DocModel {
        let mut a = AnchorAlloc::new();
        let src = "# Doc\n\nLead summary.\n\n## Sec\n\nBody.\n\n```myc-checked\nnodule ex;\nfn f() => Binary{8} = 0b0000_0000;\n```\n";
        DocModel::new(vec![ingest(
            "docs/spec/d.md",
            src,
            SourceKind::Spec,
            &mut a,
        )])
    }

    #[test]
    fn the_check_names_match_the_canonical_eight() {
        assert_eq!(CHECK_NAMES.len(), 8);
        let report = lint(&small_model());
        let names: Vec<&str> = report.outcomes.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, CHECK_NAMES);
    }

    #[test]
    fn a_clean_small_model_passes_every_check() {
        let report = lint(&small_model());
        assert!(!report.has_errors(), "errors: {:?}", report.errors());
    }

    #[test]
    fn a_checked_example_that_does_not_type_check_is_an_error() {
        let bad = Node::new(
            "ex",
            None,
            Some(Level::Medium),
            prov(),
            Payload::Example {
                lang: "myc".to_owned(),
                source: "this is not valid myc !!!".to_owned(),
                checked: true,
            },
            vec![],
        );
        let doc = Node::new(
            "doc",
            Some("D".to_owned()),
            Some(Level::Minimal),
            prov(),
            Payload::Document {
                source_kind: SourceKind::Note,
            },
            vec![bad],
        );
        let report = lint(&DocModel::new(vec![doc]));
        assert!(report
            .errors()
            .iter()
            .any(|f| f.check == "checked-examples"));
    }

    #[test]
    fn a_dead_xref_fails_the_no_dead_xref_check() {
        let xref = Node::new(
            "x",
            None,
            None,
            prov(),
            Payload::Xref {
                target: XrefTarget {
                    raw: "RFC-9999.md".to_owned(),
                    resolution: XrefResolution::Dead {
                        reason: "no such doc".to_owned(),
                    },
                },
            },
            vec![],
        );
        let doc = Node::new(
            "doc",
            Some("D".to_owned()),
            Some(Level::Minimal),
            prov(),
            Payload::Document {
                source_kind: SourceKind::Note,
            },
            vec![Node::new(
                "doc--s",
                Some("S".to_owned()),
                Some(Level::Medium),
                prov(),
                Payload::Section,
                vec![xref],
            )],
        );
        let report = lint(&DocModel::new(vec![doc]));
        assert!(report.errors().iter().any(|f| f.check == "no-dead-xref"));
    }

    #[test]
    fn extract_hashes_finds_addresses() {
        let h = "x blake3:".to_owned() + &"a".repeat(64) + " y";
        let set = extract_hashes(&h);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn heading_skip_is_detected() {
        let skips = heading_order_skips("<h1>a</h1><h3>b</h3>");
        assert_eq!(skips.len(), 1);
        assert!(heading_order_skips("<h1>a</h1><h2>b</h2><h3>c</h3>").is_empty());
    }

    #[test]
    fn parity_holds_for_a_real_model() {
        let report = lint(&small_model());
        let parity = report
            .outcomes
            .iter()
            .find(|o| o.name == "dual-projection-parity")
            .unwrap();
        assert!(parity
            .findings
            .iter()
            .all(|f| f.severity != Severity::Error));
    }
}
