//! `mycelium-sec` — **`myc-sec`**, security checks as tooling (M-367).
//!
//! v0's library core is the **`wild`-block audit** — the Mycelium-specific check no off-the-shelf scanner
//! gives. `wild` is the language's only unsafe escape hatch (LR-9/S6; DN-02 §5 — denied by default,
//! lexically marked); the audit **inventories every `wild` block** so the unsafe surface is known (never
//! ambient), and requires each to carry an **ADR-014 `// SAFETY:` justification** — an unjustified `wild`
//! is an explicit finding (G2). It is a lexical **recogniser** (like the M-141 header lints), honestly
//! scoped to *inventory + justification-presence*: it surfaces the author's `// SAFETY:` claim, it does
//! not adjudicate soundness (VR-5 — report the claim, never fabricate a verdict).
//!
//! The secrets / supply-chain families orchestrate the existing `scripts/checks/{secrets,deny}.sh` gates
//! (the bin); the load-bearing honesty rule is **skip ≠ pass** — a missing scanner is *reduced coverage*,
//! reported as such, never folded into a clean bill. KC-3: above the kernel; no new dependency.
//!
//! **M-961 (RFC-0038 / DN-77) placement note:** the inject-mode gate core (`loose`/`inoculated`
//! policy, `TrustRoot`, the `SignatureScheme` verify seam, refusals, manifest) lives at the
//! gate's insertion point — `mycelium-mlir::inject_gate` (core tier) — NOT here: this crate is
//! tools-tier, and a `core → tools` dependency would violate DN-68's `no-upward-tier-edges`
//! rule (`xtask/deps-strata.toml`; the M-883/M-884 seam precedent). Security *tooling* built in
//! this crate can depend downward on that module.

use std::path::{Path, PathBuf};

/// A finding's severity — a **fixed, declared** map (looked up, never heuristically scored; VR-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Hygiene / advisory.
    Info,
    /// Low — advisory.
    Low,
    /// An unjustified unsafe surface (an unjustified `wild` block).
    Medium,
    /// A supply-chain policy violation.
    High,
    /// Ships a secret or a known-exploitable advisory.
    Critical,
}

impl Severity {
    /// The canonical label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
            Severity::Critical => "critical",
        }
    }
}

/// One `wild` block found by the audit — located, and justified-or-not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WildBlock {
    /// The file it occurs in.
    pub file: String,
    /// 1-based line of the `wild {` opener.
    pub line: u32,
    /// Whether an ADR-014 `// SAFETY:` justification is present (same line or the preceding comment block).
    pub justified: bool,
    /// The opener line text (trimmed), for the inventory report.
    pub text: String,
}

/// A security finding — always cites *why* (G2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The check family (`wild-audit` in v0).
    pub family: &'static str,
    /// The rule code (e.g. `wild-unjustified`).
    pub rule: &'static str,
    /// Severity (fixed map).
    pub severity: Severity,
    /// Where (`file:line`).
    pub at: String,
    /// Why this is a finding, in author-facing terms.
    pub why: String,
}

/// The `wild`-audit result over a set of sources: the full inventory + the (unjustified) findings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WildAudit {
    /// Every `wild` block found (justified and not).
    pub inventory: Vec<WildBlock>,
    /// The findings (one per **unjustified** block).
    pub findings: Vec<Finding>,
}

impl WildAudit {
    /// How many blocks are justified.
    #[must_use]
    pub fn justified(&self) -> usize {
        self.inventory.iter().filter(|b| b.justified).count()
    }
    /// How many are unjustified.
    #[must_use]
    pub fn unjustified(&self) -> usize {
        self.inventory.iter().filter(|b| !b.justified).count()
    }
}

/// Audit a set of `(file, contents)` sources for `wild` blocks (LR-9/S6). Deterministic (file-ordered).
#[must_use]
pub fn audit_wild(sources: &[(String, String)]) -> WildAudit {
    let mut audit = WildAudit::default();
    let mut srcs: Vec<&(String, String)> = sources.iter().collect();
    srcs.sort_by(|a, b| a.0.cmp(&b.0));
    for (file, src) in srcs {
        let lines: Vec<&str> = src.lines().collect();
        for (i, raw) in lines.iter().enumerate() {
            if !opens_wild_block(raw) {
                continue;
            }
            let justified = has_safety(&lines, i);
            let line = (i + 1) as u32;
            audit.inventory.push(WildBlock {
                file: file.clone(),
                line,
                justified,
                text: raw.trim().to_owned(),
            });
            if !justified {
                audit.findings.push(Finding {
                    family: "wild-audit",
                    rule: "wild-unjustified",
                    severity: Severity::Medium,
                    at: format!("{file}:{line}"),
                    why: "a `wild` block (the denied-by-default unsafe escape hatch, LR-9/S6) has no \
                          adjacent `// SAFETY:` justification — every unsafe region must be justified \
                          (ADR-014); flagged, never silently trusted (G2)"
                        .to_owned(),
                });
            }
        }
    }
    audit
}

/// Whether `raw` opens a `wild` block — the `wild` keyword (whole word) followed by `{`, in the **code**
/// part of the line (a `wild` inside a `//` comment or an identifier substring does not count).
fn opens_wild_block(raw: &str) -> bool {
    // Strip a trailing line comment so a `wild` mentioned in prose isn't a false positive.
    let code = match raw.find("//") {
        Some(idx) => &raw[..idx],
        None => raw,
    };
    let bytes = code.as_bytes();
    let mut search_from = 0;
    while let Some(rel) = code[search_from..].find("wild") {
        let start = search_from + rel;
        let end = start + 4;
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        // After `wild`: optional whitespace, then `{`.
        let mut j = end;
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        let after_ok = end < bytes.len() && !is_ident_byte(bytes[end]); // `wild` is a whole word
        if before_ok && after_ok && j < bytes.len() && bytes[j] == b'{' {
            return true;
        }
        search_from = end;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether an ADR-014 `// SAFETY:` justification is present for the `wild` block on line `idx`: on the
/// same line (a trailing comment) or in the contiguous comment block immediately above it.
fn has_safety(lines: &[&str], idx: usize) -> bool {
    if line_has_safety(lines[idx]) {
        return true;
    }
    let mut k = idx;
    while k > 0 {
        k -= 1;
        let t = lines[k].trim();
        if t.is_empty() {
            // a blank line breaks the contiguous comment block above the wild
            break;
        }
        if t.starts_with("//") {
            if line_has_safety(lines[k]) {
                return true;
            }
        } else {
            break; // hit code — stop scanning upward
        }
    }
    false
}

fn line_has_safety(line: &str) -> bool {
    line.find("//")
        .map(|i| line[i..].contains("SAFETY:"))
        .unwrap_or(false)
}

/// Render the `wild`-audit `EXPLAIN` (no black box): the inventory + each unjustified finding's *why*.
#[must_use]
pub fn explain_wild(audit: &WildAudit) -> String {
    let mut out = format!(
        "wild-audit: {} block(s), {} justified, {} unjustified\n",
        audit.inventory.len(),
        audit.justified(),
        audit.unjustified()
    );
    for b in &audit.inventory {
        let tag = if b.justified {
            "justified"
        } else {
            "UNJUSTIFIED"
        };
        out.push_str(&format!("  {}:{} [{tag}] {}\n", b.file, b.line, b.text));
    }
    for f in &audit.findings {
        out.push_str(&format!(
            "  finding [{}] {} at {}: {}\n",
            f.severity.as_str(),
            f.rule,
            f.at,
            f.why
        ));
    }
    out
}

/// Collect every `.myc` under `dir` (recursively, sorted); skipping hidden entries and `target/`.
///
/// # Errors
/// Returns an I/O error string if a directory cannot be read.
pub fn collect_myc(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "myc") {
            out.push(path);
        }
    }
    Ok(())
}

// Tests live in src/tests/ (CLAUDE.md test-layout rule; extracted as-touched, M-797).
#[cfg(test)]
mod tests;
