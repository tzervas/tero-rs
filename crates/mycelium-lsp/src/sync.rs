//! **Document sync** (M-310; FR-S5; SC-5): the text → `Node` pipeline the LSP server runs on
//! `textDocument/didOpen` / `didChange` / `didClose`, now that RFC-0011 r3 + RFC-0001 r4 give the
//! surface a path into L0 (parse → check → elaborate).
//!
//! On each edit the server re-analyzes the *whole* document (full sync — `TextDocumentSyncKind.Full`)
//! and pushes `textDocument/publishDiagnostics`. The diagnostics are **honest about their spans**:
//! - a **parse/lex** failure carries a **real range** (the L1 lexer tracks `line:col`, `Pos`);
//! - a **type-check** failure carries a best-effort range at the offending function's `fn <name>`
//!   declaration (the checker tracks the failing *function* but not yet the failing *sub-expression*
//!   span — flagged, not fabricated: the precise sub-span awaits the checker carrying spans), with
//!   the function name in `data.breadcrumb`;
//! - a **clean** nodule yields **no** diagnostics.
//!
//! The parser is fail-fast (one error at a time); a recovering parser that reports many diagnostics
//! at once is a later refinement — recorded, not silently implied.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use mycelium_l1::{check_nodule, parse, CheckError, ParseError};

use crate::wire::SERVER_NAME;

/// An in-memory store of open documents (`uri → source text`), the minimal state full-sync requires.
#[derive(Debug, Clone, Default)]
pub struct DocumentStore {
    docs: BTreeMap<String, String>,
}

impl DocumentStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        DocumentStore::default()
    }

    /// Record (or replace) a document's full text (`didOpen` / `didChange` full sync).
    pub fn set(&mut self, uri: impl Into<String>, text: impl Into<String>) {
        self.docs.insert(uri.into(), text.into());
    }

    /// Drop a document (`didClose`).
    pub fn remove(&mut self, uri: &str) {
        self.docs.remove(uri);
    }

    /// The stored text for `uri`, if open.
    #[must_use]
    pub fn text(&self, uri: &str) -> Option<&str> {
        self.docs.get(uri).map(String::as_str)
    }

    /// Number of open documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
}

/// Analyze a document's source through the text → `Node` pipeline and return its LSP diagnostics
/// (JSON). `parse` failure → one ranged syntax diagnostic; `check_nodule` failure → one
/// function-located type diagnostic; a clean nodule → no diagnostics.
#[must_use]
pub fn source_diagnostics(text: &str) -> Vec<Value> {
    match parse(text) {
        Err(pe) => vec![parse_error_diagnostic(&pe)],
        Ok(nodule) => match check_nodule(&nodule) {
            Err(ce) => vec![check_error_diagnostic(text, &ce)],
            Ok(_env) => Vec::new(),
        },
    }
}

/// The full `textDocument/publishDiagnostics` notification for `uri`'s `text` (parse → check).
#[must_use]
pub fn publish_for_source(uri: &str, text: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": source_diagnostics(text) },
    })
}

/// Like [`source_diagnostics`], but **isolating an internal analysis panic** as a structured
/// diagnostic instead of letting it unwind through the server loop and kill the session.
///
/// The parse → check pipeline returns explicit `Result`s for *expected* failures (syntax / type
/// errors), and those flow through [`source_diagnostics`] untouched. This wrapper guards only the
/// *unexpected* case: a bug in the checker/elaborator that fires one of its internal invariant
/// `expect`s. Per **RFC-0013 I1** (a diagnostic is *additive*, never *substitutive*, and a refusal
/// is **never swallowed and never silent**), such a panic must be **surfaced** — turned into a
/// visible `internal`-coded diagnostic — not dropped (`logger.catch`-style swallowing, RFC-0013
/// §4.5 X3) and not allowed to crash every open document. The server stays alive on the next edit.
#[must_use]
pub fn resilient_source_diagnostics(text: &str) -> Vec<Value> {
    catching(move || source_diagnostics(text))
}

/// The resilient counterpart of [`publish_for_source`]: the server-boundary builder that the
/// JSON-RPC loop ([`crate::wire::serve`]) uses, so a single pathological document yields an
/// `internal` diagnostic rather than taking down the session (RFC-0013 I1).
#[must_use]
pub fn resilient_publish_for_source(uri: &str, text: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": resilient_source_diagnostics(text) },
    })
}

/// Run `f`, converting a panic into a single surfaced `internal` diagnostic (RFC-0013 I1). Factored
/// out so the isolation itself is unit-testable with a deliberately panicking closure (the
/// mutant-witness: a server that let the panic propagate would crash here instead of reporting).
fn catching<F>(f: F) -> Vec<Value>
where
    F: FnOnce() -> Vec<Value> + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(diags) => diags,
        Err(payload) => vec![internal_error_diagnostic(&panic_message(payload.as_ref()))],
    }
}

/// Extract a human string from a caught panic payload (`&str` / `String`, else an opaque marker).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

/// A zero-located `internal`-coded diagnostic carrying a caught analysis panic. The range is the
/// honest zero placeholder (an internal failure has no source span to point at — never a fabricated
/// location); the message is explicit that the failure was **surfaced, not swallowed** (RFC-0013 I1).
fn internal_error_diagnostic(detail: &str) -> Value {
    json!({
        "range": point_range(0, 0, 1),
        "severity": 1, // Error
        "code": "internal",
        "source": SERVER_NAME,
        "message": format!(
            "internal analysis error (surfaced, not swallowed — RFC-0013 I1): {detail}"
        ),
    })
}

/// A zero-based, one-character LSP range at a 1-based lexer [`Pos`].
fn point_range(line0: u32, col0: u32, width: u32) -> Value {
    json!({
        "start": { "line": line0, "character": col0 },
        "end":   { "line": line0, "character": col0 + width.max(1) },
    })
}

/// A ranged syntax diagnostic from a [`ParseError`] (the lexer's `line:col` is 1-based; LSP is
/// 0-based — never a fabricated location, the position is real).
fn parse_error_diagnostic(pe: &ParseError) -> Value {
    let line0 = pe.pos.line.saturating_sub(1);
    let col0 = pe.pos.col.saturating_sub(1);
    json!({
        "range": point_range(line0, col0, 1),
        "severity": 1, // Error
        "code": "parse",
        "source": SERVER_NAME,
        "message": pe.message,
    })
}

/// A best-effort-located type diagnostic from a [`CheckError`]. The checker tracks the failing
/// *function* (`site`), not yet the failing sub-expression span, so we locate `fn <site>` in the
/// source for a usable range and carry the function name as the navigable breadcrumb; if the name is
/// not found textually, we fall back to a zero range (honest, never a fabricated line).
fn check_error_diagnostic(text: &str, ce: &CheckError) -> Value {
    let range = locate_fn(text, &ce.site).map_or_else(
        || point_range(0, 0, 1),
        |(line0, col0)| point_range(line0, col0, ce.site.len() as u32),
    );
    json!({
        "range": range,
        "severity": 1, // Error
        "code": "check",
        "source": SERVER_NAME,
        "message": ce.message,
        "data": { "breadcrumb": [ce.site] },
    })
}

/// Locate a `fn <name>` declaration's name position (0-based `line, col`) in `text`, if present.
fn locate_fn(text: &str, name: &str) -> Option<(u32, u32)> {
    let needle = format!("fn {name}");
    for (li, line) in text.lines().enumerate() {
        if let Some(idx) = line.find(&needle) {
            // Column of the name itself (after `fn `).
            let col = idx + "fn ".len();
            return Some((li as u32, col as u32));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_nodule_has_no_diagnostics() {
        let src = "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);";
        assert!(source_diagnostics(src).is_empty());
    }

    #[test]
    fn a_parse_error_is_ranged_at_the_real_position() {
        // A missing policy on a swap is a parse error with a real position.
        let src = "nodule demo\nfn f(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6})";
        let diags = source_diagnostics(src);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["code"], "parse");
        assert_eq!(diags[0]["severity"], 1);
        // The range is real (not the zero placeholder the pre-M-310 wire used): line > 0.
        assert!(diags[0]["range"]["start"]["line"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn a_type_error_is_located_at_its_function_with_a_breadcrumb() {
        // `add` over Binary is a type error (it expects Ternary) — a check diagnostic at `fn bad`.
        let src = "nodule d;\nfn bad() => Binary{8} = add(0b0000_0001, 0b0000_0010);";
        let diags = source_diagnostics(src);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["code"], "check");
        assert_eq!(diags[0]["data"]["breadcrumb"], json!(["bad"]));
        // Located at the `fn bad` declaration line (line 1, 0-based), not a fabricated (0,0).
        assert_eq!(diags[0]["range"]["start"]["line"], 1);
    }

    #[test]
    fn the_store_tracks_open_and_closed_documents() {
        let mut store = DocumentStore::new();
        assert!(store.is_empty());
        store.set("mem://a", "nodule d;");
        assert_eq!(store.text("mem://a"), Some("nodule d;"));
        store.set("mem://a", "nodule d2;"); // didChange replaces (full sync)
        assert_eq!(store.text("mem://a"), Some("nodule d2;"));
        store.remove("mem://a");
        assert!(store.is_empty());
    }

    #[test]
    fn a_caught_analysis_panic_becomes_a_surfaced_internal_diagnostic_not_a_crash() {
        // Mutant-witness for RFC-0013 I1: a panic inside analysis must be *surfaced* as a visible
        // diagnostic, never let propagate (which would crash the server) and never swallowed. We
        // drive `catching` directly with a deliberately panicking closure — a real checker bug that
        // tripped one of its internal `expect`s would land here identically. (We do *not* touch the
        // global panic hook to suppress the expected backtrace: tests run in parallel and mutating
        // the process-wide hook would race other tests' panic reporting — one stderr line is the
        // honest cost.)
        let diags = catching(|| panic!("boom in the elaborator"));

        assert_eq!(
            diags.len(),
            1,
            "the panic is surfaced as exactly one diagnostic"
        );
        assert_eq!(diags[0]["code"], "internal");
        assert_eq!(diags[0]["severity"], 1);
        let msg = diags[0]["message"].as_str().unwrap();
        assert!(
            msg.contains("not swallowed"),
            "the message is explicit it was not swallowed (I1)"
        );
        assert!(
            msg.contains("boom in the elaborator"),
            "the panic detail is carried, not hidden"
        );
    }

    #[test]
    fn resilient_analysis_is_transparent_when_nothing_panics() {
        // The guard changes nothing on the normal paths: a clean nodule still yields no diagnostics,
        // and a type error still yields the same `check` diagnostic as the unguarded analysis.
        assert!(resilient_source_diagnostics(
            "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);"
        )
        .is_empty());
        let bad = "nodule d;\nfn bad() => Binary{8} = add(0b0000_0001, 0b0000_0010);";
        assert_eq!(resilient_source_diagnostics(bad), source_diagnostics(bad));
    }

    #[test]
    fn publish_for_source_has_the_lsp_notification_shape() {
        let note = publish_for_source(
            "mem://demo",
            "nodule d;\nfn main() => Binary{8} = 0b0000_0000;",
        );
        assert_eq!(note["method"], "textDocument/publishDiagnostics");
        assert_eq!(note["params"]["uri"], "mem://demo");
        assert_eq!(note["params"]["diagnostics"], json!([])); // clean
    }
}
