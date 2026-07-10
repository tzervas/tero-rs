//! The **LSP wire protocol** (M-310; FR-S5; SC-5): JSON-RPC 2.0 framing over stdio plus the
//! mapping of the [`Feedback`] surface into LSP-shaped messages — the
//! "mechanical wrapping" the facade doc (M-140) flagged as the later step.
//!
//! What this layer **is**: the byte-level [`read_message`]/[`write_message`] codec (the
//! `Content-Length` header framing every LSP transport uses), the
//! [`Diagnostic`] → LSP-`Diagnostic` mapping with the proper
//! `DiagnosticSeverity` codes, the `textDocument/publishDiagnostics` notification builder, and a
//! minimal [`serve`] lifecycle loop (`initialize` → capabilities, `shutdown`/`exit`). [`serve_stdio`]
//! runs that loop over the process's real stdin/stdout — the executable an editor launches (the
//! `mycelium-lsp` binary).
//!
//! Since M-310's document-sync step (RFC-0011 r3 / RFC-0001 r4 gave the surface a text → `Node`
//! path), [`serve`] is a **document-syncing server**: it advertises `TextDocumentSyncKind.Full`,
//! handles `didOpen`/`didChange`/`didClose`, and pushes diagnostics computed through
//! [`crate::sync`] (parse → check). **Honest about spans (VR-5):** a *parse* diagnostic carries a
//! **real** `line:col` range from the lexer; a *check* diagnostic is located at its function's
//! `fn <name>` declaration (the checker tracks the failing function, not yet the failing
//! sub-expression span — flagged, never fabricated) with the function name in `data.breadcrumb`. The
//! facade's node-analysis diagnostics ([`to_lsp_diagnostic`]) still use the zero-range + breadcrumb
//! shape (they analyze Core IR nodes, which carry no spans).

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use crate::feedback::Feedback;
use crate::lint::{Diagnostic, Severity};

/// The advertised server name (LSP `serverInfo.name`).
pub const SERVER_NAME: &str = "mycelium-lsp";

/// LSP `DiagnosticSeverity` code for a [`Severity`] (LSP spec: Error=1, Warning=2, Information=3,
/// Hint=4). The lint lattice only has Error/Warning, mapped to 1/2.
#[must_use]
pub fn lsp_severity(severity: Severity) -> u8 {
    match severity {
        Severity::Error => 1,
        Severity::Warning => 2,
    }
}

/// Map a [`Diagnostic`] to an LSP-`Diagnostic` JSON value. The `range` is a **zero placeholder**
/// (L0 Core IR has no source spans yet) and the navigable location is the structured breadcrumb in
/// `data.breadcrumb` — never a fabricated line/column (M-310; spans arrive with the L1 surface).
#[must_use]
pub fn to_lsp_diagnostic(diag: &Diagnostic) -> Value {
    json!({
        "range": {
            "start": { "line": 0, "character": 0 },
            "end": { "line": 0, "character": 0 },
        },
        "severity": lsp_severity(diag.severity),
        "code": diag.code,
        "source": SERVER_NAME,
        "message": diag.message,
        // The breadcrumb path the client navigates by until real spans exist (M-310).
        "data": { "breadcrumb": diag.path() },
    })
}

/// The `params` of a `textDocument/publishDiagnostics` notification for `feedback` at `uri`.
#[must_use]
pub fn publish_diagnostics_params(uri: &str, feedback: &Feedback) -> Value {
    json!({
        "uri": uri,
        "diagnostics": feedback
            .diagnostics
            .iter()
            .map(to_lsp_diagnostic)
            .collect::<Vec<_>>(),
    })
}

/// Build the full `textDocument/publishDiagnostics` JSON-RPC **notification** (server → client) that
/// reports `feedback`'s diagnostics for the document `uri`. This is the LSP wrapping of the M-140
/// diagnostics channel; the (future) document-analysis path emits it after each [`crate::analyze`].
#[must_use]
pub fn publish_diagnostics_notification(uri: &str, feedback: &Feedback) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": publish_diagnostics_params(uri, feedback),
    })
}

/// The `initialize` result: the server's advertised capabilities. Now that the text → `Node`
/// pipeline exists (M-310; RFC-0011 r3 / RFC-0001 r4), the server advertises **`textDocumentSync: 1`**
/// (`TextDocumentSyncKind.Full`) — it re-analyzes the whole document on each edit ([`crate::sync`])
/// and pushes diagnostics via [`publish_diagnostics_notification`] / `crate::sync::publish_for_source`.
///
/// Also advertises **`completionProvider`** for the lexical/scaffolding completion provider
/// ([`crate::completions`]). The trigger characters `["/", "@"]` prompt completion while editing
/// the `// nodule:` / `// @key:` header forms; the provider is otherwise token-prefix-triggered.
/// Scope is `Declared` — lexical/scaffolding only, no semantic/type-aware resolution.
#[must_use]
pub fn initialize_result() -> Value {
    json!({
        "capabilities": {
            "textDocumentSync": 1,
            // Lexical/scaffolding completion provider (Declared: keyword + snippet list only;
            // no type-aware or scope-aware resolution -- see crate::completions module doc).
            "completionProvider": {
                "resolveProvider": false,
                "triggerCharacters": ["/", "@"],
            },
            // M-730 position-aware providers (Declared / lexical scope -- see the crate::hover,
            // crate::definition, crate::semantic module docs). Honest about their limits: hover
            // never fabricates a type, definition is single-document, semantic tokens classify by
            // token kind only.
            "hoverProvider": true,
            "definitionProvider": true,
            "semanticTokensProvider": {
                "legend": crate::semantic::semantic_tokens_legend(),
                "full": true,
            },
        },
        "serverInfo": { "name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION") },
    })
}

/// Read one JSON-RPC message off `reader`, decoding the `Content-Length` header framing. Returns
/// `Ok(None)` at a **clean** EOF (no partial header), and an `io::Error` for a malformed frame
/// (truncated body, missing/invalid `Content-Length`, or non-JSON body) — never a silent drop.
pub fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            // EOF: clean only if we were between messages (no header seen yet).
            return if content_length.is_some() {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "EOF inside LSP message headers",
                ))
            } else {
                Ok(None)
            };
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break; // blank line terminates the headers
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            let n = rest.trim().parse::<usize>().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid Content-Length")
            })?;
            content_length = Some(n);
        }
        // Any other header (e.g. Content-Type) is ignored — LSP defines only these two.
    }
    let len = content_length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
    })?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let value =
        serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(value))
}

/// Write one JSON-RPC message to `writer` with the `Content-Length` framing, then flush.
pub fn write_message<W: Write>(writer: &mut W, msg: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(msg)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()
}

fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Drive the LSP lifecycle **with document sync** (M-310) over `reader`/`writer` (stdio in the real
/// server): answer `initialize` with [`initialize_result`], acknowledge `shutdown`, stop on `exit`,
/// reply to any other **request** (a message carrying an `id`) with JSON-RPC `MethodNotFound`
/// (-32601) — never silently — and ignore unknown notifications, as the protocol requires.
///
/// On `textDocument/didOpen` and `didChange` (full sync) it stores the document's text and **pushes
/// a `textDocument/publishDiagnostics`** computed through the text → `Node` pipeline
/// ([`crate::sync::resilient_publish_for_source`]: parse → check, with an analysis panic isolated as
/// an `internal` diagnostic per RFC-0013 I1 — a pathological document never kills the session);
/// `didClose` drops the document and clears its diagnostics. Returns when the stream ends or `exit`
/// is received. A *malformed transport frame* is a different matter — it is an explicit
/// [`read_message`] `io::Error` (the byte stream is unrecoverable), surfaced to the caller, never a
/// silent drop.
pub fn serve<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<()> {
    let mut store = crate::sync::DocumentStore::new();
    while let Some(msg) = read_message(reader)? {
        let method = msg
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = msg.get("id").cloned();
        match (method, id) {
            ("initialize", Some(id)) => write_message(writer, &response(id, initialize_result()))?,
            ("shutdown", Some(id)) => write_message(writer, &response(id, Value::Null))?,
            ("exit", _) => break,

            // --- document sync (notifications; M-310) ---
            ("textDocument/didOpen", _) => {
                if let Some((uri, text)) = did_open_params(&msg) {
                    store.set(uri.clone(), text.clone());
                    write_message(
                        writer,
                        &crate::sync::resilient_publish_for_source(&uri, &text),
                    )?;
                }
            }
            ("textDocument/didChange", _) => {
                if let Some((uri, text)) = did_change_params(&msg) {
                    store.set(uri.clone(), text.clone());
                    write_message(
                        writer,
                        &crate::sync::resilient_publish_for_source(&uri, &text),
                    )?;
                }
            }
            ("textDocument/didClose", _) => {
                if let Some(uri) = doc_uri(&msg) {
                    store.remove(&uri);
                    // Clear the document's diagnostics (an empty list, per LSP).
                    write_message(
                        writer,
                        &serde_json::json!({
                            "jsonrpc": "2.0",
                            "method": "textDocument/publishDiagnostics",
                            "params": { "uri": uri, "diagnostics": [] },
                        }),
                    )?;
                }
            }

            // --- lexical/scaffolding completions (request; id is always Some) ---
            // Scope: Declared — lexical keyword + snippet list only; no semantic analysis.
            // The full static list is returned unconditionally; prefix-filtering is the client's
            // responsibility (standard LSP: the server provides candidates, the client filters).
            ("textDocument/completion", Some(id)) => {
                write_message(writer, &response(id, crate::completions::completion_list()))?;
            }

            // --- M-730 position-aware providers (requests; id is always Some) ---
            // Each looks the document text up in the sync store; an unopened/unknown document or a
            // position off any token is a null result (never-silent: no fabricated answer, G2).
            ("textDocument/hover", Some(id)) => {
                let result = position_params(&msg)
                    .and_then(|(uri, line, ch)| {
                        store.text(&uri).map(|t| crate::hover::hover(t, line, ch))
                    })
                    .unwrap_or(Value::Null);
                write_message(writer, &response(id, result))?;
            }
            ("textDocument/definition", Some(id)) => {
                let result = position_params(&msg)
                    .and_then(|(uri, line, ch)| {
                        store
                            .text(&uri)
                            .map(|t| crate::definition::definition(&uri, t, line, ch))
                    })
                    .unwrap_or(Value::Null);
                write_message(writer, &response(id, result))?;
            }
            ("textDocument/semanticTokens/full", Some(id)) => {
                let result = doc_uri(&msg)
                    .and_then(|uri| store.text(&uri).map(crate::semantic::semantic_tokens_full))
                    .unwrap_or_else(|| serde_json::json!({ "data": [] }));
                write_message(writer, &response(id, result))?;
            }

            // Any other request must get a response (never a silent hang); -32601 = MethodNotFound.
            (other, Some(id)) => write_message(
                writer,
                &error_response(id, -32601, &format!("method not handled: {other}")),
            )?,
            // Unknown notification (no id, e.g. `initialized`): nothing to answer.
            (_, None) => {}
        }
    }
    Ok(())
}

/// Run [`serve`] over the process's **real stdio** — the entry point an editor launches
/// (`mycelium-lsp` over stdin/stdout, the transport every LSP client speaks). Locks stdin/stdout
/// once for the session and drives the loop to a clean `exit` (or stream end). A transport-level
/// `io::Error` (a malformed frame, a broken pipe) propagates to the caller — the binary reports it
/// on stderr and exits non-zero rather than dropping it silently.
pub fn serve_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    serve(&mut reader, &mut writer)
}

/// `params.textDocument.uri` of a document notification.
fn doc_uri(msg: &Value) -> Option<String> {
    msg.get("params")?
        .get("textDocument")?
        .get("uri")?
        .as_str()
        .map(str::to_owned)
}

/// `(uri, text)` from a `didOpen` notification (`params.textDocument.{uri, text}`).
fn did_open_params(msg: &Value) -> Option<(String, String)> {
    let td = msg.get("params")?.get("textDocument")?;
    let uri = td.get("uri")?.as_str()?.to_owned();
    let text = td.get("text")?.as_str()?.to_owned();
    Some((uri, text))
}

/// `(uri, full text)` from a `didChange` notification under **full sync**: the whole document is the
/// last content change's `text` (`params.contentChanges[..].text`); the uri is
/// `params.textDocument.uri`.
fn did_change_params(msg: &Value) -> Option<(String, String)> {
    let uri = doc_uri(msg)?;
    let changes = msg.get("params")?.get("contentChanges")?.as_array()?;
    let text = changes.last()?.get("text")?.as_str()?.to_owned();
    Some((uri, text))
}

/// `(uri, line, character)` (0-based position) from a `textDocument/{hover,definition}` request:
/// `params.textDocument.uri` + `params.position.{line, character}`. `None` if any field is missing —
/// the caller answers with a null result rather than guessing (G2).
fn position_params(msg: &Value) -> Option<(String, u32, u32)> {
    let uri = doc_uri(msg)?;
    let position = msg.get("params")?.get("position")?;
    let line = u32::try_from(position.get("line")?.as_u64()?).ok()?;
    let character = u32::try_from(position.get("character")?.as_u64()?).ok()?;
    Some((uri, line, character))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn framing_round_trips_one_and_many_messages() {
        let a = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" });
        let b = json!({ "jsonrpc": "2.0", "method": "exit" });
        let mut buf = Vec::new();
        write_message(&mut buf, &a).unwrap();
        write_message(&mut buf, &b).unwrap();
        // The frame is the documented header + body shape.
        let text = String::from_utf8(buf.clone()).unwrap();
        assert!(text.starts_with("Content-Length: "));
        assert!(text.contains("\r\n\r\n"));

        let mut cur = Cursor::new(buf);
        assert_eq!(read_message(&mut cur).unwrap(), Some(a));
        assert_eq!(read_message(&mut cur).unwrap(), Some(b));
        // Clean EOF after the last message.
        assert_eq!(read_message(&mut cur).unwrap(), None);
    }

    #[test]
    fn empty_stream_is_clean_eof_not_an_error() {
        let mut cur = Cursor::new(Vec::new());
        assert_eq!(read_message(&mut cur).unwrap(), None);
    }

    #[test]
    fn truncated_body_is_an_explicit_error() {
        // Mutant-witness: a header promising more bytes than the body holds must error, never return
        // a partial/silent message.
        let framed = b"Content-Length: 50\r\n\r\n{\"jsonrpc\":\"2.0\"}".to_vec();
        let mut cur = Cursor::new(framed);
        assert!(read_message(&mut cur).is_err());
    }

    #[test]
    fn severity_maps_to_lsp_codes() {
        assert_eq!(lsp_severity(Severity::Error), 1);
        assert_eq!(lsp_severity(Severity::Warning), 2);
    }

    #[test]
    fn publish_diagnostics_has_the_lsp_shape() {
        let feedback = Feedback {
            diagnostics: vec![Diagnostic {
                code: "implicit-swap",
                severity: Severity::Error,
                at: "let a/swap".to_string(),
                message: "a swap must be explicit".to_string(),
            }],
            guarantees: Vec::new(),
            swaps: Vec::new(),
            stages: Vec::new(),
            explanations: Vec::new(),
            prims: Vec::new(),
        };
        let note = publish_diagnostics_notification("mem://demo", &feedback);
        assert_eq!(note["method"], "textDocument/publishDiagnostics");
        assert_eq!(note["params"]["uri"], "mem://demo");
        let d = &note["params"]["diagnostics"][0];
        assert_eq!(d["severity"], 1); // Error
        assert_eq!(d["code"], "implicit-swap");
        assert_eq!(d["source"], SERVER_NAME);
        // Honest scope: zero range placeholder, breadcrumb carries the navigable location.
        assert_eq!(d["range"]["start"]["line"], 0);
        assert_eq!(d["data"]["breadcrumb"], json!(["let a", "swap"]));
    }

    #[test]
    fn serve_answers_initialize_and_shutdown_then_exits() {
        // Scripted client: initialize → shutdown → exit. The loop must answer the two requests and
        // stop on exit (mutant-witness: dropping the `exit` arm would block on read past EOF).
        let mut input = Vec::new();
        write_message(
            &mut input,
            &json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }),
        )
        .unwrap();
        write_message(
            &mut input,
            &json!({ "jsonrpc": "2.0", "id": 2, "method": "shutdown" }),
        )
        .unwrap();
        write_message(&mut input, &json!({ "jsonrpc": "2.0", "method": "exit" })).unwrap();

        let mut reader = Cursor::new(input);
        let mut out = Vec::new();
        serve(&mut reader, &mut out).unwrap();

        let mut rout = Cursor::new(out);
        let init = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(init["id"], 1);
        assert_eq!(init["result"]["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(init["result"]["capabilities"]["textDocumentSync"], 1); // Full (M-310)
        let shut = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(shut["id"], 2);
        assert_eq!(shut["result"], Value::Null);
        // Nothing after the shutdown response (exit produced no message).
        assert_eq!(read_message(&mut rout).unwrap(), None);
    }

    #[test]
    fn serve_publishes_diagnostics_on_did_open_and_did_change() {
        // didOpen a nodule with a type error → a `check` diagnostic; didChange to a clean nodule →
        // the diagnostics clear. The mutant-witness: a server ignoring didChange would keep stale
        // diagnostics (this asserts the second publish is empty).
        let mut input = Vec::new();
        write_message(
            &mut input,
            &json!({
                "jsonrpc": "2.0", "method": "textDocument/didOpen",
                "params": { "textDocument": {
                    "uri": "mem://x", "languageId": "mycelium", "version": 1,
                    "text": "nodule d;\nfn bad() => Binary{8} = add(0b0000_0001, 0b0000_0010);"
                }}
            }),
        )
        .unwrap();
        write_message(
            &mut input,
            &json!({
                "jsonrpc": "2.0", "method": "textDocument/didChange",
                "params": {
                    "textDocument": { "uri": "mem://x", "version": 2 },
                    "contentChanges": [ { "text": "nodule d;\nfn main() => Binary{8} = not(0b0000_0001);" } ]
                }
            }),
        )
        .unwrap();
        write_message(&mut input, &json!({ "jsonrpc": "2.0", "method": "exit" })).unwrap();

        let mut reader = Cursor::new(input);
        let mut out = Vec::new();
        serve(&mut reader, &mut out).unwrap();

        let mut rout = Cursor::new(out);
        let open = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(open["method"], "textDocument/publishDiagnostics");
        assert_eq!(open["params"]["uri"], "mem://x");
        assert_eq!(open["params"]["diagnostics"][0]["code"], "check");
        let change = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(change["params"]["diagnostics"], json!([])); // cleared on the clean edit
        assert_eq!(read_message(&mut rout).unwrap(), None);
    }

    #[test]
    fn unknown_request_gets_method_not_found_not_silence() {
        let mut input = Vec::new();
        write_message(
            &mut input,
            // `textDocument/rename` is not advertised or handled (M-730 added hover/definition/
            // semanticTokens, not rename); an unhandled request must still get an explicit -32601.
            &json!({ "jsonrpc": "2.0", "id": 7, "method": "textDocument/rename" }),
        )
        .unwrap();
        write_message(&mut input, &json!({ "jsonrpc": "2.0", "method": "exit" })).unwrap();
        let mut reader = Cursor::new(input);
        let mut out = Vec::new();
        serve(&mut reader, &mut out).unwrap();
        let mut rout = Cursor::new(out);
        let resp = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(resp["id"], 7);
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn completion_request_returns_keyword_and_snippet_list() {
        // Scripted client: textDocument/completion → must respond with a CompletionList that has
        // the LSP `isIncomplete + items` shape, contains at least one keyword item (kind=14) and
        // one snippet item (kind=15), and specifically includes `nodule` and the `swap-expr` snippet.
        // Mutant-witness: a server falling through to MethodNotFound (-32601) would fail the `items`
        // array check and the kind assertions.
        let mut input = Vec::new();
        write_message(
            &mut input,
            &json!({
                "jsonrpc": "2.0", "id": 5, "method": "textDocument/completion",
                "params": {
                    "textDocument": { "uri": "mem://t" },
                    "position": { "line": 0, "character": 2 },
                }
            }),
        )
        .unwrap();
        write_message(&mut input, &json!({ "jsonrpc": "2.0", "method": "exit" })).unwrap();

        let mut reader = Cursor::new(input);
        let mut out = Vec::new();
        serve(&mut reader, &mut out).unwrap();

        let mut rout = Cursor::new(out);
        let resp = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(resp["id"], 5);
        // Must be a success response, not an error.
        assert!(
            resp.get("error").is_none(),
            "completion must not return an error"
        );
        let result = &resp["result"];
        assert_eq!(result["isIncomplete"], false);
        let items = result["items"].as_array().expect("items must be an array");
        assert!(!items.is_empty(), "completion list must be non-empty");

        // At least one keyword item (kind=14: `nodule`) and one snippet item (kind=15).
        let has_keyword = items.iter().any(|i| i["kind"] == 14);
        let has_snippet = items.iter().any(|i| i["kind"] == 15);
        assert!(has_keyword, "must have at least one keyword item (kind=14)");
        assert!(has_snippet, "must have at least one snippet item (kind=15)");

        // `nodule` keyword must be present (it is the primary structural keyword).
        let nodule = items.iter().find(|i| i["label"] == "nodule");
        assert!(nodule.is_some(), "`nodule` keyword must appear in the list");
        assert_eq!(nodule.unwrap()["kind"], 14);

        // `swap-expr` snippet must be present and use snippet format (2).
        let swap_snip = items.iter().find(|i| i["label"] == "swap-expr");
        assert!(
            swap_snip.is_some(),
            "`swap-expr` snippet must appear in the list"
        );
        assert_eq!(swap_snip.unwrap()["insertTextFormat"], 2); // snippet grammar

        // `phylum` (still reserved-not-active) must NOT appear in completions.
        // `colony` and `hypha` moved to active in M-666 and MUST now appear.
        let has_phylum = items.iter().any(|i| i["label"] == "phylum");
        let has_colony = items.iter().any(|i| i["label"] == "colony");
        let has_hypha = items.iter().any(|i| i["label"] == "hypha");
        assert!(
            !has_phylum,
            "`phylum` (reserved-not-active) must not be offered"
        );
        assert!(
            has_colony,
            "`colony` is now an active keyword (M-666) and must appear in completions"
        );
        assert!(
            has_hypha,
            "`hypha` is now an active keyword (M-666) and must appear in completions"
        );

        // Server stops cleanly after exit.
        assert_eq!(read_message(&mut rout).unwrap(), None);
    }

    #[test]
    fn initialize_result_advertises_completion_provider() {
        // The `initialize` response must include `completionProvider` so clients know to
        // request completions. Mutant-witness: removing the field from `initialize_result()`
        // would break editor discovery and make completion requests unreachable in practice.
        let result = initialize_result();
        assert!(
            result["capabilities"]["completionProvider"].is_object(),
            "capabilities must include completionProvider"
        );
        assert_eq!(
            result["capabilities"]["completionProvider"]["resolveProvider"], false,
            "resolveProvider must be false (static list, no resolve step)"
        );
        let triggers = result["capabilities"]["completionProvider"]["triggerCharacters"]
            .as_array()
            .expect("triggerCharacters must be an array");
        assert!(
            triggers.iter().any(|t| t == "/"),
            "triggerCharacters must include '/' for the nodule-header snippet"
        );
    }

    #[test]
    fn initialize_result_advertises_the_m730_providers() {
        // M-730: hover/definition/semanticTokens must be advertised, and the semantic-tokens legend
        // must carry the type list (mutant-witness: dropping any provider breaks editor discovery).
        let caps = &initialize_result()["capabilities"];
        assert_eq!(caps["hoverProvider"], true);
        assert_eq!(caps["definitionProvider"], true);
        assert_eq!(caps["semanticTokensProvider"]["full"], true);
        let types = caps["semanticTokensProvider"]["legend"]["tokenTypes"]
            .as_array()
            .expect("semantic-token legend must list token types");
        assert!(types.iter().any(|t| t == "keyword"));
    }

    #[test]
    fn serve_answers_hover_definition_and_semantic_tokens_after_did_open() {
        // End-to-end through the document store: didOpen a nodule, then request hover (on `fn`),
        // definition (on a call site), and semanticTokens/full. Each must answer from the stored
        // text, never -32601 and never silence.
        let src = "nodule d\nfn g() -> Binary{8} = 0b0\nfn h() = g()\n";
        let mut input = Vec::new();
        write_message(
            &mut input,
            &json!({
                "jsonrpc": "2.0", "method": "textDocument/didOpen",
                "params": { "textDocument": { "uri": "mem://d.myc", "text": src } }
            }),
        )
        .unwrap();
        // hover on `fn` (line 1, char 0).
        write_message(
            &mut input,
            &json!({ "jsonrpc": "2.0", "id": 10, "method": "textDocument/hover",
                "params": { "textDocument": { "uri": "mem://d.myc" }, "position": { "line": 1, "character": 0 } } }),
        )
        .unwrap();
        // definition on the `g` call site (line 2 `fn h() = g()`, char of `g` is 9).
        write_message(
            &mut input,
            &json!({ "jsonrpc": "2.0", "id": 11, "method": "textDocument/definition",
                "params": { "textDocument": { "uri": "mem://d.myc" }, "position": { "line": 2, "character": 9 } } }),
        )
        .unwrap();
        write_message(
            &mut input,
            &json!({ "jsonrpc": "2.0", "id": 12, "method": "textDocument/semanticTokens/full",
                "params": { "textDocument": { "uri": "mem://d.myc" } } }),
        )
        .unwrap();
        write_message(&mut input, &json!({ "jsonrpc": "2.0", "method": "exit" })).unwrap();

        let mut reader = Cursor::new(input);
        let mut out = Vec::new();
        serve(&mut reader, &mut out).unwrap();

        let mut rout = Cursor::new(out);
        // The didOpen publishes diagnostics first (skip that notification).
        let first = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(first["method"], "textDocument/publishDiagnostics");
        let hover = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(hover["id"], 10);
        assert!(
            hover["result"]["contents"]["value"]
                .as_str()
                .unwrap()
                .contains("function"),
            "hover on `fn` must describe it"
        );
        let def = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(def["id"], 11);
        assert_eq!(
            def["result"]["range"]["start"]["line"], 1,
            "g is declared on line 2 (0-based 1)"
        );
        let sem = read_message(&mut rout).unwrap().unwrap();
        assert_eq!(sem["id"], 12);
        assert!(
            !sem["result"]["data"].as_array().unwrap().is_empty(),
            "semantic tokens must be non-empty for a real nodule"
        );
    }
}
