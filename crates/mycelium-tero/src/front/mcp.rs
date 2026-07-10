//! The MCP front (M-1017 / DN-87 §2.3): a Model Context Protocol server over stdio, giving native
//! tool ergonomics to any MCP-speaking platform. A thin adapter over the framework-agnostic
//! [`crate::front::core`] — its answers are byte-identical to the HTTP front's (front parity, the
//! M-1017 DoD).
//!
//! **Transport.** JSON-RPC 2.0, **newline-delimited** (one compact JSON object per line — the MCP
//! stdio framing, in contrast to LSP's `Content-Length` headers). The serve loop is modeled on
//! `mycelium-lsp`'s `wire::serve`: `initialize` handshake, request-by-`id`, notifications ignored,
//! and an explicit JSON-RPC `MethodNotFound (-32601)` for any unhandled request — never a silent
//! drop (G2). A malformed (non-JSON) line is an explicit `io::Error`, not a silent skip.
//!
//! **Tools** (one per engine operation) are advertised by `tools/list` and invoked by `tools/call`.
//! An answer/refusal is returned as an `isError:false` tool result whose `text` is the compact
//! [`crate::front::core`] envelope; a refusal is a first-class result, not a protocol error. Only a
//! malformed/unauthorized/unknown call is a JSON-RPC error.
//!
//! **Auth.** Each `tools/call` carries a `token` argument (the bearer, from `TERO_TOKENS`); it is
//! checked against the operation's required [`Scope`](crate::front::auth::Scope) before dispatch
//! (read-only by default; `refresh` needs the broader scope). `initialize`/`tools/list` are the open
//! capability handshake; the token gates the data operations.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::front::auth::TokenTable;
use crate::front::core::{self, FrontError, View};
use crate::load::load_report;
use crate::model::TeroIndexReport;

/// The advertised MCP `serverInfo.name`.
pub const SERVER_NAME: &str = "tero-mcp";

/// The MCP protocol version this server advertises in `initialize`.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// One stdio MCP session's state. Single-threaded (one client over stdio), so the report is a plain
/// owned value the `refresh` tool swaps in place — no lock needed (unlike the concurrent HTTP front).
pub(crate) struct McpState {
    report: TeroIndexReport,
    tokens: TokenTable,
    layer2_enabled: bool,
    index_path: PathBuf,
}

impl McpState {
    /// Build session state. `index_path` is the `docs/tero-index/index.json` the `refresh` tool
    /// reloads; `layer2_enabled` is the M-1018 gate (`false` until the eval gate opens).
    pub(crate) fn new(
        report: TeroIndexReport,
        tokens: TokenTable,
        layer2_enabled: bool,
        index_path: PathBuf,
    ) -> Self {
        McpState {
            report,
            tokens,
            layer2_enabled,
            index_path,
        }
    }
}

/// Run the MCP server over the process's real stdio — the entry point an MCP client launches
/// (`tero-mcp` over stdin/stdout). Locks stdio once and drives [`serve`] to stream end.
///
/// # Errors
/// Propagates a transport-level `io::Error` (a malformed frame, a broken pipe) — never dropped.
pub fn serve_mcp_stdio(
    report: TeroIndexReport,
    tokens: TokenTable,
    layer2_enabled: bool,
    index_path: PathBuf,
) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let mut state = McpState::new(report, tokens, layer2_enabled, index_path);
    serve(&mut reader, &mut writer, &mut state)
}

/// Read one newline-delimited JSON-RPC message off `reader`. Skips blank lines; returns `Ok(None)`
/// at a clean EOF (between messages) and an `io::Error` for a non-JSON line — never a silent skip.
fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(None); // clean EOF between messages
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // blank line between messages — ignore
        }
        let value = serde_json::from_str(trimmed)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        return Ok(Some(value));
    }
}

/// Write one JSON-RPC message as a single compact line (`\n`-terminated), then flush.
fn write_message<W: Write>(writer: &mut W, msg: &Value) -> io::Result<()> {
    let mut body = serde_json::to_vec(msg)?;
    body.push(b'\n');
    writer.write_all(&body)?;
    writer.flush()
}

fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Drive the MCP lifecycle over `reader`/`writer` (stdio in the real server): answer `initialize`
/// with the capability handshake, `tools/list` with the tool descriptors, `tools/call` by
/// dispatching through [`crate::front::core`], `ping` with `{}`; reply to any other **request** (a
/// message with an `id`) with `MethodNotFound (-32601)` — never silently — and ignore notifications.
/// Returns when the stream ends. Testable: the parity/mcp tests drive it over a `Cursor`.
pub(crate) fn serve<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    state: &mut McpState,
) -> io::Result<()> {
    while let Some(msg) = read_message(reader)? {
        let method = msg
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = msg.get("id").cloned();
        match (method, id) {
            ("initialize", Some(id)) => {
                write_message(writer, &response(id, initialize_result(state)))?;
            }
            ("ping", Some(id)) => write_message(writer, &response(id, json!({})))?,
            ("tools/list", Some(id)) => {
                write_message(
                    writer,
                    &response(id, json!({ "tools": tool_descriptors() })),
                )?;
            }
            ("tools/call", Some(id)) => {
                let outcome = handle_tools_call(state, &msg);
                write_message(writer, &finish_call(id, outcome))?;
            }
            // Any other request must get a response (never a silent hang); -32601 = MethodNotFound.
            (other, Some(id)) => {
                write_message(
                    writer,
                    &error_response(id, -32601, &format!("method not handled: {other}")),
                )?;
            }
            // Unknown notification (no id, e.g. `notifications/initialized`): nothing to answer.
            (_, None) => {}
        }
    }
    Ok(())
}

/// The `initialize` result: protocol version, server identity, and the `tools` capability. `Declared`
/// scope — the auth/token model is documented in `instructions`, not a protocol guarantee.
fn initialize_result(state: &McpState) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": { "name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION") },
        "capabilities": { "tools": { "listChanged": false } },
        "instructions": format!(
            "mycelium-tero memory API ({}). tools/list, then tools/call with a `token` argument \
             (from TERO_TOKENS). Every answer carries resolvable citations + an EXPLAIN trace; a \
             query that finds nothing citable is a typed refusal, not an empty answer. Layer-2 \
             (VSA) is {}.",
            crate::crate_summary(),
            if state.layer2_enabled { "enabled" } else { "gated off (serving Layer-1)" },
        ),
    })
}

/// Handle a `tools/call`: extract `name` + `arguments`, authorize the `token` argument against the
/// tool's required scope, then dispatch. Returns the envelope on success or a [`FrontError`].
fn handle_tools_call(state: &mut McpState, msg: &Value) -> Result<Value, FrontError> {
    let params = msg.get("params");
    let name = params
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| FrontError::BadRequest("tools/call requires a string `name`".into()))?;
    let empty = json!({});
    let args = params.and_then(|p| p.get("arguments")).unwrap_or(&empty);

    let token = args.get("token").and_then(Value::as_str);
    state
        .tokens
        .authorize(token, core::required_scope(name))
        .map_err(FrontError::from)?;

    dispatch(state, name, args)
}

/// Dispatch an authorized tool call to the engine (or `refresh`).
fn dispatch(state: &mut McpState, name: &str, args: &Value) -> Result<Value, FrontError> {
    let get = |k: &str| args.get(k).and_then(Value::as_str);
    match name {
        "identify" => Ok(core::identify_value(state.layer2_enabled)),
        "query_by_id" => query(state, "id", get("value"), None, None, View::Full),
        "query_by_status" => query(state, "status", get("value"), None, None, View::Full),
        "query_by_kind" => query(state, "kind", get("value"), None, None, View::Full),
        "cross_ref" => query(
            state,
            "cross_ref",
            None,
            get("start"),
            get("depth"),
            View::Full,
        ),
        "text_search" => query(state, "text", get("value"), None, None, View::Full),
        "cite" => query(
            state,
            get("kind").unwrap_or(""),
            get("value"),
            get("start"),
            get("depth"),
            View::Cite,
        ),
        "explain" => query(
            state,
            get("kind").unwrap_or(""),
            get("value"),
            get("start"),
            get("depth"),
            View::Explain,
        ),
        "refresh" => refresh(state),
        other => Err(FrontError::BadRequest(format!(
            "unknown tool {other:?} (see tools/list)"
        ))),
    }
}

/// Parse + run a query, returning the [`View`]'s envelope (identical to the HTTP front's).
fn query(
    state: &McpState,
    kind: &str,
    value: Option<&str>,
    start: Option<&str>,
    depth: Option<&str>,
    view: View,
) -> Result<Value, FrontError> {
    let q = core::parse_query(kind, value, start, depth)?;
    Ok(core::run_and_envelope(&state.report, &q, view))
}

/// Reload the served index from disk (the `refresh` tool). A load failure is a server-side
/// [`FrontError::Internal`], never a silent stale-serve (G2).
fn refresh(state: &mut McpState) -> Result<Value, FrontError> {
    let fresh = load_report(&state.index_path).map_err(|e| {
        FrontError::Internal(format!(
            "could not reload {}: {e}",
            state.index_path.display()
        ))
    })?;
    let count = fresh.items.len();
    state.report = fresh;
    Ok(json!({ "kind": "refreshed", "ok": true, "items": count }))
}

/// Wrap a dispatch outcome as a JSON-RPC response: an envelope becomes an `isError:false` tool
/// result (its compact JSON as the `text` content); a [`FrontError`] becomes a JSON-RPC error.
fn finish_call(id: Value, outcome: Result<Value, FrontError>) -> Value {
    match outcome {
        Ok(envelope) => {
            let text = serde_json::to_string(&envelope)
                .expect("tero front envelope serializes infallibly");
            response(
                id,
                json!({ "content": [ { "type": "text", "text": text } ], "isError": false }),
            )
        }
        Err(e) => error_response(id, e.jsonrpc_code(), e.message()),
    }
}

/// The `tools/list` descriptors — one per engine operation, each with a small JSON-Schema
/// `inputSchema` (parameter names + types) so MCP clients get native argument ergonomics. This IS
/// the platform-agnostic skill surface (the `.claude/skills/tero-*` are the Claude packaging).
pub fn tool_descriptors() -> Value {
    // `token` is required on every tool (the per-call bearer). Query tools add their own args.
    let tok = json!({ "type": "string", "description": "bearer token (from TERO_TOKENS)" });
    json!([
        tool(
            "identify",
            "Server identity, version, and whether the Layer-2 gate is open.",
            json!({ "token": tok }),
            &["token"],
            "introspection"
        ),
        tool(
            "query_by_id",
            "Exact lookup by corpus id (RFC-0034, M-1015, DN-87, an issue id).",
            json!({ "value": { "type": "string", "description": "the id to match" }, "token": tok }),
            &["value", "token"],
            "query"
        ),
        tool(
            "query_by_status",
            "All rows with a given status (Accepted, todo, done, …).",
            json!({ "value": { "type": "string" }, "token": tok }),
            &["value", "token"],
            "query"
        ),
        tool(
            "query_by_kind",
            "All rows of a given kind (rfc, adr, note, issue, section, …).",
            json!({ "value": { "type": "string" }, "token": tok }),
            &["value", "token"],
            "query"
        ),
        tool(
            "cross_ref",
            "Breadth-first walk of depends_on/doc_refs edges from a start id/anchor.",
            json!({ "start": { "type": "string" },
                     "depth": { "type": "string", "description": "hop count (default 1)" },
                     "token": tok }),
            &["start", "token"],
            "query"
        ),
        tool(
            "text_search",
            "Ranked free-text search over id/title/summary.",
            json!({ "value": { "type": "string", "description": "the query text" }, "token": tok }),
            &["value", "token"],
            "query"
        ),
        tool(
            "cite",
            "Citations only for a query (kind + its args, as query_*).",
            json!({ "kind": { "type": "string", "description": "id|status|kind|cross_ref|text" },
                     "value": { "type": "string" }, "start": { "type": "string" },
                     "depth": { "type": "string" }, "token": tok }),
            &["kind", "token"],
            "explain"
        ),
        tool(
            "explain",
            "EXPLAIN trace only for a query (kind + its args, as query_*).",
            json!({ "kind": { "type": "string" }, "value": { "type": "string" },
                     "start": { "type": "string" }, "depth": { "type": "string" }, "token": tok }),
            &["kind", "token"],
            "explain"
        ),
        tool(
            "refresh",
            "Reload the served index from disk (requires the `refresh` scope).",
            json!({ "token": tok }),
            &["token"],
            "maintenance"
        ),
    ])
}

/// One tool descriptor.
fn tool(
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
    category: &str,
) -> Value {
    json!({
        "name": name,
        "description": description,
        "category": category,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
}
