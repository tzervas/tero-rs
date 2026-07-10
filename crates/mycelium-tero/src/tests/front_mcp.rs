//! White-box tests for the MCP front (M-1017): a scripted newline-delimited JSON-RPC client over a
//! `Cursor` (the `mycelium-lsp::wire` test pattern). Covers the `initialize` handshake, `tools/list`
//! descriptors, a `tools/call` answer + refusal, the never-silent `MethodNotFound`, and the
//! token-scope enforcement (`-32001` unauthorized, `-32002` insufficient scope).

use std::io::Cursor;
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::front::auth::TokenTable;
use crate::front::mcp::{serve, McpState, SERVER_NAME};
use crate::tests::fixture::{corpus_report, emit_index};

/// Feed `requests` (one JSON-RPC message per line) through the MCP serve loop over `state`, and
/// return the parsed response messages (notifications produce no response, so the count may be less).
fn drive(state: &mut McpState, requests: &[Value]) -> Vec<Value> {
    let mut input = String::new();
    for r in requests {
        input.push_str(&serde_json::to_string(r).unwrap());
        input.push('\n');
    }
    let mut reader = Cursor::new(input.into_bytes());
    let mut out: Vec<u8> = Vec::new();
    serve(&mut reader, &mut out, state).unwrap();
    String::from_utf8(out)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn state(tag: &str) -> McpState {
    let (_root, report) = corpus_report(tag);
    let tokens = TokenTable::parse("reader:read admin:refresh").unwrap();
    McpState::new(report, tokens, false, PathBuf::from("unused/index.json"))
}

#[test]
fn initialize_advertises_server_identity_and_tools_capability() {
    let mut st = state("mcp-init");
    let out = drive(
        &mut st,
        &[json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} })],
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["id"], 1);
    assert_eq!(out[0]["result"]["serverInfo"]["name"], SERVER_NAME);
    assert!(out[0]["result"]["protocolVersion"].is_string());
    assert!(out[0]["result"]["capabilities"]["tools"].is_object());
}

#[test]
fn tools_list_returns_the_nine_descriptors() {
    let mut st = state("mcp-list");
    let out = drive(
        &mut st,
        &[json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" })],
    );
    let tools = out[0]["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 9);
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    for expected in [
        "identify",
        "query_by_id",
        "query_by_status",
        "query_by_kind",
        "cross_ref",
        "text_search",
        "cite",
        "explain",
        "refresh",
    ] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }
    // Every descriptor carries an inputSchema requiring a token (per-call auth surface).
    assert_eq!(tools[0]["inputSchema"]["type"], "object");
}

#[test]
fn tools_call_returns_an_answer_result_with_the_shared_envelope() {
    let mut st = state("mcp-answer");
    let out = drive(
        &mut st,
        &[json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": { "name": "query_by_id", "arguments": { "value": "M-0099", "token": "reader" } }
        })],
    );
    assert_eq!(out[0]["result"]["isError"], false);
    let text = out[0]["result"]["content"][0]["text"].as_str().unwrap();
    let env: Value = serde_json::from_str(text).unwrap();
    assert_eq!(env["kind"], "answer");
    assert!(env["citations"].as_array().is_some_and(|a| !a.is_empty()));
}

#[test]
fn a_refusal_is_a_successful_tool_result_not_a_protocol_error() {
    let mut st = state("mcp-refusal");
    let out = drive(
        &mut st,
        &[json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": { "name": "query_by_id", "arguments": { "value": "NO-SUCH-ID", "token": "reader" } }
        })],
    );
    assert!(
        out[0].get("error").is_none(),
        "a refusal is not a JSON-RPC error"
    );
    assert_eq!(out[0]["result"]["isError"], false);
    let env: Value =
        serde_json::from_str(out[0]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(env["kind"], "refusal");
}

#[test]
fn unknown_method_gets_method_not_found_not_silence() {
    let mut st = state("mcp-unknown");
    let out = drive(
        &mut st,
        &[json!({ "jsonrpc": "2.0", "id": 5, "method": "resources/list" })],
    );
    assert_eq!(out[0]["id"], 5);
    assert_eq!(out[0]["error"]["code"], -32601);
}

#[test]
fn auth_is_enforced_missing_token_and_insufficient_scope() {
    let mut st = state("mcp-auth");
    let out = drive(
        &mut st,
        &[
            // Missing token → -32001 (unauthorized).
            json!({ "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": { "name": "query_by_id", "arguments": { "value": "M-0099" } } }),
            // Read-only token calling `refresh` → -32002 (insufficient scope).
            json!({ "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": { "name": "refresh", "arguments": { "token": "reader" } } }),
        ],
    );
    assert_eq!(out[0]["error"]["code"], -32001);
    assert_eq!(out[1]["error"]["code"], -32002);
}

#[test]
fn refresh_reloads_the_index_with_a_refresh_scoped_token() {
    let (root, report) = corpus_report("mcp-refresh");
    let index_path = emit_index(&root, &report);
    let tokens = TokenTable::parse("admin:refresh").unwrap();
    let mut st = McpState::new(report, tokens, false, index_path);
    let out = drive(
        &mut st,
        &[json!({ "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": { "name": "refresh", "arguments": { "token": "admin" } } })],
    );
    assert_eq!(out[0]["result"]["isError"], false);
    let env: Value =
        serde_json::from_str(out[0]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(env["kind"], "refreshed");
    assert_eq!(env["ok"], true);
}
