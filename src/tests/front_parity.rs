//! The M-1017 DoD parity test: a fixed query battery driven three ways — the engine directly, the
//! HTTP front, and the MCP front — asserting **byte-identical** answer/refusal JSON across all
//! three. Parity holds by construction (one `front::core` serializer), and this is its differential
//! witness; a divergence in either front is a regression this catches.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header::AUTHORIZATION, Request};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::front::auth::TokenTable;
use crate::front::core::{self, View};
use crate::front::http::{router, AppState};
use crate::front::mcp::{serve, McpState};
use crate::query::Query;
use crate::tests::fixture::corpus_report;

async fn http_env(state: &Arc<AppState>, uri: &str) -> Value {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .header(AUTHORIZATION, "Bearer reader")
        .body(Body::empty())
        .unwrap();
    let resp = router(Arc::clone(state)).oneshot(req).await.unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn mcp_env(state: &mut McpState, tool: &str, args: Value) -> Value {
    let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": tool, "arguments": args } });
    let mut input = serde_json::to_string(&req).unwrap();
    input.push('\n');
    let mut reader = Cursor::new(input.into_bytes());
    let mut out = Vec::new();
    serve(&mut reader, &mut out, state).unwrap();
    let text_line = String::from_utf8(out).unwrap();
    let resp: Value = serde_json::from_str(text_line.lines().next().unwrap()).unwrap();
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    serde_json::from_str(text).unwrap()
}

#[tokio::test]
async fn engine_http_and_mcp_agree_on_every_answer_and_refusal() {
    let (_root, report) = corpus_report("parity");
    let toks = || TokenTable::parse("reader:read").unwrap();
    let http = Arc::new(AppState::new(
        report.clone(),
        toks(),
        false,
        PathBuf::from("unused"),
    ));
    let mut mcp = McpState::new(report.clone(), toks(), false, PathBuf::from("unused"));

    // (engine query, HTTP uri, MCP tool, MCP args) — the last case is a refusal (unknown id).
    let cases: Vec<(Query, &str, &str, Value)> = vec![
        (
            Query::Id("M-0099".into()),
            "/v1/query?kind=id&value=M-0099",
            "query_by_id",
            json!({ "value": "M-0099", "token": "reader" }),
        ),
        (
            Query::Status("todo".into()),
            "/v1/query?kind=status&value=todo",
            "query_by_status",
            json!({ "value": "todo", "token": "reader" }),
        ),
        (
            Query::Kind("rfc".into()),
            "/v1/query?kind=kind&value=rfc",
            "query_by_kind",
            json!({ "value": "rfc", "token": "reader" }),
        ),
        (
            Query::Text("test".into()),
            "/v1/query?kind=text&value=test",
            "text_search",
            json!({ "value": "test", "token": "reader" }),
        ),
        (
            Query::CrossRef {
                start: "M-0099".into(),
                depth: 2,
            },
            "/v1/query?kind=cross_ref&start=M-0099&depth=2",
            "cross_ref",
            json!({ "start": "M-0099", "depth": "2", "token": "reader" }),
        ),
        (
            Query::Id("NO-SUCH-ID".into()),
            "/v1/query?kind=id&value=NO-SUCH-ID",
            "query_by_id",
            json!({ "value": "NO-SUCH-ID", "token": "reader" }),
        ),
    ];

    for (query, uri, tool, args) in cases {
        let engine = core::run_and_envelope(&report, &query, View::Full);
        let http_v = http_env(&http, uri).await;
        let mcp_v = mcp_env(&mut mcp, tool, args);
        assert_eq!(engine, http_v, "HTTP diverged from the engine for {uri}");
        assert_eq!(
            engine, mcp_v,
            "MCP diverged from the engine for tool {tool}"
        );
    }
}
