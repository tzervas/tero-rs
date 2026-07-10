//! The M-1017 DoD "second-platform (non-Claude) smoke": a **raw TCP** client — the curl-equivalent,
//! nothing Claude/MCP-specific — hits a real ephemeral-port `tero-http` server and gets a cited
//! answer. The request/response byte transcript is printed (`cargo test -- --nocapture`) as the
//! recorded non-Claude demonstration.

use std::sync::Arc;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::front::auth::TokenTable;
use crate::front::http::{router, AppState};
use crate::tests::fixture::{corpus_report, emit_index};

#[tokio::test]
async fn raw_tcp_curl_equivalent_client_gets_a_cited_answer() {
    let (root, report) = corpus_report("smoke");
    let index_path = emit_index(&root, &report);
    let tokens = TokenTable::parse("demo:read").unwrap();
    let state = Arc::new(AppState::new(report, tokens, false, index_path));

    // A real socket on an ephemeral port (127.0.0.1:0 → the OS picks a free port).
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });

    // A hand-written HTTP/1.1 request — no Claude, no MCP, no HTTP client library beyond raw bytes.
    let request = format!(
        "GET /v1/query?kind=id&value=M-0099 HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Authorization: Bearer demo\r\n\
         Connection: close\r\n\r\n"
    );
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(request.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    sock.read_to_end(&mut raw).await.unwrap();
    let response = String::from_utf8_lossy(&raw).into_owned();

    // Record the transcript (visible with --nocapture) — the non-Claude smoke evidence.
    println!("=== tero-http non-Claude smoke (raw TCP / curl-equivalent) ===");
    println!("--> request:\n{request}");
    println!("<-- response:\n{response}");

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "expected a 200 status line, got:\n{response}"
    );
    let body = response
        .split("\r\n\r\n")
        .nth(1)
        .expect("response has a body")
        .trim();
    let json: Value = serde_json::from_str(body).expect("body is JSON");
    assert_eq!(json["kind"], "answer");
    assert!(
        json["citations"].as_array().is_some_and(|a| !a.is_empty()),
        "the non-Claude client got a cited answer"
    );

    server.abort();
}
