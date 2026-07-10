//! White-box tests for the HTTP front (M-1017): the `axum` router driven in-process via
//! `tower::ServiceExt::oneshot` (no real socket). Covers the endpoints, the answer/refusal envelopes
//! (both `200`), and the full `4xx` auth/validation matrix + a `refresh` round-trip.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header::AUTHORIZATION, Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt; // oneshot

use crate::front::auth::TokenTable;
use crate::front::http::{router, AppState};
use crate::tests::fixture::{corpus_report, emit_index};

fn app(tag: &str) -> Arc<AppState> {
    let (root, report) = corpus_report(tag);
    let index_path = emit_index(&root, &report);
    let tokens = TokenTable::parse("reader:read admin:refresh").unwrap();
    Arc::new(AppState::new(report, tokens, false, index_path))
}

async fn call(
    state: &Arc<AppState>,
    method: &str,
    uri: &str,
    token: Option<&str>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(t) = token {
        builder = builder.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    let req = builder.body(Body::empty()).unwrap();
    let resp = router(Arc::clone(state)).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, json)
}

#[tokio::test]
async fn identify_and_query_answer_are_200_with_the_shared_envelope() {
    let st = app("http-ok");
    let (s, body) = call(&st, "GET", "/v1/identify", Some("reader")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["name"], "mycelium-tero");

    let (s, body) = call(&st, "GET", "/v1/query?kind=id&value=M-0099", Some("reader")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["kind"], "answer");
    assert!(body["citations"].as_array().is_some_and(|a| !a.is_empty()));
}

#[tokio::test]
async fn a_refusal_is_200_not_404() {
    let st = app("http-refusal");
    let (s, body) = call(
        &st,
        "GET",
        "/v1/query?kind=id&value=NO-SUCH-ID",
        Some("reader"),
    )
    .await;
    assert_eq!(
        s,
        StatusCode::OK,
        "a refusal is a first-class outcome, not 404"
    );
    assert_eq!(body["kind"], "refusal");
    assert_eq!(body["refusal"]["variant"], "no_match");
}

#[tokio::test]
async fn cite_and_explain_endpoints_project_the_answer() {
    let st = app("http-views");
    let (s, body) = call(&st, "GET", "/v1/cite?kind=id&value=M-0099", Some("reader")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["kind"], "citations");

    let (s, body) = call(
        &st,
        "GET",
        "/v1/explain?kind=text&value=test",
        Some("reader"),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["kind"], "explain");
}

#[tokio::test]
async fn missing_and_invalid_tokens_are_401() {
    let st = app("http-401");
    let (s, body) = call(&st, "GET", "/v1/identify", None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");

    let (s, _) = call(&st, "GET", "/v1/identify", Some("ghost")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn bearer_scheme_is_matched_case_insensitively() {
    // RFC 7235 §2.1: the auth scheme name is case-insensitive — a `bearer` (lowercase) scheme is
    // accepted, not treated as a missing token.
    let st = app("http-bearer-case");
    let req = Request::builder()
        .method("GET")
        .uri("/v1/identify")
        .header(AUTHORIZATION, "bearer reader")
        .body(Body::empty())
        .unwrap();
    let resp = router(Arc::clone(&st)).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn read_token_calling_refresh_is_403() {
    let st = app("http-403");
    let (s, body) = call(&st, "POST", "/v1/refresh", Some("reader")).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["code"], "forbidden");
}

#[tokio::test]
async fn unknown_route_is_404_and_bad_kind_is_400() {
    let st = app("http-4xx");
    let (s, body) = call(&st, "GET", "/v1/nope", Some("reader")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");

    let (s, body) = call(&st, "GET", "/v1/query?kind=bogus&value=x", Some("reader")).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "bad_request");
}

#[tokio::test]
async fn refresh_with_a_refresh_token_reloads_and_returns_the_count() {
    let st = app("http-refresh");
    let (s, body) = call(&st, "POST", "/v1/refresh", Some("admin")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(body["kind"], "refreshed");
    assert_eq!(body["ok"], true);
    assert!(body["items"].as_u64().is_some_and(|n| n > 0));
}
