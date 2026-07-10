//! The HTTP front (M-1017 / DN-87 §2.3): a plain, versioned HTTP/JSON API — "the universal floor
//! (Grok, curl, anything)". An `axum` app on the `tokio` runtime, a thin async adapter over the
//! framework-agnostic [`crate::front::core`] (so its answers are byte-identical to the MCP front's —
//! front parity, the M-1017 DoD).
//!
//! Endpoints (all under `/v1/`), token-scoped via `Authorization: Bearer <token>`:
//!
//! | Method + path | Operation | Scope |
//! |---|---|---|
//! | `GET /v1/identify` | capability/version handshake | read |
//! | `GET /v1/query?kind=…&value=…` (`id`/`status`/`kind`/`text`) or `…&start=…&depth=N` (`cross_ref`) | run a query | read |
//! | `GET /v1/cite?…` | same query, citations only | read |
//! | `GET /v1/explain?…` | same query, EXPLAIN only | read |
//! | `POST /v1/refresh` | reload the served index from disk | refresh |
//!
//! Status codes: `200` for **both** an answer and a [`crate::Refusal`] (a refusal is a first-class
//! outcome carrying the same JSON envelope as over MCP — mapping it to `404` would break parity and
//! conflate it with "route not found"); `400/401/403/404/405/413` for a malformed / unauthorized /
//! insufficient-scope / unknown-route / wrong-method / oversize request. Every error body is JSON
//! (`{"error":{"code","message"}}`), never a bare status (G2).
//!
//! Security floor (DN-87 §6.4): binds `127.0.0.1` by default (a local read-only floor; TLS and any
//! public exposure are a reverse-proxy's job), a hard request-body cap (→`413`), and the token check
//! runs before any dispatch. `Declared` posture — see [`crate::front::auth`].

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Query as AxumQuery, State};
use axum::http::{header::AUTHORIZATION, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tokio::sync::RwLock;

use crate::front::auth::{Scope, TokenTable};
use crate::front::core::{self, FrontError, View};
use crate::load::load_report;
use crate::model::TeroIndexReport;

/// The hard request-body cap (a `413` past this — never-silent). The read endpoints carry no body;
/// this bounds a hostile/oversized POST body regardless.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// Shared, cheaply-cloneable server state. The report is behind an [`RwLock`] so `refresh` can
/// swap it in while readers hold a read guard.
#[derive(Debug)]
pub struct AppState {
    report: RwLock<TeroIndexReport>,
    tokens: TokenTable,
    layer2_enabled: bool,
    index_path: PathBuf,
}

impl AppState {
    /// Build server state. `index_path` is the committed `docs/tero-index/index.json` the `refresh`
    /// endpoint reloads from; `layer2_enabled` is the M-1018 gate (`false` until the eval gate opens).
    #[must_use]
    pub fn new(
        report: TeroIndexReport,
        tokens: TokenTable,
        layer2_enabled: bool,
        index_path: PathBuf,
    ) -> Self {
        AppState {
            report: RwLock::new(report),
            tokens,
            layer2_enabled,
            index_path,
        }
    }
}

/// Bind `addr` and serve the API until the process is stopped. Binds a `tokio` listener and runs the
/// `axum` app; a bind error is surfaced (never swallowed).
///
/// # Errors
/// Returns the underlying `io::Error` if the listener cannot bind `addr` or the server loop fails.
pub async fn serve_http(addr: SocketAddr, state: Arc<AppState>) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state)).await
}

/// Build the `axum` [`Router`] over `state` — the routing table + the body-size layer. Separated
/// from [`serve_http`] so the front tests can drive it in-process via
/// `tower::ServiceExt::oneshot` (no real socket).
pub(crate) fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/identify", get(identify))
        .route("/v1/query", get(query_full))
        .route("/v1/cite", get(query_cite))
        .route("/v1/explain", get(query_explain))
        .route("/v1/refresh", post(refresh))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

/// The bearer token from an `Authorization: Bearer <token>` header, if present + well-formed.
/// The scheme name is matched **case-insensitively** (RFC 7235 §2.1: auth-scheme names are
/// case-insensitive, so `bearer`/`BEARER`/… are accepted), then the token is trimmed.
fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("Bearer").then(|| token.trim())
}

/// Turn a [`FrontError`] into an HTTP `(status, json)` response.
fn err_response(e: &FrontError) -> Response {
    let status = StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(e.to_json())).into_response()
}

/// Authorize `headers` for operation `op`, returning the granted scope or a (small) [`FrontError`]
/// the caller renders. Returning `FrontError` rather than the large `axum` `Response` keeps the
/// error variant small (clippy `result_large_err`).
fn check_auth(state: &AppState, headers: &HeaderMap, op: &str) -> Result<Scope, FrontError> {
    let required = core::required_scope(op);
    state
        .tokens
        .authorize(bearer(headers), required)
        .map_err(FrontError::from)
}

async fn identify(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&state, &headers, "identify") {
        return err_response(&e);
    }
    Json(core::identify_value(state.layer2_enabled)).into_response()
}

async fn query_full(
    state: State<Arc<AppState>>,
    headers: HeaderMap,
    params: AxumQuery<HashMap<String, String>>,
) -> Response {
    run_query(&state, &headers, &params, View::Full).await
}

async fn query_cite(
    state: State<Arc<AppState>>,
    headers: HeaderMap,
    params: AxumQuery<HashMap<String, String>>,
) -> Response {
    run_query(&state, &headers, &params, View::Cite).await
}

async fn query_explain(
    state: State<Arc<AppState>>,
    headers: HeaderMap,
    params: AxumQuery<HashMap<String, String>>,
) -> Response {
    run_query(&state, &headers, &params, View::Explain).await
}

/// Shared query handler: authorize → parse `kind`/`value`/`start`/`depth` → run through the engine
/// under a read guard → the [`View`]'s JSON envelope. Auth is checked with the `query` op (read
/// scope) for all three views.
async fn run_query(
    state: &AppState,
    headers: &HeaderMap,
    params: &HashMap<String, String>,
    view: View,
) -> Response {
    if let Err(e) = check_auth(state, headers, "query") {
        return err_response(&e);
    }
    let get = |k: &str| params.get(k).map(String::as_str);
    let query = match core::parse_query(
        get("kind").unwrap_or(""),
        get("value"),
        get("start"),
        get("depth"),
    ) {
        Ok(q) => q,
        Err(e) => return err_response(&e),
    };
    let report = state.report.read().await;
    Json(core::run_and_envelope(&report, &query, view)).into_response()
}

async fn refresh(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&state, &headers, "refresh") {
        return err_response(&e);
    }
    match load_report(&state.index_path) {
        Ok(fresh) => {
            let count = fresh.items.len();
            *state.report.write().await = fresh;
            Json(json!({ "kind": "refreshed", "ok": true, "items": count })).into_response()
        }
        // A refresh failure is a *server* condition (the on-disk index went missing/unreadable),
        // not a client error — a 500 with the JSON error envelope, never a silent stale-serve (G2).
        Err(e) => err_response(&FrontError::Internal(format!(
            "could not reload {}: {e}",
            state.index_path.display()
        ))),
    }
}

/// Any unmatched route → a `404` with the JSON error envelope (never axum's default empty body).
async fn not_found() -> Response {
    err_response(&FrontError::NotFound(
        "no such route (see GET /v1/identify)".into(),
    ))
}
