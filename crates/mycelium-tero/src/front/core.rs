//! The **one core** behind the two fronts (M-1017 / DN-87 §2.3): parse a normalized request into a
//! [`Query`], run it through the M-1016 engine, and render the outcome as a stable JSON envelope —
//! *framework-agnostic*, so the MCP and HTTP fronts are thin adapters that share this exact logic
//! and therefore produce byte-identical answers (front parity, the M-1017 DoD).
//!
//! The envelope shapes (deterministic — every field order is a struct/`json!` literal, and the
//! model types serialize in declaration order):
//!
//! - answer  → `{"kind":"answer","items":[…],"citations":[…],"explain":{…}}`
//! - cite    → `{"kind":"citations","citations":[…]}`
//! - explain → `{"kind":"explain","explain":{…}}`
//! - refusal → `{"kind":"refusal","refusal":{"variant":…,…},"message":"…"}`
//! - error   → `{"error":{"code":"…","message":"…"}}`
//!
//! A refusal is a **first-class, `200`/`isError:false` outcome**, not a transport error: the engine
//! found nothing citable and said so (never-silent). Only a malformed/unauthorized/unknown request
//! is a [`FrontError`] (a real `4xx` / JSON-RPC error). Mapping "no citable row" to `404` would
//! conflate it with "route not found" and break parity — deliberately rejected.

use serde::Serialize;
use serde_json::{json, Value};

use crate::front::auth::Scope;
use crate::model::{TeroIndexItem, TeroIndexReport, SIBLING_INDICES};
use crate::query::{Answer, Citation, Explain, Query, QueryEngine, Refusal};

/// Which projection of an [`Answer`] a front asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum View {
    /// The whole answer: items + citations + EXPLAIN (`query_by_*` / `cross_ref` / `text_search`).
    Full,
    /// Citations only (the `cite` operation).
    Cite,
    /// The EXPLAIN trace only (the `explain` operation).
    Explain,
}

/// A front-agnostic client-or-transport error, mapped to an HTTP status **or** a JSON-RPC error code
/// by whichever front raised it — one error source, two mappings (SoC). Distinct from a [`Refusal`],
/// which is a *successful* "nothing citable" outcome, not an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FrontError {
    /// Malformed request: missing/invalid argument, unknown query kind/tool, unparseable depth.
    BadRequest(String),
    /// Missing or invalid auth token.
    Unauthorized(String),
    /// Valid token, but its scope does not permit this operation.
    Forbidden(String),
    /// Unknown route/operation.
    NotFound(String),
    /// A server-side failure (e.g. the on-disk index could not be reloaded on `refresh`) — a `500` /
    /// JSON-RPC internal error, never a silent stale-serve.
    Internal(String),
}

impl FrontError {
    /// The HTTP status for this error (the `http` front's mapping). `405`/`413` are emitted natively
    /// by `axum` (method routing + `DefaultBodyLimit`), so they are not modeled here.
    pub(crate) fn http_status(&self) -> u16 {
        match self {
            FrontError::BadRequest(_) => 400,
            FrontError::Unauthorized(_) => 401,
            FrontError::Forbidden(_) => 403,
            FrontError::NotFound(_) => 404,
            FrontError::Internal(_) => 500,
        }
    }

    /// The JSON-RPC error code for this error (the `mcp` front's mapping). `-32601`/`-32602`/`-32603`
    /// are the reserved JSON-RPC codes; auth uses the `-320xx` implementation-defined range.
    pub(crate) fn jsonrpc_code(&self) -> i64 {
        match self {
            FrontError::BadRequest(_) => -32602,   // Invalid params
            FrontError::NotFound(_) => -32601,     // Method not found
            FrontError::Internal(_) => -32603,     // Internal error
            FrontError::Unauthorized(_) => -32001, // impl-defined: unauthorized
            FrontError::Forbidden(_) => -32002,    // impl-defined: insufficient scope
        }
    }

    /// A stable machine slug for the error kind (goes in the JSON `error.code`).
    pub(crate) fn slug(&self) -> &'static str {
        match self {
            FrontError::BadRequest(_) => "bad_request",
            FrontError::Unauthorized(_) => "unauthorized",
            FrontError::Forbidden(_) => "forbidden",
            FrontError::NotFound(_) => "not_found",
            FrontError::Internal(_) => "internal",
        }
    }

    /// The human-readable detail.
    pub(crate) fn message(&self) -> &str {
        match self {
            FrontError::BadRequest(m)
            | FrontError::Unauthorized(m)
            | FrontError::Forbidden(m)
            | FrontError::NotFound(m)
            | FrontError::Internal(m) => m,
        }
    }

    /// The `{"error":{"code","message"}}` JSON body (never a bare status — G2).
    pub(crate) fn to_json(&self) -> Value {
        json!({ "error": { "code": self.slug(), "message": self.message() } })
    }
}

/// The required [`Scope`] for an operation name: everything is read-only except `refresh` (DN-87
/// §6.4 "read-only by default"). One source of truth, called by both fronts before dispatch.
pub(crate) fn required_scope(op: &str) -> Scope {
    match op {
        "refresh" => Scope::Refresh,
        _ => Scope::Read,
    }
}

/// Build a [`Query`] from a normalized `kind` + its string arguments (the wire form each front has
/// already extracted). `value` is required-present (but may be empty — an empty `id`/`text` is
/// passed through so the *engine* refuses it, preserving parity with direct engine use); `cross_ref`
/// requires `start` and takes an optional `depth` (default `1`).
pub(crate) fn parse_query(
    kind: &str,
    value: Option<&str>,
    start: Option<&str>,
    depth: Option<&str>,
) -> Result<Query, FrontError> {
    match kind {
        "id" => Ok(Query::Id(present(value, "value")?.to_owned())),
        "status" => Ok(Query::Status(present(value, "value")?.to_owned())),
        "kind" => Ok(Query::Kind(present(value, "value")?.to_owned())),
        "text" => Ok(Query::Text(present(value, "value")?.to_owned())),
        "cross_ref" => {
            let start = present(start, "start")?.to_owned();
            let depth = match depth {
                None => 1,
                Some(d) => d.parse::<usize>().map_err(|_| {
                    FrontError::BadRequest(format!(
                        "`depth` must be a non-negative integer, got {d:?}"
                    ))
                })?,
            };
            Ok(Query::CrossRef { start, depth })
        }
        other => Err(FrontError::BadRequest(format!(
            "unknown query kind {other:?} (expected one of: id, status, kind, cross_ref, text)"
        ))),
    }
}

/// A required argument that must be *present* (the wire key exists). An empty value is allowed
/// through — the engine's own never-silent rule handles "empty id / empty text matched nothing".
fn present<'a>(v: Option<&'a str>, name: &str) -> Result<&'a str, FrontError> {
    v.ok_or_else(|| FrontError::BadRequest(format!("missing required argument `{name}`")))
}

/// Run `query` through the M-1016 engine over `report` and render the outcome as the [`View`]'s JSON
/// envelope. An `Ok(Answer)` becomes an answer/cite/explain envelope; an `Err(Refusal)` becomes the
/// refusal envelope — both are successful (`200`) outcomes.
pub(crate) fn run_and_envelope(report: &TeroIndexReport, query: &Query, view: View) -> Value {
    match QueryEngine::new(report).run(query) {
        Ok(answer) => answer_envelope(&answer, view),
        Err(refusal) => refusal_envelope(&refusal),
    }
}

/// Render an [`Answer`] as the [`View`]'s JSON envelope (deterministic field order).
fn answer_envelope(answer: &Answer, view: View) -> Value {
    match view {
        View::Full => {
            #[derive(Serialize)]
            struct Full<'a> {
                kind: &'static str,
                items: &'a [TeroIndexItem],
                citations: Vec<Citation>,
                explain: &'a Explain,
            }
            to_value_infallible(&Full {
                kind: "answer",
                items: answer.items(),
                citations: answer.citations(),
                explain: answer.explain(),
            })
        }
        View::Cite => json!({ "kind": "citations", "citations": answer.citations() }),
        View::Explain => json!({ "kind": "explain", "explain": answer.explain() }),
    }
}

/// Render a [`Refusal`] as the refusal envelope: the structured (internally-tagged) variant plus its
/// human-readable [`Refusal`] `Display` message — a refusal is itself EXPLAIN-able (DN-87 §6.2).
fn refusal_envelope(refusal: &Refusal) -> Value {
    json!({ "kind": "refusal", "refusal": refusal, "message": refusal.to_string() })
}

/// The `identify` payload — the fronts' capability/version handshake. Uses [`crate::crate_summary`]
/// and surfaces `layer2_enabled` (the M-1018 gate; `false` until the eval gate opens — DN-87 §6.1)
/// and the sibling indices, so a client learns the whole memory surface in one call.
pub(crate) fn identify_value(layer2_enabled: bool) -> Value {
    json!({
        "name": "mycelium-tero",
        "summary": crate::crate_summary(),
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "M-1016 QueryEngine over the Layer-1 tero-index (docs/tero-index/index.json)",
        "layer2_enabled": layer2_enabled,
        "operations": [
            "identify", "query_by_id", "query_by_status", "query_by_kind",
            "cross_ref", "text_search", "cite", "explain", "refresh",
        ],
        "siblings": SIBLING_INDICES,
    })
}

/// Serialize a value that is statically known to be infallible to serialize (plain data — no
/// non-string map keys, no custom `Serialize` that can error). A failure here is a programming
/// invariant violation, not a runtime condition, so it is surfaced loudly rather than smuggled into
/// the response.
fn to_value_infallible<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("tero front envelope types serialize infallibly")
}
