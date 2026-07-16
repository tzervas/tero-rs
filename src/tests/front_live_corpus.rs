//! Skip-graceful **live-corpus front e2e** (companion to `families.rs`'s live-repo cross-checks and
//! `query_latency.rs`'s live-corpus timing): every other front test (`front_http`/`front_mcp`/
//! `front_parity`/`front_smoke`) drives the HTTP/MCP fronts over the *hermetic fixture* — a known,
//! synthetic mini-corpus whose cited files are written into a temp dir by the test itself, so a
//! citation's `file` trivially "exists" by construction. None of them prove that a front, loaded
//! with the **real, committed production index**, returns citations that resolve to real files at
//! their real repo-relative paths on disk. This closes that gap: mock transport (`oneshot` / a
//! `Cursor`-driven MCP session — no socket), the real `docs/tero-index/index.json` as the fixture,
//! and a parameterized case table (one `kind`-query per corpus family actually present), asserting
//! HTTP/MCP/engine parity *and* on-disk provenance resolution, in one loop.
//!
//! Deliberately queries by [`Query::Kind`], not [`Query::Id`]: `id` is only populated for rows whose
//! source declares one (an `RFC-*`/`ADR-*`/`M-*` heading, an `issues.yaml` entry, …), which this
//! extracted `tero-rs` repo's own corpus — plain sections + a changelog, no `issues.yaml`/skills —
//! does not carry; asserting on `id` here would either be vacuous (skipped) or brittle to a corpus
//! this repo doesn't have. `kind` is a required field on every row, so it is the query hook that is
//! guaranteed to select real, present data regardless of which families a given checkout carries.
//!
//! Skip-graceful (mirrors `families.rs`/`query_latency.rs`): a checkout without the committed index
//! (a stripped fixture repo) yields no assertion, not a failure.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header::AUTHORIZATION, Request};
use serde_json::{json, Value};
use tower::ServiceExt; // oneshot

use crate::front::auth::TokenTable;
use crate::front::core::{self, View};
use crate::front::http::{router, AppState};
use crate::front::mcp::{serve, McpState};
use crate::load::load_report;
use crate::model::{Family, TeroIndexReport};
use crate::query::Query;

/// The repo root, two levels above this crate's manifest dir (mirrors `families.rs::repo_root`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

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

fn mcp_env(state: &mut McpState, kind: &str) -> Value {
    let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": "query_by_kind", "arguments": { "value": kind, "token": "reader" } } });
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

/// One representative `(family, kind)` pair per family actually present in `report` — the
/// parameterized case table. `kind` is a required field on every row (unlike `id`), so this always
/// yields a real, queryable case for every family the checkout's corpus actually carries; a family
/// absent from this particular corpus is simply absent from the table, never faked.
fn one_kind_per_family(report: &TeroIndexReport) -> Vec<(Family, String)> {
    let families = [
        Family::Doc,
        Family::Research,
        Family::Issue,
        Family::Changelog,
        Family::Skill,
    ];
    families
        .into_iter()
        .filter_map(|fam| {
            report
                .items
                .iter()
                .find(|it| it.family == fam)
                .map(|it| (fam, it.kind.clone()))
        })
        .collect()
}

#[tokio::test]
async fn fronts_over_the_real_committed_corpus_agree_and_resolve_to_real_files() {
    let index_path = repo_root().join("docs/tero-index/index.json");
    if !index_path.exists() {
        return; // skip-graceful — a stripped checkout without the committed Layer-1 index
    }
    let report = load_report(&index_path).expect("load the committed docs/tero-index/index.json");
    let cases = one_kind_per_family(&report);
    assert!(
        !cases.is_empty(),
        "the real committed index must yield at least one family to exercise"
    );

    let toks = || TokenTable::parse("reader:read").unwrap();
    let http = Arc::new(AppState::new(
        report.clone(),
        toks(),
        false,
        index_path.clone(),
    ));
    let mut mcp = McpState::new(report.clone(), toks(), false, index_path.clone());

    for (family, kind) in cases {
        let uri = format!("/v1/query?kind=kind&value={kind}");
        let engine = core::run_and_envelope(&report, &Query::Kind(kind.clone()), View::Full);
        let http_v = http_env(&http, &uri).await;
        let mcp_v = mcp_env(&mut mcp, &kind);

        assert_eq!(
            engine, http_v,
            "HTTP diverged from the engine for {family:?} kind {kind}"
        );
        assert_eq!(
            engine, mcp_v,
            "MCP diverged from the engine for {family:?} kind {kind}"
        );
        assert_eq!(
            engine["kind"], "answer",
            "{family:?} kind {kind} must be a cited answer"
        );

        let citations = engine["citations"]
            .as_array()
            .unwrap_or_else(|| panic!("{family:?} kind {kind} answer carries no citations array"));
        assert!(
            !citations.is_empty(),
            "{family:?} kind {kind} answer must carry >=1 citation"
        );
        for cite in citations {
            let file = cite["file"].as_str().expect("citation.file is a string");
            let line = cite["line"].as_u64().expect("citation.line is a number");
            let on_disk = repo_root().join(file);
            assert!(
                on_disk.is_file(),
                "{family:?} kind {kind} cites {file}:{line}, which must exist on disk at {}",
                on_disk.display()
            );
            assert!(
                line > 0,
                "{family:?} kind {kind} citation line must be 1-based, got {line}"
            );
        }
    }
}
