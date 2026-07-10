//! White-box tests for the framework-agnostic front core (M-1017): the request→[`Query`] parse, the
//! JSON envelope shapes both fronts emit, the identify payload, and the [`FrontError`] mappings.
//! These are the parity-by-construction guarantees — one serializer, exercised directly.

use crate::front::auth::Scope;
use crate::front::core::{
    identify_value, parse_query, required_scope, run_and_envelope, FrontError, View,
};
use crate::query::Query;
use crate::tests::fixture::corpus_report;

#[test]
fn parse_query_maps_every_kind() {
    assert_eq!(
        parse_query("id", Some("M-1015"), None, None).unwrap(),
        Query::Id("M-1015".into())
    );
    assert_eq!(
        parse_query("status", Some("todo"), None, None).unwrap(),
        Query::Status("todo".into())
    );
    assert_eq!(
        parse_query("kind", Some("rfc"), None, None).unwrap(),
        Query::Kind("rfc".into())
    );
    assert_eq!(
        parse_query("text", Some("eval gate"), None, None).unwrap(),
        Query::Text("eval gate".into())
    );
    assert_eq!(
        parse_query("cross_ref", None, Some("M-1017"), Some("2")).unwrap(),
        Query::CrossRef {
            start: "M-1017".into(),
            depth: 2
        }
    );
}

#[test]
fn cross_ref_depth_defaults_to_one_when_omitted() {
    assert_eq!(
        parse_query("cross_ref", None, Some("M-1017"), None).unwrap(),
        Query::CrossRef {
            start: "M-1017".into(),
            depth: 1
        }
    );
}

#[test]
fn parse_query_rejects_unknown_kind_missing_arg_and_bad_depth() {
    // Unknown kind, missing required arg, and an unparseable depth are all 400s, never a guess.
    assert!(matches!(
        parse_query("bogus", Some("x"), None, None),
        Err(FrontError::BadRequest(_))
    ));
    assert!(matches!(
        parse_query("id", None, None, None),
        Err(FrontError::BadRequest(_))
    ));
    assert!(matches!(
        parse_query("cross_ref", None, None, None), // missing `start`
        Err(FrontError::BadRequest(_))
    ));
    assert!(matches!(
        parse_query("cross_ref", None, Some("M-1017"), Some("deep")),
        Err(FrontError::BadRequest(_))
    ));
}

#[test]
fn answer_envelope_carries_items_citations_and_explain() {
    let (_root, report) = corpus_report("core-answer");
    let env = run_and_envelope(&report, &Query::Id("M-0099".into()), View::Full);
    assert_eq!(env["kind"], "answer");
    assert!(env["items"].as_array().is_some_and(|a| !a.is_empty()));
    assert!(env["citations"].as_array().is_some_and(|a| !a.is_empty()));
    assert!(env["explain"].is_object());
    // Provenance by construction: the first citation resolves to a real anchor + file:line.
    let cite = &env["citations"][0];
    assert!(cite["anchor"].is_string());
    assert!(cite["file"].is_string());
    assert!(cite["line"].is_number());
    assert!(cite["item_tag"].is_string());
}

#[test]
fn a_query_that_matches_nothing_is_a_refusal_envelope_not_an_empty_answer() {
    let (_root, report) = corpus_report("core-refuse");
    let env = run_and_envelope(&report, &Query::Id("NO-SUCH-ID".into()), View::Full);
    assert_eq!(env["kind"], "refusal");
    assert_eq!(env["refusal"]["variant"], "no_match");
    assert!(env["message"]
        .as_str()
        .is_some_and(|m| m.contains("refusing")));
}

#[test]
fn cite_and_explain_views_project_the_answer() {
    let (_root, report) = corpus_report("core-views");
    let q = Query::Id("M-0099".into());
    let cite = run_and_envelope(&report, &q, View::Cite);
    assert_eq!(cite["kind"], "citations");
    assert!(cite["citations"].as_array().is_some_and(|a| !a.is_empty()));
    assert!(cite.get("items").is_none()); // citations-only projection

    let explain = run_and_envelope(&report, &q, View::Explain);
    assert_eq!(explain["kind"], "explain");
    assert!(explain["explain"]["order_by"].as_array().is_some());
    assert!(explain.get("items").is_none());
}

#[test]
fn identify_reports_the_engine_and_the_gated_layer2() {
    let v = identify_value(false);
    assert_eq!(v["name"], "mycelium-tero");
    assert_eq!(v["layer2_enabled"], false);
    assert!(v["summary"]
        .as_str()
        .is_some_and(|s| s.contains("mycelium-tero")));
    assert!(v["operations"].as_array().is_some_and(|a| a.len() == 9));
    assert!(v["siblings"].as_array().is_some_and(|a| !a.is_empty()));
}

#[test]
fn required_scope_is_read_by_default_and_refresh_for_refresh() {
    assert_eq!(required_scope("refresh"), Scope::Refresh);
    for op in ["identify", "query_by_id", "cite", "explain", "text_search"] {
        assert_eq!(required_scope(op), Scope::Read, "op {op} must be read-only");
    }
}

#[test]
fn front_error_maps_to_http_status_jsonrpc_code_and_slug() {
    let cases = [
        (
            FrontError::BadRequest("x".into()),
            400,
            -32602,
            "bad_request",
        ),
        (
            FrontError::Unauthorized("x".into()),
            401,
            -32001,
            "unauthorized",
        ),
        (FrontError::Forbidden("x".into()), 403, -32002, "forbidden"),
        (FrontError::NotFound("x".into()), 404, -32601, "not_found"),
        (FrontError::Internal("x".into()), 500, -32603, "internal"),
    ];
    for (e, http, rpc, slug) in cases {
        assert_eq!(e.http_status(), http);
        assert_eq!(e.jsonrpc_code(), rpc);
        assert_eq!(e.slug(), slug);
        assert_eq!(e.to_json()["error"]["code"], slug);
    }
}
