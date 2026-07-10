// Tests for `stream_parse` / `run_stream_parse` (M-820 / DN-57).
//
// Coverage:
//  - A well-formed multi-component stream parses to N clean components (data-driven).
//  - A malformed (lexically-valid, syntactically-invalid) component yields an explicit, located
//    `myc-stream-parse` error and the stream CONTINUES (good components after it still parse).
//  - A lexically-invalid stream yields an explicit OUTER `myc-stream-lex` error (whole-stream lex).
//  - EOF mid-component (unterminated last component) yields an explicit error.
//  - An empty stream (no `;`-terminated components) yields an explicit error.
//  - A single-component stream (trivial case) parses cleanly.
//  - COMMENT-SAFETY: a `//` comment containing the words `nodule` and `;` must NOT cause a
//    mis-split (the token-driven splitter ignores comments — DN-57 §2).
//
// Guarantee tags:
//  - split correctness: `Empirical` — the splitter is token-driven (lex once, segment on
//    `Tok::Nodule`/`Tok::Semi`), so it is comment-/string-safe *by construction* and that safety is
//    proven by `comment_with_nodule_and_semicolon_does_not_mis_split` below.
//  - parse correctness: `Empirical` (backed by the same `mycelium-l1::parse` oracle the tests exercise).
//  - never-silent errors: `Empirical` (every error path — parse, eof, empty — asserted below).
//  - I/O error: `Declared` (the `myc-stream-io` path is not exercised here — I/O fault injection
//    would require a mock reader; left for integration-level tests).
use crate::{run_stream_parse, stream_parse};

// --- data-driven corpus -----------------------------------------------------------------------

/// Each entry: (label, input, expected_ok_count, expected_err_count).
/// All inputs are valid-terminated or explicitly broken.
const WELL_FORMED_CORPUS: &[(&str, &str, usize, usize)] = &[
    (
        "single nodule",
        "nodule a; fn f() => Binary{8} = 0b0000_0000;",
        1,
        0,
    ),
    (
        "two nodules",
        "nodule a; fn f() => Binary{8} = 0b0000_0000;\nnodule b; fn g() => Binary{8} = 0b0000_0001;",
        2,
        0,
    ),
    (
        "three nodules — whitespace-free",
        "nodule a;nodule b;nodule c;",
        3,
        0,
    ),
];

/// Each entry: (label, input, expected_ok_count, expected_err_count, error_code).
///
/// IMPORTANT: these components are **lexically valid** but **syntactically invalid**, so the
/// whole-stream lex succeeds and the failure is isolated to one component's `parse` — which is what
/// proves *per-component* error continuation. (A lexically-invalid character like `§` aborts the
/// single whole-stream lex; that distinct path is covered by `lex_error_is_an_explicit_outer_error`
/// below.)
const MALFORMED_CORPUS: &[(&str, &str, usize, usize, &str)] = &[
    (
        "one malformed component in a two-component stream",
        // First component lexes but does not parse (empty body after `=`); second is fine.
        "nodule bad; fn f() = ;\nnodule good; fn g() => Binary{8} = 0b1111_1111;",
        1,
        1,
        "myc-stream-parse",
    ),
    (
        "first component malformed, rest clean",
        // `fn ;` lexes (keyword + terminator) but does not parse; second nodule is header-only OK.
        "nodule x; fn ;\nnodule y;",
        1,
        1,
        "myc-stream-parse",
    ),
];

// --- well-formed tests ------------------------------------------------------------------------

#[test]
fn well_formed_multi_component_stream_parses_all() {
    for (label, input, expected_ok, expected_err) in WELL_FORMED_CORPUS {
        let report = run_stream_parse(input.as_bytes(), "<test>")
            .unwrap_or_else(|e| panic!("[{label}] run_stream_parse failed fatally: {e}"));
        assert_eq!(
            report.parsed_ok, *expected_ok,
            "[{label}] expected {expected_ok} ok components, got {}",
            report.parsed_ok
        );
        assert_eq!(
            report.parsed_err, *expected_err,
            "[{label}] expected {expected_err} failed components, got {}; failures: {:?}",
            report.parsed_err, report.failures
        );
        assert!(
            report.ok(),
            "[{label}] report should be all-ok but has failures: {:?}",
            report.failures
        );
    }
}

// --- comment-safety test (the spec-fidelity fix: token-driven, not raw-text, splitting) -------

#[test]
fn comment_with_nodule_and_semicolon_does_not_mis_split() {
    // A `//` comment that literally contains the word `nodule` AND a `;` semicolon. A raw-text
    // keyword/`;` scanner would mis-split here (treating the comment's `nodule` as a new component
    // boundary and the comment's `;` as a terminator). The token-driven splitter lexes first, so
    // comments are never in the token stream and CANNOT cause a mis-split (DN-57 §2).
    //
    // This is ONE nodule with ONE item; correct behavior is exactly 1 clean component.
    let input = "nodule a; \
                 // this comment mentions a nodule and ends with a ; here\n\
                 fn f() => Binary{8} = 0b0000_0000;";
    let report = run_stream_parse(input.as_bytes(), "<comment-safety>")
        .expect("comment-bearing stream must not fail fatally");
    assert_eq!(
        report.parsed_ok, 1,
        "a `//` comment containing `nodule`/`;` must NOT create extra components; \
         expected exactly 1 clean component, got {} ok / {} err (failures: {:?})",
        report.parsed_ok, report.parsed_err, report.failures
    );
    assert_eq!(
        report.parsed_err, 0,
        "comment-bearing single nodule must have no errors; failures: {:?}",
        report.failures
    );
    assert!(report.ok());
}

#[test]
fn leading_comment_block_before_nodule_header_does_not_mis_split() {
    // A multi-line leading comment block (as in docs/spec/grammar/conformance/accept/*.myc) that
    // mentions `nodule` before the real header. The token-driven splitter must see exactly one
    // component. A raw-text scanner would have split at the comment's `nodule`.
    let input = "// exercises: a leading comment block that says the word nodule; and has a ;\n\
                 // a second comment line also naming nodule and a ; terminator\n\
                 nodule doc; fn id[T](x: T) => T = x;";
    let report = run_stream_parse(input.as_bytes(), "<leading-comment>")
        .expect("leading-comment stream must not fail fatally");
    assert_eq!(
        report.parsed_ok, 1,
        "leading comment naming `nodule`/`;` must not create extra components; \
         got {} ok / {} err (failures: {:?})",
        report.parsed_ok, report.parsed_err, report.failures
    );
    assert_eq!(report.parsed_err, 0, "failures: {:?}", report.failures);
}

// --- malformed-component tests ----------------------------------------------------------------

#[test]
fn malformed_component_yields_explicit_error_stream_continues() {
    for (label, input, expected_ok, expected_err, expected_code) in MALFORMED_CORPUS {
        let report = run_stream_parse(input.as_bytes(), "<test>")
            .unwrap_or_else(|e| panic!("[{label}] run_stream_parse failed fatally: {e}"));
        assert_eq!(
            report.parsed_ok, *expected_ok,
            "[{label}] expected {expected_ok} ok, got {}",
            report.parsed_ok
        );
        assert_eq!(
            report.parsed_err, *expected_err,
            "[{label}] expected {expected_err} errors, got {}",
            report.parsed_err
        );
        assert!(
            !report.failures.is_empty(),
            "[{label}] expected at least one failure"
        );
        let first = &report.failures[0];
        assert_eq!(
            first.code, *expected_code,
            "[{label}] expected error code {expected_code}, got {}",
            first.code
        );
        // Must carry a location (component:line:col) — never opaque (G2 / DN-22).
        assert!(
            first.location.is_some(),
            "[{label}] error must carry a location"
        );
        // Must carry a help line (DN-22 actionable).
        assert!(
            first.help.is_some(),
            "[{label}] error must carry a help line"
        );
    }
}

// --- EOF mid-component test -------------------------------------------------------------------

#[test]
fn eof_mid_component_is_an_explicit_error_not_silent() {
    // A stream that has content after its last `;` (unterminated component).
    let input = "nodule a;\nnodule b"; // `nodule b` has no `;` terminator — EOF arrives mid-component.
    let report = run_stream_parse(input.as_bytes(), "<test-eof>")
        .expect("run_stream_parse must not fatally fail (the first component is fine)");

    // The first component (nodule a;) parsed clean; the second (nodule b) is unterminated.
    assert_eq!(report.parsed_ok, 1, "first component should parse ok");
    assert_eq!(
        report.parsed_err, 1,
        "unterminated component should be one error"
    );
    let err = &report.failures[0];
    assert_eq!(err.code, "myc-stream-eof");
    assert!(
        err.help.is_some(),
        "eof error must carry a help line (DN-22)"
    );
    assert!(
        err.message.contains("unterminated"),
        "eof message must mention 'unterminated': {}",
        err.message
    );
}

// --- empty stream test ------------------------------------------------------------------------

#[test]
fn empty_stream_is_an_explicit_error_not_silent() {
    // An entirely empty stream has no components at all — must not silently succeed.
    let result = run_stream_parse("".as_bytes(), "<test-empty>");
    let err = result.expect_err("empty stream must return Err");
    assert_eq!(err.code, "myc-stream-empty");
    assert!(err.help.is_some(), "empty-stream error must carry help");
}

// --- lex-error test (whole-stream lex; never-silent outer error) ------------------------------

#[test]
fn lex_error_is_an_explicit_outer_error() {
    // A lexically-invalid character (`§`) cannot be tokenized. Because the stream is lexed once
    // (the spec-mandated token-driven split), this surfaces as an explicit OUTER `myc-stream-lex`
    // error with a position — never a silent skip/truncation (G2). This is distinct from the
    // per-component `myc-stream-parse` path (a lexically-valid component that does not parse).
    let input = "nodule a; fn f() = §;";
    let err = run_stream_parse(input.as_bytes(), "<lex-fail>")
        .expect_err("a lexically-invalid stream must return an outer Err");
    assert_eq!(err.code, "myc-stream-lex");
    assert!(err.location.is_some(), "lex error must carry a position");
    assert!(err.help.is_some(), "lex error must carry a help line");
}

// --- single-component stream ------------------------------------------------------------------

#[test]
fn single_component_stream_parses_cleanly() {
    let input = "nodule solo; fn answer() => Binary{8} = 0b0010_1010;";
    let report = run_stream_parse(input.as_bytes(), "<single>")
        .expect("single-component stream must not fail fatally");
    assert_eq!(report.parsed_ok, 1);
    assert_eq!(report.parsed_err, 0);
    assert!(report.ok());
}

// --- source_name propagation ------------------------------------------------------------------

#[test]
fn source_name_is_propagated_in_stream_report() {
    let input = "nodule x;";
    let report = run_stream_parse(input.as_bytes(), "myfile.myc").expect("stream parse ok");
    assert_eq!(report.source_name, "myfile.myc");
}

// --- stream_parse raw API lower-level check ---------------------------------------------------

#[test]
fn stream_parse_returns_per_component_results() {
    // Raw `stream_parse` returns a Vec<StreamComponent> — one entry per component.
    let input = "nodule a;nodule b;nodule c;";
    let components =
        stream_parse(input.as_bytes(), "<raw>").expect("stream_parse must not fail fatally");
    assert_eq!(components.len(), 3, "expected 3 components");
    for (i, comp) in components.iter().enumerate() {
        assert!(
            comp.is_ok(),
            "component {} should parse ok, got: {:?}",
            i + 1,
            comp
        );
    }
}
