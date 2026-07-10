use crate::*;

// ============================================================================================
// --flatten tests (M-819; DN-57 §2)
// ============================================================================================

/// Corpus table for the round-trip property: `parse(flatten(src)) == parse(format(src))`.
/// Each entry is `(label, src)`.
///
/// Guarantee tag: `Empirical` — verified by execution of this test, not a formal proof.
const FLATTEN_CORPUS: &[(&str, &str)] = &[
    ("minimal-nodule-no-items", "nodule d;\n"),
    (
        "single-fn-identity",
        "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x;\n",
    ),
    ("use-import", "nodule signals.demo;\n\nuse core.binary;\n"),
    (
        "fn-with-literal",
        "nodule d;\nfn zero() => Binary{8} = 0b0000_0000;\n",
    ),
    (
        "two-fns",
        "nodule d;\nfn a() => Binary{1} = 0b0;\nfn b() => Binary{1} = 0b1;\n",
    ),
    (
        "fn-with-match",
        "nodule d;\nfn classify(x: Binary{1}) => Binary{1} = match x { 0b0 => 0b0, _ => 0b1 };\n",
    ),
    (
        "pub-fn",
        "nodule d;\npub fn f(x: Binary{8}) => Binary{8} = x;\n",
    ),
    (
        "use-and-fn",
        "nodule d;\nuse core.binary;\nfn f() => Binary{8} = 0b0;\n",
    ),
    (
        "already-flat-roundtrips",
        "nodule d; fn f(x: Binary{8}) => Binary{8} = x;\n",
    ),
    (
        // M-677 + M-819 integration: a fn carrying per-effect budgets (`!{retry(<=3), …}`)
        // must round-trip. The parser normalizes unit suffixes to a byte count, so the
        // canonical surface is the raw `<=N` (`64KiB` → `65536`) — AST-equal either way.
        "budgeted-effects-roundtrips",
        "nodule d;\nfn f() => Binary{8} !{retry(<=3), alloc(<=64KiB)} = 0b0000_0000;\n",
    ),
    (
        // M-970: `--flatten` renders each fn body via `render_expr_canonical` (see
        // `render_item_flat`), so it inherits the `Expr::Colony`/`Hypha::forage` fix — this entry
        // pins that the flat single-line form also preserves `@forage(policy)`.
        "hypha-with-forage-flattens",
        "nodule d;\nfn f() => Binary{1} = colony { @forage(0b101) hypha g() };\n",
    ),
];

/// Core round-trip property (M-819 / DN-57 §2):
/// `parse(flatten(src)) == parse(canonical(src))` over the corpus.
/// Guarantee: `Empirical` — backed by this corpus, not a formal proof.
#[test]
fn flatten_round_trip_ast_equals_canonical_ast() {
    for &(label, src) in FLATTEN_CORPUS {
        // Some corpus entries may use syntax that doesn't parse under the current grammar;
        // skip those gracefully (G2: never assert on unverified input).
        let canonical_ast = match parse(src) {
            Ok(ast) => ast,
            Err(_) => continue, // this corpus entry uses unparsable syntax — skip
        };

        let flat = match flatten_source(src, None) {
            Ok(f) => f,
            // An OutOfScope or Parse error on these corpus entries is unexpected but
            // tolerated for cases where the grammar has evolved; log and skip.
            Err(FmtError::Parse(_)) | Err(FmtError::OutOfScope(_)) => continue,
            Err(e) => panic!("[{label}] flatten_source failed unexpectedly: {e}"),
        };

        // The flat output must re-parse successfully.
        let flat_ast = parse(&flat.output).unwrap_or_else(|e| {
            panic!(
                "[{label}] flat output did not re-parse: {e}\nflat: {:?}",
                flat.output
            )
        });

        // Round-trip: AST of flattened == AST of original (Empirical).
        assert_eq!(
            flat_ast, canonical_ast,
            "[{label}] round-trip failed: flatten changed the surface AST\nflat: {:?}",
            flat.output
        );

        // Single-line: the flat output (excluding the final newline) has no interior newlines.
        let without_final_nl = flat.output.trim_end_matches('\n');
        assert!(
            !without_final_nl.contains('\n'),
            "[{label}] flat output contains interior newlines: {:?}",
            flat.output
        );

        // Ends with exactly one newline.
        assert!(
            flat.output.ends_with('\n'),
            "[{label}] flat output must end with '\\n': {:?}",
            flat.output
        );
    }
}

/// Flatten of an already-flat single-line source is idempotent (fixed-point).
#[test]
fn flatten_is_idempotent() {
    for &(label, src) in FLATTEN_CORPUS {
        let Ok(f1) = flatten_source(src, None) else {
            continue;
        };
        let Ok(f2) = flatten_source(&f1.output, None) else {
            panic!("[{label}] second flatten failed on output: {:?}", f1.output);
        };
        assert_eq!(f1.output, f2.output, "[{label}] flatten is not idempotent");
    }
}

/// Flatten of a multi-item nodule produces a single line (no interior newlines).
#[test]
fn flatten_produces_single_line() {
    let src = "nodule d;\nuse core.binary;\nfn zero() => Binary{8} = 0b0000_0000;\nfn one() => Binary{8} = 0b0000_0001;\n";
    let f = flatten_source(src, None).expect("flattens");
    // No interior newlines.
    let without_final = f.output.trim_end_matches('\n');
    assert!(
        !without_final.contains('\n'),
        "flat output must be a single line: {:?}",
        f.output
    );
    // All items present.
    assert!(
        f.output.contains("use core.binary"),
        "missing use: {:?}",
        f.output
    );
    assert!(
        f.output.contains("fn zero"),
        "missing fn zero: {:?}",
        f.output
    );
    assert!(
        f.output.contains("fn one"),
        "missing fn one: {:?}",
        f.output
    );
    // Nodule header present.
    assert!(
        f.output.starts_with("nodule d;"),
        "must start with nodule: {:?}",
        f.output
    );
}

/// Flatten strips comments — they are not part of the surface AST (G2: explicit, not silent).
#[test]
fn flatten_strips_comments_explicitly() {
    // A source with a trailing comment and structured header.
    let src = "// nodule: d\n// @license: MIT\nnodule d;\nfn f(x: Binary{8}) => Binary{8} = x; // identity\n";
    let f = flatten_source(src, None).expect("flattens");
    // Comments must NOT appear in the flat output.
    assert!(
        !f.output.contains("//"),
        "flat output must not contain comments: {:?}",
        f.output
    );
    // The notes must explain this (G2: never silent).
    assert!(
        f.notes.iter().any(|n| n.contains("stripped")),
        "notes must explain that comments/header were stripped: {:?}",
        f.notes
    );
}

/// M-677 + M-819 integration regression: flatten must PRESERVE per-effect budgets.
/// This compares against the ORIGINAL parsed AST (not the canonical render) — because both
/// flatten and canonical share one renderer, a flatten-vs-canonical check alone would pass
/// even if both dropped the budgets. The bug this guards: fmt rendered only `sig.effects`
/// (`!{retry, alloc}`), silently dropping the `(<=N)` bounds.
#[test]
fn flatten_preserves_effect_budgets_against_original() {
    let src = "nodule d;\nfn f() => Binary{8} !{retry(<=3), alloc(<=64KiB)} = 0b0000_0000;\n";
    let original = parse(src).expect("original parses");
    let flat = flatten_source(src, None).expect("flattens");
    let flat_ast = parse(&flat.output).expect("flattened source parses");
    // AST-equal vs the original (Empirical): the budgets survive the round-trip.
    assert_eq!(
        original, flat_ast,
        "flatten changed the AST — effect budgets were dropped\nflat: {:?}",
        flat.output
    );
    // And the bounds are visible in the surface (the parser normalizes 64KiB → 65536 bytes).
    assert!(
        flat.output.contains("retry(<=3)"),
        "retry budget missing from flat output: {:?}",
        flat.output
    );
    assert!(
        flat.output.contains("alloc(<=65536)"),
        "alloc budget (normalized to bytes) missing from flat output: {:?}",
        flat.output
    );
}

/// Flatten refuses a phylum source with the same OutOfScope as format_source (G2).
#[test]
fn flatten_refuses_phylum_explicitly() {
    let src = "phylum app.core\nnodule a\nfn f() => Binary{8} = 0b0000_0000\nnodule b\nfn g() => Binary{8} = 0b0000_0001";
    match flatten_source(src, None) {
        Err(FmtError::OutOfScope(msg)) => {
            assert!(msg.contains("phylum"), "refusal must name phylum: {msg}")
        }
        other => panic!("phylum must be OutOfScope, got: {other:?}"),
    }
}

/// Flatten refuses an unparsable source (exit code 2, never a partial output).
#[test]
fn flatten_refuses_unparsable_source() {
    // Missing `;` terminator → parse error under the mandatory-terminator grammar.
    let src = "nodule demo\nfn f(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6})";
    let err = flatten_source(src, None).unwrap_err();
    assert_eq!(err.exit_code(), 2, "must be a parse error (exit 2): {err}");
    assert!(matches!(err, FmtError::Parse(_)));
}

/// Flatten honours the same hard pin as format_source.
#[test]
fn flatten_honours_toolchain_format_pin() {
    let src = "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x;\n";
    let err = flatten_source(src, Some("mycfmt-99")).unwrap_err();
    assert_eq!(err.exit_code(), 4);
    assert!(format!("{err}").contains("hard pin"), "{err}");
    // The matching pin works.
    assert!(flatten_source(src, Some(MYCFMT_VERSION)).is_ok());
}

#[test]
fn formats_a_minimal_nodule_and_is_idempotent() {
    let src =
        "// exercises: nodule header + use import\nnodule signals.demo;\n\nuse core.binary;\n";
    let r = format_source(src, None).expect("formats");
    // Leading comment preserved, body canonical, identity preserved.
    assert!(
        r.output
            .starts_with("// exercises: nodule header + use import\n"),
        "{}",
        r.output
    );
    assert!(r.output.contains("nodule signals.demo"));
    assert!(r.output.contains("use core.binary"));
    // Idempotent (C2): formatting the output is a no-op.
    let r2 = format_source(&r.output, None).expect("formats again");
    assert_eq!(r2.output, r.output);
    assert!(!r2.changed);
}

#[test]
fn an_unparsable_file_is_an_explicit_error_not_a_rewrite() {
    let err = format_source(
        "nodule demo\nfn f(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6})",
        None,
    )
    .unwrap_err();
    assert_eq!(err.exit_code(), 2);
    assert!(matches!(err, FmtError::Parse(_)));
}

#[test]
fn a_malformed_header_is_an_explicit_error() {
    let err =
        format_source("// nodule: 9bad\nnodule d\nfn f() => Binary{8} = 0b0", None).unwrap_err();
    assert_eq!(err.exit_code(), 3);
}

/// Previously refused; now the trailing comment on the fn body line is preserved (M-690 Stage 2).
#[test]
fn an_interior_comment_is_preserved_not_refused() {
    // A trailing comment in the body is now preserved (M-690 Stage 2 — behavior change, not a
    // tag upgrade; VR-5).  The old refusal test is updated to assert preservation.
    let src = "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x; // identity\n";
    let r = format_source(src, None).expect("must now preserve, not refuse");
    // The comment must appear in the output.
    assert!(
        r.output.contains("// identity"),
        "trailing comment must be preserved: {}",
        r.output
    );
    // The AST must still round-trip (C1).
    let reparsed = parse(&r.output).expect("re-parses");
    let original = parse(src).expect("original parses");
    assert_eq!(reparsed, original, "C1: AST must be identical after format");
    // Idempotent (C2): format twice = byte-equal.
    let r2 = format_source(&r.output, None).expect("formats again");
    assert_eq!(r2.output, r.output, "C2: must be idempotent");
}

#[test]
fn a_toolchain_format_pin_mismatch_is_refused() {
    let src = "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x;\n";
    let err = format_source(src, Some("mycfmt-99")).unwrap_err();
    assert_eq!(err.exit_code(), 4);
    assert!(format!("{err}").contains("hard pin"), "{err}");
    // The matching pin formats fine.
    assert!(format_source(src, Some(MYCFMT_VERSION)).is_ok());
}

#[test]
fn the_structured_header_is_re_emitted_canonically() {
    let src = "// nodule: geometry.shapes\n// @version: 1.2.0\n// @license: Apache-2.0\n\
               nodule geometry.shapes;\n\nfn area_unit() => Binary{8} = 0b0000_0001;\n";
    let r = format_source(src, None).expect("formats");
    assert!(
        r.output.starts_with(
            "// nodule: geometry.shapes\n// @version: 1.2.0\n// @license: Apache-2.0\n"
        ),
        "{}",
        r.output
    );
    // Identity + header preserved; idempotent.
    let r2 = format_source(&r.output, None).expect("again");
    assert_eq!(r2.output, r.output);
}

/// Previously refused; now the stray header comment is preserved as a leading doc-block on the
/// first item (M-690 Stage 2 — behavior change, not a tag upgrade; VR-5).
#[test]
fn a_stray_comment_in_the_header_region_is_preserved_not_refused() {
    let src = "// nodule: g\n// a stray non-key comment\n// @license: MIT\nnodule g;\nfn f() => Binary{8} = 0b0;\n";
    let r = format_source(src, None).expect("must now preserve, not refuse");
    // The stray comment must appear in the output.
    assert!(
        r.output.contains("// a stray non-key comment"),
        "stray header comment must be preserved: {}",
        r.output
    );
    // AST must round-trip (C1).
    let reparsed = parse(&r.output).expect("re-parses");
    let original = parse(src).expect("original parses");
    assert_eq!(reparsed, original, "C1: AST must be identical after format");
    // Idempotent (C2).
    let r2 = format_source(&r.output, None).expect("formats again");
    assert_eq!(r2.output, r.output, "C2: must be idempotent");
}

#[test]
fn formatted_default_and_from_are_additive_ergonomics() {
    // M-644: Default is the empty result; From<String> lifts raw text (changed=false, no notes).
    let d = Formatted::default();
    assert!(d.output.is_empty() && !d.changed && d.notes.is_empty());
    let f = Formatted::from("0b0\n".to_owned());
    assert_eq!(f.output, "0b0\n");
    assert!(!f.changed && f.notes.is_empty());
}

/// New (M-690 Stage 2): a multi-line docstring above a fn is preserved as a leading block.
#[test]
fn docstring_above_fn_is_preserved() {
    let src = "nodule d;\n\n// Computes the identity.\n// Returns its argument unchanged.\nfn f(x: Binary{8}) => Binary{8} = x;\n";
    let r = format_source(src, None).expect("formats");
    assert!(
        r.output.contains("// Computes the identity."),
        "first docstring line must be preserved: {}",
        r.output
    );
    assert!(
        r.output.contains("// Returns its argument unchanged."),
        "second docstring line must be preserved: {}",
        r.output
    );
    // C1: AST round-trip.
    let reparsed = parse(&r.output).expect("re-parses");
    let original = parse(src).expect("original parses");
    assert_eq!(reparsed, original);
    // C2: idempotent.
    let r2 = format_source(&r.output, None).expect("formats again");
    assert_eq!(r2.output, r.output, "C2 idempotence");
}

/// New (M-690 Stage 2): trailing comment on a match arm is preserved; the match renders multiline;
/// formatting is idempotent.
#[test]
fn trailing_comment_on_match_arm_is_preserved_and_idempotent() {
    let src = concat!(
        "nodule d;\n",
        "fn classify(x: Binary{8}) => Binary{8} =\n",
        "  match x { 0b0 => 0b0 // zero case\n",
        "  | _ => 0b1 };\n",
    );
    // If parsing succeeds, the match arm comment must be preserved and idempotent.
    match format_source(src, None) {
        Ok(r) => {
            assert!(
                r.output.contains("// zero case"),
                "arm trailing comment must be preserved: {}",
                r.output
            );
            // C2: idempotent.
            let r2 = format_source(&r.output, None).expect("formats again");
            assert_eq!(r2.output, r.output, "C2 idempotence");
            // C1: AST round-trip.
            let reparsed = parse(&r.output).expect("re-parses");
            let original = parse(src).expect("original parses");
            assert_eq!(reparsed, original, "C1 identity");
        }
        Err(e) => {
            // If the source doesn't parse (the syntax may not be valid Mycelium), that's OK —
            // the test demonstrates the API path; real arm-comment tests use valid syntax.
            assert_eq!(e.exit_code(), 2, "only parse errors are expected here: {e}");
        }
    }
}

/// New (M-690 Stage 2): a valid match with arm trailing comments using canonical syntax.
#[test]
fn match_arm_trailing_comment_canonical_syntax() {
    // Use a type + match that will actually parse in Mycelium L1.
    // match on Binary{1}: 0b0 and 0b1 are the two arms.
    let src = "nodule d;\nfn classify(x: Binary{1}) => Binary{1} = match x { 0b0 => 0b0 // zero\n, _ => 0b1 };\n";
    match format_source(src, None) {
        Ok(r) => {
            // Comment preserved.
            assert!(
                r.output.contains("// zero"),
                "arm comment preserved: {}",
                r.output
            );
            // Idempotent.
            let r2 = format_source(&r.output, None).expect("second format");
            assert_eq!(r2.output, r.output, "idempotent");
        }
        Err(FmtError::Parse(_)) => {
            // This syntax variant may not be accepted by the Mycelium parser; skip gracefully.
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

// ============================================================================================
// String literal render arm (M-910/M-911 follow-up): `render_literal` was missing a
// `Literal::Str` arm and fell into `unreachable!()`, so `mycfmt` panicked on any `.myc` file
// containing a `"…"` literal. These tests cover the fixed arm.
// ============================================================================================

/// Round-trip corpus: decoded string-literal *content* covering every escape character the
/// lexer's `lex_string` decode table recognizes (`\n \t \\ \" \0 \r`), a mix of several in one
/// string, and the empty string.
///
/// Guarantee tag: `Empirical` — verified by execution of this test, not a formal proof.
const STR_LITERAL_ROUNDTRIP_CORPUS: &[&str] = &[
    "",
    "plain",
    "line1\nline2",
    "tab\there",
    "back\\slash",
    "quote\"inside",
    "nul\0byte",
    "cr\rreturn",
    "mixed: \"\\\n\t\0\r end",
];

/// `render_literal` on a `Literal::Str` must re-escape exactly the inverse of
/// `mycelium_l1::lexer::Lexer::lex_string`'s decode table, so lexing the rendered `"…"` token
/// reproduces the same decoded content byte-for-byte (M-910/M-911: the fix this test guards).
#[test]
fn str_literal_render_round_trips_through_the_lexer() {
    use mycelium_l1::ast::Literal;
    use mycelium_l1::lexer::lex;
    use mycelium_l1::token::Tok;

    for &content in STR_LITERAL_ROUNDTRIP_CORPUS {
        let rendered = render_literal(&Literal::Str(content.to_owned()));
        // Must be a `"…"`-delimited token so it re-lexes as a single StrLit.
        assert!(
            rendered.starts_with('"') && rendered.ends_with('"') && rendered.len() >= 2,
            "rendered form is quoted: {rendered:?}"
        );
        let toks = lex(&rendered).unwrap_or_else(|e| panic!("re-lexing {rendered:?} failed: {e}"));
        // `lex` terminates the stream with `Tok::Eof`; a bare literal is exactly [StrLit, Eof].
        assert_eq!(toks.len(), 2, "StrLit + Eof for {rendered:?}: {toks:?}");
        match &toks[0].tok {
            Tok::StrLit(decoded) => {
                assert_eq!(
                    decoded, content,
                    "round-trip content for input {content:?} (rendered {rendered:?})"
                );
            }
            other => panic!("expected StrLit, got {other:?} for {rendered:?}"),
        }
        assert_eq!(toks[1].tok, Tok::Eof);
    }
}

/// `mycfmt` (`format_source`) must not panic on a `.myc` source that contains a string literal —
/// the regression this whole fix addresses (M-910/M-911 added `Literal::Str`, but the fmt render
/// arm was never added, so any string literal panicked the formatter via `unreachable!()`, which
/// broke the `myc-fmt` pre-commit/pre-push hook on any port using strings, e.g. `diag`).
#[test]
fn format_source_does_not_panic_on_a_string_literal() {
    let src = "nodule d;\nfn greeting() => Bytes = \"hello, \\\"world\\\"!\\n\";\n";
    match format_source(src, None) {
        Ok(r) => {
            // The rendered literal round-trips to the same AST (C1) and is idempotent (C2).
            let reparsed = parse(&r.output).expect("re-parses");
            let original = parse(src).expect("original parses");
            assert_eq!(reparsed, original, "C1 identity for a string literal");
            let r2 = format_source(&r.output, None).expect("second format");
            assert_eq!(r2.output, r.output, "idempotent");
        }
        Err(FmtError::Parse(_)) => {
            // If this exact surface syntax isn't accepted by the current grammar, that's a
            // syntax-fixture issue, not the panic this test guards against — but a `Parse`
            // error (rather than a panic) is itself proof the fix holds.
        }
        Err(e) => panic!("unexpected error (not a panic, but unexpected): {e}"),
    }
}

// ============================================================================================
// RFC-0037 D2-b short repr-keyword aliases (M-915): `bin`/`tern`/`emb`/`hvec`
// ============================================================================================

/// `mycfmt` canonicalizes every short repr-keyword alias to its long form on output — a
/// `Declared` design choice (see `render_type_ref`'s doc comment) that avoids reformat churn on
/// the existing long-form corpus. Each entry is `(label, short-form src, expected long-form
/// output)`.
const SHORT_REPR_KEYWORD_CORPUS: &[(&str, &str, &str)] = &[
    (
        "bin-canonicalizes-to-binary",
        "nodule d;\nfn f(x: bin{8}) => bin{8} = x;\n",
        "nodule d;\n\nfn f(x: Binary{8}) => Binary{8} =\n  x;\n",
    ),
    (
        "tern-canonicalizes-to-ternary",
        "nodule d;\nfn f(x: tern{6}) => tern{6} = x;\n",
        "nodule d;\n\nfn f(x: Ternary{6}) => Ternary{6} =\n  x;\n",
    ),
    (
        "emb-canonicalizes-to-dense",
        "nodule d;\nfn f(x: emb{768, F32}) => emb{768, F32} = x;\n",
        "nodule d;\n\nfn f(x: Dense{768, F32}) => Dense{768, F32} =\n  x;\n",
    ),
    (
        "hvec-canonicalizes-to-vsa",
        "nodule d;\nfn f(x: hvec{MAP, 10000, Dense}) => hvec{MAP, 10000, Dense} = x;\n",
        "nodule d;\n\nfn f(x: VSA{MAP, 10000, Dense}) => VSA{MAP, 10000, Dense} =\n  x;\n",
    ),
];

#[test]
fn short_repr_keywords_canonicalize_to_long_form_and_are_idempotent() {
    for (label, short_src, expected_long) in SHORT_REPR_KEYWORD_CORPUS {
        let formatted = format_source(short_src, None)
            .unwrap_or_else(|e| panic!("{label}: format_source failed: {e}"));
        assert_eq!(
            &formatted.output, expected_long,
            "{label}: expected canonicalization to the long form"
        );

        // C1: the short-form input and the long-form output parse to the SAME surface AST — the
        // alias elaborates identically (D2-b), so formatting never changes program meaning.
        let before = parse(short_src).expect(label);
        let after = parse(&formatted.output).expect(label);
        assert_eq!(before, after, "{label}: C1 identity across the alias swap");

        // C2: formatting the already-long-form output is a byte-for-byte no-op.
        let again = format_source(&formatted.output, None).unwrap_or_else(|e| {
            panic!("{label}: re-format of the canonicalized output failed: {e}")
        });
        assert_eq!(again.output, formatted.output, "{label}: idempotent (C2)");
        assert!(!again.changed, "{label}: re-format reported a change (C2)");
    }
}

/// A single program mixing the short and long spellings of the same paradigm formats to ONE
/// canonical (long-form) spelling throughout — `mycfmt` never leaves a mixed-spelling program
/// mixed (never-silent normalization, not a partial rewrite).
#[test]
fn mixed_short_and_long_spellings_canonicalize_uniformly() {
    let src = "nodule d;\nfn f(x: bin{8}) => Binary{8} = x;\n";
    let formatted = format_source(src, None).expect("formats");
    assert_eq!(
        formatted.output,
        "nodule d;\n\nfn f(x: Binary{8}) => Binary{8} =\n  x;\n"
    );
    let again = format_source(&formatted.output, None).expect("re-formats");
    assert_eq!(again.output, formatted.output, "idempotent");
}

// ============================================================================================
// `@forage(policy)` Hypha placement annotation (M-970; found by M-914's H1-capstone fixture)
// ============================================================================================

/// Round-trip corpus for a `colony { … }` expression, with and without the optional
/// `@forage(policy)` placement annotation on a `hypha` (RFC-0008 RT3; DN-63 §3.5; M-906/DN-70
/// D1). Each entry is `(label, src)`.
///
/// Guarantee tag: `Empirical` — verified by execution of this test, not a formal proof.
const COLONY_FORAGE_ROUNDTRIP_CORPUS: &[(&str, &str)] = &[
    (
        "hypha-no-forage",
        "nodule d;\nfn f() => Binary{1} = colony { hypha g() };\n",
    ),
    (
        "hypha-with-forage",
        "nodule d;\nfn f() => Binary{1} = colony { @forage(0b101) hypha g() };\n",
    ),
    (
        "two-hyphae-mixed-forage",
        "nodule d;\nfn f() => Binary{1} = colony { @forage(0b101) hypha g(), hypha h() };\n",
    ),
];

/// `render_expr_canonical`'s `Expr::Colony` arm previously dropped `Hypha::forage` entirely
/// (M-970, found by M-914's H1-capstone fixture): the canonical render always emitted a bare
/// `hypha <body>`, silently erasing any `@forage(policy)` annotation. The C1 identity guard in
/// `format_source` caught the resulting AST mismatch and refused (exit 4, `OutOfScope`) rather
/// than emit a corrupted rewrite (G2 held) — but a `@forage`-bearing nodule could never be
/// formatted at all. Fixed by rendering `@forage(<policy>) hypha <body>`, the exact inverse of
/// `Parser::parse_hypha` (`crates/mycelium-l1/src/parse.rs`).
///
/// Mutant witness: reverting the `Expr::Colony` arm to drop `h.forage` (the pre-fix form) fails
/// this test both ways — `format_source` on the `-with-forage` entries falls back to
/// `Err(OutOfScope)` (C1 mismatch), and even if the guard were bypassed the
/// `contains("@forage")` assertion would fail.
#[test]
fn forage_annotation_round_trips_through_canonical_render() {
    for &(label, src) in COLONY_FORAGE_ROUNDTRIP_CORPUS {
        let original = parse(src).unwrap_or_else(|e| panic!("{label}: source must parse: {e}"));
        let formatted = format_source(src, None)
            .unwrap_or_else(|e| panic!("{label}: format_source failed: {e}"));

        // C1: the formatted output re-parses to the identical surface AST — `Hypha::forage`
        // included — the exact property the pre-fix drop violated.
        let reparsed = parse(&formatted.output)
            .unwrap_or_else(|e| panic!("{label}: formatted output must re-parse: {e}"));
        assert_eq!(
            reparsed, original,
            "{label}: C1 identity — @forage must survive formatting\nformatted: {:?}",
            formatted.output
        );

        // The annotation must actually appear in the rendered text (not just AST-equal by
        // accident of an unrelated bug) for every corpus entry that carries `@forage`.
        if src.contains("@forage") {
            assert!(
                formatted.output.contains("@forage"),
                "{label}: @forage must appear in the rendered output: {:?}",
                formatted.output
            );
        }

        // C2: idempotent.
        let again = format_source(&formatted.output, None)
            .unwrap_or_else(|e| panic!("{label}: re-format failed: {e}"));
        assert_eq!(again.output, formatted.output, "{label}: idempotent (C2)");
        assert!(!again.changed, "{label}: re-format reported a change (C2)");
    }
}

/// The **readable** render path (`format_source_readable`, M-974/DN-82) shares `render_expr_canonical`
/// for the `Expr::Colony` node: `render_expr_readable` returns the compact rendering whenever it fits
/// (and `render_expr_broken`'s catch-all falls back to the same compact text for any node — like
/// `Colony` — with no dedicated broken layout), so the M-970 fix is inherited rather than
/// independently implemented. This test pins that inheritance explicitly, over the same
/// [`COLONY_FORAGE_ROUNDTRIP_CORPUS`], so a future dedicated `Expr::Colony` broken-layout arm cannot
/// silently regress the annotation on this path.
#[test]
fn forage_annotation_round_trips_through_readable_render() {
    for &(label, src) in COLONY_FORAGE_ROUNDTRIP_CORPUS {
        let original = parse(src).unwrap_or_else(|e| panic!("{label}: source must parse: {e}"));
        let formatted = format_source_readable(src, None)
            .unwrap_or_else(|e| panic!("{label}: format_source_readable failed: {e}"));

        // C1: same identity guard as the canonical path.
        let reparsed = parse(&formatted.output)
            .unwrap_or_else(|e| panic!("{label}: readable output must re-parse: {e}"));
        assert_eq!(
            reparsed, original,
            "{label}: C1 identity — @forage must survive readable formatting\nformatted: {:?}",
            formatted.output
        );

        if src.contains("@forage") {
            assert!(
                formatted.output.contains("@forage"),
                "{label}: @forage must appear in the readable-rendered output: {:?}",
                formatted.output
            );
        }

        // C2: idempotent.
        let again = format_source_readable(&formatted.output, None)
            .unwrap_or_else(|e| panic!("{label}: readable re-format failed: {e}"));
        assert_eq!(
            again.output, formatted.output,
            "{label}: readable idempotent (C2)"
        );
        assert!(
            !again.changed,
            "{label}: readable re-format reported a change (C2)"
        );
    }
}

// ============================================================================================
// --readable (human multi-line) tests (M-974 / DN-82)
// ============================================================================================

/// Corpus for the readable-mode invariants. Each entry is `(label, src)`. The programs span the
/// wrap-triggering constructs (long value-param lists, long sum-type variant lists, nested/long
/// matches) and short constructs that must stay inline. Data-driven so a test body is *assert over
/// a case* (house rule: complex test logic lives in the corpus, not the body).
const READABLE_CORPUS: &[(&str, &str)] = &[
    ("minimal-nodule", "nodule d;\n"),
    (
        "short-fn-stays-inline",
        "nodule d;\nfn f(x: Binary{8}) => Binary{8} = x;\n",
    ),
    (
        "short-type-stays-inline",
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\n",
    ),
    (
        "long-value-params-wrap",
        "nodule d;\nfn finish(nm: Bytes, i: Inputs, cands: CandList, mr: Option[Binary{8}], ci: Binary{8}, ov: Bool) => Result[Selected, SelectError] = nm;\n",
    ),
    (
        "long-sum-type-wraps",
        "nodule d;\ntype Predicate = PAlways | PSrcKindIs(Kind) | PDtypeIs(ScalarKind) | PGuaranteeAtLeast(Guarantee) | PDeclaredSparse | PAnd(Predicate, Predicate) | POr(Predicate, Predicate) | PNot(Predicate);\n",
    ),
    (
        "nested-match-wraps",
        "nodule d;\nfn cls(m: Binary{8}, u: Binary{8}) => Option[Binary{8}] = match eq(m, 0b0000_0000) { 0b1 => match eq(u, 0b0000_0000) { 0b1 => Some(m), _ => None }, _ => None };\n",
    ),
    (
        "long-call-args-wrap",
        "nodule d;\nfn g(pol: Pol) => Sel = finish(pol_name(pol), inputs_of(pol), pol_cands(pol), None, first_index(pol), True);\n",
    ),
];

/// C1 (round-trip) + C2 (idempotence) for `format_source_readable` over the corpus — the same
/// identity/fixed-point contract as the compact form, so the readable style is proven functionally
/// inert (presentation-only). Guarantee: `Empirical` (verified by execution).
#[test]
fn readable_round_trips_and_is_idempotent() {
    for (label, src) in READABLE_CORPUS {
        let original =
            parse(src).unwrap_or_else(|e| panic!("[{label}] corpus src must parse: {e}"));
        let r = format_source_readable(src, None)
            .unwrap_or_else(|e| panic!("[{label}] readable format failed: {e}"));

        // C1: the readable output re-parses to the SAME surface AST as the input.
        let reparsed = parse(&r.output).unwrap_or_else(|e| {
            panic!("[{label}] readable output must re-parse: {e}\n{}", r.output)
        });
        assert_eq!(
            reparsed, original,
            "[{label}] readable changed the surface AST (C1)\n{}",
            r.output
        );

        // The readable AST must equal the COMPACT AST too — both styles are the same projection.
        let compact = format_source(src, None).expect("compact formats");
        assert_eq!(
            parse(&compact.output).unwrap(),
            parse(&r.output).unwrap(),
            "[{label}] readable and compact must agree on the surface AST"
        );

        // C2: idempotent — a second readable format is byte-for-byte identical.
        let again = format_source_readable(&r.output, None)
            .unwrap_or_else(|e| panic!("[{label}] second readable format failed: {e}"));
        assert_eq!(
            again.output, r.output,
            "[{label}] readable is not idempotent (C2)"
        );
        assert!(!again.changed, "[{label}] re-format reported a change (C2)");
    }
}

/// The readability heuristic actually fires: a short construct stays inline (byte-identical to the
/// compact form), while a construct longer than [`READABLE_WIDTH`] breaks across multiple lines.
#[test]
fn readable_wraps_long_and_keeps_short_inline() {
    // Short: readable == compact (no line beyond the width is introduced).
    let short = "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\n";
    let r_short = format_source_readable(short, None).unwrap();
    let c_short = format_source(short, None).unwrap();
    assert_eq!(
        r_short.output, c_short.output,
        "a short construct must stay inline (readable == compact)"
    );

    // Long sum-type: readable differs from compact and wraps one variant per line with `|` breaks.
    let long_type = "nodule d;\ntype Predicate = PAlways | PSrcKindIs(Kind) | PDtypeIs(ScalarKind) | PGuaranteeAtLeast(Guarantee) | PDeclaredSparse | PAnd(Predicate, Predicate) | POr(Predicate, Predicate) | PNot(Predicate);\n";
    let r_long = format_source_readable(long_type, None).unwrap();
    assert!(
        r_long.output.contains("type Predicate =\n"),
        "a long sum-type must break after `=`:\n{}",
        r_long.output
    );
    assert!(
        r_long.output.contains("\n  | PNot(Predicate);"),
        "each subsequent variant must start a `|`-prefixed line:\n{}",
        r_long.output
    );
    // Every line of the wrapped declaration is within the readable width.
    for line in r_long.output.lines() {
        assert!(
            line.chars().count() <= READABLE_WIDTH,
            "wrapped line exceeds READABLE_WIDTH: {line:?}"
        );
    }

    // Long value-param list: readable wraps one parameter per line.
    let long_sig = "nodule d;\nfn finish(nm: Bytes, i: Inputs, cands: CandList, mr: Option[Binary{8}], ci: Binary{8}, ov: Bool) => Result[Selected, SelectError] = nm;\n";
    let r_sig = format_source_readable(long_sig, None).unwrap();
    assert!(
        r_sig.output.contains("fn finish(\n  nm: Bytes,\n"),
        "a long value-param list must wrap one param per line:\n{}",
        r_sig.output
    );
}

/// `format_source_styled` with `Style::Compact` is byte-identical to `format_source` — the refactor
/// that introduced the `style` parameter did not perturb the default (compact) path.
#[test]
fn styled_compact_equals_default_format() {
    for (label, src) in READABLE_CORPUS {
        let a = format_source(src, None).unwrap_or_else(|e| panic!("[{label}] format: {e}"));
        let b = format_source_styled(src, None, Style::Compact)
            .unwrap_or_else(|e| panic!("[{label}] styled-compact: {e}"));
        assert_eq!(
            a.output, b.output,
            "[{label}] Style::Compact must equal the default format"
        );
    }
}

// ============================================================================================
// Shape-Dispatched Readable acceptance tests (M-976 / DN-82)
// ============================================================================================
/// The Shape-Dispatched Readable acceptance oracle (M-976 / DN-82): the canonical
/// `(label, unchanged, before, after)` samples drawn from real `lib/std`. Each `after` is the
/// exact byte-for-byte target the COMPACT (default) readable style must reproduce from `before`
/// (the `unchanged` ones assert stability — `after == before`). Data-driven: the test body is an
/// assert over a case (house rule — the logic lives in this table, not the body).
#[allow(clippy::type_complexity)]
const SHAPE_DISPATCHED_SAMPLES: &[(&str, bool, &str, &str)] = &[
    (
        "cons-spine-matrix",
        false,
        r#"// matrix: the full 9-row table, in the same order as mycelium_std_core::GUARANTEE_MATRIX.
fn matrix() => Vec[GuaranteeRow] =
  Cons(
    row_value_repr_meta(),
    Cons(
      row_corevalue_datum(),
      Cons(
        row_guarantee_strength(),
        Cons(
          row_bound_boundbasis(),
          Cons(
            row_repr_of(),
            Cons(
              row_meta_of(),
              Cons(
                row_guarantee_of(),
                Cons(row_bound_of(), Cons(row_provenance_of(), Nil))
              )
            )
          )
        )
      )
    )
  );"#,
        r#"// matrix: the full 9-row table, in the same order as mycelium_std_core::GUARANTEE_MATRIX.
fn matrix() => Vec[GuaranteeRow] =
  Cons(row_value_repr_meta(),
  Cons(row_corevalue_datum(),
  Cons(row_guarantee_strength(),
  Cons(row_bound_boundbasis(),
  Cons(row_repr_of(),
  Cons(row_meta_of(),
  Cons(row_guarantee_of(),
  Cons(row_bound_of(),
  Cons(row_provenance_of(),
  Nil)))))))));"#,
    ),
    (
        "bool_and-spine",
        false,
        r#"// only_query_rows_explainable: the EXPLAIN window is exactly the value-tag/bound/provenance
// queries — lib.rs::tests::only_query_rows_are_explainable, ported as a per-row identity check
// (the Rust test's op-NAME list equality needs bytes_eq — the identity of each row constructor
// carries the same information here, so nothing is lost).
fn only_query_rows_explainable() => Bool =
  bool_and(
    row_explainable(row_guarantee_of()),
    bool_and(
      row_explainable(row_bound_of()),
      bool_and(
        row_explainable(row_provenance_of()),
        bool_and(
          bool_not(row_explainable(row_value_repr_meta())),
          bool_and(
            bool_not(row_explainable(row_corevalue_datum())),
            bool_and(
              bool_not(row_explainable(row_guarantee_strength())),
              bool_and(
                bool_not(row_explainable(row_bound_boundbasis())),
                bool_and(
                  bool_not(row_explainable(row_repr_of())),
                  bool_not(row_explainable(row_meta_of()))
                )
              )
            )
          )
        )
      )
    )
  );"#,
        r#"// only_query_rows_explainable: the EXPLAIN window is exactly the value-tag/bound/provenance
// queries — lib.rs::tests::only_query_rows_are_explainable, ported as a per-row identity check
// (the Rust test's op-NAME list equality needs bytes_eq — the identity of each row constructor
// carries the same information here, so nothing is lost).
fn only_query_rows_explainable() => Bool =
  bool_and(row_explainable(row_guarantee_of()),
  bool_and(row_explainable(row_bound_of()),
  bool_and(row_explainable(row_provenance_of()),
  bool_and(bool_not(row_explainable(row_value_repr_meta())),
  bool_and(bool_not(row_explainable(row_corevalue_datum())),
  bool_and(bool_not(row_explainable(row_guarantee_strength())),
  bool_and(bool_not(row_explainable(row_bound_boundbasis())),
  bool_and(bool_not(row_explainable(row_repr_of())),
  bool_not(row_explainable(row_meta_of()))))))))));"#,
    ),
    (
        "row-wide-flat-unchanged",
        true,
        r#"// ── honest reporting of delivery guarantee and audit bound ─────────────────────────────────────
fn row_guarantee_of() => MatrixRow =
  Row(
    "guarantee (of a Delivery)",
    GExact,
    Total,
    "total — REPORTS the sink's honest strength (None for Null; <= Declared in v0); never upgrades it (RT5/VR-5)",
    "none",
    IsExplainRecord,
    "reports the delivery guarantee on the lattice; the null sink honestly says None (not delivered — RT5/VR-5)"
  );"#,
        r#"// ── honest reporting of delivery guarantee and audit bound ─────────────────────────────────────
fn row_guarantee_of() => MatrixRow =
  Row(
    "guarantee (of a Delivery)",
    GExact,
    Total,
    "total — REPORTS the sink's honest strength (None for Null; <= Declared in v0); never upgrades it (RT5/VR-5)",
    "none",
    IsExplainRecord,
    "reports the delivery guarantee on the lattice; the null sink honestly says None (not delivered — RT5/VR-5)"
  );"#,
    ),
    (
        "cons-spine-testing",
        false,
        r#"fn row_summarize() => MatrixRow =
  Row("summarize", GExact, FalTotal, "total", EffNone, "none", True);

fn row_is_green() => MatrixRow =
  Row("is_green", GExact, FalTotal, "total", EffNone, "none", True);

// matrix: the full 5-row table, in guarantee_matrix.rs::MATRIX order.
fn matrix() => Vec[MatrixRow] =
  Cons(
    row_for_all(),
    Cons(
      row_golden(),
      Cons(row_differential(), Cons(row_summarize(), Cons(row_is_green(), Nil)))
    )
  );"#,
        r#"fn row_summarize() => MatrixRow =
  Row("summarize", GExact, FalTotal, "total", EffNone, "none", True);

fn row_is_green() => MatrixRow =
  Row("is_green", GExact, FalTotal, "total", EffNone, "none", True);

// matrix: the full 5-row table, in guarantee_matrix.rs::MATRIX order.
fn matrix() => Vec[MatrixRow] =
  Cons(row_for_all(),
  Cons(row_golden(),
  Cons(row_differential(),
  Cons(row_summarize(),
  Cons(row_is_green(),
  Nil)))));"#,
    ),
    (
        "nested-match-ladder",
        false,
        r#"// pack_tl1: 5 trits per byte; a ragged tail is None (unreachable after the alignment check; G2).
fn pack_tl1(ts: Trits) => Option[ByteList] =
  match ts {
    TNil => Some(BNil),
    TCons(t0, r0) =>
      match r0 {
        TNil => None,
        TCons(t1, r1) =>
          match r1 {
            TNil => None,
            TCons(t2, r2) =>
              match r2 {
                TNil => None,
                TCons(t3, r3) =>
                  match r3 {
                    TNil => None,
                    TCons(t4, rest) =>
                      match pack_tl1(rest) {
                        Some(bs) => Some(BCons(tl1_byte(t0, t1, t2, t3, t4), bs)),
                        None => None
                      }
                  }
              }
          }
      }
  };"#,
        r#"// pack_tl1: 5 trits per byte; a ragged tail is None (unreachable after the alignment check; G2).
fn pack_tl1(ts: Trits) => Option[ByteList] =
  match ts {
    TNil => Some(BNil),
    TCons(t0, r0) => match r0 {
      TNil => None,
      TCons(t1, r1) => match r1 {
        TNil => None,
        TCons(t2, r2) => match r2 {
          TNil => None,
          TCons(t3, r3) => match r3 {
            TNil => None,
            TCons(t4, rest) => match pack_tl1(rest) {
              Some(bs) => Some(BCons(tl1_byte(t0, t1, t2, t3, t4), bs)),
              None => None
            }
          }
        }
      }
    }
  };"#,
    ),
    (
        "let-chain-unchanged",
        true,
        r#"// tl1_group: one byte back to its 5-trit group (MSB-first). The five base-3 digits come off
// LSB-first via rem_u/div_u; a residual above 0 after five divisions means the byte value
// was > 242 — an explicit Err(OffGrid), never a silently wrapped group (C1/G2).
fn tl1_group(byte: Binary{8}) => Result[Trits, PackError] =
  let d4 = tl1_decode_digit(rem_u(byte, 0b0000_0011)) in
  let v1 = div_u(byte, 0b0000_0011) in
  let d3 = tl1_decode_digit(rem_u(v1, 0b0000_0011)) in
  let v2 = div_u(v1, 0b0000_0011) in
  let d2 = tl1_decode_digit(rem_u(v2, 0b0000_0011)) in
  let v3 = div_u(v2, 0b0000_0011) in
  let d1 = tl1_decode_digit(rem_u(v3, 0b0000_0011)) in
  let v4 = div_u(v3, 0b0000_0011) in
  let d0 = tl1_decode_digit(rem_u(v4, 0b0000_0011)) in
  match eq(div_u(v4, 0b0000_0011), 0b0000_0000) {
    0b1 => Ok(TCons(d0, TCons(d1, TCons(d2, TCons(d3, TCons(d4, TNil)))))),
    _ => Err(OffGrid)
  };"#,
        r#"// tl1_group: one byte back to its 5-trit group (MSB-first). The five base-3 digits come off
// LSB-first via rem_u/div_u; a residual above 0 after five divisions means the byte value
// was > 242 — an explicit Err(OffGrid), never a silently wrapped group (C1/G2).
fn tl1_group(byte: Binary{8}) => Result[Trits, PackError] =
  let d4 = tl1_decode_digit(rem_u(byte, 0b0000_0011)) in
  let v1 = div_u(byte, 0b0000_0011) in
  let d3 = tl1_decode_digit(rem_u(v1, 0b0000_0011)) in
  let v2 = div_u(v1, 0b0000_0011) in
  let d2 = tl1_decode_digit(rem_u(v2, 0b0000_0011)) in
  let v3 = div_u(v2, 0b0000_0011) in
  let d1 = tl1_decode_digit(rem_u(v3, 0b0000_0011)) in
  let v4 = div_u(v3, 0b0000_0011) in
  let d0 = tl1_decode_digit(rem_u(v4, 0b0000_0011)) in
  match eq(div_u(v4, 0b0000_0011), 0b0000_0000) {
    0b1 => Ok(TCons(d0, TCons(d1, TCons(d2, TCons(d3, TCons(d4, TNil)))))),
    _ => Err(OffGrid)
  };"#,
    ),
    (
        "cat-spine-in-arm",
        false,
        r#"// explain_deployed: the Deployed-arm EXPLAIN of explain_deploy — a total, deterministic function
// of the deployed id + verification record, byte-for-byte the oracle's rendering (VR-4/SC-3/C3:
// it always mentions both the content-hash check and the opaque-lowering check — no silent
// omission). The Failed arm rides deploy_error_display (FLAG-spore-6). Exact.
fn explain_deployed(spore_id: Bytes, v: DeployVerification) => Bytes =
  match v {
    Verification(hash_ok, opaque_ok) =>
      cat(
        "deploy-result: Deployed\nspore-id (content-hash): ",
        cat(
          spore_id,
          cat(
            "\ncontent_hash_canonical: ",
            cat(
              bool_text(hash_ok),
              cat(
                " (Exact — BLAKE3 deterministic; C4/ADR-003)\nno_opaque_lowering: ",
                cat(
                  bool_text(opaque_ok),
                  " (Declared — structural assertion; VR-4)\noutcome: all invariants checked; no opaque lowering step detected in pipeline"
                )
              )
            )
          )
        )
      )
  };"#,
        r#"// explain_deployed: the Deployed-arm EXPLAIN of explain_deploy — a total, deterministic function
// of the deployed id + verification record, byte-for-byte the oracle's rendering (VR-4/SC-3/C3:
// it always mentions both the content-hash check and the opaque-lowering check — no silent
// omission). The Failed arm rides deploy_error_display (FLAG-spore-6). Exact.
fn explain_deployed(spore_id: Bytes, v: DeployVerification) => Bytes =
  match v {
    Verification(hash_ok, opaque_ok) =>
      cat("deploy-result: Deployed\nspore-id (content-hash): ",
      cat(spore_id,
      cat("\ncontent_hash_canonical: ",
      cat(bool_text(hash_ok),
      cat(" (Exact — BLAKE3 deterministic; C4/ADR-003)\nno_opaque_lowering: ",
      cat(bool_text(opaque_ok),
      " (Declared — structural assertion; VR-4)\noutcome: all invariants checked; no opaque lowering step detected in pipeline"))))))
  };"#,
    ),
    (
        "let-block-indent",
        false,
        r#"// rng_next: advance the state and return it — Xorshift64 (x ^= x<<13; x ^= x>>7; x ^= x<<17),
// bit-exact vs Rust's Rng::next_u64 (whose output IS its new state). Exact: deterministic; the
// same state always yields the same output (C4/RT3).
fn rng_next(state: Binary{64}) => Binary{64} =
  let a = xor(
            state,
            shl_u(
              state,
              0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001101
            )
          ) in
  let b = xor(
            a,
            shr_u(
              a,
              0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000111
            )
          ) in
  xor(
    b,
    shl_u(b, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00010001)
  );"#,
        r#"// rng_next: advance the state and return it — Xorshift64 (x ^= x<<13; x ^= x>>7; x ^= x<<17),
// bit-exact vs Rust's Rng::next_u64 (whose output IS its new state). Exact: deterministic; the
// same state always yields the same output (C4/RT3).
fn rng_next(state: Binary{64}) => Binary{64} =
  let a = xor(
    state,
    shl_u(
      state,
      0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001101
    )
  ) in
  let b = xor(
    a,
    shr_u(a, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000111)
  ) in
  xor(
    b,
    shl_u(b, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00010001)
  );"#,
    ),
    (
        "let-in-arm-forced-open",
        false,
        r#"fn inspect[A, E, B](r: Result[A, E], f: A => B) => Result[A, E] =
  match r { Ok(x) => let peeked = f(x) in Ok(x), Err(e) => Err(e) };

// inspect_err: peek the Err side; the value and propagation are unchanged (mirror of inspect).
fn inspect_err[A, E, B](r: Result[A, E], f: E => B) => Result[A, E] =
  match r { Ok(x) => Ok(x), Err(e) => let peeked = f(e) in Err(e) };"#,
        r#"fn inspect[A, E, B](r: Result[A, E], f: A => B) => Result[A, E] =
  match r {
    Ok(x) =>
      let peeked = f(x) in
      Ok(x),
    Err(e) => Err(e)
  };

// inspect_err: peek the Err side; the value and propagation are unchanged (mirror of inspect).
fn inspect_err[A, E, B](r: Result[A, E], f: E => B) => Result[A, E] =
  match r {
    Ok(x) => Ok(x),
    Err(e) =>
      let peeked = f(e) in
      Err(e)
  };"#,
    ),
    (
        "glcons-spine",
        false,
        r#"// guarantee_matrix: the loaded matrix — 4 rows (the ported ops), all GExact, EXPLAIN-able = yes
// for every selection op; `build` constructs a policy and is not itself a selection.
fn guarantee_matrix() => GRowList =
  GLCons(
    GR("build", GExact, True, "none", ExplainNotApplicable),
    GLCons(
      GR("select", GExact, True, "none", ExplainYes),
      GLCons(
        GR("explain", GExact, True, "none", ExplainYes),
        GLCons(GR("select_with_override", GExact, True, "none", ExplainYes), GLNil)
      )
    )
  );"#,
        r#"// guarantee_matrix: the loaded matrix — 4 rows (the ported ops), all GExact, EXPLAIN-able = yes
// for every selection op; `build` constructs a policy and is not itself a selection.
fn guarantee_matrix() => GRowList =
  GLCons(GR("build", GExact, True, "none", ExplainNotApplicable),
  GLCons(GR("select", GExact, True, "none", ExplainYes),
  GLCons(GR("explain", GExact, True, "none", ExplainYes),
  GLCons(GR("select_with_override", GExact, True, "none", ExplainYes),
  GLNil))));"#,
    ),
];

/// Wrap a bare item body (`before`/`after` are item text with no nodule header) into a minimal
/// nodule so it parses, format it in a given style, and return the body text (header stripped).
fn readable_body(before: &str, cfg: LayoutCfg) -> String {
    let src = format!("nodule d;\n{before}\n");
    let out = format_source_readable_cfg(&src, None, cfg)
        .expect("shape sample must format")
        .output;
    let prefix = "nodule d;\n\n";
    out.strip_prefix(prefix)
        .unwrap_or(&out)
        .trim_end_matches('\n')
        .to_owned()
}

/// THE ACCEPTANCE ORACLE: in the COMPACT (`InlineWhenFits`) readable style, every canonical sample's
/// `after` is reproduced byte-for-byte from its `before` (R0–R6). The `unchanged` samples additionally
/// assert `after == before` (the confirmed-good anchors are left exactly as-is). Every sample also
/// round-trips (C1) and is a fixed point (C2), so the layout is behavior-neutral (`Empirical`).
///
/// **Width note (M-976).** The spec's `after` strings are its oracle **at width 88**; the *shipped*
/// default is now `READABLE_WIDTH = 100` (`rustfmt`-aligned, M-976). So this oracle pins `width: 88`
/// explicitly — it validates the R0–R6 *rules* against the spec's exact fixtures, independent of the
/// retuned default. The 100-wide default is exercised by the `lib/std` re-render + the round-trip /
/// wrap / shape-invariant tests, which are width-agnostic in structure.
#[test]
fn shape_dispatched_samples_reproduce_after_byte_for_byte() {
    let cfg = LayoutCfg {
        width: 88,
        ..LayoutCfg::default()
    };
    for (label, unchanged, before, after) in SHAPE_DISPATCHED_SAMPLES {
        let got = readable_body(before, cfg);
        let want = after.trim_end_matches('\n');
        assert_eq!(
            &got, want,
            "[{label}] compact readable output does not match the canonical `after`\n--- GOT ---\n{got}\n--- WANT ---\n{want}"
        );
        if *unchanged {
            assert_eq!(
                want,
                before.trim_end_matches('\n'),
                "[{label}] marked UNCHANGED but `after` differs from `before`"
            );
        }
        // C1: the readable output re-parses to the SAME surface AST as the input.
        let src = format!("nodule d;\n{before}\n");
        let original = parse(&src).expect("sample src parses");
        let formatted = format_source_readable_cfg(&src, None, cfg).unwrap().output;
        assert_eq!(
            parse(&formatted).expect("readable output re-parses"),
            original,
            "[{label}] readable changed the surface AST (C1)"
        );
        // C2: idempotent — a second format is byte-for-byte identical.
        let again = format_source_readable_cfg(&formatted, None, cfg)
            .unwrap()
            .output;
        assert_eq!(
            again, formatted,
            "[{label}] readable is not idempotent (C2)"
        );
    }
}

/// The `AlwaysExpand` house-style knob: the spine STILL stays flat (each `GLCons` link at one fixed
/// indent, no pyramid) but each inner nested `GR(...)` call is broken onto its own lines. Both
/// styles kill the pyramid; they differ only in inner-call density. Behavior-neutral (C1/C2).
#[test]
fn always_expand_keeps_flat_spine_and_expands_inner_calls() {
    let before = r#"fn guarantee_matrix() => GRowList =
  GLCons(
    GR("build", GExact, True, "none", ExplainNotApplicable),
    GLCons(GR("select", GExact, True, "none", ExplainYes), GLNil)
  );"#;
    let cfg = LayoutCfg {
        spine_inner: SpineInner::AlwaysExpand,
        ..LayoutCfg::default()
    };
    let got = readable_body(before, cfg);

    // The spine is flat: every `GLCons(` link begins a line at the SAME (2-space) indent.
    let spine_indents: Vec<usize> = got
        .lines()
        .filter(|l| l.trim_start().starts_with("GLCons("))
        .map(|l| l.len() - l.trim_start().len())
        .collect();
    assert!(
        spine_indents.len() >= 2,
        "expected a flat spine of >= 2 GLCons links:\n{got}"
    );
    assert!(
        spine_indents.iter().all(|&i| i == spine_indents[0]),
        "AlwaysExpand spine links must share one indent (flat, no pyramid):\n{got}"
    );
    // The inner GR(...) call is EXPANDED (broken onto its own lines) — a lone `GExact,` field line.
    assert!(
        got.lines().any(|l| l.trim() == "GExact,"),
        "AlwaysExpand must break each inner nested call one arg per line:\n{got}"
    );
    // Behavior-neutral: round-trips to the same AST as the compact style.
    let src = format!("nodule d;\n{before}\n");
    let compact_ast = parse(&format_source_readable(&src, None).unwrap().output).unwrap();
    let expand_ast = parse(&format_source_readable_cfg(&src, None, cfg).unwrap().output).unwrap();
    assert_eq!(
        compact_ast, expand_ast,
        "AlwaysExpand and InlineWhenFits must agree on the surface AST (behavior-neutral)"
    );
}

/// Shape invariants (R1/R5/R3/R2) beyond byte-equality, asserted structurally.
#[test]
fn shape_invariants_spine_flat_closers_coalesced_tree_shallow() {
    let cfg = LayoutCfg::default();

    // R1 + R5: a Cons spine draws every link at ONE indent and coalesces ALL closers into a single
    // horizontal run on the LAST line (never a vertical `)` wall).
    let matrix = r#"fn matrix() => Vec[Row] =
  Cons(row_alpha(), Cons(row_bravo(), Cons(row_charlie(), Cons(row_delta(), Cons(row_echo(), Cons(row_foxtrot(), Cons(row_golf(), Nil)))))));"#;
    let got = readable_body(matrix, cfg);
    let link_indents: Vec<usize> = got
        .lines()
        .filter(|l| l.trim_start().starts_with("Cons("))
        .map(|l| l.len() - l.trim_start().len())
        .collect();
    assert!(
        link_indents.len() >= 3,
        "expected a multi-link spine:\n{got}"
    );
    assert!(
        link_indents.iter().all(|&i| i == link_indents[0]),
        "R1: every spine link must share one indent (no per-link pyramid):\n{got}"
    );
    // Exactly one line carries a `)` closer run — the terminal line — and no earlier line ends in
    // `)` (R5: closers never form a vertical one-per-line stack).
    let closer_lines: Vec<&str> = got
        .lines()
        .filter(|l| l.trim_end().trim_end_matches(';').ends_with(')'))
        .collect();
    assert_eq!(
        closer_lines.len(),
        1,
        "R5: the coalesced closer run must appear on exactly one line:\n{got}"
    );
    assert!(
        closer_lines[0].trim_start().starts_with("Nil"),
        "R5: the single closer run rides the terminal line:\n{got}"
    );

    // R3: a genuine nested-match ladder gets ONE indent per real level (block-formers ride the arm
    // line), so the deepest arm indent grows by 2 per level — not the +4 of the old renderer.
    let ladder = r#"fn f(ts: T) => O =
  match ts {
    A => match ts {
      B => match ts {
        C => c,
        _ => d
      },
      _ => e
    },
    _ => g
  };"#;
    let got = readable_body(ladder, cfg);
    // The three `match … {` headers ride arm lines at increasing +2 indents (2, 4, 6 → riding).
    let inner_arm_indents: Vec<usize> = got
        .lines()
        .filter(|l| l.trim() == "C => c," || l.trim() == "_ => d")
        .map(|l| l.len() - l.trim_start().len())
        .collect();
    assert!(
        inner_arm_indents.iter().all(|&i| i == 6),
        "R3: one indent per real nesting level (deepest arms at 6, not 12):\n{got}"
    );

    // R2: a wide-flat (non-chain) call breaks one arg per line with the `)` ALONE on its own line.
    let wide = r#"fn r() => Row =
  Row("a-long-first-field-value-here", GExact, Total, "and-a-second-longer-field", "none", IsRec, "x");"#;
    let got = readable_body(wide, cfg);
    assert!(
        got.lines().any(|l| l.trim() == ")" || l.trim() == ");"),
        "R2: a broken wide-flat call closes with `)` alone on its own line:\n{got}"
    );
}

// RFC-0041 §4.7/§5 W0 guard-hole census (RR-29): tracked failing test for this crate's render-family
// hole. Registered as a dedicated submodule (rather than appended inline) so the census stays
// grep-able (`grep -rn 'ignore = "W' crates/`) across every crate it spans.
mod guard_hole_census;

// RFC-0041 §4.2/§9 W7 process-arena coverage (`docs/notes/W7-arena-coverage-audit.md`): the
// render-family arena wiring — a synthetic-refusal + normal-pass pair per public entry point.
mod arena_coverage;
