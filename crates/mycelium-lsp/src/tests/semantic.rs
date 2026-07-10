//! White-box tests for [`crate::semantic`] (M-730 base; M-975 completeness fix), relocated here
//! from the logic file's former inline `#[cfg(test)] mod tests` (M-797 as-touched retrofit).
//!
//! Data-driven: [`CLASSIFICATION_CASES`] is a source-snippet → expected-`tokenType` table so each
//! test body is *assert over a case*, not bespoke per-token logic (dev-workflow / CLAUDE.md test
//! layout convention).

use crate::semantic::*;
use crate::span::lex_items;

/// One (source snippet, expected `tokenType` index) case for the never-silent completeness sweep
/// (M-975): every kind of lexer token this crate is expected to classify, paired with the legend
/// index it must land on. Each snippet is a single otherwise-valid token in minimal context so the
/// classifier sees only tokens it can bucket unambiguously.
const CLASSIFICATION_CASES: &[(&str, u32)] = &[
    // Pre-existing buckets (regression coverage).
    ("fn", T_KEYWORD),
    ("Binary", T_TYPE),
    ("Exact", T_ENUM_MEMBER),
    ("0b0", T_NUMBER),
    ("+", T_OPERATOR),
    // M-975: the newer type-keyword tokens (RFC-0032/ADR-040 first-class repr types).
    ("Seq", T_TYPE),
    ("Bytes", T_TYPE),
    ("Float", T_TYPE),
    // M-975: the M-915 short repr-keyword aliases.
    ("bin", T_TYPE),
    ("tern", T_TYPE),
    ("emb", T_TYPE),
    ("hvec", T_TYPE),
    // M-975: literal kinds that previously fell through silently.
    ("0xAB", T_NUMBER),   // BytesLit
    ("1.5", T_NUMBER),    // FloatLit
    ("\"hi\"", T_STRING), // StrLit — the new `string` legend entry
    // M-975: keyword tokens missing from the original hand-maintained list.
    ("consume", T_KEYWORD),
    ("grow", T_KEYWORD),
    ("derive", T_KEYWORD),
    ("lambda", T_KEYWORD),
    ("object", T_KEYWORD),
    ("via", T_KEYWORD),
    ("lower", T_KEYWORD),
];

/// The single classified `tokenType` for a minimal one-token snippet, or `None` if the snippet
/// classified to nothing or didn't lex to exactly one item.
fn classify_snippet(src: &str) -> Option<u32> {
    let items = lex_items(src);
    let mut types = items.iter().filter_map(|it| classify(&it.kind));
    let first = types.next()?;
    // A minimal snippet must yield exactly one classified item.
    assert!(types.next().is_none(), "snippet `{src}` yielded >1 token");
    Some(first)
}

#[test]
fn legend_is_stable_and_indices_match() {
    // The encoded stream's tokenType indices must agree with the advertised legend order,
    // including the M-975 `string` addition.
    let legend = semantic_tokens_legend();
    let types = legend["tokenTypes"].as_array().unwrap();
    assert_eq!(types[T_KEYWORD as usize], "keyword");
    assert_eq!(types[T_TYPE as usize], "type");
    assert_eq!(types[T_ENUM_MEMBER as usize], "enumMember");
    assert_eq!(types[T_NUMBER as usize], "number");
    assert_eq!(types[T_COMMENT as usize], "comment");
    assert_eq!(types[T_STRING as usize], "string");
    assert!(legend["tokenModifiers"].as_array().unwrap().is_empty());
}

#[test]
fn unlexable_source_is_empty_stream_not_a_panic() {
    // G2: never-silent — an un-lexable document highlights nothing rather than fabricating.
    let v = semantic_tokens_full("fn f() = §");
    assert_eq!(v["data"].as_array().unwrap().len(), 0);
}

#[test]
fn keyword_type_strength_and_number_are_classified() {
    // `fn` is a keyword, `Binary` a type, `Exact` an enumMember, `0b0` a number.
    let src = "fn f() -> Binary{8} @ Exact = 0b0\n";
    let data = encode(&lex_items(src));
    assert_eq!(data.len() % 5, 0, "every token is a 5-tuple");
    // Collect the tokenType column (index 3 of each 5-tuple).
    let ttypes: Vec<u32> = data.chunks(5).map(|c| c[3]).collect();
    assert!(ttypes.contains(&T_KEYWORD), "fn → keyword");
    assert!(ttypes.contains(&T_TYPE), "Binary → type");
    assert!(ttypes.contains(&T_ENUM_MEMBER), "Exact → enumMember");
    assert!(ttypes.contains(&T_NUMBER), "0b0 → number");
}

#[test]
fn delta_encoding_is_relative_and_well_formed() {
    // Two lines: the first token of line 2 must encode a positive deltaLine and an absolute
    // (0-based) start column, not a column relative to the previous line.
    let src = "fn a()\nfn bb()\n";
    let data = encode(&lex_items(src));
    // First token: deltaLine 0, deltaStartChar 0 (the `fn` at col 1).
    assert_eq!(&data[0..2], &[0, 0]);
    // Find the first 5-tuple with a non-zero deltaLine: it begins line 2's `fn` at abs col 0.
    let line_break = data
        .chunks(5)
        .find(|c| c[0] == 1)
        .expect("a token starts a new line");
    assert_eq!(
        line_break[1], 0,
        "new-line token uses an absolute start column"
    );
}

#[test]
fn str_float_bytes_literals_and_new_type_keywords_are_classified() {
    // M-975: `StrLit`/`FloatLit`/`BytesLit` and the `Seq`/`Bytes`/`Float`/short-repr-keyword tokens
    // must no longer silently fall through to `None`.
    for &(src, want) in CLASSIFICATION_CASES {
        let got = classify_snippet(src);
        assert_eq!(
            got,
            Some(want),
            "snippet `{src}` classified as {got:?}, want Some({want})"
        );
    }
}

#[test]
fn comparison_and_shift_operators_are_classified_not_delimiters() {
    // M-975: post-RFC-0037-D1, `<`/`>`/`<<`/`>>` are pure infix operators (`lt`/`gt`/`shl`/`shr`),
    // not type-arg delimiters — the stale "delimiter" doc comment no longer applies to them.
    for src in ["<", ">", "<<", ">>"] {
        assert_eq!(
            classify_snippet(src),
            Some(T_OPERATOR),
            "`{src}` must classify as an operator, not fall through"
        );
    }
}

#[test]
fn delimiters_and_eof_remain_the_only_documented_none_cases() {
    // The explicit, documented "no highlight" set: real delimiters only (never `<`/`>`, see above).
    for src in ["(", ")", "{", "}", "[", "]", ":", ",", ";", "."] {
        assert_eq!(
            classify_snippet(src),
            None,
            "`{src}` is a documented delimiter and must stay unclassified"
        );
    }
}
