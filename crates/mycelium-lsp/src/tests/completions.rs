use crate::completions::*;

// ----- keyword presence -----

#[test]
fn all_active_structural_keywords_are_offered() {
    let labels: Vec<&str> = KEYWORD_COMPLETIONS.iter().map(|c| c.label).collect();
    // These are all the active structural keywords (token.rs `keyword()` -- active set).
    // `colony` and `hypha` were reserved-not-active until M-666; they are now active.
    for kw in [
        "nodule", "use", "type", "trait", "impl", "fn", "thaw", "let", "in", "if", "then", "else",
        "match", "for", "swap", "default", "paradigm", "with", "wild", "spore", "to", "policy",
        "matured", "colony", "hypha",
    ] {
        assert!(
            labels.contains(&kw),
            "active keyword `{kw}` missing from KEYWORD_COMPLETIONS"
        );
    }
}

#[test]
fn every_offered_keyword_is_a_real_active_lexer_keyword() {
    // Forward drift guard: every keyword-kind completion is recognized by the lexer's
    // authoritative `keyword()` set, so KEYWORD_COMPLETIONS cannot silently drift out of sync
    // with mycelium_l1::token::keyword() (e.g. by offering a word that is not actually a keyword).
    for c in KEYWORD_COMPLETIONS.iter() {
        assert!(
            mycelium_l1::token::keyword(c.label).is_some(),
            "completion `{}` is offered as a keyword but mycelium_l1::token::keyword() does \
             not recognize it — remove it from KEYWORD_COMPLETIONS or fix the lexer",
            c.label
        );
    }
}

#[test]
fn all_active_type_keywords_are_offered() {
    let labels: Vec<&str> = KEYWORD_COMPLETIONS.iter().map(|c| c.label).collect();
    for kw in [
        "Binary",
        "Ternary",
        "Dense",
        "VSA",
        "Substrate",
        "Sparse",
        "F16",
        "BF16",
        "F32",
        "F64",
        "Exact",
        "Proven",
        "Empirical",
        "Declared",
    ] {
        assert!(
            labels.contains(&kw),
            "active type/scalar/strength keyword `{kw}` missing from KEYWORD_COMPLETIONS"
        );
    }
}

#[test]
fn reserved_not_active_words_are_not_offered() {
    // `phylum`, the 8 remaining DN-03 §4 runtime-vocabulary words (reserved by M-665, minus
    // `colony`/`hypha` which became active in M-666), and the DN-03 §1 surface-tier words
    // `consume`/`grow` are reserved-not-active: they lex as keywords but no construct consumes
    // them yet (the parser refuses `consume`/`grow` with a teaching diagnostic until M-664) --
    // offering them as usable would violate the honesty rule (G2 / VR-5). `colony` and `hypha`
    // are now offered (see above); `consume`/`grow` were "ratified-not-yet-lexed" until they
    // were reserved into `keyword()`, mirroring the `hypha`…`reclaim` move under M-665.
    let labels: Vec<&str> = KEYWORD_COMPLETIONS
        .iter()
        .chain(SNIPPET_COMPLETIONS.iter())
        .map(|c| c.label)
        .collect();
    for banned in [
        "phylum", "fuse", "mesh", "graft", "cyst", "xloc", "forage", "backbone", "tier", "reclaim",
        "consume", "grow",
    ] {
        assert!(
            !labels.contains(&banned),
            "reserved-not-active word `{banned}` must NOT appear in completions"
        );
        // These ARE in keyword() (lexed) but excluded from the offered set. If one is dropped
        // from keyword(), this fails -- keeping the exclusion list and the lexer aligned.
        assert!(
            mycelium_l1::token::keyword(banned).is_some(),
            "`{banned}` is reserved-not-active but no longer in keyword() -- update the \
             exclusion list + this test together"
        );
    }
}

// (Historical note) `not_yet_lexed_words_are_not_offered` covered words ratified but not yet in
// `keyword()`. That category is now empty: `impl` graduated to an active keyword (M-659) and
// moved to the offered set; `consume`/`grow`, the last entries, were reserved into `keyword()`
// (DN-03 §1, lexed-not-active) and are now asserted by `reserved_not_active_words_are_not_offered`
// (the same place the `hypha`…`reclaim` runtime words landed after M-665). The test was removed
// rather than left asserting an empty set.

// ----- snippet well-formedness -----

#[test]
fn all_snippets_have_snippet_format_and_contain_tab_stops() {
    for snippet in SNIPPET_COMPLETIONS {
        assert_eq!(
            snippet.insert_text_format, FORMAT_SNIPPET,
            "snippet `{}` must use FORMAT_SNIPPET (2)",
            snippet.label
        );
        assert!(
            snippet.insert_text.contains('$'),
            "snippet `{}` has no tab stops (`$`)",
            snippet.label
        );
    }
}

#[test]
fn nodule_header_snippet_contains_nodule_and_comment_marker() {
    let nodule = SNIPPET_COMPLETIONS
        .iter()
        .find(|s| s.label == "nodule-header")
        .expect("nodule-header snippet must exist");
    assert!(nodule.insert_text.contains("// nodule:"));
    assert!(nodule.insert_text.contains("nodule "));
}

#[test]
fn swap_snippet_contains_both_to_and_policy() {
    // S1/WF2: both `to:` and `policy:` must always be present in a swap.
    let swap = SNIPPET_COMPLETIONS
        .iter()
        .find(|s| s.label == "swap-expr")
        .expect("swap-expr snippet must exist");
    assert!(
        swap.insert_text.contains("to:"),
        "swap snippet must contain `to:` (S1)"
    );
    assert!(
        swap.insert_text.contains("policy:"),
        "swap snippet must contain `policy:` (S1/WF2)"
    );
}

#[test]
fn fn_def_snippet_has_arrow_and_equals() {
    let fn_def = SNIPPET_COMPLETIONS
        .iter()
        .find(|s| s.label == "fn-def")
        .expect("fn-def snippet must exist");
    assert!(fn_def.insert_text.contains("->"), "fn-def must have `->`");
    assert!(fn_def.insert_text.contains('='), "fn-def must have `=`");
}

// ----- completion_list() shape -----

#[test]
fn completion_list_has_lsp_shape() {
    let list = completion_list();
    assert_eq!(
        list["isIncomplete"], false,
        "isIncomplete must be false for a static list"
    );
    let items = list["items"].as_array().expect("items must be an array");
    assert!(
        !items.is_empty(),
        "completion list must have at least one item"
    );
    // Every item must have the required LSP CompletionItem fields.
    for item in items {
        assert!(item["label"].is_string(), "each item must have a `label`");
        assert!(item["kind"].is_number(), "each item must have a `kind`");
        assert!(
            item["insertText"].is_string(),
            "each item must have `insertText`"
        );
    }
}

#[test]
fn completion_list_total_count_matches_constants() {
    let list = completion_list();
    let items = list["items"].as_array().unwrap();
    assert_eq!(
        items.len(),
        KEYWORD_COMPLETIONS.len() + SNIPPET_COMPLETIONS.len(),
        "completion_list() must include every keyword and every snippet"
    );
}

#[test]
fn keyword_completions_use_plain_format_and_kind_14() {
    for kw in KEYWORD_COMPLETIONS {
        assert_eq!(
            kw.kind, KIND_KEYWORD,
            "keyword `{}` must have kind=14 (Keyword)",
            kw.label
        );
        assert_eq!(
            kw.insert_text_format, FORMAT_PLAIN,
            "keyword `{}` must use plain insert-text format",
            kw.label
        );
    }
}
