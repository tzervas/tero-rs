use crate::ast::{BaseType, Expr, Item, Literal, WidthRef};
use crate::*;

#[test]
fn parses_a_nodule_with_a_swap() {
    let src =
        "nodule demo;\nfn f(x: Binary{8}) => Ternary{6} =\n  swap(x, to: Ternary{6}, policy: rt);";
    let nodule = parse(src).expect("parses");
    assert_eq!(nodule.path.0, vec!["demo"]);
    assert_eq!(nodule.items.len(), 1);
    let Item::Fn(f) = &nodule.items[0] else {
        panic!("expected a fn item");
    };
    assert!(!f.thaw);
    assert!(matches!(f.body, Expr::Swap { .. }));
    assert_eq!(f.sig.ret.base, BaseType::Ternary(WidthRef::Lit(6)));
}

#[test]
fn a_swap_without_policy_is_an_explicit_error() {
    // S1/WF2: the policy is mandatory; its absence is a diagnostic, never a silent accept.
    let src = "nodule demo;\nfn f(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6})";
    let err = parse(src).unwrap_err();
    assert!(err.message.contains("policy"), "got: {}", err.message);
}

#[test]
fn a_reserved_word_is_not_a_usable_identifier() {
    let src = "nodule demo;\nfn nodule(x: Binary{8}) => Binary{8} = x";
    assert!(parse(src).is_err());
}

#[test]
fn phylum_is_an_active_header_but_never_an_identifier() {
    // M-662 / DN-06 / RFC-0006 §4.3: `phylum` ACTIVATED as a header keyword — it opens a phylum
    // program via `parse_phylum` (`phylum <path>` then `nodule` blocks). It is **not** a
    // single-nodule program opener, so `parse` (single-nodule) still rejects it; and it remains a
    // keyword, so it can never be a silent identifier (G2). (`colony` activated as an *expression*
    // with M-666 — see `colony_and_hypha_are_active`; `phylum` activates here as a *header*.)
    // `parse` (the single-nodule entry) does not consume a `phylum` header:
    assert!(parse("phylum signals\nnodule demo;\n").is_err());
    // …but `parse_phylum` does — `phylum <path>` + a `nodule` block is a well-formed phylum.
    let ph = parse_phylum("phylum signals.demo\nnodule a;\nfn f() => Binary{8} = 0b0;")
        .expect("a phylum header + nodule parses (M-662)");
    assert_eq!(
        ph.path.as_ref().map(|p| p.0.clone()),
        Some(vec!["signals".to_owned(), "demo".to_owned()])
    );
    assert_eq!(ph.nodules.len(), 1);
    // `phylum` is still a keyword ⇒ never a usable identifier (G2).
    assert!(parse("nodule demo;\nfn phylum() => Binary{8} = 0b0").is_err());
    // A `phylum` header with no following nodule is a never-silent error (a phylum groups nodules).
    assert!(parse_phylum("phylum signals\n").is_err());
}

#[test]
fn colony_and_hypha_are_active() {
    // M-666 / RFC-0008 §4.7: `colony { hypha … }` is now an **active** L1 expression construct.
    // A well-formed colony parses; `colony`/`hypha` are still keywords, so they can never be
    // identifiers (G2) — using either as a name remains an explicit error.
    let n = parse(
            "nodule demo;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\nfn run() => Binary{8} = colony { hypha compute(0b0000_0001), hypha compute(0b0000_0010) };",
        )
        .expect("a well-formed colony parses (M-666)");
    let Item::Fn(run) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(f) if f.sig.name == "run"))
        .expect("run fn")
    else {
        panic!("run fn");
    };
    let Expr::Colony(hyphae) = &run.body else {
        panic!("run body must be a colony, got {:?}", run.body);
    };
    assert_eq!(hyphae.len(), 2, "two hyphae");

    // `colony`/`hypha` are still keywords → never usable as identifiers (G2).
    assert!(
        parse("nodule demo;\nfn f(colony: Binary{8}) => Binary{8} = colony").is_err(),
        "`colony` as a param name must stay an error (still a keyword)"
    );
    assert!(
        parse("nodule demo;\nfn hypha() => Binary{8} = 0b0").is_err(),
        "`hypha` as a fn name must stay an error (still a keyword)"
    );
    // M-906 (DN-70 D1): every hypha's `forage` field defaults to `None` when no `@forage(policy)`
    // annotation is written — the D-lite surface is additive, never a silent behavior change for
    // existing colonies.
    assert!(
        hyphae.iter().all(|h| h.forage.is_none()),
        "an un-annotated hypha's `forage` field must be `None`"
    );
    // A bare `hypha` outside a colony is a never-silent error (RT7 — no orphan hypha).
    let orphan = parse("nodule demo;\nfn f() => Binary{8} = hypha g(0b0)").unwrap_err();
    assert!(
        orphan.message.contains("only valid inside a `colony"),
        "orphan hypha must teach the colony scoping, got: {}",
        orphan.message
    );
    // An empty colony is rejected at type-check (the parser requires ≥1 hypha at parse time —
    // an empty `colony { }` fails parsing because `hypha` is required for the first element).
    let empty = parse("nodule demo;\nfn f() => Binary{8} = colony { }").unwrap_err();
    assert!(
        empty.message.contains("hypha"),
        "empty colony must mention `hypha`, got: {}",
        empty.message
    );
}

#[test]
fn runtime_vocab_keywords_are_reserved_not_active() {
    // DN-03 §4 / RFC-0008 §4.5 / M-665: the Runtime-tier names are reserved keywords — they lex
    // as keywords (never silent identifiers, G2) but no L1 construct consumes them. `hypha`
    // **left** this set with M-666 (it is now active inside a `colony`); `fuse`/`reclaim`/`tier`
    // left with M-667 / DN-58 (they are now active constructs). `forage` gained a **narrow, D-lite**
    // active surface with M-906/DN-70 D1 — but ONLY the `@forage(policy) hypha …` attribute form
    // (`parse_hypha` special-cases `Tok::At` immediately followed by `Tok::Forage`); a *bare*
    // `forage` at item/expression/fn-name/param-name position is still refused exactly as before
    // (this test's cases a–d below are unaffected — see `forage_dlite_annotation_parses_and_is_
    // recorded` for the new active-surface coverage). The remaining five (mesh/graft/cyst/xloc/
    // backbone) stay fully reserved-not-active until their own constructs land (RFC-0008 §4.6 R2).
    //
    // M-907 (DN-70 §D2, verify-only — no re-land) re-verified `backbone` specifically against this
    // test at `origin/dev` 367c601 (2026-07-02), after M-906 landed forage's D-lite surface above.
    // Inventory (`Empirical`): `backbone` still lexes as `Tok::Backbone` (token.rs:58-59) and is
    // rejected — never silently — at item position (parse.rs:513-525) and expression position
    // (parse.rs:1741-1752) with the same RFC-0008 teaching diagnostic exercised by cases (b-item)/
    // (d) below; no executing `backbone` construct, and no `BackboneRef` type, exists anywhere in
    // the tree (repo-wide grep, zero hits). The D-lite `@forage(policy)` surface (M-906) consumes
    // only a policy expression (parse.rs:2161-2189) — it does not reference or require a backbone
    // input, so DN-70 §D2's predicted outcome ("recorded residual, nothing to build" on a single
    // node) holds: **no residual to close** in this crate. The multi-node/promotion residual
    // (DN-63 FLAG-16, backbone's H2 maturity) is separately mechanized outside `mycelium-l1`, in
    // `mycelium-std-runtime::r2_residual::DeferredR2::MultiNodePlacement` (DN-78 §4 R-6, tracker
    // M-828) — a total, tested refusal ledger entry, not silently dropped. FLAG-16 itself stays
    // open, owned by the future backbone implementation RFC (unaffected by this verification pass).
    //
    // Honesty (Declared): the RFC-0008 teaching diagnostic fires when the runtime keyword is
    // reached in a position where the parser dispatches to `parse_item` or `parse_expr_inner`
    // (cases b-item and d). At positions where the parser expects a plain Ident token (the
    // fn-name slot, param binders, or program opener) it raises the standard "expected an
    // identifier / expected a `nodule` header" error — still explicit and non-silent (G2),
    // just without the RFC-0008 reference, because the never-active guard fires earlier.
    let words = ["mesh", "graft", "cyst", "xloc", "forage", "backbone"];
    for word in words {
        // Sanity: `keyword(w)` returns Some — the word lexes as a keyword, not a plain Ident.
        assert!(
            crate::token::keyword(word).is_some(),
            "`{word}` must resolve to a keyword token (keyword() must return Some)"
        );

        // (a) cannot open a program — parser sees the keyword where `nodule` is required.
        // Error is "expected a `nodule` header", not the RFC-0008 message (the parser never
        // reaches `parse_item`), but the reservation is still non-silent (G2).
        assert!(
            parse(&format!("{word} signals\n")).is_err(),
            "`{word}` opening a program must be an explicit error"
        );

        // (b-item) at item position (after a valid nodule header), `parse_item` dispatches to
        // the reserved-keyword arm and produces the RFC-0008 teaching diagnostic.
        let err = parse(&format!("nodule demo;\n{word} worker")).unwrap_err();
        assert!(
            err.message.contains("RFC-0008"),
            "`{word}` at item position: teaching diagnostic must mention RFC-0008, got: {}",
            err.message
        );

        // (b-name) fn-name slot expects an Ident: "expected an identifier" — explicit, not
        // the RFC-0008 message, because `parse_sig_tail` → `ident()` fires before `parse_item`.
        assert!(
            parse(&format!("nodule demo;\nfn {word}() => Binary{{8}} = 0b0")).is_err(),
            "`{word}` as fn name must be an explicit error"
        );

        // (c) cannot be used as a parameter name (binder expects an Ident).
        assert!(
            parse(&format!(
                "nodule demo;\nfn f({word}: Binary{{8}}) => Binary{{8}} = 0b0"
            ))
            .is_err(),
            "`{word}` as param name must be an error"
        );

        // (d) at expression position, `parse_expr_inner` dispatches to the reserved-keyword
        // arm and produces the RFC-0008 teaching diagnostic.
        let err = parse(&format!("nodule demo;\nfn f() => Binary{{8}} = {word}")).unwrap_err();
        assert!(
            err.message.contains("RFC-0008"),
            "`{word}` in expression position: teaching diagnostic must mention RFC-0008, got: {}",
            err.message
        );
    }
}

#[test]
fn forage_dlite_annotation_parses_and_is_recorded() {
    // M-906 (DN-70 D1; RFC-0008 RT3): `@forage(policy) hypha <expr>` — the D-lite active surface.
    // `forage` is still the same reserved keyword (never a silent identifier, G2); this is a
    // grammar-level special case for exactly the `@forage(…) hypha` sequence (`parse_hypha`), not
    // a general reactivation — see `runtime_vocab_keywords_are_reserved_not_active` above for the
    // bare-word behavior, which is unchanged.
    let n = parse(
        "nodule demo;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\n\
         fn run() => Binary{8} = colony { @forage(0b101) hypha compute(0b0000_0001) };",
    )
    .expect("a `@forage(…) hypha` annotation parses (M-906)");
    let Item::Fn(run) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(f) if f.sig.name == "run"))
        .expect("run fn")
    else {
        panic!("run fn");
    };
    let Expr::Colony(hyphae) = &run.body else {
        panic!("run body must be a colony, got {:?}", run.body);
    };
    assert_eq!(hyphae.len(), 1, "one hypha");
    let policy = hyphae[0]
        .forage
        .as_deref()
        .expect("the hypha's `forage` field must be `Some` after `@forage(0b101)`");
    assert!(
        matches!(policy, Expr::Lit(Literal::Bin(s)) if s == "101"),
        "the parsed policy must be the literal bitmask expression, got {policy:?}"
    );

    // A hypha with no `@forage(…)` prefix in the SAME colony still parses with `forage: None` —
    // the annotation is per-hypha, not per-colony (additive, never a silent colony-wide default).
    let mixed = parse(
        "nodule demo;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\n\
         fn run() => Binary{8} =\n  colony { @forage(0b1) hypha compute(0b0000_0001), hypha compute(0b0000_0010) };",
    )
    .expect("a mixed annotated/un-annotated colony parses");
    let Item::Fn(run2) = mixed
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(f) if f.sig.name == "run"))
        .expect("run fn")
    else {
        panic!("run fn");
    };
    let Expr::Colony(hyphae2) = &run2.body else {
        panic!("run body must be a colony");
    };
    assert!(hyphae2[0].forage.is_some(), "first hypha is annotated");
    assert!(hyphae2[1].forage.is_none(), "second hypha is not annotated");
}

#[test]
fn surface_keyword_consume_is_active_expression() {
    // DN-03 §1 / M-664: `consume <expr>` is now an **active expression** (affine acquisition of a
    // `Substrate` value, LR-8). It still lexes as a keyword (never a silent identifier, G2); at
    // item position it is a teaching error (an expression, not a top-level item); at expression
    // position it parses.
    let word = "consume";

    // (a) lexes as a keyword, not a plain Ident.
    assert!(
        crate::token::keyword(word).is_some(),
        "`{word}` must resolve to a keyword token (keyword() must return Some)"
    );

    // (b-item) item position → teaching diagnostic: `consume` is an expression, not a top-level item.
    let err = parse(&format!("nodule demo;\n{word} worker")).unwrap_err();
    assert!(
        err.message.contains("expression") && err.message.contains("M-664"),
        "`{word}` at item position: diagnostic must say it is an expression (M-664), got: {}",
        err.message
    );

    // (c) cannot be a fn name / param name (binder expects an Ident) — explicit, never silent.
    assert!(
        parse(&format!("nodule demo;\nfn {word}() => Binary{{8}} = 0b0")).is_err(),
        "`{word}` as fn name must be an explicit error"
    );
    assert!(
        parse(&format!(
            "nodule demo;\nfn f({word}: Binary{{8}}) => Binary{{8}} = 0b0"
        ))
        .is_err(),
        "`{word}` as param name must be an error"
    );

    // (d) expression position → `consume <operand>` now PARSES (type-checking the `Substrate`
    // operand is a separate pass, exercised in `tests/check.rs`).
    assert!(
        parse(&format!(
            "nodule demo;\nfn f(s: Substrate{{Sock}}) => Substrate{{Sock}} = {word} s;"
        ))
        .is_ok(),
        "`{word} s` in expression position must parse (M-664 — consume is an active expression)"
    );
}

#[test]
fn grow_is_superseded_by_derive_teaching_diagnostic() {
    // DN-38 §8.1 / M-812: `grow` is superseded by `derive` — the keyword still lexes (reserved,
    // never a silent identifier, G2) but produces a *teaching* diagnostic pointing at `derive`,
    // distinct from the plain DN-03 §1 "not yet active" message for `consume`.
    let word = "grow";

    // (a) lexes as a keyword, not a plain Ident.
    assert!(
        crate::token::keyword(word).is_some(),
        "`grow` must resolve to a keyword token"
    );

    // (b-item) item position → DN-38 §8.1 / M-812 teaching diagnostic mentioning `derive`.
    let err = parse(&format!("nodule demo;\n{word} worker")).unwrap_err();
    assert!(
        err.message.contains("DN-38") && err.message.contains("derive"),
        "`grow` at item position: diagnostic must name DN-38 + `derive`, got: {}",
        err.message
    );

    // (c) cannot be a fn name / param name.
    assert!(
        parse(&format!("nodule demo;\nfn {word}() => Binary{{8}} = 0b0")).is_err(),
        "`grow` as fn name must be an explicit error"
    );
    assert!(
        parse(&format!(
            "nodule demo;\nfn f({word}: Binary{{8}}) => Binary{{8}} = 0b0"
        ))
        .is_err(),
        "`grow` as param name must be an error"
    );

    // (d) expression position → DN-38 §8.1 / M-812 teaching diagnostic mentioning `derive`.
    let err = parse(&format!("nodule demo;\nfn f() => Binary{{8}} = {word}")).unwrap_err();
    assert!(
        err.message.contains("DN-38") && err.message.contains("derive"),
        "`grow` in expression position: diagnostic must name DN-38 + `derive`, got: {}",
        err.message
    );
}

#[test]
fn a_malformed_ternary_literal_is_explicit() {
    // RFC-0037 D4: trit literals are the `0t…` prefix form (lexed whole, like `0b…`). The lexer
    // accepts only the `+`/`0`/`-` glyphs and stops at anything else, so a malformed trit can no
    // longer carry a stray non-trit char. The never-silent (G2) malformed-ternary case is now a
    // *bare* `0t` with no glyph — an explicit lex error surfaced through `parse`, never a
    // silently-empty literal. (The former `<+x->`/"non-trit" parse path is retired with the angle
    // form; the residual "non-trit" elab check on a `Literal::Trit` is now unreachable from surface
    // syntax — the lexer prevents it upstream.)
    let src = "nodule demo;\nfn f() => Ternary{3} = 0t";
    let err = parse(src).unwrap_err();
    assert!(err.message.contains("no trits"), "got: {}", err.message);
}

#[test]
fn thaw_fn_parses_and_sets_thaw_true() {
    // RFC-0017 §4.3: `thaw fn` is the de-maturation marker; the field must be `true`.
    let src = "nodule demo;\nthaw fn k() => Binary{8} = 0b1011_0010;";
    let nodule = parse(src).unwrap();
    let Item::Fn(f) = &nodule.items[0] else {
        panic!("fn");
    };
    assert!(f.thaw);
    assert!(matches!(&f.body, Expr::Lit(Literal::Bin(s)) if s == "1011_0010"));
}

#[test]
fn matured_fn_at_item_position_is_a_parse_error_with_teaching_diagnostic() {
    // RFC-0017 §4.1: `matured fn` at item position is retired — the parser must return an
    // explicit error whose message teaches the scope form (`// @matured: true` header /
    // `thaw fn`). `matured` stays a reserved keyword token, so this is never a silent accept.
    let src = "nodule demo;\nmatured fn k() => Binary{8} = 0b00000000";
    let err = parse(src).unwrap_err();
    assert!(
        err.message.contains("maturation"),
        "teaching diagnostic must mention maturation, got: {}",
        err.message
    );
    assert!(
        err.message.contains("thaw"),
        "teaching diagnostic must mention `thaw`, got: {}",
        err.message
    );
}
