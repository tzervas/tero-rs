//! M-740 Stage 1 (DN-26 §7.3 row 1) — the self-hosted `compiler.token` / `compiler.lex` port.
//!
//! Smoke test (WIP): confirms `lib/compiler/token.myc` parses, checks, and evaluates under the
//! existing single-nodule pipeline, using the `std_*.rs` three-way harness pattern.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::{CoreValue, GuaranteeStrength, Payload};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// Extract a `Binary{N}` `CoreValue`'s bits as a `u32` (MSB-first), ignoring `Meta`/provenance —
/// used by [`assert_l1_only_u32`], which compares against a Rust-computed expected VALUE rather
/// than a second `.myc` "ref" program (a bare-literal ref would have different `Meta::provenance`
/// even when the VALUE matches, since two different source expressions never hash identically).
fn core_bits_as_u32(v: &CoreValue) -> u32 {
    let repr_val = v
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr CoreValue, got {v:?}"));
    match repr_val.payload() {
        Payload::Bits(bits) => bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b)),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

const TOKEN_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/token.myc"
));

const LEX_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/lex.myc"
));

fn program(driver: &str) -> String {
    format!("{TOKEN_SRC}\n{driver}")
}

fn lex_program(driver: &str) -> String {
    format!("{LEX_SRC}\n{driver}")
}

/// **FLAG-stage1-checker-2** (a real, reproduced performance finding, out of this leaf's scope to
/// fix — `crates/mycelium-interp`/`crates/mycelium-l1/src/elab.rs` are read-only per the task
/// boundary): for a `lex.myc`-scale program, `elaborate` completes quickly but the resulting L0
/// node then makes `mycelium_interp::Interpreter::eval_core` (the "L0 is a substitution machine"
/// path `crates/mycelium-l1/tests/depth_metric_parity.rs` already documents as `O(N²)`+ on deep
/// runtime recursion) run for minutes-plus even on a ~50-byte input that L1-eval (`Evaluator`)
/// finishes in well under a second — isolated by bisection: parse/check/monomorphize/L1-eval all
/// complete fast (verified directly), `elaborate` completes fast, but `Interpreter::eval_core` on
/// the elaborated node does not return within several minutes. This is the CLAUDE.md
/// depth_metric_parity.rs finding manifesting on real self-hosted source for the first time (that
/// file's own probes use synthetic deep terms; `lex.myc`'s natural recursive-descent style over
/// even a short program apparently already crosses into the same regime) — reported upstream, not
/// silently worked around. **Mitigation used here:** the Stage-1 `compiler.lex` differential below
/// compares the Rust oracle against the **L1-eval** leg only (`parse` → `check_nodule` →
/// `monomorphize` → `Evaluator::call` → `to_core`), which is itself a complete, correct
/// "Rust-lexer ≡ self-hosted-lexer" comparison (DN-26 §7.3's actual requirement) — it just does not
/// additionally cross-check against the L0-interp/AOT legs the way `std_*.rs`'s `assert_three_way`
/// does for smaller stdlib programs. A follow-up stage (or an `enb`-style enabler) should
/// investigate `eval_core`'s scaling before Stage 2+ ports larger self-hosted programs.
fn assert_l1_only_u32(label: &str, src: &str, expected_u32: u32) {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));
    let got = core_bits_as_u32(&l1_core);
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the Rust-oracle-computed expected value {expected_u32}"
    );
}

fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));

    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));

    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));

    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: elaborate failed: {e}"));
    let l0_core = interp
        .eval_core(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));

    let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
        .unwrap_or_else(|e| panic!("{label}: AOT run_core failed: {e}"));

    assert_eq!(
        l1_core, l0_core,
        "{label}: L1-eval(mono) vs elaborate->L0-interp diverged"
    );
    assert_eq!(l0_core, aot_core, "{label}: L0-interp vs AOT diverged");

    for (x, y, pair) in [
        (&l1_core, &l0_core, "L1<->interp"),
        (&l0_core, &aot_core, "interp<->AOT"),
    ] {
        assert_eq!(
            check_core(x, y),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "{label}: {pair} differential must validate Exact"
        );
    }

    let ref_env = check_nodule(
        &parse(expected_src).unwrap_or_else(|e| panic!("{label}: ref parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("{label}: ref check failed: {e}"));
    let ref_node = elaborate(&ref_env, "main")
        .unwrap_or_else(|e| panic!("{label}: ref elaborate failed: {e}"));
    let expected = interp
        .eval_core(&ref_node)
        .unwrap_or_else(|e| panic!("{label}: ref eval failed: {e}"));

    assert_eq!(
        l1_core, expected,
        "{label}: result does not match expected reference value"
    );
}

#[test]
fn token_myc_parses_checks_and_keyword_nodule_resolves() {
    let driver =
        "fn main() => Bool = match keyword(\"nodule\") { Some(_) => True, None => False };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("keyword(nodule) is Some", &src, expected);
}

#[test]
fn token_myc_keyword_plain_ident_is_none() {
    let driver =
        "fn main() => Bool = match keyword(\"frobnicate\") { Some(_) => True, None => False };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("keyword(frobnicate) is None", &src, expected);
}

/// All 64 keyword-classification arms, each asserting the SPECIFIC ctor (not just "some match") —
/// from the same (word, ctor) table used to write `keyword` in token.myc, itself transcribed 1:1
/// from `crates/mycelium-l1/src/token.rs::keyword`'s match arms (word-set diffed equal, 64/64,
/// during authoring). This is the Stage-1 `compiler.token` acceptance gate.
///
/// **FLAG-stage1-checker-1** (a real, minimally-reproduced usefulness-checker bug, out of this
/// leaf's scope to fix — `crates/mycelium-l1/src/usefulness.rs` is read-only per the task
/// boundary). Root cause isolated by bisection (not just observed): a COMBINED two-level nested
/// pattern in ONE match arm — `Some(Scalar(SF16))`, unwrapping `Option[Tok]` and a payload
/// constructor (`Tok::Scalar`/`Tok::Strength`, each wrapping a further sum type) in the same
/// pattern — panics `usefulness::useful_budgeted` (`panic_bounds_check`, `src/usefulness.rs:209`)
/// whenever the OUTER type has a large variant count (`Tok`, ~80 variants), and does so on a
/// SINGLE isolated site (not an aggregation effect — every other one of the 64 keywords, checked
/// in its own separate `parse`/`check_nodule` pass, is fast; only the four `Scalar(..)`-wrapped
/// and four `Strength(..)`-wrapped entries below trigger it, and wildcarding the inner payload
/// (`Some(Scalar(_))`) does NOT avoid it — the bug is the two-level nesting itself, not which
/// inner variant is named). **Workaround (verified correct, used below):** split the
/// Option-unwrap and the Tok-match into two separate `match` expressions — `match keyword(w) {
/// Some(t) => match t { Scalar(SF16) => True, _ => False }, None => False }` — one destructure per
/// match, which is already the idiomatic style throughout `lib/std/*.myc` (e.g.
/// `std.text::decode_one` never nests two constructors in one pattern). This is a genuine Stage-1
/// dogfooding finding — the exhaustiveness checker was presumably never exercised against an
/// ~80-variant sum type with nested payload constructors before this port — reported upstream (see
/// the leaf's final report); fixing `usefulness.rs` itself is a follow-up for the owning tier.
#[test]
fn token_myc_keyword_classifies_every_reserved_word_correctly() {
    let entries: &[(&str, &str)] = &[
        ("nodule", "Nodule"),
        ("phylum", "Phylum"),
        ("colony", "Colony"),
        ("hypha", "Hypha"),
        ("fuse", "Fuse"),
        ("mesh", "Mesh"),
        ("graft", "Graft"),
        ("cyst", "Cyst"),
        ("xloc", "Xloc"),
        ("forage", "Forage"),
        ("backbone", "Backbone"),
        ("tier", "Tier"),
        ("reclaim", "Reclaim"),
        ("consume", "Consume"),
        ("grow", "Grow"),
        ("lambda", "Lambda"),
        ("object", "Object"),
        ("via", "Via"),
        ("lower", "Lower"),
        ("derive", "Derive"),
        ("use", "Use"),
        ("pub", "Pub"),
        ("type", "Type"),
        ("trait", "Trait"),
        ("impl", "Impl"),
        ("fn", "Fn"),
        ("matured", "Matured"),
        ("thaw", "Thaw"),
        ("let", "Let"),
        ("in", "In"),
        ("if", "If"),
        ("then", "Then"),
        ("else", "Else"),
        ("match", "Match"),
        ("for", "For"),
        ("swap", "Swap"),
        ("default", "Default"),
        ("paradigm", "Paradigm"),
        ("with", "With"),
        ("wild", "Wild"),
        ("spore", "Spore"),
        ("to", "To"),
        ("policy", "Policy"),
        ("Binary", "KwBinary"),
        ("Ternary", "KwTernary"),
        ("Dense", "KwDense"),
        ("VSA", "Vsa"),
        ("bin", "BinShort"),
        ("tern", "TernShort"),
        ("emb", "EmbShort"),
        ("hvec", "HvecShort"),
        ("Seq", "KwSeq"),
        ("Bytes", "KwBytes"),
        ("Float", "KwFloat"),
        ("Substrate", "KwSubstrate"),
        ("Sparse", "KwSparse"),
        ("F16", "Scalar(SF16)"),
        ("BF16", "Scalar(SBf16)"),
        ("F32", "Scalar(SF32)"),
        ("F64", "Scalar(SF64)"),
        ("Exact", "Strength(GExact)"),
        ("Proven", "Strength(GProven)"),
        ("Empirical", "Strength(GEmpirical)"),
        ("Declared", "Strength(GDeclared)"),
    ];
    assert_eq!(
        entries.len(),
        64,
        "the keyword table must have exactly 64 entries"
    );
    for (word, ctor) in entries {
        // `Scalar(..)`/`Strength(..)` are payload-carrying variants wrapping a SECOND sum type
        // (ScalarTok/StrengthTok). A COMBINED two-level pattern in one match arm — e.g.
        // `Some(Scalar(SF16))` — panics the checker (FLAG-stage1-checker-1 above); splitting the
        // Option-unwrap and the Tok-match into two separate `match` expressions (one destructure
        // per match, the idiomatic style already used throughout lib/std/*.myc) avoids it and was
        // verified to give an identical, correct result.
        let driver = if ctor.contains('(') {
            format!(
                "fn main() => Bool = match keyword(\"{word}\") {{ Some(t) => match t {{ {ctor} => True, _ => False }}, None => False }};"
            )
        } else {
            format!(
                "fn main() => Bool = match keyword(\"{word}\") {{ Some({ctor}) => True, _ => False }};"
            )
        };
        let src = program(&driver);
        let expected = "nodule ref;\nfn main() => Bool = True;";
        assert_three_way(&format!("keyword({word}) is Some({ctor})"), &src, expected);
    }
}

/// Cross-checks the `.myc` `keyword` word-SET against the live Rust oracle
/// (`mycelium_l1::token::keyword`) directly — not just against the hand-authored word list above.
/// Every word the Rust lexer treats as a keyword must resolve to `Some` in the `.myc` port, and
/// every word it does NOT treat as a keyword (a battery of plain identifiers, including ones drawn
/// from the accept-corpus) must resolve to `None` — a real Rust-host vs self-host set-equality
/// check, not a self-consistency one.
#[test]
fn token_myc_keyword_set_matches_the_rust_oracle() {
    let rust_keywords = [
        "nodule",
        "phylum",
        "colony",
        "hypha",
        "fuse",
        "mesh",
        "graft",
        "cyst",
        "xloc",
        "forage",
        "backbone",
        "tier",
        "reclaim",
        "consume",
        "grow",
        "lambda",
        "object",
        "via",
        "lower",
        "derive",
        "use",
        "pub",
        "type",
        "trait",
        "impl",
        "fn",
        "matured",
        "thaw",
        "let",
        "in",
        "if",
        "then",
        "else",
        "match",
        "for",
        "swap",
        "default",
        "paradigm",
        "with",
        "wild",
        "spore",
        "to",
        "policy",
        "Binary",
        "Ternary",
        "Dense",
        "VSA",
        "bin",
        "tern",
        "emb",
        "hvec",
        "Seq",
        "Bytes",
        "Float",
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
    ];
    for w in rust_keywords {
        assert!(
            mycelium_l1::token::keyword(w).is_some(),
            "Rust oracle: {w:?} must be a keyword"
        );
    }
    let non_keywords = [
        "frobnicate",
        "x",
        "y",
        "foo",
        "Bar",
        "id",
        "shapes",
        "label",
        "samples",
        "literals",
        "app",
        "core",
        "solo",
        "operators",
        "arith",
        "bits",
        "prefix",
        "negate",
        "compare",
        "ordered",
        "mixed",
        "shift_and_compare",
        "a_sequence",
        "a_byte_string",
        "seq_and_bytes",
        "vsa",
        "bf16",
    ];
    for w in non_keywords {
        assert!(
            mycelium_l1::token::keyword(w).is_none(),
            "Rust oracle: {w:?} must NOT be a keyword"
        );
    }

    // Same battery through the .myc port (single compile; each word checked by the SAME
    // classification the property above holds the Rust oracle to). Batched into shallow groups
    // (<=8 conjuncts each) rather than one flat fold — a single ~90-deep nested expression drives
    // the hand-written recursive-descent parser/evaluator past their (un-widened, single-nodule
    // path) native stack budget; `check_phylum`'s multi-nodule path opts into
    // `mycelium_stack::with_deep_stack`, but plain `parse`/`check_nodule` do not, so depth is kept
    // shallow here rather than papering over it with an unrelated wrapper (Empirical finding, not
    // a Stage-1 language gap: nesting depth here is a test-authoring concern, not a lexer one).
    let mut checks = Vec::new();
    for w in rust_keywords {
        checks.push(format!(
            "match keyword(\"{w}\") {{ Some(_) => True, None => False }}"
        ));
    }
    for w in non_keywords {
        checks.push(format!(
            "match keyword(\"{w}\") {{ Some(_) => False, None => True }}"
        ));
    }
    let fold_shallow = |items: &[String]| -> String {
        let mut it = items.iter().rev();
        let mut acc = it.next().cloned().unwrap();
        for c in it {
            acc = format!("bool_and({c}, {acc})");
        }
        acc
    };
    let mut group_fns = Vec::new();
    let mut group_defs = String::new();
    for (gi, chunk) in checks.chunks(8).enumerate() {
        let name = format!("kw_oracle_group_{gi}");
        group_defs.push_str(&format!("fn {name}() => Bool = {};\n", fold_shallow(chunk)));
        group_fns.push(format!("{name}()"));
    }
    let body = fold_shallow(&group_fns);
    let driver = format!(
        "fn bool_and(a: Bool, b: Bool) => Bool = match a {{ True => b, False => False }};\n\
         {group_defs}\
         fn main() => Bool = {body};"
    );
    let src = program(&driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("keyword() word-set matches the Rust oracle", &src, expected);
}

// ── compiler.lex smoke tests ────────────────────────────────────────────────────────────────────

/// Format `n` as an explicit `Binary{32}` literal (`0b…`, 32 digits). Bare decimal literals do not
/// ambient-resolve in every position (constructor arguments and some prim/dedicated-branch
/// argument positions require an explicit-width literal — discovered empirically while authoring
/// `lex.myc`; RFC-0012's ambient-paradigm resolution needs a `default paradigm` declaration this
/// file never makes), so every generated driver in this file spells out widths explicitly.
fn b32(n: u32) -> String {
    format!("0b{n:032b}")
}

fn escape_myc_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[test]
fn lex_myc_parses_checks_and_empty_source_is_just_eof() {
    let driver = "fn main() => Binary{32} = tok_count(match lex(empty_bytes()) { Ok(v) => v, Err(_) => Nil });";
    let src = lex_program(driver);
    assert_l1_only_u32("lex(empty) has exactly the Eof token", &src, 1);
}

#[test]
fn lex_myc_token_count_matches_rust_oracle_on_a_small_snippet() {
    let snippet = "nodule foo;\nfn f() => Binary{8} = 0b0000_0000;\n";
    let rust_count = mycelium_l1::lexer::lex(snippet)
        .unwrap_or_else(|e| panic!("Rust oracle lex failed: {e}"))
        .len() as u32;
    let driver = format!(
        "fn main() => Binary{{32}} = tok_count(match lex(\"{}\") {{ Ok(v) => v, Err(_) => Nil }});",
        escape_myc_string(snippet)
    );
    let src = lex_program(&driver);
    assert_l1_only_u32(
        "lex(small snippet) token count matches Rust oracle",
        &src,
        rust_count,
    );
}

/// The Stage-1 gate (DN-26 §7.3): token-COUNT differential, Rust-lexer vs self-hosted `lex`, over
/// EVERY file in the L1 accept-corpus (`docs/spec/grammar/conformance/accept/`). A token count
/// mismatch is a strong, cheap signal of a classification/boundary bug (a single misclassified
/// token, a missed keyword, a mis-scanned literal, or a spurious/missing split almost always shifts
/// the count) without ever matching an individual `Tok` value (FLAG-stage1-checker-1) or paying the
/// L0-interp cost (FLAG-stage1-checker-2) once per file.
#[test]
fn lex_myc_token_count_matches_rust_oracle_over_the_accept_corpus() {
    let corpus_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/spec/grammar/conformance/accept");
    let mut files: Vec<_> = std::fs::read_dir(&corpus_dir)
        .unwrap_or_else(|e| panic!("cannot read accept-corpus dir {corpus_dir:?}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "myc"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 20,
        "expected the full accept-corpus (~27 files), found {}",
        files.len()
    );
    for path in &files {
        let source =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
        let rust_count = mycelium_l1::lexer::lex(&source)
            .unwrap_or_else(|e| panic!("Rust oracle lex failed on {path:?}: {e}"))
            .len() as u32;
        let driver = format!(
            "fn main() => Binary{{32}} = tok_count(match lex(\"{}\") {{ Ok(v) => v, Err(_) => Nil }});",
            escape_myc_string(&source)
        );
        let src = lex_program(&driver);
        assert_l1_only_u32(
            &format!("lex() token count for {}", path.display()),
            &src,
            rust_count,
        );
    }
}

/// Wrap `expr` (an `Option[Spanned]`, e.g. `nth(toks, i)`) into a driver body that yields `1` iff
/// `expr` is `Some` AND its `Tok` satisfies `tok_pattern_true`/`_ => false`, else `0`.
///
/// Uses the SAME split-match workaround as `token_myc_keyword_classifies_every_reserved_word_correctly`
/// (FLAG-stage1-checker-1): unwrapping `Option` and matching the `Tok` are two SEPARATE `match`
/// expressions (never `Some(Sp(Ctor, _))` in one combined pattern — that three-level nesting
/// (`Option` -> `Spanned` -> `Tok`) panics the checker exactly like `Some(Scalar(SF16))` did).
fn tok_at_check(expr: &str, tok_arm: &str) -> String {
    format!(
        "match {expr} {{ Some(sp) => match sp {{ Sp(t, _) => match t {{ {tok_arm}, _ => {} }} }}, None => {} }}",
        b32(0),
        b32(0)
    )
}

/// Per-token classification + content spot-check (not just the count differential above) for a
/// snippet exercising identifiers, keywords, punctuation, and the decimal `Int`/`0b` `BinLit`
/// literal forms. Each check is its OWN separate `parse`/`check_nodule` pass and never combines an
/// `Option`/`Spanned`/`Tok` unwrap into one pattern (FLAG-stage1-checker-1).
#[test]
fn lex_myc_per_token_kind_and_content_spot_checks() {
    let snippet = "nodule demo;\nfn f(x: Binary{8}) => Binary{8} = 0b0000_0001;\n";
    let rust_toks = mycelium_l1::lexer::lex(snippet).expect("rust oracle lex must succeed");
    assert_eq!(rust_toks.len(), 22, "expected token count sanity check");

    let base = escape_myc_string(snippet);
    let toks_expr = format!("match lex(\"{base}\") {{ Ok(v) => v, Err(_) => Nil }}");

    let checks: &[(u32, &str)] = &[
        (0, "Nodule"),
        (2, "Semi"),
        (3, "Fn"),
        (5, "LParen"),
        (7, "Colon"),
        (8, "KwBinary"),
        (9, "LBrace"),
        (11, "RBrace"),
        (12, "RParen"),
        (13, "FatArrow"),
        (17, "RBrace"),
        (18, "Eq"),
        (20, "Semi"),
        (21, "Eof"),
    ];
    for (idx, ctor) in checks {
        let nth_expr = format!("nth({toks_expr}, {})", b32(*idx));
        let tok_arm = format!("{ctor} => {}", b32(1));
        let driver = format!(
            "fn main() => Binary{{32}} = {};",
            tok_at_check(&nth_expr, &tok_arm)
        );
        let src = lex_program(&driver);
        assert_l1_only_u32(&format!("token[{idx}] is {ctor}"), &src, 1);
    }

    // Content checks (Bytes payload equality via bytes_eq — never a nested-Tok-matching concern:
    // `Ident(b)` binds `b` as a plain Bytes value, it does not destructure a further sum type).
    let content_checks: &[(u32, &str)] = &[(1, "demo"), (4, "f"), (6, "x")];
    for (idx, expected_ident) in content_checks {
        let nth_expr = format!("nth({toks_expr}, {})", b32(*idx));
        let tok_arm = format!(
            "Ident(b) => match bytes_eq(b, \"{expected_ident}\") {{ 0b1 => {}, _ => {} }}",
            b32(1),
            b32(0)
        );
        let driver = format!(
            "fn main() => Binary{{32}} = {};",
            tok_at_check(&nth_expr, &tok_arm)
        );
        let src = lex_program(&driver);
        assert_l1_only_u32(
            &format!("token[{idx}] Ident content is {expected_ident:?}"),
            &src,
            1,
        );
    }

    // Int / BinLit content (verbatim digits — FLAG-lex-2).
    let int_expr = format!("nth({toks_expr}, {})", b32(10));
    let int_arm = format!(
        "Int(b) => match bytes_eq(b, \"8\") {{ 0b1 => {}, _ => {} }}",
        b32(1),
        b32(0)
    );
    let driver_int = format!(
        "fn main() => Binary{{32}} = {};",
        tok_at_check(&int_expr, &int_arm)
    );
    let src_int = lex_program(&driver_int);
    assert_l1_only_u32("token[10] Int content is \"8\"", &src_int, 1);

    let bin_expr = format!("nth({toks_expr}, {})", b32(19));
    let bin_arm = format!(
        "BinLit(b) => match bytes_eq(b, \"0000_0001\") {{ 0b1 => {}, _ => {} }}",
        b32(1),
        b32(0)
    );
    let driver_bin = format!(
        "fn main() => Binary{{32}} = {};",
        tok_at_check(&bin_expr, &bin_arm)
    );
    let src_bin = lex_program(&driver_bin);
    assert_l1_only_u32("token[19] BinLit content is \"0000_0001\"", &src_bin, 1);
}

/// The trit-literal / byte-string-literal forms (0t / 0x) from the accept-corpus's own fixtures
/// (`10-literals.myc`, `20-seq-and-bytes-literals.myc`) — split-match, never a combined
/// `Option`->`Spanned`->`Tok` pattern (FLAG-stage1-checker-1).
#[test]
fn lex_myc_trit_and_bytes_literal_content_spot_checks() {
    let trit_snippet = "0t+0-;";
    let base = escape_myc_string(trit_snippet);
    let trit_expr = format!(
        "nth(match lex(\"{base}\") {{ Ok(v) => v, Err(_) => Nil }}, {})",
        b32(0)
    );
    let trit_arm = format!(
        "TritLit(b) => match bytes_eq(b, \"+0-\") {{ 0b1 => {}, _ => {} }}",
        b32(1),
        b32(0)
    );
    let driver = format!(
        "fn main() => Binary{{32}} = {};",
        tok_at_check(&trit_expr, &trit_arm)
    );
    let src = lex_program(&driver);
    assert_l1_only_u32("TritLit(\"+0-\") content matches", &src, 1);

    let bytes_snippet = "0x48_65;";
    let base2 = escape_myc_string(bytes_snippet);
    let bytes_expr = format!(
        "nth(match lex(\"{base2}\") {{ Ok(v) => v, Err(_) => Nil }}, {})",
        b32(0)
    );
    let bytes_arm = format!(
        "BytesLit(b) => match bytes_eq(b, \"48_65\") {{ 0b1 => {}, _ => {} }}",
        b32(1),
        b32(0)
    );
    let driver2 = format!(
        "fn main() => Binary{{32}} = {};",
        tok_at_check(&bytes_expr, &bytes_arm)
    );
    let src2 = lex_program(&driver2);
    assert_l1_only_u32("BytesLit(\"48_65\") content matches", &src2, 1);
}
