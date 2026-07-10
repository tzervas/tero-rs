//! The **parser conformance gate** (RFC-0006 §4.3; the WebAssembly-spec pattern, T3.1-B): the L1
//! parser must **accept** every program under `docs/spec/grammar/conformance/accept/` and
//! **reject** every program under `…/reject/` with an explicit [`ParseError`] — never a panic,
//! never a silent accept. The corpus is the ground truth; this test makes the grammar artifact
//! and the parser agree. The oracle is [`parse_phylum`] (M-662) — the top-level surface entry and a
//! strict superset of `parse` (a bare nodule is a phylum-of-one), so every pre-phylum fixture holds.

use std::fs;
use std::path::PathBuf;

use mycelium_l1::parse_phylum;

fn corpus_dir(kind: &str) -> PathBuf {
    // crate dir → repo root → the grammar conformance corpus.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/spec/grammar/conformance")
        .join(kind)
}

fn myc_files(kind: &str) -> Vec<PathBuf> {
    let dir = corpus_dir(kind);
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "myc"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .myc fixtures in {}", dir.display());
    files
}

#[test]
fn accept_corpus_all_parses() {
    for path in myc_files("accept") {
        let src = fs::read_to_string(&path).unwrap();
        match parse_phylum(&src) {
            Ok(_) => {}
            Err(e) => panic!("{} should parse but failed: {e}", path.display()),
        }
    }
}

/// Per-file expected-error fragments (A4). Each reject fixture must fail for the *intended*
/// reason: asserting only `is_err()` would let a fixture pass on an unintended failure (e.g. a
/// lexer error masking the grammar violation the fixture is meant to exercise). Each entry maps a
/// `reject/NN-*.myc` filename to a distinctive, stable fragment of the real `ParseError` message
/// its rejection must contain — making the corpus self-policing.
///
/// Every reject fixture must have an entry here; [`reject_corpus_all_fails_explicitly`] fails if a
/// new fixture lacks one, so the table cannot silently fall behind the corpus.
const REJECT_EXPECTED: &[(&str, &str)] = &[
    (
        "01-no-nodule-header.myc",
        "expected a `nodule` header to open the program",
    ),
    ("02-swap-missing-policy.myc", "a swap is never silent"),
    ("03-unclosed-brace.myc", "expected `}` to close the match"),
    (
        // RFC-0037 D4: a `0t…` literal whose first glyph is not a trit (`0tx`) leaves the literal
        // empty — the lexer scans only `+`/`0`/`-`, so a non-trit glyph is a never-silent "no trits"
        // lex refusal (the angle form `<+x->` was retired with D4).
        "04-bad-trit.myc",
        "balanced-ternary literal `0t` has no trits",
    ),
    ("05-reserved-word-ident.myc", "expected an identifier"),
    // RFC-0037 D4: the return arrow is now `=>` (the diagnostic names the new glyph).
    ("06-missing-arrow.myc", "expected `=>` and a result type"),
    (
        "07-empty.myc",
        "expected a `nodule` header to open the program",
    ),
    ("08-imperative-while.myc", "`while` is not a Mycelium form"),
    (
        "09-default-missing-paradigm.myc",
        "expected `paradigm` after `default`",
    ),
    (
        // M-662: `phylum` is now ACTIVE, so a phylum header with no nodule is a never-silent parse
        // refusal (not the former reserved-not-active rejection). The runtime vocabulary that is
        // still reserved-not-active is exercised by fixture 12.
        "10-phylum-no-nodule.myc",
        "at least one `nodule`",
    ),
    ("11-matured-fn-retired.myc", "maturation is declared per"),
    (
        "12-runtime-vocab-reserved-not-active.myc",
        "reserved for the runtime model (RFC-0008), not yet active",
    ),
    (
        // M-666 / RFC-0008 §4.7 RT7: a `hypha` outside a `colony` is not expressible.
        "13-orphan-hypha.myc",
        "only valid inside a `colony",
    ),
    (
        // M-658 / RFC-0007 §12: `impl` is reserved (DN-03 §1), so it cannot be an identifier.
        "14-impl-reserved-ident.myc",
        "expected an identifier",
    ),
    (
        // M-659 / RFC-0019 §4.1: bounds belong only on function type-params; a bound on a `type`
        // (or `trait`) parameter is a never-silent parse refusal (the bound is never dropped — G2).
        "15-trait-param-bound.myc",
        "bounds on `type`/`trait` type-parameters are deferred",
    ),
    (
        // M-660 / RFC-0014 §4.5: a duplicate effect name in one `!{…}` annotation is a never-silent
        // PARSE refusal (the set is never silently de-duplicated — G2). The *coverage* refusal
        // (performing an undeclared effect) is a checker concern exercised in `tests/check.rs`, since
        // this gate runs only the parser; the fixture's banner documents that split.
        "17-duplicate-effect.myc",
        "duplicate effect",
    ),
    (
        // M-664 / DN-03 §1: `consume <expr>` is now an ACTIVE expression (affine `Substrate`
        // acquisition, LR-8). At *item* position it is rejected — it is an expression, not a
        // top-level item; the diagnostic points into a `fn` body (never a silent accept, G2).
        "18-consume-not-an-item.myc",
        "not a top-level item",
    ),
    (
        // M-812 / DN-38 §8.1: `grow` is superseded by `derive` — the grow→derive reconciliation.
        // Its diagnostic points at `derive Name for T` (the active form), naming DN-38 §8.1.
        // This is distinct from the DN-03 §1 "not yet active" message for `consume` (fixture 18).
        "19-grow-reserved-not-active.myc",
        "DN-38",
    ),
    (
        // M-750 / RFC-0032 D4: a `0x..` byte-string literal with an odd hex-digit count is a
        // never-silent lex refusal (a byte is two hex chars — never a silent half-byte).
        "20-odd-hex-bytes.myc",
        "odd hex-digit count",
    ),
    (
        // M-750 / RFC-0032 D4: an empty `0x` literal (no hex digit) is a never-silent lex refusal.
        "21-empty-hex-bytes.myc",
        "no hex digits",
    ),
    (
        // RFC-0037 D4: the return arrow `->` is retired in favour of `=>`. A leftover `->` (still
        // lexed) gets the explicit teaching reject, never a silent acceptance of the old glyph (G2).
        "22-old-arrow-retired.myc",
        "the arrow is now `=>`, not `->`",
    ),
    (
        // RFC-0037 D1: angle-bracket type params `fn f<T>(…)` are retired (use `fn f[T](…)`). With
        // `<` now operator-only, a `<` where `(` is expected is a never-silent parse refusal (G2).
        "23-old-fn-typeparam-retired.myc",
        "expected `(` to open the parameter list",
    ),
    (
        // RFC-0037 D1: angle-bracket type params on a `trait` (`trait T<A> { … }`) are retired (use
        // `trait T[A]`). A `<` where the trait body `{` is expected is a never-silent refusal (G2).
        "24-old-trait-typeparam-retired.myc",
        "expected `{` to open the trait body",
    ),
    (
        // RFC-0037 D4: the angle balanced-ternary literal `<+0->` is retired (use `0t+0-`). With `<`
        // operator-only, the old trit at expression position is a never-silent parse refusal (G2).
        "25-old-angle-trit-retired.myc",
        "expected an expression",
    ),
    (
        // DN-54 §3 / M-812: a `lower` declaration without `=` is a never-silent parse refusal (G2).
        // The parser expects `lower Name[params]? = <rhs>`.
        "26-lower-missing-eq.myc",
        "expected `=` after the rule name/params in a `lower` declaration",
    ),
    (
        // DN-54 §4 / M-812 / DN-38 §8.1: a `derive` application without `for` is a never-silent
        // parse refusal (G2). The parser expects `derive Name for T`.
        "27-derive-missing-for.myc",
        "expected `for` after the rule name in a `derive` application",
    ),
    (
        // DN-53 / M-811: an `object` body must start with a constructor clause; an empty body is a
        // never-silent parse refusal (DN-53 §A.3.1). The constructor provides the data form for the
        // `TypeDecl` the `object` desugars to; without it the desugar has no `Ctor` (never-silent, G2).
        // (Renumbered 26→28 at integration to avoid the 26-* number collision with DN-54's fixture.)
        "28-object-empty-body.myc",
        "must have at least one constructor clause",
    ),
    (
        // DN-57 §3 / M-818: the `;` component terminator is now MANDATORY. An item not terminated
        // by `;` (the first `fn` running into the second) is a never-silent parse refusal (G2). The
        // diagnostic names the unterminated component and points at where the `;` belongs.
        "29-missing-semicolon-terminator.myc",
        "expected `;` to terminate this item",
    ),
    (
        // RFC-0037 D2-b / DN-02 (M-915): `vec` was explicitly REJECTED as the short repr-keyword
        // alias for `VSA` (collides with `std.collections.Vec`) — it is never a keyword, so
        // `vec{...}` in type position parses `vec` as a bare identifier and then fails on the
        // unexpected `{` where the parameter list's closing `)` was expected (never a silent
        // accept, G2).
        "30-vec-short-alias-rejected.myc",
        "expected `)` to close the parameter list",
    ),
    (
        // M-916 (RFC-0025 §4.2, resolved by RFC-0037 D1): `<=`/`>=` are retired glyphs — no such
        // token exists, so `a <= b` lexes as `LAngle`, `Eq` and the parser reads `a < (= b)`,
        // failing on the `=` where a right-hand-side expression was expected (never a silent
        // reinterpretation of the old two-char glyph, G2). The word forms `lte`/`gte` are the only
        // valid spelling (see accept/20-operator-syntax.myc).
        "31-old-le-ge-glyph-retired.myc",
        "expected an expression",
    ),
];

#[test]
fn reject_corpus_all_fails_explicitly() {
    for path in myc_files("reject") {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("fixture has a UTF-8 name");
        let src = fs::read_to_string(&path).unwrap();
        // A reject fixture must fail — and fail as an explicit ParseError, not a panic (the call
        // returning at all proves no panic; the `Err` arm proves no silent accept).
        let err = match parse_phylum(&src) {
            Ok(_) => panic!(
                "{} should be rejected but parsed successfully",
                path.display()
            ),
            Err(e) => e,
        };
        // …and it must fail for the *intended* reason: a new fixture with no entry is a hard
        // failure (the gate can't grow blind spots), and a fixture failing for an unexpected
        // reason is caught instead of silently passing on `is_err()` alone.
        let Some((_, expected)) = REJECT_EXPECTED.iter().find(|(f, _)| *f == name) else {
            panic!(
                "{name} has no expected-error entry in REJECT_EXPECTED — every reject fixture must \
                 declare the distinctive fragment its rejection message must contain (A4)"
            );
        };
        let msg = err.to_string();
        assert!(
            msg.contains(expected),
            "{name} rejected for an unexpected reason:\n  expected message to contain: {expected:?}\n  \
             actual message: {msg:?}\n(if the fixture or diagnostic legitimately changed, update \
             REJECT_EXPECTED)"
        );
    }
}

/// The accept/reject split is meaningful: at least one fixture in each bucket, and the buckets are
/// disjoint in outcome (guards against a vacuous gate).
#[test]
fn the_gate_is_non_vacuous() {
    assert!(!myc_files("accept").is_empty());
    assert!(!myc_files("reject").is_empty());
}

/// Every entry in [`REJECT_EXPECTED`] must correspond to an actual fixture file in the reject
/// corpus — an orphaned entry (pointing at a deleted or renamed fixture) would silently pass
/// while exercising nothing, creating a blind spot the gate is supposed to close. Mutant-witness:
/// adding a `REJECT_EXPECTED` entry for a non-existent file trips this test, keeping the table
/// and the corpus in sync **in both directions** (A4 bidirectional integrity).
#[test]
fn reject_expected_table_has_no_orphaned_entries() {
    let existing: std::collections::BTreeSet<String> = myc_files("reject")
        .into_iter()
        .map(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .expect("fixture has a UTF-8 name")
                .to_owned()
        })
        .collect();
    for (name, _) in REJECT_EXPECTED {
        assert!(
            existing.contains(*name),
            "REJECT_EXPECTED entry {name:?} has no corresponding reject fixture — \
             either add the fixture or remove the orphaned entry (A4 bidirectional integrity)"
        );
    }
}
