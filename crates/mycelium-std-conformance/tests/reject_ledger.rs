//! The **reject-ledger regression guard** (DN-56 condition 1 / DN-80 / M-959, kickoff `frz` Lane A).
//!
//! DN-80 (`docs/notes/DN-80-Reject-Ledger-Exhaustive-Never-Silent-Refusal-Inventory.md`) is the
//! exhaustive, `{construct, reason, alternative}` ledger of every construct the kernel *rejects* —
//! across the parse (`ParseError`), check (`CheckError`/`AmbientError`), and runtime/kernel
//! (`EvalError`/`WfError`) strata. A ledger with no regression guard can silently fall behind the
//! moment a new reject path lands, so this test **audits the exact counts/variant-sets DN-80 §4-§7
//! cite** and fails, never-silently, the moment source drifts from what the ledger describes.
//!
//! # Why this lives here, not in `mycelium-l1`
//! `crates/mycelium-l1` (the frontend: `token`/`lexer`/`parse`/`checkty`) and
//! `crates/mycelium-interp/src/prims.rs` are **read-only** for this task (a concurrent M-915 leaf
//! owns the L1 frontend). This crate (`mycelium-std-conformance`) is a pre-existing **test-only**
//! conformance crate that already dev-depends on `mycelium-l1`/`mycelium-interp`/`mycelium-core` for
//! its three-way differentials (RFC-0031 D5/D6) — a natural, non-frontend home for a cross-crate
//! audit. This test touches **no** frontend file: it only *reads* the audited files as plain text
//! (`std::fs::read_to_string`), so it needs no new dependency and edits nothing read-only.
//!
//! # Honesty (`Empirical`, VR-5 — never `Proven`)
//! The check-level assertions (§2 below) are a **line/regex heuristic over source text** — the same
//! posture `docs/api-index/` states for itself ("source is ground truth"). They prove the *audited
//! call-site counts* still match DN-80 §4's table; they do **not** prove semantic completeness (that
//! no other reject path could exist) nor localize *which* site moved within a family — see DN-80 §8
//! for the full grounded discussion of this limitation. The closed-enum assertions (§3) are strictly
//! stronger: a `BTreeSet` equality over extracted variant names catches any addition, removal, *or
//! rename* exactly, since `AmbientError`/`EvalError`/`WfError` are genuinely closed Rust enums.
//!
//! If this test fails, the fix is: (1) read the failure message (it names the file/pattern/count or
//! enum/variant that drifted), (2) add/update the corresponding row(s) in DN-80, (3) update the
//! pinned constant/set below to match the new audited reality, in the same commit.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// The repo root, resolved from this crate's manifest dir (two levels up: `crates/<this>/..`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

fn read(rel_path: &str) -> String {
    let p = repo_root().join(rel_path);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// §1 — Parse-level reject corpus (DN-80 Part A): exactly the 30 fixtures the ledger names.
// ════════════════════════════════════════════════════════════════════════════════════════════

/// The 30 fixture names DN-80 §3 ledgers (note: no `16-*` — a documented numbering gap, not a
/// hidden reject; see DN-80 row 16).
const LEDGERED_REJECT_FIXTURES: &[&str] = &[
    "01-no-nodule-header.myc",
    "02-swap-missing-policy.myc",
    "03-unclosed-brace.myc",
    "04-bad-trit.myc",
    "05-reserved-word-ident.myc",
    "06-missing-arrow.myc",
    "07-empty.myc",
    "08-imperative-while.myc",
    "09-default-missing-paradigm.myc",
    "10-phylum-no-nodule.myc",
    "11-matured-fn-retired.myc",
    "12-runtime-vocab-reserved-not-active.myc",
    "13-orphan-hypha.myc",
    "14-impl-reserved-ident.myc",
    "15-trait-param-bound.myc",
    "17-duplicate-effect.myc",
    "18-consume-not-an-item.myc",
    "19-grow-reserved-not-active.myc",
    "20-odd-hex-bytes.myc",
    "21-empty-hex-bytes.myc",
    "22-old-arrow-retired.myc",
    "23-old-fn-typeparam-retired.myc",
    "24-old-trait-typeparam-retired.myc",
    "25-old-angle-trit-retired.myc",
    "26-lower-missing-eq.myc",
    "27-derive-missing-for.myc",
    "28-object-empty-body.myc",
    "29-missing-semicolon-terminator.myc",
    "30-vec-short-alias-rejected.myc",
    "31-old-le-ge-glyph-retired.myc",
];

#[test]
fn parse_level_reject_corpus_matches_the_ledger() {
    let dir = repo_root().join("docs/spec/grammar/conformance/reject");
    let mut actual: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "myc"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let ledgered: BTreeSet<String> = LEDGERED_REJECT_FIXTURES
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    let missing_from_ledger: Vec<_> = actual.difference(&ledgered).collect();
    assert!(
        missing_from_ledger.is_empty(),
        "reject fixture(s) exist with no DN-80 §3 ledger row: {missing_from_ledger:?} — \
         add a {{construct, reason, alternative}} row to DN-80 before this can pass (G2: a reject \
         path may not go unledgered)"
    );

    let orphaned_in_ledger: Vec<_> = ledgered.difference(&actual).collect();
    assert!(
        orphaned_in_ledger.is_empty(),
        "DN-80 §3 ledgers fixture(s) that no longer exist: {orphaned_in_ledger:?} — \
         update DN-80 to match the current corpus"
    );

    // Belt: keep this test's own list honest against a raw count too (30, DN-80 §3/§8.1).
    actual.retain(|_| true);
    assert_eq!(
        actual.len(),
        30,
        "expected exactly 30 reject fixtures (DN-80 §3); found {} — update DN-80 and this list",
        actual.len()
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// §2 — Check-level reject call-site counts (DN-80 Part B): a line/regex heuristic over source
// text, pinned to the exact counts DN-80 §4 audited at `dev` tip `ca42fd2` (2026-07-02).
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Count non-overlapping occurrences of `pattern` in `haystack` — the same thing `grep -c` counts
/// per line, applied over the whole file (every call site in this audit's patterns appears at most
/// once per source line, so this agrees with the `grep -c` figures DN-80 §4 cites).
fn count_occurrences(haystack: &str, pattern: &str) -> usize {
    haystack.matches(pattern).count()
}

#[test]
fn checkty_direct_checkerror_construction_count_matches_the_ledger() {
    let src = read("crates/mycelium-l1/src/checkty.rs");
    let direct_new = count_occurrences(&src, "CheckError::new(");
    let direct_at = count_occurrences(&src, "CheckError::at(");
    let total = direct_new + direct_at;
    assert_eq!(
        total, 103,
        "checkty.rs direct `CheckError::new(`/`CheckError::at(` construction sites: found {total}, \
         DN-80 §4 audited 103 (dev 7b933a3, 2026-07-02; +6 vs the original ca42fd2 audit — the \
         M-919/M-973 lower/derive extension-checker work, family 8; +2 — M-965's two `Fuse` \
         built-in-prelude redeclaration refusals, family 5; +1 — M-966's `via`-delegation \
         ambiguity refusal, family 6; +1 — RFC-0041 W1 (M-979, 2026-07-03) the recursion-depth \
         `BudgetError` → `CheckError` refusal mapping, family: resource-exhaustion) — a reject path \
         was added or removed without updating DN-80 §4's construct-family table and this pinned \
         count (one of these is the shared `Cx::err` helper's own body at line ~3000 — plumbing, not \
         a distinct construct; see DN-80 §4's audited-totals note)"
    );
}

#[test]
fn fuse_law_checker_checkerror_construction_count_matches_the_ledger() {
    // fuse.rs (M-965, DN-58 §A) is the `Fuse` semilattice-**law** checker — a new audited reject
    // file (DN-80 §4 row 40). Its four `CheckError::new(` sites are the idempotence /
    // commutativity / associativity violations plus the probe-time eval-failure refusal.
    let src = read("crates/mycelium-l1/src/fuse.rs");
    let total =
        count_occurrences(&src, "CheckError::new(") + count_occurrences(&src, "CheckError::at(");
    assert_eq!(
        total, 4,
        "fuse.rs `CheckError::new(`/`CheckError::at(` construction sites: found {total}, \
         DN-80 §4 audited 4 (dev 4e2c389, 2026-07-02 — the Fuse semilattice-law reject family, \
         DN-80 §4 row 40) — a law-reject path was added or removed without updating the ledger and \
         this pinned count together"
    );
}

#[test]
fn checkty_self_err_call_count_matches_the_ledger() {
    let src = read("crates/mycelium-l1/src/checkty.rs");
    let total = count_occurrences(&src, "self.err(");
    assert_eq!(
        total, 116,
        "checkty.rs `self.err(` call sites: found {total}, DN-80 §4 audited 116 (dev ca42fd2, \
         2026-07-02; +1 — RFC-0041 W1 (M-979, 2026-07-03) recursion-depth `BudgetError` refusal; \
         +5 — trx2 Lane C CU-3 (2026-07-08) the never-silent Binary↔Float conversion prims \
         `bin_to_flt`/`flt_to_bin` — arity, operand-type, and DN-41 width-witness refusals, \
         ADR-040 §2.4) — a reject path was added or removed without updating DN-80 §4's \
         construct-family table and this pinned count"
    );
}

#[test]
fn grade_checkerror_construction_count_matches_the_ledger() {
    let src = read("crates/mycelium-l1/src/grade.rs");
    let total =
        count_occurrences(&src, "CheckError::at(") + count_occurrences(&src, "CheckError::new(");
    assert_eq!(
        total, 3,
        "grade.rs `CheckError::at(`/`CheckError::new(` construction sites: found {total}, \
         DN-80 §4 audited 3 (dev ca42fd2, 2026-07-02; +1 — RFC-0041 W1 (M-979, 2026-07-03) the \
         grade-pass recursion-depth `BudgetError` → `CheckError` refusal) — the guarantee/grade-lattice \
         reject family (DN-80 §4 row 39) drifted; update the ledger and this pinned count together"
    );
}

// ════════════════════════════════════════════════════════════════════════════════════════════
// §3 — Closed-enum reject surfaces (DN-80 Parts C/D/E): exact variant-name-set equality. This is
// the strong half of the guard — any added/removed/renamed variant fails.
// ════════════════════════════════════════════════════════════════════════════════════════════

/// Strip `//`-style line comments (covers `///` doc comments too) from `src`, so brace/paren depth
/// tracking below only ever sees code, never comment prose that might itself contain punctuation.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| match line.find("//") {
            Some(idx) => &line[..idx],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract the top-level variant names of `pub enum <enum_name> { … }` from `src` (comments already
/// stripped by the caller). Depth-tracks `{}`/`()` so struct-like and tuple-like variant payloads
/// don't confuse the top-level comma split; each variant's name is the leading identifier of its
/// chunk.
fn extract_enum_variants(src_no_comments: &str, enum_name: &str) -> Vec<String> {
    let needle = format!("enum {enum_name}");
    let start = src_no_comments
        .find(&needle)
        .unwrap_or_else(|| panic!("`enum {enum_name}` not found in source"));
    let after = &src_no_comments[start..];
    let open = after
        .find('{')
        .unwrap_or_else(|| panic!("no `{{` found after `enum {enum_name}`"));
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut body_end = None;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    body_end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let body_end = body_end.unwrap_or_else(|| panic!("unbalanced braces in `enum {enum_name}`"));
    let body = &after[open + 1..body_end];

    // Split the body on TOP-LEVEL commas (depth 0 relative to the body, tracking both {} and ()).
    let mut variants = Vec::new();
    let mut depth = 0i32;
    let mut chunk_start = 0usize;
    let body_bytes = body.as_bytes();
    for (i, &b) in body_bytes.iter().enumerate() {
        match b {
            b'{' | b'(' => depth += 1,
            b'}' | b')' => depth -= 1,
            b',' if depth == 0 => {
                push_variant_name(&body[chunk_start..i], &mut variants);
                chunk_start = i + 1;
            }
            _ => {}
        }
    }
    // Trailing chunk (the enum may or may not end with a trailing comma).
    push_variant_name(&body[chunk_start..], &mut variants);
    variants
}

fn push_variant_name(chunk: &str, out: &mut Vec<String>) {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return;
    }
    let name: String = trimmed
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if !name.is_empty() {
        out.push(name);
    }
}

fn assert_variant_set(src_path: &str, enum_name: &str, expected: &[&str]) {
    let src = read(src_path);
    let no_comments = strip_line_comments(&src);
    let actual: BTreeSet<String> = extract_enum_variants(&no_comments, enum_name)
        .into_iter()
        .collect();
    let expected_set: BTreeSet<String> = expected.iter().map(|s| (*s).to_owned()).collect();

    let added: Vec<_> = actual.difference(&expected_set).collect();
    let removed: Vec<_> = expected_set.difference(&actual).collect();
    assert!(
        added.is_empty() && removed.is_empty(),
        "`{enum_name}` in {src_path} has drifted from DN-80's ledgered variant set: \
         added (in source, not in DN-80) = {added:?}, removed (in DN-80, not in source) = \
         {removed:?} — update DN-80's variant table (and this test's `expected` list) together \
         (G2: a reject variant may not go unledgered)"
    );
}

#[test]
fn ambient_error_variants_match_the_ledger() {
    assert_variant_set(
        "crates/mycelium-l1/src/ambient.rs",
        "AmbientError",
        &[
            "MultipleDefaults",
            "UnresolvedAmbient",
            "ParadigmShapeMismatch",
            "BareDecimalNoEncoding",
            "DepthExceeded",
        ],
    );
}

#[test]
fn eval_error_variants_match_the_ledger() {
    assert_variant_set(
        "crates/mycelium-interp/src/lib.rs",
        "EvalError",
        &[
            "FreeVariable",
            "UnknownPrim",
            "PrimType",
            "ApproxCompositionUnsupported",
            "UnsupportedSwap",
            "Overflow",
            "FuelExhausted",
            "DepthLimit",
            "EffectBudget",
            "Swap",
            "Wf",
            "NonExhaustiveMatch",
            "DataMalformed",
            "GuaranteeMeetUnsupported",
            "DataResult",
            "ApplyNonFunction",
            "FunctionResult",
        ],
    );
}

#[test]
fn wf_error_variants_match_the_ledger() {
    assert_variant_set(
        "crates/mycelium-core/src/lib.rs",
        "WfError",
        &[
            "GuaranteeBoundMismatch",
            "MalformedBound",
            "MalformedRepr",
            "DimensionTooLarge",
            "PayloadReprMismatch",
            "MalformedReconstruction",
            "MalformedSparsity",
        ],
    );
}

/// The gate is non-vacuous: each of the three closed enums yields a non-empty variant set (guards
/// against `extract_enum_variants` silently returning nothing on a parsing regression, which would
/// make the equality checks above vacuously pass on an empty actual/expected pair for any type that
/// unexpectedly matched an unrelated `enum` snippet first).
#[test]
fn variant_extraction_is_non_vacuous() {
    for (path, name) in [
        ("crates/mycelium-l1/src/ambient.rs", "AmbientError"),
        ("crates/mycelium-interp/src/lib.rs", "EvalError"),
        ("crates/mycelium-core/src/lib.rs", "WfError"),
    ] {
        let src = read(path);
        let no_comments = strip_line_comments(&src);
        let variants = extract_enum_variants(&no_comments, name);
        assert!(
            !variants.is_empty(),
            "extracted zero variants for `{name}` in {path} — the extractor likely regressed \
             (unbalanced braces, or the enum was renamed/moved)"
        );
    }
}
