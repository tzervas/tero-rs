//! Differential tests for `std.testing` (M-932, E29-1, kickoff `opp`) — the structural testing
//! surface whose honesty crux (C1/G2) must survive translation: a skipped or undetermined check
//! is a first-class reported verdict, never a silent pass.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-op three-way cases, and — the row this port owns
//! per the harness doc (§4) — the live comparisons against the **retained Rust oracle**,
//! `mycelium-std-testing` (RFC-0031 D6; the crate is NOT retired). Scalar oracle cases reduce
//! both sides to `u64`/`u8` bit-folds ([`eval_u64`]/[`eval_u8`], the `std_error.rs::eval_byte`
//! precedent widened to 64 bits); matrix oracle cases compare **live against
//! `mycelium_std_testing::guarantee_matrix::MATRIX`** (the `std_diag.rs`/`std_recover.rs`
//! precedent) so a real Rust↔`.myc` divergence flips the oracle and fails; the mode-resolver
//! cases build their expected `.myc` value **live from the Rust resolver's output** (the
//! FLAG-testing-7 anti-drift pin).
//!
//! # Surface-check (D5 row 1) and substitutions
//! See `lib/std/testing.myc`'s header comment for the full surface-check. PORTED: the verdict
//! sums + the honest aggregator, Budget, the Xorshift64 Rng as pure state threading (bit-exact),
//! for_all + the bounded shrink loop, golden (closed name code + width-generic snapshot),
//! differential (thunked width-generic backends), the M-796 ModeScope/ModeTestConfig/ModeVisit
//! surface with a reimplemented most-specific-wins resolver, the reified violation cores of the
//! two panic asserts, and the 5-row guarantee matrix as checked data. FLAGGED, not forced
//! (VR-5/G2): the `Gen<T>` trait *as a trait* (fn-pair substitution), `next_usize_below`
//! (platform-width `usize`), Debug-formatted descriptions + `make_diff` (typed reified payloads
//! instead), open-String golden names (no `bytes_eq` prim), `FailRecord::to_diag` (kernel `Diag`,
//! D1 boundary), and the panic-based `assert_mode_scope`/`assert_mode_negative` wrappers (no host
//! refusal primitive — FLAG-error-1 precedent).
//!
//! # Self-application smoke (the M-932 DoD extra)
//! [`self_application_shl_conformance_case_is_green`] expresses an EXISTING conformance case —
//! the `enablement.rs` RFC-0033 Gap-B worked example `shl_u(1, 3) = 8` — through the ported
//! harness itself: the `.myc` `differential` op checks the case, the `.myc` `summarize`/
//! `is_green` aggregate it, and the three-way differential proves the ported harness reaches the
//! same verdict on every execution path. Its negative twin proves a wrong expectation is a
//! `Fail`, never a silent pass.
//!
//! # Honesty tags
//! - **`Declared`** — each ported op's type-level contract and the 5-row matrix transcription,
//!   carried at the SAME strength as `mycelium-std-testing`'s own guarantee matrix (VR-5: never
//!   upgraded in translation; the harness ops are Exact *mechanisms* there and stay so here).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) AND the
//!   Rust-oracle differential below, both validated by trial on the programs in this file;
//!   neither is a machine-checked proof.

mod harness;

use mycelium_core::{CoreValue, Payload};
use mycelium_std_testing::cert_mode_test::{CertDecl, CertScope, ModeTestConfig};
use mycelium_std_testing::guarantee_matrix::MATRIX;
use mycelium_std_testing::{
    differential, for_all, golden, is_green, summarize, Budget, Gen, GoldenBaseline, Rng, Verdict,
};

/// The std.testing nodule source, loaded at compile time — the single source of truth.
const TESTING_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/testing.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(TESTING_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`] (same pattern as `std_error.rs`).
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Expected-side type mirrors — constructor order matches `lib/std/testing.myc` exactly
// (structural identity for the CoreValue comparison; the `std_recover.rs` precedent).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The verdict sum + its component types (enough for any `Verdict[..]`-valued expected `main`).
const T_VERDICT: &str = "nodule ref;\n\
type SkipReason = Ignored | UnmetPrecondition | NeedsRecord | BackendUnavailable | ToolMissing;\n\
type UndetReason = OracleUnavailable | BudgetExhaustedInconclusive | NonDeterministicInput;\n\
type FailRecord[F] = FRec(F, Binary{64}, Binary{8}, Bytes);\n\
type Verdict[V] = Pass | Fail(FailRecord[V]) | Skipped(SkipReason) | Undetermined(UndetReason);\n";

/// `GoldenFail` / `DiffFail` payload mirrors (appended to [`T_VERDICT`] where needed).
const T_GOLDEN_FAIL: &str = "type GoldenFail[W] = GFail(Binary{8}, W, W);\n";
const T_DIFF_FAIL: &str = "type DiffFail[D] = DFail(D, D);\n";

/// The aggregate mirror.
const T_SUMMARY: &str =
    "nodule ref;\ntype Summary = Counts(Binary{8}, Binary{8}, Binary{8}, Binary{8});\n";

/// Budget + Option mirrors.
const T_BUDGET: &str =
    "nodule ref;\ntype Option[Opt] = Some(Opt) | None;\ntype Budget = Trials(Binary{8});\n";

/// The mode-parametric vocabulary mirrors.
const T_MODE: &str = "nodule ref;\n\
type Option[Opt] = Some(Opt) | None;\n\
type CertMode = Fast | Balanced | Certified;\n\
type CertScope = ScGlobal | ScPhylum | ScNodule;\n\
type ResolvedMode = RMode(CertMode, Option[CertScope]);\n\
type ModeScope = MScope(Bool, Bool, Bool);\n\
type ModeViolation = NegativeFires(CertMode) | PositiveAbsent(CertMode);\n";

/// A plain-Bool expected program.
fn expect_bool(b: bool) -> String {
    format!(
        "nodule ref;\nfn main() => Bool = {};",
        if b { "True" } else { "False" }
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Literal builders + L1-eval extraction helpers (the `std_error.rs::eval_byte` precedent,
// widened for Binary{64}/Binary{32}/Bytes results).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A `Binary{64}` literal for `n` (64 digits, MSB-first).
fn lit64(n: u64) -> String {
    format!("0b{n:064b}")
}

/// A `Binary{8}` literal for `n`.
fn lit8(n: u8) -> String {
    format!("0b{n:08b}")
}

/// Run `driver`'s `main` through the L1 evaluator (parse → check → monomorphize → eval → to_core)
/// and return the CoreValue. Reuses the same path as [`harness::assert_three_way`]; used only for
/// bridging to the Rust oracle (the three-way obligation is covered by the cases above).
fn eval_core(driver: &str) -> CoreValue {
    use mycelium_l1::elab::build_registry;
    use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

    let src = program(driver);
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("parse failed: {e}")))
        .unwrap_or_else(|e| panic!("check failed: {e}"));
    let mono = monomorphize(&env, "main").unwrap_or_else(|e| panic!("monomorphize failed: {e}"));
    let registry = build_registry(&mono).unwrap_or_else(|e| panic!("build_registry failed: {e}"));
    let val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("L1-eval failed: {e}"));
    val.to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("result is outside the r3 data fragment"))
}

/// Fold an MSB-first bit payload into a `u64` (bit-exact; no sign extension — the unsigned
/// reading `lt`/the Xorshift state use).
fn bits_to_u64(cv: &CoreValue) -> u64 {
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Binary repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bits(bits) => bits.iter().fold(0u64, |acc, &b| (acc << 1) | u64::from(b)),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Evaluate a `Binary{64}`-valued `main` to a `u64`.
fn eval_u64(driver: &str) -> u64 {
    bits_to_u64(&eval_core(driver))
}

/// Evaluate a `Binary{8}`-valued `main` to a `u8`.
fn eval_u8(driver: &str) -> u8 {
    let v = bits_to_u64(&eval_core(driver));
    u8::try_from(v).unwrap_or_else(|_| panic!("expected an 8-bit value, got {v}"))
}

/// Evaluate a `Binary{32}`-valued `main` to a `u32`.
fn eval_u32(driver: &str) -> u32 {
    let v = bits_to_u64(&eval_core(driver));
    u32::try_from(v).unwrap_or_else(|_| panic!("expected a 32-bit value, got {v}"))
}

/// Evaluate a `Bytes`-valued `main` to its byte content.
fn eval_bytes(driver: &str) -> Vec<u8> {
    let cv = eval_core(driver);
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Bytes repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bytes(bytes) => bytes.clone(),
        other => panic!("expected a Bytes payload, got {other:?}"),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Three-way differential cases (L1-eval ≡ elaborate→L0-interp ≡ AOT), one section per ported op.
// Each against a hand-computed reference (Declared data; agreement Empirical). Expected sides
// replicate the SAME primitive-op composition as the path under test (provenance is part of
// value identity — Derived vs Root).
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ── the honest aggregator (C1/G2 — skips aggregate DISTINCTLY from passes) ──────────────────────

/// `summarize` keeps all four verdict classes distinct — one of each ⇒ `Counts(1,1,1,1)`, each
/// count derived by exactly one `add_u` bump over the zero base.
#[test]
fn summarize_counts_each_class_distinct() {
    let driver = "fn one_of_each() => Vec[Verdict[Unit]] = Cons(Pass, Cons(Fail(FRec(U, zero64(), 0b0000_0000, \"x\")), Cons(Skipped(Ignored), Cons(Undetermined(OracleUnavailable), Nil))));\nfn main() => Summary = summarize(one_of_each());";
    let src = program(driver);
    let expected = format!(
        "{T_SUMMARY}fn main() => Summary = Counts(add_u(0b0000_0000, 0b0000_0001), add_u(0b0000_0000, 0b0000_0001), add_u(0b0000_0000, 0b0000_0001), add_u(0b0000_0000, 0b0000_0001));"
    );
    assert_three_way("summarize one-of-each", &src, &expected);
}

/// A suite with skips but no failures IS green — the skip is SURFACED in the Summary, not
/// treated as a failure (treating "could not run" as "failed" would itself violate C1 — the
/// Rust doc's own reasoning, carried over).
#[test]
fn is_green_true_with_skips_surfaced() {
    let driver = "fn suite() => Vec[Verdict[Unit]] = Cons(Pass, Cons(Skipped(NeedsRecord), Nil));\nfn main() => Bool = is_green(summarize(suite()));";
    let src = program(driver);
    assert_three_way("is_green with surfaced skip", &src, &expect_bool(true));
}

/// One failure makes the suite not-green.
#[test]
fn is_green_false_on_any_failure() {
    let driver = "fn suite() => Vec[Verdict[Unit]] = Cons(Pass, Cons(Fail(FRec(U, zero64(), 0b0000_0000, \"x\")), Nil));\nfn main() => Bool = is_green(summarize(suite()));";
    let src = program(driver);
    assert_three_way("is_green with a failure", &src, &expect_bool(false));
}

// ── budget (C6 — declared, bounded; 0 is an explicit None) ──────────────────────────────────────

/// `budget_new(0)` is an explicit `None` — a zero-trial budget is refused, never silently
/// accepted (C1).
#[test]
fn budget_new_zero_is_none() {
    let driver = "fn main() => Option[Budget] = budget_new(0b0000_0000);";
    let src = program(driver);
    let expected = format!("{T_BUDGET}fn main() => Option[Budget] = None;");
    assert_three_way("budget_new(0)", &src, &expected);
}

/// `budget_new(5)` wraps the count.
#[test]
fn budget_new_five_is_some() {
    let driver = "fn main() => Option[Budget] = budget_new(0b0000_0101);";
    let src = program(driver);
    let expected = format!("{T_BUDGET}fn main() => Option[Budget] = Some(Trials(0b0000_0101));");
    assert_three_way("budget_new(5)", &src, &expected);
}

/// `budget_default` carries 100 trials (Budget::DEFAULT).
#[test]
fn budget_default_is_one_hundred_trials() {
    let driver = "fn main() => Binary{8} = budget_trials(budget_default());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0110_0100;";
    assert_three_way("budget_default trials", &src, expected);
}

// ── Rng (RT3/C6 — seeded, deterministic, zero-seed promoted) ────────────────────────────────────

/// A zero seed is promoted to the non-degenerate default constant (0xDEAD_BEEF_CAFE_1337).
#[test]
fn rng_new_zero_seed_promotes() {
    let driver = format!("fn main() => Binary{{64}} = rng_new({});", lit64(0));
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\nfn main() => Binary{{64}} = {};",
        lit64(0xDEAD_BEEF_CAFE_1337)
    );
    assert_three_way("rng_new(0) promotion", &src, &expected);
}

/// A non-zero seed passes through unchanged.
#[test]
fn rng_new_nonzero_seed_passes_through() {
    let driver = format!("fn main() => Binary{{64}} = rng_new({});", lit64(42));
    let src = program(&driver);
    let expected = format!("nodule ref;\nfn main() => Binary{{64}} = {};", lit64(42));
    assert_three_way("rng_new(42) pass-through", &src, &expected);
}

/// One Xorshift64 step from seed 1, three-way — the expected side replicates the same
/// xor/shl_u/shr_u composition (same ops, same shift constants).
#[test]
fn rng_next_xorshift_step_three_way() {
    let driver = format!(
        "fn main() => Binary{{64}} = rng_next(rng_new({}));",
        lit64(1)
    );
    let src = program(&driver);
    let expected = format!(
        "nodule ref;\nfn main() => Binary{{64}} =\n  let a = xor({one}, shl_u({one}, {c13})) in\n  let b = xor(a, shr_u(a, {c7})) in\n  xor(b, shl_u(b, {c17}));",
        one = lit64(1),
        c13 = lit64(13),
        c7 = lit64(7),
        c17 = lit64(17),
    );
    assert_three_way("rng_next(1) xorshift step", &src, &expected);
}

// ── for_all (spec §3; the C2 crux: a pass backs Empirical, never Proven) ────────────────────────

/// Shared generator/property driver fragments for the for_all cases.
const GEN_PRELUDE: &str = "fn gen_state(s: Binary{64}) => GenStep[Binary{64}] = Stepped(Some(rng_next(s)), rng_next(s));\n\
fn no_shrink64(_x: Binary{64}) => Vec[Binary{64}] = Nil;\n\
fn prop_true64(_x: Binary{64}) => Bool = True;\n\
fn gen_none64(s: Binary{64}) => GenStep[Binary{64}] = Stepped(None, s);\n";

/// All trials pass ⇒ `Pass`.
#[test]
fn for_all_all_trials_pass() {
    let driver = format!(
        "{GEN_PRELUDE}fn main() => Verdict[Binary{{64}}] = for_all(gen_state, no_shrink64, {}, Trials(0b0000_0011), prop_true64);",
        lit64(7)
    );
    let src = program(&driver);
    let expected = format!("{T_VERDICT}fn main() => Verdict[Binary{{64}}] = Pass;");
    assert_three_way("for_all all-pass", &src, &expected);
}

/// A generator that never produces ⇒ `Skipped(NeedsRecord)` — a property that could not run is
/// REPORTED, never a silent pass (the C1 crux).
#[test]
fn for_all_empty_generator_is_skipped() {
    let driver = format!(
        "{GEN_PRELUDE}fn main() => Verdict[Binary{{64}}] = for_all(gen_none64, no_shrink64, {}, Trials(0b0000_0011), prop_true64);",
        lit64(7)
    );
    let src = program(&driver);
    let expected = format!("{T_VERDICT}fn main() => Verdict[Binary{{64}}] = Skipped(NeedsRecord);");
    assert_three_way("for_all empty generator", &src, &expected);
}

/// A violated property with no shrink candidates fails at trial 0 with the REIFIED
/// counterexample, the reproducing seed, and the op context (the C3 EXPLAIN artifact in value
/// form — FLAG-testing-3).
#[test]
fn for_all_failure_carries_reified_counterexample() {
    let driver = format!(
        "fn gen_five(s: Binary{{64}}) => GenStep[Binary{{8}}] = Stepped(Some(0b0000_0101), s);\nfn no_shrink8(_x: Binary{{8}}) => Vec[Binary{{8}}] = Nil;\nfn prop_lt4(x: Binary{{8}}) => Bool = match lt(x, 0b0000_0100) {{ 0b1 => True, _ => False }};\nfn main() => Verdict[Binary{{8}}] = for_all(gen_five, no_shrink8, {}, Trials(0b0000_0011), prop_lt4);",
        lit64(2)
    );
    let src = program(&driver);
    let expected = format!(
        "{T_VERDICT}fn main() => Verdict[Binary{{8}}] = Fail(FRec(0b0000_0101, {}, 0b0000_0000, \"for_all property violated\"));",
        lit64(2)
    );
    assert_three_way("for_all fail, no shrink", &src, &expected);
}

/// The shrink loop descends to the MINIMAL still-failing value: constant 8, property `x < 3`,
/// decrement-shrinker ⇒ counterexample 3 (2 passes the property, so the descent stops).
#[test]
fn for_all_shrinks_to_minimal_counterexample() {
    let driver = format!(
        "fn gen_eight(s: Binary{{64}}) => GenStep[Binary{{8}}] = Stepped(Some(0b0000_1000), s);\nfn shrink_dec(x: Binary{{8}}) => Vec[Binary{{8}}] = match eq(x, 0b0000_0000) {{ 0b1 => Nil, _ => Cons(sub_u(x, 0b0000_0001), Nil) }};\nfn prop_lt3(x: Binary{{8}}) => Bool = match lt(x, 0b0000_0011) {{ 0b1 => True, _ => False }};\nfn main() => Verdict[Binary{{8}}] = for_all(gen_eight, shrink_dec, {}, Trials(0b0000_0011), prop_lt3);",
        lit64(2)
    );
    let src = program(&driver);
    // 8 → 7 → 6 → 5 → 4 → 3 (all still fail `x < 3`); shrink(3) = [2], 2 passes ⇒ minimal is 3,
    // derived by five sub_u decrements from the generated 8.
    let expected = format!(
        "{T_VERDICT}fn main() => Verdict[Binary{{8}}] = Fail(FRec(sub_u(sub_u(sub_u(sub_u(sub_u(0b0000_1000, 0b0000_0001), 0b0000_0001), 0b0000_0001), 0b0000_0001), 0b0000_0001), {}, 0b0000_0000, \"for_all property violated\"));",
        lit64(2)
    );
    assert_three_way("for_all shrink-to-minimal", &src, &expected);
}

// ── golden (C1: a missing baseline is NEVER auto-accepted) ──────────────────────────────────────

/// A missing baseline is `Skipped(NeedsRecord)` — the golden-test honesty crux.
#[test]
fn golden_missing_baseline_is_skipped() {
    let driver = "fn no_baseline() => Option[GoldenBaseline[Binary{8}]] = None;\nfn main() => Verdict[GoldenFail[Binary{8}]] = golden(no_baseline(), 0b0000_0001, 0b1010_1011);";
    let src = program(driver);
    let expected =
        format!("{T_VERDICT}{T_GOLDEN_FAIL}fn main() => Verdict[GoldenFail[Binary{{8}}]] = Skipped(NeedsRecord);");
    assert_three_way("golden missing baseline", &src, &expected);
}

/// A baseline under the WRONG name code is treated as missing (the caller supplied the wrong
/// baseline) — `Skipped(NeedsRecord)`, never a cross-name comparison.
#[test]
fn golden_name_mismatch_is_skipped() {
    let driver = "fn baseline() => Option[GoldenBaseline[Binary{8}]] = Some(GB(0b0000_0010, 0b1010_1011));\nfn main() => Verdict[GoldenFail[Binary{8}]] = golden(baseline(), 0b0000_0001, 0b1010_1011);";
    let src = program(driver);
    let expected =
        format!("{T_VERDICT}{T_GOLDEN_FAIL}fn main() => Verdict[GoldenFail[Binary{{8}}]] = Skipped(NeedsRecord);");
    assert_three_way("golden name mismatch", &src, &expected);
}

/// A matching baseline passes.
#[test]
fn golden_matching_snapshot_passes() {
    let driver = "fn baseline() => Option[GoldenBaseline[Binary{8}]] = Some(GB(0b0000_0001, 0b1010_1011));\nfn main() => Verdict[GoldenFail[Binary{8}]] = golden(baseline(), 0b0000_0001, 0b1010_1011);";
    let src = program(driver);
    let expected =
        format!("{T_VERDICT}{T_GOLDEN_FAIL}fn main() => Verdict[GoldenFail[Binary{{8}}]] = Pass;");
    assert_three_way("golden match", &src, &expected);
}

/// A mismatch fails with the TYPED diff — name code, expected, produced all reified (C3;
/// FLAG-testing-3: the typed pair substitutes the rendered string diff).
#[test]
fn golden_mismatch_fails_with_typed_diff() {
    let driver = "fn baseline() => Option[GoldenBaseline[Binary{8}]] = Some(GB(0b0000_0001, 0b1010_1011));\nfn main() => Verdict[GoldenFail[Binary{8}]] = golden(baseline(), 0b0000_0001, 0b1100_1101);";
    let src = program(driver);
    let expected = format!(
        "{T_VERDICT}{T_GOLDEN_FAIL}fn main() => Verdict[GoldenFail[Binary{{8}}]] = Fail(FRec(GFail(0b0000_0001, 0b1010_1011, 0b1100_1101), {}, 0b0000_0000, \"golden\"));",
        lit64(0)
    );
    assert_three_way("golden mismatch typed diff", &src, &expected);
}

/// The snapshot is width-generic: the SAME `golden` specialises to Binary{16} at the call site
/// (the `map_get` width-genericity precedent).
#[test]
fn golden_second_width_binary16() {
    let driver = "fn baseline16() => Option[GoldenBaseline[Binary{16}]] = Some(GB(0b0000_0011, 0b1010_1011_0000_0001));\nfn main() => Verdict[GoldenFail[Binary{16}]] = golden(baseline16(), 0b0000_0011, 0b1010_1011_0000_0001);";
    let src = program(driver);
    let expected =
        format!("{T_VERDICT}{T_GOLDEN_FAIL}fn main() => Verdict[GoldenFail[Binary{{16}}]] = Pass;");
    assert_three_way("golden at Binary{16}", &src, &expected);
}

// ── differential (C1: an unavailable backend is NEVER a silent pass) ────────────────────────────

/// Shared backend thunks: the `enablement.rs` RFC-0033 Gap-B worked example `shl_u(1,3) = 8`
/// as the case under test (the self-application conformance case).
const DIFF_PRELUDE: &str = "fn lhs_shl(_u: Unit) => Binary{8} = shl_u(0b0000_0001, 0b0000_0011);\n\
fn rhs_eight(_u: Unit) => Binary{8} = 0b0000_1000;\n\
fn rhs_nine(_u: Unit) => Binary{8} = 0b0000_1001;\n";

/// An unavailable backend is `Skipped(BackendUnavailable)` — reported, never silent (C1/G2).
#[test]
fn differential_unavailable_backend_is_skipped() {
    let driver = format!(
        "{DIFF_PRELUDE}fn main() => Verdict[DiffFail[Binary{{8}}]] = differential(\"shl 1<<3\", False, lhs_shl, True, rhs_eight);"
    );
    let src = program(&driver);
    let expected = format!(
        "{T_VERDICT}{T_DIFF_FAIL}fn main() => Verdict[DiffFail[Binary{{8}}]] = Skipped(BackendUnavailable);"
    );
    assert_three_way("differential backend unavailable", &src, &expected);
}

/// Agreement passes: `shl_u(1,3)` ≡ the hand-computed 8.
#[test]
fn differential_agreement_passes() {
    let driver = format!(
        "{DIFF_PRELUDE}fn main() => Verdict[DiffFail[Binary{{8}}]] = differential(\"shl 1<<3\", True, lhs_shl, True, rhs_eight);"
    );
    let src = program(&driver);
    let expected =
        format!("{T_VERDICT}{T_DIFF_FAIL}fn main() => Verdict[DiffFail[Binary{{8}}]] = Pass;");
    assert_three_way("differential agreement", &src, &expected);
}

/// A disagreement fails with BOTH outputs reified (C3) and the input description composed into
/// the context via the Exact `bytes_concat` prim (parity with Rust's `differential({desc})`).
#[test]
fn differential_disagreement_fails_with_both_outputs() {
    let driver = format!(
        "{DIFF_PRELUDE}fn main() => Verdict[DiffFail[Binary{{8}}]] = differential(\"shl 1<<3\", True, lhs_shl, True, rhs_nine);"
    );
    let src = program(&driver);
    let expected = format!(
        "{T_VERDICT}{T_DIFF_FAIL}fn main() => Verdict[DiffFail[Binary{{8}}]] = Fail(FRec(DFail(shl_u(0b0000_0001, 0b0000_0011), 0b0000_1001), {}, 0b0000_0000, bytes_concat(bytes_concat(\"differential(\", \"shl 1<<3\"), \")\")));",
        lit64(0)
    );
    assert_three_way("differential disagreement", &src, &expected);
}

// ── the self-application smoke (the M-932 DoD extra) ────────────────────────────────────────────

/// **Self-application:** the ported `.myc` harness expresses an EXISTING conformance case — the
/// `enablement.rs` RFC-0033 Gap-B worked example `shl_u(1, 3) = 8` — end to end: the `.myc`
/// `differential` op checks the case, `summarize` aggregates it, `is_green` reports it, and the
/// three-way differential proves every execution path reaches the same green verdict. `.myc`
/// testing infrastructure is testing `.myc`-visible language behaviour.
#[test]
fn self_application_shl_conformance_case_is_green() {
    let driver = format!(
        "{DIFF_PRELUDE}fn suite() => Vec[Verdict[DiffFail[Binary{{8}}]]] = Cons(differential(\"shl 1<<3\", True, lhs_shl, True, rhs_eight), Nil);\nfn main() => Bool = is_green(summarize(suite()));"
    );
    let src = program(&driver);
    assert_three_way(
        "self-application: shl conformance case",
        &src,
        &expect_bool(true),
    );
}

/// The negative twin: a WRONG expected value flips the suite to not-green — the ported harness
/// cannot green-wash a failing conformance case (C1/G2).
#[test]
fn self_application_wrong_expectation_is_not_green() {
    let driver = format!(
        "{DIFF_PRELUDE}fn suite() => Vec[Verdict[DiffFail[Binary{{8}}]]] = Cons(differential(\"shl 1<<3\", True, lhs_shl, True, rhs_nine), Nil);\nfn main() => Bool = is_green(summarize(suite()));"
    );
    let src = program(&driver);
    assert_three_way(
        "self-application: wrong expectation not green",
        &src,
        &expect_bool(false),
    );
}

// ── mode-parametric surface (M-796; RFC-0034 §13) ───────────────────────────────────────────────

/// `FAST_ONLY` contains Fast and not Balanced — the scope predicate over the closed tier set.
#[test]
fn scope_contains_fast_only() {
    let driver = "fn main() => Bool = bool_and(scope_contains(scope_fast_only(), Fast), bool_not(scope_contains(scope_fast_only(), Balanced)));";
    let src = program(driver);
    assert_three_way("scope_contains FAST_ONLY", &src, &expect_bool(true));
}

/// The resolver→scope bridge: Balanced resolves to the NON_FAST scope.
#[test]
fn from_resolved_balanced_is_non_fast() {
    let driver = "fn no_src() => Option[CertScope] = None;\nfn main() => ModeScope = from_resolved_mode(RMode(Balanced, no_src()));";
    let src = program(driver);
    let expected = format!("{T_MODE}fn main() => ModeScope = MScope(False, True, True);");
    assert_three_way("from_resolved_mode(Balanced)", &src, &expected);
}

/// `scope_union`/`scope_intersect` compose scopes point-wise.
#[test]
fn scope_union_and_intersect_compose() {
    let driver = "fn main() => Bool = bool_and(scope_contains(scope_union(scope_fast_only(), scope_certified_only()), Certified), bool_not(scope_contains(scope_intersect(scope_fast_only(), scope_certified_only()), Fast)));";
    let src = program(driver);
    assert_three_way("scope union/intersect", &src, &expect_bool(true));
}

/// The most-specific declaration wins: Nodule:Certified beats Phylum:Fast, and the provenance
/// names the winning scope (never ambient — G2).
#[test]
fn config_provenance_nodule_wins() {
    let driver = "fn decls() => Vec[CertDecl] = Cons(Decl(ScPhylum, Fast), Cons(Decl(ScNodule, Certified), Nil));\nfn main() => ResolvedMode = config_provenance(config_new(decls()));";
    let src = program(driver);
    let expected = format!("{T_MODE}fn main() => ResolvedMode = RMode(Certified, Some(ScNodule));");
    assert_three_way("provenance: nodule wins", &src, &expected);
}

/// The granular override wins over ALL scope tiers, with source None (above the lattice).
#[test]
fn config_granular_override_wins() {
    let driver = "fn decls() => Vec[CertDecl] = Cons(Decl(ScNodule, Certified), Nil);\nfn main() => ResolvedMode = config_provenance(config_with_granular(config_new(decls()), Fast));";
    let src = program(driver);
    let expected = format!("{T_MODE}fn main() => ResolvedMode = RMode(Fast, None);");
    assert_three_way("provenance: granular wins", &src, &expected);
}

/// No declarations ⇒ the Fast project default ⇒ the ALL_MODES scope (widening from fast is
/// always safe — the Rust Default impl's contract).
#[test]
fn config_default_resolves_all_modes() {
    let driver = "fn main() => ModeScope = config_resolve(config_default());";
    let src = program(driver);
    let expected = format!("{T_MODE}fn main() => ModeScope = MScope(True, True, True);");
    assert_three_way("default config → ALL_MODES", &src, &expected);
}

/// `for_each_mode_in` visits exactly the in-scope modes (the never-silent visited/skipped audit).
#[test]
fn for_each_mode_in_matches_scope() {
    let driver = "fn main() => Bool = matches_scope(for_each_mode_in(scope_non_fast(), mode_code), scope_non_fast());";
    let src = program(driver);
    assert_three_way("for_each_mode_in audit", &src, &expect_bool(true));
}

/// A property correctly scoped FAST_ONLY has no violation (the assert_mode_scope pass case,
/// reified — FLAG-testing-6).
#[test]
fn mode_scope_violation_none_when_conformant() {
    let driver = "fn pred_is_fast(m: CertMode) => Bool = mode_eq(m, Fast);\nfn main() => Option[ModeViolation] = mode_scope_violation(scope_fast_only(), pred_is_fast);";
    let src = program(driver);
    let expected = format!("{T_MODE}fn main() => Option[ModeViolation] = None;");
    assert_three_way("mode_scope_violation conformant", &src, &expected);
}

/// A property ABSENT where the scope requires it is a reified PositiveAbsent violation (the
/// positive panic direction of the Rust assert).
#[test]
fn mode_scope_violation_positive_absent() {
    let driver = "fn pred_is_fast(m: CertMode) => Bool = mode_eq(m, Fast);\nfn main() => Option[ModeViolation] = mode_scope_violation(scope_all_modes(), pred_is_fast);";
    let src = program(driver);
    let expected =
        format!("{T_MODE}fn main() => Option[ModeViolation] = Some(PositiveAbsent(Balanced));");
    assert_three_way("mode_scope_violation positive-absent", &src, &expected);
}

/// A property FIRING outside its scope is a reified NegativeFires violation (the cross-mode
/// negative pattern — "the invariant fires where it doesn't apply").
#[test]
fn mode_negative_violation_fires_outside_scope() {
    let driver = "fn pred_always(_m: CertMode) => Bool = True;\nfn main() => Option[ModeViolation] = mode_negative_violation(scope_certified_only(), pred_always);";
    let src = program(driver);
    let expected =
        format!("{T_MODE}fn main() => Option[ModeViolation] = Some(NegativeFires(Fast));");
    assert_three_way("mode_negative_violation fires", &src, &expected);
}

// ── guarantee matrix (spec §4 — the table as checked data) ──────────────────────────────────────

/// The matrix has exactly 5 rows (spec §4 lists five ops) — the count derived by the vec_len
/// add_u spine.
#[test]
fn matrix_has_five_rows() {
    let driver = "fn main() => Binary{8} = vec_len(matrix());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, 0b0000_0000)))));";
    assert_three_way("matrix has 5 rows", &src, expected);
}

/// Every row's tag is Exact — the harness ops are Exact MECHANISMS; tagging one
/// Empirical/Declared would overclaim subject strength (spec §4 / VR-5, the Rust test's guard).
#[test]
fn matrix_all_rows_exact() {
    let driver = "fn main() => Bool = all_rows_exact(matrix());";
    let src = program(driver);
    assert_three_way("matrix all Exact", &src, &expect_bool(true));
}

/// Every row is EXPLAIN-able (C3/G11/SC-3 — no black boxes).
#[test]
fn matrix_all_rows_explainable() {
    let driver = "fn main() => Bool = all_rows_explainable(matrix());";
    let src = program(driver);
    assert_three_way("matrix all explainable", &src, &expect_bool(true));
}

/// Only `golden` and `differential` declare IO (spec §4) — read off the TYPED effects column.
#[test]
fn matrix_only_golden_and_differential_declare_io() {
    let driver = "fn main() => Bool = only_golden_and_differential_declare_io();";
    let src = program(driver);
    assert_three_way("matrix io discipline", &src, &expect_bool(true));
}

/// `summarize`/`is_green` are total (spec §4) — read off the TYPED fallibility column.
#[test]
fn matrix_aggregator_rows_are_total() {
    let driver = "fn main() => Bool = aggregator_rows_are_total();";
    let src = program(driver);
    assert_three_way("matrix aggregators total", &src, &expect_bool(true));
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Rust-oracle differential (D5 row 4) — wired against the RETAINED `mycelium-std-testing` crate
// (RFC-0031 D6: the crate is NOT retired). Scalar cases reduce both sides to u64/u32/u8;
// matrix cases compare LIVE against guarantee_matrix::MATRIX; mode-resolver cases build their
// expected `.myc` value LIVE from the Rust resolver's output (FLAG-testing-7 anti-drift pin).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The `.myc` Xorshift64 is BIT-EXACT vs the Rust `Rng` across seeds — including the zero-seed
/// promotion and a multi-step chain (`Empirical`, by trial on these seeds).
#[test]
fn oracle_rng_bit_exact_across_seeds() {
    for seed in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX] {
        let driver = format!(
            "fn main() => Binary{{64}} = rng_next(rng_new({}));",
            lit64(seed)
        );
        let myc = eval_u64(&driver);
        let rust = Rng::new(seed).next_u64();
        assert_eq!(
            myc, rust,
            "rng_next(rng_new({seed})) must match Rust bit-for-bit"
        );
    }
    // Three chained steps from seed 1 (state threading ≡ &mut mutation).
    let driver = format!(
        "fn main() => Binary{{64}} = rng_next(rng_next(rng_next(rng_new({}))));",
        lit64(1)
    );
    let myc = eval_u64(&driver);
    let mut rng = Rng::new(1);
    rng.next_u64();
    rng.next_u64();
    let rust = rng.next_u64();
    assert_eq!(myc, rust, "three chained rng_next steps must match Rust");
}

/// `rng_next_u32` (high 32 bits via the never-silent width_cast narrow) matches Rust's
/// `next_u32`.
#[test]
fn oracle_rng_next_u32_matches_rust() {
    for seed in [0u64, 7, 123_456_789] {
        let driver = format!(
            "fn main() => Binary{{32}} = rng_next_u32(rng_new({}));",
            lit64(seed)
        );
        let myc = eval_u32(&driver);
        let rust = Rng::new(seed).next_u32();
        assert_eq!(myc, rust, "rng_next_u32({seed}) must match Rust");
    }
}

/// Budget parity: 0 is refused on both sides; the default is 100 on both sides.
#[test]
fn oracle_budget_matches_rust() {
    assert!(Budget::new(0).is_none(), "Rust Budget::new(0) is None");
    // (.myc side: `budget_new_zero_is_none` above proves the same three-way.)
    let myc_default = eval_u8("fn main() => Binary{8} = budget_trials(budget_default());");
    assert_eq!(
        u32::from(myc_default),
        Budget::DEFAULT.trials(),
        "budget_default must match Budget::DEFAULT"
    );
    let myc_five = eval_u8("fn main() => Binary{8} = budget_trials(Trials(0b0000_0101));");
    assert_eq!(
        u32::from(myc_five),
        Budget::new(5).expect("5 is a valid budget").trials(),
        "budget_trials(Trials(5)) must match the Rust accessor"
    );
}

/// A test-local Rust `Gen<u64>` mirroring the `.myc` `gen_state` fn (one draw per trial, no
/// shrink candidates) — the FLAG-testing-1 fn-pair substitution's oracle twin.
struct DrawU64;
impl Gen<u64> for DrawU64 {
    fn generate(&mut self, rng: &mut Rng) -> Option<u64> {
        Some(rng.next_u64())
    }
}

/// A generator that never produces (both sides must report `Skipped`, never a silent pass).
struct NoDraw;
impl Gen<u64> for NoDraw {
    fn generate(&mut self, _rng: &mut Rng) -> Option<u64> {
        None
    }
}

/// for_all parity — failing property: the SAME seeded draw sequence violates `x < 2^63` on the
/// SAME trial with the SAME counterexample on both sides, and Rust's rendered description equals
/// `trial={t} value={v}` over the `.myc` REIFIED values (the FLAG-testing-3 substitution, pinned).
#[test]
fn oracle_for_all_failure_parity() {
    const SEED: u64 = 1;
    const BUDGET: u8 = 16;
    let thresh: u64 = 1 << 63;

    // Rust side.
    let budget = Budget::new(u32::from(BUDGET)).expect("non-zero");
    let verdict = for_all(&mut DrawU64, SEED, budget, |x: &u64| *x < thresh);
    let Verdict::Fail { record } = verdict else {
        panic!(
            "the seed-1 draw sequence must violate `x < 2^63` within 16 trials, got {verdict:?}"
        );
    };
    assert_eq!(
        record.seed, SEED,
        "Rust record carries the reproducing seed"
    );

    // .myc side — shared driver body, projected per component (counterexample, then trial).
    let body = format!(
        "fn gen_state(s: Binary{{64}}) => GenStep[Binary{{64}}] = Stepped(Some(rng_next(s)), rng_next(s));\nfn no_shrink64(_x: Binary{{64}}) => Vec[Binary{{64}}] = Nil;\nfn prop_below(x: Binary{{64}}) => Bool = match lt(x, {th}) {{ 0b1 => True, _ => False }};\nfn run() => Verdict[Binary{{64}}] = for_all(gen_state, no_shrink64, {seed}, Trials({budget}), prop_below);",
        th = lit64(thresh),
        seed = lit64(SEED),
        budget = lit8(BUDGET),
    );
    let cx_driver = format!(
        "{body}\nfn main() => Binary{{64}} = match run() {{ Fail(rec) => match rec {{ FRec(cx, _, _, _) => cx }}, Pass => zero64(), Skipped(_) => zero64(), Undetermined(_) => zero64() }};"
    );
    let trial_driver = format!(
        "{body}\nfn main() => Binary{{8}} = match run() {{ Fail(rec) => match rec {{ FRec(_, _, t, _) => t }}, Pass => 0b1111_1111, Skipped(_) => 0b1111_1111, Undetermined(_) => 0b1111_1111 }};"
    );
    let myc_cx = eval_u64(&cx_driver);
    let myc_trial = eval_u8(&trial_driver);

    assert_eq!(
        u32::from(myc_trial),
        record.trial,
        "the failing trial index must match the Rust oracle"
    );
    assert_eq!(
        format!("trial={} value={}", myc_trial, myc_cx),
        record.description,
        "Rust's rendered description must equal the .myc reified (trial, counterexample) pair"
    );
}

/// for_all parity — pass and skip classes: an all-pass run is `Pass` on both sides; an empty
/// generator is `Skipped` on both sides (never a silent pass — C1).
#[test]
fn oracle_for_all_pass_and_skip_parity() {
    let budget = Budget::new(5).expect("non-zero");
    let pass = for_all(&mut DrawU64, 9, budget, |_x: &u64| true);
    assert_eq!(pass, Verdict::Pass, "Rust all-pass run is Pass");
    // (.myc side: `for_all_all_trials_pass` proves Pass three-way.)

    let skipped = for_all(&mut NoDraw, 9, budget, |_x: &u64| true);
    assert!(
        matches!(
            skipped,
            Verdict::Skipped {
                reason: mycelium_std_testing::SkipReason::NeedsRecord
            }
        ),
        "Rust empty-generator run is Skipped(NeedsRecord), got {skipped:?}"
    );
    // (.myc side: `for_all_empty_generator_is_skipped` proves the same three-way.)
}

/// golden parity — all four verdict shapes agree class-for-class with the Rust oracle: missing
/// baseline ⇒ Skipped(NeedsRecord); wrong name ⇒ Skipped(NeedsRecord); match ⇒ Pass;
/// mismatch ⇒ Fail. (The `.myc` payloads are proven three-way above; the OPEN string
/// name/snapshot is the FLAG-testing-4 substitution — class parity is the frozen observable.)
#[test]
fn oracle_golden_class_parity() {
    use mycelium_std_testing::SkipReason;

    let missing = golden(None, "case", "value");
    assert!(matches!(
        missing,
        Verdict::Skipped {
            reason: SkipReason::NeedsRecord
        }
    ));
    let wrong = GoldenBaseline::new("other", "value");
    let wrong_name = golden(Some(&wrong), "case", "value");
    assert!(matches!(
        wrong_name,
        Verdict::Skipped {
            reason: SkipReason::NeedsRecord
        }
    ));
    let base = GoldenBaseline::new("case", "value");
    assert_eq!(golden(Some(&base), "case", "value"), Verdict::Pass);
    assert!(matches!(
        golden(Some(&base), "case", "other"),
        Verdict::Fail { .. }
    ));
    // (.myc side: the four `golden_*` three-way cases above prove the same class per shape.)
}

/// differential parity — the same thunked case agrees with the Rust oracle in all three shapes,
/// and on disagreement the `.myc` reified DFail pair equals the Rust closures' outputs.
#[test]
fn oracle_differential_parity() {
    use mycelium_std_testing::SkipReason;

    let pass = differential("shl 1<<3", true, || 1i8 << 3, true, || 8i8);
    assert_eq!(pass, Verdict::Pass, "Rust agreement is Pass");

    let skipped = differential("shl 1<<3", false, || 1i8 << 3, true, || 8i8);
    assert!(matches!(
        skipped,
        Verdict::Skipped {
            reason: SkipReason::BackendUnavailable
        }
    ));

    let fail = differential("shl 1<<3", true, || 1i8 << 3, true, || 9i8);
    assert!(
        matches!(fail, Verdict::Fail { .. }),
        "Rust disagreement is Fail"
    );

    // .myc reified pair equals the Rust closures' outputs (8, 9).
    let lhs_driver = format!(
        "{DIFF_PRELUDE}fn main() => Binary{{8}} = match differential(\"shl 1<<3\", True, lhs_shl, True, rhs_nine) {{ Fail(rec) => match rec {{ FRec(d, _, _, _) => match d {{ DFail(l, _) => l }} }}, Pass => 0b0000_0000, Skipped(_) => 0b0000_0000, Undetermined(_) => 0b0000_0000 }};"
    );
    let rhs_driver = format!(
        "{DIFF_PRELUDE}fn main() => Binary{{8}} = match differential(\"shl 1<<3\", True, lhs_shl, True, rhs_nine) {{ Fail(rec) => match rec {{ FRec(d, _, _, _) => match d {{ DFail(_, r) => r }} }}, Pass => 0b0000_0000, Skipped(_) => 0b0000_0000, Undetermined(_) => 0b0000_0000 }};"
    );
    assert_eq!(
        eval_u8(&lhs_driver),
        8,
        "reified lhs equals the Rust lhs output"
    );
    assert_eq!(
        eval_u8(&rhs_driver),
        9,
        "reified rhs equals the Rust rhs output"
    );
}

/// summarize/is_green parity — the same verdict multiset aggregates to the same counts and the
/// same green-ness on both sides ("green" = checked-and-passed; the skip stays surfaced,
/// distinct, and non-blocking on BOTH sides).
#[test]
fn oracle_summarize_parity() {
    use mycelium_std_testing::{FailRecord, SkipReason, UndetReason};

    let rust_verdicts = vec![
        Verdict::Pass,
        Verdict::Fail {
            record: FailRecord {
                description: "x".to_owned(),
                seed: 0,
                trial: 0,
                context: "x".to_owned(),
            },
        },
        Verdict::Skipped {
            reason: SkipReason::Ignored,
        },
        Verdict::Undetermined {
            reason: UndetReason::OracleUnavailable,
        },
        Verdict::Pass,
    ];
    let rust_summary = summarize(&rust_verdicts);
    assert!(!is_green(&rust_summary));

    let body = "fn suite() => Vec[Verdict[Unit]] = Cons(Pass, Cons(Fail(FRec(U, zero64(), 0b0000_0000, \"x\")), Cons(Skipped(Ignored), Cons(Undetermined(OracleUnavailable), Cons(Pass, Nil)))));";
    for (proj, rust_count) in [
        ("summary_passed", rust_summary.passed),
        ("summary_failed", rust_summary.failed),
        ("summary_skipped", rust_summary.skipped),
        ("summary_undetermined", rust_summary.undetermined),
        ("summary_total", rust_summary.total()),
    ] {
        let driver = format!("{body}\nfn main() => Binary{{8}} = {proj}(summarize(suite()));");
        assert_eq!(
            u32::from(eval_u8(&driver)),
            rust_count,
            "{proj} must match the Rust oracle"
        );
    }
    let green_driver = format!("{body}\nfn main() => Bool = is_green(summarize(suite()));");
    let src = program(&green_driver);
    assert_three_way(
        "oracle is_green parity",
        &src,
        &expect_bool(is_green(&rust_summary)),
    );
}

// ── mode-resolver parity (the FLAG-testing-7 anti-drift pin) ────────────────────────────────────

/// The `.myc` constructor name for a Rust `CertMode`.
fn mode_ctor(m: mycelium_core::cert_mode::CertMode) -> &'static str {
    use mycelium_core::cert_mode::CertMode;
    match m {
        CertMode::Fast => "Fast",
        CertMode::Balanced => "Balanced",
        CertMode::Certified => "Certified",
    }
}

/// The `.myc` constructor name for a Rust `CertScope`.
fn scope_ctor(s: CertScope) -> &'static str {
    match s {
        CertScope::Global => "ScGlobal",
        CertScope::Phylum => "ScPhylum",
        CertScope::Nodule => "ScNodule",
    }
}

/// The `.myc` cons-list source for a Rust `CertDecl` slice.
fn decls_src(decls: &[CertDecl]) -> String {
    decls.iter().rev().fold("Nil".to_owned(), |acc, d| {
        format!(
            "Cons(Decl({}, {}), {acc})",
            scope_ctor(d.scope),
            mode_ctor(d.mode)
        )
    })
}

/// The reimplemented `.myc` most-specific-wins resolver agrees with the SHARED Rust resolver
/// (`ModeTestConfig::provenance` → `mycelium_proj::resolve_mode`) on every declaration stack in
/// the table — the expected `.myc` value is built LIVE from the Rust resolver's output, so a real
/// divergence (drift in either direction) fails this test (`Empirical` — the FLAG-testing-7 pin).
#[test]
fn oracle_mode_resolver_parity_live() {
    use mycelium_core::cert_mode::CertMode;

    let cases: Vec<(&str, Vec<CertDecl>)> = vec![
        ("empty (project default)", vec![]),
        (
            "global balanced",
            vec![CertDecl {
                scope: CertScope::Global,
                mode: CertMode::Balanced,
            }],
        ),
        (
            "phylum balanced + nodule certified",
            vec![
                CertDecl {
                    scope: CertScope::Phylum,
                    mode: CertMode::Balanced,
                },
                CertDecl {
                    scope: CertScope::Nodule,
                    mode: CertMode::Certified,
                },
            ],
        ),
        (
            "nodule fast listed before global certified",
            vec![
                CertDecl {
                    scope: CertScope::Nodule,
                    mode: CertMode::Fast,
                },
                CertDecl {
                    scope: CertScope::Global,
                    mode: CertMode::Certified,
                },
            ],
        ),
    ];

    for (label, decls) in &cases {
        let config = ModeTestConfig::new(decls);
        let rust_prov = config.provenance();
        let rust_scope = config.resolve();

        // provenance parity (live-built expected).
        let driver = format!(
            "fn decls() => Vec[CertDecl] = {};\nfn main() => ResolvedMode = config_provenance(config_new(decls()));",
            decls_src(decls)
        );
        let src = program(&driver);
        let source_src = rust_prov
            .source
            .map_or("None".to_owned(), |s| format!("Some({})", scope_ctor(s)));
        let expected = format!(
            "{T_MODE}fn main() => ResolvedMode = RMode({}, {});",
            mode_ctor(rust_prov.mode),
            source_src
        );
        assert_three_way(&format!("resolver parity: {label}"), &src, &expected);

        // resolved-scope parity (live-built expected from the Rust ModeScope's membership).
        let scope_driver = format!(
            "fn decls() => Vec[CertDecl] = {};\nfn main() => ModeScope = config_resolve(config_new(decls()));",
            decls_src(decls)
        );
        let scope_src = program(&scope_driver);
        let flag = |m: CertMode| {
            if rust_scope.contains(m) {
                "True"
            } else {
                "False"
            }
        };
        let scope_expected = format!(
            "{T_MODE}fn main() => ModeScope = MScope({}, {}, {});",
            flag(CertMode::Fast),
            flag(CertMode::Balanced),
            flag(CertMode::Certified)
        );
        assert_three_way(
            &format!("scope parity: {label}"),
            &scope_src,
            &scope_expected,
        );
    }

    // Granular override parity.
    let config = ModeTestConfig::new(&[CertDecl {
        scope: CertScope::Nodule,
        mode: mycelium_core::cert_mode::CertMode::Certified,
    }])
    .with_granular(mycelium_core::cert_mode::CertMode::Fast);
    let rust_prov = config.provenance();
    let driver = "fn decls() => Vec[CertDecl] = Cons(Decl(ScNodule, Certified), Nil);\nfn main() => ResolvedMode = config_provenance(config_with_granular(config_new(decls()), Fast));";
    let src = program(driver);
    let source_src = rust_prov
        .source
        .map_or("None".to_owned(), |s| format!("Some({})", scope_ctor(s)));
    let expected = format!(
        "{T_MODE}fn main() => ResolvedMode = RMode({}, {});",
        mode_ctor(rust_prov.mode),
        source_src
    );
    assert_three_way("resolver parity: granular override", &src, &expected);
}

/// The panic-based Rust asserts agree with the `.myc` reified violation cores: a conformant
/// scope/predicate pair does not panic (`.myc`: None), a violating pair panics (`.myc`:
/// Some(violation)) — the FLAG-testing-6 substitution, pinned at class level.
#[test]
fn oracle_mode_assert_parity() {
    use mycelium_core::cert_mode::CertMode;
    use mycelium_std_testing::{assert_mode_scope, ModeScope};

    // Conformant: does not panic; .myc `mode_scope_violation_none_when_conformant` is None.
    assert_mode_scope(
        ModeScope::FAST_ONLY,
        |mode| mode == CertMode::Fast,
        "fast-only property",
    );

    // Violating (property absent where required): panics; .myc
    // `mode_scope_violation_positive_absent` is Some(PositiveAbsent(Balanced)).
    let panicked = std::panic::catch_unwind(|| {
        assert_mode_scope(
            ModeScope::ALL_MODES,
            |mode| mode == CertMode::Fast,
            "fast-only property declared all-modes",
        );
    });
    assert!(
        panicked.is_err(),
        "the Rust assert must panic exactly where the .myc core reifies a violation"
    );
}

// ── guarantee-matrix parity — LIVE from guarantee_matrix::MATRIX (the std_diag precedent) ───────

/// Row-for-row live parity: op names and fallibility/effects PROSE are byte-identical between
/// the `.myc` matrix and the Rust MATRIX; the typed tag/explainable/total/io columns agree with
/// the live Rust values. A real transcription drift fails here, not in a stale copy.
#[test]
fn oracle_matrix_live_parity() {
    use mycelium_core::GuaranteeStrength;

    let myc_len = eval_u8("fn main() => Binary{8} = vec_len(matrix());");
    assert_eq!(
        usize::from(myc_len),
        MATRIX.len(),
        "row count must match MATRIX.len()"
    );

    let row_fns = [
        "row_for_all",
        "row_golden",
        "row_differential",
        "row_summarize",
        "row_is_green",
    ];
    for (i, row_fn) in row_fns.iter().enumerate() {
        let rust_row = &MATRIX[i];

        let op_bytes = eval_bytes(&format!("fn main() => Bytes = row_op({row_fn}());"));
        assert_eq!(
            op_bytes,
            rust_row.op.as_bytes(),
            "row {i} op name must be byte-identical to MATRIX"
        );

        let fal_bytes = eval_bytes(&format!(
            "fn main() => Bytes = row_fallibility({row_fn}());"
        ));
        assert_eq!(
            fal_bytes,
            rust_row.fallibility.as_bytes(),
            "row {i} fallibility prose must be byte-identical to MATRIX"
        );

        let eff_bytes = eval_bytes(&format!("fn main() => Bytes = row_effects({row_fn}());"));
        assert_eq!(
            eff_bytes,
            rust_row.effects.as_bytes(),
            "row {i} effects prose must be byte-identical to MATRIX"
        );

        // Typed columns vs live Rust values (tag Exact-ness, explainability, totality, io).
        let tag_driver = format!("fn main() => Bool = guarantee_eq(row_tag({row_fn}()), GExact);");
        assert_three_way(
            &format!("matrix row {i} tag parity"),
            &program(&tag_driver),
            &expect_bool(rust_row.tag == GuaranteeStrength::Exact),
        );

        let expl_driver = format!("fn main() => Bool = row_explainable({row_fn}());");
        assert_three_way(
            &format!("matrix row {i} explainable parity"),
            &program(&expl_driver),
            &expect_bool(rust_row.explainable),
        );

        let total_driver =
            format!("fn main() => Bool = is_fal_total(row_fallibility_class({row_fn}()));");
        assert_three_way(
            &format!("matrix row {i} totality parity"),
            &program(&total_driver),
            &expect_bool(rust_row.fallibility == "total"),
        );

        let io_driver = format!("fn main() => Bool = declares_io(row_effects_class({row_fn}()));");
        assert_three_way(
            &format!("matrix row {i} io parity"),
            &program(&io_driver),
            &expect_bool(rust_row.effects.contains("io")),
        );
    }
}
