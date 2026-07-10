//! Differential tests for `std.recover` (M-930, E29-1, kickoff `opp`) — the never-silent
//! declarative recovery bridge: `handle_classified` over a reified policy, the budget ledger,
//! the declared-effect checker, and the RFC-0016 §4.5 guarantee matrix as checked data.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-op three-way cases, and — the row this port owns
//! per the harness doc (§4) — the live comparisons against the **retained Rust oracle**,
//! `mycelium-std-recover` (RFC-0031 D6; the crate is NOT retired). Driver-semantics oracle cases
//! reduce both sides to bytes ([`eval_byte`], the `std_error.rs` precedent); matrix oracle cases
//! compute the expected value **live from `mycelium_std_recover::guarantee_matrix::MATRIX`**
//! (the `std_diag.rs` precedent) so a real Rust↔`.myc` divergence flips the oracle and fails.
//!
//! # Surface-check (D5 row 1) and substitutions
//! See `lib/std/recover.myc`'s header comment for the full surface-check. PORTED: the outcome/
//! resolution sums + predicates, the closed 4-action set, the class registry (closed typed
//! vocabulary), the policy (`on`/`action_for`/`policy_effects`), the budget ledger
//! (`budget_set`/`budget_remaining`/`budget_consume`), `check_effects`, the driver
//! (`handle_classified`/`recover_classified`), and the 11-row guarantee matrix. FLAGGED, not
//! forced (VR-5/G2): `PolicyRef`/`policy_ref`/`PolicyHashError` (kernel BLAKE3 `ContentHash` —
//! the D1 boundary; the `Resolution` field is substituted by the presence-only `PolicyWitness`),
//! `DiagError` (kernel `Diag`), `EffectKind::Named`/open string class names (no `bytes_eq` prim),
//! and the Rust source's substring-assertion tests (same gap; their semantics are asserted
//! executably on `handle_classified` below instead).
//!
//! # Honesty tags
//! - **`Declared`** — each ported op's type-level contract and the 11-row matrix transcription,
//!   carried at the SAME strength as `mycelium-std-recover`'s own guarantee matrix (VR-5: never
//!   upgraded in translation).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) AND the
//!   Rust-oracle differential below, both validated by trial on the programs in this file;
//!   neither is a machine-checked proof.

mod harness;

use mycelium_core::{binary::bits_to_int, CoreValue, GuaranteeStrength, Payload};
use mycelium_interp::budget::{Budgets, EffectBudget, EffectKind};
use mycelium_std_recover::guarantee_matrix::{Explainable, Fallibility, MATRIX};
use mycelium_std_recover::{
    check_effects, handle_classified, ClassRegistry, EffectSet, Outcome, RecoveryAction,
    RecoveryPolicy, Resolution,
};

/// The std.recover nodule source, loaded at compile time — the single source of truth.
const RECOVER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/recover.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(RECOVER_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`] (same pattern as `std_error.rs`).
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Shared driver fragments — the closed test vocabulary every case builds on. `empty_pol` pins the
// policy's `T` by annotation (no bare generic `PNil` in argument position).
// ══════════════════════════════════════════════════════════════════════════════════════════════

const PRELUDE: &str = "fn classify_io(_e: Binary{8}) => ClassName = ClsIoError;\n\
fn reg_io() => ClassRegistry = RegCons(ClsIoError, RegNil);\n\
fn empty_pol() => Policy[Binary{8}] = PNil;\n\
fn act_retry(n: Binary{8}) => RecoveryAction[Binary{8}] = Retry(n);\n\
fn act_escalate(c: ClassName) => RecoveryAction[Binary{8}] = Escalate(c);\n\
fn act_cleanup(ef: EffectKind) => RecoveryAction[Binary{8}] = CleanupThenPropagate(ef);\n\
fn attempt_fail(_u: Unit) => AttemptOut[Binary{8}, Binary{8}] = Attempted(OErr(0b1111_1110), GDeclared);\n\
fn mk_err_in() => Outcome[Binary{8}, Binary{8}] = OErr(0b0000_0001);\n\
fn mk_ok_in() => Outcome[Binary{8}, Binary{8}] = OOk(0b0000_0101);\n\
fn attempt_ok_emp(_u: Unit) => AttemptOut[Binary{8}, Binary{8}] = Attempted(OOk(0b0000_0111), GEmpirical);\n";

/// The expected-side type mirror for `Resolution` results — constructor order matches
/// `lib/std/recover.myc` exactly (structural identity for the CoreValue comparison).
const EXPECT_RESOLUTION_TYPES: &str = "nodule ref;\n\
type Guarantee = GExact | GProven | GEmpirical | GDeclared;\n\
type PolicyWitness = ByPolicy | NoPolicy;\n\
type Resolution[T, E] = Recovered(T, Guarantee, PolicyWitness) | Propagated(E, PolicyWitness, Bool);\n";

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Three-way differential cases (L1-eval ≡ elaborate→L0-interp ≡ AOT) — the driver's I1/I2/I4/I5
// semantics, each against a hand-computed reference value (Declared data, Empirical agreement).
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ── handle_classified: the never-silent spine (I1) ──────────────────────────────────────────────

/// `OOk` passes through as `Recovered(v, GExact, NoPolicy)` — FR-R3: a clean pass-through is
/// Exact, never Declared (the P5-B exact-tag bug fix, carried into the port).
#[test]
fn handle_ok_pass_through_is_exact() {
    let driver = format!(
        "{PRELUDE}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_ok_in(), empty_pol(), BNil, classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Recovered(0b0000_0101, GExact, NoPolicy);"
    );
    assert_three_way("handle(OOk) pass-through Exact", &src, &expected);
}

/// An `OErr` with NO matching rule propagates UNCHANGED with `NoPolicy` — the I1 floor.
#[test]
fn handle_err_no_rule_propagates_unchanged() {
    let driver = format!(
        "{PRELUDE}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), empty_pol(), BNil, classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Propagated(0b0000_0001, NoPolicy, False);"
    );
    assert_three_way("handle(OErr, no rule) I1 floor", &src, &expected);
}

/// A `Fallback` rule recovers with the fixed `GDeclared` ceiling (I2/VR-5 — a substituted value
/// has no checked basis) and the `ByPolicy` witness.
#[test]
fn handle_fallback_recovers_declared() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, Fallback(0b0010_1010)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), BNil, classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Recovered(0b0010_1010, GDeclared, ByPolicy);"
    );
    assert_three_way("handle(fallback) Declared", &src, &expected);
}

/// A `Retry` success inherits the attempt's OWN tag (`GEmpirical` here) — never upgraded
/// (I2/VR-5/FR-R3).
#[test]
fn handle_retry_success_inherits_attempt_tag() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, act_retry(0b0000_0010)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), budget_set(BNil, Attempts(0b0000_0010)), classify_io, attempt_ok_emp);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Recovered(0b0000_0111, GEmpirical, ByPolicy);"
    );
    assert_three_way("handle(retry success) inherits tag", &src, &expected);
}

/// `Retry` exhausted (every attempt fails) propagates the ORIGINAL error — additive, never a
/// drop (I1).
#[test]
fn handle_retry_exhausted_propagates_original() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, act_retry(0b0000_0010)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), budget_set(BNil, Attempts(0b0000_0010)), classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Propagated(0b0000_0001, ByPolicy, False);"
    );
    assert_three_way(
        "handle(retry exhausted) original propagates",
        &src,
        &expected,
    );
}

/// `Retry` with NO declared budget (I5: an undeclared effect cannot run) propagates immediately —
/// the graceful I4 overrun, never a hang.
#[test]
fn handle_retry_absent_budget_propagates() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, act_retry(0b0000_0101)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), BNil, classify_io, attempt_ok_emp);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Propagated(0b0000_0001, ByPolicy, False);"
    );
    assert_three_way("handle(retry, absent budget) I5", &src, &expected);
}

/// `Escalate` re-propagates explicitly — never a recovered value, never a drop (I1). The target
/// class is X1-validated at `on` time (both classes registered here).
#[test]
fn handle_escalate_propagates() {
    let driver = format!(
        "{PRELUDE}fn reg2() => ClassRegistry = RegCons(ClsFatal, RegCons(ClsIoError, RegNil));\n\
fn mk_pol() => Policy[Binary{{8}}] = match on(reg2(), empty_pol(), ClsIoError, act_escalate(ClsFatal)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), BNil, classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Propagated(0b0000_0001, ByPolicy, False);"
    );
    assert_three_way("handle(escalate) propagates", &src, &expected);
}

/// `CleanupThenPropagate` with NO cleanup budget: the overrun is RECORDED (`True` — spec §7-Q4,
/// legible not swallowed) and the original error propagates regardless (I1).
#[test]
fn handle_cleanup_overrun_is_recorded() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, act_cleanup(EkIo)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), BNil, classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Propagated(0b0000_0001, ByPolicy, True);"
    );
    assert_three_way("handle(cleanup) overrun recorded", &src, &expected);
}

/// `CleanupThenPropagate` WITHIN budget: `cleanup_overrun` stays `False`; the original error
/// still propagates (I1 — cleanup never converts an error into success).
#[test]
fn handle_cleanup_within_budget_sets_overrun_false() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, act_cleanup(EkIo)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = handle_classified(mk_err_in(), mk_pol(), budget_set(BNil, Ops(0b0000_0001)), classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Propagated(0b0000_0001, ByPolicy, False);"
    );
    assert_three_way("handle(cleanup) within budget", &src, &expected);
}

/// `recover_classified` — the Result bridge: an `Err` input routes through `from_result` into the
/// same fallback path (error.md §7-Q1: Recovered | Propagated, no drop variant).
#[test]
fn recover_classified_bridges_result_err() {
    let driver = format!(
        "{PRELUDE}fn mk_pol() => Policy[Binary{{8}}] = match on(reg_io(), empty_pol(), ClsIoError, Fallback(0b0010_1010)) {{ Ok(p) => p, Err(_) => empty_pol() }};\n\
fn mk_res() => Result[Binary{{8}}, Binary{{8}}] = Err(0b0000_0001);\n\
fn main() => Resolution[Binary{{8}}, Binary{{8}}] = recover_classified(mk_res(), mk_pol(), BNil, classify_io, attempt_fail);"
    );
    let src = program(&driver);
    let expected = format!(
        "{EXPECT_RESOLUTION_TYPES}fn main() => Resolution[Binary{{8}}, Binary{{8}}] = Recovered(0b0010_1010, GDeclared, ByPolicy);"
    );
    assert_three_way("recover_classified(Err) bridge", &src, &expected);
}

// ── on: X1 — both class names are registry-validated, never fabricated ──────────────────────────

/// `on` with an unregistered LHS class is an explicit `Err(UnknownCls)` — never a silent
/// fabrication (X1/G2).
#[test]
fn on_unknown_class_is_explicit_err() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[Policy[Binary{{8}}], UnknownClass] = on(reg_io(), empty_pol(), ClsFatal, Fallback(0b0000_0001));"
    );
    let src = program(&driver);
    let expected = "nodule ref;\n\
type ClassName = ClsIoError | ClsParseError | ClsTimeout | ClsFatal;\n\
type UnknownClass = UnknownCls(ClassName);\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type RecoveryAction[T] = Fallback(T) | Retry(Binary{8}) | Escalate(ClassName) | CleanupThenPropagate(EffectKind);\n\
type Policy[T] = PNil | PRule(ClassName, RecoveryAction[T], Policy[T]);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Policy[Binary{8}], UnknownClass] = Err(UnknownCls(ClsFatal));";
    assert_three_way("on(unknown class) X1", &src, expected);
}

/// `on` with an unregistered `Escalate` TARGET is likewise an explicit `Err(UnknownCls)` —
/// the X1 fix carried over from the Rust source (the escalation target is validated too).
#[test]
fn on_unregistered_escalate_target_is_explicit_err() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[Policy[Binary{{8}}], UnknownClass] = on(reg_io(), empty_pol(), ClsIoError, act_escalate(ClsFatal));"
    );
    let src = program(&driver);
    let expected = "nodule ref;\n\
type ClassName = ClsIoError | ClsParseError | ClsTimeout | ClsFatal;\n\
type UnknownClass = UnknownCls(ClassName);\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type RecoveryAction[T] = Fallback(T) | Retry(Binary{8}) | Escalate(ClassName) | CleanupThenPropagate(EffectKind);\n\
type Policy[T] = PNil | PRule(ClassName, RecoveryAction[T], Policy[T]);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Policy[Binary{8}], UnknownClass] = Err(UnknownCls(ClsFatal));";
    assert_three_way("on(unregistered escalate target) X1", &src, expected);
}

// ── registry: resolve (X1 — looked up, never evaluated) ─────────────────────────────────────────

/// `resolve` on a registered class returns it; on an unregistered one it is an explicit
/// `Err(UnknownCls)` (X1/G2).
#[test]
fn resolve_registered_ok_unregistered_err() {
    let driver = format!(
        "{PRELUDE}fn main() => Result[ClassName, UnknownClass] = resolve(reg_io(), ClsIoError);"
    );
    let src = program(&driver);
    let expected = "nodule ref;\n\
type ClassName = ClsIoError | ClsParseError | ClsTimeout | ClsFatal;\n\
type UnknownClass = UnknownCls(ClassName);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[ClassName, UnknownClass] = Ok(ClsIoError);";
    assert_three_way("resolve(registered)", &src, expected);

    let driver = format!(
        "{PRELUDE}fn main() => Result[ClassName, UnknownClass] = resolve(reg_io(), ClsFatal);"
    );
    let src = program(&driver);
    let expected = "nodule ref;\n\
type ClassName = ClsIoError | ClsParseError | ClsTimeout | ClsFatal;\n\
type UnknownClass = UnknownCls(ClassName);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[ClassName, UnknownClass] = Err(UnknownCls(ClsFatal));";
    assert_three_way("resolve(unregistered) X1", &src, expected);
}

// ── budget ledger: consume drains / refuses explicitly (I4/I5) ──────────────────────────────────

/// A declared budget drains: `consume(Attempts(2), Retry, 1)` returns the decremented ledger
/// (the functional form of the Rust `&mut` decrement). The expected side recomputes the
/// decrement via the SAME `sub_u` prim (Derived provenance — the harness convention).
#[test]
fn budget_consume_drains_declared_budget() {
    let driver = "fn main() => Result[Budgets, EffectBudgetExhausted] = budget_consume(budget_set(BNil, Attempts(0b0000_0010)), EkRetry, 0b0000_0001);";
    let src = program(driver);
    let expected = "nodule ref;\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type EffectBudgetExhausted = Exhausted(EffectKind, Binary{8}, Binary{8});\n\
type Budgets = BNil | BEntry(EffectKind, Binary{8}, Budgets);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Budgets, EffectBudgetExhausted] = Ok(BEntry(EkRetry, sub_u(0b0000_0010, 0b0000_0001), BNil));";
    assert_three_way("budget_consume drains", &src, expected);
}

/// An ABSENT budget refuses immediately — `Exhausted(kind, requested, 0)` (I5: tightly scoped by
/// default; I4: explicit, graceful, never a hang).
#[test]
fn budget_consume_absent_budget_refuses() {
    let driver = "fn main() => Result[Budgets, EffectBudgetExhausted] = budget_consume(BNil, EkRetry, 0b0000_0001);";
    let src = program(driver);
    let expected = "nodule ref;\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type EffectBudgetExhausted = Exhausted(EffectKind, Binary{8}, Binary{8});\n\
type Budgets = BNil | BEntry(EffectKind, Binary{8}, Budgets);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Budgets, EffectBudgetExhausted] = Err(Exhausted(EkRetry, 0b0000_0001, 0b0000_0000));";
    assert_three_way("budget_consume absent refuses (I5)", &src, expected);
}

/// A zero-declared budget overruns explicitly, naming kind + requested + remaining (I4).
#[test]
fn budget_consume_overrun_names_kind_requested_remaining() {
    let driver = "fn main() => Result[Budgets, EffectBudgetExhausted] = budget_consume(budget_set(BNil, Attempts(0b0000_0000)), EkRetry, 0b0000_0001);";
    let src = program(driver);
    let expected = "nodule ref;\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type EffectBudgetExhausted = Exhausted(EffectKind, Binary{8}, Binary{8});\n\
type Budgets = BNil | BEntry(EffectKind, Binary{8}, Budgets);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Budgets, EffectBudgetExhausted] = Err(Exhausted(EkRetry, 0b0000_0001, 0b0000_0000));";
    assert_three_way("budget_consume overrun fields (I4)", &src, expected);
}

// ── check_effects (I3 — manual-declare + compositional-check) ───────────────────────────────────

/// Every performed effect declared → `Ok(U)`.
#[test]
fn check_effects_all_declared_is_ok() {
    let driver = "fn main() => Result[Unit, UndeclaredEffect] = check_effects(ECons(EkRetry, ECons(EkIo, ENil)), ECons(EkIo, ENil));";
    let src = program(driver);
    let expected = "nodule ref;\n\
type Unit = U;\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type UndeclaredEffect = Undeclared(EffectKind);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Unit, UndeclaredEffect] = Ok(U);";
    assert_three_way("check_effects declared ok", &src, expected);
}

/// A performed-but-undeclared effect is the explicit `Err(Undeclared(kind))` — never silent (I3).
#[test]
fn check_effects_undeclared_is_explicit_err() {
    let driver = "fn main() => Result[Unit, UndeclaredEffect] = check_effects(ECons(EkRetry, ENil), ECons(EkIo, ENil));";
    let src = program(driver);
    let expected = "nodule ref;\n\
type Unit = U;\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type UndeclaredEffect = Undeclared(EffectKind);\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Unit, UndeclaredEffect] = Err(Undeclared(EkIo));";
    assert_three_way("check_effects undeclared err (I3)", &src, expected);
}

// ── policy_effects (I3 — the policy's declared, closed effect set) ──────────────────────────────

/// A retry + cleanup policy declares exactly `{Retry, Io}` (dedup cons-list, deterministic
/// order: the walk folds right-to-left and prepends).
#[test]
fn policy_effects_collects_declared_kinds() {
    let driver = format!(
        "{PRELUDE}fn reg2() => ClassRegistry = RegCons(ClsParseError, RegCons(ClsIoError, RegNil));\n\
fn mk_pol() => Policy[Binary{{8}}] = match on(reg2(), empty_pol(), ClsIoError, act_retry(0b0000_0001)) {{ Ok(p1) => match on(reg2(), p1, ClsParseError, act_cleanup(EkIo)) {{ Ok(p2) => p2, Err(_) => empty_pol() }}, Err(_) => empty_pol() }};\n\
fn main() => EffectList = policy_effects(mk_pol());"
    );
    let src = program(&driver);
    let expected = "nodule ref;\n\
type EffectKind = EkRetry | EkAlloc | EkIo | EkCascade | EkTime;\n\
type EffectList = ENil | ECons(EffectKind, EffectList);\n\
fn main() => EffectList = ECons(EkRetry, ECons(EkIo, ENil));";
    assert_three_way("policy_effects collects kinds", &src, expected);
}

// ── outcome conversions ─────────────────────────────────────────────────────────────────────────

/// `into_result(from_result(Ok(x)))` round-trips — Outcome ≅ Result, no information moved.
#[test]
fn outcome_result_round_trip() {
    let driver = "fn mk() => Result[Binary{8}, Binary{8}] = Ok(0b0000_0101);\nfn main() => Result[Binary{8}, Binary{8}] = into_result(from_result(mk()));";
    let src = program(driver);
    let expected = "nodule ref;\n\
type Result[A, E] = Ok(A) | Err(E);\n\
fn main() => Result[Binary{8}, Binary{8}] = Ok(0b0000_0101);";
    assert_three_way("outcome/result round-trip", &src, expected);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Rust-oracle differential (D5 row 4) — wired against the RETAINED `mycelium-std-recover` crate
// (RFC-0031 D6: the crate is NOT retired). Driver-semantics cases reduce BOTH sides to bytes:
// the `.myc` driver reduces its Resolution to a raw `Binary{8}` via a local match, [`eval_byte`]
// decodes it exactly as the L1 evaluator/AOT paths do (`bits_to_int`, MSB-first two's-complement
// — the `std_error.rs` precedent), and the Rust side runs the ACTUAL `handle_classified` /
// `Budgets::consume` / `check_effects` with the same inputs. Guarantee tags and policy witnesses
// are compared through explicit shared code maps (`strength_code` / `witness presence`) — VR-5:
// the tag comparison is exact equality on the lattice, never a weaker "both recovered".
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Decode a `Binary{8}` [`CoreValue`] to its signed byte (the `std_error.rs` codec).
fn extract_byte(cv: &CoreValue) -> i8 {
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Binary{{8}} repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bits(bits) => bits_to_int(bits) as i8,
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Run `driver`'s `main` (a raw `Binary{8}`) through the L1 evaluator and return the decoded
/// byte (the `std_error.rs` precedent — the three-way obligation is covered by the cases above;
/// this helper only bridges to the Rust oracle).
fn eval_byte(driver: &str) -> i8 {
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
    let core = val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("result is outside the r3 data fragment"));
    extract_byte(&core)
}

/// The shared lattice code map: `Exact=0 ⊐ Proven=1 ⊐ Empirical=2 ⊐ Declared=3` — mirrors the
/// `.myc` drivers' `tag_code` fn below, so tag equality is compared exactly (VR-5).
fn strength_code(s: GuaranteeStrength) -> i8 {
    match s {
        GuaranteeStrength::Exact => 0,
        GuaranteeStrength::Proven => 1,
        GuaranteeStrength::Empirical => 2,
        GuaranteeStrength::Declared => 3,
    }
}

/// The shared `EffectKind` code map — mirrors `lib/std/recover.myc::effect_code`.
fn effect_kind_code(k: &EffectKind) -> i8 {
    match k {
        EffectKind::Retry => 0,
        EffectKind::Alloc => 1,
        EffectKind::Io => 2,
        EffectKind::Cascade => 3,
        EffectKind::Time => 4,
        EffectKind::Named(n) => panic!("Named({n}) is FLAG-recover-3 — not in the ported kernel"),
    }
}

/// The `.myc`-side reducers appended to oracle drivers: value/tag/witness/overrun projections of
/// a `Resolution` (code maps mirror `strength_code` / presence / bool-as-byte).
const ORACLE_REDUCERS: &str = "fn tag_code(g: Guarantee) => Binary{8} = match g { GExact => 0b0000_0000, GProven => 0b0000_0001, GEmpirical => 0b0000_0010, GDeclared => 0b0000_0011 };\n\
fn wit_code(w: PolicyWitness) => Binary{8} = match w { ByPolicy => 0b0000_0001, NoPolicy => 0b0000_0000 };\n\
fn bool_code(b: Bool) => Binary{8} = match b { True => 0b0000_0001, False => 0b0000_0000 };\n\
fn res_value(r: Resolution[Binary{8}, Binary{8}]) => Binary{8} = match r { Recovered(v, _, _) => v, Propagated(e, _, _) => e };\n\
fn res_tag(r: Resolution[Binary{8}, Binary{8}]) => Binary{8} = match r { Recovered(_, g, _) => tag_code(g), Propagated(_, _, _) => 0b0111_1111 };\n\
fn res_wit(r: Resolution[Binary{8}, Binary{8}]) => Binary{8} = match r { Recovered(_, _, w) => wit_code(w), Propagated(_, w, _) => wit_code(w) };\n\
fn res_overrun(r: Resolution[Binary{8}, Binary{8}]) => Binary{8} = match r { Recovered(_, _, _) => 0b0111_1111, Propagated(_, _, ov) => bool_code(ov) };\n";

/// Build the Rust oracle fixture the driver cases mirror: registry with `io-error`, a one-rule
/// policy for it, and a `Budgets` ledger.
fn rust_fixture(
    action: RecoveryAction<i8>,
    budgets: Budgets,
) -> (RecoveryPolicy<i8>, Budgets, mycelium_std_recover::ClassName) {
    let mut reg = ClassRegistry::new();
    reg.register("io-error");
    reg.register("fatal");
    let class = reg.resolve("io-error").expect("registered");
    let mut policy = RecoveryPolicy::<i8>::new();
    policy.on(&reg, "io-error", action).expect("known class");
    (policy, budgets, class)
}

/// Fallback — Rust oracle: `Recovered { value: 42, tag: Declared, policy: Some(_) }`; the `.myc`
/// side must match value byte, lattice code, AND witness presence.
#[test]
fn oracle_fallback_matches_rust() {
    let (policy, mut budgets, class) = rust_fixture(
        RecoveryAction::Fallback {
            value: Box::new(42i8),
        },
        Budgets::new(),
    );
    let r = handle_classified(
        Outcome::Err(1i8),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || (Outcome::Ok(0i8), GuaranteeStrength::Exact),
    );
    let Resolution::Recovered { value, tag, policy } = r else {
        panic!("rust oracle: fallback must recover");
    };

    let myc_pol = "fn mk_pol() => Policy[Binary{8}] = match on(reg_io(), empty_pol(), ClsIoError, Fallback(0b0010_1010)) { Ok(p) => p, Err(_) => empty_pol() };\n";
    let call = "handle_classified(mk_err_in(), mk_pol(), BNil, classify_io, attempt_fail)";
    let value_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_value({call});");
    let tag_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_tag({call});");
    let wit_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_wit({call});");

    assert_eq!(eval_byte(&value_driver), value, "fallback value byte");
    assert_eq!(
        eval_byte(&tag_driver),
        strength_code(tag),
        "fallback tag must be Declared on BOTH sides (I2/VR-5)"
    );
    assert_eq!(
        eval_byte(&wit_driver),
        i8::from(policy.is_some()),
        "fallback witness: a policy acted (ByPolicy ↔ Some(PolicyRef) presence — FLAG-recover-1)"
    );
}

/// Ok pass-through — Rust oracle: `Recovered { tag: Exact, policy: None }` (FR-R3).
#[test]
fn oracle_ok_pass_through_matches_rust() {
    let (policy, mut budgets, class) = rust_fixture(
        RecoveryAction::Fallback {
            value: Box::new(0i8),
        },
        Budgets::new(),
    );
    let r = handle_classified(
        Outcome::Ok(5i8),
        &policy,
        &mut budgets,
        |_: &i8| class.clone(),
        || (Outcome::Ok(0i8), GuaranteeStrength::Exact),
    );
    let Resolution::Recovered { value, tag, policy } = r else {
        panic!("rust oracle: Ok must pass through as Recovered (I1)");
    };

    let myc_pol = "fn mk_pol() => Policy[Binary{8}] = match on(reg_io(), empty_pol(), ClsIoError, Fallback(0b0000_0000)) { Ok(p) => p, Err(_) => empty_pol() };\n";
    let call = "handle_classified(mk_ok_in(), mk_pol(), BNil, classify_io, attempt_fail)";
    let value_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_value({call});");
    let tag_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_tag({call});");
    let wit_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_wit({call});");

    assert_eq!(eval_byte(&value_driver), value, "pass-through value byte");
    assert_eq!(
        eval_byte(&tag_driver),
        strength_code(tag),
        "Ok pass-through must be Exact on BOTH sides (FR-R3) — never Declared"
    );
    assert_eq!(
        eval_byte(&wit_driver),
        i8::from(policy.is_some()),
        "Ok pass-through: no policy acted (NoPolicy ↔ None)"
    );
}

/// Retry success — Rust oracle: the attempt's own `Empirical` tag is inherited, never upgraded.
#[test]
fn oracle_retry_success_inherits_attempt_tag() {
    let (policy, mut budgets, class) = rust_fixture(
        RecoveryAction::Retry { max_attempts: 2 },
        Budgets::new().with(EffectBudget::Attempts(2)),
    );
    let r = handle_classified(
        Outcome::Err(1i8),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || (Outcome::Ok(7i8), GuaranteeStrength::Empirical),
    );
    let Resolution::Recovered { value, tag, .. } = r else {
        panic!("rust oracle: retry with a succeeding attempt must recover");
    };

    let myc_pol = "fn mk_pol() => Policy[Binary{8}] = match on(reg_io(), empty_pol(), ClsIoError, act_retry(0b0000_0010)) { Ok(p) => p, Err(_) => empty_pol() };\n";
    let call = "handle_classified(mk_err_in(), mk_pol(), budget_set(BNil, Attempts(0b0000_0010)), classify_io, attempt_ok_emp)";
    let value_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_value({call});");
    let tag_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_tag({call});");

    assert_eq!(eval_byte(&value_driver), value, "retry success value byte");
    assert_eq!(
        eval_byte(&tag_driver),
        strength_code(tag),
        "retry success must inherit the attempt's Empirical tag on BOTH sides (I2/VR-5)"
    );
}

/// Retry exhausted — Rust oracle: the ORIGINAL error propagates (never the retry's error, never
/// a drop — I1).
#[test]
fn oracle_retry_exhausted_propagates_original() {
    let (policy, mut budgets, class) = rust_fixture(
        RecoveryAction::Retry { max_attempts: 2 },
        Budgets::new().with(EffectBudget::Attempts(2)),
    );
    let r = handle_classified(
        Outcome::Err(1i8),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || (Outcome::Err(-2i8), GuaranteeStrength::Exact),
    );
    let Resolution::Propagated { error, .. } = r else {
        panic!("rust oracle: exhausted retry must propagate");
    };

    let myc_pol = "fn mk_pol() => Policy[Binary{8}] = match on(reg_io(), empty_pol(), ClsIoError, act_retry(0b0000_0010)) { Ok(p) => p, Err(_) => empty_pol() };\n";
    let call = "handle_classified(mk_err_in(), mk_pol(), budget_set(BNil, Attempts(0b0000_0010)), classify_io, attempt_fail)";
    let value_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_value({call});");

    assert_eq!(
        eval_byte(&value_driver),
        error,
        "retry exhaustion must propagate the ORIGINAL error byte on BOTH sides (I1)"
    );
}

/// Cleanup overrun flag — Rust oracle: absent Io budget ⇒ `cleanup_overrun: true` (recorded, not
/// swallowed — spec §7-Q4); within budget ⇒ `false`. The original error propagates in BOTH cases.
#[test]
fn oracle_cleanup_overrun_flag_matches_rust() {
    // Absent budget → overrun recorded.
    let (policy, mut budgets, class) = rust_fixture(
        RecoveryAction::CleanupThenPropagate {
            effect: EffectKind::Io,
        },
        Budgets::new(),
    );
    let r = handle_classified(
        Outcome::Err(1i8),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || (Outcome::Ok(0i8), GuaranteeStrength::Exact),
    );
    let Resolution::Propagated {
        error,
        cleanup_overrun,
        ..
    } = r
    else {
        panic!("rust oracle: cleanup_then_propagate must propagate");
    };

    let myc_pol = "fn mk_pol() => Policy[Binary{8}] = match on(reg_io(), empty_pol(), ClsIoError, act_cleanup(EkIo)) { Ok(p) => p, Err(_) => empty_pol() };\n";
    let call_absent = "handle_classified(mk_err_in(), mk_pol(), BNil, classify_io, attempt_fail)";
    let value_driver = format!(
        "{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_value({call_absent});"
    );
    let overrun_driver = format!(
        "{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_overrun({call_absent});"
    );
    assert_eq!(eval_byte(&value_driver), error, "original error (I1)");
    assert_eq!(
        eval_byte(&overrun_driver),
        i8::from(cleanup_overrun),
        "absent-budget cleanup overrun must be RECORDED on BOTH sides (spec §7-Q4)"
    );

    // Within budget → overrun false.
    let (policy, mut budgets, class) = rust_fixture(
        RecoveryAction::CleanupThenPropagate {
            effect: EffectKind::Io,
        },
        Budgets::new().with(EffectBudget::Ops(1)),
    );
    let r = handle_classified(
        Outcome::Err(1i8),
        &policy,
        &mut budgets,
        |_| class.clone(),
        || (Outcome::Ok(0i8), GuaranteeStrength::Exact),
    );
    let Resolution::Propagated {
        cleanup_overrun, ..
    } = r
    else {
        panic!("rust oracle: cleanup_then_propagate must propagate");
    };
    let call_within = "handle_classified(mk_err_in(), mk_pol(), budget_set(BNil, Ops(0b0000_0001)), classify_io, attempt_fail)";
    let overrun_driver = format!(
        "{PRELUDE}{ORACLE_REDUCERS}{myc_pol}fn main() => Binary{{8}} = res_overrun({call_within});"
    );
    assert_eq!(
        eval_byte(&overrun_driver),
        i8::from(cleanup_overrun),
        "within-budget cleanup overrun must be false on BOTH sides (spec §7-Q4)"
    );
}

/// No-rule propagation — Rust oracle: the error propagates unchanged with `policy: None`
/// (↔ `NoPolicy`).
#[test]
fn oracle_no_rule_propagates_with_no_witness() {
    let policy = RecoveryPolicy::<i8>::new();
    let mut budgets = Budgets::new();
    let mut reg = ClassRegistry::new();
    reg.register("io-error");
    let class = reg.resolve("io-error").expect("registered");
    let r = handle_classified(
        Outcome::Err(1i8),
        &policy,
        &mut budgets,
        |_: &i8| class.clone(),
        || (Outcome::Ok(0i8), GuaranteeStrength::Exact),
    );
    let Resolution::Propagated { error, policy, .. } = r else {
        panic!("rust oracle: no rule must propagate (I1 floor)");
    };

    let call = "handle_classified(mk_err_in(), empty_pol(), BNil, classify_io, attempt_fail)";
    let value_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = res_value({call});");
    let wit_driver =
        format!("{PRELUDE}{ORACLE_REDUCERS}fn main() => Binary{{8}} = res_wit({call});");
    assert_eq!(eval_byte(&value_driver), error, "unchanged error byte (I1)");
    assert_eq!(
        eval_byte(&wit_driver),
        i8::from(policy.is_some()),
        "no-rule path must carry NO acting-policy witness on BOTH sides"
    );
}

/// Budget consume — Rust oracle: `Budgets::consume` overrun names kind/requested/remaining; the
/// `.myc` `Exhausted` fields must match byte-for-byte.
#[test]
fn oracle_budget_consume_overrun_fields_match_rust() {
    let mut b = Budgets::new().with(EffectBudget::Attempts(1));
    assert!(b.consume(EffectKind::Retry, 1).is_ok());
    let err = b
        .consume(EffectKind::Retry, 1)
        .expect_err("drained budget overruns");

    // The .myc side: consume twice from Attempts(1); project the second consume's Exhausted
    // fields. The first consume's Ok ledger threads through the nested match.
    let requested_driver = "fn main() => Binary{8} = match budget_consume(budget_set(BNil, Attempts(0b0000_0001)), EkRetry, 0b0000_0001) { Ok(b2) => match budget_consume(b2, EkRetry, 0b0000_0001) { Ok(_) => 0b0111_1111, Err(x) => match x { Exhausted(_, req, _) => req } }, Err(_) => 0b0111_1110 };";
    let remaining_driver = "fn main() => Binary{8} = match budget_consume(budget_set(BNil, Attempts(0b0000_0001)), EkRetry, 0b0000_0001) { Ok(b2) => match budget_consume(b2, EkRetry, 0b0000_0001) { Ok(_) => 0b0111_1111, Err(x) => match x { Exhausted(_, _, rem) => rem } }, Err(_) => 0b0111_1110 };";
    let kind_driver = "fn main() => Binary{8} = match budget_consume(budget_set(BNil, Attempts(0b0000_0001)), EkRetry, 0b0000_0001) { Ok(b2) => match budget_consume(b2, EkRetry, 0b0000_0001) { Ok(_) => 0b0111_1111, Err(x) => match x { Exhausted(k, _, _) => effect_code(k) } }, Err(_) => 0b0111_1110 };";

    assert_eq!(
        eval_byte(requested_driver),
        i8::try_from(err.requested).expect("fits"),
        "overrun `requested` must match the Rust oracle (I4)"
    );
    assert_eq!(
        eval_byte(remaining_driver),
        i8::try_from(err.remaining).expect("fits"),
        "overrun `remaining` must match the Rust oracle (I4)"
    );
    assert_eq!(
        eval_byte(kind_driver),
        effect_kind_code(&err.kind),
        "overrun `kind` must match the Rust oracle (I4)"
    );
}

/// check_effects — Rust oracle: the FIRST undeclared performed effect is named; kinds must agree
/// through the shared code map (I3).
#[test]
fn oracle_check_effects_undeclared_kind_matches_rust() {
    let declared: EffectSet = [EffectKind::Retry].into_iter().collect();
    let performed: EffectSet = [EffectKind::Io].into_iter().collect();
    let err = check_effects(&declared, &performed).expect_err("Io is undeclared");

    let driver = "fn main() => Binary{8} = match check_effects(ECons(EkRetry, ENil), ECons(EkIo, ENil)) { Ok(_) => 0b0111_1111, Err(u) => match u { Undeclared(k) => effect_code(k) } };";
    assert_eq!(
        eval_byte(driver),
        effect_kind_code(&err.effect),
        "the undeclared effect kind must match the Rust oracle (I3)"
    );
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Guarantee-matrix oracle cases (the `std_diag.rs` precedent): every expected value is computed
// LIVE from `mycelium_std_recover::guarantee_matrix::MATRIX` — a real divergence between the Rust
// source and the `.myc` transcription flips the oracle and fails the case.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Render a Rust `bool` as the `.myc` literal denoting the same `Bool` value.
fn myc_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// The `n`-deep `add_u` chain `matrix_len` expands to (the `std_diag.rs` provenance
/// convention: recompute via the SAME prims, not a bare literal).
fn myc_len_chain(n: u8) -> String {
    let mut expr = "0b0000_0000".to_owned();
    for _ in 0..n {
        expr = format!("add_u(0b0000_0001, {expr})");
    }
    expr
}

/// `matrix_len(matrix())` equals the live Rust oracle's `MATRIX.len()` (11 rows).
#[test]
fn matrix_len_matches_rust_oracle_row_count() {
    let expected_count = u8::try_from(MATRIX.len()).expect("row count fits u8");
    let driver = "fn main() => Binary{8} = matrix_len(matrix());";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\nfn main() => Binary{{8}} = {};",
        myc_len_chain(expected_count)
    );
    assert_three_way("matrix_len == rust MATRIX.len()", &src, &expected);
}

/// Every row states a non-empty `never_silent_property` (C1/I1) — live oracle.
#[test]
fn all_rows_state_never_silent_property_matches_rust_oracle() {
    let expected = MATRIX.iter().all(|r| !r.never_silent_property.is_empty());
    let driver = "fn main() => Bool = all_never_silent_nonempty(matrix());";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way("all never_silent nonempty == rust", &src, &expected_src);
}

/// The driver rows (`handle*`/`recover*`) are `Total` — live oracle (I1/I4).
#[test]
fn driver_rows_are_total_matches_rust_oracle() {
    let expected = MATRIX
        .iter()
        .filter(|r| r.op.starts_with("handle") || r.op.starts_with("recover"))
        .all(|r| r.fallibility == Fallibility::Total);
    let driver = "fn main() => Bool = driver_rows_are_total();";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way("driver rows Total == rust", &src, &expected_src);
}

/// The driver rows carry `PolicyRef` for EXPLAIN (C3) — live oracle.
#[test]
fn driver_rows_are_policy_ref_explainable_matches_rust_oracle() {
    let expected = MATRIX
        .iter()
        .filter(|r| r.op.starts_with("handle") || r.op.starts_with("recover"))
        .all(|r| r.explainable == Explainable::PolicyRef);
    let driver = "fn main() => Bool = driver_rows_are_policy_ref_explainable();";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way("driver rows PolicyRef == rust", &src, &expected_src);
}

/// The typed fallibility column — `on`/`check_effects` FallibleConfig, `consume` FallibleBudget,
/// `policy_ref`/`action_for` Total — live oracle (the ADT-typed half of the Rust exact-ops
/// check; its substring half is FLAGged, see the nodule header).
#[test]
fn config_ops_fallibility_matches_rust_oracle() {
    let find = |op: &str| {
        MATRIX
            .iter()
            .find(|r| r.op == op)
            .unwrap_or_else(|| panic!("rust oracle must contain op {op:?}"))
    };
    let expected = find("on (policy registration)").fallibility == Fallibility::FallibleConfig
        && find("check_effects (I3)").fallibility == Fallibility::FallibleConfig
        && find("consume (budget ledger)").fallibility == Fallibility::FallibleBudget
        && find("policy_ref").fallibility == Fallibility::Total
        && find("action_for").fallibility == Fallibility::Total;
    let driver = "fn main() => Bool = config_ops_fallibility_is_typed();";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way("config/budget fallibility == rust", &src, &expected_src);
}
