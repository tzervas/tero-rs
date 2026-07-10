//! Differential tests for `std.result` (M-649) — the first self-hosted generic stdlib nodule.
//!
//! These tests prove that `std.result` RUNS to closed L0 on all three paths (L1-eval ≡
//! elaborate→L0-interp ≡ AOT) and that each combinator returns the correct reference value.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies only the nodule's `include_str!` (the path is a macro literal, so it stays local) and
//! the per-op cases. The nodule source is loaded verbatim (the single source of truth), then a
//! typed driver `fn` is appended to pin the generic parameters `A` and `E` to `Binary{8}`.
//! Without explicit pinning, the monomorphizer emits a never-silent `Residual` (undetermined type
//! parameters — G2), so every driver uses explicitly-typed helper functions (`mk_ok`, `mk_err`)
//! to carry the full `Result<Binary{8},Binary{8}>` type to the call site.
//!
//! # Honesty tags
//! - **`Declared`** — the type-level contract of each combinator (a structural check, not a theorem).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), validated
//!   by trial on the programs below; not a machine-checked proof.
//!
//! # HOF combinators (M-649 / M-688 capstone)
//! `map`, `and_then`, and `fold` are now **present and executable** (RFC-0024 static
//! defunctionalization, M-685/686/687). The tests below pass a named top-level helper function as
//! the function argument; the monomorphizer specializes the combinator body at the call site
//! (defunctionalization), yielding closed first-order L0. Differential agreement is `Empirical`
//! (trials over the programs below); the type-level contract is `Declared` (VR-5).

mod harness;

/// The std.result nodule source, loaded at compile time — the single source of truth.
const RESULT_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/result.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
/// The driver must supply `mk_ok` / `mk_err` helpers with explicit `Result<Binary{8},Binary{8}>`
/// return types so the monomorphizer can determine both `A` and `E` from the call site.
fn program(driver: &str) -> String {
    harness::program(RESULT_SRC, driver)
}

/// Run the three-way differential on `src` — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT —
/// and assert all three paths agree AND equal the `expected` reference value (a hand-computed
/// `CoreValue` produced by evaluating a trivial reference program through the same path).
///
/// Honesty: differential agreement is `Empirical` (trials); the type-level contract is `Declared`.
/// Thin re-export of the shared [`harness::assert_three_way`] so the per-case bodies below stay
/// unchanged.
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ── is_ok ────────────────────────────────────────────────────────────────────────────────────────

/// `is_ok(Ok(x))` → `True` (Declared: the Ok-arm always returns True).
/// Both A and E are pinned to Binary{8} via explicit return types on mk_ok/mk_err.
#[test]
fn is_ok_on_ok_returns_true() {
    let driver = "fn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Bool = is_ok(mk_ok());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_ok(Ok)", &src, expected);
}

/// `is_ok(Err(e))` → `False` (Declared: the Err-arm always returns False).
#[test]
fn is_ok_on_err_returns_false() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Bool = is_ok(mk_err());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_ok(Err)", &src, expected);
}

// ── is_err ───────────────────────────────────────────────────────────────────────────────────────

/// `is_err(Ok(x))` → `False` (Declared: mirror of is_ok, Ok-arm returns False).
#[test]
fn is_err_on_ok_returns_false() {
    let driver = "fn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Bool = is_err(mk_ok());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_err(Ok)", &src, expected);
}

/// `is_err(Err(e))` → `True` (Declared: mirror of is_ok, Err-arm returns True).
#[test]
fn is_err_on_err_returns_true() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Bool = is_err(mk_err());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_err(Err)", &src, expected);
}

// ── unwrap_or ────────────────────────────────────────────────────────────────────────────────────

/// `unwrap_or(Ok(x), d)` → `x` (Declared: returns the wrapped value, ignores default).
/// Never-silent (G2): the default is caller-supplied; no panic, no sentinel.
#[test]
fn unwrap_or_on_ok_returns_wrapped_value() {
    let driver = "fn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Binary{8} = unwrap_or(mk_ok(), 0b0000_0000);";
    let src = program(driver);
    // Expected: the wrapped value 0b0000_0001, not the default 0b0000_0000.
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way("unwrap_or(Ok)", &src, expected);
}

/// `unwrap_or(Err(e), d)` → `d` (Declared: returns the default, discards the error).
/// Never-silent (G2): the caller-supplied default is the explicit recovery path.
#[test]
fn unwrap_or_on_err_returns_default() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Binary{8} = unwrap_or(mk_err(), 0b0000_0000);";
    let src = program(driver);
    // Expected: the default 0b0000_0000, not the discarded error 0b1111_1111.
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0000;";
    assert_three_way("unwrap_or(Err)", &src, expected);
}

/// Edge case: `unwrap_or(Err(e), d)` where `d = e` — both values are 0b1111_1111.
/// This confirms the combinator returns `d` for the right reason (match-arm, not identity):
/// the reference value is hand-computed as `0b1111_1111`.
#[test]
fn unwrap_or_on_err_with_same_default_as_error() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Binary{8} = unwrap_or(mk_err(), 0b1111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b1111_1111;";
    assert_three_way("unwrap_or(Err, d=e)", &src, expected);
}

// ── map (M-649 / M-688 HOF capstone) ─────────────────────────────────────────────────────────────
//
// `map` is now executable via RFC-0024 static defunctionalization: the named helper `not_val` is
// passed as a first-class value; the monomorphizer specializes `map` at the call site, replacing
// `f(x)` with a direct call to `not_val`. Differential agreement: `Empirical` (trials). Contract:
// `Declared`.
//
// Helper: `not_val(x: Binary{8}) -> Binary{8} = not(x)`.
// Hand-computed expected values:
//   map(Ok(0b0000_0001), not_val) → Ok(not(0b0000_0001)) = Ok(0b1111_1110)
//   map(Err(0b1111_1111), not_val) → Err(0b1111_1111)  [Err passes through]

/// `map(Ok(x), not_val)` → `Ok(not(x))` — the success value is transformed.
/// `not_val` is a named top-level fn, defunctionalized at the call site (RFC-0024 §4, Declared).
/// Hand-computed: not(0b0000_0001) = 0b1111_1110.
#[test]
fn map_on_ok_applies_function_to_success_value() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_ok(), not_val);";
    let src = program(driver);
    // Expected: Ok wrapping the computed value not(0b0000_0001). The reference program computes via
    // not() to match the Derived provenance of the test result (literal 0b1111_1110 would be Root).
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(not(0b0000_0001));";
    assert_three_way("map(Ok, not_val)", &src, expected);
}

/// `map(Err(e), not_val)` → `Err(e)` — the error passes through untouched (never-silent, G2).
/// Hand-computed: Err(0b1111_1111) with no transformation applied.
#[test]
fn map_on_err_passes_through_untouched() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_err(), not_val);";
    let src = program(driver);
    // Expected: the original Err is preserved; not_val is not applied.
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("map(Err, not_val)", &src, expected);
}

// ── and_then (M-649 / M-688 HOF capstone) ────────────────────────────────────────────────────────
//
// `and_then` sequences a Result-returning step. Helper: `mk_ok_inner(x) = Ok(not(x))`.
// Hand-computed expected values:
//   and_then(Ok(0b0000_0001), mk_ok_inner) → Ok(not(0b0000_0001)) = Ok(0b1111_1110)
//   and_then(Err(0b1111_1111), mk_ok_inner) → Err(0b1111_1111)  [Err short-circuits]

/// `and_then(Ok(x), mk_ok_inner)` → `Ok(not(x))` — the step is applied on success.
/// `mk_ok_inner` wraps `not(x)` in an Ok; defunctionalized at the call site (RFC-0024 §4, Declared).
/// Hand-computed: Ok(not(0b0000_0001)) = Ok(0b1111_1110).
#[test]
fn and_then_on_ok_chains_the_step() {
    let driver = "fn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = and_then(mk_ok(), mk_ok_inner);";
    let src = program(driver);
    // Expected: the chained step returns Ok(not(0b0000_0001)). The reference program computes via
    // not() to match the Derived provenance of the test result (literal would be Root).
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(not(0b0000_0001));";
    assert_three_way("and_then(Ok, mk_ok_inner)", &src, expected);
}

/// `and_then(Err(e), mk_ok_inner)` → `Err(e)` — the Err short-circuits; the step is not applied.
/// Hand-computed: Err(0b1111_1111) unchanged.
#[test]
fn and_then_on_err_short_circuits() {
    let driver = "fn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = and_then(mk_err(), mk_ok_inner);";
    let src = program(driver);
    // Expected: the Err propagates unchanged; mk_ok_inner is never called.
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("and_then(Err, mk_ok_inner)", &src, expected);
}

// ── fold (M-649 / M-688 HOF capstone) ────────────────────────────────────────────────────────────
//
// `fold` eliminates a Result to a common type B via two single-arg fns (the catamorphism).
// Helpers:
//   `id_val(x: Binary{8}) -> Binary{8} = x`           (identity — returns the success value)
//   `const_zero(e: Binary{8}) -> Binary{8} = xor(e, e)` (any XOR itself = 0b0000_0000)
// Hand-computed expected values:
//   fold(Ok(0b1010_1010), id_val, const_zero) → id_val(0b1010_1010) = 0b1010_1010
//   fold(Err(0b1111_0000), id_val, const_zero) → const_zero(0b1111_0000) = xor(0b1111_0000, 0b1111_0000) = 0b0000_0000

/// `fold(Ok(x), id_val, const_zero)` → `x` — the on_ok branch is taken.
/// Hand-computed: id_val(0b1010_1010) = 0b1010_1010.
#[test]
fn fold_on_ok_applies_on_ok_branch() {
    let driver = "fn id_val(x: Binary{8}) => Binary{8} = x;\nfn const_zero(e: Binary{8}) => Binary{8} = xor(e, e);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b1010_1010);\nfn main() => Binary{8} = fold(mk_ok(), id_val, const_zero);";
    let src = program(driver);
    // Expected: the success value 0b1010_1010 (on_ok branch; const_zero never called).
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b1010_1010;";
    assert_three_way("fold(Ok, id_val, const_zero)", &src, expected);
}

/// `fold(Err(e), id_val, const_zero)` → `xor(e, e) = 0b0000_0000` — the on_err branch is taken.
/// Hand-computed: xor(0b1111_0000, 0b1111_0000) = 0b0000_0000.
#[test]
fn fold_on_err_applies_on_err_branch() {
    let driver = "fn id_val(x: Binary{8}) => Binary{8} = x;\nfn const_zero(e: Binary{8}) => Binary{8} = xor(e, e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_0000);\nfn main() => Binary{8} = fold(mk_err(), id_val, const_zero);";
    let src = program(driver);
    // Expected: xor(0b1111_0000, 0b1111_0000). The reference program computes via xor() to match the
    // Derived provenance of the test result (literal 0b0000_0000 would be Root).
    let expected = "nodule ref;\nfn main() => Binary{8} = xor(0b1111_0000, 0b1111_0000);";
    assert_three_way("fold(Err, id_val, const_zero)", &src, expected);
}

// ── map_err (M-715 — the Err-side mirror of map) ──────────────────────────────────────────────────
//
// Helper: `not_val(e) = not(e)`. The error is transformed; an Ok passes through untouched.
//   map_err(Ok(0b0000_0001), not_val) → Ok(0b0000_0001)  [Ok preserved]
//   map_err(Err(0b0000_1111), not_val) → Err(not(0b0000_1111)) = Err(0b1111_0000)

/// `map_err(Ok(x), not_val)` → `Ok(x)` — the success value passes through; the error fn is not run.
#[test]
fn map_err_on_ok_passes_through() {
    let driver = "fn not_val(e: Binary{8}) => Binary{8} = not(e);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map_err(mk_ok(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("map_err(Ok, not_val)", &src, expected);
}

/// `map_err(Err(e), not_val)` → `Err(not(e))` — the error is transformed.
/// Hand-computed: not(0b0000_1111) = 0b1111_0000.
#[test]
fn map_err_on_err_transforms_error() {
    let driver = "fn not_val(e: Binary{8}) => Binary{8} = not(e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Result[Binary{8},Binary{8}] = map_err(mk_err(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(not(0b0000_1111));";
    assert_three_way("map_err(Err, not_val)", &src, expected);
}

// ── or_else (M-715 — the Err-side bind, dual of and_then) ─────────────────────────────────────────
//
// Helper: `recover(e) = Ok(not(e))` — a recovery step that turns an error into a success.
//   or_else(Ok(0b0000_0001), recover) → Ok(0b0000_0001)  [Ok kept; recover not run]
//   or_else(Err(0b0000_1111), recover) → Ok(not(0b0000_1111)) = Ok(0b1111_0000)

/// `or_else(Ok(x), recover)` → `Ok(x)` — the success value is kept; the recovery step is not run.
#[test]
fn or_else_on_ok_keeps_value() {
    let driver = "fn recover(e: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(e));\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = or_else(mk_ok(), recover);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("or_else(Ok, recover)", &src, expected);
}

/// `or_else(Err(e), recover)` → `recover(e) = Ok(not(e))` — the recovery step runs on the error.
/// Hand-computed: Ok(not(0b0000_1111)) = Ok(0b1111_0000).
#[test]
fn or_else_on_err_runs_recovery() {
    let driver = "fn recover(e: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(e));\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Result[Binary{8},Binary{8}] = or_else(mk_err(), recover);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(not(0b0000_1111));";
    assert_three_way("or_else(Err, recover)", &src, expected);
}
