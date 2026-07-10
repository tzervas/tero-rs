//! Differential tests for `std.error` (M-931, E29-1, kickoff `opp`) — the errors-as-values
//! ergonomics layer over `Result[A,E]`/`Option[A]`.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-op three-way cases (mirroring
//! `std_result.rs`/`std_option.rs`), and — the row this port owns per the harness doc (§4) —
//! [`extract_byte`]-based comparisons against the **retained Rust oracle**,
//! `mycelium-std-error` (RFC-0031 D6; the crate is NOT retired).
//!
//! # Surface-check (D5 row 1) and substitutions
//! See `lib/std/error.myc`'s header comment for the full surface-check: 15 of the crate's 21+1
//! ops are ported (`map`/`map_err`/`and_then`/`or_else`/`filter`/`inspect`/`inspect_err`/`ok_or`/
//! `ok_or_else`/`ok`/`flatten`/`unwrap_or`/`unwrap_or_else`/`unwrap_or_option`/
//! `unwrap_or_else_option`). FLAGGED, not forced (VR-5/G2): `unwrap`/`expect`/`unwrap_err` (no
//! host panic/refusal primitive); the RFC-0014 `recover` bridge (depends on `std.recover`'s own
//! unlanded `.myc` port, M-930, plus the kernel-only `PolicyRef` type); and `transpose`/`zip`
//! (FLAG-error-3, discovered during this port: the v0 checker refuses a function reusing its own
//! type parameter at two different instantiation depths of the same polymorphic sum type in one
//! body).
//!
//! # Honesty tags
//! - **`Declared`** — the type-level contract of each ported combinator (a structural check, not
//!   a theorem) — carried at the SAME strength as `mycelium-std-error`'s own guarantee matrix
//!   (VR-5: never upgraded in translation).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) AND the
//!   Rust-oracle differential below, both validated by trial on the programs in this file; neither
//!   is a machine-checked proof.

mod harness;

use mycelium_core::{binary::bits_to_int, CoreValue, Payload};

/// The std.error nodule source, loaded at compile time — the single source of truth.
const ERROR_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/error.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(ERROR_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`] so the per-case bodies below stay
/// unchanged (same pattern as `std_result.rs`/`std_option.rs`).
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Three-way differential cases (L1-eval ≡ elaborate→L0-interp ≡ AOT), one section per ported op.
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ── map / map_err ────────────────────────────────────────────────────────────────────────────────

/// `map(Ok(x), not_val)` → `Ok(not(x))` — the success value is transformed.
#[test]
fn map_on_ok_applies_function() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_ok(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(not(0b0000_0001));";
    assert_three_way("map(Ok, not_val)", &src, expected);
}

/// `map(Err(e), not_val)` → `Err(e)` — the error passes through untouched (never-silent, G2).
#[test]
fn map_on_err_passes_through() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_err(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("map(Err, not_val)", &src, expected);
}

/// `map_err(Ok(x), not_val)` → `Ok(x)` — the success value passes through untouched.
#[test]
fn map_err_on_ok_passes_through() {
    let driver = "fn not_val(e: Binary{8}) => Binary{8} = not(e);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map_err(mk_ok(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("map_err(Ok, not_val)", &src, expected);
}

/// `map_err(Err(e), not_val)` → `Err(not(e))` — the error is transformed.
#[test]
fn map_err_on_err_transforms_error() {
    let driver = "fn not_val(e: Binary{8}) => Binary{8} = not(e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Result[Binary{8},Binary{8}] = map_err(mk_err(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(not(0b0000_1111));";
    assert_three_way("map_err(Err, not_val)", &src, expected);
}

// ── and_then / or_else ───────────────────────────────────────────────────────────────────────────

/// `and_then(Ok(x), mk_ok_inner)` → `Ok(not(x))` — the step is applied on success.
#[test]
fn and_then_on_ok_chains_the_step() {
    let driver = "fn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = and_then(mk_ok(), mk_ok_inner);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(not(0b0000_0001));";
    assert_three_way("and_then(Ok, mk_ok_inner)", &src, expected);
}

/// `and_then(Err(e), mk_ok_inner)` → `Err(e)` — the Err short-circuits.
#[test]
fn and_then_on_err_short_circuits() {
    let driver = "fn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = and_then(mk_err(), mk_ok_inner);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("and_then(Err, mk_ok_inner)", &src, expected);
}

/// `or_else(Ok(x), recover)` → `Ok(x)` — the success value is kept; recovery not run.
#[test]
fn or_else_on_ok_keeps_value() {
    let driver = "fn recover(e: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(e));\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = or_else(mk_ok(), recover);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("or_else(Ok, recover)", &src, expected);
}

/// `or_else(Err(e), recover)` → `recover(e) = Ok(not(e))` — the recovery step runs.
#[test]
fn or_else_on_err_runs_recovery() {
    let driver = "fn recover(e: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(e));\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Result[Binary{8},Binary{8}] = or_else(mk_err(), recover);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(not(0b0000_1111));";
    assert_three_way("or_else(Err, recover)", &src, expected);
}

// ── filter (Option) ─────────────────────────────────────────────────────────────────────────────

/// `filter(Some(x), pred)` → `Some(x)` when `pred(x)` is `True` — a typed transition, not a drop.
#[test]
fn filter_on_some_true_keeps_value() {
    let driver = "fn is_high(x: Binary{8}) => Bool = match lt(0b0111_1111, x) { 0b1 => True, _ => False };\nfn mk_some() => Option[Binary{8}] = Some(0b1111_1111);\nfn main() => Option[Binary{8}] = filter(mk_some(), is_high);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b1111_1111);";
    assert_three_way("filter(Some, True)", &src, expected);
}

/// `filter(Some(x), pred)` → `None` when `pred(x)` is `False` — the typed absence.
#[test]
fn filter_on_some_false_yields_none() {
    let driver = "fn is_high(x: Binary{8}) => Bool = match lt(0b0111_1111, x) { 0b1 => True, _ => False };\nfn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Option[Binary{8}] = filter(mk_some(), is_high);";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = None;";
    assert_three_way("filter(Some, False)", &src, expected);
}

/// `filter(None, pred)` → `None` unconditionally (the predicate is never applied).
#[test]
fn filter_on_none_is_always_none() {
    let driver = "fn is_high(x: Binary{8}) => Bool = match lt(0b0111_1111, x) { 0b1 => True, _ => False };\nfn mk_none() => Option[Binary{8}] = None;\nfn main() => Option[Binary{8}] = filter(mk_none(), is_high);";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = None;";
    assert_three_way("filter(None)", &src, expected);
}

// ── inspect / inspect_err ────────────────────────────────────────────────────────────────────────

/// `inspect(Ok(x), f)` → `Ok(x)` — the value and sum shape are unchanged (a structural peek).
#[test]
fn inspect_on_ok_leaves_value_unchanged() {
    let driver = "fn peek(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = inspect(mk_ok(), peek);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("inspect(Ok)", &src, expected);
}

/// `inspect_err(Err(e), f)` → `Err(e)` — the error and propagation are unchanged.
#[test]
fn inspect_err_on_err_leaves_error_unchanged() {
    let driver = "fn peek(e: Binary{8}) => Binary{8} = not(e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = inspect_err(mk_err(), peek);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("inspect_err(Err)", &src, expected);
}

// ── ok_or / ok_or_else (Option → Result) ─────────────────────────────────────────────────────────

/// `ok_or(Some(x), err)` → `Ok(x)`.
#[test]
fn ok_or_on_some_returns_ok() {
    let driver = "fn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = ok_or(mk_some(), 0b1111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("ok_or(Some)", &src, expected);
}

/// `ok_or(None, err)` → `Err(err)` — the absence is explicitly named (never a silent drop).
#[test]
fn ok_or_on_none_names_absence_as_err() {
    let driver = "fn mk_none() => Option[Binary{8}] = None;\nfn main() => Result[Binary{8},Binary{8}] = ok_or(mk_none(), 0b1111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("ok_or(None)", &src, expected);
}

/// `ok_or_else(None, f)` → `Err(f(U))` — the lazily-computed error (Unit substitution).
#[test]
fn ok_or_else_on_none_computes_error() {
    let driver = "fn lazy_err(_u: Unit) => Binary{8} = not(0b0000_0000);\nfn mk_none() => Option[Binary{8}] = None;\nfn main() => Result[Binary{8},Binary{8}] = ok_or_else(mk_none(), lazy_err);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(not(0b0000_0000));";
    assert_three_way("ok_or_else(None)", &src, expected);
}

/// `ok_or_else(Some(x), f)` → `Ok(x)` — `f` is never called (never observable here since `.myc`
/// has no effect surface, but the match arm structurally never selects the `None` branch).
#[test]
fn ok_or_else_on_some_returns_ok() {
    let driver = "fn lazy_err(_u: Unit) => Binary{8} = not(0b0000_0000);\nfn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = ok_or_else(mk_some(), lazy_err);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("ok_or_else(Some)", &src, expected);
}

// ── ok (Result → Option, FLAGGED lossy conversion) ──────────────────────────────────────────────

/// `ok(Ok(x))` → `Some(x)`.
#[test]
fn ok_on_ok_returns_some() {
    let driver = "fn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Option[Binary{8}] = ok(mk_ok());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = Some(0b0000_0001);";
    assert_three_way("ok(Ok)", &src, expected);
}

/// `ok(Err(e))` → `None` — the flagged lossy conversion (spec §7-Q2); the loss is explicit.
#[test]
fn ok_on_err_returns_none() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Option[Binary{8}] = ok(mk_err());";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] = None;";
    assert_three_way("ok(Err)", &src, expected);
}

// ── flatten ──────────────────────────────────────────────────────────────────────────────────────
//
// (transpose is NOT ported — see `lib/std/error.myc`'s FLAG-error-3: the v0 checker refuses a
// function that reuses one of its own type parameters at two different instantiation depths of
// the same polymorphic sum type within one body. `transpose[A,E](Option[Result[A,E]]) ->
// Result[Option[A],E]` hits exactly that — reproduced and documented in the nodule.)

/// `flatten(Ok(Ok(x)))` → `Ok(x)`.
#[test]
fn flatten_ok_ok() {
    let driver = "fn mk() => Result[Result[Binary{8},Binary{8}],Binary{8}] = Ok(Ok(0b0000_0001));\nfn main() => Result[Binary{8},Binary{8}] = flatten(mk());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);";
    assert_three_way("flatten(Ok(Ok))", &src, expected);
}

/// `flatten(Ok(Err(e)))` → `Err(e)` — the inner Err propagates to the outer.
#[test]
fn flatten_ok_err_propagates_inner() {
    let driver = "fn mk() => Result[Result[Binary{8},Binary{8}],Binary{8}] = Ok(Err(0b1111_1111));\nfn main() => Result[Binary{8},Binary{8}] = flatten(mk());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);";
    assert_three_way("flatten(Ok(Err))", &src, expected);
}

/// `flatten(Err(e))` → `Err(e)` — the outer Err propagates; no wrapping discarded.
#[test]
fn flatten_err_propagates_outer() {
    let driver = "fn mk() => Result[Result[Binary{8},Binary{8}],Binary{8}] = Err(0b0000_1111);\nfn main() => Result[Binary{8},Binary{8}] = flatten(mk());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Result[A,E] = Ok(A) | Err(E);\nfn main() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);";
    assert_three_way("flatten(Err)", &src, expected);
}

// (zip is NOT ported — see `lib/std/error.myc`'s FLAG-error-3: `zip[A,B](Option[A],Option[B]) ->
// Option[Pair[A,B]]` reuses `Option`'s own type parameter at two different instantiation depths
// (`A` vs `Pair<A,B>`) within one body, hitting the identical v0-checker refusal as `transpose`.)

// ── unwrap_or / unwrap_or_else (Result) ──────────────────────────────────────────────────────────

/// `unwrap_or(Ok(x), d)` → `x` — the wrapped value, not the default.
#[test]
fn unwrap_or_on_ok_returns_wrapped_value() {
    let driver = "fn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Binary{8} = unwrap_or(mk_ok(), 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way("unwrap_or(Ok)", &src, expected);
}

/// `unwrap_or(Err(e), d)` → `d` — the default, discarding the error.
#[test]
fn unwrap_or_on_err_returns_default() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Binary{8} = unwrap_or(mk_err(), 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0000;";
    assert_three_way("unwrap_or(Err)", &src, expected);
}

/// `unwrap_or_else(Err(e), f)` → `f(e)` — the computed fallback.
#[test]
fn unwrap_or_else_on_err_computes_fallback() {
    let driver = "fn not_val(e: Binary{8}) => Binary{8} = not(e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Binary{8} = unwrap_or_else(mk_err(), not_val);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = not(0b0000_1111);";
    assert_three_way("unwrap_or_else(Err)", &src, expected);
}

// ── unwrap_or_option / unwrap_or_else_option (Option) ───────────────────────────────────────────

/// `unwrap_or_option(Some(x), d)` → `x`.
#[test]
fn unwrap_or_option_on_some_returns_value() {
    let driver = "fn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Binary{8} = unwrap_or_option(mk_some(), 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way("unwrap_or_option(Some)", &src, expected);
}

/// `unwrap_or_option(None, d)` → `d`.
#[test]
fn unwrap_or_option_on_none_returns_default() {
    let driver = "fn mk_none() => Option[Binary{8}] = None;\nfn main() => Binary{8} = unwrap_or_option(mk_none(), 0b1111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b1111_1111;";
    assert_three_way("unwrap_or_option(None)", &src, expected);
}

/// `unwrap_or_else_option(None, f)` → `f(U)` — the computed default (Unit substitution).
#[test]
fn unwrap_or_else_option_on_none_computes_default() {
    let driver = "fn lazy_default(_u: Unit) => Binary{8} = not(0b0000_0000);\nfn mk_none() => Option[Binary{8}] = None;\nfn main() => Binary{8} = unwrap_or_else_option(mk_none(), lazy_default);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = not(0b0000_0000);";
    assert_three_way("unwrap_or_else_option(None)", &src, expected);
}

/// `unwrap_or_else_option(Some(x), f)` → `x`.
#[test]
fn unwrap_or_else_option_on_some_returns_value() {
    let driver = "fn lazy_default(_u: Unit) => Binary{8} = not(0b0000_0000);\nfn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Binary{8} = unwrap_or_else_option(mk_some(), lazy_default);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way("unwrap_or_else_option(Some)", &src, expected);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Rust-oracle differential (D5 row 4) — wired against the RETAINED `mycelium-std-error` crate
// (RFC-0031 D6: the crate is NOT retired). Every driver below reduces its combinator's output to
// a raw `Binary{8}` via a local `match` (never `unwrap`, which this port does not have — see the
// nodule's FLAG-error-1), then [`extract_byte`] decodes the L1-eval CoreValue the same way the L1
// evaluator/AOT paths already do (`mycelium_core::binary::bits_to_int`, MSB-first two's-complement
// — the same codec std.math.myc's differential relies on), so it is compared directly against the
// Rust oracle's `i8` output. The M-931 DoD calls out **error-formatting parity**: the `map_err`/
// `ok_or` cases below additionally assert the *formatted* (`{:08b}`) representation of the
// transformed/named error value is identical between the `.myc` result and the Rust oracle's —
// not just the raw byte — so a formatting-level regression (e.g. a byte-order swap that happened
// to still compare numerically equal) would also be caught.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Decode a `Binary{8}` [`CoreValue`] to its signed byte, MSB-first two's-complement — the same
/// codec [`bits_to_int`] gives the L1 evaluator/AOT paths (`crates/mycelium-core/src/binary.rs`).
fn extract_byte(cv: &CoreValue) -> i8 {
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Binary{{8}} repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bits(bits) => bits_to_int(bits) as i8,
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Run `driver`'s `main` (a raw `Binary{8}` — no sum wrapper) through the L1 evaluator and return
/// the decoded byte. Reuses the same monomorphize/build_registry/eval path as
/// [`harness::assert_three_way`], but returns the decoded scalar instead of asserting three-way
/// agreement (the three-way obligation is already covered by the cases above; this helper is only
/// for bridging to the Rust oracle).
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

/// `map` — Rust oracle: `mycelium_std_error::map(Ok(1i8), |x| !x)` = `Ok(-2)`. `.myc` side: the
/// same driver as [`map_on_ok_applies_function`], reduced to a raw byte via a local match.
#[test]
fn oracle_map_on_ok_matches_rust() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Binary{8} = match map(mk_ok(), not_val) { Ok(x) => x, Err(e) => e };";
    let myc_byte = eval_byte(driver);

    let rust_result: Result<i8, i8> = mycelium_std_error::map(Ok(1i8), |x: i8| !x);
    let rust_byte = rust_result.expect("Ok stays Ok");

    assert_eq!(
        myc_byte, rust_byte,
        "map(Ok) must match the Rust oracle byte-for-byte"
    );
}

/// `map_err` (error-formatting parity case) — Rust oracle:
/// `mycelium_std_error::map_err(Err(15i8), |e| !e)` = `Err(-16)`. Compares BOTH the raw byte and
/// its `{:08b}` formatted representation between the `.myc` result and the Rust oracle's error
/// value — the M-931 DoD's explicit "error-formatting parity" obligation.
#[test]
fn oracle_map_err_on_err_matches_rust_including_formatting() {
    let driver = "fn not_val(e: Binary{8}) => Binary{8} = not(e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Binary{8} = match map_err(mk_err(), not_val) { Ok(x) => x, Err(e) => e };";
    let myc_byte = eval_byte(driver);

    let rust_result: Result<i8, i8> = mycelium_std_error::map_err(Err(15i8), |e: i8| !e);
    let rust_err = rust_result.expect_err("Err stays Err");

    assert_eq!(
        myc_byte, rust_err,
        "map_err(Err) must match the Rust oracle byte-for-byte"
    );
    // Error-formatting parity: the two implementations' error-byte formatted representations
    // (8-bit binary string, matching the source literal's `0b…` presentation) must agree exactly.
    assert_eq!(
        format!("{:08b}", myc_byte as u8),
        format!("{:08b}", rust_err as u8),
        "map_err's transformed error value must format identically (.myc vs Rust oracle) — \
         error-formatting parity (M-931 DoD)"
    );
}

/// `and_then` — Rust oracle: `mycelium_std_error::and_then(Ok(1i8), |x| Ok(!x))` = `Ok(-2)`.
#[test]
fn oracle_and_then_on_ok_matches_rust() {
    let driver = "fn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Binary{8} = match and_then(mk_ok(), mk_ok_inner) { Ok(x) => x, Err(e) => e };";
    let myc_byte = eval_byte(driver);

    let rust_result: Result<i8, i8> = mycelium_std_error::and_then(Ok(1i8), |x: i8| Ok(!x));
    let rust_byte = rust_result.expect("Ok stays Ok via the chained step");

    assert_eq!(
        myc_byte, rust_byte,
        "and_then(Ok) must match the Rust oracle byte-for-byte"
    );
}

/// `or_else` — Rust oracle: `mycelium_std_error::or_else(Err(15i8), |e| Ok(!e))` = `Ok(-16)`.
#[test]
fn oracle_or_else_on_err_matches_rust() {
    let driver = "fn recover(e: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(e));\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Binary{8} = match or_else(mk_err(), recover) { Ok(x) => x, Err(e) => e };";
    let myc_byte = eval_byte(driver);

    let rust_result: Result<i8, i8> = mycelium_std_error::or_else(Err(15i8), |e: i8| Ok(!e));
    let rust_byte = rust_result.expect("Err recovers to Ok");

    assert_eq!(
        myc_byte, rust_byte,
        "or_else(Err) must match the Rust oracle byte-for-byte"
    );
}

/// `ok_or` (error-formatting parity case) — `None` names its absence as the SAME error byte
/// `mycelium_std_error::ok_or` would (`ok_or(None, 0b1111_1111)`). Named-absence bytes must be
/// formatted identically between the two implementations.
#[test]
fn oracle_ok_or_on_none_matches_rust_including_formatting() {
    let driver = "fn mk_none() => Option[Binary{8}] = None;\nfn main() => Binary{8} = match ok_or(mk_none(), 0b1111_1111) { Ok(x) => x, Err(e) => e };";
    let myc_byte = eval_byte(driver);

    let rust_result: Result<i8, i8> = mycelium_std_error::ok_or(None, -1i8);
    let rust_err = rust_result.expect_err("None names Err(err)");

    assert_eq!(
        myc_byte, rust_err,
        "ok_or(None) must name the SAME error byte as the Rust oracle"
    );
    assert_eq!(
        format!("{:08b}", myc_byte as u8),
        format!("{:08b}", rust_err as u8),
        "ok_or's named-absence error value must format identically (.myc vs Rust oracle) — \
         error-formatting parity (M-931 DoD)"
    );
}

/// `ok` (the flagged lossy conversion) — `mycelium_std_error::ok(Ok(1i8))` = `Some(1)`; the
/// surviving Ok-side value must still match byte-for-byte across the lossy boundary.
#[test]
fn oracle_ok_lossy_conversion_preserves_surviving_value() {
    let driver = "fn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Binary{8} = match ok(mk_ok()) { Some(x) => x, None => 0b0000_0000 };";
    let myc_byte = eval_byte(driver);

    let rust_result: Option<i8> = mycelium_std_error::ok(Ok::<i8, i8>(1i8));
    let rust_byte = rust_result.expect("Ok survives the lossy conversion as Some");

    assert_eq!(
        myc_byte, rust_byte,
        "ok(Ok) surviving value must match the Rust oracle byte-for-byte"
    );
}

/// `unwrap_or` — Rust oracle: `mycelium_std_error::unwrap_or(Err(15i8), 0i8).0` = `0`.
#[test]
fn oracle_unwrap_or_on_err_matches_rust() {
    let driver = "fn mk_err() => Result[Binary{8},Binary{8}] = Err(0b0000_1111);\nfn main() => Binary{8} = unwrap_or(mk_err(), 0b0000_0000);";
    let myc_byte = eval_byte(driver);

    let (rust_byte, record) = mycelium_std_error::unwrap_or(Err::<i8, i8>(15), 0i8);
    assert_eq!(
        record.guarantee_tag, "Declared",
        "VR-5: unwrap_or is Declared, never Exact"
    );

    assert_eq!(
        myc_byte, rust_byte,
        "unwrap_or(Err) must match the Rust oracle byte-for-byte"
    );
}
