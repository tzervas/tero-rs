//! Differential tests for `std.option` (M-715) — the self-hosted `Option<A>` core nodule.
//!
//! Sibling of `std_result.rs`: both share the [`harness`] fixture (M-925). The nodule source is
//! loaded verbatim via `include_str!` (the single source of truth; the path is a macro literal, so
//! the load itself stays local to this file), then a typed driver `fn` is appended to pin the
//! generic parameter `A` to `Binary{8}`. Without explicit pinning the monomorphizer emits a
//! never-silent `Residual` (undetermined type parameter — G2), so every driver carries the full
//! `Option<Binary{8}>` type to the call site via explicitly-typed helpers (`mk_some`, `mk_none`).
//!
//! # Honesty tags
//! - **`Exact`** — the `Some`/`None` constructors and the total Bool discriminators `is_some`/
//!   `is_none` (total, RFC-0016 / core spec §3).
//! - **`Declared`** — the type-level contract of each value combinator (`unwrap_or`/`map`/`and_then`/
//!   `fold`/`or_else`/`flatten`) — a structural check, not a theorem.
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), validated
//!   by trial on the programs below; not a machine-checked proof.

mod harness;

/// The std.option nodule source, loaded at compile time — the single source of truth.
const OPTION_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/option.myc"
));

/// Build a full test program by appending a typed driver to the nodule source. The driver supplies
/// `mk_some` / `mk_none` helpers with explicit `Option<Binary{8}>` return types so the monomorphizer
/// can determine `A` from the call site.
fn program(driver: &str) -> String {
    harness::program(OPTION_SRC, driver)
}

/// Run the three-way differential on `src` — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT — and assert
/// all three paths agree AND equal the `expected` reference value.
///
/// Honesty: differential agreement is `Empirical` (trials); the type-level contract is `Declared`.
/// Thin re-export of the shared [`harness::assert_three_way`] so the per-case bodies below stay
/// unchanged.
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ── is_some / is_none ──────────────────────────────────────────────────────────────────────────────

/// `is_some(Some(x))` → `True` (Exact: the Some-arm always returns True).
#[test]
fn is_some_on_some_returns_true() {
    let driver = "fn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Bool =is_some(mk_some());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool =True;";
    assert_three_way("is_some(Some)", &src, expected);
}

/// `is_some(None)` → `False`. `A` is pinned to Binary{8} via the explicit return type on mk_none.
#[test]
fn is_some_on_none_returns_false() {
    let driver =
        "fn mk_none() => Option[Binary{8}] = None;\nfn main() => Bool =is_some(mk_none());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool =False;";
    assert_three_way("is_some(None)", &src, expected);
}

/// `is_none(Some(x))` → `False` (mirror of is_some).
#[test]
fn is_none_on_some_returns_false() {
    let driver = "fn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Bool =is_none(mk_some());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool =False;";
    assert_three_way("is_none(Some)", &src, expected);
}

/// `is_none(None)` → `True`.
#[test]
fn is_none_on_none_returns_true() {
    let driver =
        "fn mk_none() => Option[Binary{8}] = None;\nfn main() => Bool =is_none(mk_none());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool =True;";
    assert_three_way("is_none(None)", &src, expected);
}

// ── unwrap_or ──────────────────────────────────────────────────────────────────────────────────────

/// `unwrap_or(Some(x), d)` → `x` (Declared: returns the held value, ignores the fallback).
#[test]
fn unwrap_or_on_some_returns_held_value() {
    let driver = "fn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Binary{8} =unwrap_or(mk_some(), 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} =0b0000_0001;";
    assert_three_way("unwrap_or(Some)", &src, expected);
}

/// `unwrap_or(None, d)` → `d` (Never-silent G2: the caller-supplied fallback is the explicit
/// recovery path; None never becomes a fabricated value).
#[test]
fn unwrap_or_on_none_returns_fallback() {
    let driver = "fn mk_none() => Option[Binary{8}] = None;\nfn main() => Binary{8} =unwrap_or(mk_none(), 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} =0b0000_0000;";
    assert_three_way("unwrap_or(None)", &src, expected);
}

// ── map ────────────────────────────────────────────────────────────────────────────────────────────
//
// `map` is executable via RFC-0024 static defunctionalization: the named helper `not_val` is passed
// as a first-class value; the monomorphizer specializes `map` at the call site. Hand-computed:
//   map(Some(0b0000_0001), not_val) → Some(not(0b0000_0001)) = Some(0b1111_1110)
//   map(None, not_val) → None  [None passes through]

/// `map(Some(x), not_val)` → `Some(not(x))` — the held value is transformed.
#[test]
fn map_on_some_applies_function() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Option[Binary{8}] =map(mk_some(), not_val);";
    let src = program(driver);
    // Compute via not() so the reference shares the Derived provenance of the test result.
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =Some(not(0b0000_0001));";
    assert_three_way("map(Some, not_val)", &src, expected);
}

/// `map(None, not_val)` → `None` — the empty case passes through untouched (never-silent, G2).
#[test]
fn map_on_none_passes_through() {
    let driver = "fn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_none() => Option[Binary{8}] = None;\nfn main() => Option[Binary{8}] =map(mk_none(), not_val);";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =None;";
    assert_three_way("map(None, not_val)", &src, expected);
}

// ── and_then ───────────────────────────────────────────────────────────────────────────────────────
//
// Helper: `mk_some_inner(x) = Some(not(x))`.
//   and_then(Some(0b0000_0001), mk_some_inner) → Some(not(0b0000_0001)) = Some(0b1111_1110)
//   and_then(None, mk_some_inner) → None  [None short-circuits]

/// `and_then(Some(x), mk_some_inner)` → `Some(not(x))` — the step is applied on Some.
#[test]
fn and_then_on_some_chains_step() {
    let driver = "fn mk_some_inner(x: Binary{8}) => Option[Binary{8}] = Some(not(x));\nfn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn main() => Option[Binary{8}] =and_then(mk_some(), mk_some_inner);";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =Some(not(0b0000_0001));";
    assert_three_way("and_then(Some, mk_some_inner)", &src, expected);
}

/// `and_then(None, mk_some_inner)` → `None` — the empty case short-circuits; the step is not applied.
#[test]
fn and_then_on_none_short_circuits() {
    let driver = "fn mk_some_inner(x: Binary{8}) => Option[Binary{8}] = Some(not(x));\nfn mk_none() => Option[Binary{8}] = None;\nfn main() => Option[Binary{8}] =and_then(mk_none(), mk_some_inner);";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =None;";
    assert_three_way("and_then(None, mk_some_inner)", &src, expected);
}

// ── fold ───────────────────────────────────────────────────────────────────────────────────────────
//
// `fold` eliminates an Option to a B via on_some (A -> B) and on_none (a B value).
// Helper: `id_val(x) = x`.
//   fold(Some(0b1010_1010), id_val, 0b0000_0000) → id_val(0b1010_1010) = 0b1010_1010
//   fold(None, id_val, 0b0000_0000) → 0b0000_0000  [on_none default]

/// `fold(Some(x), id_val, d)` → `x` — the on_some branch is taken.
#[test]
fn fold_on_some_applies_on_some_branch() {
    let driver = "fn id_val(x: Binary{8}) => Binary{8} = x;\nfn mk_some() => Option[Binary{8}] = Some(0b1010_1010);\nfn main() => Binary{8} =fold(mk_some(), id_val, 0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} =0b1010_1010;";
    assert_three_way("fold(Some, id_val, d)", &src, expected);
}

/// `fold(None, id_val, d)` → `d` — the on_none default is returned (never-silent, G2).
#[test]
fn fold_on_none_returns_default() {
    let driver = "fn id_val(x: Binary{8}) => Binary{8} = x;\nfn mk_none() => Option[Binary{8}] = None;\nfn main() => Binary{8} =fold(mk_none(), id_val, 0b1111_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} =0b1111_0000;";
    assert_three_way("fold(None, id_val, d)", &src, expected);
}

// ── or_else ──────────────────────────────────────────────────────────────────────────────────────

/// `or_else(Some(x), alt)` → `Some(x)` — the present value wins; the alternative is ignored.
#[test]
fn or_else_on_some_keeps_value() {
    let driver = "fn mk_some() => Option[Binary{8}] = Some(0b0000_0001);\nfn mk_alt() => Option[Binary{8}] = Some(0b1111_1111);\nfn main() => Option[Binary{8}] =or_else(mk_some(), mk_alt());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =Some(0b0000_0001);";
    assert_three_way("or_else(Some, alt)", &src, expected);
}

/// `or_else(None, alt)` → `alt` — the caller-supplied alternative is taken (never-silent, G2).
#[test]
fn or_else_on_none_takes_alternative() {
    let driver = "fn mk_none() => Option[Binary{8}] = None;\nfn mk_alt() => Option[Binary{8}] = Some(0b1111_1111);\nfn main() => Option[Binary{8}] =or_else(mk_none(), mk_alt());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =Some(0b1111_1111);";
    assert_three_way("or_else(None, alt)", &src, expected);
}

// ── flatten ──────────────────────────────────────────────────────────────────────────────────────

/// `flatten(Some(Some(x)))` → `Some(x)` — the nested value survives one level of collapse.
#[test]
fn flatten_some_some_yields_inner() {
    let driver = "fn mk() => Option[Option[Binary{8}]] = Some(Some(0b0000_0001));\nfn main() => Option[Binary{8}] =flatten(mk());";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =Some(0b0000_0001);";
    assert_three_way("flatten(Some(Some))", &src, expected);
}

/// `flatten(Some(None))` → `None` — an inner None collapses to None.
#[test]
fn flatten_some_none_yields_none() {
    let driver = "fn mk() => Option[Option[Binary{8}]] = Some(None);\nfn main() => Option[Binary{8}] =flatten(mk());";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =None;";
    assert_three_way("flatten(Some(None))", &src, expected);
}

/// `flatten(None)` → `None` — an outer None collapses to None.
#[test]
fn flatten_none_yields_none() {
    let driver = "fn mk() => Option[Option[Binary{8}]] = None;\nfn main() => Option[Binary{8}] =flatten(mk());";
    let src = program(driver);
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{8}] =None;";
    assert_three_way("flatten(None)", &src, expected);
}
