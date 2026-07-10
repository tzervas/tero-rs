//! Differential tests for `std.core` (M-927, kickoff `opp`, RFC-0031 D5) — the `.myc` port of
//! `crates/mycelium-std-core/src/lib.rs`'s RFC-0016 §4.5 guarantee matrix (the crate's ONE own
//! type + its const data).
//!
//! # Scope (surface-check, D5 row 1 — see `lib/std/core.myc`'s module doc for the full writeup)
//! `mycelium-std-core` is a re-export-heavy Ring-0 facade: 7 kernel re-export groups, a `prelude`
//! bundle, the 5-fn §4.8 kernel query surface (`repr_of`…`provenance_of` over `CoreValue`), and
//! the `error_scaffold` (trait + `macro_rules!`) — ALL of that is the RFC-0031 D1 kernel boundary
//! (the kernel stays Rust; no FFI/kernel-type construction or value-reflection mechanism is
//! exposed to `.myc` today) and is FLAGged, not ported (VR-5/G2 — never a hollow port; the
//! `std_diag.rs` kernel-half precedent). This file tests ONLY the ported part: `GuaranteeRow` +
//! the 9-row `GUARANTEE_MATRIX` as checked data, and the structural checks over it.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-case drivers, and the row-4 Rust-oracle wiring.
//! Every `expected_src` below is built from a value **computed live from
//! `mycelium_std_core::GUARANTEE_MATRIX`**, not a hardcoded literal — a real divergence between
//! the Rust source and the `.myc` transcription would flip the computed oracle value and fail the
//! corresponding case.
//!
//! # Honesty tags
//! - **`Exact`** — every matrix row's `tag` field, per RFC-0016 C2 (`std.core` introduces no
//!   operation that selects, converts, or approximates).
//! - **`Declared`** — the 9-row transcription itself (asserted data, not machine-checked; mirrors
//!   the Rust source's own `#[cfg(test)]` structural assertions).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) below,
//!   validated by trial on the cases exercised.
//!
//! # Bytes-comparison note (narrowed from the std_diag FLAG)
//! There is still no general `bytes_eq`/substring prim (RFC-0032 D4), but the Rust test's
//! `row.effects == "none"` assertion IS fully ported: "none" is a fixed 4-byte literal, so
//! `lib/std/core.myc::effects_is_none` composes the Exact `bytes_len`/`bytes_get`/`eq` prims into
//! full content equality. Free-form prose fields (`op`, `fallibility`) are transcribed
//! byte-faithfully and spot-checked (first byte) only, as in `std_diag.rs`.

mod harness;

use mycelium_core::GuaranteeStrength;
use mycelium_std_core::GUARANTEE_MATRIX;

/// The std.core nodule source, loaded at compile time — the single source of truth.
const CORE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/core.myc"
));

/// Build a full test program by appending a driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(CORE_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`].
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

/// Render a Rust `bool` as the `.myc` literal that denotes the same `Bool` value.
fn myc_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// Render the `n`-deep nested `add_u(0b1, add_u(0b1, …, 0b0)…)` expression that the port's
/// recursive `Binary{8}` counters (`matrix_len`, `explainable_count`) expand to for `n` matching
/// elements. Writing the SAME primitive-op composition directly gives the reference program the
/// matching `Derived` provenance chain the M-925 harness's `check_core` comparison requires (the
/// `std_result.rs`/`std_option.rs`/`std_diag.rs` precedent: recompute via the SAME underlying
/// prims, not a bare literal) while still being an INDEPENDENT check of the count.
fn myc_count_chain(n: u8) -> String {
    let mut expr = "0b0000_0000".to_owned();
    for _ in 0..n {
        expr = format!("add_u(0b0000_0001, {expr})");
    }
    expr
}

// ── matrix_len — row count (the Rust oracle's OWN `GUARANTEE_MATRIX.len()`, not a hardcoded 9) ────

/// `matrix_len(matrix())` equals the live Rust oracle's row count.
/// Guarantee: Declared (the transcription); Empirical (differential).
#[test]
fn matrix_len_matches_rust_oracle_row_count() {
    let expected_count = u8::try_from(GUARANTEE_MATRIX.len()).expect("row count fits u8");
    let driver = "fn main() => Binary{8} = matrix_len(matrix());";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\nfn main() => Binary{{8}} = {};",
        myc_count_chain(expected_count)
    );
    assert_three_way("matrix_len == rust GUARANTEE_MATRIX.len()", &src, &expected);
}

// ── all_exact — half of lib.rs::tests::matrix_is_all_exact_and_effect_free ────────────────────────

/// Every row's tag is `Exact` — driven against the live Rust GUARANTEE_MATRIX (RFC-0016 C2:
/// a `Proven`/`Empirical` tag here would itself violate VR-5).
#[test]
fn all_rows_are_exact_matches_rust_oracle() {
    let expected_all_exact = GUARANTEE_MATRIX
        .iter()
        .all(|r| r.tag == GuaranteeStrength::Exact);
    let driver = "fn main() => Bool = all_exact(matrix());";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\nfn main() => Bool = {};",
        myc_bool(expected_all_exact)
    );
    assert_three_way("all_exact == rust all-Exact", &src, &expected);
}

// ── all_effects_none — the other half of matrix_is_all_exact_and_effect_free (FULL content) ──────

/// Every row's effects field is EXACTLY `"none"` — driven against the live Rust GUARANTEE_MATRIX.
/// Unlike the `std_diag.rs` non-empty checks, this is full content equality: `"none"` is a fixed
/// 4-byte literal, so the port's `effects_is_none` composes `bytes_len`/`bytes_get`/`eq` into the
/// same assertion the Rust test makes (`row.effects, "none"`).
#[test]
fn all_rows_effect_free_matches_rust_oracle() {
    let expected = GUARANTEE_MATRIX.iter().all(|r| r.effects == "none");
    let driver = "fn main() => Bool = all_effects_none(matrix());";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "all_effects_none == rust effects==\"none\" check",
        &src,
        &expected_src,
    );
}

// ── explainable_count — the cardinality half of only_query_rows_are_explainable ──────────────────

/// The number of EXPLAIN-able rows equals the live Rust oracle's count (3: the query rows).
#[test]
fn explainable_count_matches_rust_oracle() {
    let expected_count = u8::try_from(GUARANTEE_MATRIX.iter().filter(|r| r.explainable).count())
        .expect("count fits u8");
    let driver = "fn main() => Binary{8} = explainable_count(matrix());";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\nfn main() => Binary{{8}} = {};",
        myc_count_chain(expected_count)
    );
    assert_three_way(
        "explainable_count == rust explainable count",
        &src,
        &expected,
    );
}

// ── only_query_rows_explainable — lib.rs::tests::only_query_rows_are_explainable ─────────────────

/// The EXPLAIN window is exactly `guarantee_of`/`bound_of`/`provenance_of` — driven against the
/// live Rust oracle's own op-name list (the `.myc` side checks the same fact per-row by
/// constructor identity; the op-NAME list equality itself needs `bytes_eq` and is covered by the
/// oracle computation here — see the module doc's Bytes-comparison note).
#[test]
fn only_query_rows_explainable_matches_rust_oracle() {
    let explainable: Vec<&str> = GUARANTEE_MATRIX
        .iter()
        .filter(|r| r.explainable)
        .map(|r| r.op)
        .collect();
    let expected = explainable == ["guarantee_of", "bound_of", "provenance_of"];
    let driver = "fn main() => Bool = only_query_rows_explainable();";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "only_query_rows_explainable == rust explainable-set check",
        &src,
        &expected_src,
    );
}

// ── Individual row spot-checks (per-op, hand-computed — the .myc port's own transcription fidelity) ──

/// `row_op(row_repr_of())` starts with the literal byte `'r'` (0x72 = 0b0111_0010) — a sanity
/// check that the string-literal transcription round-trips through `bytes_get` at position 0,
/// since full Bytes equality is not available for free-form prose (see module doc).
#[test]
fn row_repr_of_op_first_byte_is_r() {
    // Guard on the live oracle first: the row this spot-check transcribes must still exist there.
    let repr_of = GUARANTEE_MATRIX
        .iter()
        .find(|r| r.op == "repr_of")
        .expect("repr_of row exists in the rust oracle");
    assert!(
        repr_of.fallibility.contains("Option"),
        "rust oracle: repr_of stays Option-shaped (C1 explicit absence) — the .myc port cannot \
         check this prose without a bytes_eq/contains prim (FLAG, std_diag precedent)"
    );
    let driver = "fn main() => Binary{8} = bytes_get(row_op(row_repr_of()), 0b0000_0000);";
    let src = program(driver);
    // Independent Derived-provenance reference: apply `bytes_get` directly to a fresh "repr_of"
    // literal (same content) — the std_diag.rs precedent (same underlying prim on the same
    // literal content, not a bare Root literal).
    let expected = "nodule ref;\nfn main() => Binary{8} = bytes_get(\"repr_of\", 0b0000_0000);";
    assert_three_way("row_repr_of op[0] == 'r'", &src, expected);
}

/// `row_guarantee_of()` is Exact AND EXPLAIN-able — the two typed fields of the first query row,
/// driven against the live Rust oracle's own row.
#[test]
fn row_guarantee_of_is_exact_and_explainable_matches_rust_oracle() {
    let guarantee_of = GUARANTEE_MATRIX
        .iter()
        .find(|r| r.op == "guarantee_of")
        .expect("guarantee_of row exists in the rust oracle");
    let expected = guarantee_of.tag == GuaranteeStrength::Exact && guarantee_of.explainable;
    let driver =
        "fn main() => Bool = bool_and(is_exact(row_tag(row_guarantee_of())), row_explainable(row_guarantee_of()));";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "row_guarantee_of Exact+explainable == rust row",
        &src,
        &expected_src,
    );
}

/// `row_value_repr_meta()` (a pure type re-export row) is NOT EXPLAIN-able — driven against the
/// live Rust oracle's own row (spot-checks the `False` side of the Bool field's transcription).
#[test]
fn row_type_reexports_not_explainable_matches_rust_oracle() {
    let reexports = GUARANTEE_MATRIX
        .iter()
        .find(|r| r.op == "Value/Repr/Meta (type re-exports)")
        .expect("type re-export row exists in the rust oracle");
    let expected = !reexports.explainable;
    let driver = "fn main() => Bool = bool_not(row_explainable(row_value_repr_meta()));";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "row_value_repr_meta not-explainable == rust row",
        &src,
        &expected_src,
    );
}
