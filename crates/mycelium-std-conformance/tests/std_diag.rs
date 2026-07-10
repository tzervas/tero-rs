//! Differential tests for `std.diag` (M-926, kickoff `opp`, RFC-0031 D5) — the `.myc` port of
//! `crates/mycelium-std-diag/src/guarantee_matrix.rs`'s RFC-0016 §4.5 guarantee matrix.
//!
//! # Scope (surface-check, D5 row 1 — see `lib/std/diag.myc`'s module doc for the full writeup)
//! `mycelium-std-diag`'s public surface has two halves: (a) the guarantee-matrix DATA
//! (`guarantee_matrix::MATRIX`) — pure algebraic data + prose, now expressible via ADTs + the
//! M-910/M-911 textual string literal; (b) the re-exported KERNEL record types (`Diag`/`Locus`/
//! `Trace`/`Code`/`Severity` from `mycelium-diag`) — these carry Rust `String`/`Vec<String>` fields,
//! BLAKE3 content-hashing, and `serde` JSON (de)serialization with no `.myc`-surface equivalent
//! (RFC-0031 D1 — the kernel stays Rust; no FFI/kernel-construction mechanism is exposed to `.myc`
//! today). This file tests ONLY (a); (b) is FLAGged, not ported (VR-5/G2 — never a hollow port).
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-case drivers, and the row-4 Rust-oracle wiring
//! (M-925's harness doc: "each port leaf FLAGs/implements its own Rust-oracle call" — this crate DOES
//! retain a Rust predecessor, `mycelium-std-diag`, unlike `std_result`/`std_option`). Every
//! `expected_src` below is built from a value **computed live from
//! `mycelium_std_diag::guarantee_matrix::MATRIX`**, not a hardcoded literal — a real divergence
//! between the Rust source and this `.myc` transcription would flip the computed oracle value and
//! fail the corresponding case.
//!
//! # Honesty tags
//! - **`Exact`** — every matrix row's `guarantee` field, per RFC-0016 C2 (`diag` has no accuracy
//!   semantics of its own).
//! - **`Declared`** — the 14-row transcription itself (asserted data, not machine-checked; mirrors
//!   the Rust source's own `#[cfg(test)]` assertions, which are likewise structural, not proofs).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) below,
//!   validated by trial on the cases exercised.
//!
//! # Bytes-comparison gap (FLAG, carried from `lib/std/diag.myc`)
//! RFC-0032 D4 exposes only `bytes_len`/`bytes_get`/`bytes_slice`/`bytes_concat` — no
//! `bytes_eq`/substring prim. So the Rust source's STRING-CONTENT assertions
//! (`error_set.contains("UNCHANGED")`, `effects.contains("io")`, the X1/RT5/VR-5 citation checks,
//! `only_sink_has_io_effect`) are NOT ported: they need a prim this port does not have and does not
//! introduce (never forced — VR-5/G2). Everything ELSE in `guarantee_matrix.rs::tests` (row count,
//! all-Exact, non-empty never_silent_property/effects/error_set, the `present`/`content_id`/
//! `policy_ref` structural crux checks) IS ported and driven against the live Rust oracle below.

mod harness;

use mycelium_std_diag::guarantee_matrix::{Explainable, Fallibility, MATRIX};

/// The std.diag nodule source, loaded at compile time — the single source of truth.
const DIAG_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/diag.myc"
));

/// Build a full test program by appending a driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(DIAG_SRC, driver)
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

/// Render the `n`-deep nested `add_u(0b1, add_u(0b1, …, add_u(0b1, 0b0)…))` expression that
/// `matrix_len`'s recursive spine-walk expands to for a list of `n` elements (`len(Cons(x,rest)) =
/// add_u(0b1, len(rest))`, `len(Nil) = 0b0`). Writing the SAME primitive-op composition directly
/// (rather than re-invoking `matrix_len(matrix())`) gives the reference program the matching
/// `Derived` provenance chain the M-925 harness's `check_core` comparison requires (the
/// `std_result.rs`/`std_option.rs` precedent: recompute via the SAME underlying prims, not a bare
/// literal, so the reference is Derived, not Root) while still being an INDEPENDENT check of the
/// row count (it does not call `matrix()`/`matrix_len` at all).
fn myc_len_chain(n: u8) -> String {
    let mut expr = "0b0000_0000".to_owned();
    for _ in 0..n {
        expr = format!("add_u(0b0000_0001, {expr})");
    }
    expr
}

// ── matrix_len — row count (row 4: the Rust oracle's OWN `MATRIX.len()`, not a hardcoded 14) ──────

/// `matrix_len(matrix())` equals the live Rust oracle's row count.
/// Guarantee: Declared (the transcription); Empirical (differential).
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

// ── all_exact — guarantee_matrix.rs::all_diag_ops_are_exact ───────────────────────────────────────

/// Every row's guarantee is `Exact` — driven against the live Rust MATRIX (RFC-0016 C2).
#[test]
fn all_rows_are_exact_matches_rust_oracle() {
    let expected_all_exact = MATRIX.iter().all(|r| r.guarantee == "Exact");
    let driver = "fn main() => Bool = all_exact(matrix());";
    let src = program(driver);
    let expected = format!(
        "nodule ref;\nfn main() => Bool = {};",
        myc_bool(expected_all_exact)
    );
    assert_three_way("all_exact == rust all-Exact", &src, &expected);
}

// ── all_never_silent_nonempty — guarantee_matrix.rs::every_row_states_never_silent_property ───────

/// Every row states a non-empty `never_silent_property` — driven against the live Rust MATRIX.
#[test]
fn all_rows_state_never_silent_property_matches_rust_oracle() {
    let expected = MATRIX.iter().all(|r| !r.never_silent_property.is_empty());
    let driver = "fn main() => Bool = all_never_silent_nonempty(matrix());";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "all_never_silent_nonempty == rust non-empty check",
        &src,
        &expected_src,
    );
}

// ── all_effects_nonempty — guarantee_matrix.rs::every_row_states_effects ──────────────────────────

/// Every row states its effects (C6) — driven against the live Rust MATRIX.
#[test]
fn all_rows_state_effects_matches_rust_oracle() {
    let expected = MATRIX.iter().all(|r| !r.effects.is_empty());
    let driver = "fn main() => Bool = all_effects_nonempty(matrix());";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "all_effects_nonempty == rust non-empty check",
        &src,
        &expected_src,
    );
}

// ── explicit_ops_have_error_set — guarantee_matrix.rs::explicit_ops_have_nonempty_error_set ───────

/// Every `Explicit`-fallibility row states a non-empty `error_set` — driven against the live Rust
/// MATRIX (a `Total` row is vacuously true, matching the `.myc` port's `match` semantics).
#[test]
fn explicit_ops_have_nonempty_error_set_matches_rust_oracle() {
    let expected = MATRIX
        .iter()
        .filter(|r| r.fallibility == Fallibility::Explicit)
        .all(|r| !r.error_set.is_empty());
    let driver = "fn main() => Bool = explicit_ops_have_error_set(matrix());";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "explicit_ops_have_error_set == rust non-empty-on-Explicit check",
        &src,
        &expected_src,
    );
}

// ── present_is_i1_crux — the structural half of guarantee_matrix.rs::present_is_the_i1_crux ───────

/// `present` is `Total`, `Exact`, and `IsExplainRecord` — the I1 structural crux (the
/// "error_set contains UNCHANGED" substring half is FLAGged, not ported — see module doc).
#[test]
fn present_is_i1_crux_matches_rust_oracle() {
    let present = MATRIX
        .iter()
        .find(|r| r.op == "present")
        .expect("present row exists in the rust oracle");
    let expected = present.guarantee == "Exact"
        && present.fallibility == Fallibility::Total
        && present.explainable == Explainable::IsExplainRecord;
    // The rust oracle's own crux test additionally requires "UNCHANGED" in error_set (not portable
    // here — no bytes_eq/contains prim); assert that half holds too, so this test would catch a
    // rust-side regression even though the `.myc` side can't check it structurally.
    assert!(
        present.error_set.contains("UNCHANGED"),
        "rust oracle: present's error_set must still say UNCHANGED (I1) — the .myc port cannot \
         check this without a bytes_eq/contains prim (FLAG)"
    );
    let driver = "fn main() => Bool = present_is_i1_crux();";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "present_is_i1_crux == rust structural crux",
        &src,
        &expected_src,
    );
}

// ── content_id_and_policy_ref_are_handles — guarantee_matrix.rs::content_id_and_policy_ref_are_content_addressed_handles ──

/// `content_id` and `policy_ref` are both `ContentAddressedHandle` — fully portable (pure ADT
/// equality, no substring needed) — driven against the live Rust MATRIX.
#[test]
fn content_id_and_policy_ref_are_handles_matches_rust_oracle() {
    let content_id = MATRIX
        .iter()
        .find(|r| r.op == "content_id")
        .expect("content_id row exists in the rust oracle");
    let policy_ref = MATRIX
        .iter()
        .find(|r| r.op == "policy_ref")
        .expect("policy_ref row exists in the rust oracle");
    let expected = content_id.explainable == Explainable::ContentAddressedHandle
        && policy_ref.explainable == Explainable::ContentAddressedHandle;
    let driver = "fn main() => Bool = content_id_and_policy_ref_are_handles();";
    let src = program(driver);
    let expected_src = format!("nodule ref;\nfn main() => Bool = {};", myc_bool(expected));
    assert_three_way(
        "content_id_and_policy_ref_are_handles == rust ADT check",
        &src,
        &expected_src,
    );
}

// ── Individual row spot-checks (per-op, hand-computed — the .myc port's own transcription fidelity) ──

/// `row_op(row_present())` decodes back to the literal bytes `"present"` — a sanity check that the
/// string-literal transcription round-trips through `bytes_get` at position 0 (the first byte,
/// `'p'` = 0x70 = 0b0111_0000), since full Bytes equality is not available (FLAG, module doc).
#[test]
fn row_present_op_first_byte_is_p() {
    let driver = "fn main() => Binary{8} = bytes_get(row_op(row_present()), 0b0000_0000);";
    let src = program(driver);
    // Independent Derived-provenance reference: apply `bytes_get` directly to a fresh `"present"`
    // literal (same content), rather than re-deriving through `matrix()`/`row_present`/`row_op` —
    // the std_result.rs/std_option.rs precedent (recompute via the SAME underlying prim on the
    // SAME literal content, not a bare Root literal, so the reference is Derived).
    let expected = "nodule ref;\nfn main() => Binary{8} = bytes_get(\"present\", 0b0000_0000);";
    assert_three_way("row_present op[0] == 'p'", &src, expected);
}

/// `row_guarantee(row_sink())` is `Exact` (every row is — spot-checked on the one row with a
/// non-trivial `io` effect, to also exercise that field's construction).
#[test]
fn row_sink_guarantee_is_exact() {
    let driver = "fn main() => Bool = is_exact(row_guarantee(row_sink()));";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("row_sink guarantee is Exact", &src, expected);
}

/// `row_fallibility(row_from_json())` is `Explicit` (the parse-failure row).
#[test]
fn row_from_json_fallibility_is_explicit() {
    let driver = "fn main() => Bool = is_explicit(row_fallibility(row_from_json()));";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("row_from_json fallibility is Explicit", &src, expected);
}
