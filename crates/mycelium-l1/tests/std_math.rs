//! Differential tests for `std.math` (M-718) — the self-hosted WIDTH-GENERIC arithmetic/logic surface.
//!
//! The nodule source is loaded verbatim via `include_str!` (the single source of truth), then a typed
//! `fn main` driver is appended that pins the width param from concrete operand literals. The
//! `assert_three_way` harness mirrors `std_cmp.rs`: L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT, all
//! three agree AND equal the reference value. The reference programs reuse the same kernel prim the
//! surface fn delegates to, so the computed value shares `Derived` provenance with the reference.
//!
//! # Width-generic coverage (M-753 / DN-42)
//! Each op is exercised at **two distinct widths** (Binary{8} + Binary{16}; Ternary{3} + Ternary{6}),
//! proving the single definition genuinely monomorphizes to the call-site width — not a renamed
//! fixed-width monomorph.
//!
//! # Honesty tags (grounded in crates/mycelium-interp/src/prims.rs)
//! - **`Exact`** — every op is Exact on its in-range result: `bit.add`/`bit.sub` (ripple-carry,
//!   never-silent overflow), `bit.and/or/xor/not` (total logical), `trit.add/sub/mul` (fixed-width,
//!   never-silent overflow), `trit.neg` (always in range).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), by trial.
//! - **never-silent (G2)** — an overflow/underflow (`badd`/`bsub`/`tadd`/`tsub`/`tmul` out of range) is
//!   an explicit refusal on EVERY path, never a silent modular wrap; a width mismatch is a static
//!   refusal, never a silent coercion.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// The std.math nodule source, loaded at compile time — the single source of truth.
const MATH_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/math.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    format!("{MATH_SRC}\n{driver}")
}

/// Run the three-way differential on `src` — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT — and assert
/// all three paths agree AND equal the `expected` reference value.
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

    assert!(
        mono.fns.values().all(|fd| fd.sig.params.is_empty())
            && mono.types.values().all(|d| d.params.is_empty())
            && mono.traits.is_empty()
            && mono.instances.is_empty()
            && mono.impls.is_empty(),
        "{label}: monomorphized env must be closed (no generics/traits)"
    );

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
        "{label}: L1-eval(mono) vs elaborate→L0-interp diverged"
    );
    assert_eq!(l0_core, aot_core, "{label}: L0-interp vs AOT diverged");

    for (x, y, pair) in [
        (&l1_core, &l0_core, "L1↔interp"),
        (&l0_core, &aot_core, "interp↔AOT"),
    ] {
        assert_eq!(
            check_core(x, y),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "{label}: the shared checker must validate the {pair} pair"
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

/// A `main` of the given `ret` type calling `call`; the reference recomputes via `ref_body`.
fn assert_op(label: &str, ret: &str, call: &str, ref_body: &str) {
    let src = program(&format!("fn main() => {ret} = {call};"));
    let expected = format!("nodule ref;\nfn main() => {ret} = {ref_body};");
    assert_three_way(label, &src, &expected);
}

// ── Binary arithmetic (badd / bsub) at Binary{8} and Binary{16} ─────────────────────────────────────

/// `badd(3, 5)` at Binary{8} → 8. Exact; ref recomputes via the same `add_u` (Derived provenance).
#[test]
fn badd_binary8() {
    assert_op(
        "badd8(3,5)",
        "Binary{8}",
        "badd(0b0000_0011, 0b0000_0101)",
        "add_u(0b0000_0011, 0b0000_0101)",
    );
}

/// `badd(256, 1)` at Binary{16} → 257. The operands carry 16 bits, so the SAME `badd` specialises to
/// N=16 (256 is unrepresentable at Binary{8}). Exact.
#[test]
fn badd_binary16() {
    assert_op(
        "badd16(256,1)",
        "Binary{16}",
        "badd(0b0000_0001_0000_0000, 0b0000_0000_0000_0001)",
        "add_u(0b0000_0001_0000_0000, 0b0000_0000_0000_0001)",
    );
}

/// `bsub(5, 3)` at Binary{8} → 2. Exact.
#[test]
fn bsub_binary8() {
    assert_op(
        "bsub8(5,3)",
        "Binary{8}",
        "bsub(0b0000_0101, 0b0000_0011)",
        "sub_u(0b0000_0101, 0b0000_0011)",
    );
}

/// `bsub(256, 1)` at Binary{16} → 255. Exact (crosses the Binary{8} boundary downward).
#[test]
fn bsub_binary16() {
    assert_op(
        "bsub16(256,1)",
        "Binary{16}",
        "bsub(0b0000_0001_0000_0000, 0b0000_0000_0000_0001)",
        "sub_u(0b0000_0001_0000_0000, 0b0000_0000_0000_0001)",
    );
}

// ── Binary bitwise logic (band / bor / bxor / bnot) ─────────────────────────────────────────────────

/// `band(0b1100, 0b1010)` at Binary{8} → 0b1000. Exact.
#[test]
fn band_binary8() {
    assert_op(
        "band8",
        "Binary{8}",
        "band(0b0000_1100, 0b0000_1010)",
        "and(0b0000_1100, 0b0000_1010)",
    );
}

/// `bor(0b1100, 0b1010)` at Binary{8} → 0b1110. Exact.
#[test]
fn bor_binary8() {
    assert_op(
        "bor8",
        "Binary{8}",
        "bor(0b0000_1100, 0b0000_1010)",
        "or(0b0000_1100, 0b0000_1010)",
    );
}

/// `bxor(0b1100, 0b1010)` at Binary{8} → 0b0110. Exact.
#[test]
fn bxor_binary8() {
    assert_op(
        "bxor8",
        "Binary{8}",
        "bxor(0b0000_1100, 0b0000_1010)",
        "xor(0b0000_1100, 0b0000_1010)",
    );
}

/// `bnot(0b0000_1111)` at Binary{8} → 0b1111_0000. Exact (one's complement).
#[test]
fn bnot_binary8() {
    assert_op(
        "bnot8",
        "Binary{8}",
        "bnot(0b0000_1111)",
        "not(0b0000_1111)",
    );
}

/// `band` at Binary{16} — the same definition at a second width. Exact.
#[test]
fn band_binary16() {
    assert_op(
        "band16",
        "Binary{16}",
        "band(0b1111_1111_0000_0000, 0b0000_1111_1111_0000)",
        "and(0b1111_1111_0000_0000, 0b0000_1111_1111_0000)",
    );
}

/// `bnot` at Binary{16} — second width. Exact.
#[test]
fn bnot_binary16() {
    assert_op(
        "bnot16",
        "Binary{16}",
        "bnot(0b0000_0000_1111_1111)",
        "not(0b0000_0000_1111_1111)",
    );
}

// ── Balanced-ternary arithmetic (tadd / tsub / tmul / tneg) at Ternary{3} and Ternary{6} ────────────

/// `tadd(<00+>, <00->)` at Ternary{3} → <000> (+1 + -1 = 0). Exact.
#[test]
fn tadd_ternary3() {
    assert_op(
        "tadd3",
        "Ternary{3}",
        "tadd(0t00+, 0t00-)",
        "add(0t00+, 0t00-)",
    );
}

/// `tsub(<00+>, <00->)` at Ternary{3} → <00+> + <00+> ... = +2 = <0+->. Exact.
#[test]
fn tsub_ternary3() {
    assert_op(
        "tsub3",
        "Ternary{3}",
        "tsub(0t00+, 0t00-)",
        "sub(0t00+, 0t00-)",
    );
}

/// `tmul(<00+>, <0+->)` at Ternary{3} → +1 * +2 = +2 = <0+->. Exact.
#[test]
fn tmul_ternary3() {
    assert_op(
        "tmul3",
        "Ternary{3}",
        "tmul(0t00+, 0t0+-)",
        "mul(0t00+, 0t0+-)",
    );
}

/// `tneg(<0+->)` at Ternary{3} → -(+2) = -2 = <0-+>. Exact (digit-wise sign flip).
#[test]
fn tneg_ternary3() {
    assert_op("tneg3", "Ternary{3}", "tneg(0t0+-)", "neg(0t0+-)");
}

/// `tadd` at Ternary{6} — the same definition at a second width. Exact.
#[test]
fn tadd_ternary6() {
    assert_op(
        "tadd6",
        "Ternary{6}",
        "tadd(0t00+0+-, 0t00000+)",
        "add(0t00+0+-, 0t00000+)",
    );
}

/// `tneg` at Ternary{6} — second width. Exact.
#[test]
fn tneg_ternary6() {
    assert_op("tneg6", "Ternary{6}", "tneg(0t00+0+-)", "neg(0t00+0+-)");
}

// ── Second-width coverage for bor / bxor / tsub / tmul (M-719 gap-closure) ───────────────────────────
//
// The consolidated conformance gate (std_generic_conformance.rs) states every width-generic math op is
// checked at ≥ 2 distinct widths. `badd`/`bsub`/`band`/`bnot` and `tadd`/`tneg` already had two widths
// above; these add the SECOND width for `bor`/`bxor` (Binary{16}) and `tsub`/`tmul` (Ternary{6}) so the
// "≥ 2 widths each" claim holds for the whole binary-bitwise and ternary-arithmetic surface — proving each
// is genuinely width-generic (the same definition specialised at a second width), not a Binary{8}/Ternary{3}
// monomorph. All Exact; three-way agreement Empirical. Reference reuses the kernel prim (Derived provenance).

/// `bor` at Binary{16} — the same definition at a second width. Exact.
#[test]
fn bor_binary16() {
    assert_op(
        "bor16",
        "Binary{16}",
        "bor(0b1111_1111_0000_0000, 0b0000_1111_1111_0000)",
        "or(0b1111_1111_0000_0000, 0b0000_1111_1111_0000)",
    );
}

/// `bxor` at Binary{16} — the same definition at a second width. Exact.
#[test]
fn bxor_binary16() {
    assert_op(
        "bxor16",
        "Binary{16}",
        "bxor(0b1111_1111_0000_0000, 0b0000_1111_1111_0000)",
        "xor(0b1111_1111_0000_0000, 0b0000_1111_1111_0000)",
    );
}

/// `tsub` at Ternary{6} — the same definition at a second width. Exact.
#[test]
fn tsub_ternary6() {
    assert_op(
        "tsub6",
        "Ternary{6}",
        "tsub(0t00+0+-, 0t00000+)",
        "sub(0t00+0+-, 0t00000+)",
    );
}

/// `tmul` at Ternary{6} — the same definition at a second width. Exact (in-range result).
#[test]
fn tmul_ternary6() {
    assert_op(
        "tmul6",
        "Ternary{6}",
        "tmul(0t00000+, 0t0000+-)",
        "mul(0t00000+, 0t0000+-)",
    );
}

// ── Never-silent overflow/underflow refusals (G2) on every path ─────────────────────────────────────

/// Run the generic surface program and assert all three paths REFUSE (never-silent overflow / G2).
fn assert_eval_refuses(label: &str, driver: &str) {
    let src = program(driver);
    let env =
        check_nodule(&parse(&src).unwrap_or_else(|e| panic!("{label}: parse must succeed: {e}")))
            .unwrap_or_else(|e| {
                panic!("{label}: check must succeed (overflow is runtime, not static): {e}")
            });
    let mono = monomorphize(&env, "main")
        .unwrap_or_else(|e| panic!("{label}: monomorphize must succeed: {e}"));

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&mono).call("main", vec![]).is_err(),
        "{label}: L1-eval must refuse (never a silent wrap)"
    );
    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: elaborate failed: {e}"));
    assert!(
        interp.eval(&node).is_err(),
        "{label}: L0-interp must refuse"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "{label}: AOT must refuse"
    );
}

/// `badd(255, 1)` at Binary{8} overflows (carry-out of bit 7) — a never-silent `Overflow` refusal on
/// every path, never a silent wrap to 0 (G2). The exact `bit.add` ripple-carry contract.
#[test]
fn badd_overflow_refuses_on_every_path() {
    assert_eval_refuses(
        "badd(255,1) overflow",
        "fn main() => Binary{8} = badd(0b1111_1111, 0b0000_0001);",
    );
}

/// `bsub(0, 1)` at Binary{8} underflows (borrow-out below 0) — a never-silent `Overflow` refusal on
/// every path, never a silent wrap to 255 (G2).
#[test]
fn bsub_underflow_refuses_on_every_path() {
    assert_eval_refuses(
        "bsub(0,1) underflow",
        "fn main() => Binary{8} = bsub(0b0000_0000, 0b0000_0001);",
    );
}

// ── Never-silent width-mismatch refusal (static) ────────────────────────────────────────────────────

/// `badd` with a Binary{8} and a Binary{16} operand is a static width-mismatch refusal — the single
/// width param `N` cannot be both 8 and 16 (DN-42 §4 / VR-5 / S1). Never a silent widen.
#[test]
fn badd_mixed_widths_refuses() {
    let src = program("fn main() => Binary{16} = badd(0b0000_0001, 0b0000_0001_0000_0000);");
    let parsed = parse(&src).expect("parse should succeed");
    let err = check_nodule(&parsed)
        .expect_err("expected a never-silent width-mismatch refusal, but check succeeded")
        .to_string();
    assert!(
        err.contains("Binary{16}")
            && (err.contains("cannot match") || err.contains("width") || err.contains("swap")),
        "refusal must name the width mismatch (never-silent), got: {err}"
    );
}
