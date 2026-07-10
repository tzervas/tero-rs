//! Differential tests for `std.fmt` (M-717, #462) — the self-hosted first-order formatting
//! utilities: hex-digit conversion (`hex_digit`), low-nibble extraction (`nibble_lo`), high-nibble
//! extraction (`nibble_hi`), and two-digit hex encoding (`to_hex`).
//!
//! The nodule source is loaded verbatim via `include_str!` (the single source of truth), then a
//! driver `fn main` is appended. `assert_three_way` mirrors `std_option.rs` exactly.
//!
//! # Honesty tags
//! - **`Exact`** — `hex_digit` (total over 0..15 via `lt`/`add_u`; the ≥16 fallback arm is `Declared`), `nibble_lo` (total bit-mask),
//!   `nibble_hi` (total 4-level lt binary-search tree over 16 possible masked values).
//! - **`Declared`** — `to_hex` (structural composition of Exact parts; correct for all Binary{8}
//!   inputs by construction).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT), validated
//!   by trial on the programs below; not a machine-checked proof.
//!
//! # Anchor
//! Expected values are hand-computed and verified three-way (L1≡L0≡AOT). The Rust crate
//! crates/mycelium-std-fmt exists but exposes a different Ring-2 surface (no hex_digit/to_hex),
//! so it is the value oracle for shared semantics only — not a structural reference.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// The std.fmt nodule source, loaded at compile time — the single source of truth.
const FMT_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/fmt.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    format!("{FMT_SRC}\n{driver}")
}

/// Run the three-way differential on `src` — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT — and
/// assert all three paths agree AND equal the `expected` reference value.
///
/// Honesty: differential agreement is `Empirical` (trials); the type-level contract is `Declared`.
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

// ── hex_digit ─────────────────────────────────────────────────────────────────────────────────────
//
// hex_digit maps nibble values 0..15 to ASCII hex characters:
//   0  → '0'=0x30=0b0011_0000    5  → '5'=0x35=0b0011_0101
//   1  → '1'=0x31=0b0011_0001    9  → '9'=0x39=0b0011_1001
//   10 → 'a'=0x61=0b0110_0001   15  → 'f'=0x66=0b0110_0110
// Hand-computed: 0..9: 0x30+n; 10..15: 0x57+n (since 0x57+10=0x61='a').
//
// Reference programs use `add_u` to match the `Derived` provenance of the computed result
// (a literal `0b0011_0000` would have `Root` provenance — see std_option.rs `map` comment).

/// `hex_digit(0)` → `add_u(0, 0x30)` = '0' = 0x30 (Exact: 0 < 10, add_u(0, 0x30)).
#[test]
fn hex_digit_zero() {
    let driver = "fn main() => Binary{8} = hex_digit(0b0000_0000);";
    let src = program(driver);
    // Reference: add_u(0b0000_0000, 0b0011_0000) — Derived provenance to match computed result.
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0000, 0b0011_0000);";
    assert_three_way("hex_digit(0)='0'", &src, expected);
}

/// `hex_digit(9)` → `add_u(9, 0x30)` = '9' = 0x39 (Exact: 9 < 10, add_u(9, 0x30)).
#[test]
fn hex_digit_nine() {
    // 0b0000_1001 = 9; '9'=0x39=57. Reference: add_u(0b0000_1001, 0b0011_0000).
    let driver = "fn main() => Binary{8} = hex_digit(0b0000_1001);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_1001, 0b0011_0000);";
    assert_three_way("hex_digit(9)='9'", &src, expected);
}

/// `hex_digit(10)` → `add_u(10, 0x57)` = 'a' = 0x61 (Exact: 10 >= 10, 10 < 16).
#[test]
fn hex_digit_ten() {
    // 0b0000_1010 = 10; 'a'=0x61=97. Reference: add_u(0b0000_1010, 0b0101_0111).
    let driver = "fn main() => Binary{8} = hex_digit(0b0000_1010);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_1010, 0b0101_0111);";
    assert_three_way("hex_digit(10)='a'", &src, expected);
}

/// `hex_digit(15)` → `add_u(15, 0x57)` = 'f' = 0x66 (Exact: 15 >= 10, 15 < 16).
#[test]
fn hex_digit_fifteen() {
    // 0b0000_1111 = 15; 'f'=0x66=102. Reference: add_u(0b0000_1111, 0b0101_0111).
    let driver = "fn main() => Binary{8} = hex_digit(0b0000_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_1111, 0b0101_0111);";
    assert_three_way("hex_digit(15)='f'", &src, expected);
}

/// `hex_digit(5)` → `add_u(5, 0x30)` = '5' = 0x35 (Exact: 5 < 10, add_u(5, 0x30)).
#[test]
fn hex_digit_five() {
    // 0b0000_0101 = 5; '5'=0x35=53. Reference: add_u(0b0000_0101, 0b0011_0000).
    let driver = "fn main() => Binary{8} = hex_digit(0b0000_0101);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0101, 0b0011_0000);";
    assert_three_way("hex_digit(5)='5'", &src, expected);
}

/// `hex_digit(16)` → `'?'` = 0x3F = 0b0011_1111 — never-silent out-of-range fallback (G2).
/// The fallback arm returns a literal constant (Root provenance — no computation applied).
#[test]
fn hex_digit_out_of_range_returns_fallback() {
    // 0b0001_0000 = 16; not a valid nibble → fallback '?'=0x3F=63=0b0011_1111.
    // The fallback arm is a literal: Root provenance, same as the reference literal.
    let driver = "fn main() => Binary{8} = hex_digit(0b0001_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0011_1111;"; // '?'=0x3F, Root provenance
    assert_three_way("hex_digit(16)='?'", &src, expected);
}

// ── nibble_lo ─────────────────────────────────────────────────────────────────────────────────────
//
// nibble_lo(x) = and(x, 0b0000_1111). `and` produces Derived provenance, so reference programs
// use `and` to match.

/// `nibble_lo(0b1010_0101)` → `and(0b1010_0101, 0b0000_1111)` = 5 (Exact).
#[test]
fn nibble_lo_extracts_low_bits() {
    // 0b1010_0101 = 0xa5; low nibble = 0b0000_0101 = 5.
    let driver = "fn main() => Binary{8} = nibble_lo(0b1010_0101);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = and(0b1010_0101, 0b0000_1111);";
    assert_three_way("nibble_lo(0xa5)=5", &src, expected);
}

/// `nibble_lo(0b1111_0000)` → `and(0b1111_0000, 0b0000_1111)` = 0 (Exact).
#[test]
fn nibble_lo_zero_low_nibble() {
    let driver = "fn main() => Binary{8} = nibble_lo(0b1111_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = and(0b1111_0000, 0b0000_1111);";
    assert_three_way("nibble_lo(0xf0)=0", &src, expected);
}

/// `nibble_lo(0b0000_1111)` → `and(0b0000_1111, 0b0000_1111)` = 15 (Exact).
#[test]
fn nibble_lo_full_low_nibble() {
    let driver = "fn main() => Binary{8} = nibble_lo(0b0000_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = and(0b0000_1111, 0b0000_1111);";
    assert_three_way("nibble_lo(0x0f)=15", &src, expected);
}

// ── nibble_hi ─────────────────────────────────────────────────────────────────────────────────────
//
// nibble_hi extracts the high 4 bits of a byte via a 4-level lt binary-search match tree.
// The masked value `and(x, 0b1111_0000)` is one of 16 values (0x00, 0x10, ..., 0xF0), each
// mapping to nibble digits 0..15 respectively.
//
// Hand-computed (Exact):
//   nibble_hi(0b0000_0000) = 0   (masked=0x00 → 0)
//   nibble_hi(0b0001_0000) = 1   (masked=0x10 → 1)
//   nibble_hi(0b0100_0000) = 4   (masked=0x40 → 4)
//   nibble_hi(0b1010_0101) = 10  (masked=0xa0 → 10)
//   nibble_hi(0b1111_1111) = 15  (masked=0xf0 → 15)
// Grounding: hand-computed, three-way verified; mycelium-std-fmt exists but is a different Ring-2 surface, not the oracle.

/// `nibble_hi(0b0000_0101)` → `0b0000_0000` (= 0; masked=0x00; Exact).
#[test]
fn nibble_hi_zero_high_nibble() {
    let driver = "fn main() => Binary{8} = nibble_hi(0b0000_0101);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0000;";
    assert_three_way("nibble_hi(0x05)=0", &src, expected);
}

/// `nibble_hi(0b0001_0111)` → `0b0000_0001` (= 1; masked=0x10; Exact).
#[test]
fn nibble_hi_one() {
    // 0b0001_0111 = 0x17; masked=0x10 → nibble 1.
    let driver = "fn main() => Binary{8} = nibble_hi(0b0001_0111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way("nibble_hi(0x17)=1", &src, expected);
}

/// `nibble_hi(0b0100_1010)` → `0b0000_0100` (= 4; masked=0x40; Exact).
#[test]
fn nibble_hi_four() {
    // 0b0100_1010 = 0x4a; masked=0x40 → nibble 4.
    let driver = "fn main() => Binary{8} = nibble_hi(0b0100_1010);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0100;";
    assert_three_way("nibble_hi(0x4a)=4", &src, expected);
}

/// `nibble_hi(0b1010_0101)` → `0b0000_1010` (= 10; masked=0xa0; Exact).
#[test]
fn nibble_hi_ten() {
    // 0b1010_0101 = 0xa5; masked=0xa0 → nibble 10.
    let driver = "fn main() => Binary{8} = nibble_hi(0b1010_0101);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_1010;";
    assert_three_way("nibble_hi(0xa5)=10", &src, expected);
}

/// `nibble_hi(0b1111_1111)` → `0b0000_1111` (= 15; masked=0xf0; Exact).
#[test]
fn nibble_hi_fifteen() {
    // 0b1111_1111 = 0xFF; masked=0xf0 → nibble 15.
    let driver = "fn main() => Binary{8} = nibble_hi(0b1111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_1111;";
    assert_three_way("nibble_hi(0xFF)=15", &src, expected);
}

// ── to_hex ────────────────────────────────────────────────────────────────────────────────────────
//
// to_hex(x) = HP(hex_digit(nibble_hi(x)), hex_digit(nibble_lo(x))).
//
// The `to_hex` reference re-runs the SAME function bodies (FMT_SRC with the nodule header renamed
// to `nodule ref`). This is forced by provenance-sensitive CoreValue equality: `to_hex` composes
// `nibble_hi`/`nibble_lo`/`hex_digit`, whose results carry `Derived` provenance that a literal
// `HP(..)` (Root provenance) cannot match — a literal value-pin would fail on provenance, not value.
// CONSEQUENCE (honest, VR-5): `to_hex_ref` proves L1 ≡ L0 ≡ AOT cross-engine agreement + provenance
// stability — it does NOT independently confirm bit values (a swapped-nibble composition bug would
// replicate into the reference). Independent value-grounding comes from the dedicated
// `hex_digit`/`nibble_lo`/`nibble_hi` tests above (hand-built `add_u` oracles, not self-referential);
// `to_hex` is the visible 1-line composition `HP(hex_digit(nibble_hi(x)), hex_digit(nibble_lo(x)))`,
// and the per-case hand-computed values are documented just below.
//
// Hand-computed expected values (Declared; Empirical three-way):
//   to_hex(0x00) = HP('0','0') = HP(0x30, 0x30)
//   to_hex(0x4a) = HP('4','a') = HP(0x34, 0x61)   nibble_hi(0x4a)=4→'4'; nibble_lo(0x4a)=10→'a'
//   to_hex(0xff) = HP('f','f') = HP(0x66, 0x66)   nibble_hi=15→'f'; nibble_lo=15→'f'
//   to_hex(0x0f) = HP('0','f') = HP(0x30, 0x66)   nibble_hi=0→'0'; nibble_lo=15→'f'
//   to_hex(0xa0) = HP('a','0') = HP(0x61, 0x30)   nibble_hi=10→'a'; nibble_lo=0→'0'
// Grounding: hand-computed, three-way verified; mycelium-std-fmt exists but is a different Ring-2 surface, not the oracle.

/// Build a `to_hex` reference by reusing the full FMT_SRC (nodule header renamed to `ref`) + the
/// same driver. This matches the test program's `Derived` provenance (a literal `HP` cannot — see
/// the note above), so it checks cross-engine agreement, NOT independent bit values.
fn to_hex_ref(driver: &str) -> String {
    // Replace "nodule std.fmt;" with "nodule ref;" — all function definitions are preserved.
    format!(
        "{}\n{}",
        FMT_SRC.replace("nodule std.fmt;", "nodule ref;"),
        driver
    )
}

/// `to_hex(0x00)` → `HP('0','0')` = `HP(0x30, 0x30)` (Declared/Empirical).
/// nibble_hi(0x00)=0→'0'=0x30; nibble_lo(0x00)=0→'0'=0x30.
#[test]
fn to_hex_zero() {
    let driver = "fn main() => HexPair = to_hex(0b0000_0000);";
    let src = program(driver);
    let expected = to_hex_ref(driver);
    assert_three_way("to_hex(0x00)=HP('0','0')", &src, &expected);
}

/// `to_hex(0x4a)` → `HP('4','a')` = `HP(0x34, 0x61)` (Declared/Empirical).
/// nibble_hi(0x4a)=4 → hex_digit(4)='4'=0x34; nibble_lo(0x4a)=10 → hex_digit(10)='a'=0x61.
#[test]
fn to_hex_0x4a() {
    // 0b0100_1010 = 0x4a.
    let driver = "fn main() => HexPair = to_hex(0b0100_1010);";
    let src = program(driver);
    let expected = to_hex_ref(driver);
    assert_three_way("to_hex(0x4a)=HP('4','a')", &src, &expected);
}

/// `to_hex(0xff)` → `HP('f','f')` = `HP(0x66, 0x66)` (Declared/Empirical).
/// nibble_hi(0xff)=15 → 'f'=0x66; nibble_lo(0xff)=15 → 'f'=0x66.
#[test]
fn to_hex_0xff() {
    // 0b1111_1111 = 0xff.
    let driver = "fn main() => HexPair = to_hex(0b1111_1111);";
    let src = program(driver);
    let expected = to_hex_ref(driver);
    assert_three_way("to_hex(0xff)=HP('f','f')", &src, &expected);
}

/// `to_hex(0x0f)` → `HP('0','f')` = `HP(0x30, 0x66)` (Declared/Empirical).
/// nibble_hi(0x0f)=0 → '0'=0x30; nibble_lo(0x0f)=15 → 'f'=0x66.
#[test]
fn to_hex_0x0f() {
    // 0b0000_1111 = 0x0f.
    let driver = "fn main() => HexPair = to_hex(0b0000_1111);";
    let src = program(driver);
    let expected = to_hex_ref(driver);
    assert_three_way("to_hex(0x0f)=HP('0','f')", &src, &expected);
}

/// `to_hex(0xa0)` → `HP('a','0')` = `HP(0x61, 0x30)` (Declared/Empirical).
/// nibble_hi(0xa0)=10 → 'a'=0x61; nibble_lo(0xa0)=0 → '0'=0x30.
#[test]
fn to_hex_0xa0() {
    // 0b1010_0000 = 0xa0.
    let driver = "fn main() => HexPair = to_hex(0b1010_0000);";
    let src = program(driver);
    let expected = to_hex_ref(driver);
    assert_three_way("to_hex(0xa0)=HP('a','0')", &src, &expected);
}
