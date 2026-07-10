//! Differential tests for `std.ternary` (M-933, E29-1, kickoff `opp`) — the balanced-ternary
//! value surface: Trit/Bit digit primitives, the `int <-> trits` codec over `div_u`/`rem_u`,
//! fixed-width `Trits` arithmetic with explicit `Option` fallibility, the I2S/TL1/TL2 packed
//! codecs, and the RFC-0016 §4.5 guarantee matrix as data.
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925). This file
//! supplies the nodule's `include_str!`, per-op three-way cases, and — the row this port owns per
//! the harness doc (§4) — a **live Rust-oracle differential** against the retained
//! `mycelium-std-ternary` crate (RFC-0031 D6; the crate is NOT retired): the numeric-edge corpus
//! below (carries, bounds, codec round-trips, packed-byte values — the M-933 DoD's explicit
//! "numeric edge cases (carries, bounds) in the differential corpus" obligation) is evaluated on
//! BOTH sides at test time, never hand-copied into only one.
//!
//! # Surface-check (D5 row 1) and substitutions
//! See `lib/std/ternary.myc`'s header: all 18 matrix ops ported (renames documented there);
//! FLAGged, not forced (VR-5/G2): the char-typed wire-glyph pair (no `char` type — ported as
//! ASCII-byte substitutes), `i64` (sign-magnitude `SInt` over `Binary{16}`, ceiling `m <= 10`),
//! `Vec<u8>` (ByteList), `Packed` field privacy (unpack widens to `Result`), and the matrix's
//! name-keyed assertions (no `bytes_eq` prim).
//!
//! # Honesty tags (VR-5 — never upgraded in translation)
//! - **`Exact`** claims live in the ported matrix DATA (mirroring the crate's C2 rows); the row
//!   data itself is `Declared` (hand-transcribed, structurally asserted below).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) AND the
//!   Rust-oracle differential below, validated by trial on the corpus in this file; neither is a
//!   machine-checked proof.

mod harness;

use mycelium_core::{binary::bits_to_uint, CoreValue, Payload};
use mycelium_std_ternary::{
    add as rust_add, int_to_trits as rust_int_to_trits, max_magnitude as rust_max_magnitude,
    mul as rust_mul, neg as rust_neg, pack as rust_pack, sub as rust_sub,
    trits_to_int as rust_trits_to_int, Scheme as RustScheme, Trit as RustTrit,
};

/// The std.ternary nodule source, loaded at compile time — the single source of truth.
const TERNARY_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/ternary.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(TERNARY_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`] (same pattern as `std_error.rs`).
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Driver-construction fixtures — the corpus is data; test bodies are asserts over cases (house
// test-layout rule). `.myc`-side values are built as constructor text from the SAME Rust-oracle
// values the comparison uses, so the two sides can never drift apart silently.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Render a signed integer as a `.myc` `SInt` constructor expression (sign-magnitude over
/// `Binary{16}` — the FLAG-ternary-2 substitution).
fn sint_expr(v: i64) -> String {
    let mag = v.unsigned_abs();
    assert!(
        mag <= 0xFFFF,
        "corpus value {v} exceeds the Binary{{16}} port ceiling"
    );
    let bits = format!("0b{:016b}", mag);
    if v < 0 {
        format!("SNeg({bits})")
    } else {
        format!("SPos({bits})")
    }
}

/// Render a Rust-oracle trit slice as a `.myc` `Trits` cons-list expression (MSB-first).
fn trits_expr(ts: &[RustTrit]) -> String {
    let mut out = String::from("TNil");
    for t in ts.iter().rev() {
        let c = match t {
            RustTrit::Neg => "TNeg",
            RustTrit::Zero => "TZero",
            RustTrit::Pos => "TPos",
        };
        out = format!("TCons({c}, {out})");
    }
    out
}

/// Decode a `Binary{N}` [`CoreValue`] to its **unsigned** integer value ([`bits_to_uint`],
/// MSB-first) — the `Binary` surface is sign-free (ADR-028), and the packed-byte observables
/// reach 242 (> i8::MAX), so the two's-complement reading would mis-decode them.
fn extract_uint(cv: &CoreValue) -> i64 {
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Binary repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bits(bits) => i64::try_from(bits_to_uint(bits)).expect("observable fits i64"),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Run `driver`'s `main` (a raw `Binary{N}` observable) through the L1 evaluator and return the
/// decoded integer — the Rust-oracle bridge (same monomorphize/eval path as
/// [`harness::assert_three_way`]; the three-way obligation is carried by the cases above it).
fn eval_uint(label: &str, driver: &str) -> i64 {
    use mycelium_l1::elab::build_registry;
    use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

    let src = program(driver);
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));
    let val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let core = val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: result is outside the r3 data fragment"));
    extract_uint(&core)
}

/// The biased-scalar encoding both sides share for a signed-or-absent result:
/// `Some(v)` -> `BIAS + v` (v in [-BIAS+1, …]); `None` -> `0`. Never a silent sentinel — the
/// encoding is this fixture's explicit, documented observable, applied identically to the `.myc`
/// driver (via match arms) and the Rust oracle (via [`bias_encode`]).
const BIAS: i64 = 1000;

fn bias_encode(v: Option<i64>) -> i64 {
    match v {
        Some(v) => BIAS + v,
        None => 0,
    }
}

/// A `.myc` `main` that reduces `Option[Trits]`-producing `expr` to the shared biased scalar at
/// `Binary{16}`: None -> 0, Some(ts) -> BIAS + trits_to_int(ts) (sign folded via match).
fn biased_option_trits_driver(expr: &str) -> String {
    format!(
        "fn main() => Binary{{16}} = match {expr} {{ \
           None => 0b0000_0000_0000_0000, \
           Some(ts) => match trits_to_int(ts) {{ \
             SPos(mag) => add_u(0b0000_0011_1110_1000, mag), \
             SNeg(mag) => sub_u(0b0000_0011_1110_1000, mag) }} }};"
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Three-way differential cases (L1-eval ≡ elaborate→L0-interp ≡ AOT), one section per op group.
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ── Trit / Bit primitives ───────────────────────────────────────────────────────────────────────

/// `trit_new` accepts exactly {-1, 0, +1} and refuses off-domain — C1's explicit None.
#[test]
fn trit_new_domain_and_off_domain() {
    let driver = "fn main() => Option[Trit] = trit_new(SNeg(0b0000_0000_0000_0001));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Option[Trit] = Some(TNeg);";
    assert_three_way("trit_new(-1)", &program(driver), expected);

    let driver = "fn main() => Option[Trit] = trit_new(SPos(0b0000_0000_0000_0010));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Option[Trit] = None;";
    assert_three_way("trit_new(2) is None", &program(driver), expected);

    let driver = "fn main() => Option[Trit] = trit_new(SNeg(0b0000_0000_0000_0010));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Option[Trit] = None;";
    assert_three_way("trit_new(-2) is None", &program(driver), expected);
}

/// `trit_digit` inverts `trit_new` on the whole 3-value domain; `trit_neg` is the exact sign
/// flip and an involution (finite-domain exhaustive — all three trits).
#[test]
fn trit_digit_and_neg_exhaustive() {
    for (ctor, digit, negated) in [
        ("TNeg", "SNeg(0b0000_0000_0000_0001)", "TPos"),
        ("TZero", "SPos(0b0000_0000_0000_0000)", "TZero"),
        ("TPos", "SPos(0b0000_0000_0000_0001)", "TNeg"),
    ] {
        let driver = format!("fn main() => SInt = trit_digit({ctor});");
        let expected = format!(
            "nodule ref;\ntype SInt = SPos(Binary{{16}}) | SNeg(Binary{{16}});\nfn main() => SInt = {digit};"
        );
        assert_three_way(&format!("trit_digit({ctor})"), &program(&driver), &expected);

        let driver = format!("fn main() => Trit = trit_neg({ctor});");
        let expected = format!(
            "nodule ref;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Trit = {negated};"
        );
        assert_three_way(&format!("trit_neg({ctor})"), &program(&driver), &expected);

        let driver = format!("fn main() => Trit = trit_neg(trit_neg({ctor}));");
        let expected =
            format!("nodule ref;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Trit = {ctor};");
        assert_three_way(
            &format!("trit_neg involution ({ctor})"),
            &program(&driver),
            &expected,
        );
    }
}

/// Wire-byte round trip over all three trits, and explicit None on a non-glyph byte
/// (FLAG-ternary-1: the ASCII-byte substitution for the Rust char pair).
#[test]
fn wire_byte_round_trips_and_rejects() {
    for ctor in ["TNeg", "TZero", "TPos"] {
        let driver =
            format!("fn main() => Option[Trit] = trit_from_wire_byte(trit_to_wire_byte({ctor}));");
        let expected = format!(
            "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Option[Trit] = Some({ctor});"
        );
        assert_three_way(
            &format!("wire round-trip {ctor}"),
            &program(&driver),
            &expected,
        );
    }
    // 'a' = 97 is not a wire glyph.
    let driver = "fn main() => Option[Trit] = trit_from_wire_byte(0b0110_0001);";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\nfn main() => Option[Trit] = None;";
    assert_three_way("wire byte 'a' is None", &program(driver), expected);
}

/// `Bit` truth tables — exhaustive over the 2x2 domain for and/or/xor (Exact total algebra).
#[test]
fn bit_truth_tables_exhaustive() {
    let cases = [
        (
            "bit_and",
            [
                ("BZero", "BZero", "BZero"),
                ("BZero", "BOne", "BZero"),
                ("BOne", "BZero", "BZero"),
                ("BOne", "BOne", "BOne"),
            ],
        ),
        (
            "bit_or",
            [
                ("BZero", "BZero", "BZero"),
                ("BZero", "BOne", "BOne"),
                ("BOne", "BZero", "BOne"),
                ("BOne", "BOne", "BOne"),
            ],
        ),
        (
            "bit_xor",
            [
                ("BZero", "BZero", "BZero"),
                ("BZero", "BOne", "BOne"),
                ("BOne", "BZero", "BOne"),
                ("BOne", "BOne", "BZero"),
            ],
        ),
    ];
    for (op, table) in cases {
        for (a, b, out) in table {
            let driver = format!("fn main() => Bit = {op}({a}, {b});");
            let expected =
                format!("nodule ref;\ntype Bit = BZero | BOne;\nfn main() => Bit = {out};");
            assert_three_way(&format!("{op}({a},{b})"), &program(&driver), &expected);
        }
    }
}

/// `bit_new` accepts {0, 1} and refuses 2 and -1 (C1 explicit None).
#[test]
fn bit_new_domain_and_off_domain() {
    let driver = "fn main() => Option[Bit] = bit_new(SPos(0b0000_0000_0000_0001));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Bit = BZero | BOne;\nfn main() => Option[Bit] = Some(BOne);";
    assert_three_way("bit_new(1)", &program(driver), expected);

    let driver = "fn main() => Option[Bit] = bit_new(SNeg(0b0000_0000_0000_0001));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Bit = BZero | BOne;\nfn main() => Option[Bit] = None;";
    assert_three_way("bit_new(-1) is None", &program(driver), expected);
}

// ── codec: worked example + bounds (the spec's own witness values) ─────────────────────────────

/// The binary-ternary.md §5 worked example: -78 in 6 trits is <0,-1,0,0,+1,0> — and it decodes
/// back. (The expected side composes the SAME constructors — Derived, not Root, provenance.)
#[test]
fn worked_example_neg78_in_6_trits() {
    let driver =
        "fn main() => Option[Trits] = int_to_trits(SNeg(0b0000_0000_0100_1110), 0b0000_0110);";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\ntype Trits = TNil | TCons(Trit, Trits);\nfn main() => Option[Trits] = Some(TCons(TZero, TCons(TNeg, TCons(TZero, TCons(TZero, TCons(TPos, TCons(TZero, TNil)))))));";
    assert_three_way("int_to_trits(-78, 6)", &program(driver), expected);

    // Decode back: trits_to_int(<0,-1,0,0,+1,0>) == -78. (The scalar is compared in-driver via
    // `eq` so the observable is a provenance-free Bool datum — the expected side of a raw
    // Derived scalar would otherwise need the identical op-composition, per the harness
    // provenance convention.)
    let driver = "fn main() => Bool = match trits_to_int(TCons(TZero, TCons(TNeg, TCons(TZero, TCons(TZero, TCons(TPos, TCons(TZero, TNil))))))) { SNeg(mag) => match eq(mag, 0b0000_0000_0100_1110) { 0b1 => True, _ => False }, SPos(_) => False };";
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("trits_to_int(<0,-1,0,0,+1,0>)", &program(driver), expected);
}

/// Bounds (M-933 DoD): max_magnitude(6) = 364; ±364 encode, ±365 are explicit None (C1 — the
/// mutant witness: without the residual check, 365 would produce a wrong trit string).
#[test]
fn codec_bounds_at_width_6() {
    let driver = "fn main() => Bool = match max_magnitude(0b0000_0110) { Some(v) => match eq(v, 0b0000_0001_0110_1100) { 0b1 => True, _ => False }, None => False };";
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("max_magnitude(6) = 364", &program(driver), expected);

    for (v, is_some) in [(364i64, true), (-364, true), (365, false), (-365, false)] {
        let driver = format!(
            "fn main() => Bool = match int_to_trits({}, 0b0000_0110) {{ Some(ts) => True, None => False }};",
            sint_expr(v)
        );
        let expected = format!(
            "nodule ref;\nfn main() => Bool = {};",
            if is_some { "True" } else { "False" }
        );
        assert_three_way(
            &format!("int_to_trits({v}, 6) bound"),
            &program(&driver),
            &expected,
        );
    }
}

/// The Binary{16} ceiling (FLAG-ternary-2): max_magnitude(10) is the largest supported width;
/// max_magnitude(11) is an explicit None — the ported analogue of Rust's m >= 41 => None.
#[test]
fn max_magnitude_ceiling_at_m10() {
    // (3^10 - 1)/2 = 29524.
    let driver = "fn main() => Bool = match max_magnitude(0b0000_1010) { Some(v) => match eq(v, 0b0111_0011_0101_0100) { 0b1 => True, _ => False }, None => False };";
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("max_magnitude(10) = 29524", &program(driver), expected);

    let driver = "fn main() => Option[Binary{16}] = max_magnitude(0b0000_1011);";
    let expected =
        "nodule ref;\ntype Option[A] = Some(A) | None;\nfn main() => Option[Binary{16}] = None;";
    assert_three_way(
        "max_magnitude(11) = None (ceiling)",
        &program(driver),
        expected,
    );

    // Like the Rust crate: 0 encodes at any width — int_to_trits does not consult the ceiling.
    let driver = "fn main() => Bool = match int_to_trits(SPos(0b0000_0000_0000_0000), 0b0000_1011) { Some(ts) => True, None => False };";
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("int_to_trits(0, 11) is Some", &program(driver), expected);
}

// ── arithmetic: carries, overflow, identities (M-933 DoD: carries in the corpus) ───────────────

/// A carry-propagating addition: 1 + 1 = 2 at width 2 is <+,-> (2 = 3 - 1) — the digit sum
/// 1+1 emits digit -1 carry +1 (the balanced half-adder's carry case).
#[test]
fn add_carry_case_one_plus_one() {
    let driver = "fn main() => Option[Trits] = trits_add(TCons(TZero, TCons(TPos, TNil)), TCons(TZero, TCons(TPos, TNil)));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\ntype Trits = TNil | TCons(Trit, Trits);\nfn main() => Option[Trits] = Some(TCons(TPos, TCons(TNeg, TNil)));";
    assert_three_way("1 + 1 = <+,-> (carry)", &program(driver), expected);
}

/// Fixed-width overflow is an explicit None (mutant witness: without the final-carry check the
/// sum would silently wrap): at width 1, 1 + 1 = 2 > max_magnitude(1) = 1.
#[test]
fn add_overflow_is_none() {
    let driver = "fn main() => Bool = match trits_add(TCons(TPos, TNil), TCons(TPos, TNil)) { Some(ts) => True, None => False };";
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("1 + 1 overflows width 1", &program(driver), expected);
}

/// Unequal widths are an explicit None, not a silent partial result (C1).
#[test]
fn add_rejects_unequal_widths() {
    let driver = "fn main() => Bool = match trits_add(TCons(TPos, TNil), TCons(TZero, TCons(TPos, TNil))) { Some(ts) => True, None => False };";
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("unequal-width add is None", &program(driver), expected);
}

/// `trits_neg` is value negation and an involution on the worked-example string.
#[test]
fn neg_value_negation_and_involution() {
    // neg(<0,-1,0,0,+1,0>) = <0,+1,0,0,-1,0> — i.e. +78.
    let driver = "fn main() => Bool = match trits_to_int(trits_neg(TCons(TZero, TCons(TNeg, TCons(TZero, TCons(TZero, TCons(TPos, TCons(TZero, TNil)))))))) { SPos(mag) => match eq(mag, 0b0000_0000_0100_1110) { 0b1 => True, _ => False }, SNeg(_) => False };";
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("value(neg(-78)) = +78", &program(driver), expected);

    let driver =
        "fn main() => Trits = trits_neg(trits_neg(TCons(TZero, TCons(TNeg, TCons(TPos, TNil)))));";
    let expected = "nodule ref;\ntype Trit = TNeg | TZero | TPos;\ntype Trits = TNil | TCons(Trit, Trits);\nfn main() => Trits = TCons(TZero, TCons(TNeg, TCons(TPos, TNil)));";
    assert_three_way("neg is an involution", &program(driver), expected);
}

/// `trits_mul`: 2 * 2 = 4 at width 2 (max_magnitude(2) = 4 — exactly at the bound), and
/// 2 * 3 = 6 at width 2 is an explicit None (overflow past the high-trit check).
#[test]
fn mul_at_bound_and_overflow() {
    // 2 = <+,->, 4 = <+,+> at width 2.
    let driver = "fn main() => Option[Trits] = trits_mul(TCons(TPos, TCons(TNeg, TNil)), TCons(TPos, TCons(TNeg, TNil)));";
    let expected = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\ntype Trits = TNil | TCons(Trit, Trits);\nfn main() => Option[Trits] = Some(TCons(TPos, TCons(TPos, TNil)));";
    assert_three_way(
        "2 * 2 = 4 at width 2 (at the bound)",
        &program(driver),
        expected,
    );

    // 3 = <+,0>; 2 * 3 = 6 > 4 — None (mutant witness: a skipped high-trit check would
    // silently return the low trits).
    let driver = "fn main() => Bool = match trits_mul(TCons(TPos, TCons(TNeg, TNil)), TCons(TPos, TCons(TZero, TNil))) { Some(ts) => True, None => False };";
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("2 * 3 overflows width 2", &program(driver), expected);
}

// ── packing: schemes, worked bytes, round trips, misalignment ──────────────────────────────────

/// The Rust crate's own worked TL1/TL2 bytes: [Pos,0,0,0,0] packs to TL1 byte 202 and TL2 byte
/// 40 (242 - 202) — same trits, different bytes, same logical value (DN-01).
#[test]
fn tl1_tl2_worked_bytes() {
    let ts = "TCons(TPos, TCons(TZero, TCons(TZero, TCons(TZero, TCons(TZero, TNil)))))";
    for (scheme, byte) in [("Tl1", "0b1100_1010"), ("Tl2", "0b0010_1000")] {
        let driver = format!(
            "fn main() => Bool = match pack({ts}, {scheme}) {{ Ok(p) => match packed_bytes(p) {{ BCons(b, _) => match eq(b, {byte}) {{ 0b1 => True, _ => False }}, BNil => False }}, Err(_) => False }};"
        );
        let expected = "nodule ref;\nfn main() => Bool = True;".to_string();
        assert_three_way(
            &format!("{scheme} byte for [+,0,0,0,0]"),
            &program(&driver),
            &expected,
        );
    }
}

/// Pack/unpack round-trips losslessly under every scheme (I2S at 4 trits; TL1/TL2 at 5) — the
/// DN-01 §2 losslessness contract, exercised on a non-symmetric string.
#[test]
fn pack_unpack_round_trips() {
    // 4-trit string for I2S: <+,-,0,+> ; 5-trit for TL1/TL2: <+,-,0,+,->.
    let cases = [
        (
            "I2S",
            "TCons(TPos, TCons(TNeg, TCons(TZero, TCons(TPos, TNil))))",
        ),
        (
            "Tl1",
            "TCons(TPos, TCons(TNeg, TCons(TZero, TCons(TPos, TCons(TNeg, TNil)))))",
        ),
        (
            "Tl2",
            "TCons(TPos, TCons(TNeg, TCons(TZero, TCons(TPos, TCons(TNeg, TNil)))))",
        ),
    ];
    for (scheme, ts) in cases {
        let driver = format!(
            "fn main() => Result[Trits, PackError] = match pack({ts}, {scheme}) {{ Ok(p) => unpack(p), Err(e) => Err(e) }};"
        );
        let expected = format!(
            "nodule ref;\ntype Result[A, E] = Ok(A) | Err(E);\ntype Trit = TNeg | TZero | TPos;\ntype Trits = TNil | TCons(Trit, Trits);\ntype PackError = OffGrid | Misaligned;\nfn main() => Result[Trits, PackError] = Ok({ts});"
        );
        assert_three_way(
            &format!("{scheme} round-trip"),
            &program(&driver),
            &expected,
        );
    }
}

/// A misaligned trit count is an explicit Err(Misaligned) (mutant witness: without the
/// alignment check a 3-trit I2S input would produce a malformed partial byte).
#[test]
fn pack_rejects_misaligned() {
    let driver = "fn main() => Bool = match pack(TCons(TPos, TCons(TZero, TCons(TNeg, TNil))), I2S) { Ok(p) => True, Err(e) => False };";
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("3 trits misaligned for I2S", &program(driver), expected);
}

/// `scheme_of` and `explain` expose the inspectable Meta.physical record (C3): the scheme, the
/// explicit-caller selection note, and the trit/byte counts.
#[test]
fn scheme_of_and_explain_records() {
    let ts = "TCons(TPos, TCons(TZero, TCons(TZero, TCons(TZero, TNil))))";
    let driver = format!(
        "fn main() => Scheme = match pack({ts}, I2S) {{ Ok(p) => scheme_of(p), Err(e) => Tl1 }};"
    );
    let expected = "nodule ref;\ntype Scheme = I2S | Tl1 | Tl2;\nfn main() => Scheme = I2S;";
    assert_three_way(
        "scheme_of is the packed scheme",
        &program(&driver),
        expected,
    );

    let driver = format!(
        "fn main() => Bool = match pack({ts}, I2S) {{ Err(_) => False, Ok(p) => match explain(p) {{ ExplainRec(s, _, n, bc) => match s {{ I2S => bool_and(match eq(n, 0b0000_0100) {{ 0b1 => True, _ => False }}, match eq(bc, 0b0000_0001) {{ 0b1 => True, _ => False }}), Tl1 => False, Tl2 => False }} }} }};"
    );
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way(
        "explain: 4 trits, 1 byte, explicit caller",
        &program(&driver),
        expected,
    );
}

/// FLAG-ternary-4's defensive gate: a forged Packed whose trit_count disagrees with its bytes is
/// an explicit Err(OffGrid), never a silent wrong-length answer.
#[test]
fn unpack_rejects_forged_trit_count() {
    let driver = "fn main() => Bool = match unpack(MkPacked(BCons(0b0000_0000, BNil), I2S, 0b0000_0101)) { Ok(ts) => True, Err(e) => False };";
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("forged trit_count is Err", &program(driver), expected);
}

// ── guarantee matrix: the structural invariants (RFC-0016 §4.5) ────────────────────────────────

/// The matrix has exactly 18 rows (the crate's op count), every row tags GExact (C2/VR-5),
/// every row states effects, every fallible row names its error set (C1/G2), and exactly the
/// 4 pack ops are EXPLAIN-able (the count form of the name-keyed Rust check — FLAG-ternary-5).
#[test]
fn matrix_structural_invariants() {
    let cases = [
        ("matrix has 18 rows", "fn main() => Bool = match eq(matrix_len(matrix()), 0b0001_0010) { 0b1 => True, _ => False };", "fn main() => Bool = True;"),
        ("all tags Exact", "fn main() => Bool = all_exact(matrix());", "fn main() => Bool = True;"),
        ("all effects stated", "fn main() => Bool = all_effects_nonempty(matrix());", "fn main() => Bool = True;"),
        ("fallible rows name error sets", "fn main() => Bool = fallible_rows_name_their_error_set(matrix());", "fn main() => Bool = True;"),
        ("exactly 4 EXPLAIN-able ops", "fn main() => Bool = match eq(count_explainable(matrix()), 0b0000_0100) { 0b1 => True, _ => False };", "fn main() => Bool = True;"),
    ];
    for (label, driver, expected_main) in cases {
        let expected = format!("nodule ref;\n{expected_main}");
        assert_three_way(label, &program(driver), &expected);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Rust-oracle differential (D5 row 4) — wired against the RETAINED `mycelium-std-ternary` crate
// (RFC-0031 D6: the crate is NOT retired). The numeric-edge corpus (bounds, carries, round
// trips) is evaluated on BOTH sides at test time and compared through the shared biased-scalar
// observable — the M-933 DoD's "numeric edge cases (carries, bounds) in the differential
// corpus" obligation, live, never hand-copied into one side only.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The numeric-edge value corpus at width 6 (max ±364): zero, units, the worked example, the
/// exact bounds, and just-past-the-bounds.
fn edge_values_w6() -> Vec<i64> {
    vec![
        0, 1, -1, 2, -2, 78, -78, 121, -121, 363, -363, 364, -364, 365, -365, 500, -500,
    ]
}

/// Codec round-trip vs the Rust oracle: for every edge value v,
/// `.myc` `int_to_trits(v, 6) |> trits_to_int` (biased) must equal the oracle's
/// `int_to_trits(v, 6).map(|t| trits_to_int(&t))` (biased) — including the None edges.
#[test]
fn oracle_codec_round_trip_edges() {
    for v in edge_values_w6() {
        let driver =
            biased_option_trits_driver(&format!("int_to_trits({}, 0b0000_0110)", sint_expr(v)));
        let myc = eval_uint(&format!("codec round-trip v={v}"), &driver);

        let rust = bias_encode(rust_int_to_trits(v, 6).map(|ts| rust_trits_to_int(&ts)));
        assert_eq!(
            myc, rust,
            "int_to_trits({v}, 6) |> trits_to_int must match the Rust oracle (biased observable)"
        );
    }
}

/// Digit-exact codec agreement: for in-range edge values the `.myc` trit string must equal the
/// oracle's digit-for-digit (the expected side is BUILT from the oracle's own output at test
/// time, then compared through the three-way harness — no hand-copied digits).
#[test]
fn oracle_codec_digits_match() {
    for v in edge_values_w6() {
        let Some(oracle_trits) = rust_int_to_trits(v, 6) else {
            continue; // the None edges are covered by the biased round-trip test above
        };
        let driver = format!(
            "fn main() => Option[Trits] = int_to_trits({}, 0b0000_0110);",
            sint_expr(v)
        );
        let expected = format!(
            "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Trit = TNeg | TZero | TPos;\ntype Trits = TNil | TCons(Trit, Trits);\nfn main() => Option[Trits] = Some({});",
            trits_expr(&oracle_trits)
        );
        assert_three_way(
            &format!("oracle digits v={v}"),
            &program(&driver),
            &expected,
        );
    }
}

/// max_magnitude agrees with the oracle across widths 0..=10 (and both ceilings refuse:
/// the oracle at m=41, the port at m=11 — each side's ceiling is its own documented bound,
/// FLAG-ternary-2, asserted here rather than papered over).
#[test]
fn oracle_max_magnitude_agrees_on_shared_domain() {
    for m in 0u32..=10 {
        let driver = format!(
            "fn main() => Binary{{16}} = match max_magnitude(0b{m:08b}) {{ Some(v) => v, None => 0b0000_0000_0000_0000 }};"
        );
        let myc = eval_uint(&format!("max_magnitude({m})"), &driver);
        let rust = rust_max_magnitude(m).expect("m <= 10 is in the oracle's i64 range");
        assert_eq!(myc, rust, "max_magnitude({m}) must match the Rust oracle");
    }
    // The port's ceiling (m = 11) refuses explicitly; the oracle still answers (i64 headroom).
    let driver = "fn main() => Binary{8} = match max_magnitude(0b0000_1011) { Some(_) => 0b0000_0001, None => 0b0000_0000 };";
    assert_eq!(
        eval_uint("ceiling m=11", driver),
        0,
        "m=11 must be None in the port"
    );
    assert!(
        rust_max_magnitude(11).is_some(),
        "the oracle's i64 ceiling is higher (m <= 40)"
    );
    assert!(
        rust_max_magnitude(41).is_none(),
        "the oracle's own ceiling refuses at m = 41"
    );
}

/// Addition vs the Rust oracle over a carry-heavy pair corpus at width 4 (max ±40): both sides
/// must agree on every sum INCLUDING the overflow Nones (the biased observable).
#[test]
fn oracle_add_carry_and_bound_pairs() {
    let pairs: &[(i64, i64)] = &[
        (1, 1),     // single carry
        (13, 14),   // multi-digit carries
        (-13, -14), // negative carries
        (40, -40),  // bound + inverse = 0
        (40, 1),    // overflow past +max
        (-40, -1),  // overflow past -max
        (39, 1),    // lands exactly on +max
        (-39, -1),  // lands exactly on -max
        (20, 21),   // overflow by 1
        (0, 0),
    ];
    for &(x, y) in pairs {
        let a = trits_expr(&rust_int_to_trits(x, 4).expect("x fits width 4"));
        let b = trits_expr(&rust_int_to_trits(y, 4).expect("y fits width 4"));
        let driver = biased_option_trits_driver(&format!("trits_add({a}, {b})"));
        let myc = eval_uint(&format!("add({x},{y})"), &driver);

        let ra = rust_int_to_trits(x, 4).expect("in range");
        let rb = rust_int_to_trits(y, 4).expect("in range");
        let rust = bias_encode(rust_add(&ra, &rb).map(|ts| rust_trits_to_int(&ts)));
        assert_eq!(
            myc, rust,
            "add({x},{y}) at width 4 must match the Rust oracle"
        );
    }
}

/// Subtraction and negation vs the oracle on the same edge pairs (sub = add . neg identity on
/// both sides).
#[test]
fn oracle_sub_and_neg_pairs() {
    let pairs: &[(i64, i64)] = &[(1, 1), (13, -14), (-40, 40), (40, 40), (0, 40), (-39, 1)];
    for &(x, y) in pairs {
        let a = trits_expr(&rust_int_to_trits(x, 4).expect("x fits width 4"));
        let b = trits_expr(&rust_int_to_trits(y, 4).expect("y fits width 4"));
        let driver = biased_option_trits_driver(&format!("trits_sub({a}, {b})"));
        let myc = eval_uint(&format!("sub({x},{y})"), &driver);

        let ra = rust_int_to_trits(x, 4).expect("in range");
        let rb = rust_int_to_trits(y, 4).expect("in range");
        let rust = bias_encode(rust_sub(&ra, &rb).map(|ts| rust_trits_to_int(&ts)));
        assert_eq!(
            myc, rust,
            "sub({x},{y}) at width 4 must match the Rust oracle"
        );
    }
    // neg: value(neg(t)) == -value(t) for the worked example, on both sides.
    let a = trits_expr(&rust_int_to_trits(-78, 6).expect("in range"));
    let driver = format!(
        "fn main() => Binary{{16}} = match trits_to_int(trits_neg({a})) {{ SPos(mag) => add_u(0b0000_0011_1110_1000, mag), SNeg(mag) => sub_u(0b0000_0011_1110_1000, mag) }};"
    );
    let myc = eval_uint("neg(-78)", &driver);
    let ra = rust_int_to_trits(-78, 6).expect("in range");
    let rust = BIAS + rust_trits_to_int(&rust_neg(&ra));
    assert_eq!(myc, rust, "value(neg(-78)) must match the Rust oracle");
}

/// Multiplication vs the oracle at width 3 (max ±13): products at and past the bound.
#[test]
fn oracle_mul_bound_pairs() {
    let pairs: &[(i64, i64)] = &[
        (2, 2),   // 4 in range
        (3, 4),   // 12 in range
        (-3, 4),  // -12 in range
        (13, 1),  // exactly max
        (13, -1), // exactly min
        (4, 4),   // 16 overflows
        (-13, 2), // -26 overflows
        (0, 13),  // zero edge
    ];
    for &(x, y) in pairs {
        let a = trits_expr(&rust_int_to_trits(x, 3).expect("x fits width 3"));
        let b = trits_expr(&rust_int_to_trits(y, 3).expect("y fits width 3"));
        let driver = biased_option_trits_driver(&format!("trits_mul({a}, {b})"));
        let myc = eval_uint(&format!("mul({x},{y})"), &driver);

        let ra = rust_int_to_trits(x, 3).expect("in range");
        let rb = rust_int_to_trits(y, 3).expect("in range");
        let rust = bias_encode(rust_mul(&ra, &rb).map(|ts| rust_trits_to_int(&ts)));
        assert_eq!(
            myc, rust,
            "mul({x},{y}) at width 3 must match the Rust oracle"
        );
    }
}

/// Packed bytes vs the oracle: every 5-trit encoding of the width-5 range's edge values packs to
/// the SAME byte value under TL1 and TL2 as `mycelium_std_ternary::pack` produces (and the same
/// I2S byte for a 4-trit edge string) — byte-for-byte codec agreement, not just round-trip.
#[test]
fn oracle_packed_bytes_match() {
    // TL1/TL2 over 5-trit strings.
    for v in [0i64, 1, -1, 121, -121, 60, -60] {
        let ts = rust_int_to_trits(v, 5).expect("v fits width 5");
        let myc_ts = trits_expr(&ts);
        for (scheme_myc, scheme_rust) in [("Tl1", RustScheme::Tl1), ("Tl2", RustScheme::Tl2)] {
            let driver = format!(
                "fn main() => Binary{{8}} = match pack({myc_ts}, {scheme_myc}) {{ Ok(p) => match packed_bytes(p) {{ BCons(b, _) => b, BNil => 0b1111_1111 }}, Err(e) => 0b1111_1110 }};"
            );
            let myc = eval_uint(&format!("{scheme_myc} byte v={v}"), &driver);
            let rust_packed = rust_pack(&ts, scheme_rust).expect("aligned 5-trit pack");
            let rust_byte = i64::from(rust_packed.bytes()[0]);
            assert_eq!(
                myc, rust_byte,
                "{scheme_myc} byte for v={v} must match the Rust oracle byte-for-byte"
            );
        }
    }
    // I2S over a 4-trit string.
    let ts = rust_int_to_trits(7, 4).expect("7 fits width 4");
    let driver = format!(
        "fn main() => Binary{{8}} = match pack({}, I2S) {{ Ok(p) => match packed_bytes(p) {{ BCons(b, _) => b, BNil => 0b1111_1111 }}, Err(e) => 0b1111_1110 }};",
        trits_expr(&ts)
    );
    let myc = eval_uint("I2S byte v=7", &driver);
    let rust_packed = rust_pack(&ts, RustScheme::I2S).expect("aligned 4-trit pack");
    assert_eq!(
        myc,
        i64::from(rust_packed.bytes()[0]),
        "I2S byte for v=7 must match the Rust oracle byte-for-byte"
    );
}

/// The ported matrix mirrors the oracle's: same row count and same all-Exact/pure/explainable
/// shape, read from the ACTUAL `mycelium_std_ternary::guarantee_matrix::MATRIX` at test time.
#[test]
fn oracle_matrix_shape_matches() {
    use mycelium_std_ternary::guarantee_matrix::{Explainable, Tag, MATRIX};

    let myc_len = eval_uint(
        "matrix len",
        "fn main() => Binary{8} = matrix_len(matrix());",
    );
    assert_eq!(
        myc_len,
        MATRIX.len() as i64,
        "row count must match the oracle's MATRIX"
    );

    let myc_explainable = eval_uint(
        "matrix explainable count",
        "fn main() => Binary{8} = count_explainable(matrix());",
    );
    let rust_explainable = MATRIX
        .iter()
        .filter(|r| r.explainable == Explainable::Yes)
        .count() as i64;
    assert_eq!(
        myc_explainable, rust_explainable,
        "EXPLAIN-able op count must match the oracle's MATRIX"
    );

    assert!(
        MATRIX.iter().all(|r| r.tag == Tag::Exact),
        "oracle invariant: every row Exact (C2/VR-5)"
    );
    let myc_all_exact = eval_uint(
        "matrix all exact",
        "fn main() => Binary{8} = match all_exact(matrix()) { True => 0b0000_0001, False => 0b0000_0000 };",
    );
    assert_eq!(
        myc_all_exact, 1,
        "ported matrix must also be all-Exact (VR-5, same strength)"
    );
}
