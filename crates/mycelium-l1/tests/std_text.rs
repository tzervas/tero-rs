//! Differential tests for `std.text` (M-717, #462) — the self-hosted UTF-8 byte/text utilities.
//!
//! The nodule source is loaded verbatim via `include_str!` (the single source of truth), then a
//! typed driver `fn main` is appended to exercise each operation. The `assert_three_way` harness
//! mirrors `std_option.rs` exactly: L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT, all three paths
//! agree AND equal the `expected` reference value.
//!
//! # Generic pinning
//! `Option<A>` and `Result<A,E>` in `std.text` are pinned to concrete `Binary{8}` / `Utf8Error`
//! types via explicit return-type annotations on the driver strings (and the `DECODE_REF_PREAMBLE`
//! constant for `decode_ascii` tests) — without pinning, the monomorphizer
//! emits a never-silent `Residual` (G2).
//!
//! # Honesty tags
//! - **`Exact`** — `byte_len` (delegates to `bytes_len`), `is_ascii_byte`/`is_cont_byte` (total via
//!   `lt`+match), the `width_cast`/`lt`/`and`/`or`/`add_u` bit ops the decode is assembled from.
//! - **`Declared`** — `byte_at` (Option bounds-check contract), `decode_ascii`/`decode_one`
//!   (never-silent type-level contracts; structural composition of Exact parts, not machine-proven).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT),
//!   validated by trial on the programs below; not a machine-checked proof.
//!
//! # Scope / FLAGs (honest boundary — VR-5)
//! - FLAG-text-1: **CLOSED** (DN-41 / M-798). `byte_at` is now an Option-returning bounds-checked
//!   access via `lt(width_cast(i, bytes_len(b)), bytes_len(b))` — the `width_cast` widen bridges the
//!   `Binary{8}` index to the `Binary{32}` length, the gap wave-n1 flagged.
//! - FLAG-text-2: **CLOSED** (DN-41 / M-798). `decode_one` returns the full `Binary{32}` codepoint
//!   (1/2/3/4-byte UTF-8); `width_cast` lifts the masked payloads, shifts are repeated `add_u`
//!   doublings (no shift prim). `decode_ascii` is retained as the `Binary{8}` 1-byte fast path.
//! - **UTF-8 validity layer (M-717 remainder): CLOSED.** `decode_one` now rejects overlong encodings,
//!   surrogate-range codepoints (U+D800–DFFF), and codepoints > U+10FFFF via the `reject_two/three/four`
//!   gates — each a never-silent `Err(Overlong/Surrogate/TooLarge(lead))`. Boundary values (U+0080,
//!   U+10FFFF) are accepted, not over-rejected. Structural malformations remain `Err(Invalid(byte))`.
//! - FLAG-text-3: **CLOSED** (DN-43 / M-799). `bytes_slice`/`bytes_concat` are now surface-callable
//!   over the kernel `Bytes` (`text.myc`'s `slice`/`concat` delegate to them); three-way coverage
//!   lives in `std_bytes_slice.rs`. The `Bytes8` cons-list type is now superseded (kept declared for
//!   append-only minimalism). `decode_one` still returns a `Pair(codepoint, byte_width)` (the caller
//!   advances by the width) — that is a decode convention, independent of slice availability.
//!
//! # Anchor
//! Expected values are hand-computed and verified three-way (L1≡L0≡AOT). The Rust crate
//! crates/mycelium-std-text exists but exposes a different Ring-2 surface (no decode_ascii over a
//! .myc port), so it is the value oracle for shared semantics only — not a structural reference.

use mycelium_cert::{check_core, BinaryTernarySwapEngine, CheckVerdict};
use mycelium_core::GuaranteeStrength;
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator};

/// The std.text nodule source, loaded at compile time — the single source of truth.
const TEXT_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/text.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    format!("{TEXT_SRC}\n{driver}")
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

// ── byte_len ──────────────────────────────────────────────────────────────────────────────────────

/// `byte_len(0x48_65_6c_6c_6f)` → `Binary{32}(5)`.
/// Reference: `bytes_len` on the UTF-8 encoding of "Hello" is exactly 5 (Exact).
/// Hand-computed: 0x48=H, 0x65=e, 0x6c=l, 0x6c=l, 0x6f=o — 5 bytes.
/// Grounding: hand-computed + enablement.rs bytes_len tests; mycelium-std-text exists but is a
/// different Ring-2 surface, not the oracle.
#[test]
fn byte_len_returns_count() {
    let driver = "fn main() => Binary{32} = byte_len(0x48_65_6c_6c_6f);";
    let src = program(driver);
    // Binary{32}(5) MSB-first: 0b00000000_00000000_00000000_00000101
    let expected = "nodule ref;\nfn main() => Binary{32} = bytes_len(0x48_65_6c_6c_6f);";
    assert_three_way("byte_len(Hello)", &src, expected);
}

/// `byte_len(0x01_02_03)` → `Binary{32}(3)` — mirrors enablement.rs bytes_len_surface_three_way.
#[test]
fn byte_len_three_byte_input() {
    let driver = "fn main() => Binary{32} = byte_len(0x01_02_03);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{32} = bytes_len(0x01_02_03);";
    assert_three_way("byte_len(3 bytes)", &src, expected);
}

// ── is_ascii_byte ─────────────────────────────────────────────────────────────────────────────────

/// `is_ascii_byte(0b0100_0001)` → `True` ('A' = 0x41 < 0x80; Exact).
/// Hand-computed: 0x41 = 65 < 128; the lt prim returns 0b1 → True.
#[test]
fn is_ascii_byte_true_for_ascii() {
    // 0b0100_0001 = 0x41 = 'A': high bit clear → ASCII.
    let driver = "fn main() => Bool = is_ascii_byte(0b0100_0001);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_ascii_byte(0x41=A → True)", &src, expected);
}

/// `is_ascii_byte(0b0000_0000)` → `True` (NUL byte = 0x00; Exact).
#[test]
fn is_ascii_byte_true_for_nul() {
    let driver = "fn main() => Bool = is_ascii_byte(0b0000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_ascii_byte(0x00=NUL → True)", &src, expected);
}

/// `is_ascii_byte(0b0111_1111)` → `True` (DEL = 0x7F = 127 < 128; Exact).
#[test]
fn is_ascii_byte_true_for_max_ascii() {
    // 0b0111_1111 = 0x7F = 127: last valid ASCII value.
    let driver = "fn main() => Bool = is_ascii_byte(0b0111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_ascii_byte(0x7F → True)", &src, expected);
}

/// `is_ascii_byte(0b1000_0000)` → `False` (= 0x80; first non-ASCII byte; Exact).
/// Hand-computed: 0x80 = 128, not < 128 → lt returns `_` arm → False.
#[test]
fn is_ascii_byte_false_for_continuation() {
    // 0b1000_0000 = 0x80: the first byte with the high bit set — a 2-byte UTF-8 lead range start.
    let driver = "fn main() => Bool = is_ascii_byte(0b1000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_ascii_byte(0x80 → False)", &src, expected);
}

/// `is_ascii_byte(0b1111_1111)` → `False` (= 0xFF; Exact).
#[test]
fn is_ascii_byte_false_for_0xff() {
    let driver = "fn main() => Bool = is_ascii_byte(0b1111_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_ascii_byte(0xFF → False)", &src, expected);
}

// ── decode_ascii ──────────────────────────────────────────────────────────────────────────────────
//
// `decode_ascii` returns Result[Binary{8}, Utf8Error] — the generic parameters are pinned by
// annotating the expected-ref type explicitly. The three `Result`/`Utf8Error` types must be
// redeclared in the reference program so `eval_core` produces a compatible CoreValue.

/// Reference program preamble for decode_ascii tests: re-declare the local types so that
/// `eval_core` produces a compatible CoreValue (same type ContentHash as the test program).
const DECODE_REF_PREAMBLE: &str = "nodule ref;\ntype Option[A] = Some(A) | None;\ntype Result[A, E] = Ok(A) | Err(E);\ntype Utf8Error = Invalid(Binary{8}) | Overlong(Binary{8}) | Surrogate(Binary{8}) | TooLarge(Binary{8});\n";

/// `decode_ascii(0x41_42_43, 0b0000_0000)` → `Ok(bytes_get(…, 0))` (= Ok(0x41='A'); Declared/Empirical).
/// The byte at index 0 of [0x41, 0x42, 0x43] is 0x41 = 'A' — ASCII, so Ok.
/// The reference program uses `bytes_get` to match the `Derived` provenance of the computed value
/// (a literal `Ok(0b0100_0001)` would have `Root` provenance — see std_option.rs `map` comment).
/// Grounding: hand-computed, three-way verified; mycelium-std-text exists but is a different Ring-2 surface, not the oracle.
#[test]
fn decode_ascii_ok_on_valid_ascii() {
    let driver =
        "fn main() => Result[Binary{8}, Utf8Error] = decode_ascii(0x41_42_43, 0b0000_0000);";
    let src = program(driver);
    // Reference: Ok wrapping the same bytes_get call to share Derived provenance.
    let expected = format!(
        "{DECODE_REF_PREAMBLE}fn main() => Result[Binary{{8}}, Utf8Error] = Ok(bytes_get(0x41_42_43, 0b0000_0000));"
    );
    assert_three_way("decode_ascii(ABC, 0)=Ok(A)", &src, &expected);
}

/// `decode_ascii(0x43_44_45, 0b0000_0010)` → `Ok(bytes_get(…, 2))` (= Ok(0x45='E'); Declared/Empirical).
/// Index 2 of [0x43, 0x44, 0x45] = 0x45 = 'E'; ASCII → Ok.
#[test]
fn decode_ascii_ok_at_offset() {
    let driver =
        "fn main() => Result[Binary{8}, Utf8Error] = decode_ascii(0x43_44_45, 0b0000_0010);";
    let src = program(driver);
    // Reference: Ok(bytes_get(…, 2)) — Derived provenance to match computed result.
    let expected = format!(
        "{DECODE_REF_PREAMBLE}fn main() => Result[Binary{{8}}, Utf8Error] = Ok(bytes_get(0x43_44_45, 0b0000_0010));"
    );
    assert_three_way("decode_ascii(CDE, 2)=Ok(E)", &src, &expected);
}

/// `decode_ascii(0xc3_a9, 0b0000_0000)` → `Err(Invalid(bytes_get(…, 0)))` — never-silent (G2).
/// 0xc3 = 0b1100_0011 is the UTF-8 lead byte for U+00E9 (é); it has the high bit set → not ASCII.
/// Never-silent: the malformed lead is returned as the offending byte, never U+FFFD.
/// Hand-computed: is_ascii_byte(0xC3) = False → Err(Invalid(0xC3)).
/// Grounding: hand-computed, three-way verified; mycelium-std-text exists but is a different Ring-2 surface, not the oracle.
#[test]
fn decode_ascii_err_on_multibyte_lead() {
    // 0xc3_a9 is the UTF-8 encoding of 'é' (U+00E9). The lead byte 0xC3 has high bit set → Err.
    let driver = "fn main() => Result[Binary{8}, Utf8Error] = decode_ascii(0xc3_a9, 0b0000_0000);";
    let src = program(driver);
    // Reference: Err(Invalid(bytes_get(…, 0))) — Derived provenance to match computed result.
    let expected = format!(
        "{DECODE_REF_PREAMBLE}fn main() => Result[Binary{{8}}, Utf8Error] = Err(Invalid(bytes_get(0xc3_a9, 0b0000_0000)));"
    );
    assert_three_way("decode_ascii(é-lead)=Err(Invalid(0xC3))", &src, &expected);
}

/// `decode_ascii(0x80_bf, 0b0000_0000)` → `Err(Invalid(bytes_get(…, 0)))` — never-silent (G2).
/// 0x80 is a bare UTF-8 continuation byte (not valid as a lead); its high bit is set → Err.
#[test]
fn decode_ascii_err_on_continuation_byte() {
    // 0x80 = 0b1000_0000: bare continuation byte — invalid lead, non-ASCII.
    let driver = "fn main() => Result[Binary{8}, Utf8Error] = decode_ascii(0x80_bf, 0b0000_0000);";
    let src = program(driver);
    // Reference: Err(Invalid(bytes_get(…, 0))) — Derived provenance to match computed result.
    let expected = format!(
        "{DECODE_REF_PREAMBLE}fn main() => Result[Binary{{8}}, Utf8Error] = Err(Invalid(bytes_get(0x80_bf, 0b0000_0000)));"
    );
    assert_three_way("decode_ascii(0x80-continuation)=Err", &src, &expected);
}

// ── byte_at (FLAG-text-1 closed by DN-41 width_cast) ────────────────────────────────────────────────
//
// `byte_at(b, i)` bounds-checks the `Binary{8}` index `i` against the `Binary{32}` `bytes_len(b)` via
// `lt(width_cast(i, bytes_len(b)), bytes_len(b))` — the exact DN-41/M-798 pattern. In range yields
// `Some(byte)`; out of range yields `None` (never-silent, G2). Reference programs reuse `bytes_get`
// (in range) so the wrapped value shares `Derived` provenance with the computed result; `None` is a
// nullary constructor (no provenance to match). Both are pinned to `Option[Binary{8}]`.

/// `byte_at(0x41_42_43, 0b0000_0001)` → `Some(bytes_get(…, 1))` (= Some(0x42='B'); Declared/Empirical).
/// Index 1 of [0x41, 0x42, 0x43] is in range (1 < 3) → Some. Grounding: hand-computed, three-way verified.
#[test]
fn byte_at_some_in_range() {
    let driver = "fn main() => Option[Binary{8}] = byte_at(0x41_42_43, 0b0000_0001);";
    let src = program(driver);
    // Reference: Some(bytes_get(…, 1)) — Derived provenance to match the computed in-range byte.
    let expected =
        program("fn main() => Option[Binary{8}] = Some(bytes_get(0x41_42_43, 0b0000_0001));");
    assert_three_way("byte_at(ABC, 1)=Some(B)", &src, &expected);
}

/// `byte_at(0x41_42_43, 0b0000_0000)` → `Some(bytes_get(…, 0))` (= Some(0x41='A')) — boundary index 0.
#[test]
fn byte_at_some_at_zero() {
    let driver = "fn main() => Option[Binary{8}] = byte_at(0x41_42_43, 0b0000_0000);";
    let src = program(driver);
    let expected =
        program("fn main() => Option[Binary{8}] = Some(bytes_get(0x41_42_43, 0b0000_0000));");
    assert_three_way("byte_at(ABC, 0)=Some(A)", &src, &expected);
}

/// `byte_at(0x41_42_43, 0b0000_0011)` → `None` — index 3 is out of range (3 is NOT < 3); never-silent.
/// The out-of-range index is an explicit `None`, never a kernel refusal and never a silent wrap (G2).
/// Grounding: hand-computed (len 3, index 3 past end), three-way verified.
#[test]
fn byte_at_none_out_of_range() {
    let driver = "fn main() => Option[Binary{8}] = byte_at(0x41_42_43, 0b0000_0011);";
    let src = program(driver);
    let expected = program("fn main() => Option[Binary{8}] = None;");
    assert_three_way("byte_at(ABC, 3)=None (oob)", &src, &expected);
}

/// `byte_at(0x41_42_43, 0b1111_1111)` → `None` — index 255 (far past end) is out of range; never-silent.
#[test]
fn byte_at_none_far_out_of_range() {
    let driver = "fn main() => Option[Binary{8}] = byte_at(0x41_42_43, 0b1111_1111);";
    let src = program(driver);
    let expected = program("fn main() => Option[Binary{8}] = None;");
    assert_three_way("byte_at(ABC, 255)=None (oob)", &src, &expected);
}

// ── decode_one — full multi-byte UTF-8 decode (FLAG-text-2 closed by DN-41 width_cast) ──────────────
//
// `decode_one(b, i)` returns `Ok(Pr(codepoint : Binary{32}, byte_width : Binary{8}))` for the UTF-8
// sequence starting at `i`, or `Err(Invalid(byte))` on any structural malformation (never-silent, G2).
// The codepoint is the full `Binary{32}` Unicode scalar (FLAG-text-2 was the `Binary{8}` cap; the
// `width_cast` widen lifts it). Reference programs **recompute** the codepoint via the same nodule
// helper expressions (`shl6`/`widen8`/`cont_payload`/`or` etc.) so the computed value shares its
// `Derived` provenance with the reference — a literal codepoint would have `Root` provenance and would
// not compare equal (Meta carries provenance). All expected values are hand-computed and cross-checked
// against Python's UTF-8 decoder (é=233, €=8364, 😀=128512).

/// `decode_one(0x41_42_43, 0b0000_0000)` → `Ok(Pr(widen8('A'), 1))` — 1-byte ASCII path.
/// 0x41='A' < 0x80 → 1-byte; codepoint = widen8(0x41) = 65, width 1. Declared/Empirical.
#[test]
fn decode_one_ascii_one_byte() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0x41_42_43, 0b0000_0000);";
    let src = program(driver);
    // Reference: recompute via the same `widen8(bytes_get(…))` so the Binary{32} codepoint shares
    // Derived provenance with decode_one's `widen8(lead)`.
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(widen8(bytes_get(0x41_42_43, 0b0000_0000)), 0b0000_0001));",
    );
    assert_three_way("decode_one(A)=Ok(Pr(65,1))", &src, &expected);
}

/// `decode_one(0xc3_a9, 0b0000_0000)` → `Ok(Pr(233, 2))` — 2-byte path (é = U+00E9).
/// Lead 0xC3 ∈ 0xC0..0xDF → 2-byte; cont 0xA9 is valid (0x80..0xBF). cp = (0xC3 & 0x1F)<<6 | (0xA9 &
/// 0x3F) = 3<<6 | 41 = 192+41 = 233 = U+00E9. Hand-computed + Python-verified. Declared/Empirical.
#[test]
fn decode_one_two_byte_e_acute() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xc3_a9, 0b0000_0000);";
    let src = program(driver);
    // Reference: recompute the codepoint with the same assembly `decode_two` uses (matching provenance).
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(shl6(widen8(and(bytes_get(0xc3_a9, 0b0000_0000), 0b0001_1111))), cont_payload(bytes_get(0xc3_a9, 0b0000_0001))), 0b0000_0010));",
    );
    assert_three_way("decode_one(é)=Ok(Pr(233,2))", &src, &expected);
}

/// `decode_one(0xe2_82_ac, 0b0000_0000)` → `Ok(Pr(8364, 3))` — 3-byte path (€ = U+20AC).
/// Lead 0xE2 ∈ 0xE0..0xEF → 3-byte; conts 0x82, 0xAC valid. cp = (0xE2 & 0x0F)<<12 | (0x82 & 0x3F)<<6
/// | (0xAC & 0x3F) = 2<<12 | 2<<6 | 44 = 8192+128+44 = 8364 = U+20AC. Hand-computed + Python-verified.
#[test]
fn decode_one_three_byte_euro() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xe2_82_ac, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(or(shl12(widen8(and(bytes_get(0xe2_82_ac, 0b0000_0000), 0b0000_1111))), shl6(cont_payload(bytes_get(0xe2_82_ac, 0b0000_0001)))), cont_payload(bytes_get(0xe2_82_ac, 0b0000_0010))), 0b0000_0011));",
    );
    assert_three_way("decode_one(€)=Ok(Pr(8364,3))", &src, &expected);
}

/// `decode_one(0xf0_9f_98_80, 0b0000_0000)` → `Ok(Pr(128512, 4))` — 4-byte path (😀 = U+1F600).
/// Lead 0xF0 ∈ 0xF0..0xF7 → 4-byte; conts 0x9F, 0x98, 0x80 valid. cp = (0xF0 & 0x07)<<18 | (0x9F &
/// 0x3F)<<12 | (0x98 & 0x3F)<<6 | (0x80 & 0x3F) = 0 | 31<<12 | 24<<6 | 0 = 126976+1536 = 128512 =
/// U+1F600. Hand-computed + Python-verified. Declared/Empirical.
#[test]
fn decode_one_four_byte_emoji() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xf0_9f_98_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(or(or(shl18(widen8(and(bytes_get(0xf0_9f_98_80, 0b0000_0000), 0b0000_0111))), shl12(cont_payload(bytes_get(0xf0_9f_98_80, 0b0000_0001)))), shl6(cont_payload(bytes_get(0xf0_9f_98_80, 0b0000_0010)))), cont_payload(bytes_get(0xf0_9f_98_80, 0b0000_0011))), 0b0000_0100));",
    );
    assert_three_way("decode_one(😀)=Ok(Pr(128512,4))", &src, &expected);
}

// ── decode_one — never-silent malformations (G2) on all three lead paths ────────────────────────────
//
// Each malformed input produces `Err(Invalid(byte))` carrying the offending byte — never a U+FFFD
// substitution, never a silent truncation/wrap. The reference reuses `bytes_get` so the offending byte
// shares `Derived` provenance.

/// `decode_one(0x80_41, 0b0000_0000)` → `Err(Invalid(0x80))` — a bare continuation byte cannot lead.
/// 0x80 ∈ 0x80..0xBF (lt 0xC0 but not lt 0x80) → the "continuation as lead" Err arm; never-silent.
#[test]
fn decode_one_err_bare_continuation_lead() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0x80_41, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Invalid(bytes_get(0x80_41, 0b0000_0000)));",
    );
    assert_three_way("decode_one(0x80 lead)=Err(Invalid(0x80))", &src, &expected);
}

/// `decode_one(0xc3_41, 0b0000_0000)` → `Err(Invalid(0x41))` — 2-byte lead but the continuation slot
/// holds 0x41 ('A'), which is NOT a continuation byte (0x41 < 0x80). Never-silent: the offending
/// continuation byte is reported, never a half-decoded codepoint.
#[test]
fn decode_one_err_bad_continuation_two_byte() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xc3_41, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Invalid(bytes_get(0xc3_41, 0b0000_0001)));",
    );
    assert_three_way("decode_one(0xC3 0x41)=Err(Invalid(0x41))", &src, &expected);
}

/// `decode_one(0xc3, 0b0000_0000)` → `Err(Invalid(0xC3))` — a truncated 2-byte sequence (the
/// continuation byte at index 1 is past the 1-byte input). `byte_at(b, 1)` is `None` → the lead byte
/// is reported. Never-silent: a missing continuation is an explicit Err, never a kernel OOB refusal.
#[test]
fn decode_one_err_truncated_two_byte() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xc3, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Invalid(bytes_get(0xc3, 0b0000_0000)));",
    );
    assert_three_way(
        "decode_one(0xC3 truncated)=Err(Invalid(0xC3))",
        &src,
        &expected,
    );
}

/// `decode_one(0xe2_82, 0b0000_0000)` → `Err(Invalid(0x82))` — a 3-byte lead truncated after the first
/// continuation (the second continuation at index 2 is past the 2-byte input). `byte_at(b, 2)` is
/// `None` → the last-seen continuation 0x82 is reported. Never-silent on the 3-byte truncation path.
#[test]
fn decode_one_err_truncated_three_byte() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xe2_82, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Invalid(bytes_get(0xe2_82, 0b0000_0001)));",
    );
    assert_three_way(
        "decode_one(0xE2 0x82 truncated)=Err(Invalid(0x82))",
        &src,
        &expected,
    );
}

/// `decode_one(0xf8_80_80_80, 0b0000_0000)` → `Err(Invalid(0xF8))` — 0xF8 is not a valid UTF-8 lead
/// (no 5+-byte form exists). The lead is in none of the 1/2/3/4-byte ranges → the final Err arm;
/// never-silent (G2): the invalid lead is reported, never decoded as a phantom 5-byte sequence.
#[test]
fn decode_one_err_invalid_high_lead() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xf8_80_80_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Invalid(bytes_get(0xf8_80_80_80, 0b0000_0000)));",
    );
    assert_three_way("decode_one(0xF8 lead)=Err(Invalid(0xF8))", &src, &expected);
}

/// `decode_one(0x41, 0b0000_0001)` → `Err(Invalid(0b0000_0000))` — the start index `1` is past the end
/// of the single-byte input (`byte_at(b, 1)` is `None`), so `decode_one`'s `None` arm reports the
/// synthetic offending byte `0` (there is no byte to report). Never-silent (G2): an out-of-range start
/// is an explicit `Err`, never a kernel OOB refusal leaking through. The reference uses the **literal**
/// `Invalid(0b0000_0000)` to match the `Root` provenance of `decode_one`'s own literal `0` (unlike the
/// `bytes_get`-derived bytes in the other Err cases).
#[test]
fn decode_one_err_oob_start_index() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0x41, 0b0000_0001);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Invalid(0b0000_0000));",
    );
    assert_three_way(
        "decode_one(0x41, idx 1 oob)=Err(Invalid(0))",
        &src,
        &expected,
    );
}

// ── decode_one — UTF-8 validity rejection (M-717 validity layer) ────────────────────────────────────
//
// The structural decode is well-formed but a sequence may still encode a NON-canonical or NON-scalar
// codepoint: an overlong form (fewer bytes would do), a surrogate (U+D800–DFFF), or a value above the
// Unicode ceiling (U+10FFFF). Each is a never-silent `Err(Overlong/Surrogate/TooLarge(lead))` — the
// lead byte is `Derived` (via `bytes_get`), so the reference reuses `bytes_get` to match provenance.
// Hand-computed codepoints cross-checked against the UTF-8 well-formedness rules (RFC-3629).

/// `decode_one(0xc0_80, 0)` → `Err(Overlong(0xC0))` — the classic overlong NUL: cp = (0xC0 & 0x1F)<<6 |
/// (0x80 & 0x3F) = 0, which is < 0x80 (1 byte would suffice). Never-silent (G2): overlong is rejected,
/// never decoded as U+0000.
#[test]
fn decode_one_rejects_overlong_two_byte() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xc0_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Overlong(bytes_get(0xc0_80, 0b0000_0000)));",
    );
    assert_three_way("decode_one(0xC0 0x80)=Err(Overlong)", &src, &expected);
}

/// `decode_one(0xe0_80_80, 0)` → `Err(Overlong(0xE0))` — overlong 3-byte: cp = 0 < 0x800.
#[test]
fn decode_one_rejects_overlong_three_byte() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xe0_80_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Overlong(bytes_get(0xe0_80_80, 0b0000_0000)));",
    );
    assert_three_way("decode_one(0xE0 0x80 0x80)=Err(Overlong)", &src, &expected);
}

/// `decode_one(0xed_a0_80, 0)` → `Err(Surrogate(0xED))` — U+D800 (the first UTF-16 high surrogate):
/// cp = (0xED & 0x0F)<<12 | (0xA0 & 0x3F)<<6 | (0x80 & 0x3F) = 0xD000 | 0x800 = 0xD800, in the surrogate
/// gap 0xD800..0xDFFF. Never-silent: a surrogate is not a Unicode scalar value, so it is rejected.
#[test]
fn decode_one_rejects_surrogate() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xed_a0_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Surrogate(bytes_get(0xed_a0_80, 0b0000_0000)));",
    );
    assert_three_way(
        "decode_one(U+D800 surrogate)=Err(Surrogate)",
        &src,
        &expected,
    );
}

/// `decode_one(0xf4_90_80_80, 0)` → `Err(TooLarge(0xF4))` — U+110000, one above the ceiling: cp =
/// (0xF4 & 0x07)<<18 | (0x90 & 0x3F)<<12 | … = 0x100000 | 0x10000 = 0x110000 > 0x10FFFF. Never-silent.
#[test]
fn decode_one_rejects_above_max() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xf4_90_80_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(TooLarge(bytes_get(0xf4_90_80_80, 0b0000_0000)));",
    );
    assert_three_way("decode_one(U+110000)=Err(TooLarge)", &src, &expected);
}

// ── decode_one — validity BOUNDARIES are accepted (not over-rejected) ────────────────────────────────

/// `decode_one(0xc2_80, 0)` → `Ok(Pr(0x80, 2))` — U+0080, the SMALLEST canonical 2-byte codepoint, is
/// accepted (cp = 0x80 is NOT < 0x80). Proves the overlong gate does not over-reject the boundary.
#[test]
fn decode_one_accepts_min_two_byte_boundary() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xc2_80, 0b0000_0000);";
    let src = program(driver);
    // Reference recomputes via the decode_two assembly (matching Derived provenance).
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(shl6(widen8(and(bytes_get(0xc2_80, 0b0000_0000), 0b0001_1111))), cont_payload(bytes_get(0xc2_80, 0b0000_0001))), 0b0000_0010));",
    );
    assert_three_way("decode_one(U+0080 min 2-byte)=Ok", &src, &expected);
}

/// `decode_one(0xf4_8f_bf_bf, 0)` → `Ok(Pr(0x10FFFF, 4))` — U+10FFFF, the MAXIMUM Unicode scalar value,
/// is accepted (cp = 0x10FFFF is NOT > 0x10FFFF). Proves the ceiling gate does not over-reject the
/// boundary. cp = 0x100000 | 0xF000 | 0xFC0 | 0x3F = 0x10FFFF.
#[test]
fn decode_one_accepts_max_codepoint_boundary() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xf4_8f_bf_bf, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(or(or(shl18(widen8(and(bytes_get(0xf4_8f_bf_bf, 0b0000_0000), 0b0000_0111))), shl12(cont_payload(bytes_get(0xf4_8f_bf_bf, 0b0000_0001)))), shl6(cont_payload(bytes_get(0xf4_8f_bf_bf, 0b0000_0010)))), cont_payload(bytes_get(0xf4_8f_bf_bf, 0b0000_0011))), 0b0000_0100));",
    );
    assert_three_way("decode_one(U+10FFFF max)=Ok", &src, &expected);
}

// ── decode_one — validity boundary EDGES (review finding #4: surrogate upper edge + per-length mins) ─
//
// Complements the validity tests above by pinning the exact edges of each gate, so an off-by-one in a
// threshold would be caught: the surrogate UPPER edge (U+DFFF rejects, U+E000 the first scalar after it
// accepts) and the 3-/4-byte minimum codepoints (U+0800, U+10000 accept — not over-rejected as overlong).

/// `decode_one(0xed_bf_bf, 0)` → `Err(Surrogate(0xED))` — U+DFFF, the LAST surrogate: cp = (0xED&0xF)<<12
/// | (0xBF&0x3F)<<6 | (0xBF&0x3F) = 0xD000 | 0xFC0 | 0x3F = 0xDFFF, still in 0xD800..0xDFFF. Reject.
#[test]
fn decode_one_rejects_surrogate_upper_edge() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xed_bf_bf, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Err(Surrogate(bytes_get(0xed_bf_bf, 0b0000_0000)));",
    );
    assert_three_way("decode_one(U+DFFF surrogate edge)=Err", &src, &expected);
}

/// `decode_one(0xee_80_80, 0)` → `Ok(Pr(0xE000, 3))` — U+E000, the FIRST scalar above the surrogate
/// gap, is accepted (cp = 0xE000 is NOT < 0xE000... and NOT < 0xD800 so passes the surrogate gate). The
/// edge just past the surrogate range must NOT be rejected.
#[test]
fn decode_one_accepts_first_scalar_after_surrogates() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xee_80_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(or(shl12(widen8(and(bytes_get(0xee_80_80, 0b0000_0000), 0b0000_1111))), shl6(cont_payload(bytes_get(0xee_80_80, 0b0000_0001)))), cont_payload(bytes_get(0xee_80_80, 0b0000_0010))), 0b0000_0011));",
    );
    assert_three_way(
        "decode_one(U+E000 first post-surrogate)=Ok",
        &src,
        &expected,
    );
}

/// `decode_one(0xe0_a0_80, 0)` → `Ok(Pr(0x800, 3))` — U+0800, the SMALLEST canonical 3-byte codepoint,
/// accepted (cp = 0x800 is NOT < 0x800). Proves the 3-byte overlong gate does not over-reject its min.
#[test]
fn decode_one_accepts_min_three_byte_boundary() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xe0_a0_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(or(shl12(widen8(and(bytes_get(0xe0_a0_80, 0b0000_0000), 0b0000_1111))), shl6(cont_payload(bytes_get(0xe0_a0_80, 0b0000_0001)))), cont_payload(bytes_get(0xe0_a0_80, 0b0000_0010))), 0b0000_0011));",
    );
    assert_three_way("decode_one(U+0800 min 3-byte)=Ok", &src, &expected);
}

/// `decode_one(0xf0_90_80_80, 0)` → `Ok(Pr(0x10000, 4))` — U+10000, the SMALLEST canonical 4-byte
/// codepoint, accepted (cp = 0x10000 is NOT < 0x10000). Proves the 4-byte overlong gate does not
/// over-reject its min.
#[test]
fn decode_one_accepts_min_four_byte_boundary() {
    let driver =
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = decode_one(0xf0_90_80_80, 0b0000_0000);";
    let src = program(driver);
    let expected = program(
        "fn main() => Result[Pair[Binary{32}, Binary{8}], Utf8Error] = Ok(Pr(or(or(or(shl18(widen8(and(bytes_get(0xf0_90_80_80, 0b0000_0000), 0b0000_0111))), shl12(cont_payload(bytes_get(0xf0_90_80_80, 0b0000_0001)))), shl6(cont_payload(bytes_get(0xf0_90_80_80, 0b0000_0010)))), cont_payload(bytes_get(0xf0_90_80_80, 0b0000_0011))), 0b0000_0100));",
    );
    assert_three_way("decode_one(U+10000 min 4-byte)=Ok", &src, &expected);
}

// ── is_cont_byte (M-719 gap-closure: isolated per-op differential) ──────────────────────────────────
//
// `is_cont_byte` is exercised internally by `decode_two/three/four`, but the RFC-0031 §5 D5 bar is
// per-EXPORTED-op, so it gets its own isolated three-way tests over its three regions: below 0x80
// (not a continuation), 0x80..=0xBF (continuation), and 0xC0.. (not a continuation). Total over
// Binary{8} (Exact); three-way agreement Empirical.

/// `is_cont_byte(0x41)` → `False` — an ASCII byte (< 0x80) is not a continuation byte. Exact.
#[test]
fn is_cont_byte_false_below_0x80() {
    let driver = "fn main() => Bool = is_cont_byte(0b0100_0001);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_cont_byte(0x41 → False)", &src, expected);
}

/// `is_cont_byte(0x80)` → `True` — the first continuation byte (0x80 = 0b1000_0000). Exact.
#[test]
fn is_cont_byte_true_at_0x80() {
    let driver = "fn main() => Bool = is_cont_byte(0b1000_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_cont_byte(0x80 → True)", &src, expected);
}

/// `is_cont_byte(0xBF)` → `True` — the last continuation byte (0xBF = 0b1011_1111). Exact.
#[test]
fn is_cont_byte_true_at_0xbf() {
    let driver = "fn main() => Bool = is_cont_byte(0b1011_1111);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("is_cont_byte(0xBF → True)", &src, expected);
}

/// `is_cont_byte(0xC0)` → `False` — 0xC0 = 0b1100_0000 is a lead byte, not a continuation (the high
/// two bits are 0b11, not 0b10). The upper boundary of the continuation region. Exact.
#[test]
fn is_cont_byte_false_at_0xc0() {
    let driver = "fn main() => Bool = is_cont_byte(0b1100_0000);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("is_cont_byte(0xC0 → False)", &src, expected);
}

// ── concat / slice — the `.myc`-SURFACE wrappers over the kernel Bytes prims (M-719 gap-closure) ──────
//
// `std_bytes_slice.rs` covers the kernel `bytes_concat`/`bytes_slice` prims directly and the `.myc`
// `slice_opt` wrapper, but NOT the thin `.myc`-surface `concat`/`slice` wrappers themselves. The
// RFC-0031 §5 D5 bar is per-EXPORTED-op, so these wrappers get their own three-way differential here.
// The reference reuses the SAME kernel prim (`bytes_concat`/`bytes_slice`) so the result shares the
// `Derived` provenance the computed value carries (a `0x…` literal would be `Root` and not compare equal —
// Meta carries provenance). Both wrappers are `Exact` on their domain; three-way agreement Empirical.

/// `concat(0xDEAD, 0xBEEF)` → `0xDEADBEEF` — the `.myc` `concat` wrapper delegates to the Exact
/// `bytes_concat` kernel prim. Reference reuses `bytes_concat` (Derived provenance). Exact.
#[test]
fn concat_surface_wrapper_joins_bytes() {
    let driver = "fn main() => Bytes = concat(0xDEAD, 0xBEEF);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bytes = bytes_concat(0xDEAD, 0xBEEF);";
    assert_three_way("concat(0xDEAD,0xBEEF)=0xDEADBEEF", &src, expected);
}

/// `slice(0xDEADBEEF, 1, 3)` → `0xADBE` — the `.myc` `slice` wrapper delegates to the never-silent
/// `bytes_slice` kernel prim on the in-range half-open `[1, 3)`. Reference reuses `bytes_slice`
/// (Derived provenance). Exact on the in-range domain (start <= end <= len). Bounds are Binary{32}.
#[test]
fn slice_surface_wrapper_in_range() {
    let driver = "fn main() => Bytes = slice(0xDEADBEEF, 0b0000_0000_0000_0000_0000_0000_0000_0001, 0b0000_0000_0000_0000_0000_0000_0000_0011);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, 0b0000_0000_0000_0000_0000_0000_0000_0001, 0b0000_0000_0000_0000_0000_0000_0000_0011);";
    assert_three_way("slice(0xDEADBEEF,[1,3))=0xADBE", &src, expected);
}

/// `slice(0xDEADBEEF, 0, 4)` → the whole `Bytes` (the full half-open `[0, len)`) — an identity slice.
/// Exact; reference reuses `bytes_slice` (Derived provenance).
#[test]
fn slice_surface_wrapper_full_range_is_identity() {
    let driver = "fn main() => Bytes = slice(0xDEADBEEF, 0b0000_0000_0000_0000_0000_0000_0000_0000, 0b0000_0000_0000_0000_0000_0000_0000_0100);";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, 0b0000_0000_0000_0000_0000_0000_0000_0000, 0b0000_0000_0000_0000_0000_0000_0000_0100);";
    assert_three_way("slice(0xDEADBEEF,[0,4))=identity", &src, expected);
}
