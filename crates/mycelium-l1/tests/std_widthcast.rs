//! `width_cast` conformance (DN-41 / M-798) — the differential proof that the never-silent `Binary`
//! width-cast prim genuinely re-widths an unsigned value and refuses a lossy narrow.
//!
//! `width_cast(value: Binary{N}, into: Binary{M}) -> Binary{M}` re-widths a sign-free `Binary`
//! (ADR-028) value (MSB-first), with the target width `M` carried by the **second operand's width**
//! (a *width witness* — its bits are ignored). It is the enabler wave-n1 flagged as missing for
//! E13-1 M-717 (multi-byte UTF-8 / `byte_at`): widening a `Binary{8}` byte index to `Binary{32}` so
//! it can be `lt`-compared against a `bytes_len` result.
//!
//! Each case lands a **three-way differential** (L1-eval ≡ elaborate→L0-interp ≡ AOT) over the same
//! trusted prim registry, mirroring `enablement.rs`/`differential.rs`.
//!
//! # Honesty tags
//! - **`Exact`** — a widen (`M > N`) is zero-extension and an in-range narrow (`M < N`) drops only
//!   zero high bits; both equal the unsigned reference value exactly.
//! - **`Declared`/never-silent** — the *narrowing-overflow contract*: a value that does not fit `M`
//!   bits is an explicit refusal, asserted (not a proven theorem) and exhibited by the refusal test.
//! - **`Empirical`** — the three-way agreement is established by trial on the programs below.
//!
//! # Never-silent (G2/VR-5)
//! A narrowing whose dropped high bits are not all zero is an **explicit refusal on every path**
//! (L1-eval, L0-interp, AOT) — never a silent truncation to the low `M` bits.

use mycelium_core::{Payload, Repr};
use mycelium_interp::{EvalError, Interpreter, PrimRegistry};
use mycelium_l1::{check_nodule, elaborate, parse, Evaluator, L1Error};

/// Run the three-way differential on `src` (L1-eval ≡ elaborate→L0-interp ≡ AOT) and assert all
/// three paths agree on the observable (`repr + payload`) AND equal the `expected` reference value.
/// (A faithful copy of `enablement.rs::assert_three_way`, kept local so this enabler's conformance is
/// self-contained.)
fn assert_three_way(label: &str, src: &str, expected_repr: &Repr, expected_payload: &Payload) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    // Path 1: the L1 fuel-guarded evaluator.
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();

    // Path 2: elaborate to L0, run on the reference interpreter.
    let node =
        elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: must be in the fragment: {e}"));
    let l0 = interp
        .eval(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));

    // Path 3: the same L0 term through the AOT path.
    let aot = mycelium_mlir::run(&node, &prims, &engine)
        .unwrap_or_else(|e| panic!("{label}: AOT failed: {e}"));

    for (path, v) in [("L1-eval", &l1), ("L0-interp", &l0), ("AOT", &aot)] {
        assert_eq!(v.repr(), expected_repr, "{label}: {path} repr mismatch");
        assert_eq!(
            v.payload(),
            expected_payload,
            "{label}: {path} payload mismatch"
        );
    }
    assert_eq!(
        (l1.repr(), l1.payload()),
        (l0.repr(), l0.payload()),
        "{label}: L1-eval vs L0-interp diverged"
    );
    assert_eq!(
        (l0.repr(), l0.payload()),
        (aot.repr(), aot.payload()),
        "{label}: L0-interp vs AOT diverged"
    );
}

/// The `Binary{w}` MSB-first encoding of the unsigned value `n`.
fn bin(w: u32, n: u64) -> (Repr, Payload) {
    let bits: Vec<bool> = (0..w).rev().map(|k| (n >> k) & 1 == 1).collect();
    (Repr::Binary { width: w }, Payload::Bits(bits))
}

/// A `Binary{1}` truth payload (the realized `Bool` of the comparison prims — RFC-0032 D1).
fn b1(truth: bool) -> (Repr, Payload) {
    (Repr::Binary { width: 1 }, Payload::Bits(vec![truth]))
}

/// A literal for `Binary{32}(n)` as the explicit 32 bits (the surface has no decimal-with-width
/// shorthand without a `default paradigm`, so write the witness/operand bits directly).
fn lit32(n: u32) -> String {
    let s: String = (0..32)
        .rev()
        .map(|k| if (n >> k) & 1 == 1 { '1' } else { '0' })
        .collect();
    format!("0b{s}")
}

// ── Widen (M > N): zero-extension is Exact, value preserved ───────────────────────────────────────

/// `width_cast(0b1010_0101 : Binary{8}, witness : Binary{32}) -> Binary{32}` zero-extends `0xA5`
/// (165) to a `Binary{32}` of the same value on all three paths. The witness is a `Binary{32}` whose
/// *value* is irrelevant (here `0`) — only its width drives the cast.
#[test]
fn widen_8_to_32_zero_extends_exactly() {
    let (r, p) = bin(32, 0xA5);
    let src = format!(
        "nodule d;\nfn main() => Binary{{32}} = width_cast(0b1010_0101, {});",
        lit32(0)
    );
    assert_three_way("widen 8->32 (0xA5)", &src, &r, &p);
}

/// Widening the maximum `Binary{8}` value (`255`) to `Binary{32}` is still exactly `255` (no sign
/// bit, no overflow — `Binary` is sign-free, ADR-028).
#[test]
fn widen_8_to_32_preserves_max_byte() {
    let (r, p) = bin(32, 255);
    let src = format!(
        "nodule d;\nfn main() => Binary{{32}} = width_cast(0b1111_1111, {});",
        lit32(0)
    );
    assert_three_way("widen 8->32 (255)", &src, &r, &p);
}

// ── Identity (M == N) ─────────────────────────────────────────────────────────────────────────────

/// A same-width cast is the identity (the value is unchanged, the width unchanged).
#[test]
fn same_width_is_identity() {
    let (r, p) = bin(8, 0x3c);
    assert_three_way(
        "identity 8->8 (0x3c)",
        "nodule d;\nfn main() => Binary{8} = width_cast(0b0011_1100, 0b0000_0000);",
        &r,
        &p,
    );
}

// ── Narrow (M < N) that fits: Exact, value preserved ──────────────────────────────────────────────

/// `width_cast(Binary{32}(5), witness : Binary{8}) -> Binary{8}` narrows `5` (whose high 24 bits are
/// all zero) to `Binary{8}(5)` exactly — a fitting narrow is lossless on all three paths.
#[test]
fn narrow_32_to_8_fits_exactly() {
    let (r, p) = bin(8, 5);
    let src = format!(
        "nodule d;\nfn main() => Binary{{8}} = width_cast({}, 0b0000_0000);",
        lit32(5)
    );
    assert_three_way("narrow 32->8 (5 fits)", &src, &r, &p);
}

/// The boundary value `255` fits exactly in `Binary{8}` (the largest in-range narrow).
#[test]
fn narrow_32_to_8_fits_at_boundary() {
    let (r, p) = bin(8, 255);
    let src = format!(
        "nodule d;\nfn main() => Binary{{8}} = width_cast({}, 0b0000_0000);",
        lit32(255)
    );
    assert_three_way("narrow 32->8 (255 boundary)", &src, &r, &p);
}

// ── Narrow that overflows: never-silent refusal on ALL THREE paths (G2/VR-5) ──────────────────────

/// `width_cast(Binary{32}(256), into Binary{8})` does NOT fit (`256` needs bit 8, dropped by the
/// narrow) — an explicit refusal on **every** path, never a silent truncation to `0`. The program
/// type-checks: the value→M fit is a runtime contract (DN-41), exactly like `add_u` overflow.
#[test]
fn narrow_overflow_refuses_on_every_path() {
    let src = format!(
        "nodule d;\nfn main() => Binary{{8}} = width_cast({}, 0b0000_0000);",
        lit32(256)
    );
    // Check-first (the strengthening): the program **type-checks** — so the refusal below is a
    // genuine *runtime* contract (DN-41), not a static error caught at the wrong layer. The narrow
    // surfaces uniformly as `EvalError::Overflow` on all three paths (L1 wraps it in `L1Error::Kernel`
    // since `mycelium_l1`'s `KernelError` *is* `mycelium_interp::EvalError`; the AOT env-machine reuses
    // the same prim registry, so it too yields `EvalError::Overflow`).
    let env =
        check_nodule(&parse(&src).expect("parses")).expect("checks (fit is a runtime contract)");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        matches!(
            Evaluator::new(&env).call("main", vec![]),
            Err(L1Error::Kernel(EvalError::Overflow { .. }))
        ),
        "L1-eval must refuse the lossy narrow with Overflow (never a silent truncation to 0)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        matches!(interp.eval(&node), Err(EvalError::Overflow { .. })),
        "L0-interp must refuse the lossy narrow with Overflow"
    );
    assert!(
        matches!(
            mycelium_mlir::run(&node, &prims, &engine),
            Err(EvalError::Overflow { .. })
        ),
        "AOT must refuse the lossy narrow with Overflow"
    );
}

/// A high value (`0xFFFF_FFFF`) narrowed to `Binary{8}` also refuses on all three paths — the
/// dropped high bits are set, so it cannot fit (never a silent low-byte truncation to `0xFF`).
#[test]
fn narrow_overflow_high_value_refuses_on_every_path() {
    let src = format!(
        "nodule d;\nfn main() => Binary{{8}} = width_cast({}, 0b0000_0000);",
        lit32(0xFFFF_FFFF)
    );
    let env = check_nodule(&parse(&src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(Evaluator::new(&env).call("main", vec![]).is_err());
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(interp.eval(&node).is_err());
    assert!(mycelium_mlir::run(&node, &prims, &engine).is_err());
}

// ── The motivating composite (M-717): widen a byte index to compare against a length ──────────────

/// The M-717 use case made concrete: `lt(width_cast(idx8, len32), len32)` — widen a `Binary{8}` byte
/// index to `Binary{32}` (reusing the `Binary{32}` length as the very width witness it is compared
/// against), then `lt` it against the length. `idx = 3 < len = 16` ⇒ `0b1` on all three paths.
#[test]
fn motivating_composite_index_lt_length_true() {
    let (r, p) = b1(true);
    let src = format!(
        "nodule d;\nfn main() => Binary{{1}} = lt(width_cast(0b0000_0011, {len}), {len});",
        len = lit32(16)
    );
    assert_three_way("lt(width_cast(idx8,len32), len32) true", &src, &r, &p);
}

/// The false branch of the same composite: an index `20` is NOT `< 16` ⇒ `0b0` (the widened index is
/// compared at full `Binary{32}` width — never a width-mismatch refusal, which is exactly the gap
/// `width_cast` closes for M-717).
#[test]
fn motivating_composite_index_lt_length_false() {
    let (r, p) = b1(false);
    let src = format!(
        "nodule d;\nfn main() => Binary{{1}} = lt(width_cast(0b0001_0100, {len}), {len});",
        len = lit32(16)
    );
    assert_three_way("lt(width_cast(idx8,len32), len32) false", &src, &r, &p);
}

// ── Never-silent type refusals (G2): a non-Binary operand is a static error ───────────────────────

/// `width_cast` over a non-`Binary` value operand is a **static** type refusal (never a silent
/// coercion). `<00+->` is `Ternary{4}`.
#[test]
fn width_cast_non_binary_value_refuses_statically() {
    let src = format!(
        "nodule d;\nfn main() => Binary{{32}} = width_cast(0t00+-, {});",
        lit32(0)
    );
    assert!(
        check_nodule(&parse(&src).expect("parses")).is_err(),
        "a Ternary value operand to width_cast must be a static type error (DN-41)"
    );
}

/// `width_cast` with a non-`Binary` width witness is a static refusal — the witness must be a
/// `Binary{M}` (it supplies the target width).
#[test]
fn width_cast_non_binary_witness_refuses_statically() {
    let src = "nodule d;\nfn main() => Ternary{4} = width_cast(0b0000_0011, 0t00+-);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a Ternary width witness to width_cast must be a static type error (DN-41)"
    );
}

/// Wrong arity is an explicit refusal (one operand is missing the width witness).
#[test]
fn width_cast_wrong_arity_refuses() {
    let src = "nodule d;\nfn main() => Binary{8} = width_cast(0b0000_0011);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "width_cast requires two operands (value + width witness); one is a static error"
    );
}
