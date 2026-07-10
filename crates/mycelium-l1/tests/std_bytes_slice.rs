//! `bytes_slice` / `bytes_concat` surface conformance (DN-43 / M-799) — the differential proof that
//! the already-landed never-silent byte slice/concat kernel prims (RFC-0032 D4 / M-750) are now
//! callable from the `.myc` surface and remain never-silent on an out-of-range / inverted slice.
//!
//! `bytes_slice(b: Bytes, start: Binary{W}, end: Binary{W}) -> Bytes` is the half-open sub-slice
//! `b[start, end)`; `bytes_concat(b1: Bytes, b2: Bytes) -> Bytes` is byte concatenation. This is the
//! **surface** mapping (the kernel prims already existed + were registered — see
//! `mycelium-interp/src/prims.rs::prim_bytes_slice`/`prim_bytes_concat`); DN-43 adds the
//! `prim_kernel_name` + `try_check_seq_bytes_prim` wiring. It closes **FLAG-text-3** and M-717's
//! slicing Definition-of-Done clause.
//!
//! Each case lands a **three-way differential** (L1-eval ≡ elaborate→L0-interp ≡ AOT) over the same
//! trusted prim registry, mirroring `std_widthcast.rs`/`enablement.rs`.
//!
//! # Honesty tags
//! - **`Exact`** — `bytes_concat` (total/lossless) and `bytes_slice` over the in-range domain
//!   (`start <= end <= len`) reproduce the reference byte sequence exactly.
//! - **`Declared`/never-silent** — `bytes_slice`'s *out-of-range / inverted-range contract*: an
//!   out-of-bounds or inverted `[start, end)` is an explicit refusal, asserted (not a proven theorem)
//!   and exhibited by the refusal test.
//! - **`Empirical`** — the three-way agreement is established by trial on the programs below.
//!
//! # Never-silent (G2/VR-5)
//! An out-of-range (`end > len`) or inverted (`start > end`) slice is an **explicit refusal on every
//! path** (L1-eval, L0-interp, AOT) — never a silent clamp to `[..len)` and never a silent truncation
//! to the empty slice.

use mycelium_cert::{check_core, CheckVerdict};
use mycelium_core::{GuaranteeStrength, Payload, Repr};
use mycelium_interp::{EvalError, Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, elaborate, monomorphize, parse, Evaluator, L1Error};

/// Run the three-way differential on `src` (L1-eval ≡ elaborate→L0-interp ≡ AOT) and assert all
/// three paths agree on the observable (`repr + payload`) AND equal the `expected` reference value.
/// (Mirrors `std_widthcast.rs::assert_three_way`, kept local so this surface's conformance is
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

/// A `Bytes` repr + payload reference value (the expected byte sequence).
fn bytes(b: &[u8]) -> (Repr, Payload) {
    (Repr::Bytes, Payload::Bytes(b.to_vec()))
}

/// A `Binary{32}` index literal, written as explicit 32 bits (the surface has no decimal-with-width
/// shorthand without a `default paradigm`). Mirrors `std_widthcast.rs::lit32`.
fn lit32(n: u32) -> String {
    let s: String = (0..32)
        .rev()
        .map(|k| if (n >> k) & 1 == 1 { '1' } else { '0' })
        .collect();
    format!("0b{s}")
}

// ── In-range slice: the value is preserved exactly ────────────────────────────────────────────────

/// `bytes_slice(0xDEADBEEF, 1, 3)` is the half-open sub-slice `[1, 3)` = `0xADBE` — preserved exactly
/// on all three paths (Exact over the in-range domain).
#[test]
fn slice_in_range_preserves_value() {
    let (r, p) = bytes(&[0xAD, 0xBE]);
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, {}, {});",
        lit32(1),
        lit32(3)
    );
    assert_three_way("slice [1,3) of 0xDEADBEEF", &src, &r, &p);
}

/// A full-span slice `[0, len)` is the identity (the whole byte string back), and an empty slice
/// `[2, 2)` is the empty `Bytes` — both in-range, both Exact.
#[test]
fn slice_full_span_is_identity() {
    let (r, p) = bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, {}, {});",
        lit32(0),
        lit32(4)
    );
    assert_three_way("slice [0,4) (identity)", &src, &r, &p);
}

#[test]
fn slice_empty_range_is_empty_bytes() {
    let (r, p) = bytes(&[]);
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, {}, {});",
        lit32(2),
        lit32(2)
    );
    assert_three_way("slice [2,2) (empty)", &src, &r, &p);
}

// ── concat of two `0x…` literals ──────────────────────────────────────────────────────────────────

/// `bytes_concat(0xDEAD, 0xBEEF)` = `0xDEADBEEF` on all three paths (Exact / total).
#[test]
fn concat_two_literals() {
    let (r, p) = bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let src = "nodule d;\nfn main() => Bytes = bytes_concat(0xDEAD, 0xBEEF);";
    assert_three_way("concat 0xDEAD ++ 0xBEEF", src, &r, &p);
}

/// `concat` with an empty operand is the identity (the other operand back) — total, Exact.
#[test]
fn concat_with_empty_is_identity() {
    let (r, p) = bytes(&[0xDE, 0xAD]);
    // `0x` cannot be empty (lexer requires >=1 byte), so exercise identity via concat(x, slice(x,len,len)).
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_concat(0xDEAD, bytes_slice(0xDEAD, {}, {}));",
        lit32(2),
        lit32(2)
    );
    assert_three_way("concat 0xDEAD ++ empty", &src, &r, &p);
}

// ── Out-of-range / inverted slice: never-silent refusal on ALL THREE paths (G2/VR-5) ──────────────

/// `bytes_slice(0xDEADBEEF, 2, 9)` — `end = 9 > len = 4` — is out of bounds: an explicit refusal on
/// **every** path (L1-eval, L0-interp, AOT), never a silent clamp to `[2, 4)`. The program
/// type-checks: the range fit is a runtime contract (DN-43), exactly like `add_u`/`width_cast`
/// overflow.
#[test]
fn slice_out_of_range_refuses_on_every_path() {
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, {}, {});",
        lit32(2),
        lit32(9)
    );
    // Check-first (the strengthening): the program **type-checks** — so the refusal below is a
    // genuine *runtime* contract (DN-43), not a static error caught at the wrong layer. The
    // out-of-range slice surfaces uniformly as `EvalError::PrimType` (the kernel `prim_bytes_slice`
    // guard `start > end || end > len`) on all three paths: L1 wraps it in `L1Error::Kernel`
    // (`mycelium_l1`'s `KernelError` *is* `mycelium_interp::EvalError`), and the AOT env-machine reuses
    // the same prim registry, so it too yields `EvalError::PrimType`.
    let env = check_nodule(&parse(&src).expect("parses"))
        .expect("checks (range fit is a runtime contract)");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        matches!(
            Evaluator::new(&env).call("main", vec![]),
            Err(L1Error::Kernel(EvalError::PrimType { .. }))
        ),
        "L1-eval must refuse the out-of-range slice (never a silent clamp to [2,4))"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        matches!(interp.eval(&node), Err(EvalError::PrimType { .. })),
        "L0-interp must refuse the out-of-range slice"
    );
    assert!(
        matches!(
            mycelium_mlir::run(&node, &prims, &engine),
            Err(EvalError::PrimType { .. })
        ),
        "AOT must refuse the out-of-range slice"
    );
}

/// `bytes_slice(0xDEADBEEF, 3, 1)` — `start = 3 > end = 1` — is inverted: an explicit refusal on
/// **every** path, never a silent empty-slice substitution.
#[test]
fn slice_inverted_range_refuses_on_every_path() {
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0xDEADBEEF, {}, {});",
        lit32(3),
        lit32(1)
    );
    // Check-first (the strengthening): the program **type-checks**, so this is a *runtime* refusal
    // (DN-43), not a static error. The inverted range (`start > end`) hits the same kernel guard and
    // surfaces as `EvalError::PrimType` on all three paths (L1 wraps it in `L1Error::Kernel`; the AOT
    // env-machine reuses the same prim registry).
    let env = check_nodule(&parse(&src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(matches!(
        Evaluator::new(&env).call("main", vec![]),
        Err(L1Error::Kernel(EvalError::PrimType { .. }))
    ));
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(matches!(
        interp.eval(&node),
        Err(EvalError::PrimType { .. })
    ));
    assert!(matches!(
        mycelium_mlir::run(&node, &prims, &engine),
        Err(EvalError::PrimType { .. })
    ));
}

// ── Never-silent static type refusals (G2): a non-`Bytes` operand is a static error ───────────────

/// `bytes_slice` over a non-`Bytes` receiver is a **static** type refusal (never a silent coercion).
/// `0b0000_0000` is `Binary{8}`.
#[test]
fn slice_non_bytes_receiver_refuses_statically() {
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0b0000_0000, {}, {});",
        lit32(0),
        lit32(1)
    );
    assert!(
        check_nodule(&parse(&src).expect("parses")).is_err(),
        "a Binary receiver to bytes_slice must be a static type error (DN-43/RFC-0032 D4)"
    );
}

/// `bytes_concat` over a non-`Bytes` operand is a static type refusal.
#[test]
fn concat_non_bytes_operand_refuses_statically() {
    let src = "nodule d;\nfn main() => Bytes = bytes_concat(0xDEAD, 0b0000_0000);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a Binary second operand to bytes_concat must be a static type error (DN-43/RFC-0032 D4)"
    );
}

/// Wrong arity is an explicit refusal (`bytes_slice` needs three operands; `bytes_concat` two).
#[test]
fn slice_wrong_arity_refuses() {
    let src = format!(
        "nodule d;\nfn main() => Bytes = bytes_slice(0xDEAD, {});",
        lit32(0)
    );
    assert!(
        check_nodule(&parse(&src).expect("parses")).is_err(),
        "bytes_slice requires three operands (bytes + start + end); two is a static error"
    );
}

#[test]
fn concat_wrong_arity_refuses() {
    let src = "nodule d;\nfn main() => Bytes = bytes_concat(0xDEAD);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "bytes_concat requires two operands; one is a static error"
    );
}

// ── slice_opt — the bounds-checked Option<Bytes> form (DN-43 / M-799) ─────────────────────────────
//
// `slice_opt(b, start, end) -> Option<Bytes>` lifts the kernel's never-silent out-of-range refusal
// into explicit `Some`/`None` data (the range-analog of `byte_at`). Because the result is a **data**
// value (`Option<Bytes>`), these tests use the full data-value three-way harness (mirroring
// `std_text.rs::assert_three_way`): they include the `text.myc` source verbatim (the single source of
// truth for `slice_opt`/`Option`/`byte_len`) and compare the monomorphized L1 / elaborated-L0 / AOT
// CoreValues against a reference program. The reference reuses the same `bytes_slice` call for the
// `Some` case so the wrapped value shares `Derived` provenance with the computed result (a literal
// would carry `Root` provenance and would not compare equal — Meta carries provenance).

/// The `std.text` nodule source, loaded at compile time — the single source of truth for `slice_opt`.
const TEXT_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/text.myc"
));

/// Build a full test program by appending a typed driver to the `text.myc` nodule source.
fn text_program(driver: &str) -> String {
    format!("{TEXT_SRC}\n{driver}")
}

/// Data-value three-way differential (L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT) over an
/// `Option<Bytes>`-returning `src`, asserting all three agree AND equal the `expected` reference
/// program's value. A faithful copy of `std_text.rs::assert_three_way`, kept local so this surface's
/// `slice_opt` conformance is self-contained.
fn assert_three_way_opt(label: &str, src: &str, expected_src: &str) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );

    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));

    let l1_core = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"))
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));

    let node = elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: elaborate failed: {e}"));
    let l0_core = interp
        .eval_core(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot_core = mycelium_mlir::run_core(
        &node,
        &PrimRegistry::with_builtins(),
        &mycelium_cert::BinaryTernarySwapEngine,
    )
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

/// `slice_opt(0xDEADBEEF, 1, 3)` → `Some(bytes_slice(…, 1, 3))` (= Some(0xADBE)) — in range
/// (`1 <= 3 <= 4`), so the bounds check passes and the sub-slice is wrapped in `Some`. The reference
/// reuses `bytes_slice` so the wrapped value shares `Derived` provenance. Declared/Empirical.
#[test]
fn slice_opt_in_range_is_some() {
    let driver = format!(
        "fn main() => Option[Bytes] = slice_opt(0xDEADBEEF, {}, {});",
        lit32(1),
        lit32(3)
    );
    let src = text_program(&driver);
    let expected = text_program(&format!(
        "fn main() => Option[Bytes] = Some(bytes_slice(0xDEADBEEF, {}, {}));",
        lit32(1),
        lit32(3)
    ));
    assert_three_way_opt("slice_opt [1,3) = Some(0xADBE)", &src, &expected);
}

/// `slice_opt(0xDEADBEEF, 3, 1)` → `None` — inverted range (`start = 3 > end = 1`): the never-silent
/// out-of-range refusal is lifted to an explicit `None`, never a fabricated empty slice (G2).
#[test]
fn slice_opt_inverted_range_is_none() {
    let driver = format!(
        "fn main() => Option[Bytes] = slice_opt(0xDEADBEEF, {}, {});",
        lit32(3),
        lit32(1)
    );
    let src = text_program(&driver);
    let expected = text_program("fn main() => Option[Bytes] = None;");
    assert_three_way_opt("slice_opt [3,1) inverted = None", &src, &expected);
}

/// `slice_opt(0xDEADBEEF, 2, 9)` → `None` — out of range (`end = 9 > len = 4`): the explicit `None`,
/// never a silent clamp to `[2, 4)` (G2).
#[test]
fn slice_opt_out_of_range_is_none() {
    let driver = format!(
        "fn main() => Option[Bytes] = slice_opt(0xDEADBEEF, {}, {});",
        lit32(2),
        lit32(9)
    );
    let src = text_program(&driver);
    let expected = text_program("fn main() => Option[Bytes] = None;");
    assert_three_way_opt("slice_opt [2,9) oob = None", &src, &expected);
}
