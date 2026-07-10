//! Width-generic free functions conformance (DN-42 Option A / M-753 step-d) — the differential
//! proof that `fn f<N>(x: Binary{N}) -> Binary{N}` and friends genuinely specialize to the
//! concrete width at every call site, with never-silent refusals on mismatch.
//!
//! # What is tested
//! - **Three-way differential** (`L1-eval ≡ elaborate→L0-interp ≡ AOT`) on width-generic identity
//!   and delegation wrappers at `Binary{8}`, `Binary{16}`, `Ternary{3}`, `Ternary{6}`.
//! - **Width mismatch refuses** — calling a width-generic fn where the inferred width conflicts
//!   with the expected return type is an explicit error (never a silent coercion — S1/VR-5/G2).
//! - **Undetermined width param refuses** — a width param not inferable from value params is an
//!   explicit check error (never a guessed default — DN-42 §4 / VR-5).
//!
//! # Honesty tags
//! - **`Exact`** — identity and delegation are exact (no arithmetic rounding or swap).
//! - **`Empirical`** — three-way agreement is established by trial on the programs below.
//! - **`Declared`** — the refusal contracts are asserted by test; no independent proof.

use mycelium_core::{Payload, Repr, Trit};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::{check_nodule, elaborate, parse, Evaluator};

/// Run the three-way differential on `src` (L1-eval ≡ elaborate→L0-interp ≡ AOT) and assert
/// all three paths agree on the observable and equal the `expected` reference value.
fn assert_three_way(label: &str, src: &str, expected_repr: &Repr, expected_payload: &Payload) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    // Path 1: the L1 fuel-guarded evaluator (direct on the checked env, before monomorphization).
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();

    // Path 2: elaborate to L0 (monomorphizes first), run on the reference interpreter.
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

/// A `Binary{w}` MSB-first unsigned encoding of `n`.
fn bin(w: u32, n: u64) -> (Repr, Payload) {
    let bits: Vec<bool> = (0..w).rev().map(|k| (n >> k) & 1 == 1).collect();
    (Repr::Binary { width: w }, Payload::Bits(bits))
}

/// A `Ternary{w}` balanced-ternary encoding of signed `v` (MSB-first, digit in `{Neg,Zero,Pos}`).
fn trit(w: u32, v: i64) -> (Repr, Payload) {
    // Standard balanced-ternary digit extraction.
    let mut rem = v;
    let mut ds: Vec<i8> = Vec::with_capacity(w as usize);
    for _ in 0..w {
        let d = ((rem % 3) + 3) % 3; // in {0, 1, 2}
        let d: i8 = if d <= 1 { d as i8 } else { d as i8 - 3 }; // map 2 to -1
        ds.push(d);
        rem = (rem - d as i64) / 3;
    }
    ds.reverse(); // MSB first
    let trits: Vec<Trit> = ds
        .into_iter()
        .map(|d| match d {
            -1 => Trit::Neg,
            0 => Trit::Zero,
            1 => Trit::Pos,
            _ => unreachable!(),
        })
        .collect();
    (Repr::Ternary { trits: w }, Payload::Trits(trits))
}

// ── Width-generic identity function ─────────────────────────────────────────────────────────────

/// `fn id_bits<N>(x: Binary{N}) -> Binary{N} = x` — roundtrip at Binary{8}.
/// `Exact`: identity produces the input unchanged.
#[test]
fn width_generic_identity_binary_8() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn main() => Binary{8} = id_bits(0b1010_0101);";
    let (r, p) = bin(8, 0b1010_0101);
    assert_three_way("id_bits<8>", src, &r, &p);
}

/// `fn id_bits<N>(x: Binary{N}) -> Binary{N} = x` — roundtrip at Binary{16}.
/// `Exact`: identity produces the input unchanged.
#[test]
fn width_generic_identity_binary_16() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn main() => Binary{16} = id_bits(0b1100_1001_0111_1110);";
    let (r, p) = bin(16, 0b1100_1001_0111_1110);
    assert_three_way("id_bits<16>", src, &r, &p);
}

/// `fn id_trits<M>(x: Ternary{M}) -> Ternary{M} = x` — roundtrip at Ternary{3}.
/// `<0+->` = `[Zero, Pos, Neg]` = 0*9 + 1*3 + (-1)*1 = 2 in balanced ternary.
/// `Exact`: identity produces the input unchanged.
#[test]
fn width_generic_identity_ternary_3() {
    let src = "nodule d;\nfn id_trits{M}(x: Ternary{M}) => Ternary{M} = x;\nfn main() => Ternary{3} = id_trits(0t0+-);";
    let (r, p) = trit(3, 2); // <0+-> = 0*9 + 1*3 + (-1)*1 = 2
    assert_three_way("id_trits<3>", src, &r, &p);
}

/// `fn id_trits<M>(x: Ternary{M}) -> Ternary{M} = x` — roundtrip at Ternary{6}.
/// `Exact`: identity produces the input unchanged.
#[test]
fn width_generic_identity_ternary_6() {
    // <00+0+-> (6 trits) = 0*243 + 0*81 + 1*27 + 0*9 + 1*3 + (-1)*1 = 29
    let src = "nodule d;\nfn id_trits{M}(x: Ternary{M}) => Ternary{M} = x;\nfn main() => Ternary{6} = id_trits(0t00+0+-);";
    let (r, p) = trit(6, 29);
    assert_three_way("id_trits<6>", src, &r, &p);
}

// ── Width-generic delegation (wraps a prim) ─────────────────────────────────────────────────────

/// `fn add_n<N>(a: Binary{N}, b: Binary{N}) -> Binary{N} = add_u(a, b)` — at Binary{8}.
/// `Empirical`: three-way agreement — each path agrees with the reference 3+5=8.
#[test]
fn width_generic_add_binary_8() {
    let src = "nodule d;\nfn add_n{N}(a: Binary{N}, b: Binary{N}) => Binary{N} = add_u(a, b);\nfn main() => Binary{8} = add_n(0b0000_0011, 0b0000_0101);";
    let (r, p) = bin(8, 3 + 5); // 3 + 5 = 8
    assert_three_way("add_n<8>", src, &r, &p);
}

/// `fn add_n<N>(a: Binary{N}, b: Binary{N}) -> Binary{N} = add_u(a, b)` — at Binary{16}.
/// `Empirical`: three-way agreement — each path agrees with the reference 1+2=3.
#[test]
fn width_generic_add_binary_16() {
    let src = "nodule d;\nfn add_n{N}(a: Binary{N}, b: Binary{N}) => Binary{N} = add_u(a, b);\nfn main() => Binary{16} = add_n(0b0000_0000_0000_0001, 0b0000_0000_0000_0010);";
    let (r, p) = bin(16, 1 + 2); // 1 + 2 = 3
    assert_three_way("add_n<16>", src, &r, &p);
}

/// `fn add_trits<M>(a: Ternary{M}, b: Ternary{M}) -> Ternary{M} = add(a, b)` — at Ternary{3}.
/// `<00+>` = +1, `<00->` = -1; sum = 0 = `<000>`. `Empirical`: three-way agreement.
#[test]
fn width_generic_add_ternary_3() {
    let src = "nodule d;\nfn add_trits{M}(a: Ternary{M}, b: Ternary{M}) => Ternary{M} = add(a, b);\nfn main() => Ternary{3} = add_trits(0t00+, 0t00-);";
    let (r, p) = trit(3, 0); // +1 + (-1) = 0
    assert_three_way("add_trits<3>", src, &r, &p);
}

// ── Same function called at two distinct widths — both specialisations present ───────────────────

/// `id_bits<N>` called at both Binary{8} and Binary{16} from the same nodule — the Binary{8}
/// path. Verifies two distinct monomorphisations are emitted and both produce correct results.
/// `Empirical`: three-way differentials agree.
#[test]
fn width_generic_two_widths_same_function_8() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn get8() => Binary{8} = id_bits(0b1111_0000);\nfn get16() => Binary{16} = id_bits(0b1111_0000_1111_0000);\nfn main() => Binary{8} = get8();";
    let (r, p) = bin(8, 0b1111_0000);
    assert_three_way("two_widths get8", src, &r, &p);
}

/// The Binary{16} path of the same two-width nodule.
#[test]
fn width_generic_two_widths_same_function_16() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn get8() => Binary{8} = id_bits(0b1111_0000);\nfn get16() => Binary{16} = id_bits(0b1111_0000_1111_0000);\nfn main() => Binary{16} = get16();";
    let (r, p) = bin(16, 0b1111_0000_1111_0000);
    assert_three_way("two_widths get16", src, &r, &p);
}

// ── Recursive + delegated width-generics (the `unify` width-var pass-through — M-718) ─────────────
//
// A width-generic fn calling ANOTHER width-generic fn — or ITSELF — with a still-ABSTRACT width is the
// case the M-718 `unify` pass-through enables: the callee's width var binds to the caller's (mirroring
// the type-var pass-through) and is resolved to a concrete `Width::Lit` at monomorphization. Before
// M-718 every such call was refused ("width does not determine the width"). These lock the capability
// that the self-hosted recursive `map_get`/`set_contains` and the `le→cmp` delegation depend on.

/// Width-generic DELEGATION: `wrap_id<N>` calls the width-generic `id_bits<N>` with its own abstract
/// `x: Binary{N}`. The callee's width is determined by the enclosing scope, resolved to 8 at mono.
/// `Empirical`: three-way agreement.
#[test]
fn width_generic_delegation_binary_8() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn wrap_id{N}(x: Binary{N}) => Binary{N} = id_bits(x);\nfn main() => Binary{8} = wrap_id(0b1010_0101);";
    let (r, p) = bin(8, 0b1010_0101);
    assert_three_way("wrap_id<8>→id_bits<8>", src, &r, &p);
}

/// Width-generic delegation at a second width (Binary{16}) — proves the delegated call specialises
/// per width, not once. `Empirical`.
#[test]
fn width_generic_delegation_binary_16() {
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn wrap_id{N}(x: Binary{N}) => Binary{N} = id_bits(x);\nfn main() => Binary{16} = wrap_id(0b1100_1001_0111_1110);";
    let (r, p) = bin(16, 0b1100_1001_0111_1110);
    assert_three_way("wrap_id<16>→id_bits<16>", src, &r, &p);
}

/// RECURSIVE width-generic fn: `rec_id<N>(x, n)` returns `x` after `n` self-calls, recursing with its
/// abstract `x: Binary{N}` (`n` is a concrete Binary{8} step counter). The recursive call runs at the
/// abstract width before monomorphizing to N=8. `Empirical`: three-way agreement on the fixed point.
#[test]
fn width_generic_recursive_binary_8() {
    let src = "nodule d;\nfn rec_id{N}(x: Binary{N}, n: Binary{8}) => Binary{N} = match eq(n, 0b0000_0000) { 0b1 => x, _ => rec_id(x, sub_u(n, 0b0000_0001)) };\nfn main() => Binary{8} = rec_id(0b0110_0110, 0b0000_0011);";
    let (r, p) = bin(8, 0b0110_0110);
    assert_three_way("rec_id<8>", src, &r, &p);
}

/// The same recursive width-generic fn at Binary{16} — the recursion is width-polymorphic. `Empirical`.
#[test]
fn width_generic_recursive_binary_16() {
    let src = "nodule d;\nfn rec_id{N}(x: Binary{N}, n: Binary{8}) => Binary{N} = match eq(n, 0b0000_0000) { 0b1 => x, _ => rec_id(x, sub_u(n, 0b0000_0001)) };\nfn main() => Binary{16} = rec_id(0b1111_0000_0000_1111, 0b0000_0010);";
    let (r, p) = bin(16, 0b1111_0000_0000_1111);
    assert_three_way("rec_id<16>", src, &r, &p);
}

// ── Never-silent refusals (Declared / G2 / VR-5) ────────────────────────────────────────────────

/// A call where the width param cannot be inferred from the value arguments is an explicit
/// refusal — never a guessed width (G2 / VR-5 / DN-42 §4). `Declared`.
///
/// `fn phantom_n<N>(x: Binary{8}) -> Binary{N} = x` has `N` only in the return type; the
/// call `phantom_n(0b0)` provides no argument that pins `N`, so the checker refuses.
#[test]
fn width_generic_undetermined_param_refuses() {
    let src = "nodule d;\nfn phantom_n{N}(x: Binary{8}) => Binary{N} = x;\nfn main() => Binary{8} = phantom_n(0b0000_0000);";
    let parsed = parse(src).expect("parses");
    let result = check_nodule(&parsed);
    assert!(
        result.is_err(),
        "expected check to fail: width param `N` is not in value params and cannot be inferred, \
         but check succeeded"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains('N')
            || err.contains("width")
            || err.contains("undetermined")
            || err.contains("phantom_n"),
        "error message should mention the problematic parameter or undetermined width: {err}"
    );
}

/// Calling a width-generic fn where the inferred width conflicts with the declared return type
/// is an explicit type error — not a silent widening coercion (DN-42 §4 / VR-5 / S1).
/// `Declared`: the checker refuses (never a silent swap or coercion).
#[test]
fn width_mismatch_in_generic_call_refuses() {
    // `id_bits<N>` infers N=8 from the argument `0b0000_0000`, but `main()` declares `Binary{16}`
    // as the return type. The inferred return `Binary{8}` ≠ `Binary{16}` — type error.
    let src = "nodule d;\nfn id_bits{N}(x: Binary{N}) => Binary{N} = x;\nfn main() => Binary{16} = id_bits(0b0000_0000);";
    let parsed = parse(src).expect("parses");
    let result = check_nodule(&parsed);
    assert!(
        result.is_err(),
        "expected check to fail: Binary{{8}} inferred but Binary{{16}} declared, but check succeeded"
    );
}

// ── Ternary path through the M-718 var-var unify arm (review finding #2 — coverage parity) ─────────
//
// The new `unify` width-var pass-through handles BOTH Binary and Ternary (`Ty::Ternary(Width::Var)`).
// The Binary path is covered by the delegation/recursion tests above; this exercises the Ternary arm
// directly via a Ternary{M}-generic fn delegating to another, so the comment's "mirrors for Ternary
// too" parity claim is Empirically covered, not merely Declared.

/// Width-generic Ternary DELEGATION: `wrap_trits<M>` calls `id_trits<M>` with its abstract
/// `x: Ternary{M}` — the callee's width is determined by the enclosing scope, resolved to 3 at mono.
/// `<0+->` = 0*9 + 1*3 + (-1)*1 = 2. `Empirical`: three-way agreement.
#[test]
fn width_generic_delegation_ternary_3() {
    let src = "nodule d;\nfn id_trits{M}(x: Ternary{M}) => Ternary{M} = x;\nfn wrap_trits{M}(x: Ternary{M}) => Ternary{M} = id_trits(x);\nfn main() => Ternary{3} = wrap_trits(0t0+-);";
    let (r, p) = trit(3, 2);
    assert_three_way("wrap_trits<3>→id_trits<3>", src, &r, &p);
}

// ── Cross-argument width-conflict refusals — both orders (review findings #1 and #3) ───────────────
//
// When one argument pins a width param to a CONCRETE width and another argument leaves it ABSTRACT,
// the two cannot be proven equal — a never-silent refusal (DN-42 §4 / VR-5 / S1), never a silent
// coercion. The two argument orders hit two distinct `unify` arms; both are exercised here, and both
// error messages must name the abstract width HONESTLY (never a phantom width `0` — the M-718 fix to
// the var-vs-Lit conflict formatter).

/// CONCRETE-then-ABSTRACT: `f(0b…8, x)` where `x: Binary{M}` is abstract. The first arg pins `N=8`,
/// the second hits the var-var conflict arm (the callee's `N` is already `Lit(8)` ≠ abstract `M`).
/// Refusal, never silent. The message names the abstract width `M`, not a phantom `0`.
#[test]
fn width_generic_concrete_then_abstract_refuses() {
    let src = "nodule d;\nfn f{N}(a: Binary{N}, b: Binary{N}) => Binary{N} = a;\nfn outer{M}(x: Binary{M}) => Binary{8} = f(0b0000_0000, x);\nfn main() => Binary{8} = outer(0b0000_0001);";
    let result = check_nodule(&parse(src).expect("parses"));
    let err = result
        .expect_err("expected a never-silent width-conflict refusal")
        .to_string();
    assert!(
        err.contains('M') && !err.contains(" 0 "),
        "conflict message must name the abstract width `M`, never a phantom `0`: {err}"
    );
}

/// ABSTRACT-then-CONCRETE: `f(x, 0b…8)` where `x: Binary{M}` is abstract. The first arg binds `N` to
/// the abstract carrier `Binary{Var(M)}` (var-var arm); the second hits the var-vs-Lit conflict arm
/// with that abstract carrier already bound — the path whose formatter used to print a phantom `0`.
/// After the M-718 fix it names `M`. Refusal, never silent.
#[test]
fn width_generic_abstract_then_concrete_refuses() {
    let src = "nodule d;\nfn f{N}(a: Binary{N}, b: Binary{N}) => Binary{N} = a;\nfn outer{M}(x: Binary{M}) => Binary{8} = f(x, 0b0000_0000);\nfn main() => Binary{8} = outer(0b0000_0001);";
    let result = check_nodule(&parse(src).expect("parses"));
    let err = result
        .expect_err("expected a never-silent width-conflict refusal")
        .to_string();
    assert!(
        err.contains('M') && !err.contains(" 0 "),
        "conflict message must name the abstract width `M`, never a phantom `0`: {err}"
    );
}
