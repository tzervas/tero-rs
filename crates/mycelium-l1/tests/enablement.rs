//! Kernel self-hosting-enablement conformance (E19-1 / M-752) — the differential proof that the
//! RFC-0032 enablers genuinely unblock the E13-1 self-hosted-stdlib tiers.
//!
//! Each enabler lands with a **three-way differential** (L1-eval ≡ elaborate→L0-interp ≡ AOT) over
//! the same trusted prim registry, mirroring `differential.rs`. This file pins the *new* surface:
//!
//! - **M-747** — the reduce-to-`Bool` comparison prims `eq`/`lt` over `Binary{N}`/`Ternary{N}`
//!   (RFC-0032 D1). The kernel prim returns `Binary{1}` (`0b1` = true); the `.myc` `std.cmp` lift to
//!   the `Bool` ADT is demonstrated by the `bool`-bridge smoke port. **Unblocks** E13-1 M-718
//!   (width-typed `cmp`/`Eq`/`Ord`).
//! - **M-748** — never-silent fixed-width binary arithmetic `add_u`/`sub_u` + the surfaced
//!   `and`/`or` (RFC-0032 D2). **Unblocks** E13-1 M-718 (binary `math`).
//!
//! # Honesty tags
//! - **`Exact`** — every prim here is total/decidable over its in-range domain; each result equals
//!   its reference value exactly.
//! - **`Empirical`** — the three-way agreement is established by trial on the programs below.
//!
//! # Never-silent (G2/VR-5)
//! Overflow (`add_u`/`sub_u` out of `[0, 2^N)`) and paradigm/width mismatch are **explicit
//! refusals on every path**, never a silent wrap or a silent `false` — pinned by the refusal tests.
//!
//! # Scope boundary
//! - **M-749** (`Repr::Seq`) — the **`.myc` surface is now wired** (lexer/parser/checker/elaborator:
//!   the `Seq{T, N}` type, the `[e1, …]` list literal, and the `seq_get`/`seq_len` prims). The full
//!   **three-way** (`L1-eval ≡ elaborate→L0-interp ≡ AOT`) differential over the surface runs in the
//!   `seq_*_surface_*` tests below, alongside the original **prim-level** differential
//!   (`seq.get`/`seq.len` over directly-built L0 `Node`s: **L0-interp ≡ AOT**) and the never-silent
//!   out-of-bounds refusal on both paths. Both layers are real (no faked/upgraded basis — G2/VR-5).
//! - **M-750** (`Repr::Bytes`) — the **`.myc` surface is now wired** (the `Bytes` type, the `0x…` hex
//!   literal, and the `bytes_get`/`bytes_len` prims). The full **three-way** differential over the
//!   surface runs in the `bytes_*_surface_*` tests below, alongside the original **prim-level**
//!   differential (`bytes.get`/`bytes.len`/`bytes.slice`/`bytes.concat` over directly-built L0
//!   `Node`s: **L0-interp ≡ AOT**) and the never-silent out-of-range/inverted-range refusals. UTF-8
//!   decode is written in `.myc` over these byte prims (per RFC-0032 D4) and is not exercised here.
//! - **Never-silent surface rejects** (G2): a **heterogeneous** list literal and an **odd-hex** `0x…`
//!   literal are explicit refusals at check/parse time — pinned by the `*_rejects` tests below.
//! - **M-910/M-911** (kickoff `enb` Phase-I H1) — the **`.myc` surface is now wired** for a textual
//!   string literal `"…"` (lexer/parser/checker/elaborator): it lowers to the SAME `Repr::Bytes`
//!   value form as the `0x…` literal (KC-3 — no new L0 node), so it is a legal operand to the SAME
//!   `bytes_get`/`bytes_len` prims exercised by M-750 above. The full three-way differential runs
//!   in the `string_literal_*_surface_three_way` tests below; the explicit, minimal escape set
//!   (`\n \t \\ \" \0 \r`) and its never-silent termination/escape errors are pinned by the
//!   `string_*_reject` tests.
//! - **M-897** (ADR-040, kickoff `enb` Phase-I H1 Gap A) — the **`.myc` surface is now wired** for
//!   the decimal float literal (`1.5` / `0.0` / `1e10` / `2.5e-3`) and the nullary `Float` type
//!   (binary64 only — ADR-040 FLAG-1): it lowers to the **existing** `Repr::Float`/`Payload::Float`
//!   scalar value form landed by M-896 (KC-3 — no new L0 node). The literal denotes the
//!   **correctly-rounded** (RNE) binary64 of its decimal text (FLAG-3); that claim is pinned
//!   `Empirical` by the bit-exact `float_literal_round_trip_corpus` differential against rustc's
//!   own decimal→binary64 conversion. The full three-way runs in the `float_literal_*_three_way`
//!   tests; the never-silent form/range/pattern refusals are pinned by the `float_*_reject` tests.
//! - **M-898** (ADR-040 §2.5, kickoff `enb` Phase-I H1 Gap A) — the **scalar-float arithmetic
//!   prims** `flt_add`/`flt_sub`/`flt_mul`/`flt_div`/`flt_neg` (kernel `flt.*`): IEEE-754 binary64
//!   under RNE, arithmetic specials **in-band** per the ratified FLAG-2 (overflow → ±inf,
//!   `x/0` → ±inf, `0/0` → NaN — never a trap; the distinguished sentinel is the never-silent
//!   signal), every NaN canonical (§2.3). Per-op tag **`Empirical`** per the ratified ADR-040
//!   §2.6 (host-RNE conformance, zero-deviation-vs-spec bound; no `Proven` anywhere), inspected
//!   off the value on every path below. Because M-897's float literal landed, the **nullary-main
//!   surface three-way closes** for float arithmetic (`flt_arith_*_three_way` below) — unlike the
//!   dense group, whose surface leg still injects kernel-built arguments (see the M-890 note).
//!   Static conformance accept/reject in the `flt_prims_conformance_*` tests.
//! - **M-899** (ADR-040 §2.4, kickoff `enb` Phase-I H1 Gap A) — the **scalar-float comparison
//!   prims** `flt_lt`/`flt_le`/`flt_gt`/`flt_ge`/`flt_eq` (the IEEE-754 §5.11 partial-order
//!   predicates: **NaN is unordered — any NaN operand yields the defined value `false`**,
//!   `flt_eq(NaN, NaN)` included) plus the **named, opt-in total order** `flt_total_le`
//!   (IEEE-754 §5.10 `totalOrder`: `−inf < … < −0 < +0 < … < +inf < NaN`, reflexive, canonical
//!   NaN last, signed zeros directed), kernel `flt.lt`/…/`flt.eq`/`flt.total_le`. Two `Float`
//!   operands collapse to `Binary{1}` (the realized `Bool`). Per-op tag **`Empirical`** per
//!   ADR-040 §2.6; **the `flt_total_le` total-order property is the M-511 proof debt — it stays
//!   `Empirical` until a proof lands, never `Proven` on host documentation (VR-5)**. The
//!   nullary-main surface three-way closes (`flt_cmp_*_three_way` below) with the NaN-unordered
//!   behavior pinned on every path; static accept/reject in the `flt_cmp_conformance_*` tests.
//! - **M-900** (`enb` Phase-I H1 Gap A capstone) — the **float three-way conformance closeout**:
//!   verification + recording, not new prims. Confirms the M-896…M-899 float suite above is
//!   comprehensive and coherent across the literal, `flt.{add,sub,mul,div,neg}`,
//!   `flt.{lt,le,gt,ge,eq,total_le}`, in-band specials, signed zeros, and canonical-NaN identity —
//!   adding exactly one genuinely-missing corner (`flt_arith_nan_propagates_and_recanonicalizes_three_way`:
//!   a NaN operand *propagates* through arithmetic, not only *produced* by `0/0`, three-way).
//!   **Recorded honestly (G2/VR-5): every float form here closes three-way — there is no AOT
//!   refusal to record for Gap A** (unlike the M-890/M-891 dense group, which has no nullary-main
//!   surface form yet). `docs/spec/stdlib/self-hosting-readiness.md` §0 blocker-1 is re-verified
//!   and recorded closed against this evidence. Residual FLAGs (not this task's scope, carried
//!   forward): `is_nan`/`is_finite` classification prims are still OPEN (`mycelium-core/src/prim.rs`
//!   flt.add doc comment) — NaN is detectable today via `¬flt_eq(x, x)` and finiteness via
//!   `flt_lt(-inf, x) ∧ flt_lt(x, +inf)`, so the float gate itself does not need them; and the
//!   `flt.*`/`Float` surface-name ratification is deferred to the `integration` tier.
//! - **M-892** (RFC-0003 §3/§4, ADR-008, kickoff `enb` Phase-I H1 Gap C) — the **model-dispatched
//!   VSA bind group** `vsa_bind`/`vsa_unbind`/`vsa_permute` (kernel `vsa.*`, dispatch set
//!   MAP-I/FHRR/BSC), riding the new `Ty::Vsa` type-level lift (`VSA{model, dim, sparsity}` now
//!   resolves; the surface `MAP_I` ident canonicalizes to the kernel `MAP-I` id). Per-op tags are
//!   **per-model**, constructed by the `mycelium-vsa` kernel and carried unchanged (MAP-I/BSC ops
//!   `Exact`; FHRR `unbind` `Empirical` with its trial-validated δ — VR-5). Like the M-890 dense
//!   group, there is **no surface hypervector construction form**, so the three-way's surface leg
//!   injects `Repr::Vsa` argument values (see the M-892 section note below — recorded honestly,
//!   not silently skipped); AOT closes over the equivalent hand-built `Node::Op` on every case,
//!   including the runtime refusals. Static accept/reject in the `vsa_prims_conformance_*` tests.
//! - **M-893** (RFC-0003 §4/§5, ADR-008, kickoff `enb` Phase-I H1 Gap C) — **`vsa_bundle`**, the
//!   **certified superposition path** (kernel `vsa.bundle` → MAP-I's `bundle_values_certified`,
//!   the M-131 checked-instantiation pattern): `Seq{VSA{…}, N≥1}` × `Float` δ → `VSA{…}`, the
//!   dispatch set the **certified singleton {MAP-I}** (FHRR/BSC bundles are Empirical-profile
//!   kernel ops — statically refused naming the certified set; an append-only future surfacing,
//!   never a silent re-tag — VR-5). The result carries the kernel's **`Proven`** tag + its
//!   checked `CapacityBound` (the value's own m/dim, `ProvenThm` citation) unchanged on every
//!   path; an under-dimensioned bundle is an explicit `InsufficientCapacity` refusal naming the
//!   required dim, never an unbacked `Proven`. Same three-way posture as M-892 (injected
//!   hypervector arguments — no VSA construction form yet; the `Seq` rides the surface list
//!   literal, δ rides a `Float` param/literal). Static accept/reject in the
//!   `vsa_bundle_conformance_*` tests, incl. the *static* empty-bundle (`N = 0`) refusal (N
//!   lives in the `Seq` type).
//! - **M-894** (RFC-0003 §3/§5/§6, ADR-008, FR-S4, kickoff `enb` Phase-I H1 Gap C) — the
//!   **cleanup/reconstruction pair + the capacity query**: `vsa_cleanup` (query × codebook →
//!   the `Seq{Float, 3}` `[index, confidence, margin]` decision triple — retrieval is never a
//!   silent nearest-neighbour pick; a tie is the RFC-0010 §4.4 identifiability refusal),
//!   `vsa_reconstruct` (record × role × codebook × `Float` threshold → the triple; dispatch set
//!   **{MAP-I, BSC}** — an FHRR record is a *static* refusal naming its unbind profile's regime;
//!   a below-threshold retrieval is an explicit runtime refusal naming confidence vs threshold),
//!   and `vsa_required_dim` (`Binary{W}` items × `Float` δ → `Binary{64}`, the M-131
//!   `requiredDim` checked instantiation, its result carrying the kernel's **`Proven`**
//!   `CapacityBound` — the capacity-bound query made inspectable). The triple carries the
//!   query/record's own (strength, bound) pair through the §4.7 meet — reconstructing from a
//!   certified bundle re-discloses its `Proven` `CapacityBound` (the disclosed bound is the
//!   value's own — VR-5). Same three-way posture as M-892/M-893 (injected hypervector
//!   arguments — no VSA construction form yet; codebooks ride the surface list literal,
//!   thresholds/δ ride `Float` params/literals). Static accept/reject in the
//!   `vsa_cleanup_reconstruct_conformance_*` tests, incl. the *static* empty-codebook (`N = 0`)
//!   and FHRR-reconstruct refusals.

use mycelium_core::{
    Bound, BoundBasis, BoundKind, FloatWidth, GuaranteeStrength, Meta, Node, NormKind, Payload,
    Provenance, Repr, Value,
};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::{check_nodule, elaborate, parse, Evaluator};

/// Run the three-way differential on `src` (L1-eval ≡ elaborate→L0-interp ≡ AOT) and assert all
/// three paths agree on the observable (`repr + payload`) AND equal the `expected` reference value.
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

/// `Binary{1}` truth payload — the realized `Bool` of RFC-0032 D1.
fn b1(truth: bool) -> (Repr, Payload) {
    (Repr::Binary { width: 1 }, Payload::Bits(vec![truth]))
}

// ── M-747: width-typed comparison/equality (unblocks E13-1 M-718 cmp/Eq/Ord) ────────────────────

#[test]
fn eq_binary_width_typed() {
    let (r, p) = b1(true);
    assert_three_way(
        "eq Binary equal",
        "nodule d;\nfn main() => Binary{1} = eq(0b1010_0000, 0b1010_0000);",
        &r,
        &p,
    );
    let (r, p) = b1(false);
    assert_three_way(
        "eq Binary unequal",
        "nodule d;\nfn main() => Binary{1} = eq(0b1010_0000, 0b1010_0001);",
        &r,
        &p,
    );
}

#[test]
fn eq_ternary_width_typed() {
    let (r, p) = b1(true);
    assert_three_way(
        "eq Ternary equal",
        "nodule d;\nfn main() => Binary{1} = eq(0t00+-, 0t00+-);",
        &r,
        &p,
    );
    let (r, p) = b1(false);
    assert_three_way(
        "eq Ternary unequal",
        "nodule d;\nfn main() => Binary{1} = eq(0t00+-, 0t0+0-);",
        &r,
        &p,
    );
}

#[test]
fn lt_binary_unsigned_magnitude() {
    // 0b1000_0000 (128) < 0b1010_0000 (160).
    let (r, p) = b1(true);
    assert_three_way(
        "lt Binary true",
        "nodule d;\nfn main() => Binary{1} = lt(0b1000_0000, 0b1010_0000);",
        &r,
        &p,
    );
    // Not strictly less when equal.
    let (r, p) = b1(false);
    assert_three_way(
        "lt Binary equal-is-false",
        "nodule d;\nfn main() => Binary{1} = lt(0b1010_0000, 0b1010_0000);",
        &r,
        &p,
    );
}

#[test]
fn lt_ternary_balanced_value() {
    // <00+-> = 2, <0+0-> = 8, so 2 < 8 is true.
    let (r, p) = b1(true);
    assert_three_way(
        "lt Ternary true",
        "nodule d;\nfn main() => Binary{1} = lt(0t00+-, 0t0+0-);",
        &r,
        &p,
    );
    // <0-00> = -9 < <00+-> = 2 → true; reversed → false (negative magnitude ordering).
    let (r, p) = b1(false);
    assert_three_way(
        "lt Ternary negative-false",
        "nodule d;\nfn main() => Binary{1} = lt(0t00+-, 0t0-00);",
        &r,
        &p,
    );
}

/// The `.myc` `Bool`-bridge smoke port: match the `Binary{1}` comparison bit into the `Bool` ADT,
/// exactly the one-line lift the E13-1 `std.cmp` port (M-718) bottoms out on.
#[test]
fn bool_bridge_from_comparison_bit() {
    // `match eq(a, b) { 0b1 => True, _ => False }` ≡ the data value `True`.
    let src = "nodule d;\nfn main() => Bool = match eq(0b1010_0000, 0b1010_0000) { 0b1 => True, _ => False };";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let val = Evaluator::new(&env).call("main", vec![]).expect("L1-eval");
    // The result is the `True` data constructor (the lift succeeded).
    let core = format!("{val:?}");
    assert!(
        core.contains("True"),
        "bool-bridge must yield the `True` constructor, got {core}"
    );
}

// ── M-748: never-silent binary arithmetic (unblocks E13-1 M-718 binary math) ────────────────────

#[test]
fn and_or_surfaced() {
    assert_three_way(
        "and",
        "nodule d;\nfn main() => Binary{8} = and(0b1100_1010, 0b1010_1010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("10001010".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "or",
        "nodule d;\nfn main() => Binary{8} = or(0b1100_1010, 0b1010_1010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11101010".chars().map(|c| c == '1').collect()),
    );
}

#[test]
fn add_u_in_range() {
    // 0b0000_0001 + 0b0000_0010 = 0b0000_0011 (1 + 2 = 3).
    assert_three_way(
        "add_u",
        "nodule d;\nfn main() => Binary{8} = add_u(0b0000_0001, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
}

#[test]
fn sub_u_in_range() {
    // 0b0000_0101 - 0b0000_0010 = 0b0000_0011 (5 - 2 = 3).
    assert_three_way(
        "sub_u",
        "nodule d;\nfn main() => Binary{8} = sub_u(0b0000_0101, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
}

// ── Never-silent (G2/VR-5): overflow + mismatch refuse on every path ─────────────────────────────

/// `add_u` overflow (`255 + 1` at `Binary{8}`) is an explicit refusal on **all three** paths —
/// never a silent wrap to `0`. (The program type-checks: overflow is a runtime contract, D2.)
#[test]
fn add_u_overflow_refuses_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = add_u(0b1111_1111, 0b0000_0001);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "L1-eval must refuse the overflow (never a silent wrap)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "L0-interp must refuse the overflow"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "AOT must refuse the overflow"
    );
}

/// `sub_u` underflow (`0 - 1` at `Binary{8}`, a negative with no unsigned form) refuses on **all
/// three** paths — never a silent wrap to `255` — exactly like the overflow test above.
#[test]
fn sub_u_underflow_refuses_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = sub_u(0b0000_0000, 0b0000_0001);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "L1-eval must refuse the underflow (never a silent wrap to 255)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "L0-interp must refuse the underflow"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "AOT must refuse the underflow"
    );
}

/// A cross-paradigm comparison (`Binary` vs `Ternary`) is a **static** never-silent refusal — caught
/// at check time, never a silent `false` (RFC-0032 D1).
#[test]
fn eq_cross_paradigm_refuses_statically() {
    let src = "nodule d;\nfn main() => Binary{1} = eq(0b0000_0001, 0t00+-);";
    let err = check_nodule(&parse(src).expect("parses"));
    assert!(
        err.is_err(),
        "a Binary-vs-Ternary `eq` must be a static type error, never a silent false"
    );
}

/// When **both** comparands are bare ambient decimals neither pins a *width* (and the `Binary{1}`
/// result can't anchor them, comparison being width-collapsing) — refused, never a defaulted width
/// (RFC-0032 D1 / RFC-0012 §4.3). `default paradigm Binary` makes `5`/`6` ambient (paradigm known,
/// width unknown), so this exercises the width-anchor refusal specifically.
#[test]
fn eq_both_bare_decimals_refuse() {
    let src = "nodule d;\ndefault paradigm Binary;\nfn main() => Binary{1} = eq(5, 6);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a both-bare-decimal `eq` must refuse (no width anchor), never a default width"
    );
}

/// A **concrete** operand anchors a bare ambient comparand's width (consistent with the
/// width-preserving prims, e.g. `xor(0b1111_0000, 15)`): under `default paradigm Binary`,
/// `eq(0b0000_0101, 5)` type-checks with `5` resolving to `Binary{8}`, evaluating to `0b1`.
#[test]
fn eq_concrete_operand_anchors_bare_decimal() {
    let (r, p) = b1(true);
    assert_three_way(
        "eq concrete anchors bare decimal",
        "nodule d;\ndefault paradigm Binary;\nfn main() => Binary{1} = eq(0b0000_0101, 5);",
        &r,
        &p,
    );
    // Order-independent: bare decimal first, concrete second (`4` ≠ `5`).
    let (r, p) = b1(false);
    assert_three_way(
        "eq bare-first anchored false",
        "nodule d;\ndefault paradigm Binary;\nfn main() => Binary{1} = eq(4, 0b0000_0101);",
        &r,
        &p,
    );
}

// ── M-887 (`enb` Gap B): never-silent two's-complement multiply ─────────────────────────────────
//
// `mul_s` (kernel `bin.mul`) is the first Gap-B prim of the RFC-0033 §4.1.2/§4.1.3 shared
// two's-complement arithmetic set (ADR-028). It reads its `Binary{N}` operands under the
// two's-complement (signed) interpretation — distinct from `add_u`/`sub_u`'s existing
// **unsigned** overflow contract (RFC-0032 D2) — and refuses out-of-`B_N` products explicitly,
// never a silent wrap (G2/VR-5).

#[test]
fn mul_s_in_range_positive_and_negative() {
    // 3 * 4 = 12.
    assert_three_way(
        "mul_s positive",
        "nodule d;\nfn main() => Binary{8} = mul_s(0b0000_0011, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001100".chars().map(|c| c == '1').collect()),
    );
    // -3 * 4 = -12 (two's complement: -3 = 0b1111_1101, -12 = 0b1111_0100).
    assert_three_way(
        "mul_s negative operand",
        "nodule d;\nfn main() => Binary{8} = mul_s(0b1111_1101, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11110100".chars().map(|c| c == '1').collect()),
    );
    // -3 * -4 = 12.
    assert_three_way(
        "mul_s both negative",
        "nodule d;\nfn main() => Binary{8} = mul_s(0b1111_1101, 0b1111_1100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001100".chars().map(|c| c == '1').collect()),
    );
}

/// `mul_s` overflow (`127 * 2` at `Binary{8}`, out of `B_8 = [-128, 127]`) is an explicit refusal
/// on **all three** paths — never a silent wrap. (The program type-checks: the two's-complement
/// overflow bound is a runtime contract, like `add_u`/`sub_u`'s unsigned one.)
#[test]
fn mul_s_overflow_refuses_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = mul_s(0b0111_1111, 0b0000_0010);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "L1-eval must refuse the overflow (never a silent wrap)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "L0-interp must refuse the overflow"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "AOT must refuse the overflow"
    );
}

/// The classic two's-complement multiply-overflow edge (`i8::MIN * -1 = 128`, out of `B_8`) refuses
/// on all three paths — never a silent wrap back to `-128`.
#[test]
fn mul_s_min_times_neg_one_refuses_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = mul_s(0b1000_0000, 0b1111_1111);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "i8::MIN * -1 must refuse on L1-eval (never a silent wrap to -128)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "i8::MIN * -1 must refuse on L0-interp"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "i8::MIN * -1 must refuse on AOT"
    );
}

// ── CU-1 (RFC-0033 §4.1.2): never-silent UNSIGNED multiply `mul_u` (kernel `bit.mul`) ────────────
//
// The unsigned member of the `bit.*` family — overflow-distinct from the signed `mul_s`/`bin.mul`
// (the `lib/std/math.myc` FLAG-math-1 missing op). Reads operands as unsigned bitvectors; an
// out-of-`U_N` product is an explicit refusal on all three paths, never a wrap (G2/VR-5).

#[test]
fn mul_u_in_range_including_high_bit() {
    // 3 * 4 = 12.
    assert_three_way(
        "mul_u small",
        "nodule d;\nfn main() => Binary{8} = mul_u(0b0000_0011, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001100".chars().map(|c| c == '1').collect()),
    );
    // 15 * 17 = 255 — the high boundary of U_8, in range.
    assert_three_way(
        "mul_u high boundary",
        "nodule d;\nfn main() => Binary{8} = mul_u(0b0000_1111, 0b0001_0001);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111111".chars().map(|c| c == '1').collect()),
    );
    // 9 * 20 = 180 — in U_8 = [0,255] but OUT of signed B_8 = [-128,127]: the criterion that
    // distinguishes `mul_u` from `mul_s` (which refuses this exact product).
    assert_three_way(
        "mul_u unsigned-only product",
        "nodule d;\nfn main() => Binary{8} = mul_u(0b0000_1001, 0b0001_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("10110100".chars().map(|c| c == '1').collect()),
    );
}

/// `mul_u` overflow (`16 * 16 = 256`, out of `U_8 = [0, 255]`) is an explicit refusal on **all
/// three** paths — never a silent wrap to `0`. (The program type-checks: the unsigned-overflow
/// bound is a runtime contract, like `add_u`/`sub_u`'s.)
#[test]
fn mul_u_overflow_refuses_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = mul_u(0b0001_0000, 0b0001_0000);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "L1-eval must refuse the unsigned overflow (never a silent wrap to 0)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "L0-interp must refuse the unsigned overflow"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "AOT must refuse the unsigned overflow"
    );
}

// ── CU-6: width-preserving bit-manipulation counts `popcount`/`clz`/`ctz` (kernel bit.*) ─────────
//
// popcount/clz/ctz are kernel prims (single host instruction, not efficiently `.myc`-derivable);
// rotate/reverse_bits ride `std.math`. Unary `Binary{N} → Binary{N}`, total (a count always fits N
// bits), agreeing on all three paths.
#[test]
fn bit_manip_counts_three_way() {
    // popcount(0b0110_1000) = 3; clz = 1; ctz = 3 (0b0000_0011, 0b0000_0001, 0b0000_0011).
    assert_three_way(
        "popcount",
        "nodule d;\nfn main() => Binary{8} = popcount(0b0110_1000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "clz",
        "nodule d;\nfn main() => Binary{8} = clz(0b0110_1000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000001".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "ctz",
        "nodule d;\nfn main() => Binary{8} = ctz(0b0110_1000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
    // All-zero: clz = ctz = 8 (0b0000_1000), popcount = 0.
    assert_three_way(
        "clz all-zero is n",
        "nodule d;\nfn main() => Binary{8} = clz(0b0000_0000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001000".chars().map(|c| c == '1').collect()),
    );
}

/// A width/paradigm mismatch (`Binary{8}` vs `Binary{1}`) is a **static** never-silent refusal —
/// caught at check time, mirroring `add_u`/`sub_u`'s width-preserving contract.
#[test]
fn mul_s_width_mismatch_refuses_statically() {
    let src = "nodule d;\nfn main() => Binary{8} = mul_s(0b0000_0001, 0b0);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a width-mismatched mul_s must be a static type error, never a silent coercion"
    );
}

// ── M-888 (`enb` Gap B): never-silent unsigned division/remainder ───────────────────────────────
//
// `div_u`/`rem_u` (kernel `bin.div`/`bin.rem`) are the second Gap-B prims of the RFC-0033
// §4.1.2/§4.1.3 arithmetic set. Division *differs* by signedness (§4.1.2), so — unlike `mul_s` —
// it MUST be a distinct-named op per signedness; this lands the **unsigned** reading first (the
// signed reading rides M-767 under its own name). Division by zero refuses explicitly on every
// path, never a panic or a silent value (G2/VR-5).

#[test]
fn div_u_and_rem_u_worked_examples() {
    // 7 / 2 = 3, 7 % 2 = 1.
    assert_three_way(
        "div_u 7/2",
        "nodule d;\nfn main() => Binary{8} = div_u(0b0000_0111, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "rem_u 7%2",
        "nodule d;\nfn main() => Binary{8} = rem_u(0b0000_0111, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000001".chars().map(|c| c == '1').collect()),
    );
    // 255 / 1 = 255, 255 % 1 = 0 (upper boundary at Binary{8}).
    assert_three_way(
        "div_u 255/1",
        "nodule d;\nfn main() => Binary{8} = div_u(0b1111_1111, 0b0000_0001);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111111".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "rem_u 255%1",
        "nodule d;\nfn main() => Binary{8} = rem_u(0b1111_1111, 0b0000_0001);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000000".chars().map(|c| c == '1').collect()),
    );
    // 0 / 17 = 0, 0 % 17 = 0.
    assert_three_way(
        "div_u 0/17",
        "nodule d;\nfn main() => Binary{8} = div_u(0b0000_0000, 0b0001_0001);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000000".chars().map(|c| c == '1').collect()),
    );
}

/// Division/remainder by zero (`7 / 0`, `7 % 0` at `Binary{8}`) is an explicit refusal on **all
/// three** paths — never a panic or a silently-defined value. (The program type-checks: div-by-zero
/// is a runtime contract, like `mul_s`'s overflow.)
#[test]
fn div_u_and_rem_u_by_zero_refuse_on_every_path() {
    for src in [
        "nodule d;\nfn main() => Binary{8} = div_u(0b0000_0111, 0b0000_0000);",
        "nodule d;\nfn main() => Binary{8} = rem_u(0b0000_0111, 0b0000_0000);",
    ] {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        let interp = Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(mycelium_cert::BinaryTernarySwapEngine),
        );
        let prims = PrimRegistry::with_builtins();
        let engine = mycelium_cert::BinaryTernarySwapEngine;

        assert!(
            Evaluator::new(&env).call("main", vec![]).is_err(),
            "L1-eval must refuse division by zero (never a silent value): {src}"
        );
        let node = elaborate(&env, "main").expect("in fragment");
        assert!(
            interp.eval(&node).is_err(),
            "L0-interp must refuse division by zero: {src}"
        );
        assert!(
            mycelium_mlir::run(&node, &prims, &engine).is_err(),
            "AOT must refuse division by zero: {src}"
        );
    }
}

/// A width/paradigm mismatch (`Binary{8}` vs `Binary{1}`) is a **static** never-silent refusal —
/// caught at check time, mirroring `mul_s`'s width-preserving contract.
#[test]
fn div_u_width_mismatch_refuses_statically() {
    let src = "nodule d;\nfn main() => Binary{8} = div_u(0b0000_0001, 0b0);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a width-mismatched div_u must be a static type error, never a silent coercion"
    );
}

// ── M-889 (`enb` Gap B): never-silent logical left/right shift ──────────────────────────────────
//
// `shl_u`/`shr_u` (kernel `bin.shl`/`bin.shr`) are the third Gap-B prim pair of the RFC-0033
// §4.1.2/§4.1.3 shared shift op set — the **logical** (unsigned) reading, landed first per the
// signedness-split requirement (§4.1.2), mirroring `div_u`/`rem_u`. Both operands are
// `Binary{N}` (the shift amount is itself read as an unsigned `N`-bit bitvector); a shift amount
// `>= N` refuses explicitly on every path, never UB, a wrapped shift amount, or a silently-zeroed
// result (G2/VR-5). The arithmetic/signed right shift rides M-767 under its own distinct name.

#[test]
fn shl_u_and_shr_u_worked_examples() {
    // 1 << 3 = 8.
    assert_three_way(
        "shl_u 1<<3",
        "nodule d;\nfn main() => Binary{8} = shl_u(0b0000_0001, 0b0000_0011);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001000".chars().map(|c| c == '1').collect()),
    );
    // 8 >> 3 = 1.
    assert_three_way(
        "shr_u 8>>3",
        "nodule d;\nfn main() => Binary{8} = shr_u(0b0000_1000, 0b0000_0011);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000001".chars().map(|c| c == '1').collect()),
    );
    // Logical (zero-filling) right shift: 0b1000_0000 >> 4 = 0b0000_1000, never sign-extended.
    assert_three_way(
        "shr_u logical zero-fill",
        "nodule d;\nfn main() => Binary{8} = shr_u(0b1000_0000, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001000".chars().map(|c| c == '1').collect()),
    );
    // Shift by 0 is the identity.
    assert_three_way(
        "shl_u by 0 is identity",
        "nodule d;\nfn main() => Binary{8} = shl_u(0b1010_1010, 0b0000_0000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("10101010".chars().map(|c| c == '1').collect()),
    );
}

/// A shift amount `>= width` (`8 << 8`/`8 >> 8` at `Binary{8}`) is an explicit refusal on **all
/// three** paths — never UB, a silently wrapped shift amount, or a silently-zeroed result. (The
/// program type-checks: an out-of-range shift amount is a runtime contract, like `div_u`'s
/// div-by-zero.)
#[test]
fn shl_u_and_shr_u_out_of_range_shift_refuse_on_every_path() {
    for src in [
        "nodule d;\nfn main() => Binary{8} = shl_u(0b0000_0001, 0b0000_1000);",
        "nodule d;\nfn main() => Binary{8} = shr_u(0b0000_0001, 0b0000_1000);",
    ] {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        let interp = Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(mycelium_cert::BinaryTernarySwapEngine),
        );
        let prims = PrimRegistry::with_builtins();
        let engine = mycelium_cert::BinaryTernarySwapEngine;

        assert!(
            Evaluator::new(&env).call("main", vec![]).is_err(),
            "L1-eval must refuse an out-of-range shift amount (never UB/wrap/silent): {src}"
        );
        let node = elaborate(&env, "main").expect("in fragment");
        assert!(
            interp.eval(&node).is_err(),
            "L0-interp must refuse an out-of-range shift amount: {src}"
        );
        assert!(
            mycelium_mlir::run(&node, &prims, &engine).is_err(),
            "AOT must refuse an out-of-range shift amount: {src}"
        );
    }
}

/// A width/paradigm mismatch (`Binary{8}` vs `Binary{1}`) is a **static** never-silent refusal —
/// caught at check time, mirroring `div_u`'s width-preserving contract.
#[test]
fn shl_u_width_mismatch_refuses_statically() {
    let src = "nodule d;\nfn main() => Binary{8} = shl_u(0b0000_0001, 0b0);";
    assert!(
        check_nodule(&parse(src).expect("parses")).is_err(),
        "a width-mismatched shl_u must be a static type error, never a silent coercion"
    );
}

// ── M-766 (`enb` Gap B): never-silent two's-complement add/sub/neg ──────────────────────────────
//
// `add_s`/`sub_s`/`neg_s` (kernel `bin.add`/`bin.sub`/`bin.neg`) complete the *shared*
// two's-complement arithmetic set `mul_s` (M-887) started — RFC-0033 §4.1.2/§4.1.3, ADR-028.
//
// **Inventory (verified before landing, per the M-766 task's "reconcile against the kpr-landed
// add/sub" instruction).** The pre-existing `add_u`/`sub_u` (kernel `bit.add`/`bit.sub`,
// kpr/E19-1, RFC-0032 D2) are a **different, unsigned-committed** family: their overflow criterion
// is the unsigned carry/borrow-out, which *under-refuses* relative to the signed range `B_N` (e.g.
// at `Binary{4}`, `5 + 3 = 8` is unsigned-in-range `[0,15]` but signed-out-of-range `B_4 = [-8,7]`),
// so they do not stand in for the RFC-0033 shared `add`/`sub`. `add_s`/`sub_s` are therefore
// genuinely missing (not a re-land of E19-1's work), completed here alongside `neg_s` (which has
// no pre-existing counterpart at all — negation is inherently a signed concept). Naming: the
// `_u`/`_s` signedness suffixes follow the DN-72 surface-naming convention (ratified 2026-07-02;
// ADR-028); see the `checkty::prim_family` naming comment.

#[test]
fn add_s_and_sub_s_worked_examples() {
    // 3 + 4 = 7.
    assert_three_way(
        "add_s positive",
        "nodule d;\nfn main() => Binary{8} = add_s(0b0000_0011, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000111".chars().map(|c| c == '1').collect()),
    );
    // -3 + 4 = 1 (two's complement: -3 = 0b1111_1101).
    assert_three_way(
        "add_s negative operand",
        "nodule d;\nfn main() => Binary{8} = add_s(0b1111_1101, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000001".chars().map(|c| c == '1').collect()),
    );
    // 7 - 4 = 3.
    assert_three_way(
        "sub_s positive",
        "nodule d;\nfn main() => Binary{8} = sub_s(0b0000_0111, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
    // 4 - (-3) = 7 (two's complement: -3 = 0b1111_1101).
    assert_three_way(
        "sub_s subtract-negative",
        "nodule d;\nfn main() => Binary{8} = sub_s(0b0000_0100, 0b1111_1101);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000111".chars().map(|c| c == '1').collect()),
    );
}

#[test]
fn neg_s_worked_examples() {
    // -(3) = -3 (0b1111_1101).
    assert_three_way(
        "neg_s positive operand",
        "nodule d;\nfn main() => Binary{8} = neg_s(0b0000_0011);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111101".chars().map(|c| c == '1').collect()),
    );
    // -(-3) = 3.
    assert_three_way(
        "neg_s negative operand",
        "nodule d;\nfn main() => Binary{8} = neg_s(0b1111_1101);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
    // -(0) = 0.
    assert_three_way(
        "neg_s zero",
        "nodule d;\nfn main() => Binary{8} = neg_s(0b0000_0000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000000".chars().map(|c| c == '1').collect()),
    );
}

/// `add_s`/`sub_s` overflow (`127 + 1`, `-128 - 1` at `Binary{8}`, both out of `B_8 = [-128,
/// 127]`) is an explicit refusal on **all three** paths — never a silent wrap. (The program
/// type-checks: the two's-complement overflow bound is a runtime contract, like `mul_s`'s.)
#[test]
fn add_s_and_sub_s_overflow_refuse_on_every_path() {
    for src in [
        "nodule d;\nfn main() => Binary{8} = add_s(0b0111_1111, 0b0000_0001);",
        "nodule d;\nfn main() => Binary{8} = sub_s(0b1000_0000, 0b0000_0001);",
    ] {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        let interp = Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(mycelium_cert::BinaryTernarySwapEngine),
        );
        let prims = PrimRegistry::with_builtins();
        let engine = mycelium_cert::BinaryTernarySwapEngine;

        assert!(
            Evaluator::new(&env).call("main", vec![]).is_err(),
            "L1-eval must refuse the overflow (never a silent wrap): {src}"
        );
        let node = elaborate(&env, "main").expect("in fragment");
        assert!(
            interp.eval(&node).is_err(),
            "L0-interp must refuse the overflow: {src}"
        );
        assert!(
            mycelium_mlir::run(&node, &prims, &engine).is_err(),
            "AOT must refuse the overflow: {src}"
        );
    }
}

/// The classic two's-complement negate-overflow edge (`i8::MIN` negated at `Binary{8}`, out of
/// `B_8 = [-128, 127]`) refuses on all three paths — never a silent wrap back to `-128`.
#[test]
fn neg_s_min_value_refuses_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = neg_s(0b1000_0000);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "-(i8::MIN) must refuse on L1-eval (never a silent wrap to -128)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "-(i8::MIN) must refuse on L0-interp"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "-(i8::MIN) must refuse on AOT"
    );
}

/// A width/paradigm mismatch (`Binary{8}` vs `Binary{1}`) is a **static** never-silent refusal for
/// `add_s`/`sub_s` — caught at check time, mirroring `mul_s`'s width-preserving contract.
#[test]
fn add_s_and_sub_s_width_mismatch_refuse_statically() {
    for src in [
        "nodule d;\nfn main() => Binary{8} = add_s(0b0000_0001, 0b0);",
        "nodule d;\nfn main() => Binary{8} = sub_s(0b0000_0001, 0b0);",
    ] {
        assert!(
            check_nodule(&parse(src).expect("parses")).is_err(),
            "a width-mismatched add_s/sub_s must be a static type error, never a silent \
             coercion: {src}"
        );
    }
}

// ── M-767 (`enb` Gap B): the signedness-split signed op set — div_s/rem_s/shr_s/lt_s ────────────
//
// `div_s`/`rem_s`/`shr_s`/`lt_s` (kernel `bin.div_s`/`bin.rem_s`/`bin.shr_s`/`cmp.lt_s`) complete
// the RFC-0033 §4.1.2 **signedness split**: division, ordering, right shift, and overflow
// *detection* differ by signedness, so each signed reading is a **distinct named op** from its
// landed `_u` counterpart (ADR-028; DN-72 `_s` suffixes — the names this task's slot pre-assigned).
//
// **Rounding convention (grounding, VR-5):** signed division is **truncated toward zero**, the
// remainder's sign following the dividend — the ADR-028-cited SMT-LIB `bvsdiv`/`bvsrem` semantics
// (RFC-0033's text does not literally pin the rounding; the choice is grounded on that citation
// and FLAGged in the M-767 report, never silently made). The signed-division overflow-detect case
// (`min ÷ −1`, quotient `+2^(N-1)` out of `B_N`) refuses explicitly on every path — §4.1.3's
// never-silent overflow outranks SMT-LIB's defined wrap. `shr_s` is the **arithmetic**
// (sign-extending) right shift; `lt_s` the two's-complement order (`Binary`-only: balanced
// ternary's `lt` is already the signed order).

#[test]
fn div_s_and_rem_s_worked_examples_pin_truncation() {
    // 7 / 2 = 3 r 1.
    assert_three_way(
        "div_s 7/2",
        "nodule d;\nfn main() => Binary{8} = div_s(0b0000_0111, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000011".chars().map(|c| c == '1').collect()),
    );
    // -7 / 2 = -3 (truncated toward zero — a floored convention would answer -4 = 0b1111_1100).
    assert_three_way(
        "div_s -7/2 truncates toward zero",
        "nodule d;\nfn main() => Binary{8} = div_s(0b1111_1001, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111101".chars().map(|c| c == '1').collect()),
    );
    // -7 % 2 = -1 (sign follows the dividend — floored would answer +1).
    assert_three_way(
        "rem_s -7%2 sign follows the dividend",
        "nodule d;\nfn main() => Binary{8} = rem_s(0b1111_1001, 0b0000_0010);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111111".chars().map(|c| c == '1').collect()),
    );
    // 7 / -2 = -3 r 1.
    assert_three_way(
        "div_s 7/-2",
        "nodule d;\nfn main() => Binary{8} = div_s(0b0000_0111, 0b1111_1110);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111101".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "rem_s 7%-2",
        "nodule d;\nfn main() => Binary{8} = rem_s(0b0000_0111, 0b1111_1110);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000001".chars().map(|c| c == '1').collect()),
    );
}

/// The §4.1.3 signed **overflow-detect** case: `-128 ÷ -1` (true quotient `+128`, out of `B_8`)
/// refuses on **all three** paths — never a silent wrap back to `-128`. `rem_s(-128, -1) = 0`
/// fits `B_8` exactly and succeeds three-way (deliberately not Rust `checked_rem`'s
/// hardware-motivated over-refusal — see `mycelium_core::binary::rem_signed`).
#[test]
fn div_s_min_by_neg_one_refuses_on_every_path_rem_s_succeeds() {
    let src = "nodule d;\nfn main() => Binary{8} = div_s(0b1000_0000, 0b1111_1111);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "-128 / -1 must refuse on L1-eval (never a silent wrap to -128)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "-128 / -1 must refuse on L0-interp"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "-128 / -1 must refuse on AOT"
    );

    // The remainder's exact result 0 fits B_8 — it succeeds on the same operands, three-way.
    assert_three_way(
        "rem_s -128%-1 = 0",
        "nodule d;\nfn main() => Binary{8} = rem_s(0b1000_0000, 0b1111_1111);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000000".chars().map(|c| c == '1').collect()),
    );
}

/// Signed division/remainder by zero refuses on **all three** paths, exactly as the unsigned pair
/// does — an explicit runtime refusal, never a panic or a silently-defined value.
#[test]
fn div_s_and_rem_s_by_zero_refuse_on_every_path() {
    for src in [
        "nodule d;\nfn main() => Binary{8} = div_s(0b1111_1001, 0b0000_0000);",
        "nodule d;\nfn main() => Binary{8} = rem_s(0b1111_1001, 0b0000_0000);",
    ] {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        let interp = Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(mycelium_cert::BinaryTernarySwapEngine),
        );
        let prims = PrimRegistry::with_builtins();
        let engine = mycelium_cert::BinaryTernarySwapEngine;

        assert!(
            Evaluator::new(&env).call("main", vec![]).is_err(),
            "L1-eval must refuse signed division by zero: {src}"
        );
        let node = elaborate(&env, "main").expect("in fragment");
        assert!(
            interp.eval(&node).is_err(),
            "L0-interp must refuse signed division by zero: {src}"
        );
        assert!(
            mycelium_mlir::run(&node, &prims, &engine).is_err(),
            "AOT must refuse signed division by zero: {src}"
        );
    }
}

/// `shr_s` **sign-extends**: `-128 >> 4 = -8` (`0b1000_0000` → `0b1111_1000`), where the logical
/// `shr_u` answers `+8` (`0b0000_1000`) — both pinned three-way so the signedness split is
/// visible in the differential itself.
#[test]
fn shr_s_sign_extends_three_way_where_shr_u_zero_fills() {
    assert_three_way(
        "shr_s -128>>4 sign-extends",
        "nodule d;\nfn main() => Binary{8} = shr_s(0b1000_0000, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111000".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "shr_u -128>>4 zero-fills (the unsigned twin, for contrast)",
        "nodule d;\nfn main() => Binary{8} = shr_u(0b1000_0000, 0b0000_0100);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001000".chars().map(|c| c == '1').collect()),
    );
    // -1 >> 3 = -1 (all-ones is a fixed point of sign extension).
    assert_three_way(
        "shr_s -1>>3 = -1",
        "nodule d;\nfn main() => Binary{8} = shr_s(0b1111_1111, 0b0000_0011);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111111".chars().map(|c| c == '1').collect()),
    );
    // A non-negative value agrees with the logical shift: 64 >> 3 = 8.
    assert_three_way(
        "shr_s 64>>3 = 8",
        "nodule d;\nfn main() => Binary{8} = shr_s(0b0100_0000, 0b0000_0011);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00001000".chars().map(|c| c == '1').collect()),
    );
}

/// An out-of-range shift amount (`k >= N`) refuses on **all three** paths for the arithmetic
/// shift, exactly as for the logical one — never an implicit "all sign bits" result.
#[test]
fn shr_s_out_of_range_shift_refuses_on_every_path() {
    for src in [
        "nodule d;\nfn main() => Binary{8} = shr_s(0b1111_1111, 0b0000_1000);",
        "nodule d;\nfn main() => Binary{8} = shr_s(0b1111_1111, 0b1111_1111);",
    ] {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        let interp = Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(mycelium_cert::BinaryTernarySwapEngine),
        );
        let prims = PrimRegistry::with_builtins();
        let engine = mycelium_cert::BinaryTernarySwapEngine;

        assert!(
            Evaluator::new(&env).call("main", vec![]).is_err(),
            "L1-eval must refuse an out-of-range arithmetic shift amount: {src}"
        );
        let node = elaborate(&env, "main").expect("in fragment");
        assert!(
            interp.eval(&node).is_err(),
            "L0-interp must refuse an out-of-range arithmetic shift amount: {src}"
        );
        assert!(
            mycelium_mlir::run(&node, &prims, &engine).is_err(),
            "AOT must refuse an out-of-range arithmetic shift amount: {src}"
        );
    }
}

/// `lt_s` is the two's-complement order: `0b1111_1111` is `-1 < 0` under `lt_s` but `255 > 0`
/// under the unsigned `lt` — the distinguishing pair pinned three-way against both prims.
#[test]
fn lt_s_orders_two_complement_where_lt_orders_magnitude() {
    let (r, p) = b1(true);
    assert_three_way(
        "lt_s -1 < 0",
        "nodule d;\nfn main() => Binary{1} = lt_s(0b1111_1111, 0b0000_0000);",
        &r,
        &p,
    );
    let (r, p) = b1(false);
    assert_three_way(
        "lt 255 !< 0 (the unsigned twin, for contrast)",
        "nodule d;\nfn main() => Binary{1} = lt(0b1111_1111, 0b0000_0000);",
        &r,
        &p,
    );
    // min < max; equal is not strictly less.
    let (r, p) = b1(true);
    assert_three_way(
        "lt_s -128 < 127",
        "nodule d;\nfn main() => Binary{1} = lt_s(0b1000_0000, 0b0111_1111);",
        &r,
        &p,
    );
    let (r, p) = b1(false);
    assert_three_way(
        "lt_s equal-is-false",
        "nodule d;\nfn main() => Binary{1} = lt_s(0b0000_0101, 0b0000_0101);",
        &r,
        &p,
    );
}

/// `lt_s` static refusal surface (conformance rejects, G2): a width mismatch and a cross-paradigm
/// pair are static type errors; a **ternary** pair refuses with the real routing (balanced
/// ternary's `lt` order is already the signed order — a distinct ternary `lt_s` would silently
/// duplicate it); a width-mismatched `div_s` likewise refuses statically.
#[test]
fn signed_op_static_rejects() {
    for (src, why) in [
        (
            "nodule d;\nfn main() => Binary{1} = lt_s(0b0000_0001, 0b0);",
            "width-mismatched lt_s",
        ),
        (
            "nodule d;\nfn main() => Binary{1} = lt_s(0b0000_0001, 0t00+-);",
            "cross-paradigm lt_s",
        ),
        (
            "nodule d;\nfn main() => Binary{1} = lt_s(0t00+-, 0t00+-);",
            "ternary lt_s (use lt — already the signed order)",
        ),
        (
            "nodule d;\nfn main() => Binary{8} = div_s(0b0000_0001, 0b0);",
            "width-mismatched div_s",
        ),
        (
            "nodule d;\nfn main() => Binary{8} = shr_s(0b0000_0001, 0b0);",
            "width-mismatched shr_s",
        ),
    ] {
        assert!(
            check_nodule(&parse(src).expect("parses")).is_err(),
            "{why} must be a static type error, never a silent coercion/order: {src}"
        );
    }
}

// ── M-749: indexed-sequence prims — prim-level differential (L0-interp ≡ AOT) ────────────────────
//
// `Repr::Seq` has no `.myc` surface literal yet (lexer/parser wiring deferred — FLAGGED in the
// module header), so the three-way `.myc` path can't run. Instead we build the L0 `Node` tree
// directly and exercise the achievable, trusted-base differential: the reference interpreter
// (`L0-interp`) and the AOT env-machine (`mycelium_mlir::run_core`) dispatch `seq.get`/`seq.len`
// through the *same* prim registry, so they must agree on the observable — and refuse an
// out-of-bounds index identically (never-silent on both paths, G2).

/// A `Binary{1}` value (a sequence element / an index bit-source).
fn b1_val(truth: bool) -> Value {
    Value::new(
        Repr::Binary { width: 1 },
        Payload::Bits(vec![truth]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed bit")
}

/// An unsigned `Binary{8}` index literal value (MSB-first).
fn idx8(n: u8) -> Value {
    let bits: Vec<bool> = (0..8).rev().map(|k| (n >> k) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed index")
}

/// A `Seq<Binary{1}, 3>` const value `[true, false, true]`.
fn seq3() -> Value {
    Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 3,
        },
        Payload::Seq(vec![b1_val(true), b1_val(false), b1_val(true)]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed seq")
}

/// Run a single-`Op` L0 program on **both** the reference interpreter and the AOT env-machine and
/// return `(l0_interp, aot)` results (each a `Result`), so a test can assert agreement on success
/// *and* on refusal.
fn run_l0_and_aot(node: &Node) -> (Result<Value, String>, Result<Value, String>) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;
    let l0 = interp.eval(node).map_err(|e| format!("{e:?}"));
    // `run` returns the repr `Value` (the seq prims always yield a repr value, never a data value).
    let aot = mycelium_mlir::run(node, &prims, &engine).map_err(|e| format!("{e:?}"));
    (l0, aot)
}

#[test]
fn seq_get_in_range_l0_interp_equals_aot() {
    // seq.get([t,f,t], 0) == t ; index 2 == t ; index 1 == f.
    for (i, want) in [(0u8, true), (1, false), (2, true)] {
        let node = Node::Op {
            prim: "seq.get".to_owned(),
            args: vec![Node::Const(seq3()), Node::Const(idx8(i))],
        };
        let (l0, aot) = run_l0_and_aot(&node);
        let l0 = l0.unwrap_or_else(|e| panic!("seq.get({i}) L0-interp failed: {e}"));
        let aot = aot.unwrap_or_else(|e| panic!("seq.get({i}) AOT failed: {e}"));
        assert_eq!(
            (l0.repr(), l0.payload()),
            (aot.repr(), aot.payload()),
            "seq.get({i}): L0-interp vs AOT diverged"
        );
        assert_eq!(l0.repr(), &Repr::Binary { width: 1 });
        assert_eq!(l0.payload(), &Payload::Bits(vec![want]));
    }
}

#[test]
fn seq_len_l0_interp_equals_aot() {
    let node = Node::Op {
        prim: "seq.len".to_owned(),
        args: vec![Node::Const(seq3())],
    };
    let (l0, aot) = run_l0_and_aot(&node);
    let l0 = l0.expect("seq.len L0-interp");
    let aot = aot.expect("seq.len AOT");
    assert_eq!(
        (l0.repr(), l0.payload()),
        (aot.repr(), aot.payload()),
        "seq.len: L0-interp vs AOT diverged"
    );
    // 3 as Binary{32}, MSB-first.
    assert_eq!(l0.repr(), &Repr::Binary { width: 32 });
    let want: Vec<bool> = (0..32).rev().map(|k| (3u32 >> k) & 1 == 1).collect();
    assert_eq!(l0.payload(), &Payload::Bits(want));
}

/// Never-silent (G2): an out-of-bounds `seq.get` is an explicit refusal on **both** paths — never a
/// panic, never a silent default. (`len == 3`, so index 3 is out of range.)
#[test]
fn seq_get_out_of_bounds_refuses_on_both_paths() {
    let node = Node::Op {
        prim: "seq.get".to_owned(),
        args: vec![Node::Const(seq3()), Node::Const(idx8(3))],
    };
    let (l0, aot) = run_l0_and_aot(&node);
    assert!(
        l0.is_err(),
        "L0-interp must refuse an out-of-bounds seq.get (never a silent default)"
    );
    assert!(
        aot.is_err(),
        "AOT must refuse an out-of-bounds seq.get (never a silent default)"
    );
}

/// `seq.get`/`seq.len` over a **non-sequence** operand is an explicit type refusal on both paths
/// (never a silent coercion).
#[test]
fn seq_prims_refuse_non_sequence_operand() {
    let get_bad = Node::Op {
        prim: "seq.get".to_owned(),
        args: vec![Node::Const(b1_val(true)), Node::Const(idx8(0))],
    };
    let (l0, aot) = run_l0_and_aot(&get_bad);
    assert!(
        l0.is_err() && aot.is_err(),
        "seq.get on a non-seq must refuse on both paths"
    );

    let len_bad = Node::Op {
        prim: "seq.len".to_owned(),
        args: vec![Node::Const(b1_val(true))],
    };
    let (l0, aot) = run_l0_and_aot(&len_bad);
    assert!(
        l0.is_err() && aot.is_err(),
        "seq.len on a non-seq must refuse on both paths"
    );
}

/// A `Binary{1}` value carrying a **`Declared`** guarantee (a user-asserted, unvalidated bound) — the
/// pre-image for the VR-5 no-upgrade test below.
fn b1_declared(truth: bool) -> Value {
    let bound = Bound {
        kind: BoundKind::Error {
            eps: 0.1,
            norm: NormKind::L2,
        },
        basis: BoundBasis::UserDeclared,
    };
    let meta = Meta::new(
        Provenance::Root,
        GuaranteeStrength::Declared,
        Some(bound),
        None,
        None,
        None,
    )
    .expect("well-formed Declared meta (M-I4)");
    Value::new(Repr::Binary { width: 1 }, Payload::Bits(vec![truth]), meta)
        .expect("well-formed declared bit")
}

/// VR-5 (regression for the pr-review Medium): `seq.get` must return the indexed element at **its
/// own** established basis, never upgraded. A `Declared` element retrieved from an otherwise-`Exact`
/// sequence+index must come back **`Declared`** (carrying its bound), not silently re-stamped `Exact`.
/// Before the fix, `seq.get` propagated the guarantee from the container/index only and dropped the
/// element's `Meta`, yielding an `Exact` result — a silent upgrade past basis.
#[test]
fn seq_get_preserves_a_declared_elements_guarantee() {
    // An Exact-container `Seq<Binary{1}, 2>` whose element 0 is *Declared*, element 1 is Exact.
    let seq = Value::new(
        Repr::Seq {
            elem: Box::new(Repr::Binary { width: 1 }),
            len: 2,
        },
        Payload::Seq(vec![b1_declared(true), b1_val(false)]),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed seq with a declared element");

    // get(seq, 0) → the Declared element: guarantee stays Declared, bound carried, value preserved.
    let node = Node::Op {
        prim: "seq.get".to_owned(),
        args: vec![Node::Const(seq.clone()), Node::Const(idx8(0))],
    };
    let (l0, aot) = run_l0_and_aot(&node);
    let l0 = l0.expect("seq.get(declared elem) L0-interp");
    let aot = aot.expect("seq.get(declared elem) AOT");
    assert_eq!(
        l0.meta().guarantee(),
        GuaranteeStrength::Declared,
        "VR-5: seq.get must NOT upgrade a Declared element to Exact"
    );
    assert!(
        l0.meta().bound().is_some(),
        "the Declared element's bound must carry through, never silently dropped (G2)"
    );
    assert_eq!(l0.payload(), &Payload::Bits(vec![true]), "value preserved");
    assert_eq!(
        l0.meta().guarantee(),
        aot.meta().guarantee(),
        "L0-interp and AOT must agree on the preserved guarantee"
    );

    // get(seq, 1) → the Exact element stays Exact (no spurious downgrade either).
    let node1 = Node::Op {
        prim: "seq.get".to_owned(),
        args: vec![Node::Const(seq), Node::Const(idx8(1))],
    };
    let (l0_1, _) = run_l0_and_aot(&node1);
    assert_eq!(
        l0_1.expect("seq.get(exact elem)").meta().guarantee(),
        GuaranteeStrength::Exact,
        "an Exact element from an Exact container stays Exact"
    );
}

// ── M-750: byte-string prims — prim-level differential (L0-interp ≡ AOT) ─────────────────────────
//
// As with the seq prims, `Repr::Bytes` has no `.myc` surface literal yet (FLAGGED), so we build the
// L0 `Node` tree directly. The reference interpreter and the AOT env-machine dispatch
// `bytes.{len,get,slice,concat}` through the same prim registry, so they must agree — and refuse an
// out-of-range / inverted access identically (never-silent on both paths, G2).

/// A `Repr::Bytes` const value over `bytes`.
fn bytes_val(bytes: Vec<u8>) -> Value {
    Value::new(
        Repr::Bytes,
        Payload::Bytes(bytes),
        Meta::exact(Provenance::Root),
    )
    .expect("well-formed bytes")
}

#[test]
fn bytes_get_and_len_l0_interp_equal_aot() {
    let bytes = bytes_val(vec![0x01, 0x02, 0x03]);

    // bytes.len → Binary{32}(3).
    let len_node = Node::Op {
        prim: "bytes.len".to_owned(),
        args: vec![Node::Const(bytes.clone())],
    };
    let (l0, aot) = run_l0_and_aot(&len_node);
    let l0 = l0.expect("bytes.len L0-interp");
    let aot = aot.expect("bytes.len AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(l0.repr(), &Repr::Binary { width: 32 });

    // bytes.get(b, 1) → Binary{8}(0x02).
    let get_node = Node::Op {
        prim: "bytes.get".to_owned(),
        args: vec![Node::Const(bytes), Node::Const(idx8(1))],
    };
    let (l0, aot) = run_l0_and_aot(&get_node);
    let l0 = l0.expect("bytes.get L0-interp");
    let aot = aot.expect("bytes.get AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(l0.repr(), &Repr::Binary { width: 8 });
    // 0x02 == 0b0000_0010.
    let want: Vec<bool> = (0..8).rev().map(|k| (0x02u8 >> k) & 1 == 1).collect();
    assert_eq!(l0.payload(), &Payload::Bits(want));
}

#[test]
fn bytes_slice_and_concat_l0_interp_equal_aot() {
    let bytes = bytes_val(vec![0x0a, 0x0b, 0x0c, 0x0d]);

    // bytes.slice(b, 1, 3) → Bytes(0x0b 0x0c).
    let slice_node = Node::Op {
        prim: "bytes.slice".to_owned(),
        args: vec![
            Node::Const(bytes.clone()),
            Node::Const(idx8(1)),
            Node::Const(idx8(3)),
        ],
    };
    let (l0, aot) = run_l0_and_aot(&slice_node);
    let l0 = l0.expect("bytes.slice L0-interp");
    let aot = aot.expect("bytes.slice AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(l0.payload(), &Payload::Bytes(vec![0x0b, 0x0c]));

    // bytes.concat(b, b) → 8 bytes.
    let concat_node = Node::Op {
        prim: "bytes.concat".to_owned(),
        args: vec![Node::Const(bytes.clone()), Node::Const(bytes)],
    };
    let (l0, aot) = run_l0_and_aot(&concat_node);
    let l0 = l0.expect("bytes.concat L0-interp");
    let aot = aot.expect("bytes.concat AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(
        l0.payload(),
        &Payload::Bytes(vec![0x0a, 0x0b, 0x0c, 0x0d, 0x0a, 0x0b, 0x0c, 0x0d])
    );
}

/// Never-silent (G2): an out-of-bounds `bytes.get` and an inverted/out-of-range `bytes.slice` are
/// explicit refusals on **both** paths — never a panic, never a silently-clamped result.
#[test]
fn bytes_out_of_range_refuses_on_both_paths() {
    let bytes = bytes_val(vec![0x01, 0x02, 0x03]); // len 3

    // index 3 is out of range.
    let get_oob = Node::Op {
        prim: "bytes.get".to_owned(),
        args: vec![Node::Const(bytes.clone()), Node::Const(idx8(3))],
    };
    let (l0, aot) = run_l0_and_aot(&get_oob);
    assert!(
        l0.is_err() && aot.is_err(),
        "OOB bytes.get must refuse on both paths"
    );

    // slice [2, 1) is inverted; [0, 4) overruns len — both refuse.
    let slice_inv = Node::Op {
        prim: "bytes.slice".to_owned(),
        args: vec![
            Node::Const(bytes.clone()),
            Node::Const(idx8(2)),
            Node::Const(idx8(1)),
        ],
    };
    let (l0, aot) = run_l0_and_aot(&slice_inv);
    assert!(
        l0.is_err() && aot.is_err(),
        "inverted bytes.slice must refuse on both paths"
    );

    let slice_over = Node::Op {
        prim: "bytes.slice".to_owned(),
        args: vec![
            Node::Const(bytes),
            Node::Const(idx8(0)),
            Node::Const(idx8(4)),
        ],
    };
    let (l0, aot) = run_l0_and_aot(&slice_over);
    assert!(
        l0.is_err() && aot.is_err(),
        "out-of-range bytes.slice must refuse on both paths"
    );
}

/// `bytes.*` over a non-bytes operand is an explicit type refusal on both paths.
#[test]
fn bytes_prims_refuse_non_bytes_operand() {
    let len_bad = Node::Op {
        prim: "bytes.len".to_owned(),
        args: vec![Node::Const(b1_val(true))],
    };
    let (l0, aot) = run_l0_and_aot(&len_bad);
    assert!(
        l0.is_err() && aot.is_err(),
        "bytes.len on a non-bytes must refuse on both paths"
    );
}

// ── M-912: `bytes.eq` / `hash.blake3` — prim-level differential (L0-interp ≡ AOT) ────────────────
//
// Like the M-750 group above, these are kernel prims exercised directly as L0 `Node` trees before
// any `.myc` surface existed to name them; the full surface three-way (now that `bytes_eq`/
// `hash_blake3` are surface-callable) follows below with the rest of the D3/D4 surface tests.

#[test]
fn bytes_eq_l0_interp_equal_aot() {
    let a = bytes_val(vec![0x01, 0x02, 0x03]);
    let a2 = bytes_val(vec![0x01, 0x02, 0x03]);
    let b = bytes_val(vec![0x01, 0x02, 0x04]);

    // Equal byte strings -> true, on both paths.
    let eq_node = Node::Op {
        prim: "bytes.eq".to_owned(),
        args: vec![Node::Const(a.clone()), Node::Const(a2)],
    };
    let (l0, aot) = run_l0_and_aot(&eq_node);
    let l0 = l0.expect("bytes.eq(equal) L0-interp");
    let aot = aot.expect("bytes.eq(equal) AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(l0.repr(), &Repr::Binary { width: 1 });
    assert_eq!(l0.payload(), &Payload::Bits(vec![true]));

    // Unequal byte strings -> false, on both paths.
    let neq_node = Node::Op {
        prim: "bytes.eq".to_owned(),
        args: vec![Node::Const(a), Node::Const(b)],
    };
    let (l0, aot) = run_l0_and_aot(&neq_node);
    let l0 = l0.expect("bytes.eq(unequal) L0-interp");
    let aot = aot.expect("bytes.eq(unequal) AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(l0.payload(), &Payload::Bits(vec![false]));
}

/// `bytes.eq` over a non-bytes operand is an explicit type refusal on both paths.
#[test]
fn bytes_eq_refuses_non_bytes_operand() {
    let bad = Node::Op {
        prim: "bytes.eq".to_owned(),
        args: vec![
            Node::Const(b1_val(true)),
            Node::Const(bytes_val(vec![0x01])),
        ],
    };
    let (l0, aot) = run_l0_and_aot(&bad);
    assert!(
        l0.is_err() && aot.is_err(),
        "bytes.eq on a non-bytes operand must refuse on both paths"
    );
}

/// `hash.blake3` reproduces the official BLAKE3 empty-input test vector on both paths — the
/// prim-level twin of the known-digest conformance in `mycelium-interp` (M-912).
#[test]
fn hash_blake3_l0_interp_equal_aot() {
    let node = Node::Op {
        prim: "hash.blake3".to_owned(),
        args: vec![Node::Const(bytes_val(vec![]))],
    };
    let (l0, aot) = run_l0_and_aot(&node);
    let l0 = l0.expect("hash.blake3 L0-interp");
    let aot = aot.expect("hash.blake3 AOT");
    assert_eq!((l0.repr(), l0.payload()), (aot.repr(), aot.payload()));
    assert_eq!(l0.repr(), &Repr::Bytes);
    // The official BLAKE3 empty-input digest (github.com/BLAKE3-team/BLAKE3 test vectors).
    let expected: Vec<u8> = [
        0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9,
        0x49, 0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f,
        0x32, 0x62,
    ]
    .to_vec();
    assert_eq!(expected.len(), 32, "BLAKE3 digest is 32 bytes");
    assert_eq!(l0.payload(), &Payload::Bytes(expected));
}

/// `hash.blake3` over a non-bytes operand is an explicit type refusal on both paths.
#[test]
fn hash_blake3_refuses_non_bytes_operand() {
    let bad = Node::Op {
        prim: "hash.blake3".to_owned(),
        args: vec![Node::Const(b1_val(true))],
    };
    let (l0, aot) = run_l0_and_aot(&bad);
    assert!(
        l0.is_err() && aot.is_err(),
        "hash.blake3 on a non-bytes operand must refuse on both paths"
    );
}

// ── M-749 surface: Seq{T,N} / `[..]` literal — full three-way differential (RFC-0032 D3) ─────────
//
// Now that the `.myc` surface exists (the `Seq{T, N}` type, the `[e1, …]` list literal, and the
// `seq_get`/`seq_len` prims), the seq enabler runs the **full three-way** (L1-eval ≡
// elaborate→L0-interp ≡ AOT) differential over a parsed `.myc` program — not only the prim-level
// L0≡AOT layer above. `assert_three_way` checks all three paths agree AND equal the reference.

/// The `Binary{32}` MSB-first encoding of `n` (the `seq_len`/`bytes_len` result shape).
fn b32(n: u32) -> (Repr, Payload) {
    let bits: Vec<bool> = (0..32).rev().map(|k| (n >> k) & 1 == 1).collect();
    (Repr::Binary { width: 32 }, Payload::Bits(bits))
}

/// `[0b1, 0b0, 0b1]` ascribed to `Seq{Binary{1}, 3}` round-trips as a `Repr::Seq` value on all three
/// paths — the first end-to-end proof the surface list literal builds a kernel sequence.
#[test]
fn seq_literal_surface_three_way() {
    let expected_repr = Repr::Seq {
        elem: Box::new(Repr::Binary { width: 1 }),
        len: 3,
    };
    let expected_payload = Payload::Seq(vec![b1_val(true), b1_val(false), b1_val(true)]);
    assert_three_way(
        "seq literal [0b1,0b0,0b1]",
        "nodule d;\nfn main() => Seq{Binary{1}, 3} = [0b1, 0b0, 0b1];",
        &expected_repr,
        &expected_payload,
    );
}

/// `seq_get([0b1,0b0,0b1], i)` over the surface agrees on all three paths for each in-range index.
/// The index is written as an explicit 8-bit `Binary{8}` literal (MSB-first).
#[test]
fn seq_get_surface_three_way() {
    for (i, want) in [(0u8, true), (1, false), (2, true)] {
        let (r, p) = b1(want);
        let src =
            format!("nodule d;\nfn main() => Binary{{1}} = seq_get([0b1, 0b0, 0b1], 0b{i:08b});");
        assert_three_way(&format!("seq_get index {i}"), &src, &r, &p);
    }
}

/// `seq_len([0b1,0b0,0b1])` over the surface is `Binary{32}(3)` on all three paths.
#[test]
fn seq_len_surface_three_way() {
    let (r, p) = b32(3);
    assert_three_way(
        "seq_len",
        "nodule d;\nfn main() => Binary{32} = seq_len([0b1, 0b0, 0b1]);",
        &r,
        &p,
    );
}

/// Never-silent (G2): a **heterogeneous** list literal is a static check refusal — the elements must
/// be homogeneous, never silently coerced (RFC-0032 D3). `0b1` is `Binary{1}`, `0b00` is `Binary{2}`.
#[test]
fn seq_heterogeneous_elements_reject() {
    let src = "nodule d;\nfn main() => Seq{Binary{1}, 2} = [0b1, 0b00];";
    let err = check_nodule(&parse(src).expect("parses"))
        .expect_err("a heterogeneous list literal must be a static check error, never a coercion");
    assert!(
        err.to_string().contains("homogeneous"),
        "the refusal must name the homogeneity cause (never-silent): {err}"
    );
}

/// Never-silent (G2): a list literal whose count disagrees with the ascribed `Seq{T, N}` length is a
/// static refusal — never a silent truncation/padding (RFC-0032 D3).
#[test]
fn seq_length_mismatch_reject() {
    let src = "nodule d;\nfn main() => Seq{Binary{1}, 5} = [0b1, 0b0, 0b1];";
    let err = check_nodule(&parse(src).expect("parses"))
        .expect_err("a list-length vs Seq{N} mismatch must be a static check error");
    assert!(
        err.to_string().contains("expected `Seq` length"),
        "the refusal must name the length-mismatch cause (never-silent): {err}"
    );
}

// ── M-750 surface: Bytes / `0x..` literal — full three-way differential (RFC-0032 D4) ────────────

/// `0x48_65_6c_6c_6f` ("Hello") round-trips as a `Repr::Bytes` value on all three paths.
#[test]
fn bytes_literal_surface_three_way() {
    let expected_payload = Payload::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
    assert_three_way(
        "bytes literal 0x48_65_6c_6c_6f",
        "nodule d;\nfn main() => Bytes = 0x48_65_6c_6c_6f;",
        &Repr::Bytes,
        &expected_payload,
    );
}

/// `bytes_get(0x_…, i)` over the surface is the indexed byte (`Binary{8}`) on all three paths.
#[test]
fn bytes_get_surface_three_way() {
    // 0x01_02_03, index 1 → 0x02 == 0b0000_0010.
    let want: Vec<bool> = (0..8).rev().map(|k| (0x02u8 >> k) & 1 == 1).collect();
    assert_three_way(
        "bytes_get index 1",
        "nodule d;\nfn main() => Binary{8} = bytes_get(0x01_02_03, 0b0000_0001);",
        &Repr::Binary { width: 8 },
        &Payload::Bits(want),
    );
}

/// `bytes_len(0x01_02_03)` over the surface is `Binary{32}(3)` on all three paths.
#[test]
fn bytes_len_surface_three_way() {
    let (r, p) = b32(3);
    assert_three_way(
        "bytes_len",
        "nodule d;\nfn main() => Binary{32} = bytes_len(0x01_02_03);",
        &r,
        &p,
    );
}

// ── M-912 surface: `bytes_eq`/`hash_blake3` — full three-way differential ────────────────────────

/// `bytes_eq(0x01_02_03, 0x01_02_03)` over the surface is `true` on all three paths.
#[test]
fn bytes_eq_surface_three_way_equal() {
    let (r, p) = b1(true);
    assert_three_way(
        "bytes_eq equal",
        "nodule d;\nfn main() => Binary{1} = bytes_eq(0x01_02_03, 0x01_02_03);",
        &r,
        &p,
    );
}

/// `bytes_eq(0x01_02_03, 0x01_02_04)` over the surface is `false` on all three paths.
#[test]
fn bytes_eq_surface_three_way_unequal() {
    let (r, p) = b1(false);
    assert_three_way(
        "bytes_eq unequal",
        "nodule d;\nfn main() => Binary{1} = bytes_eq(0x01_02_03, 0x01_02_04);",
        &r,
        &p,
    );
}

/// `bytes_eq` over different-length byte strings is `false` on all three paths (a length-sensitive
/// comparison, never a prefix match).
#[test]
fn bytes_eq_surface_three_way_different_length() {
    let (r, p) = b1(false);
    assert_three_way(
        "bytes_eq different length",
        "nodule d;\nfn main() => Binary{1} = bytes_eq(0x01_02, 0x01_02_03);",
        &r,
        &p,
    );
}

/// `hash_blake3("")` over the surface is the official BLAKE3 empty-input digest on all three paths
/// — the surface twin of the known-digest prim-level test above.
#[test]
fn hash_blake3_surface_three_way_empty() {
    let expected_payload = Payload::Bytes(vec![
        0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9,
        0x49, 0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f,
        0x32, 0x62,
    ]);
    assert_three_way(
        "hash_blake3 empty string",
        "nodule d;\nfn main() => Bytes = hash_blake3(\"\");",
        &Repr::Bytes,
        &expected_payload,
    );
}

/// `hash_blake3(0x00_01_02)` over the surface matches the official BLAKE3 length-3 test vector
/// (the `0, 1, 2, …` repeating-sequence input rule) on all three paths.
#[test]
fn hash_blake3_surface_three_way_len3() {
    let expected_payload = Payload::Bytes(vec![
        0xe1, 0xbe, 0x4d, 0x7a, 0x8a, 0xb5, 0x56, 0x0a, 0xa4, 0x19, 0x9e, 0xea, 0x33, 0x98, 0x49,
        0xba, 0x8e, 0x29, 0x3d, 0x55, 0xca, 0x0a, 0x81, 0x00, 0x67, 0x26, 0xd1, 0x84, 0x51, 0x9e,
        0x64, 0x7f,
    ]);
    assert_three_way(
        "hash_blake3 0x00_01_02 (BLAKE3 test vector, len 3)",
        "nodule d;\nfn main() => Bytes = hash_blake3(0x00_01_02);",
        &Repr::Bytes,
        &expected_payload,
    );
}

/// Never-silent (G2): an **odd-hex** `0x…` literal is a lex/parse refusal — a byte is two hex chars,
/// never a silent half-byte (RFC-0032 D4). `0x123` has three hex digits.
#[test]
fn bytes_odd_hex_reject() {
    let src = "nodule d\nfn main() => Bytes = 0x123";
    let err = parse(src).expect_err("an odd-hex `0x…` literal must be a parse error");
    assert!(
        err.to_string().contains("odd hex-digit count"),
        "the refusal must name the odd-hex cause, never a silent half-byte: {err}"
    );
}

/// Never-silent (G2): an empty `0x` (no hex digits) is a lex/parse refusal.
#[test]
fn bytes_empty_hex_reject() {
    let src = "nodule d\nfn main() => Bytes = 0x";
    let err = parse(src).expect_err("an empty `0x` literal must be a parse error");
    assert!(
        err.to_string().contains("no hex digits"),
        "the refusal must name the empty-hex cause: {err}"
    );
}

// ── M-910/M-911 surface: textual string literal `"…"` — full three-way differential ──────────────
//
// `"…"` lowers to the SAME `Repr::Bytes`/`Payload::Bytes` value form as the `0x…` literal above
// (UTF-8-encoded; KC-3 — no new L0 node), so it types as `Bytes` and is a legal operand to the
// SAME `bytes_get`/`bytes_len` prims exercised above. The escape set is explicit and minimal:
// `\n \t \\ \" \0 \r` (ergonomic, not expressive — `\xNN` is deliberately not included, see the
// lexer's `lex_string` doc). Escape/termination errors are the lexer's never-silent gate (G2).

/// `"Hello"` round-trips as the identical `Repr::Bytes` value the `0x48_65_6c_6c_6f` literal
/// produces above — the direct evidence that the two literal forms share one value form.
#[test]
fn string_literal_surface_three_way() {
    let expected_payload = Payload::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
    assert_three_way(
        "string literal \"Hello\"",
        "nodule d;\nfn main() => Bytes = \"Hello\";",
        &Repr::Bytes,
        &expected_payload,
    );
}

/// The empty string literal `""` is a legal, zero-length `Bytes` value on all three paths.
#[test]
fn string_literal_empty_surface_three_way() {
    assert_three_way(
        "empty string literal",
        "nodule d;\nfn main() => Bytes = \"\";",
        &Repr::Bytes,
        &Payload::Bytes(vec![]),
    );
}

/// Every escape in the minimal set decodes to its target byte in the elaborated value, on all
/// three paths — `"\n\t\\\"\0\r"` is the 6-byte sequence `0A 09 5C 22 00 0D`.
#[test]
fn string_literal_escape_set_surface_three_way() {
    assert_three_way(
        "string literal escape set",
        "nodule d;\nfn main() => Bytes = \"\\n\\t\\\\\\\"\\0\\r\";",
        &Repr::Bytes,
        &Payload::Bytes(vec![0x0A, 0x09, 0x5C, 0x22, 0x00, 0x0D]),
    );
}

/// `bytes_get("Hello", i)` over the surface is the indexed byte on all three paths — proof that a
/// string literal is a legal operand to the existing `Bytes` prims (RFC-0032 D4), not a distinct
/// surface type.
#[test]
fn string_literal_bytes_get_surface_three_way() {
    // "Hello"[1] == 'e' == 0x65.
    let want: Vec<bool> = (0..8).rev().map(|k| (0x65u8 >> k) & 1 == 1).collect();
    assert_three_way(
        "bytes_get over a string literal, index 1",
        "nodule d;\nfn main() => Binary{8} = bytes_get(\"Hello\", 0b0000_0001);",
        &Repr::Binary { width: 8 },
        &Payload::Bits(want),
    );
}

/// `bytes_len("Hello")` over the surface is `Binary{32}(5)` on all three paths.
#[test]
fn string_literal_bytes_len_surface_three_way() {
    let (r, p) = b32(5);
    assert_three_way(
        "bytes_len over a string literal",
        "nodule d;\nfn main() => Binary{32} = bytes_len(\"Hello\");",
        &r,
        &p,
    );
}

/// Never-silent (G2): an unterminated string literal (no closing `"` before EOF) is a lex/parse
/// refusal — never a silent truncation.
#[test]
fn string_unterminated_reject() {
    let src = "nodule d\nfn main() => Bytes = \"abc";
    let err = parse(src).expect_err("an unterminated string literal must be a parse error");
    assert!(
        err.to_string().contains("unterminated"),
        "the refusal must name the unterminated cause: {err}"
    );
}

/// Never-silent (G2): an unknown escape sequence (`\q`) is a lex/parse refusal — never a silently
/// dropped backslash or a silently-literal escape char.
#[test]
fn string_unknown_escape_reject() {
    let src = "nodule d\nfn main() => Bytes = \"a\\qb\"";
    let err = parse(src).expect_err("an unknown escape sequence must be a parse error");
    assert!(
        err.to_string().contains("unknown escape"),
        "the refusal must name the unknown-escape cause: {err}"
    );
}

/// Never-silent (G2): a raw newline inside `"…"` is a lex/parse refusal — a multi-line string is
/// not part of the minimal surface (use `\n`).
#[test]
fn string_raw_newline_reject() {
    let src = "nodule d\nfn main() => Bytes = \"a\nb\"";
    let err = parse(src).expect_err("a raw newline inside a string literal must be a parse error");
    assert!(
        err.to_string().contains("unterminated"),
        "the refusal must name the unterminated (raw-newline) cause: {err}"
    );
}

// ── M-897 (ADR-040, `enb` Gap A): the decimal float literal — full three-way differential ───────
//
// `1.5` lowers to the EXISTING `Repr::Float{width: F64}`/`Payload::Float` scalar value form landed
// by M-896 (KC-3 — no new L0 node); the nullary `Float` type keyword names it (binary64 only at
// introduction — ADR-040 FLAG-1). The literal denotes the **correctly-rounded** (RNE) binary64 of
// its decimal text (FLAG-3 — the documented, EXPLAIN-able conversion posture); the conversion runs
// once, at elaboration, via `f64::from_str`. Honesty tags: the denotation is `Exact` as a
// definition (ADR-040 §2.6); the host-conversion claim ("`from_str` is correctly rounded") is
// `Declared` (Rust-std) pinned `Empirical` here by the bit-exact round-trip corpus against rustc's
// own compile-time decimal→binary64 conversion — two independent implementations of the same
// IEEE-754 conversion agreeing bit-for-bit. Never-silent (G2): form, empty-exponent, and
// out-of-range (rounds-to-±inf) errors are the lexer's; the float-pattern refusal is the
// checker's (FLAG-4). **AOT closure (recorded honestly):** the three-way — L1-eval ≡
// elaborate→L0-interp ≡ AOT — closes over nullary `main` programs returning a float value; all
// three paths run below (no refusal to record).

/// The `Float` value observable, bit-exact: repr is `Float{F64}` and the payload carries exactly
/// the expected bits (payload `==` would pass `-0.0 == 0.0`; bits do not — the ADR-040 §2.3
/// identity posture).
#[track_caller]
fn assert_float_three_way_bits(label: &str, src: &str, expected: f64) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();
    let node =
        elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: must be in the fragment: {e}"));
    let l0 = interp
        .eval(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot = mycelium_mlir::run(&node, &prims, &engine)
        .unwrap_or_else(|e| panic!("{label}: AOT failed: {e}"));

    for (path, v) in [("L1-eval", &l1), ("L0-interp", &l0), ("AOT", &aot)] {
        assert_eq!(
            v.repr(),
            &Repr::Float {
                width: FloatWidth::F64
            },
            "{label}: {path} repr mismatch"
        );
        let Payload::Float(x) = v.payload() else {
            panic!("{label}: {path} payload is not Float: {:?}", v.payload());
        };
        assert_eq!(
            x.to_bits(),
            expected.to_bits(),
            "{label}: {path} bits mismatch (got {x:?}, want {expected:?})"
        );
    }
}

/// `1.5` (exactly representable in binary64) round-trips identically on all three paths.
#[test]
fn float_literal_surface_three_way() {
    assert_float_three_way_bits(
        "float literal 1.5",
        "nodule d;\nfn main() => Float = 1.5;",
        1.5,
    );
}

/// `0.0` is positive zero — bit-exactly (`+0.0`/`-0.0` are distinct identities, ADR-040 §2.3;
/// a payload `==` check alone could not see the difference).
#[test]
fn float_literal_zero_three_way() {
    assert_float_three_way_bits(
        "float literal 0.0",
        "nodule d;\nfn main() => Float = 0.0;",
        0.0,
    );
}

/// The exponent forms: integer-mantissa `1e10`, fractional `2.5e-3`, and uppercase `1E+5`.
#[test]
fn float_literal_exponent_forms_three_way() {
    assert_float_three_way_bits(
        "float literal 1e10",
        "nodule d;\nfn main() => Float = 1e10;",
        1e10,
    );
    assert_float_three_way_bits(
        "float literal 2.5e-3",
        "nodule d;\nfn main() => Float = 2.5e-3;",
        2.5e-3,
    );
    assert_float_three_way_bits(
        "float literal 1E+5",
        "nodule d;\nfn main() => Float = 1E+5;",
        1E+5,
    );
}

/// `0.1` is NOT exactly representable — the literal denotes its correctly-rounded binary64
/// (FLAG-3), which is exactly what rustc's `0.1` denotes too; both conversions agree bit-for-bit.
#[test]
fn float_literal_inexact_decimal_three_way() {
    assert_float_three_way_bits(
        "float literal 0.1 (correctly rounded)",
        "nodule d;\nfn main() => Float = 0.1;",
        0.1,
    );
}

/// The `Float` type annotation flows through params and returns: a float literal is a legal
/// argument to a `Float -> Float` function on all three paths.
#[test]
fn float_type_annotation_param_return_three_way() {
    assert_float_three_way_bits(
        "Float param/return",
        "nodule d;\nfn id(x: Float) => Float = x;\nfn main() => Float = id(2.5);",
        2.5,
    );
}

/// **The round-trip property, on a corpus (Empirical):** for each reference value `v`, rendering
/// its shortest decimal form (`{v:?}` — Rust's shortest round-trip render) and running that text
/// through the full surface pipeline (lex → parse → check → L1-eval, and elaborate → L0-interp)
/// reproduces `v` **bit-for-bit**. This is decimal→binary64→render→binary64 closure, and — since
/// the pipeline converts via `f64::from_str` while the reference bits come from rustc's own
/// compile-time conversion — a two-implementation differential of the correctly-rounded
/// conversion (VR-5: the FLAG-3 claim is tested, not asserted). Corpus rows pin the boundary
/// cases: exact/inexact decimals, exponent extremes, `f64::MAX` (largest finite), subnormals down
/// to `5e-324` (smallest positive), and the 2^53 exact-integer edge.
#[test]
// The near-MAX / deep-subnormal rows below carry their full shortest-round-trip digit strings on
// purpose (they pin conversion at the representability boundaries); trimming the "excessive"
// digits would change which binary64 the row denotes.
#[allow(clippy::excessive_precision)]
fn float_literal_round_trip_corpus() {
    let corpus: &[f64] = &[
        0.0,
        1.0,
        1.5,
        0.1,
        0.2,
        1.0 / 3.0,
        2.5e-3,
        1e10,
        std::f64::consts::PI,
        std::f64::consts::E,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,                  // smallest positive subnormal
        9007199254740992.0,      // 2^53 — the exact-integer representability edge
        1.7976931348623155e308,  // one ULP below f64::MAX (…157e308)
        4.9406564584124654e-321, // a deep subnormal
    ];
    for &v in corpus {
        let text = format!("{v:?}");
        let src = format!("nodule d;\nfn main() => Float = {text};");
        assert_float_three_way_bits(&format!("round-trip {text}"), &src, v);
    }
}

/// Never-silent (G2): an exponent with no digits (`1e`) is an explicit lex/parse refusal naming
/// the cause — never a silent `Int` + identifier split.
#[test]
fn float_exponent_no_digits_reject() {
    let src = "nodule d;\nfn main() => Float = 1e;";
    let err = parse(src).expect_err("an exponent with no digits must be a parse error");
    assert!(
        err.to_string().contains("exponent with no digits"),
        "the refusal must name the empty-exponent cause: {err}"
    );
}

/// Never-silent (G2, ADR-040 §2.4): a literal whose correctly-rounded binary64 value is not
/// finite (`1e999`) is an explicit out-of-range refusal — a literal is a conversion boundary; it
/// never silently lands on ±inf (in-band IEEE specials arise only from arithmetic).
#[test]
fn float_out_of_range_reject() {
    let src = "nodule d;\nfn main() => Float = 1e999;";
    let err = parse(src).expect_err("a literal rounding to +inf must be a parse error");
    assert!(
        err.to_string().contains("float literal out of range"),
        "the refusal must name the out-of-range cause: {err}"
    );
}

/// The Int-disambiguation boundary, pinned at the surface: `1.` is NOT a float (no digit after
/// the dot — `.` stays the path glyph), so the trailing dot is an explicit parse refusal; and a
/// leading-dot `.5` never opens a number.
#[test]
fn float_trailing_and_leading_dot_reject() {
    parse("nodule d;\nfn main() => Float = 1.;")
        .expect_err("`1.` must not parse as a float literal (Int `1` + a dangling `.`)");
    parse("nodule d;\nfn main() => Float = .5;")
        .expect_err("`.5` must not parse as a float literal (no leading-dot form)");
}

/// Never-silent type discipline: a float literal where a `Binary{8}` is expected is an explicit
/// check refusal naming both types — never a silent conversion (S1/G2).
#[test]
fn float_type_mismatch_reject() {
    let src = "nodule d;\nfn main() => Binary{8} = 1.5;";
    let env = parse(src).expect("parses");
    let err = check_nodule(&env).expect_err("Float where Binary{8} is expected must be refused");
    assert!(
        err.to_string().contains("Float"),
        "the refusal must name the Float type: {err}"
    );
}

/// ADR-040 FLAG-4: floats cannot be matched by literal patterns — IEEE `==` and content identity
/// diverge on floats (`-0.0`/NaN), so a literal-pattern arm would have to silently pick one
/// semantic. Pinned at the first gate that fires: the checker's scrutinee rule refuses `match`
/// over a `Float` outright (explicit, names the type). A second, defense-in-depth refusal sits in
/// `normalize_pattern` (naming ADR-040 FLAG-4) should a float scrutinee ever become matchable.
#[test]
fn float_pattern_reject() {
    let src = "nodule d;\nfn f(x: Float) => Float = match x { 1.5 => x, _ => x };\nfn main() => Float = f(0.0);";
    let env = parse(src).expect("parses");
    let err = check_nodule(&env).expect_err("a match over a Float scrutinee must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("match scrutinee") && msg.contains("Float"),
        "the refusal must name the scrutinee rule and the Float type: {err}"
    );
}

// ── M-898 (ADR-040 §2.5, `enb` Gap A): the scalar-float arithmetic prims ────────────────────────
//
// `flt_add`/`flt_sub`/`flt_mul`/`flt_div`/`flt_neg` (kernel `flt.add`/`flt.sub`/`flt.mul`/
// `flt.div`/`flt.neg`) — IEEE-754 binary64 arithmetic under **round-to-nearest-even only**
// (rounding is a property of the operation, never hidden state — ADR-040 §2.2, the ADR-028
// parallel). The never-silent contract has two distinct halves (G2):
//   - **static** — every operand must be exactly `Float` (a non-`Float` operand, a bare decimal,
//     and a wrong arity are explicit check-time refusals — `flt_prims_conformance_reject`);
//   - - none at runtime by design — the ops are **total** over `Float`: arithmetic specials are
//     **in-band, inspectable, propagating values** per the ratified ADR-040 §2.4 FLAG-2
//     (overflow → ±inf, `x/0` → ±inf with the IEEE sign rule, `0/0` → canonical NaN), pinned
//     three-way by the `flt_arith_specials_*` tests — never a trap, never a silent alias of an
//     ordinary value (the in-band sentinel IS the signal; contrast integer `div_u`, which has
//     no sentinel and must refuse).
// Per-op tag: **`Empirical`** per the ratified ADR-040 §2.6 — the correctly-rounded-RNE
// *definition* is the spec (`Exact` as a definition), the host-delivers-those-bits
// *implementation claim* is `Empirical` (pinned by the 40-case hand-derived IEEE reference corpus
// in `mycelium-interp/src/tests/prims.rs`), the platform IEEE statement stays `Declared`; no
// `Proven` anywhere. The disclosed bound is zero-deviation-vs-spec (`eps = 0`, `Linf`,
// `EmpiricalFit`), EXPLAIN-able off the value — checked on every path below.
//
// **Where the three-way closes (recorded honestly — G2/VR-5).** M-897's float literal makes a
// *nullary* `main` over float values expressible, so the **full surface three-way**
// (L1-eval ≡ elaborate→L0-interp ≡ AOT over `assert_float_three_way_bits`) **closes** for the
// whole group — including the in-band specials — with no refusal to record (the AOT env-machine
// dispatches `Op` nodes through the same trusted `PrimRegistry`). This is the Gap-A closure the
// dense group's section note anticipated.

/// Like [`assert_float_three_way_bits`], and additionally asserts the ADR-040 §2.6 tag contract
/// on **every** path: guarantee `Empirical`, bound `eps = 0`/`Linf` on an `EmpiricalFit` basis
/// (the zero-deviation-vs-spec claim, EXPLAIN-able off the value — G2/SC-3).
#[track_caller]
fn assert_flt_three_way_with_tag(label: &str, src: &str, expected: f64) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();
    let node =
        elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: must be in the fragment: {e}"));
    let l0 = interp
        .eval(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot = mycelium_mlir::run(
        &node,
        &PrimRegistry::with_builtins(),
        &mycelium_cert::BinaryTernarySwapEngine,
    )
    .unwrap_or_else(|e| panic!("{label}: AOT failed: {e}"));
    for (path, v) in [("L1-eval", &l1), ("L0-interp", &l0), ("AOT", &aot)] {
        let Payload::Float(x) = v.payload() else {
            panic!("{label}: {path} payload is not Float: {:?}", v.payload());
        };
        assert_eq!(
            x.to_bits(),
            expected.to_bits(),
            "{label}: {path} bits mismatch (got {x:?}, want {expected:?})"
        );
        assert_eq!(
            v.meta().guarantee(),
            GuaranteeStrength::Empirical,
            "{label}: {path} must carry the ratified ADR-040 §2.6 Empirical tag (VR-5)"
        );
        match v.meta().bound() {
            Some(Bound {
                kind: BoundKind::Error { eps, norm },
                basis: BoundBasis::EmpiricalFit { trials, .. },
            }) => {
                assert_eq!(*eps, 0.0, "{label}: {path} zero-deviation-vs-spec bound");
                assert_eq!(*norm, NormKind::Linf);
                assert!(*trials >= 1, "an Empirical basis is never evidence-free");
            }
            other => panic!("{label}: {path} expected the EmpiricalFit bound, got {other:?}"),
        }
    }
}

/// The nullary-main surface three-way closes for each arithmetic op over exact dyadic operands
/// (bit-exact reference results), and the ADR-040 §2.6 tag rides every path.
#[test]
fn flt_arith_ops_three_way() {
    assert_flt_three_way_with_tag(
        "flt_add",
        "nodule d;\nfn main() => Float = flt_add(1.5, 2.25);",
        3.75,
    );
    assert_flt_three_way_with_tag(
        "flt_sub",
        "nodule d;\nfn main() => Float = flt_sub(3.75, 1.5);",
        2.25,
    );
    assert_flt_three_way_with_tag(
        "flt_mul",
        "nodule d;\nfn main() => Float = flt_mul(1.5, 2.0);",
        3.0,
    );
    assert_flt_three_way_with_tag(
        "flt_div",
        "nodule d;\nfn main() => Float = flt_div(3.0, 2.0);",
        1.5,
    );
    assert_flt_three_way_with_tag(
        "flt_neg",
        "nodule d;\nfn main() => Float = flt_neg(1.5);",
        -1.5,
    );
}

/// RNE is observable at the surface: `0.1 + 0.2` is the correctly-rounded binary64
/// `0.30000000000000004` (not `0.3`) on all three paths — the canonical rounding witness.
#[test]
fn flt_arith_rne_rounding_three_way() {
    assert_flt_three_way_with_tag(
        "flt_add rounds RNE",
        "nodule d;\nfn main() => Float = flt_add(0.1, 0.2);",
        0.300_000_000_000_000_04,
    );
}

/// Chained float arithmetic composes (an `Empirical` intermediate is a legal operand):
/// `(1.5 × 2.0) + 0.25 = 3.25`, bit-exact on all three paths.
#[test]
fn flt_arith_composition_three_way() {
    assert_flt_three_way_with_tag(
        "flt composition",
        "nodule d;\nfn main() => Float = flt_add(flt_mul(1.5, 2.0), 0.25);",
        3.25,
    );
}

/// `Float` params/returns flow through functions: the ops accept function-bound `Float` values.
#[test]
fn flt_arith_through_functions_three_way() {
    assert_flt_three_way_with_tag(
        "flt through fn",
        "nodule d;\nfn scale2(x: Float) => Float = flt_mul(x, 2.0);\nfn main() => Float = scale2(2.25);",
        4.5,
    );
}

/// **In-band specials (the ratified ADR-040 FLAG-2), three-way:** div-by-zero → ±inf with the
/// IEEE sign rule, `0/0` → the canonical NaN, and overflow → +inf — **values on every path**,
/// never a trap/refusal and never a silent alias of an ordinary number (the sentinel is the
/// never-silent signal; every path agrees bit-for-bit, NaN included, because NaN is canonical).
#[test]
fn flt_arith_specials_are_in_band_three_way() {
    assert_flt_three_way_with_tag(
        "1/0 → +inf",
        "nodule d;\nfn main() => Float = flt_div(1.0, 0.0);",
        f64::INFINITY,
    );
    assert_flt_three_way_with_tag(
        "-1/0 → -inf",
        "nodule d;\nfn main() => Float = flt_div(flt_neg(1.0), 0.0);",
        f64::NEG_INFINITY,
    );
    assert_flt_three_way_with_tag(
        "1/-0 → -inf (signed zero is observable)",
        "nodule d;\nfn main() => Float = flt_div(1.0, flt_neg(0.0));",
        f64::NEG_INFINITY,
    );
    assert_flt_three_way_with_tag(
        "0/0 → canonical NaN",
        "nodule d;\nfn main() => Float = flt_div(0.0, 0.0);",
        f64::from_bits(mycelium_core::CANONICAL_NAN_BITS),
    );
    // Overflow: MAX + MAX → +inf, in-band (f64::MAX's shortest round-trip literal).
    assert_flt_three_way_with_tag(
        "overflow → +inf",
        "nodule d;\nfn main() => Float = flt_add(1.7976931348623157e308, 1.7976931348623157e308);",
        f64::INFINITY,
    );
    // The signed-zero identity: neg(+0) is −0, bit-distinct (ADR-040 §2.3 — a payload `==`
    // could not see this; the bit assertion can).
    assert_flt_three_way_with_tag(
        "neg(0.0) → -0.0 bit-exactly",
        "nodule d;\nfn main() => Float = flt_neg(0.0);",
        -0.0,
    );
}

/// **NaN propagation through arithmetic, end-to-end three-way (M-900 closeout addition).**
/// `flt_arith_specials_are_in_band_three_way` above pins the canonical NaN as a *produced*
/// value (`0/0`); this pins it as a *propagated* value — a NaN operand infects every arithmetic
/// op and every result stays the SAME canonical bits (ADR-040 §2.3: one NaN, one identity, never
/// re-derived or drifted) on all three paths. Closes the last un-covered "specials in-band"
/// corner of the M-900 comprehensiveness sweep (the `mycelium-interp` prim-level corpus already
/// covers this per-op; this is the surface-program three-way closure over it).
#[test]
fn flt_arith_nan_propagates_and_recanonicalizes_three_way() {
    let cnan = f64::from_bits(mycelium_core::CANONICAL_NAN_BITS);
    // NaN, constructed in-language via the in-band 0/0 special, infects flt_add/flt_mul/flt_sub.
    assert_flt_three_way_with_tag(
        "NaN + 1.0 → canonical NaN (propagation)",
        "nodule d;\nfn main() => Float = flt_add(flt_div(0.0, 0.0), 1.0);",
        cnan,
    );
    assert_flt_three_way_with_tag(
        "1.0 * NaN → canonical NaN (propagation, either operand side)",
        "nodule d;\nfn main() => Float = flt_mul(1.0, flt_div(0.0, 0.0));",
        cnan,
    );
    assert_flt_three_way_with_tag(
        "NaN - NaN → canonical NaN (not 0 — NaN algebra, not ordinary subtraction)",
        "nodule d;\nfn main() => Float = flt_sub(flt_div(0.0, 0.0), flt_div(0.0, 0.0));",
        cnan,
    );
    // neg(NaN) re-canonicalizes: sign/payload bits are never observable (ADR-040 §2.3) — the
    // result is bit-identical to the canonical NaN, not a sign-flipped variant.
    assert_flt_three_way_with_tag(
        "neg(NaN) re-canonicalizes to the SAME canonical bits",
        "nodule d;\nfn main() => Float = flt_neg(flt_div(0.0, 0.0));",
        cnan,
    );
}

/// Static conformance — accept: every float-prim signature the checker must admit.
#[test]
fn flt_prims_conformance_accept() {
    for src in [
        "nodule d;\nfn f(a: Float, b: Float) => Float = flt_add(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Float = flt_sub(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Float = flt_mul(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Float = flt_div(a, b);",
        "nodule d;\nfn f(a: Float) => Float = flt_neg(a);",
        // Literal operands (M-897) and composition (dim-free: Float is nullary).
        "nodule d;\nfn main() => Float = flt_add(1.5, 2.5e-3);",
        "nodule d;\nfn f(a: Float) => Float = flt_neg(flt_mul(a, 2.0));",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent operand/arity contract is a *check-time*
/// refusal with a message naming the offense (G2). A bare decimal has no `Float` anchor
/// (RFC-0012 §4.3 — never a cross-family default), and a non-`Float` operand points at the
/// missing explicit `swap`.
#[test]
fn flt_prims_conformance_reject() {
    for (src, needle) in [
        // Non-Float operand: never a silent conversion.
        (
            "nodule d;\nfn f(a: Binary{8}, b: Float) => Float = flt_add(a, b);",
            "must be a `Float`",
        ),
        (
            "nodule d;\nfn f(a: Float, b: Binary{8}) => Float = flt_add(a, b);",
            "must be a `Float`",
        ),
        (
            "nodule d;\nfn f(t: Ternary{4}) => Float = flt_neg(t);",
            "must be a `Float`",
        ),
        // A bare decimal is not a float literal: no cross-family defaulting (Q6/RFC-0012 §4.3) —
        // without an ambient it has no representation family at all, and even under a declared
        // `default paradigm` it cannot fill a `Float` context (no silent int→float).
        (
            "nodule d;\nfn main() => Float = flt_add(1, 1.5);",
            "no representation family",
        ),
        (
            "nodule d;\ndefault paradigm Binary;\nfn main() => Float = flt_add(1, 1.5);",
            "cannot fill a Float context",
        ),
        // Arity: explicit.
        (
            "nodule d;\nfn main() => Float = flt_add(1.5);",
            "takes 2 operand(s)",
        ),
        (
            "nodule d;\nfn main() => Float = flt_neg(1.5, 2.5);",
            "takes 1 operand(s)",
        ),
        // The result is Float: a non-Float return edge is an explicit mismatch naming the type.
        (
            "nodule d;\nfn main() => Binary{8} = flt_add(1.5, 2.5);",
            "Float",
        ),
    ] {
        let err =
            check_nodule(&parse(src).expect("parses")).expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── M-899 (ADR-040 §2.4, `enb` Gap A): the scalar-float comparison prims ────────────────────────
//
// `flt_lt`/`flt_le`/`flt_gt`/`flt_ge`/`flt_eq` (kernel `flt.lt`/…/`flt.eq`) — the IEEE-754 §5.11
// quiet comparison **predicates** — plus `flt_total_le` (kernel `flt.total_le`), the **named,
// opt-in total order** (IEEE-754 §5.10 `totalOrder`). Two `Float` operands collapse to
// `Binary{1}` (the realized `Bool` — the RFC-0032 D1 note, exactly the `eq`/`lt` result shape).
//
// **The explicit NaN semantics (ADR-040 §2.4 — the point of this group):** float ordering is
// *partial*. NaN is unordered against everything, itself included, and every one of the five
// predicates yields the IEEE-*defined* value **false** on a NaN operand — `flt_eq(NaN, NaN)` is
// false, and NaN is not "the biggest" (`flt_gt(NaN, x)` is false too). That false is never a
// silent ordering: unordered is observable from the predicates themselves (`¬flt_le ∧ ¬flt_gt`;
// `¬flt_eq(x, x)` is the NaN test), and the D1 `eq`/`lt` refuse Float operands *by routing*
// (`flt_cmp_conformance_reject`) rather than inventing a bitwise order (G2). Sorting/keying —
// which a partial order cannot serve — goes through `flt_total_le` **by name**: total,
// reflexive (`flt_total_le(NaN, NaN)` is true), canonical NaN last, and the signed zeros
// *directed* (`−0` precedes `+0`) where `flt_eq` calls them equal (the ADR-040 FLAG-4 seam).
//
// Per-op tag: **`Empirical`** per the ratified ADR-040 §2.6, with the zero-deviation-vs-spec
// comparison bound (EXPLAIN-able off the value; the `EmpiricalFit` method string names the
// M-511 caveat). **The `flt_total_le` total-order property (totality/antisymmetry/transitivity/
// placement) is the M-511 proof debt — corpus/property evidence only, no checked theorem; the
// tag stays `Empirical` until M-511 discharges it (VR-5, never upgraded).** The nullary-main
// surface three-way closes exactly as for the M-898 arithmetic (same trusted `PrimRegistry` on
// every path), NaN rows included — NaN operands are *produced in-language* via `flt_div(0.0,
// 0.0)` (the in-band FLAG-2 specials), so the unordered behavior is exercised end-to-end.

/// Like [`assert_flt_three_way_with_tag`] but for the comparison group: asserts the `Binary{1}`
/// truth value bit-for-bit on **every** path (L1-eval ≡ L0-interp ≡ AOT), plus the ADR-040 §2.6
/// tag contract — guarantee `Empirical`, bound `eps = 0`/`Linf` on an `EmpiricalFit` basis whose
/// method names the M-511 total-order proof debt (G2/SC-3: the unproven status is EXPLAIN-able,
/// never hidden).
#[track_caller]
fn assert_flt_cmp_three_way_with_tag(label: &str, src: &str, expected: bool) {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();
    let node =
        elaborate(&env, "main").unwrap_or_else(|e| panic!("{label}: must be in the fragment: {e}"));
    let l0 = interp
        .eval(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot = mycelium_mlir::run(
        &node,
        &PrimRegistry::with_builtins(),
        &mycelium_cert::BinaryTernarySwapEngine,
    )
    .unwrap_or_else(|e| panic!("{label}: AOT failed: {e}"));
    for (path, v) in [("L1-eval", &l1), ("L0-interp", &l0), ("AOT", &aot)] {
        assert_eq!(
            v.repr(),
            &Repr::Binary { width: 1 },
            "{label}: {path} result must be the Binary{{1}} truth value"
        );
        let Payload::Bits(bits) = v.payload() else {
            panic!("{label}: {path} payload is not Bits: {:?}", v.payload());
        };
        assert_eq!(
            bits.as_slice(),
            &[expected],
            "{label}: {path} truth bit mismatch (want {expected})"
        );
        assert_eq!(
            v.meta().guarantee(),
            GuaranteeStrength::Empirical,
            "{label}: {path} must carry the ratified ADR-040 §2.6 Empirical tag (VR-5)"
        );
        match v.meta().bound() {
            Some(Bound {
                kind: BoundKind::Error { eps, norm },
                basis: BoundBasis::EmpiricalFit { trials, method },
            }) => {
                assert_eq!(*eps, 0.0, "{label}: {path} zero-deviation-vs-spec bound");
                assert_eq!(*norm, NormKind::Linf);
                assert!(*trials >= 1, "an Empirical basis is never evidence-free");
                assert!(
                    method.contains("M-511"),
                    "{label}: {path} basis must surface the M-511 total-order proof debt"
                );
            }
            other => panic!("{label}: {path} expected the EmpiricalFit bound, got {other:?}"),
        }
    }
}

/// The nullary-main surface three-way closes for each comparison op, with a true and a false
/// row per op (bit-asserted on all three paths, tag inspected on all three paths).
#[test]
fn flt_cmp_ops_three_way() {
    for (label, src, expected) in [
        (
            "flt_lt true",
            "nodule d;\nfn main() => Binary{1} = flt_lt(1.5, 2.5);",
            true,
        ),
        (
            "flt_lt false",
            "nodule d;\nfn main() => Binary{1} = flt_lt(2.5, 1.5);",
            false,
        ),
        (
            "flt_le reflexive",
            "nodule d;\nfn main() => Binary{1} = flt_le(1.5, 1.5);",
            true,
        ),
        (
            "flt_le false",
            "nodule d;\nfn main() => Binary{1} = flt_le(2.5, 1.5);",
            false,
        ),
        (
            "flt_gt true",
            "nodule d;\nfn main() => Binary{1} = flt_gt(2.5, 1.5);",
            true,
        ),
        (
            "flt_gt false",
            "nodule d;\nfn main() => Binary{1} = flt_gt(1.5, 2.5);",
            false,
        ),
        (
            "flt_ge reflexive",
            "nodule d;\nfn main() => Binary{1} = flt_ge(1.5, 1.5);",
            true,
        ),
        (
            "flt_ge false",
            "nodule d;\nfn main() => Binary{1} = flt_ge(1.5, 2.5);",
            false,
        ),
        (
            "flt_eq true",
            "nodule d;\nfn main() => Binary{1} = flt_eq(1.5, 1.5);",
            true,
        ),
        (
            "flt_eq false",
            "nodule d;\nfn main() => Binary{1} = flt_eq(1.5, 2.5);",
            false,
        ),
        (
            "flt_total_le true",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(1.5, 2.5);",
            true,
        ),
        (
            "flt_total_le false",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(2.5, 1.5);",
            false,
        ),
    ] {
        assert_flt_cmp_three_way_with_tag(label, src, expected);
    }
}

/// CU-2 (ADR-040 §2.5): the mandated float classification predicates `flt_is_nan`/`flt_is_finite`/
/// `flt_is_infinite` — the in-band never-silent tests for the propagating ±inf/NaN sentinels (§2.4).
/// NaN is produced by `0/0`, +inf by `1/0` (the ratified in-band specials, §2.4/FLAG-2 — never a
/// trap); each predicate agrees on all three paths (L1/L0/AOT), Binary{1} truth.
#[test]
fn flt_classification_three_way() {
    for (label, src, expected) in [
        (
            "is_nan on 0/0",
            "nodule d;\nfn main() => Binary{1} = flt_is_nan(flt_div(0.0, 0.0));",
            true,
        ),
        (
            "is_nan on finite",
            "nodule d;\nfn main() => Binary{1} = flt_is_nan(1.5);",
            false,
        ),
        (
            "is_finite on finite",
            "nodule d;\nfn main() => Binary{1} = flt_is_finite(1.5);",
            true,
        ),
        (
            "is_finite on +inf",
            "nodule d;\nfn main() => Binary{1} = flt_is_finite(flt_div(1.0, 0.0));",
            false,
        ),
        (
            "is_finite on NaN",
            "nodule d;\nfn main() => Binary{1} = flt_is_finite(flt_div(0.0, 0.0));",
            false,
        ),
        (
            "is_infinite on +inf",
            "nodule d;\nfn main() => Binary{1} = flt_is_infinite(flt_div(1.0, 0.0));",
            true,
        ),
        (
            "is_infinite on finite",
            "nodule d;\nfn main() => Binary{1} = flt_is_infinite(2.5);",
            false,
        ),
        (
            "is_infinite on NaN",
            "nodule d;\nfn main() => Binary{1} = flt_is_infinite(flt_div(0.0, 0.0));",
            false,
        ),
    ] {
        assert_flt_cmp_three_way_with_tag(label, src, expected);
    }
}

/// **NaN is unordered, end-to-end on every path (ADR-040 §2.4).** The NaN operand is produced
/// *in-language* by `flt_div(0.0, 0.0)` (the in-band FLAG-2 special), so this is the full
/// surface→kernel NaN story: every §5.11 predicate is `false` with NaN on either side —
/// including `flt_eq(NaN, NaN)` — while the *named* total order places NaN deterministically
/// (reflexive, above +inf).
#[test]
fn flt_cmp_nan_is_unordered_three_way() {
    for (label, src, expected) in [
        (
            "lt(NaN, 1) is false",
            "nodule d;\nfn main() => Binary{1} = flt_lt(flt_div(0.0, 0.0), 1.0);",
            false,
        ),
        (
            "gt(NaN, 1) is false (NaN is not \"the biggest\")",
            "nodule d;\nfn main() => Binary{1} = flt_gt(flt_div(0.0, 0.0), 1.0);",
            false,
        ),
        (
            "le(1, NaN) is false (either operand side)",
            "nodule d;\nfn main() => Binary{1} = flt_le(1.0, flt_div(0.0, 0.0));",
            false,
        ),
        (
            "ge(1, NaN) is false",
            "nodule d;\nfn main() => Binary{1} = flt_ge(1.0, flt_div(0.0, 0.0));",
            false,
        ),
        (
            "eq(NaN, NaN) is false — NaN ≠ NaN",
            "nodule d;\nfn main() => Binary{1} = flt_eq(flt_div(0.0, 0.0), flt_div(0.0, 0.0));",
            false,
        ),
        (
            "total_le(NaN, NaN) is true — the total order IS reflexive on NaN",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(flt_div(0.0, 0.0), flt_div(0.0, 0.0));",
            true,
        ),
        (
            "total_le(+inf, NaN) is true — canonical NaN sorts last",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(flt_div(1.0, 0.0), flt_div(0.0, 0.0));",
            true,
        ),
        (
            "total_le(NaN, +inf) is false — NaN precedes nothing but itself",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(flt_div(0.0, 0.0), flt_div(1.0, 0.0));",
            false,
        ),
    ] {
        assert_flt_cmp_three_way_with_tag(label, src, expected);
    }
}

/// **The signed-zero seam, three-way (ADR-040 FLAG-4):** `−0` and `+0` are IEEE-**equal** under
/// `flt_eq` (and unordered by `flt_lt` in both directions) yet **distinct and directed** under
/// the named total order — `flt_total_le(−0, +0)` but not `flt_total_le(+0, −0)`.
#[test]
fn flt_cmp_signed_zero_three_way() {
    for (label, src, expected) in [
        (
            "eq(+0, −0) — IEEE-equal",
            "nodule d;\nfn main() => Binary{1} = flt_eq(0.0, flt_neg(0.0));",
            true,
        ),
        (
            "lt(−0, +0) — equal zeros are not less",
            "nodule d;\nfn main() => Binary{1} = flt_lt(flt_neg(0.0), 0.0);",
            false,
        ),
        (
            "total_le(−0, +0) — −0 precedes +0",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(flt_neg(0.0), 0.0);",
            true,
        ),
        (
            "total_le(+0, −0) — the zeros are DISTINCT under the total order",
            "nodule d;\nfn main() => Binary{1} = flt_total_le(0.0, flt_neg(0.0));",
            false,
        ),
    ] {
        assert_flt_cmp_three_way_with_tag(label, src, expected);
    }
}

/// Static conformance — accept: every comparison signature the checker must admit (two `Float`
/// operands → `Binary{1}`, params/literals/composed `flt.*` results all admissible operands).
#[test]
fn flt_cmp_conformance_accept() {
    for src in [
        "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = flt_lt(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = flt_le(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = flt_gt(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = flt_ge(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = flt_eq(a, b);",
        "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = flt_total_le(a, b);",
        // Literal operands (M-897) and composed flt.* operands.
        "nodule d;\nfn main() => Binary{1} = flt_eq(1.5, 2.5e-3);",
        "nodule d;\nfn f(a: Float) => Binary{1} = flt_lt(flt_mul(a, 2.0), 8.0);",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent operand/arity/result contract is a *check-time*
/// refusal naming the offense (G2), and — the M-899 routing rule — the D1 `eq`/`lt` refuse
/// `Float` operands by **pointing at the float predicates and the named total order**, never by
/// silently inventing an order for NaN.
#[test]
fn flt_cmp_conformance_reject() {
    for (src, needle) in [
        // Non-Float operand: never a silent conversion.
        (
            "nodule d;\nfn f(a: Binary{8}, b: Float) => Binary{1} = flt_lt(a, b);",
            "must be a `Float`",
        ),
        (
            "nodule d;\nfn f(t: Ternary{4}, b: Float) => Binary{1} = flt_total_le(t, b);",
            "must be a `Float`",
        ),
        // A bare decimal has no Float anchor (Q6/RFC-0012 §4.3): no cross-family defaulting.
        (
            "nodule d;\nfn main() => Binary{1} = flt_lt(1, 1.5);",
            "no representation family",
        ),
        // Arity: explicit.
        (
            "nodule d;\nfn main() => Binary{1} = flt_lt(1.5);",
            "takes 2 operand(s)",
        ),
        (
            "nodule d;\nfn main() => Binary{1} = flt_total_le(1.5);",
            "takes 2 operand(s)",
        ),
        // The result is Binary{1}, not Float: a wrong return edge is an explicit mismatch.
        (
            "nodule d;\nfn main() => Float = flt_lt(1.5, 2.5);",
            "Binary",
        ),
        // The D1 comparisons route floats to the flt_* surface — the refusal names it.
        (
            "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = eq(a, b);",
            "flt_total_le",
        ),
        (
            "nodule d;\nfn f(a: Float, b: Float) => Binary{1} = lt(a, b);",
            "flt_lt",
        ),
    ] {
        let err =
            check_nodule(&parse(src).expect("parses")).expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── M-890 (`enb` Gap C): the dense elementwise prim group ───────────────────────────────────────
//
// `dense_add`/`dense_sub`/`dense_neg`/`dense_scale` (kernel `dense.add`/`dense.sub`/`dense.neg`/
// `dense.scale`, the `mycelium-dense` surface) — the first **tensor-valued** prims. Dim + dtype
// live in the type (`Dense{d, s}`), so the never-silent shape contract is *static* (a mismatch is
// a check-time refusal, never a broadcast); the numeric-domain contracts (overflow, subnormal,
// off-grid, approximate sources) stay *runtime* refusals owned by the kernel, which also
// constructs the result's honest per-op tag (`op_guarantee`: `neg` `Exact`, the rest `Proven`
// with the ProvenThm relative-ε bound) — carried through every path unchanged (VR-5).
//
// **Where the three-way closes (recorded honestly — G2/VR-5).** L1 has **no dense
// value-construction form yet**: at this suite's writing there was no float literal (Gap A —
// M-897 has since landed the *scalar* float literal below, but a **dense** construction form
// still does not exist), a bare decimal under a `Dense` ambient is an explicit refusal
// (RFC-0012 §4.3; `tests/ambient.rs`), and the Binary→Dense swap is an Explicit-Residual on all
// paths (DN-52 FLAG-1, `tests/differential.rs`). So a *nullary* `main` over dense values is
// **inexpressible**, and the surface-program three-way of `assert_three_way` cannot run. The
// three-way below therefore closes over the forms that DO exist: **L1-eval with kernel-built
// `Dense` argument values injected through `Evaluator::call`** ≡ **L0-interp over the equivalent
// hand-built `Node::Op`** ≡ **AOT (`mycelium_mlir::run`) over the same node** — agreement on
// repr + payload + the carried tag, and on the never-silent overflow refusal. The nullary-main
// surface closure is deferred to the dense value-construction form (Gap A / a dense literal),
// not silently skipped.

use mycelium_core::ScalarKind;
use mycelium_dense::DenseSpace;
use mycelium_interp::EvalError;
use mycelium_l1::{L1Error, L1Value};

/// A `Dense{n, F32}` value from on-grid elements, built through the kernel's own constructor —
/// the only dense value-construction form until a surface literal lands (Gap A).
fn dense_f32(xs: Vec<f64>) -> Value {
    let n = u32::try_from(xs.len()).expect("test dims are small");
    DenseSpace::new(n, ScalarKind::F32)
        .expect("F32 is a supported dtype")
        .value(xs)
        .expect("fixture elements are finite and on-grid")
}

/// Run the M-890 three-way on one dense prim application (see the section note for why the
/// surface leg takes injected argument values): L1-eval (`Evaluator::call` on the checked surface
/// program `entry`) ≡ L0-interp ≡ AOT (both over the equivalent hand-built `Node::Op`). Asserts
/// all three agree on repr + payload and returns the L0 value for tag inspection.
fn assert_dense_three_way(
    label: &str,
    src: &str,
    entry: &str,
    kernel: &str,
    args: &[Value],
) -> Value {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    // Path 1: the L1 fuel-guarded evaluator, with the kernel-built Dense values as arguments.
    let l1 = Evaluator::new(&env)
        .call(entry, args.iter().cloned().map(L1Value::Repr).collect())
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();

    // Paths 2+3: the equivalent L0 term (the prim over `Const`s of the same values), on the
    // reference interpreter and through the AOT path.
    let node = Node::Op {
        prim: kernel.to_owned(),
        args: args.iter().cloned().map(Node::Const).collect(),
    };
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let l0 = interp
        .eval(&node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot = mycelium_mlir::run(
        &node,
        &PrimRegistry::with_builtins(),
        &mycelium_cert::BinaryTernarySwapEngine,
    )
    .unwrap_or_else(|e| panic!("{label}: AOT failed: {e}"));

    for (path, v) in [("L1-eval", &l1), ("L0-interp", &l0), ("AOT", &aot)] {
        assert_eq!(v.repr(), l0.repr(), "{label}: {path} repr diverged");
        assert_eq!(
            v.payload(),
            l0.payload(),
            "{label}: {path} payload diverged"
        );
        assert_eq!(
            v.meta().guarantee(),
            l0.meta().guarantee(),
            "{label}: {path} carried tag diverged"
        );
    }
    l0
}

#[test]
fn dense_add_three_way_carries_the_proven_tag() {
    let a = dense_f32(vec![1.5, 2.5]);
    let b = dense_f32(vec![0.25, -1.0]);
    let y = assert_dense_three_way(
        "dense_add",
        "nodule d;\nfn f(a: Dense{2, F32}, b: Dense{2, F32}) => Dense{2, F32} = dense_add(a, b);",
        "f",
        "dense.add",
        &[a, b],
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![1.75, 1.5]));
    // The kernel's tag, carried on every path: Proven + the ProvenThm relative-ε bound.
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    let space = DenseSpace::new(2, ScalarKind::F32).unwrap();
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { .. },
        }) => {
            assert_eq!(*eps, space.op_rel_eps());
            assert_eq!(*norm, NormKind::Rel);
        }
        other => panic!("expected the kernel's ProvenThm Error bound, got {other:?}"),
    }
}

#[test]
fn dense_sub_neg_scale_three_way() {
    let a = dense_f32(vec![1.5, 2.5]);
    let b = dense_f32(vec![0.5, -1.0]);
    let y = assert_dense_three_way(
        "dense_sub",
        "nodule d;\nfn f(a: Dense{2, F32}, b: Dense{2, F32}) => Dense{2, F32} = dense_sub(a, b);",
        "f",
        "dense.sub",
        &[a.clone(), b],
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![1.0, 3.5]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);

    // neg: Exact (never rounds), no bound — the carried-tag distinction inside one group.
    let n = assert_dense_three_way(
        "dense_neg",
        "nodule d;\nfn f(a: Dense{2, F32}) => Dense{2, F32} = dense_neg(a);",
        "f",
        "dense.neg",
        std::slice::from_ref(&a),
    );
    assert_eq!(n.payload(), &Payload::Scalars(vec![-1.5, -2.5]));
    assert_eq!(n.meta().guarantee(), GuaranteeStrength::Exact);
    assert!(n.meta().bound().is_none(), "Exact results carry no bound");

    // scale: the factor is a Dense{1, F32} scalar (the pre-Gap-A scalar form).
    let c = dense_f32(vec![2.0]);
    let s = assert_dense_three_way(
        "dense_scale",
        "nodule d;\nfn f(a: Dense{2, F32}, c: Dense{1, F32}) => Dense{2, F32} = dense_scale(a, c);",
        "f",
        "dense.scale",
        &[a, c],
    );
    assert_eq!(s.payload(), &Payload::Scalars(vec![3.0, 5.0]));
    assert_eq!(s.meta().guarantee(), GuaranteeStrength::Proven);
}

/// Runtime reject, three-way: an out-of-range result refuses explicitly and *consistently* on
/// every path (L1-eval wraps the kernel's refusal in `L1Error::Kernel`; L0-interp and AOT surface
/// it directly) — never a silent ±Inf (G2).
#[test]
fn dense_add_overflow_refuses_on_every_path() {
    let max = dense_f32(vec![f64::from(f32::MAX)]);
    let src =
        "nodule d;\nfn f(a: Dense{1, F32}, b: Dense{1, F32}) => Dense{1, F32} = dense_add(a, b);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let l1 = Evaluator::new(&env).call(
        "f",
        vec![L1Value::Repr(max.clone()), L1Value::Repr(max.clone())],
    );
    assert!(
        matches!(l1, Err(L1Error::Kernel(EvalError::Overflow { .. }))),
        "L1-eval must refuse the overflow explicitly, got {l1:?}"
    );

    let node = Node::Op {
        prim: "dense.add".to_owned(),
        args: vec![Node::Const(max.clone()), Node::Const(max)],
    };
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    assert!(
        matches!(interp.eval(&node), Err(EvalError::Overflow { .. })),
        "L0-interp must refuse the overflow explicitly"
    );
    assert!(
        matches!(
            mycelium_mlir::run(
                &node,
                &PrimRegistry::with_builtins(),
                &mycelium_cert::BinaryTernarySwapEngine
            ),
            Err(EvalError::Overflow { .. })
        ),
        "AOT must refuse the overflow explicitly"
    );
}

/// Static conformance — accept: every dense prim signature the checker must admit.
#[test]
fn dense_prims_conformance_accept() {
    for src in [
        "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{4, F32} = dense_add(a, b);",
        "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{4, F32} = dense_sub(a, b);",
        "nodule d;\nfn f(a: Dense{4, F32}) => Dense{4, F32} = dense_neg(a);",
        "nodule d;\nfn f(a: Dense{4, F32}, c: Dense{1, F32}) => Dense{4, F32} = dense_scale(a, c);",
        // BF16 spaces type identically (the dtype rides the type).
        "nodule d;\nfn f(a: Dense{8, BF16}, b: Dense{8, BF16}) => Dense{8, BF16} = dense_add(a, b);",
        // Composition: the result of one dense prim feeds the next (dim/dtype-preserving).
        "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{4, F32} = dense_neg(dense_add(a, b));",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent shape/dtype contract is a *check-time* refusal
/// (dim + dtype live in the type), with a message naming the offense (G2).
#[test]
fn dense_prims_conformance_reject() {
    for (src, needle) in [
        // Dim mismatch: never a broadcast.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{8, F32}) => Dense{4, F32} = dense_add(a, b);",
            "share one dim and dtype",
        ),
        // Dtype mismatch: never a silent re-round.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, BF16}) => Dense{4, F32} = dense_sub(a, b);",
            "share one dim and dtype",
        ),
        // Cross-paradigm operand: an explicit refusal pointing at the missing swap.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Binary{8}) => Dense{4, F32} = dense_add(a, b);",
            "must be a `Dense{dim, scalar}`",
        ),
        (
            "nodule d;\nfn f(a: Binary{8}) => Binary{8} = dense_neg(a);",
            "must be a `Dense{dim, scalar}`",
        ),
        // The scale factor must be Dense{1, s} of the SAME dtype.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, c: Dense{3, F32}) => Dense{4, F32} = dense_scale(a, c);",
            "factor must be a `Dense{1, scalar}`",
        ),
        (
            "nodule d;\nfn f(a: Dense{4, F32}, c: Dense{1, BF16}) => Dense{4, F32} = dense_scale(a, c);",
            "factor dtype",
        ),
        // Arity: explicit.
        (
            "nodule d;\nfn f(a: Dense{4, F32}) => Dense{4, F32} = dense_add(a);",
            "takes 2 operand(s)",
        ),
    ] {
        let err = check_nodule(&parse(src).expect("parses"))
            .expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── M-891 (`enb` Gap C): the dense measurement pair `dense_dot`/`dense_similarity` ──────────────
//
// Kernel `dense.dot`/`dense.similarity` — two equal-dim/dtype `Dense{d, s}` operands reduce to a
// **`Dense{1, F64}` measurement** (the binary64 the kernel computed, delivered exactly), carrying
// the kernel's `Proven` **accumulation bound**: absolute (`Linf`), `dot_abs_eps`/
// `similarity_abs_eps` under a ProvenThm citation — deliberately NOT the dtype's per-element
// `op_rel_eps` (inputs are exact on-grid and the accumulation is binary64, so the dtype ε never
// enters; a per-element relative claim on a dot would be false under cancellation — VR-5). The
// M-891 crux is that this bound **flows into the per-op tag and is EXPLAIN-able**: the tests
// below inspect guarantee + ε + norm + citation off the result `Value` itself, on every path.
//
// **Where the three-way closes:** same as M-890 (see the section note above) — a nullary `main`
// over dense values is inexpressible until a dense value-construction form lands (Gap A float /
// a dense literal), so the surface leg takes kernel-built `Dense` arguments through
// `Evaluator::call`; recorded honestly, not silently skipped.

/// M-891 three-way + EXPLAIN: `dense_dot` agrees on L1-eval ≡ L0-interp ≡ AOT, its result is the
/// `Dense{1, F64}` measurement form, and the disclosed accumulation bound (ε, norm, ProvenThm
/// citation) is **inspectable off the value** — the accuracy claim is never a black box (G2/SC-3).
#[test]
fn dense_dot_three_way_measurement_with_inspectable_bound() {
    let a = dense_f32(vec![1.5, 2.0, -0.5]);
    let b = dense_f32(vec![2.0, 0.25, 4.0]);
    let y = assert_dense_three_way(
        "dense_dot",
        "nodule d;\nfn f(a: Dense{3, F32}, b: Dense{3, F32}) => Dense{1, F64} = dense_dot(a, b);",
        "f",
        "dense.dot",
        &[a, b],
    );
    // 3.0 + 0.5 − 2.0 = 1.5 (every product and partial sum exact in binary64).
    assert_eq!(
        y.repr(),
        &Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64
        },
        "the measurement result form is Dense{{1, F64}}"
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![1.5]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    // EXPLAIN: the kernel's guarantee metadata is inspectable — ε is `dot_abs_eps` over the
    // computed abs-product sum (3.0 + 0.5 + 2.0), the norm is absolute (Linf), and the
    // ProvenThm citation names its theorem basis.
    let space = DenseSpace::new(3, ScalarKind::F32).unwrap();
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { citation },
        }) => {
            assert_eq!(*eps, space.dot_abs_eps(3.0 + 0.5 + 2.0));
            assert_eq!(*norm, NormKind::Linf);
            assert!(
                citation.contains("Higham"),
                "the EXPLAIN-able citation must name its theorem basis: {citation}"
            );
        }
        other => panic!("expected the kernel's ProvenThm Linf bound, got {other:?}"),
    }
}

#[test]
fn dense_similarity_three_way_and_zero_convention() {
    // Orthogonal on-grid vectors: the cosine is exactly 0 (every product is 0).
    let a = dense_f32(vec![1.0, 0.0]);
    let b = dense_f32(vec![0.0, 1.0]);
    let y = assert_dense_three_way(
        "dense_similarity",
        "nodule d;\nfn f(a: Dense{2, F32}, b: Dense{2, F32}) => Dense{1, F64} = \
         dense_similarity(a, b);",
        "f",
        "dense.similarity",
        &[a.clone(), b],
    );
    assert_eq!(
        y.repr(),
        &Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F64
        }
    );
    assert_eq!(y.payload(), &Payload::Scalars(vec![0.0]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    // The similarity bound is input-independent (normalization caps the absolute error).
    let space = DenseSpace::new(2, ScalarKind::F32).unwrap();
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Error { eps, norm },
            basis: BoundBasis::ProvenThm { .. },
        }) => {
            assert_eq!(*eps, space.similarity_abs_eps());
            assert_eq!(*norm, NormKind::Linf);
        }
        other => panic!("expected the kernel's ProvenThm Linf bound, got {other:?}"),
    }
    // The zero-norm convention (documented in the kernel citation): exactly 0 on every path.
    let z = dense_f32(vec![0.0, 0.0]);
    let zc = assert_dense_three_way(
        "dense_similarity_zero",
        "nodule d;\nfn f(a: Dense{2, F32}, b: Dense{2, F32}) => Dense{1, F64} = \
         dense_similarity(a, b);",
        "f",
        "dense.similarity",
        &[a, z],
    );
    assert_eq!(zc.payload(), &Payload::Scalars(vec![0.0]));
}

/// Static conformance — accept: the measurement-pair signatures the checker must admit
/// (the result type is always `Dense{1, F64}`, whatever the operand dim/dtype).
#[test]
fn dense_measurement_conformance_accept() {
    for src in [
        "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{1, F64} = dense_dot(a, b);",
        "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{1, F64} = \
         dense_similarity(a, b);",
        // BF16 operands measure identically (the dtype rides the operand type; result is F64).
        "nodule d;\nfn f(a: Dense{8, BF16}, b: Dense{8, BF16}) => Dense{1, F64} = dense_dot(a, b);",
        // Composition: a dense-elementwise result feeds the measurement.
        "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{1, F64} = \
         dense_dot(dense_add(a, b), b);",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent shape/dtype contract, plus the
/// measurement-result form itself (the result is `Dense{1, F64}`, not the operand type —
/// mis-declaring it is a check-time refusal, so the F64 measurement can never silently pose as
/// an on-grid operand value).
#[test]
fn dense_measurement_conformance_reject() {
    for (src, needle) in [
        // Dim mismatch: never a broadcast.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{8, F32}) => Dense{1, F64} = dense_dot(a, b);",
            "share one dim and dtype",
        ),
        // Dtype mismatch: never a silent re-round.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, BF16}) => Dense{1, F64} = \
             dense_similarity(a, b);",
            "share one dim and dtype",
        ),
        // Cross-paradigm operand: an explicit refusal pointing at the missing swap.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Binary{8}) => Dense{1, F64} = dense_dot(a, b);",
            "must be a `Dense{dim, scalar}`",
        ),
        // Arity: explicit.
        (
            "nodule d;\nfn f(a: Dense{4, F32}) => Dense{1, F64} = dense_dot(a);",
            "takes 2 operand(s)",
        ),
        // The result is the Dense{1, F64} measurement form — declaring the operand type is a
        // static mismatch, never a silent re-round of the measurement onto the operand grid.
        (
            "nodule d;\nfn f(a: Dense{4, F32}, b: Dense{4, F32}) => Dense{4, F32} = dense_dot(a, b);",
            "Dense{1, F64}",
        ),
    ] {
        let err = check_nodule(&parse(src).expect("parses"))
            .expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── M-892 (`enb` Gap C): the model-dispatched VSA bind group ────────────────────────────────────
//
// `vsa_bind`/`vsa_unbind`/`vsa_permute` (kernel `vsa.bind`/`vsa.unbind`/`vsa.permute`, the
// `mycelium-vsa` surface) — model-dispatched across MAP-I / FHRR / BSC on the operand's
// `VSA{model, dim, sparsity}` type. Model + dim live in the type, so the never-silent model/shape
// contract is *static* (a cross-model or cross-dim call is a check-time refusal, never a
// coercion), and an out-of-set model is a *static* refusal naming the dispatch set; the
// numeric-domain contracts (alphabet violations, the FHRR unbind regime gate, non-Exact-operand
// composition) stay *runtime* refusals owned by the kernel, which also constructs the result's
// honest **per-model** tag (MAP-I/BSC ops `Exact`; FHRR `unbind` `Empirical` with its
// trial-validated δ bound) — carried through every path unchanged (VR-5).
//
// **Where the three-way closes (recorded honestly — G2/VR-5).** L1 has **no hypervector
// value-construction form** (no VSA literal; a bare decimal has no VSA anchor — RFC-0012 §4.3),
// so a *nullary* `main` over hypervectors is **inexpressible** and the surface-program three-way
// of `assert_three_way` cannot run — exactly the M-890 dense posture. The three-way below
// therefore closes over the forms that DO exist: **L1-eval with kernel-shaped `Repr::Vsa`
// argument values injected through `Evaluator::call`** ≡ **L0-interp over the equivalent
// hand-built `Node::Op`** ≡ **AOT (`mycelium_mlir::run`) over the same node** — agreement on
// repr + payload + the carried per-model tag, and on the never-silent runtime refusals. The
// nullary-main surface closure is deferred to a VSA value-construction form (a later wave), not
// silently skipped. (This suite builds hypervectors through `mycelium-core` alone —
// `Repr::Vsa` + `Payload::Hypervector` are core value forms; `mycelium-l1` takes no
// `mycelium-vsa` dependency.)

use mycelium_core::SparsityClass;

/// A hypervector `Value` of `model` at `dim` (dense class, `Exact`/`Root` meta) — the injected
/// argument form (core value forms only; the kernel models validate the alphabet at op time).
fn vsa_hv(model: &str, dim: u32, data: Vec<f64>) -> Value {
    Value::new(
        Repr::Vsa {
            model: model.to_owned(),
            dim,
            sparsity: SparsityClass::Dense,
        },
        Payload::Hypervector(data),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A deterministic FHRR phasor atom (phases in `(−π, π]`; tiny LCG — house style).
fn phasor_atom(dim: u32, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            let t = std::f64::consts::TAU * u;
            if t > std::f64::consts::PI {
                t - std::f64::consts::TAU
            } else {
                t
            }
        })
        .collect()
}

/// An unsigned `Binary{8}` shift-amount value (MSB-first) — `vsa_permute`'s second operand.
fn shift8(v: u64) -> Value {
    let bits: Vec<bool> = (0..8u32).rev().map(|i| (v >> i) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// Run the M-892 three-way on one checked surface program + the equivalent hand-built L0 node
/// (see the section note for why the surface leg takes injected argument values): L1-eval
/// (`Evaluator::call` on `entry`) ≡ L0-interp ≡ AOT (both over `node`). Asserts all three agree
/// on repr + payload + the carried tag and returns the L0 value for tag/bound inspection.
fn assert_vsa_three_way(label: &str, src: &str, entry: &str, node: &Node, args: &[Value]) -> Value {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));

    // Path 1: the L1 fuel-guarded evaluator, with the injected hypervector arguments.
    let l1 = Evaluator::new(&env)
        .call(entry, args.iter().cloned().map(L1Value::Repr).collect())
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1 = l1
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: result must be a repr value"))
        .clone();

    // Paths 2+3: the equivalent L0 term on the reference interpreter and through the AOT path.
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let l0 = interp
        .eval(node)
        .unwrap_or_else(|e| panic!("{label}: L0-interp failed: {e}"));
    let aot = mycelium_mlir::run(
        node,
        &PrimRegistry::with_builtins(),
        &mycelium_cert::BinaryTernarySwapEngine,
    )
    .unwrap_or_else(|e| panic!("{label}: AOT failed: {e}"));

    for (path, v) in [("L1-eval", &l1), ("L0-interp", &l0), ("AOT", &aot)] {
        assert_eq!(v.repr(), l0.repr(), "{label}: {path} repr diverged");
        assert_eq!(
            v.payload(),
            l0.payload(),
            "{label}: {path} payload diverged"
        );
        assert_eq!(
            v.meta().guarantee(),
            l0.meta().guarantee(),
            "{label}: {path} carried tag diverged"
        );
    }
    l0
}

/// One prim application as a hand-built L0 node (the prim over `Const`s of the argument values).
fn op_node(kernel: &str, args: &[Value]) -> Node {
    Node::Op {
        prim: kernel.to_owned(),
        args: args.iter().cloned().map(Node::Const).collect(),
    }
}

/// Three-way, per self-inverse model: `vsa_bind` carries the kernel's **`Exact`** tag on every
/// path, with the algebraically-checkable payloads (MAP-I elementwise product; BSC XOR).
#[test]
fn vsa_bind_three_way_carries_the_exact_tag_per_model() {
    // MAP-I: [1,-1,1,-1] ⊗ [1,1,-1,-1] = [1,-1,-1,1] (elementwise product).
    let a = vsa_hv("MAP-I", 4, vec![1.0, -1.0, 1.0, -1.0]);
    let b = vsa_hv("MAP-I", 4, vec![1.0, 1.0, -1.0, -1.0]);
    let y = assert_vsa_three_way(
        "vsa_bind/MAP-I",
        "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 4, Dense}) => \
         VSA{MAP_I, 4, Dense} = vsa_bind(a, b);",
        "f",
        &op_node("vsa.bind", &[a.clone(), b.clone()]),
        &[a, b],
    );
    assert_eq!(
        y.payload(),
        &Payload::Hypervector(vec![1.0, -1.0, -1.0, 1.0])
    );
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Exact);
    assert!(y.meta().bound().is_none(), "Exact results carry no bound");

    // BSC: [0,1,1,0] ⊕ [0,1,0,1] = [0,0,1,1] (XOR).
    let a = vsa_hv("BSC", 4, vec![0.0, 1.0, 1.0, 0.0]);
    let b = vsa_hv("BSC", 4, vec![0.0, 1.0, 0.0, 1.0]);
    let y = assert_vsa_three_way(
        "vsa_bind/BSC",
        "nodule v;\nfn f(a: VSA{BSC, 4, Dense}, b: VSA{BSC, 4, Dense}) => \
         VSA{BSC, 4, Dense} = vsa_bind(a, b);",
        "f",
        &op_node("vsa.bind", &[a.clone(), b.clone()]),
        &[a, b],
    );
    assert_eq!(y.payload(), &Payload::Hypervector(vec![0.0, 0.0, 1.0, 1.0]));
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Exact);
}

/// Three-way composition: `vsa_unbind(vsa_bind(a, b), b)` recovers `a` **exactly** for the
/// self-inverse MAP-I (tag `Exact` carried), and for FHRR carries the kernel's **`Empirical`**
/// weak-link tag with its disclosed probability bound — the carried-tag distinction *inside one
/// prim*, decided by the dispatched model (VR-5: per-model, never re-stamped).
#[test]
fn vsa_unbind_composition_three_way_is_model_tagged() {
    // MAP-I (self-inverse, Exact): recover a exactly.
    let a = vsa_hv("MAP-I", 4, vec![1.0, -1.0, 1.0, 1.0]);
    let b = vsa_hv("MAP-I", 4, vec![-1.0, 1.0, 1.0, -1.0]);
    let src = "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 4, Dense}) => \
               VSA{MAP_I, 4, Dense} = vsa_unbind(vsa_bind(a, b), b);";
    let node = Node::Op {
        prim: "vsa.unbind".to_owned(),
        args: vec![
            op_node("vsa.bind", &[a.clone(), b.clone()]),
            Node::Const(b.clone()),
        ],
    };
    let y = assert_vsa_three_way("vsa_unbind/MAP-I", src, "f", &node, &[a.clone(), b]);
    assert_eq!(y.payload(), a.payload(), "self-inverse recovery is exact");
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Exact);

    // FHRR (approximate-inverse role): Empirical + the disclosed probability bound.
    let a = vsa_hv("FHRR", 256, phasor_atom(256, 1));
    let b = vsa_hv("FHRR", 256, phasor_atom(256, 2));
    let src = "nodule v;\nfn f(a: VSA{FHRR, 256, Dense}, b: VSA{FHRR, 256, Dense}) => \
               VSA{FHRR, 256, Dense} = vsa_unbind(vsa_bind(a, b), b);";
    let node = Node::Op {
        prim: "vsa.unbind".to_owned(),
        args: vec![
            op_node("vsa.bind", &[a.clone(), b.clone()]),
            Node::Const(b.clone()),
        ],
    };
    let y = assert_vsa_three_way("vsa_unbind/FHRR", src, "f", &node, &[a, b]);
    assert_eq!(
        y.meta().guarantee(),
        GuaranteeStrength::Empirical,
        "FHRR unbind carries the kernel's weak-link Empirical tag (RFC-0003 §4; VR-5)"
    );
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Probability { .. },
            basis: BoundBasis::EmpiricalFit { .. },
        }) => {} // the disclosed trial-validated δ (its constants are pinned in mycelium-interp)
        other => panic!("expected the kernel's EmpiricalFit probability bound, got {other:?}"),
    }
}

/// Three-way: `vsa_permute` is `Exact` on every path, the shift rides an unsigned `Binary{8}`
/// (surface literal `0b0000_0001`), and the complementary shift is its inverse (cyclic rotation).
#[test]
fn vsa_permute_three_way_is_exact_and_cyclic() {
    // MAP-I: rotate [1,-1,1,1] by 1 → [-1,1,1,1] (out[i] = a[(i+shift) mod d]).
    let a = vsa_hv("MAP-I", 4, vec![1.0, -1.0, 1.0, 1.0]);
    let s = shift8(1);
    let y = assert_vsa_three_way(
        "vsa_permute/MAP-I",
        "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, s: Binary{8}) => VSA{MAP_I, 4, Dense} = \
         vsa_permute(a, s);",
        "f",
        &op_node("vsa.permute", &[a.clone(), s.clone()]),
        &[a.clone(), s],
    );
    assert_eq!(
        y.payload(),
        &Payload::Hypervector(vec![-1.0, 1.0, 1.0, 1.0])
    );
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Exact);

    // The complementary shift (dim − s = 3) restores the original — the inverse permute is
    // expressible with the unsigned shift operand (no negative-shift form needed).
    let back = assert_vsa_three_way(
        "vsa_permute/inverse",
        "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, s: Binary{8}) => VSA{MAP_I, 4, Dense} = \
         vsa_permute(a, s);",
        "f",
        &op_node("vsa.permute", &[y.clone(), shift8(3)]),
        &[y, shift8(3)],
    );
    assert_eq!(back.payload(), a.payload(), "cyclic inverse");
}

/// Runtime reject, three-way: a **model mismatch reachable only at runtime** (injected arguments
/// bypass the static types — the declared params say MAP-I, the injected second argument is BSC)
/// refuses explicitly and *consistently* on every path (L1-eval wraps the kernel's refusal in
/// `L1Error::Kernel`; L0-interp and AOT surface it directly) — never a cross-model coercion (G2).
#[test]
fn vsa_model_mismatch_refuses_on_every_path() {
    let a = vsa_hv("MAP-I", 4, vec![1.0, -1.0, 1.0, -1.0]);
    let foreign = vsa_hv("BSC", 4, vec![0.0, 1.0, 1.0, 0.0]);
    let src = "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 4, Dense}) => \
               VSA{MAP_I, 4, Dense} = vsa_bind(a, b);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let l1 = Evaluator::new(&env).call(
        "f",
        vec![L1Value::Repr(a.clone()), L1Value::Repr(foreign.clone())],
    );
    assert!(
        matches!(l1, Err(L1Error::Kernel(EvalError::PrimType { .. }))),
        "L1-eval must refuse the model mismatch explicitly, got {l1:?}"
    );

    let node = op_node("vsa.bind", &[a, foreign]);
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    assert!(
        matches!(interp.eval(&node), Err(EvalError::PrimType { .. })),
        "L0-interp must refuse the model mismatch explicitly"
    );
    assert!(
        matches!(
            mycelium_mlir::run(
                &node,
                &PrimRegistry::with_builtins(),
                &mycelium_cert::BinaryTernarySwapEngine
            ),
            Err(EvalError::PrimType { .. })
        ),
        "AOT must refuse the model mismatch explicitly"
    );
}

/// Runtime reject, three-way: the FHRR unbind **regime gate** — unbinding a value that is not a
/// single `vsa.fhrr.bind` product is an explicit refusal on every path (the kernel's
/// `OutsideEmpiricalProfile`: the Empirical tag is issued only inside its trial-validated
/// regime; VR-5), never a silently mis-tagged decode.
#[test]
fn vsa_fhrr_unbind_regime_gate_refuses_on_every_path() {
    let a = vsa_hv("FHRR", 256, phasor_atom(256, 1));
    let b = vsa_hv("FHRR", 256, phasor_atom(256, 2));
    let src = "nodule v;\nfn f(a: VSA{FHRR, 256, Dense}, b: VSA{FHRR, 256, Dense}) => \
               VSA{FHRR, 256, Dense} = vsa_unbind(a, b);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    // Root provenance → outside the validated single-factor regime, on every path.
    let l1 = Evaluator::new(&env).call(
        "f",
        vec![L1Value::Repr(a.clone()), L1Value::Repr(b.clone())],
    );
    assert!(
        matches!(l1, Err(L1Error::Kernel(EvalError::PrimType { .. }))),
        "L1-eval must refuse the out-of-regime unbind explicitly, got {l1:?}"
    );
    let node = op_node("vsa.unbind", &[a, b]);
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    assert!(
        matches!(interp.eval(&node), Err(EvalError::PrimType { .. })),
        "L0-interp must refuse the out-of-regime unbind explicitly"
    );
    assert!(
        matches!(
            mycelium_mlir::run(
                &node,
                &PrimRegistry::with_builtins(),
                &mycelium_cert::BinaryTernarySwapEngine
            ),
            Err(EvalError::PrimType { .. })
        ),
        "AOT must refuse the out-of-regime unbind explicitly"
    );
}

/// Static conformance — accept: every VSA bind-group signature the checker must admit, across
/// the model set (incl. the surface `MAP_I` → kernel `MAP-I` ident canonicalization, prim
/// composition, and an out-of-set model as a legal type *mention* — the ADR-008 mention/algebra
/// split: the type checks; only the prims are gated).
#[test]
fn vsa_prims_conformance_accept() {
    for src in [
        // The model set, bind/unbind/permute.
        "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 4, Dense}) => \
         VSA{MAP_I, 4, Dense} = vsa_bind(a, b);",
        "nodule v;\nfn f(a: VSA{FHRR, 256, Dense}, b: VSA{FHRR, 256, Dense}) => \
         VSA{FHRR, 256, Dense} = vsa_unbind(a, b);",
        "nodule v;\nfn f(a: VSA{BSC, 1024, Dense}, s: Binary{16}) => VSA{BSC, 1024, Dense} = \
         vsa_permute(a, s);",
        // Composition: model + dim are preserved, so results feed sibling prims.
        "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 4, Dense}, s: Binary{8}) => \
         VSA{MAP_I, 4, Dense} = vsa_permute(vsa_bind(a, b), s);",
        // An out-of-set model is a legal *mention* (no algebra invoked).
        "nodule v;\nfn f(a: VSA{HRR, 4, Dense}) => VSA{HRR, 4, Dense} = a;",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent model/shape contract is a *check-time* refusal
/// (model + dim live in the type), with a message naming the offense (G2).
#[test]
fn vsa_prims_conformance_reject() {
    for (src, needle) in [
        // Cross-model: never a coercion.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{BSC, 4, Dense}) => \
             VSA{MAP_I, 4, Dense} = vsa_bind(a, b);",
            "share one model and dim",
        ),
        // Dim mismatch: never a resize.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 8, Dense}) => \
             VSA{MAP_I, 4, Dense} = vsa_unbind(a, b);",
            "share one model and dim",
        ),
        // A model outside the introduction dispatch set: static refusal naming the set.
        (
            "nodule v;\nfn f(a: VSA{HRR, 4, Dense}, b: VSA{HRR, 4, Dense}) => \
             VSA{HRR, 4, Dense} = vsa_bind(a, b);",
            "outside the vsa prim dispatch set",
        ),
        // Sparse operands: refused at introduction (kernel results are dense-class).
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Sparse{2}}, b: VSA{MAP_I, 4, Sparse{2}}) => \
             VSA{MAP_I, 4, Sparse{2}} = vsa_bind(a, b);",
            "requires a `Dense`-sparsity hypervector",
        ),
        // Cross-paradigm operand: an explicit refusal pointing at the missing swap.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: Binary{8}) => VSA{MAP_I, 4, Dense} = \
             vsa_bind(a, b);",
            "must be a `VSA{model, dim, sparsity}`",
        ),
        // The permute shift must be a Binary magnitude, not a hypervector.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, s: VSA{MAP_I, 4, Dense}) => \
             VSA{MAP_I, 4, Dense} = vsa_permute(a, s);",
            "shift must be an unsigned `Binary{W}`",
        ),
        // Arity: explicit.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}) => VSA{MAP_I, 4, Dense} = vsa_bind(a);",
            "takes 2 operand(s)",
        ),
        // The result type is computed (model + dim preserved) — declaring another is a static
        // mismatch, never a silent re-model.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 4, Dense}, b: VSA{MAP_I, 4, Dense}) => \
             VSA{BSC, 4, Dense} = vsa_bind(a, b);",
            "VSA{MAP-I, 4, Dense}",
        ),
    ] {
        let err =
            check_nodule(&parse(src).expect("parses")).expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── M-893 (`enb` Gap C): `vsa_bundle` — the certified superposition path ────────────────────────
//
// `vsa_bundle(items: Seq{VSA{MAP_I, d, Dense}, N≥1}, δ: Float) → VSA{MAP_I, d, Dense}` (kernel
// `vsa.bundle` → MAP-I's `bundle_values_certified`, the M-131 checked-instantiation pattern).
// The static rules ride the types: item model/dim/sparsity + the item count N live in the `Seq`
// type, so the certified-singleton dispatch ({MAP-I} — FHRR/BSC bundles are Empirical-profile
// kernel ops, refused naming the certified set), the Dense-sparsity introduction scope, and the
// **empty bundle (N = 0)** are all *static* refusals. The capacity side-condition
// `dim ≥ requiredDim(N, δ)` stays **runtime** (δ is a run-time `Float`): the kernel refuses
// `InsufficientCapacity` naming the required dim rather than issuing an unbacked `Proven` bound
// (M-I2/VR-5), and issues the **`Proven` `CapacityBound`** — the value's OWN m/dim with the
// `ProvenThm` citation — only when the check passes, carried unchanged on every path.
//
// **Where the three-way closes (recorded honestly — G2/VR-5).** Exactly the M-892 posture: L1
// has no hypervector value-construction form, so the surface leg injects `Repr::Vsa` argument
// values through `Evaluator::call` (the `Seq` is built by the surface **list literal** over the
// injected params; δ rides a surface **float literal**) ≡ L0-interp over the equivalent
// hand-built `Node::Op` (a `Const` of the kernel-shaped `Repr::Seq` value + a `Const` `Float`)
// ≡ AOT over the same node — agreement on repr + payload + the carried `Proven` tag, and on the
// never-silent runtime refusals. The nullary-main surface closure is deferred to a VSA
// value-construction form (a later wave), not silently skipped.

/// A `Float` δ value (`Exact`/`Root` meta) — the injected/`Const` form of the second operand.
fn fval(x: f64) -> Value {
    Value::new(
        Repr::Float {
            width: FloatWidth::F64,
        },
        Payload::Float(x),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A `Seq` value over hypervector items (`Exact`/`Root` meta — the shape the surface list
/// literal evaluates to), for the hand-built L0 leg.
fn vsa_seq(items: &[Value]) -> Value {
    let first = items.first().expect("test seqs are non-empty");
    Value::new(
        Repr::Seq {
            elem: Box::new(first.repr().clone()),
            len: u32::try_from(items.len()).expect("small"),
        },
        Payload::Seq(items.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A deterministic bipolar (`±1`) MAP-I atom (tiny LCG — house style).
fn bipolar_atom(dim: u32, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 63) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Three-way: `vsa_bundle` over three MAP-I atoms at a sufficient dim (2048 ≥ requiredDim(3,
/// 1e-2) = 1141) carries the kernel's **`Proven`** tag and its checked `CapacityBound` — the
/// **value's own** items/dim with the `ProvenThm` citation — identically on every path, with the
/// elementwise-sum payload. The `Seq` rides the surface list literal over the injected params;
/// δ rides a surface float literal.
#[test]
fn vsa_bundle_three_way_carries_the_proven_tag_and_its_own_capacity_bound() {
    const DIM: u32 = 2048;
    let items: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 40 + i)))
        .collect();
    let src = "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}, b: VSA{MAP_I, 2048, Dense}, \
               c: VSA{MAP_I, 2048, Dense}) => VSA{MAP_I, 2048, Dense} = \
               vsa_bundle([a, b, c], 0.01);";
    let node = Node::Op {
        prim: "vsa.bundle".to_owned(),
        args: vec![Node::Const(vsa_seq(&items)), Node::Const(fval(0.01))],
    };
    let y = assert_vsa_three_way("vsa_bundle/MAP-I", src, "f", &node, &items);
    // Payload: the elementwise integer superposition of the three bipolar atoms.
    let expected: Vec<f64> = (0..DIM as usize)
        .map(|k| {
            items
                .iter()
                .map(|v| match v.payload() {
                    Payload::Hypervector(h) => h[k],
                    _ => unreachable!(),
                })
                .sum()
        })
        .collect();
    assert_eq!(y.payload(), &Payload::Hypervector(expected));
    // The kernel's Proven tag + its checked bound, carried unchanged (VR-5) — and the disclosed
    // bound is the value's OWN (this bundle's m and d), with the checked basis recorded.
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Capacity { items: m, dim: d },
            basis: BoundBasis::ProvenThm { citation },
        }) => {
            assert_eq!(*m, 3, "the bound discloses this bundle's item count");
            assert_eq!(*d, u64::from(DIM), "the bound discloses this bundle's dim");
            assert!(
                citation.contains("Clarkson") && citation.contains("requiredDim"),
                "the ProvenThm basis records the citation + checked side-condition: {citation}"
            );
        }
        other => panic!("expected the kernel's checked Capacity/ProvenThm bound, got {other:?}"),
    }
}

/// Runtime reject, three-way: an **under-dimensioned certified bundle** (dim 16 < requiredDim(3,
/// 1e-2) = 1141 — the theorem's side-condition fails) refuses explicitly and consistently on
/// every path, naming the required dim — never an unbacked `Proven` bound and never a
/// silently-weaker result (M-I2/VR-5; G2).
#[test]
fn vsa_bundle_insufficient_capacity_refuses_on_every_path() {
    const DIM: u32 = 16;
    let items: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 50 + i)))
        .collect();
    let src = "nodule v;\nfn f(a: VSA{MAP_I, 16, Dense}, b: VSA{MAP_I, 16, Dense}, \
               c: VSA{MAP_I, 16, Dense}) => VSA{MAP_I, 16, Dense} = \
               vsa_bundle([a, b, c], 0.01);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let l1 = Evaluator::new(&env).call("f", items.iter().cloned().map(L1Value::Repr).collect());
    match l1 {
        Err(L1Error::Kernel(EvalError::PrimType { why, .. })) => assert!(
            why.contains("insufficient capacity") && why.contains("1141"),
            "L1-eval names the failed side-condition + required dim, got: {why}"
        ),
        other => panic!("L1-eval must refuse the under-dimensioned bundle, got {other:?}"),
    }

    let node = Node::Op {
        prim: "vsa.bundle".to_owned(),
        args: vec![Node::Const(vsa_seq(&items)), Node::Const(fval(0.01))],
    };
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    assert!(
        matches!(interp.eval(&node), Err(EvalError::PrimType { .. })),
        "L0-interp must refuse the under-dimensioned bundle explicitly"
    );
    assert!(
        matches!(
            mycelium_mlir::run(
                &node,
                &PrimRegistry::with_builtins(),
                &mycelium_cert::BinaryTernarySwapEngine
            ),
            Err(EvalError::PrimType { .. })
        ),
        "AOT must refuse the under-dimensioned bundle explicitly"
    );
}

/// Runtime reject, three-way: an **uncertified model reachable only at runtime** (the declared
/// params say MAP-I, the injected items are BSC — injected arguments bypass the static types)
/// refuses explicitly on every path naming the certified set — never a silent re-tag of the
/// BSC empirical-profile bundle (VR-5/G2). (The static twin is in the conformance-reject suite.)
#[test]
fn vsa_bundle_uncertified_model_refuses_on_every_path() {
    const DIM: u32 = 16;
    let items: Vec<Value> = (0..3)
        .map(|i| {
            vsa_hv(
                "BSC",
                DIM,
                bipolar_atom(DIM, 60 + i)
                    .into_iter()
                    .map(|x| if x > 0.0 { 1.0 } else { 0.0 })
                    .collect(),
            )
        })
        .collect();
    let src = "nodule v;\nfn f(a: VSA{MAP_I, 16, Dense}, b: VSA{MAP_I, 16, Dense}, \
               c: VSA{MAP_I, 16, Dense}) => VSA{MAP_I, 16, Dense} = \
               vsa_bundle([a, b, c], 0.01);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    let l1 = Evaluator::new(&env).call("f", items.iter().cloned().map(L1Value::Repr).collect());
    match l1 {
        Err(L1Error::Kernel(EvalError::PrimType { why, .. })) => assert!(
            why.contains("certified singleton"),
            "L1-eval names the certified set, got: {why}"
        ),
        other => panic!("L1-eval must refuse the uncertified model, got {other:?}"),
    }

    let node = Node::Op {
        prim: "vsa.bundle".to_owned(),
        args: vec![Node::Const(vsa_seq(&items)), Node::Const(fval(0.01))],
    };
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    assert!(
        matches!(interp.eval(&node), Err(EvalError::PrimType { .. })),
        "L0-interp must refuse the uncertified model explicitly"
    );
    assert!(
        matches!(
            mycelium_mlir::run(
                &node,
                &PrimRegistry::with_builtins(),
                &mycelium_cert::BinaryTernarySwapEngine
            ),
            Err(EvalError::PrimType { .. })
        ),
        "AOT must refuse the uncertified model explicitly"
    );
}

/// Static conformance — accept: the `vsa_bundle` signatures the checker must admit (a single
/// item; several items; the bundle result feeding a sibling vsa prim — model + dim preserved;
/// δ as a param instead of a literal).
#[test]
fn vsa_bundle_conformance_accept() {
    for src in [
        "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}) => VSA{MAP_I, 2048, Dense} = \
         vsa_bundle([a], 0.01);",
        "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}, b: VSA{MAP_I, 2048, Dense}, \
         c: VSA{MAP_I, 2048, Dense}) => VSA{MAP_I, 2048, Dense} = vsa_bundle([a, b, c], 0.01);",
        // Composition: the bundle's model + dim are preserved, so the result feeds vsa_permute.
        "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}, b: VSA{MAP_I, 2048, Dense}, s: Binary{8}) \
         => VSA{MAP_I, 2048, Dense} = vsa_permute(vsa_bundle([a, b], 0.01), s);",
        // δ as a Float parameter (runtime δ — the capacity check is the kernel's).
        "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}, b: VSA{MAP_I, 2048, Dense}, d: Float) => \
         VSA{MAP_I, 2048, Dense} = vsa_bundle([a, b], d);",
        // A Seq param (not a literal) works too — the item count is in the type.
        "nodule v;\nfn f(xs: Seq{VSA{MAP_I, 2048, Dense}, 3}, d: Float) => \
         VSA{MAP_I, 2048, Dense} = vsa_bundle(xs, d);",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent contract at check time, each refusal naming
/// the offense (G2): an uncertified model (FHRR/BSC — the certified singleton), an out-of-set
/// model, Sparse items, a non-Seq first operand, non-VSA items, the static empty bundle
/// (`N = 0` lives in the type), a non-Float δ, arity, and a wrong declared result.
#[test]
fn vsa_bundle_conformance_reject() {
    for (src, needle) in [
        // FHRR/BSC: in the M-892 bind-group set, but their bundles are not certified — static
        // refusal naming the certified singleton (never a silent re-tag of Empirical evidence).
        (
            "nodule v;\nfn f(a: VSA{FHRR, 2048, Dense}, b: VSA{FHRR, 2048, Dense}) => \
             VSA{FHRR, 2048, Dense} = vsa_bundle([a, b], 0.01);",
            "certified singleton",
        ),
        (
            "nodule v;\nfn f(a: VSA{BSC, 2048, Dense}, b: VSA{BSC, 2048, Dense}) => \
             VSA{BSC, 2048, Dense} = vsa_bundle([a, b], 0.01);",
            "certified singleton",
        ),
        // A model outside the introduction dispatch set: the shared static refusal.
        (
            "nodule v;\nfn f(a: VSA{HRR, 2048, Dense}, b: VSA{HRR, 2048, Dense}) => \
             VSA{HRR, 2048, Dense} = vsa_bundle([a, b], 0.01);",
            "outside the vsa prim dispatch set",
        ),
        // Sparse items: refused at introduction (kernel results are dense-class).
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 2048, Sparse{2}}, b: VSA{MAP_I, 2048, Sparse{2}}) \
             => VSA{MAP_I, 2048, Sparse{2}} = vsa_bundle([a, b], 0.01);",
            "requires `Dense`-sparsity hypervectors",
        ),
        // A non-Seq first operand: never a silent lift of one hypervector into a bundle.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}) => VSA{MAP_I, 2048, Dense} = \
             vsa_bundle(a, 0.01);",
            "first operand must be a `Seq",
        ),
        // Non-VSA items: an explicit refusal pointing at the missing swap.
        (
            "nodule v;\nfn f(a: Binary{8}, b: Binary{8}) => VSA{MAP_I, 2048, Dense} = \
             vsa_bundle([a, b], 0.01);",
            "items must be `VSA{model, dim, sparsity}`",
        ),
        // The static empty bundle: N = 0 lives in the Seq type — no superposition is defined.
        (
            "nodule v;\nfn f(xs: Seq{VSA{MAP_I, 2048, Dense}, 0}, d: Float) => \
             VSA{MAP_I, 2048, Dense} = vsa_bundle(xs, d);",
            "at least one item",
        ),
        // δ must be a Float — never a defaulted or coerced parameter.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}, b: VSA{MAP_I, 2048, Dense}, \
             d: Binary{8}) => VSA{MAP_I, 2048, Dense} = vsa_bundle([a, b], d);",
            "probability \u{3b4} must be a `Float`",
        ),
        // Arity: explicit.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}) => VSA{MAP_I, 2048, Dense} = \
             vsa_bundle([a]);",
            "takes 2 operand(s)",
        ),
        // The result type is computed (model + dim preserved) — declaring another is a static
        // mismatch, never a silent re-model.
        (
            "nodule v;\nfn f(a: VSA{MAP_I, 2048, Dense}, b: VSA{MAP_I, 2048, Dense}) => \
             VSA{BSC, 2048, Dense} = vsa_bundle([a, b], 0.01);",
            "VSA{MAP-I, 2048, Dense}",
        ),
    ] {
        let err =
            check_nodule(&parse(src).expect("parses")).expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── M-894 (`enb` Gap C): `vsa_cleanup` + `vsa_reconstruct` + `vsa_required_dim` ─────────────────
//
// The cleanup-memory retrieval, the RFC-0003 §6 compositional role-reconstruction, and the M-131
// capacity-bound query (FR-S4). Signatures:
//   vsa_cleanup(q: VSA{m, d, Dense}, cb: Seq{VSA{m, d, Dense}, N≥1}) → Seq{Float, 3}
//   vsa_reconstruct(r: VSA{m, d, Dense}, role: VSA{m, d, Dense},
//                   cb: Seq{VSA{m, d, Dense}, N≥1}, thr: Float) → Seq{Float, 3}
//   vsa_required_dim(items: Binary{W}, δ: Float) → Binary{64}
// The `Seq{Float, 3}` result is the `[index, confidence, margin]` decision triple — the retrieval
// decision is a first-class, inspectable value (never a silent nearest-neighbour pick — G2), and
// it carries the query/record's own (strength, bound) pair through the RFC-0001 §4.7 meet (a
// certified bundle record re-discloses its `Proven` `CapacityBound`: the disclosed bound is the
// value's own — VR-5).
//
// **Where the three-way closes (recorded honestly — G2/VR-5).** Exactly the M-892/M-893 posture:
// L1 has no hypervector value-construction form, so the surface leg injects `Repr::Vsa` argument
// values through `Evaluator::call` (codebooks ride the surface **list literal** over injected
// params; thresholds/δ ride surface **float literals**/params; `vsa_required_dim`'s items ride a
// `Binary{8}` param) ≡ L0-interp over the equivalent hand-built `Node::Op` ≡ AOT
// (`mycelium_mlir::run`) over the same node — agreement on repr + payload + the carried tag, and
// on the never-silent runtime refusals. The nullary-main surface closure is deferred to a VSA
// value-construction form (a later wave), not silently skipped.

/// Decode a `Binary{64}` result (MSB-first) to a u64 — `vsa_required_dim`'s result form.
fn u64_of_bits(v: &Value) -> u64 {
    match (v.repr(), v.payload()) {
        (Repr::Binary { width: 64 }, Payload::Bits(bits)) => {
            bits.iter().fold(0u64, |acc, &b| (acc << 1) | u64::from(b))
        }
        other => panic!("expected a Binary{{64}} dimension, got {other:?}"),
    }
}

/// Unpack the `Seq{Float, 3}` decision triple.
fn triple_of(v: &Value) -> [f64; 3] {
    match v.payload() {
        Payload::Seq(elems) => {
            let xs: Vec<f64> = elems
                .iter()
                .map(|e| match e.payload() {
                    Payload::Float(x) => *x,
                    other => panic!("triple element must be a Float, got {other:?}"),
                })
                .collect();
            [xs[0], xs[1], xs[2]]
        }
        other => panic!("expected the Seq{{Float, 3}} decision triple, got {other:?}"),
    }
}

/// Three-way: `vsa_cleanup` over an exact-atom query returns the `[index, confidence, margin]`
/// decision triple — index of the matching atom, confidence ≈ 1, positive margin — as an
/// `Exact`/no-bound `Seq{Float, 3}` identically on every path (all operands `Exact`).
#[test]
fn vsa_cleanup_three_way_returns_the_decision_triple() {
    const DIM: u32 = 64;
    let atoms: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 70 + i)))
        .collect();
    let query = atoms[1].clone();
    let src = "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: VSA{MAP_I, 64, Dense}, \
               b: VSA{MAP_I, 64, Dense}, c: VSA{MAP_I, 64, Dense}) => Seq{Float, 3} = \
               vsa_cleanup(q, [a, b, c]);";
    let node = Node::Op {
        prim: "vsa.cleanup".to_owned(),
        args: vec![Node::Const(query.clone()), Node::Const(vsa_seq(&atoms))],
    };
    let args: Vec<Value> = std::iter::once(query)
        .chain(atoms.iter().cloned())
        .collect();
    let y = assert_vsa_three_way("vsa_cleanup/MAP-I", src, "f", &node, &args);
    let [index, confidence, margin] = triple_of(&y);
    assert_eq!(index, 1.0, "the exact atom cleans to its own index");
    assert!((confidence - 1.0).abs() < 1e-9, "confidence={confidence}");
    assert!(margin > 0.0, "unique arg-max, margin={margin}");
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Exact);
    assert!(y.meta().bound().is_none(), "Exact results carry no bound");
}

/// Three-way: `vsa_reconstruct` from a **certified bundle** record recovers the bound filler and
/// **re-discloses the record's own `Proven` `CapacityBound`** on every path — the disclosed bound
/// is the value's own (VR-5), never re-derived and never dropped. (The record is built through
/// the interpreter's own certified-bundle prim, then injected — the M-892 injected-argument
/// posture.)
#[test]
fn vsa_reconstruct_three_way_rediscloses_the_records_own_bound() {
    const DIM: u32 = 2048;
    let roles: Vec<Value> = (0..2)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 80 + i)))
        .collect();
    let fillers: Vec<Value> = (0..2)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 90 + i)))
        .collect();
    // record = certified_bundle([role0 ⊗ filler0, role1 ⊗ filler1], δ = 0.01), built through the
    // registry's own prims so the Proven tag + CapacityBound are the kernel's.
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let pairs: Vec<Node> = roles
        .iter()
        .zip(&fillers)
        .map(|(r, x)| op_node("vsa.bind", &[r.clone(), x.clone()]))
        .collect();
    let record = {
        let products: Vec<Value> = pairs
            .iter()
            .map(|n| interp.eval(n).expect("bind accepts"))
            .collect();
        interp
            .eval(&Node::Op {
                prim: "vsa.bundle".to_owned(),
                args: vec![Node::Const(vsa_seq(&products)), Node::Const(fval(0.01))],
            })
            .expect("certified bundle accepts")
    };
    assert_eq!(record.meta().guarantee(), GuaranteeStrength::Proven);
    let record_bound = record.meta().bound().cloned().expect("Proven has a bound");

    let src = "nodule v;\nfn f(r: VSA{MAP_I, 2048, Dense}, role: VSA{MAP_I, 2048, Dense}, \
               x: VSA{MAP_I, 2048, Dense}, y: VSA{MAP_I, 2048, Dense}) => Seq{Float, 3} = \
               vsa_reconstruct(r, role, [x, y], 0.2);";
    let node = Node::Op {
        prim: "vsa.reconstruct".to_owned(),
        args: vec![
            Node::Const(record.clone()),
            Node::Const(roles[1].clone()),
            Node::Const(vsa_seq(&fillers)),
            Node::Const(fval(0.2)),
        ],
    };
    let args = vec![
        record,
        roles[1].clone(),
        fillers[0].clone(),
        fillers[1].clone(),
    ];
    let y = assert_vsa_three_way("vsa_reconstruct/MAP-I", src, "f", &node, &args);
    let [index, confidence, margin] = triple_of(&y);
    assert_eq!(index, 1.0, "role 1 recovers filler 1 below capacity");
    assert!(confidence >= 0.2 && margin > 0.0);
    // The carried pair is the record's own: Proven + its OWN CapacityBound (m = 2, d = 2048).
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    assert_eq!(
        y.meta().bound(),
        Some(&record_bound),
        "the disclosed bound is the record's own (VR-5)"
    );
    match y.meta().bound() {
        Some(Bound {
            kind: BoundKind::Capacity { items: 2, dim },
            basis: BoundBasis::ProvenThm { .. },
        }) => assert_eq!(*dim, u64::from(DIM)),
        other => panic!("expected the record's Capacity/ProvenThm bound, got {other:?}"),
    }
}

/// Three-way: `vsa_required_dim` answers the M-001 probe row (requiredDim(3, 1e-2) = 1141) as a
/// `Binary{64}` carrying the kernel's **`Proven`** `CapacityBound` for exactly that
/// instantiation, identically on every path — the capacity-bound query is inspectable (the
/// `ProvenThm` basis records the citation + μ + the checked side-condition).
#[test]
fn vsa_required_dim_three_way_carries_the_proven_capacity_bound() {
    let items = shift8(3);
    let src = "nodule v;\nfn f(m: Binary{8}, d: Float) => Binary{64} = vsa_required_dim(m, d);";
    let node = Node::Op {
        prim: "vsa.required_dim".to_owned(),
        args: vec![Node::Const(items.clone()), Node::Const(fval(0.01))],
    };
    let y = assert_vsa_three_way("vsa_required_dim", src, "f", &node, &[items, fval(0.01)]);
    assert_eq!(u64_of_bits(&y), 1141, "the M-001 probe instantiation");
    assert_eq!(y.meta().guarantee(), GuaranteeStrength::Proven);
    match y.meta().bound() {
        Some(Bound {
            kind:
                BoundKind::Capacity {
                    items: 3,
                    dim: 1141,
                },
            basis: BoundBasis::ProvenThm { citation },
        }) => assert!(
            citation.contains("Clarkson") && citation.contains("requiredDim"),
            "the ProvenThm basis records the citation + checked side-condition: {citation}"
        ),
        other => panic!("expected the kernel's Capacity/ProvenThm bound, got {other:?}"),
    }
}

/// Runtime reject, three-way: a **below-threshold retrieval** refuses explicitly and consistently
/// on every path, naming confidence vs threshold — never a silent low-quality answer (RFC-0003
/// §6; G2). (The m = 3 crosstalk confidence ≈ 1/√3 cannot clear the 0.9 threshold.)
#[test]
fn vsa_reconstruct_below_threshold_refuses_on_every_path() {
    const DIM: u32 = 2048;
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let roles: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 170 + i)))
        .collect();
    let fillers: Vec<Value> = (0..3)
        .map(|i| vsa_hv("MAP-I", DIM, bipolar_atom(DIM, 180 + i)))
        .collect();
    let products: Vec<Value> = roles
        .iter()
        .zip(&fillers)
        .map(|(r, x)| {
            interp
                .eval(&op_node("vsa.bind", &[r.clone(), x.clone()]))
                .expect("bind accepts")
        })
        .collect();
    let record = interp
        .eval(&Node::Op {
            prim: "vsa.bundle".to_owned(),
            args: vec![Node::Const(vsa_seq(&products)), Node::Const(fval(0.01))],
        })
        .expect("certified bundle accepts");

    let src = "nodule v;\nfn f(r: VSA{MAP_I, 2048, Dense}, role: VSA{MAP_I, 2048, Dense}, \
               x: VSA{MAP_I, 2048, Dense}, y: VSA{MAP_I, 2048, Dense}, \
               z: VSA{MAP_I, 2048, Dense}) => Seq{Float, 3} = \
               vsa_reconstruct(r, role, [x, y, z], 0.9);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let mut l1_args = vec![record.clone(), roles[0].clone()];
    l1_args.extend(fillers.iter().cloned());
    let l1 = Evaluator::new(&env).call("f", l1_args.into_iter().map(L1Value::Repr).collect());
    match l1 {
        Err(L1Error::Kernel(EvalError::PrimType { why, .. })) => assert!(
            why.contains("below the threshold 0.9"),
            "L1-eval names confidence vs threshold, got: {why}"
        ),
        other => panic!("L1-eval must refuse the below-threshold retrieval, got {other:?}"),
    }

    let node = Node::Op {
        prim: "vsa.reconstruct".to_owned(),
        args: vec![
            Node::Const(record),
            Node::Const(roles[0].clone()),
            Node::Const(vsa_seq(&fillers)),
            Node::Const(fval(0.9)),
        ],
    };
    assert!(
        matches!(interp.eval(&node), Err(EvalError::PrimType { .. })),
        "L0-interp must refuse the below-threshold retrieval explicitly"
    );
    assert!(
        matches!(
            mycelium_mlir::run(
                &node,
                &PrimRegistry::with_builtins(),
                &mycelium_cert::BinaryTernarySwapEngine
            ),
            Err(EvalError::PrimType { .. })
        ),
        "AOT must refuse the below-threshold retrieval explicitly"
    );
}

/// Static conformance — accept: the M-894 signatures the checker must admit — cleanup across the
/// model set (incl. FHRR: the arg-max is model-generic), reconstruct over its {MAP-I, BSC}
/// dispatch set, threshold/δ as params or literals, a `Seq` param codebook, composition (the
/// capacity query's `Binary{64}` feeding the D1 `eq`), and the triple feeding `seq_len`.
#[test]
fn vsa_cleanup_reconstruct_conformance_accept() {
    for src in [
        // Cleanup, per dispatch-set model (the arg-max procedure is model-generic).
        "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: VSA{MAP_I, 64, Dense}, \
         b: VSA{MAP_I, 64, Dense}) => Seq{Float, 3} = vsa_cleanup(q, [a, b]);",
        "nodule v;\nfn f(q: VSA{FHRR, 256, Dense}, a: VSA{FHRR, 256, Dense}) => \
         Seq{Float, 3} = vsa_cleanup(q, [a]);",
        "nodule v;\nfn f(q: VSA{BSC, 64, Dense}, a: VSA{BSC, 64, Dense}) => Seq{Float, 3} = \
         vsa_cleanup(q, [a]);",
        // A Seq param codebook (the atom count is in the type).
        "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, cb: Seq{VSA{MAP_I, 64, Dense}, 4}) => \
         Seq{Float, 3} = vsa_cleanup(q, cb);",
        // Reconstruct over its dispatch set, threshold as a literal and as a param.
        "nodule v;\nfn f(r: VSA{MAP_I, 2048, Dense}, role: VSA{MAP_I, 2048, Dense}, \
         x: VSA{MAP_I, 2048, Dense}) => Seq{Float, 3} = vsa_reconstruct(r, role, [x], 0.3);",
        "nodule v;\nfn f(r: VSA{BSC, 64, Dense}, role: VSA{BSC, 64, Dense}, \
         x: VSA{BSC, 64, Dense}, t: Float) => Seq{Float, 3} = vsa_reconstruct(r, role, [x], t);",
        // The capacity query; δ as a param; its Binary{64} result feeds sibling prims.
        "nodule v;\nfn f(m: Binary{8}, d: Float) => Binary{64} = vsa_required_dim(m, d);",
        "nodule v;\nfn f(m: Binary{8}, n: Binary{8}) => Binary{1} = \
         eq(vsa_required_dim(m, 0.01), vsa_required_dim(n, 0.01));",
        // The decision triple is a first-class Seq — it composes (e.g. its length).
        "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: VSA{MAP_I, 64, Dense}) => Binary{32} = \
         seq_len(vsa_cleanup(q, [a]));",
    ] {
        check_nodule(&parse(src).expect("parses"))
            .unwrap_or_else(|e| panic!("must accept: {src}\n  got: {e}"));
    }
}

/// Static conformance — reject: the never-silent contract at check time, each refusal naming the
/// offense (G2): codebook model/dim mismatches, the static empty codebook (`N = 0`), non-Seq /
/// non-VSA codebooks, the FHRR-reconstruct dispatch refusal, out-of-set models, Sparse operands,
/// non-Float thresholds/δ, a bare-decimal / non-Binary items operand, arity, and a wrong declared
/// result type (the triple's type is computed).
#[test]
fn vsa_cleanup_reconstruct_conformance_reject() {
    for (src, needle) in [
        // Codebook model mismatch: never a coercion.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: VSA{BSC, 64, Dense}) => \
             Seq{Float, 3} = vsa_cleanup(q, [a]);",
            "share the query's model and dim",
        ),
        // Codebook dim mismatch: never a resize.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, cb: Seq{VSA{MAP_I, 128, Dense}, 2}) => \
             Seq{Float, 3} = vsa_cleanup(q, cb);",
            "share the query's model and dim",
        ),
        // The static empty codebook: N = 0 lives in the Seq type.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, cb: Seq{VSA{MAP_I, 64, Dense}, 0}) => \
             Seq{Float, 3} = vsa_cleanup(q, cb);",
            "at least one codebook atom",
        ),
        // A non-Seq codebook: never a silent lift.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: VSA{MAP_I, 64, Dense}) => \
             Seq{Float, 3} = vsa_cleanup(q, a);",
            "codebook must be a `Seq",
        ),
        // Non-VSA codebook atoms: an explicit refusal pointing at the missing swap.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: Binary{8}) => Seq{Float, 3} = \
             vsa_cleanup(q, [a]);",
            "atoms must be `VSA{model, dim, sparsity}`",
        ),
        // FHRR reconstruct: a static refusal naming the dispatch set + the profile ground.
        (
            "nodule v;\nfn f(r: VSA{FHRR, 256, Dense}, role: VSA{FHRR, 256, Dense}, \
             x: VSA{FHRR, 256, Dense}) => Seq{Float, 3} = vsa_reconstruct(r, role, [x], 0.3);",
            "outside the reconstruct dispatch set",
        ),
        // An out-of-set model: the shared static refusal naming the M-892 set.
        (
            "nodule v;\nfn f(q: VSA{HRR, 64, Dense}, a: VSA{HRR, 64, Dense}) => \
             Seq{Float, 3} = vsa_cleanup(q, [a]);",
            "outside the vsa prim dispatch set",
        ),
        // Sparse operands: refused at introduction.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Sparse{2}}, a: VSA{MAP_I, 64, Sparse{2}}) => \
             Seq{Float, 3} = vsa_cleanup(q, [a]);",
            "requires a `Dense`-sparsity hypervector",
        ),
        // Role/record mismatch: named.
        (
            "nodule v;\nfn f(r: VSA{MAP_I, 64, Dense}, role: VSA{MAP_I, 128, Dense}, \
             x: VSA{MAP_I, 64, Dense}) => Seq{Float, 3} = vsa_reconstruct(r, role, [x], 0.3);",
            "record and role must share one model and dim",
        ),
        // A non-Float threshold: never a defaulted or coerced parameter.
        (
            "nodule v;\nfn f(r: VSA{MAP_I, 64, Dense}, role: VSA{MAP_I, 64, Dense}, \
             x: VSA{MAP_I, 64, Dense}, t: Binary{8}) => Seq{Float, 3} = \
             vsa_reconstruct(r, role, [x], t);",
            "threshold must be a `Float`",
        ),
        // The items operand must be a Binary magnitude (a bare decimal has no width anchor).
        (
            "nodule v;\nfn f(d: Float) => Binary{64} = vsa_required_dim(d, d);",
            "items must be an unsigned `Binary{W}`",
        ),
        // δ must be a Float.
        (
            "nodule v;\nfn f(m: Binary{8}, d: Binary{8}) => Binary{64} = \
             vsa_required_dim(m, d);",
            "probability \u{3b4} must be a `Float`",
        ),
        // Arity: explicit.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}) => Seq{Float, 3} = vsa_cleanup(q);",
            "takes 2 operand(s)",
        ),
        (
            "nodule v;\nfn f(r: VSA{MAP_I, 64, Dense}, role: VSA{MAP_I, 64, Dense}, \
             x: VSA{MAP_I, 64, Dense}) => Seq{Float, 3} = vsa_reconstruct(r, role, [x]);",
            "takes 4 operand(s)",
        ),
        // The result type is computed (`Seq{Float, 3}`) — declaring another is a static
        // mismatch, never a silent re-shape.
        (
            "nodule v;\nfn f(q: VSA{MAP_I, 64, Dense}, a: VSA{MAP_I, 64, Dense}) => \
             VSA{MAP_I, 64, Dense} = vsa_cleanup(q, [a]);",
            "Seq{Float, 3}",
        ),
    ] {
        let err =
            check_nodule(&parse(src).expect("parses")).expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── CU-3 (ADR-040 §2.4): never-silent Binary↔Float conversions ─────────────────────────────────
//
// `bin_to_flt` (kernel `bin.to_flt`) is checked-exact `Binary{N} -> Float`; `flt_to_bin` (kernel
// `flt.to_bin`) is `(Float, Binary{M}) -> Binary{M}`, the second operand a **width witness** —
// exactly `width_cast`'s DN-41 shape (only the witness's width is read, its bits are ignored).
// Both refuse rather than silently round/truncate on every path (G2/VR-5); the **lossy** rounding
// `bin→flt` direction for magnitudes past the binary64 exact-integer bound (`2^53`) is explicitly
// out of scope — a reified swap, not a prim (see the CU-3 leaf report FLAG).

/// Assert `src` refuses on **all three** paths (L1-eval, L0-interp, AOT) — never a silent
/// success on any path (G2). Mirrors the manual pattern `mul_u_overflow_refuses_on_every_path`
/// uses, factored once for the CU-3 refusal cases below.
fn assert_refuses_on_every_path(label: &str, src: &str) {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(mycelium_cert::BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = mycelium_cert::BinaryTernarySwapEngine;

    assert!(
        Evaluator::new(&env).call("main", vec![]).is_err(),
        "{label}: L1-eval must refuse (never-silent, ADR-040 §2.4)"
    );
    let node = elaborate(&env, "main").expect("in fragment");
    assert!(
        interp.eval(&node).is_err(),
        "{label}: L0-interp must refuse (never-silent, ADR-040 §2.4)"
    );
    assert!(
        mycelium_mlir::run(&node, &prims, &engine).is_err(),
        "{label}: AOT must refuse (never-silent, ADR-040 §2.4)"
    );
}

/// `bin_to_flt` round-trips small in-range magnitudes to `Float`, bit-exact on all three paths.
#[test]
fn bin_to_flt_round_trips_three_way() {
    assert_three_way(
        "bin_to_flt 0",
        "nodule d;\nfn main() => Float = bin_to_flt(0b0000_0000);",
        &Repr::Float {
            width: FloatWidth::F64,
        },
        &Payload::Float(0.0),
    );
    assert_three_way(
        "bin_to_flt 255",
        "nodule d;\nfn main() => Float = bin_to_flt(0b1111_1111);",
        &Repr::Float {
            width: FloatWidth::F64,
        },
        &Payload::Float(255.0),
    );
}

/// `bin_to_flt` refuses a magnitude past the binary64 exact-integer bound (`2^53`) on every
/// path — never a silent lossy round (ADR-040 §2.4/§5: the lossy direction is a reified swap,
/// not this prim).
#[test]
fn bin_to_flt_refuses_past_the_exact_bound_on_every_path() {
    // 2^54 as a 55-bit Binary literal (MSB `1`, the rest `0`) exceeds FLOAT_EXACT_MAX = 2^53.
    let lit: String = std::iter::once('1')
        .chain(std::iter::repeat_n('0', 54))
        .collect();
    assert_refuses_on_every_path(
        "bin_to_flt past 2^53",
        &format!("nodule d;\nfn main() => Float = bin_to_flt(0b{lit});"),
    );
}

/// `flt_to_bin` round-trips small in-range integers back to `Binary{M}` — the target width read
/// from the witness operand only (its bits are ignored, exactly `width_cast`'s DN-41 shape).
#[test]
fn flt_to_bin_round_trips_three_way() {
    assert_three_way(
        "flt_to_bin 0",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(0.0, 0b0000_0000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000000".chars().map(|c| c == '1').collect()),
    );
    assert_three_way(
        "flt_to_bin 255",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(255.0, 0b0000_0000);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("11111111".chars().map(|c| c == '1').collect()),
    );
    // The witness's own value/bits are ignored — only its width matters.
    assert_three_way(
        "flt_to_bin witness bits ignored",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(7.0, 0b1111_1111);",
        &Repr::Binary { width: 8 },
        &Payload::Bits("00000111".chars().map(|c| c == '1').collect()),
    );
}

/// `flt_to_bin` refuses NaN, ±inf, a negative value, and a nonzero fractional part on every path
/// — never a silent coercion (ADR-040 §2.4; G2).
#[test]
fn flt_to_bin_refuses_the_never_silent_domain_on_every_path() {
    assert_refuses_on_every_path(
        "flt_to_bin NaN",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(flt_div(0.0, 0.0), 0b0000_0000);",
    );
    assert_refuses_on_every_path(
        "flt_to_bin +inf",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(flt_div(1.0, 0.0), 0b0000_0000);",
    );
    assert_refuses_on_every_path(
        "flt_to_bin negative",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(flt_neg(1.0), 0b0000_0000);",
    );
    assert_refuses_on_every_path(
        "flt_to_bin fractional",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(1.5, 0b0000_0000);",
    );
}

/// `flt_to_bin` refuses a magnitude that does not fit the witness's target width on every path —
/// never a silent truncation (ADR-040 §2.4/DN-41).
#[test]
fn flt_to_bin_refuses_out_of_target_width_on_every_path() {
    assert_refuses_on_every_path(
        "flt_to_bin 256 does not fit Binary{8}",
        "nodule d;\nfn main() => Binary{8} = flt_to_bin(256.0, 0b0000_0000);",
    );
}

/// A non-`Binary` `bin_to_flt` operand, and a non-`Float` `flt_to_bin` value operand, are
/// **static** never-silent refusals — caught at check time, never a runtime coercion.
#[test]
fn conversion_prims_reject_wrong_paradigm_statically() {
    for (src, needle) in [
        (
            "nodule d;\nfn main() => Float = bin_to_flt(1.5);",
            "must be a concrete `Binary{N}`",
        ),
        (
            "nodule d;\nfn main() => Binary{8} = flt_to_bin(0b0000_0001, 0b0000_0000);",
            "must be a `Float`",
        ),
        (
            "nodule d;\nfn main() => Binary{8} = flt_to_bin(1.0, 1.0);",
            "width witness must be a concrete `Binary{M}`",
        ),
    ] {
        let err =
            check_nodule(&parse(src).expect("parses")).expect_err(&format!("must reject: {src}"));
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "the refusal must name the offense.\n  src: {src}\n  want: {needle}\n  got: {msg}"
        );
    }
}

// ── CU-7 (RFC-0033 §4.2.2 / ADR-029): the fixed-width `trit.*` arithmetic is ALREADY arbitrary-
// width — a verify-first correction ────────────────────────────────────────────────────────────
//
// **Recon finding (mitigation #14 — verify against the codebase before implementing).** The trx2
// kickoff notes describe the runnable `trit.add`/`trit.sub`/`trit.mul`/`trit.neg` prims as capped
// at "~40 trits" and attribute that cap to `mycelium_core::ternary` being "i64-internal". Reading
// the actual kernel (`crates/mycelium-core/src/ternary/mod.rs::add`/`mul`) shows this is **not
// accurate for the arithmetic itself**: `add` is a digit-wise ripple-carry adder and `mul` a
// shifted-accumulation multiplier, both operating directly on `&[Trit]` with **no `i64` in the
// algorithm** — overflow is detected structurally (a nonzero final carry / nonzero high digits),
// not via an integer-range check. The **only** `i64`-capped pieces are the *conversion* utilities
// `max_magnitude`/`trits_to_int`/`int_to_trits` (used for decimal-literal encoding and test
// oracles), which are a genuinely different concern from the arithmetic RFC-0033 §4.2.2 names. The
// lexer's `0t…` trit-literal ([`crates/mycelium-l1/src/lexer.rs::lex_trit`]) likewise has no width
// cap (RFC-0037 D4). So **any width reachable via a trit-glyph literal already gets arbitrary-width
// arithmetic today** — RFC-0033 §4.2.2's fixed-width-side mandate is met by the *existing* code,
// zero-risk, and this three-way locks it in at 80 trits (double the assumed cap) so it can never
// silently regress.
//
// **What CU-7 does NOT cover (correctly deferred, not guessed).** A genuinely **growable** Ternary
// value form — no fixed `N`, an arbitrary-precision "BigInt"-shaped surface type built on
// `BigTernary` — is explicitly **out of scope**: RFC-0033's own Accepted-status changelog entry
// states "the value-model growth beyond the already-landed V0 `BigTernary` (M-754…M-757) is a
// post-1.0 wave" and couples it to the **content-address one-way doors** (the V1–V5 kernel
// implementation, M-760…M-784, deferred to post-1.0). Surfacing that growable form would touch the
// E20-1 content-address rehash this leaf was told to FLAG rather than guess (G2/VR-5) — see the
// CU-7 leaf report.

/// MSB-first `Trit` vector from a glyph string (`+`/`0`/`-`), mirroring [`bits`]'s Binary analogue.
fn trits(s: &str) -> Vec<mycelium_core::Trit> {
    s.chars()
        .map(|c| match c {
            '+' => mycelium_core::Trit::Pos,
            '-' => mycelium_core::Trit::Neg,
            '0' => mycelium_core::Trit::Zero,
            _ => panic!("trit glyph must be one of +/0/-, got {c:?}"),
        })
        .collect()
}

/// `add` at `Ternary{80}` — double the kickoff notes' assumed "~40-trit cap" — three-way. Operands
/// are all-zero except their low two digits (`+0` = 3, `0+` = 1); the sum `3 + 1 = 4` is `++` in
/// the low two digits (`4 = 1·3 + 1·1`).
#[test]
fn trit_add_beyond_the_claimed_40_trit_cap_three_way() {
    let zeros = "0".repeat(78);
    let a = format!("0t{zeros}+0"); // 80 trits, value 3
    let b = format!("0t{zeros}0+"); // 80 trits, value 1
    let expected_digits = format!("{zeros}++"); // 80 trits, value 4
    assert_three_way(
        "trit add beyond the assumed 40-trit cap",
        &format!("nodule d;\nfn main() => Ternary{{80}} = add({a}, {b});"),
        &Repr::Ternary { trits: 80 },
        &Payload::Trits(trits(&expected_digits)),
    );
}

/// `mul` at `Ternary{80}` three-way — the shifted-accumulation multiplier is likewise structural
/// (no `i64`), so it holds at this width too. `3 · 3 = 9`; balanced-ternary `9 = 1·9 + 0·3 + 0·1`
/// → low three digits `+00`.
#[test]
fn trit_mul_beyond_the_claimed_40_trit_cap_three_way() {
    let zeros = "0".repeat(77);
    let a = format!("0t{zeros}0+0"); // 80 trits, value 3
    let expected_digits = format!("{zeros}+00"); // 80 trits, value 9
    assert_three_way(
        "trit mul beyond the assumed 40-trit cap",
        &format!("nodule d;\nfn main() => Ternary{{80}} = mul({a}, {a});"),
        &Repr::Ternary { trits: 80 },
        &Payload::Trits(trits(&expected_digits)),
    );
}
