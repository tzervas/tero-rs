//! Differential tests for `std.select` (M-928, E29-1, kickoff `opp`) — the `.myc` port of the
//! structural selection/decision surface of `crates/mycelium-std-select` (RFC-0005; RFC-0016).
//!
//! # Harness design
//! Execution/comparison machinery lives in the shared [`harness`] fixture (M-925) — this file
//! supplies the nodule's `include_str!`, the per-op three-way cases, and — the row this port owns
//! per the harness doc (§4) — live comparisons against the **retained Rust oracle**,
//! `mycelium-std-select` (RFC-0031 D6; the crate is NOT retired). The M-928 DoD names **EXPLAIN
//! parity** explicitly: the oracle section below compares the `Explanation` **field by field**
//! (policy_name bytes, the full cost ranking, matched_rule, chosen_index, overridden) between the
//! `.myc` port and the Rust oracle on mirrored policies/inputs — at the oracle's declared
//! `storage_weight = 1.0` (the ported cost is raw declared storage bits — FLAG-select-2 in the
//! nodule).
//!
//! # Surface-check (D5 row 1) and substitutions
//! See `lib/std/select.myc`'s header for the full surface-check: the structural selection surface
//! (predicates, decision table, cost-in-bits, build validation, select/explain/override with the
//! mandatory Explanation, the guarantee matrix as data) is ported; FLAGged, not forced (VR-5/G2):
//! the content-addressed `PolicyRef`/`Explanation.policy` (kernel BLAKE3 — FLAG-select-1), the
//! f64 cost weight + eps predicate (FLAG-select-2), the site adapters + Packing/Decode candidates
//! (the D1 kernel-dispatch boundary — FLAG-select-3), the decode-site predicates/inputs fields
//! (FLAG-select-4), and `PolicyRegistry` (FLAG-select-5). `explain` is Result-typed here where
//! the Rust op is total (FLAG-select-6 — no private fields in `.myc` v0).
//!
//! # Honesty tags
//! - **`Exact`** — every ported op's decision semantics (a total predicate over exact integer
//!   metadata; C2), carried at the SAME strength as the Rust crate's matrix (VR-5).
//! - **`Declared`** — the guarantee-matrix row data (asserted, mirrors the Rust source's own
//!   structural tests).
//! - **`Empirical`** — the three-way differential agreement (L1-eval ≡ L0-interp ≡ AOT) AND the
//!   Rust-oracle EXPLAIN-parity differential below, both validated by trial on the programs in
//!   this file; neither is a machine-checked proof.

mod harness;

use mycelium_core::{binary::bits_to_int, CoreValue, Payload};

/// The std.select nodule source, loaded at compile time — the single source of truth.
const SELECT_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/std/select.myc"
));

/// Build a full test program by appending a typed driver to the nodule source.
fn program(driver: &str) -> String {
    harness::program(SELECT_SRC, driver)
}

/// Thin re-export of the shared [`harness::assert_three_way`] (std_error.rs pattern).
fn assert_three_way(label: &str, src: &str, expected_src: &str) {
    harness::assert_three_way(label, src, expected_src);
}

// ── shared `.myc` driver fragments ───────────────────────────────────────────────────────────────
//
// The standard two-candidate policy every case below builds on — mirrors the Rust oracle tests'
// `two_repr_policy`: name "two-repr.v1", candidates [Binary{8} (8 bits), Dense{8, F32} (256
// bits)], one Always → Cheapest rule, default arm 0. Built via `build` (so the FLAG-select-6
// DanglingChoice arms are unreachable — exactly the domain the Rust oracle's private-field
// invariant guarantees).
const MK_TWO_REPR: &str = "fn mk_cands() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_1000)), CLCons(CRepr(RDense(0b0000_0000_0000_1000, SkF32)), CLNil));\n\
fn mk_rules() => RuleList = RLCons(When(PAlways, ACheapest), RLNil);\n\
fn mk_pol() => Result[SelectionPolicy, PolicyError] = build(\"two-repr.v1\", mk_cands(), mk_rules(), 0b0000_0000);\n\
fn mk_inputs() => Inputs = SelIn(RBinary(0b0000_0000_0000_1000), GExact);\n";

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Three-way differential cases (L1-eval ≡ elaborate→L0-interp ≡ AOT), one section per obligation.
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ── C1 never-silent: build refusals ─────────────────────────────────────────────────────────────

/// `build` refuses an empty candidate set with the exact `NoCandidates` variant (C1) — a
/// selection over nothing is not total.
#[test]
fn build_refuses_empty_candidates() {
    let driver = "fn main() => Bool = match build(\"empty\", CLNil, RLNil, 0b0000_0000) { Ok(_) => False, Err(e) => match e { NoCandidates => True, IndexOutOfRange(_) => False } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("build(empty) refuses NoCandidates", &src, expected);
}

/// `build` refuses an out-of-range default arm with `IndexOutOfRange` carrying the offending
/// index (C1 — never a silent clamp). The Err payload passes through untouched (literal 5).
#[test]
fn build_refuses_out_of_range_default() {
    let driver = "fn mk_c() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_0001)), CLNil);\nfn main() => Binary{8} = match build(\"bad-default\", mk_c(), RLNil, 0b0000_0101) { Ok(_) => 0b1111_1110, Err(e) => match e { NoCandidates => 0b1111_1111, IndexOutOfRange(i) => i } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0101;";
    assert_three_way(
        "build(bad default) refuses IndexOutOfRange(5)",
        &src,
        expected,
    );
}

/// `build` refuses a rule whose `Choose(i)` points outside the candidate list (C1).
#[test]
fn build_refuses_out_of_range_choose() {
    let driver = "fn mk_c() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_0001)), CLNil);\nfn mk_r() => RuleList = RLCons(When(PAlways, AChoose(0b0110_0011)), RLNil);\nfn main() => Binary{8} = match build(\"bad-choose\", mk_c(), mk_r(), 0b0000_0000) { Ok(_) => 0b1111_1110, Err(e) => match e { NoCandidates => 0b1111_1111, IndexOutOfRange(i) => i } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0110_0011;";
    assert_three_way(
        "build(Choose(99)) refuses IndexOutOfRange(99)",
        &src,
        expected,
    );
}

/// `build` accepts a well-formed table — and the accepted policy carries its name (the
/// load-bearing Bytes payload) intact: probe byte 0 of the name ('t' = 0x74).
#[test]
fn build_accepts_well_formed_and_carries_name() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_1111, Ok(p) => bytes_get(pol_name(p), 0b0000_0000) }};"
    );
    let src = program(driver);
    // Independent Derived-provenance reference: bytes_get over a fresh literal (std_diag pattern).
    let expected = "nodule ref;\nfn main() => Binary{8} = bytes_get(\"two-repr.v1\", 0b0000_0000);";
    assert_three_way("build accepts + name byte 0 = 't'", &src, expected);
}

// ── C3 + cost model: Cheapest selects the minimum, in declared bits ─────────────────────────────

/// `Cheapest` selects Binary{8} (8 bits) over Dense{8, F32} (256 bits): the chosen candidate's
/// width passes through untouched.
#[test]
fn cheapest_selects_minimum_cost_candidate() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{16}} = match mk_pol() {{ Err(_) => 0b1111_1111_1111_1111, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_1111_1111_1110, Ok(s) => match sel_cand(s) {{ CRepr(r) => match r {{ RBinary(w) => w, _ => 0b1111_1111_1111_1101 }} }} }} }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{16} = 0b0000_0000_0000_1000;";
    assert_three_way("Cheapest picks Binary{8} over Dense{8,F32}", &src, expected);
}

/// The EXPLAIN record ranks EVERY candidate, not just the winner (RFC-0005 §2.2): the Dense
/// line's cost is dim × dtype_bits = 8 × 32 = 256, computed by the same `mul_s` composition.
#[test]
fn explanation_costs_covers_all_candidates_with_exact_bits() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{16}} = match mk_pol() {{ Err(_) => 0b1111_1111_1111_1111, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_1111_1111_1110, Ok(s) => match cost_at(expl_costs(sel_expl(s)), 0b0000_0001) {{ None => 0b1111_1111_1111_1101, Some(l) => line_cost(l) }} }} }};"
    );
    let src = program(driver);
    // Same primitive-op composition (Derived provenance): dim × dtype_bits(F32).
    let expected = "nodule ref;\nfn main() => Binary{16} = mul_s(0b0000_0000_0000_1000, 0b0000_0000_0010_0000);";
    assert_three_way("costs[1] = Dense bits = 8×32", &src, expected);
}

/// The ranking length equals the candidate count (C3 — complete, never truncated).
#[test]
fn explanation_costs_length_matches_candidates() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Bool = match mk_pol() {{ Err(_) => False, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => False, Ok(s) => match eq(costs_len(expl_costs(sel_expl(s))), cand_len(pol_cands(p))) {{ 0b1 => True, _ => False }} }} }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("costs length = candidate count", &src, expected);
}

/// `Cheapest` ties break deterministically to the LOWEST index (kernel `cheapest` verbatim):
/// two equal-cost Binary{8} candidates → chosen_index 0.
#[test]
fn cheapest_tie_breaks_to_lowest_index() {
    let driver = "fn mk_c() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_1000)), CLCons(CRepr(RBinary(0b0000_0000_0000_1000)), CLNil));\nfn mk_r() => RuleList = RLCons(When(PAlways, ACheapest), RLNil);\nfn main() => Binary{8} = match build(\"tie.v1\", mk_c(), mk_r(), 0b0000_0000) { Err(_) => 0b1111_1111, Ok(p) => match select(p, SelIn(RBinary(0b0000_0000_0000_1000), GExact)) { Err(_) => 0b1111_1110, Ok(s) => expl_chosen_index(sel_expl(s)) } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0000;";
    assert_three_way("Cheapest tie → lowest index", &src, expected);
}

// ── fixed declared precedence: first match wins; default arm ────────────────────────────────────

/// Rule predicates fire in table order — the first matching rule wins (RFC-0005 §2.3), recorded
/// as `matched_rule = Some(0)`.
#[test]
fn predicate_rule_first_match_wins() {
    let driver = "fn mk_c() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_0001)), CLCons(CRepr(RDense(0b0000_0000_0000_0001, SkF32)), CLNil));\nfn mk_r() => RuleList = RLCons(When(PSrcKindIs(KBinary), AChoose(0b0000_0000)), RLCons(When(PAlways, AChoose(0b0000_0001)), RLNil));\nfn main() => Option[Binary{8}] = match build(\"first-match.v1\", mk_c(), mk_r(), 0b0000_0000) { Err(_) => None, Ok(p) => match select(p, SelIn(RBinary(0b0000_0000_0000_0001), GExact)) { Err(_) => None, Ok(s) => expl_matched_rule(sel_expl(s)) } };";
    let src = program(driver);
    let expected = "nodule ref;\ntype Option[OptA] = Some(OptA) | None;\nfn main() => Option[Binary{8}] = Some(0b0000_0000);";
    assert_three_way(
        "first matching rule wins (matched_rule = Some(0))",
        &src,
        expected,
    );
}

/// The mandatory default arm fires when no rule matches: `matched_rule = None` and the default
/// index is chosen — a Dense source falls through a Binary-only rule.
#[test]
fn default_arm_fires_when_no_rule_matches() {
    let driver = "fn mk_c() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_0001)), CLCons(CRepr(RDense(0b0000_0000_0000_0001, SkF32)), CLNil));\nfn mk_r() => RuleList = RLCons(When(PSrcKindIs(KBinary), AChoose(0b0000_0000)), RLNil);\nfn main() => Binary{8} = match build(\"default-arm.v1\", mk_c(), mk_r(), 0b0000_0001) { Err(_) => 0b1111_1111, Ok(p) => match select(p, SelIn(RDense(0b0000_0000_0000_0001, SkF32), GExact)) { Err(_) => 0b1111_1110, Ok(s) => match expl_matched_rule(sel_expl(s)) { Some(_) => 0b1111_1101, None => expl_chosen_index(sel_expl(s)) } } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way(
        "default arm fires (matched None, chose default 1)",
        &src,
        expected,
    );
}

// ── predicate language semantics ────────────────────────────────────────────────────────────────

/// `PDtypeIs(SkF32)` matches a Dense F32 source (and the rule fires).
#[test]
fn dtype_predicate_matches_dense_f32() {
    let driver = "fn mk_i() => Inputs = SelIn(RDense(0b0000_0000_0000_0100, SkF32), GExact);\nfn main() => Bool = eval_pred(PDtypeIs(SkF32), mk_i());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("DtypeIs(F32) on Dense F32", &src, expected);
}

/// `PDtypeIs` asks for a Dense source: a Binary source is `False` (checked evidence, not its
/// absence — kernel semantics verbatim).
#[test]
fn dtype_predicate_false_on_non_dense() {
    let driver = "fn mk_i() => Inputs = SelIn(RBinary(0b0000_0000_0000_0100), GExact);\nfn main() => Bool = eval_pred(PDtypeIs(SkF32), mk_i());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("DtypeIs(F32) on Binary is False", &src, expected);
}

/// `PGuaranteeAtLeast`: Exact (rank 0) satisfies "at least Empirical" (rank 2); Declared
/// (rank 3) does not — the kernel's lattice-rank ≤, verbatim.
#[test]
fn guarantee_at_least_uses_lattice_rank() {
    let driver = "fn mk_e() => Inputs = SelIn(RBytes, GExact);\nfn mk_d() => Inputs = SelIn(RBytes, GDeclared);\nfn main() => Bool = match eval_pred(PGuaranteeAtLeast(GEmpirical), mk_e()) { False => False, True => match eval_pred(PGuaranteeAtLeast(GEmpirical), mk_d()) { True => False, False => True } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("GuaranteeAtLeast lattice rank", &src, expected);
}

/// `PDeclaredSparse` reads the source descriptor: sparse Vsa → True; dense Vsa → False.
#[test]
fn declared_sparse_reads_src_descriptor() {
    let driver = "fn main() => Bool = match eval_pred(PDeclaredSparse, SelIn(RVsaSparse(0b0000_0000_0000_0100), GExact)) { False => False, True => match eval_pred(PDeclaredSparse, SelIn(RVsaDense(0b0000_0000_0000_0100), GExact)) { True => False, False => True } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("DeclaredSparse on Vsa sparse/dense", &src, expected);
}

/// The connectives: `PAnd`/`POr`/`PNot` — `PNot(PAlways)` is False; `POr(PNot(PAlways),
/// PAlways)` is True; `PAnd(PAlways, PNot(PAlways))` is False. (The kernel's n-ary All/Any is
/// the nested binary form — nodule substitution note.)
#[test]
fn connectives_and_or_not() {
    let driver = "fn mk_i() => Inputs = SelIn(RBytes, GExact);\nfn main() => Bool = match eval_pred(PNot(PAlways), mk_i()) { True => False, False => match eval_pred(POr(PNot(PAlways), PAlways), mk_i()) { False => False, True => match eval_pred(PAnd(PAlways, PNot(PAlways)), mk_i()) { True => False, False => True } } };";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("PAnd/POr/PNot semantics", &src, expected);
}

/// The Seq cost recurses through the element repr (RFC-0032 D3): Seq{Binary{8}, 4} = 4 × 8 bits,
/// by the same `mul_s` composition.
#[test]
fn seq_cost_recurses_through_element() {
    let driver = "fn main() => Binary{16} = repr_bits(RSeq(RBinary(0b0000_0000_0000_1000), 0b0000_0000_0000_0100));";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{16} = mul_s(0b0000_0000_0000_0100, 0b0000_0000_0000_1000);";
    assert_three_way("Seq cost = len × elem bits", &src, expected);
}

// ── C3: override recorded, never hidden; C1: out-of-range refused ───────────────────────────────

/// `select_with_override` records `overridden = True` and the forced index in the Explanation —
/// the override state is never hidden (C3; RFC-0005 §2.4). Cheapest would choose 0; force 1.
#[test]
fn select_with_override_records_override_in_explanation() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_1111, Ok(p) => match select_with_override(p, mk_inputs(), 0b0000_0001) {{ Err(_) => 0b1111_1110, Ok(s) => match expl_overridden(sel_expl(s)) {{ False => 0b1111_1101, True => expl_chosen_index(sel_expl(s)) }} }} }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0000_0001;";
    assert_three_way(
        "override records overridden=True + forced index",
        &src,
        expected,
    );
}

/// A normal `select` emits `overridden = False` — the override state is recorded faithfully.
#[test]
fn select_emits_not_overridden() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Bool = match mk_pol() {{ Err(_) => True, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => True, Ok(s) => expl_overridden(sel_expl(s)) }} }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = False;";
    assert_three_way("normal select: overridden=False", &src, expected);
}

/// `select_with_override` refuses an out-of-range forced index with `OverrideOutOfRange` — never
/// a snap to the nearest legal choice (C1). The refusal carries the offending index AND the
/// candidate count (the count is the same `add_u` spine-walk composition).
#[test]
fn select_with_override_refuses_out_of_range() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_1111, Ok(p) => match select_with_override(p, mk_inputs(), 0b0110_0011) {{ Ok(_) => 0b1111_1110, Err(e) => match e {{ DanglingChoice(_) => 0b1111_1101, OverrideOutOfRange(i, _) => i }} }} }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Binary{8} = 0b0110_0011;";
    assert_three_way(
        "override(99) refused with the offending index",
        &src,
        expected,
    );
}

// ── C2/C4 determinism + explain ≡ select consistency ────────────────────────────────────────────

/// `select` is deterministic: two calls on the same `(policy, inputs)` agree on the chosen
/// index (C2/C4 — same inputs → same record).
#[test]
fn select_is_deterministic() {
    // Both selections are bound via match patterns before the `eq` (the AOT env-machine refuses
    // a nullary-fn application directly in prim-operand position — a driver-shape workaround,
    // not a semantics change).
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Bool = match mk_pol() {{ Err(_) => False, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => False, Ok(s1) => match select(p, mk_inputs()) {{ Err(_) => False, Ok(s2) => match eq(expl_chosen_index(sel_expl(s1)), expl_chosen_index(sel_expl(s2))) {{ 0b1 => True, _ => False }} }} }} }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("select deterministic", &src, expected);
}

/// `explain` and `select` produce the same decision content for the same `(policy, inputs)`:
/// chosen index and winner cost agree (C3 — the record is re-derivable).
#[test]
fn explain_and_select_consistent() {
    let driver = &format!(
        "{MK_TWO_REPR}fn sidx() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_1111, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_1110, Ok(s) => expl_chosen_index(sel_expl(s)) }} }};\nfn eidx() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0111, Ok(p) => match explain(p, mk_inputs()) {{ Err(_) => 0b1111_0110, Ok(e) => expl_chosen_index(e) }} }};\nfn main() => Bool = match eq(sidx(), eidx()) {{ 0b1 => True, _ => False }};"
    );
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("explain ≡ select decision content", &src, expected);
}

// ── guarantee matrix (RFC-0016 §4.5) as checked data ────────────────────────────────────────────

/// The matrix has exactly 4 rows (one per ported op — the Rust matrix's `policy_ref` row is
/// FLAG-select-1, not fabricated).
#[test]
fn matrix_has_four_rows() {
    let driver = "fn main() => Binary{8} = matrix_len(guarantee_matrix());";
    let src = program(driver);
    // Same composition: four add_u(1, ·) steps over the spine, ending at the literal 0.
    let expected = "nodule ref;\nfn main() => Binary{8} = add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, add_u(0b0000_0001, 0b0000_0000))));";
    assert_three_way("matrix has 4 rows", &src, expected);
}

/// Every matrix row carries the `GExact` tag — the honest tag for a total predicate over exact
/// metadata (VR-5; the Rust suite's `matrix_all_rows_exact`, ported).
#[test]
fn matrix_all_rows_exact() {
    let driver = "fn main() => Bool = matrix_all_exact(guarantee_matrix());";
    let src = program(driver);
    let expected = "nodule ref;\nfn main() => Bool = True;";
    assert_three_way("matrix all rows GExact", &src, expected);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Rust-oracle differential (D5 row 4 + the M-928 EXPLAIN-parity DoD) — wired against the
// RETAINED `mycelium-std-select` crate (RFC-0031 D6: the crate is NOT retired). Each case builds
// the SAME policy/inputs on both sides (oracle `storage_weight = 1.0` — the ported cost is raw
// declared storage bits, FLAG-select-2) and compares the Explanation FIELD BY FIELD: policy_name
// (byte-for-byte — the load-bearing string payload), the full cost ranking, matched_rule,
// chosen_index, chosen-candidate identity, and the overridden flag. The `Explanation.policy`
// content hash is FLAG-select-1 (kernel BLAKE3) — compared via its carrier `policy_name` here,
// never fabricated.
// ══════════════════════════════════════════════════════════════════════════════════════════════

use mycelium_core::{Meta, Provenance, Repr, ScalarKind};
use mycelium_std_select::{
    build as oracle_build, explain as oracle_explain, select as oracle_select,
    select_with_override as oracle_select_with_override, Action, Candidate, CostModel, Explanation,
    Predicate, Rule, SelectError, SelectionInputs, SelectionPolicy,
};

/// Run `driver`'s `main` through the L1 evaluator and return the resulting [`CoreValue`]
/// (std_error.rs `eval_byte` pattern, generalized to any observable). The three-way obligation
/// is covered by the cases above; this helper only bridges to the Rust oracle.
fn eval_core(driver: &str) -> CoreValue {
    use mycelium_l1::elab::build_registry;
    use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

    let src = program(driver);
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("parse failed: {e}")))
        .unwrap_or_else(|e| panic!("check failed: {e}"));
    let mono = monomorphize(&env, "main").unwrap_or_else(|e| panic!("monomorphize failed: {e}"));
    let registry = build_registry(&mono).unwrap_or_else(|e| panic!("build_registry failed: {e}"));
    let val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("L1-eval failed: {e}"));
    val.to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("result is outside the r3 data fragment"))
}

/// Decode a `Binary{N}` result to its integer (MSB-first two's-complement — the same
/// [`bits_to_int`] codec the L1 evaluator/AOT paths use).
fn eval_int(driver: &str) -> i64 {
    let cv = eval_core(driver);
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Binary repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bits(bits) => bits_to_int(bits),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Decode a `Bytes` result to its raw bytes (the load-bearing string payload).
fn eval_bytes(driver: &str) -> Vec<u8> {
    let cv = eval_core(driver);
    let repr = cv
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Bytes repr value, got {cv:?}"));
    match repr.payload() {
        Payload::Bytes(bytes) => bytes.clone(),
        other => panic!("expected a Bytes payload, got {other:?}"),
    }
}

/// The Rust-oracle mirror of [`MK_TWO_REPR`]: same name, candidates, rule, default arm —
/// `storage_weight = 1.0` (the declared-bits unit the port carries; FLAG-select-2).
fn oracle_two_repr_policy() -> SelectionPolicy {
    oracle_build(
        "two-repr.v1",
        vec![
            Candidate::Repr(Repr::Binary { width: 8 }),
            Candidate::Repr(Repr::Dense {
                dim: 8,
                dtype: ScalarKind::F32,
            }),
        ],
        vec![Rule {
            when: Predicate::Always,
            action: Action::Cheapest,
        }],
        0,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect("well-formed two-repr policy")
}

/// The oracle inputs mirroring `mk_inputs()`: a Binary{8} source with Exact-guarantee metadata.
fn oracle_inputs() -> SelectionInputs {
    SelectionInputs::from_meta(Repr::Binary { width: 8 }, &Meta::exact(Provenance::Root))
}

/// The oracle explanation for the standard select — shared by the parity cases below.
fn oracle_explanation() -> Explanation {
    let (_, expl) =
        oracle_select(&oracle_two_repr_policy(), &oracle_inputs()).expect("oracle select");
    expl
}

/// EXPLAIN parity: `chosen_index` — the `.myc` port and the Rust oracle choose the same
/// candidate index (Cheapest → Binary{8} at index 0).
#[test]
fn oracle_explain_parity_chosen_index() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_1111, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_1110, Ok(s) => expl_chosen_index(sel_expl(s)) }} }};"
    );
    let expl = oracle_explanation();
    assert_eq!(
        eval_int(driver),
        i64::try_from(expl.chosen_index).expect("index fits"),
        "chosen_index must match the Rust oracle (EXPLAIN parity, M-928 DoD)"
    );
}

/// EXPLAIN parity: `matched_rule` — both sides record rule 0 (the Always → Cheapest row).
/// Encoding: the driver maps `Some(i) => i`, `None => -1` (a TEST-driver encoding only, matched
/// on the oracle side by `map_or(-1, …)` — the library carries the honest Option/None).
#[test]
fn oracle_explain_parity_matched_rule() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_0001, Ok(s) => match expl_matched_rule(sel_expl(s)) {{ Some(ri) => ri, None => 0b1111_1111 }} }} }};"
    );
    let expl = oracle_explanation();
    let oracle_encoded = expl
        .matched_rule
        .map_or(-1i64, |r| i64::try_from(r).expect("rule index fits"));
    assert_eq!(
        eval_int(driver),
        oracle_encoded,
        "matched_rule must match the Rust oracle (EXPLAIN parity, M-928 DoD)"
    );
}

/// EXPLAIN parity: `overridden` — False on the normal path, on both sides. Encoding:
/// True => 1, False => 0 (test-driver encoding; the library carries the honest Bool).
#[test]
fn oracle_explain_parity_overridden_flag() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_0001, Ok(s) => match expl_overridden(sel_expl(s)) {{ True => 0b0000_0001, False => 0b0000_0000 }} }} }};"
    );
    let expl = oracle_explanation();
    assert_eq!(
        eval_int(driver),
        i64::from(expl.overridden),
        "overridden flag must match the Rust oracle (EXPLAIN parity, M-928 DoD)"
    );
}

/// EXPLAIN parity: the FULL cost ranking — every candidate's cost, in candidate order, equals
/// the oracle's (at storage_weight 1.0 the oracle costs are integral: Binary{8} = 8.0 bits,
/// Dense{8,F32} = 256.0 bits — asserted integral before comparing).
#[test]
fn oracle_explain_parity_full_cost_ranking() {
    let expl = oracle_explanation();
    assert_eq!(expl.costs.len(), 2, "oracle ranks both candidates");
    for (i, line) in expl.costs.iter().enumerate() {
        assert!(
            line.cost.fract() == 0.0,
            "oracle cost at weight 1.0 must be integral bits, got {}",
            line.cost
        );
        let idx_lit = format!("0b0000_{:04b}", i);
        let driver = &format!(
            "{MK_TWO_REPR}fn main() => Binary{{16}} = match mk_pol() {{ Err(_) => 0b1111_1111_1111_1111, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_1111_1111_1110, Ok(s) => match cost_at(expl_costs(sel_expl(s)), {idx_lit}) {{ None => 0b1111_1111_1111_1101, Some(l) => line_cost(l) }} }} }};"
        );
        #[allow(clippy::cast_possible_truncation)]
        let oracle_bits = line.cost as i64;
        assert_eq!(
            eval_int(driver),
            oracle_bits,
            "cost line {i} must match the Rust oracle in declared bits (EXPLAIN parity)"
        );
    }
}

/// EXPLAIN parity: `policy_name` — byte-for-byte identical to the oracle's (the load-bearing
/// string payload the M-928 brief calls out; `Explanation.policy` itself is FLAG-select-1).
#[test]
fn oracle_explain_parity_policy_name_bytes() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Bytes = match mk_pol() {{ Err(_) => \"ERR-BUILD\", Ok(p) => match select(p, mk_inputs()) {{ Err(_) => \"ERR-SELECT\", Ok(s) => expl_policy_name(sel_expl(s)) }} }};"
    );
    let expl = oracle_explanation();
    assert_eq!(
        eval_bytes(driver),
        expl.policy_name.as_bytes(),
        "policy_name must match the Rust oracle byte-for-byte (EXPLAIN parity, M-928 DoD)"
    );
}

/// EXPLAIN parity: `explain` — the standalone EXPLAIN op derives the same decision content as
/// the oracle's `explain` (chosen_index compared; both sides also assert explain ≡ select
/// internally — the three-way case above and the oracle's own contract).
#[test]
fn oracle_explain_op_parity() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match explain(p, mk_inputs()) {{ Err(_) => 0b1111_0001, Ok(e) => expl_chosen_index(e) }} }};"
    );
    let oracle_expl = oracle_explain(&oracle_two_repr_policy(), &oracle_inputs());
    assert_eq!(
        eval_int(driver),
        i64::try_from(oracle_expl.chosen_index).expect("index fits"),
        "explain must derive the same choice as the Rust oracle (EXPLAIN parity)"
    );
}

/// EXPLAIN parity on the override path: forced index 1 → both sides record overridden = true
/// and chosen_index = 1 (the override is IN the record, never hidden — C3).
#[test]
fn oracle_override_parity_records_override() {
    let driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select_with_override(p, mk_inputs(), 0b0000_0001) {{ Err(_) => 0b1111_0001, Ok(s) => match expl_overridden(sel_expl(s)) {{ False => 0b1111_0010, True => expl_chosen_index(sel_expl(s)) }} }} }};"
    );
    let (_, oracle_expl) =
        oracle_select_with_override(&oracle_two_repr_policy(), &oracle_inputs(), 1)
            .expect("oracle override");
    assert!(oracle_expl.overridden, "oracle records the override");
    assert_eq!(
        eval_int(driver),
        i64::try_from(oracle_expl.chosen_index).expect("index fits"),
        "override path: chosen_index must match the Rust oracle (EXPLAIN parity)"
    );
}

/// Refusal parity: an out-of-range override is refused on BOTH sides with the same offending
/// index and candidate count (`OverrideOutOfRange {index: 99, candidates: 2}` — C1, never a
/// snap to the nearest legal choice).
#[test]
fn oracle_override_out_of_range_refusal_parity() {
    let idx_driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select_with_override(p, mk_inputs(), 0b0110_0011) {{ Ok(_) => 0b1111_0001, Err(e) => match e {{ DanglingChoice(_) => 0b1111_0010, OverrideOutOfRange(i, _) => i }} }} }};"
    );
    let n_driver = &format!(
        "{MK_TWO_REPR}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select_with_override(p, mk_inputs(), 0b0110_0011) {{ Ok(_) => 0b1111_0001, Err(e) => match e {{ DanglingChoice(_) => 0b1111_0010, OverrideOutOfRange(_, n) => n }} }} }};"
    );
    let err = oracle_select_with_override(&oracle_two_repr_policy(), &oracle_inputs(), 99)
        .expect_err("oracle refuses the out-of-range override");
    match err {
        SelectError::OverrideOutOfRange { index, candidates } => {
            assert_eq!(
                eval_int(idx_driver),
                i64::try_from(index).expect("fits"),
                "refused index must match the Rust oracle"
            );
            assert_eq!(
                eval_int(n_driver),
                i64::try_from(candidates).expect("fits"),
                "candidate count in the refusal must match the Rust oracle"
            );
        }
        other => panic!("oracle must refuse with OverrideOutOfRange, got {other:?}"),
    }
}

/// Default-arm parity: a Dense source falling through a Binary-only rule lands on the default
/// arm on BOTH sides — matched_rule None (encoded -1) and chosen_index = the default (1).
#[test]
fn oracle_default_arm_parity() {
    const MK_DEFAULT: &str = "fn mk_cands() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_0001)), CLCons(CRepr(RDense(0b0000_0000_0000_0001, SkF32)), CLNil));\n\
fn mk_rules() => RuleList = RLCons(When(PSrcKindIs(KBinary), AChoose(0b0000_0000)), RLNil);\n\
fn mk_pol() => Result[SelectionPolicy, PolicyError] = build(\"default-arm.v1\", mk_cands(), mk_rules(), 0b0000_0001);\n\
fn mk_inputs() => Inputs = SelIn(RDense(0b0000_0000_0000_0001, SkF32), GExact);\n";
    let mr_driver = &format!(
        "{MK_DEFAULT}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_0001, Ok(s) => match expl_matched_rule(sel_expl(s)) {{ Some(ri) => ri, None => 0b1111_1111 }} }} }};"
    );
    let ci_driver = &format!(
        "{MK_DEFAULT}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_0001, Ok(s) => expl_chosen_index(sel_expl(s)) }} }};"
    );

    let oracle_pol = oracle_build(
        "default-arm.v1",
        vec![
            Candidate::Repr(Repr::Binary { width: 1 }),
            Candidate::Repr(Repr::Dense {
                dim: 1,
                dtype: ScalarKind::F32,
            }),
        ],
        vec![Rule {
            when: Predicate::SrcKindIs(mycelium_std_select::ParadigmKind::Binary),
            action: Action::Choose(0),
        }],
        1,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect("well-formed default-arm policy");
    let oracle_in = SelectionInputs::from_meta(
        Repr::Dense {
            dim: 1,
            dtype: ScalarKind::F32,
        },
        &Meta::exact(Provenance::Root),
    );
    let (_, oracle_expl) = oracle_select(&oracle_pol, &oracle_in).expect("oracle select");

    let oracle_mr = oracle_expl
        .matched_rule
        .map_or(-1i64, |r| i64::try_from(r).expect("fits"));
    assert_eq!(
        eval_int(mr_driver),
        oracle_mr,
        "default arm: matched_rule must match the Rust oracle (None on both sides)"
    );
    assert_eq!(
        eval_int(ci_driver),
        i64::try_from(oracle_expl.chosen_index).expect("fits"),
        "default arm: chosen_index must match the Rust oracle"
    );
}

/// Tie-break parity: two equal-cost candidates → BOTH sides choose the lowest index (the
/// deterministic tie rule — RFC-0005 §2.3 determinism).
#[test]
fn oracle_tie_break_parity() {
    const MK_TIE: &str = "fn mk_cands() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_1000)), CLCons(CRepr(RBinary(0b0000_0000_0000_1000)), CLNil));\n\
fn mk_rules() => RuleList = RLCons(When(PAlways, ACheapest), RLNil);\n\
fn mk_pol() => Result[SelectionPolicy, PolicyError] = build(\"tie.v1\", mk_cands(), mk_rules(), 0b0000_0000);\n\
fn mk_inputs() => Inputs = SelIn(RBinary(0b0000_0000_0000_1000), GExact);\n";
    let driver = &format!(
        "{MK_TIE}fn main() => Binary{{8}} = match mk_pol() {{ Err(_) => 0b1111_0000, Ok(p) => match select(p, mk_inputs()) {{ Err(_) => 0b1111_0001, Ok(s) => expl_chosen_index(sel_expl(s)) }} }};"
    );

    let oracle_pol = oracle_build(
        "tie.v1",
        vec![
            Candidate::Repr(Repr::Binary { width: 8 }),
            Candidate::Repr(Repr::Binary { width: 8 }),
        ],
        vec![Rule {
            when: Predicate::Always,
            action: Action::Cheapest,
        }],
        0,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect("well-formed tie policy");
    let (_, oracle_expl) = oracle_select(&oracle_pol, &oracle_inputs()).expect("oracle select");

    assert_eq!(
        eval_int(driver),
        i64::try_from(oracle_expl.chosen_index).expect("fits"),
        "tie must break to the lowest index on both sides (EXPLAIN parity)"
    );
}

/// Build-refusal parity: an out-of-range default arm is refused on BOTH sides with the same
/// offending index (`PolicyError::IndexOutOfRange {index: 5}` — C1).
#[test]
fn oracle_build_refusal_parity() {
    let driver = "fn mk_c() => CandList = CLCons(CRepr(RBinary(0b0000_0000_0000_0001)), CLNil);\nfn main() => Binary{8} = match build(\"bad-default\", mk_c(), RLNil, 0b0000_0101) { Ok(_) => 0b1111_1110, Err(e) => match e { NoCandidates => 0b1111_1111, IndexOutOfRange(i) => i } };";
    let err = oracle_build(
        "bad-default",
        vec![Candidate::Repr(Repr::Binary { width: 1 })],
        vec![],
        5,
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect_err("oracle refuses the out-of-range default");
    match err {
        mycelium_std_select::PolicyError::IndexOutOfRange { index } => {
            assert_eq!(
                eval_int(driver),
                i64::try_from(index).expect("fits"),
                "build refusal index must match the Rust oracle"
            );
        }
        other => panic!("oracle must refuse with IndexOutOfRange, got {other:?}"),
    }
}
