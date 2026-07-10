//! M-740 Stage 5, increment 2 (M-1007; DN-26 §7.3 row 5 / §9 flag-1) — the self-hosted
//! `compiler.semcore` type-algebra quartet: the LIVE-ORACLE differential gate for checkty.rs's four
//! pure type-algebra leaves ported into `lib/compiler/semcore.myc`:
//!   * `has_var`   (checkty.rs::has_var)   — does a `Ty` mention any abstract type/width variable?
//!   * `type_head` (checkty.rs::type_head) — the stage-1 coherence head-key of a `Ty` (or `None`).
//!   * `subst_ty`  (checkty.rs::subst_ty)  — type-variable / width-variable substitution.
//!   * `param_subst` (checkty.rs::param_subst) — build the parameter→argument substitution map.
//!
//! **Live-oracle posture (the honest accounting VR-5 requires).** Every case calls the REAL Rust
//! `checkty` fn on a `Ty` fixture and asserts the `.myc` port's verdict against THAT computed result
//! — never a hand-derived constant. The four fns are `pub`/`pub(crate)`, reachable from this in-crate
//! `src/tests/` module (the established white-box `src/tests/*.rs` convention — no visibility change
//! to `checkty.rs`; only this test module and its one `mod` line in `src/tests/mod.rs` were added,
//! exactly as increment 1's `compiler_stage5_semcore.rs` did). `subst_ty`/`param_subst` produce a
//! `Ty`; the `.myc` result is compared against the oracle's `Ty` via the in-file `ty_eq`
//! (FLAG-semcore-12 scaffolding faithful to `Ty: PartialEq`) so the comparison stays in-language.
//!
//! M-981 applies as in every prior self-hosted-compiler-scale stage: only the L1-eval leg is
//! exercised (every input is a small synthetic `Ty`, not a corpus program — the marginal value of an
//! L0/AOT three-way leg is low relative to its eval-depth cost, M-987).

use crate::ast::{Scalar, Sparsity};
use crate::checkty::{check_nodule, has_var, param_subst, subst_ty, type_head, Ty, Width};
use crate::elab::build_registry;
use crate::eval::Evaluator;
use crate::mono::monomorphize;
use crate::parse;
use mycelium_core::Payload;

/// Extract a `Binary{N}` `CoreValue`'s bits as a `u32` (MSB-first) — the established convention from
/// every prior stage's harness (`compiler_stage1.rs`'s own `core_bits_as_u32`).
fn core_bits_as_u32(v: &mycelium_core::CoreValue) -> u32 {
    let repr_val = v
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr CoreValue, got {v:?}"));
    match repr_val.payload() {
        Payload::Bits(bits) => bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b)),
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

const SEMCORE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/semcore.myc"
));

/// The driver prelude: the single `type_head` result-encoder every `type_head` scenario shares
/// (the substitution/has_var scenarios need no extra encoder — they use the ported fns directly).
fn driver_prelude() -> String {
    r#"
// th_matches: 1 iff the ported `type_head` agrees with the oracle on Some/None AND (when Some) on
// the head bytes — `want_some` = 1 if the oracle is Some else 0, `want` = the oracle's head (or "").
// (Named `th_matches`, not `head_matches` — semcore.myc already defines a decision.rs `head_matches`.)
fn th_matches(res: Option[Bytes], want_some: Binary{32}, want: Bytes) => Binary{32} =
  match res {
    None => match eq(want_some, zero32()) { 0b1 => one32(), _ => zero32() },
    Some(h) => match eq(want_some, one32()) {
      0b1 => match bytes_eq(h, want) { 0b1 => one32(), _ => zero32() },
      _ => zero32()
    }
  };
"#
    .to_owned()
}

fn program(driver: &str) -> String {
    format!("{SEMCORE_SRC}\n{}\n{driver}", driver_prelude())
}

/// L1-eval-only assertion (the M-981 convention every self-hosted-compiler-scale stage uses):
/// parse → check → monomorphize → build_registry → eval `main` → compare the `Binary{32}` result to
/// the LIVE-ORACLE-derived `expected_u32`.
fn assert_l1_only_u32(label: &str, src: &str, expected_u32: u32) {
    let env = check_nodule(&parse(src).unwrap_or_else(|e| panic!("{label}: parse failed: {e}")))
        .unwrap_or_else(|e| panic!("{label}: check failed: {e}"));
    let mono =
        monomorphize(&env, "main").unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{label}: build_registry failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call("main", vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let l1_core = l1_val
        .to_core(&mono, &registry)
        .unwrap_or_else(|| panic!("{label}: L1 result is outside the r3 data fragment"));
    let got = core_bits_as_u32(&l1_core);
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the LIVE-ORACLE expected value {expected_u32}"
    );
}

// ── Rust → `.myc` fixture encoders (a Rust `Ty` value → its `semcore.myc` constructor expression) ──

/// A `Binary{32}` `.myc` literal (`0bxxxx_..._xxxx`, 32 bits grouped by 4 — the semcore driver style).
fn encode_u32(n: u32) -> String {
    let mut s = String::from("0b");
    for (count, i) in (0..32).rev().enumerate() {
        if count != 0 && count % 4 == 0 {
            s.push('_');
        }
        s.push(if (n >> i) & 1 == 1 { '1' } else { '0' });
    }
    s
}

/// A `Bytes` `.myc` literal — Rust's debug quoting is exactly the double-quoted form the lexer reads.
fn encode_bytes(s: &str) -> String {
    format!("{s:?}")
}

fn encode_scalar(s: Scalar) -> &'static str {
    match s {
        Scalar::F16 => "SF16",
        Scalar::Bf16 => "SBf16",
        Scalar::F32 => "SF32",
        Scalar::F64 => "SF64",
    }
}

fn encode_sparsity(sp: &Sparsity) -> String {
    match sp {
        Sparsity::Dense => "SpDense".to_owned(),
        Sparsity::Sparse(k) => format!("SpSparse({})", encode_u32(*k)),
    }
}

fn encode_width(w: &Width) -> String {
    match w {
        Width::Lit(n) => format!("WdLit({})", encode_u32(*n)),
        Width::Var(v) => format!("WdVar({})", encode_bytes(v)),
    }
}

fn encode_ty(t: &Ty) -> String {
    match t {
        Ty::Binary(w) => format!("TyBinary({})", encode_width(w)),
        Ty::Ternary(w) => format!("TyTernary({})", encode_width(w)),
        Ty::Dense(d, s) => format!("TyDense({}, {})", encode_u32(*d), encode_scalar(*s)),
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => format!(
            "TyVsa({}, {}, {})",
            encode_bytes(model),
            encode_u32(*dim),
            encode_sparsity(sparsity)
        ),
        Ty::Data(n, args) => format!("TyData({}, {})", encode_bytes(n), encode_ty_list(args)),
        Ty::Substrate(t) => format!("TySubstrate({})", encode_bytes(t)),
        Ty::Seq(elem, n) => format!("TySeq({}, {})", encode_ty(elem), encode_u32(*n)),
        Ty::Bytes => "TyBytes".to_owned(),
        Ty::Float => "TyFloat".to_owned(),
        Ty::Var(v) => format!("TyVar({})", encode_bytes(v)),
        Ty::Fn(a, r) => format!("TyFn({}, {})", encode_ty(a), encode_ty(r)),
    }
}

fn encode_ty_list(ts: &[Ty]) -> String {
    let mut s = String::from("Nil");
    for t in ts.iter().rev() {
        s = format!("Cons({}, {})", encode_ty(t), s);
    }
    s
}

fn encode_bytes_list(names: &[&str]) -> String {
    let mut s = String::from("Nil");
    for n in names.iter().rev() {
        s = format!("Cons({}, {})", encode_bytes(n), s);
    }
    s
}

// ── per-leaf live-oracle assertions ───────────────────────────────────────────────────────────────

/// `has_var`: the `.myc` verdict (`True`→1 / `False`→0) must equal the live `checkty::has_var(&ty)`.
fn assert_has_var(label: &str, ty: &Ty) {
    let expected = u32::from(has_var(ty));
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match has_var({}) {{ True => one32(), False => zero32() }};\n",
        encode_ty(ty)
    );
    assert_l1_only_u32(label, &program(&driver), expected);
}

/// `type_head`: the `.myc` `type_head(ty)` must match the live `checkty::type_head(&ty)` Some/None +
/// head bytes (encoded via `head_matches`, which returns 1 on full agreement).
fn assert_type_head(label: &str, ty: &Ty) {
    let (want_some, want) = match type_head(ty) {
        Some(h) => (1u32, h),
        None => (0u32, String::new()),
    };
    let driver = format!(
        "fn main() => Binary{{32}} =\n  th_matches(type_head({}), {}, {});\n",
        encode_ty(ty),
        encode_u32(want_some),
        encode_bytes(&want)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `subst_ty` ∘ `param_subst`: build the substitution from `params`/`args` and apply it to `ty`; the
/// `.myc` result must be structurally equal (`ty_eq`) to the live `checkty` composition's `Ty`. This
/// exercises `param_subst` (building the map) AND `subst_ty` (consuming it) against one oracle.
fn assert_subst(label: &str, ty: &Ty, params: &[&str], args: &[Ty]) {
    let param_strings: Vec<String> = params.iter().map(|s| (*s).to_owned()).collect();
    let map = param_subst(&param_strings, args);
    let expected_ty = subst_ty(ty, &map);
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match ty_eq(subst_ty({}, param_subst({}, {})), {}) {{ True => one32(), False => zero32() }};\n",
        encode_ty(ty),
        encode_bytes_list(params),
        encode_ty_list(args),
        encode_ty(&expected_ty)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `ty_eq` discrimination probe (FLAG-semcore-12 scaffolding validation): asserts `ty_eq` is a real
/// structural equality — reflexive on equal `Ty`s, discriminating on distinct ones — so the
/// `subst_ty`/`param_subst` witness above is NOT vacuous (a degenerate always-`True` `ty_eq` would
/// pass those hollowly). `expected_equal` here is a direct assertion about the helper, not an oracle
/// differential (ty_eq is support scaffolding, pinned directly).
fn assert_ty_eq(label: &str, a: &Ty, b: &Ty, expected_equal: bool) {
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match ty_eq({}, {}) {{ True => one32(), False => zero32() }};\n",
        encode_ty(a),
        encode_ty(b)
    );
    assert_l1_only_u32(label, &program(&driver), u32::from(expected_equal));
}

// Small fixture constructors keeping the test bodies to `assert over a case` (test-layout rule).
fn data(n: &str, args: Vec<Ty>) -> Ty {
    Ty::Data(n.to_owned(), args)
}
fn var(n: &str) -> Ty {
    Ty::Var(n.to_owned())
}
fn bin(n: u32) -> Ty {
    Ty::Binary(Width::Lit(n))
}
fn bin_var(n: &str) -> Ty {
    Ty::Binary(Width::Var(n.to_owned()))
}
fn tern_var(n: &str) -> Ty {
    Ty::Ternary(Width::Var(n.to_owned()))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Structural gate: `semcore.myc` (with the increment-2 additions) parses and type-checks green.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn semcore_typealg_parses_and_checks() {
    let nodule = parse(SEMCORE_SRC).unwrap_or_else(|e| panic!("semcore.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("semcore.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// ty_eq discrimination (scaffolding validation — keeps the subst_ty/param_subst witness non-vacuous).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn typealg_ty_eq_discriminates() {
    // Equal pairs → True (reflexivity across variant kinds).
    assert_ty_eq("eq_bytes", &Ty::Bytes, &Ty::Bytes, true);
    assert_ty_eq("eq_binary8", &bin(8), &bin(8), true);
    assert_ty_eq(
        "eq_data",
        &data("List", vec![bin(8)]),
        &data("List", vec![bin(8)]),
        true,
    );
    // Distinct pairs → False (real discrimination on kind / width / name / arg / var-name).
    assert_ty_eq("neq_bytes_float", &Ty::Bytes, &Ty::Float, false);
    assert_ty_eq("neq_binary_width", &bin(8), &bin(16), false);
    assert_ty_eq(
        "neq_data_name",
        &data("List", vec![bin(8)]),
        &data("Vec", vec![bin(8)]),
        false,
    );
    assert_ty_eq(
        "neq_data_arg",
        &data("List", vec![bin(8)]),
        &data("List", vec![bin(16)]),
        false,
    );
    assert_ty_eq("neq_ctor_kind", &var("A"), &Ty::Bytes, false);
    assert_ty_eq("neq_var_name", &var("A"), &var("B"), false);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// has_var (LIVE — `checkty::has_var`): every arm class (var, width-var, nested, Fn/Seq, concrete).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn typealg_has_var_cases() {
    assert_has_var("var", &var("A"));
    assert_has_var("binary_lit_concrete", &bin(8));
    assert_has_var("binary_width_var", &bin_var("N"));
    assert_has_var("ternary_width_var", &tern_var("M"));
    assert_has_var("data_nested_var", &data("List", vec![var("A")]));
    assert_has_var(
        "data_all_concrete",
        &data("Pair", vec![Ty::Bytes, Ty::Float]),
    );
    assert_has_var(
        "data_nested_deep_var",
        &data("Box", vec![data("List", vec![var("A")])]),
    );
    assert_has_var(
        "fn_rhs_var",
        &Ty::Fn(Box::new(Ty::Bytes), Box::new(var("B"))),
    );
    assert_has_var(
        "fn_all_concrete",
        &Ty::Fn(Box::new(Ty::Bytes), Box::new(Ty::Float)),
    );
    assert_has_var("seq_elem_var", &Ty::Seq(Box::new(var("E")), 4));
    assert_has_var("seq_elem_concrete", &Ty::Seq(Box::new(Ty::Bytes), 4));
    assert_has_var(
        "vsa_concrete",
        &Ty::Vsa {
            model: "MAP-I".to_owned(),
            dim: 256,
            sparsity: Sparsity::Dense,
        },
    );
    assert_has_var("bytes_concrete", &Ty::Bytes);
    // PR #1231 review nit: the remaining top-level concrete arms (Dense/Substrate/Float → no var).
    assert_has_var("dense_concrete", &Ty::Dense(512, Scalar::F16));
    assert_has_var("substrate_concrete", &Ty::Substrate("file".to_owned()));
    assert_has_var("float_concrete", &Ty::Float);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// type_head (LIVE — `checkty::type_head`): the head-key of each kind, plus the `Var`/`Fn` → None arms.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn typealg_type_head_cases() {
    assert_type_head("binary", &bin(8));
    assert_type_head("ternary", &tern_var("N"));
    assert_type_head("dense", &Ty::Dense(1024, Scalar::F32));
    assert_type_head(
        "vsa",
        &Ty::Vsa {
            model: "FHRR".to_owned(),
            dim: 512,
            sparsity: Sparsity::Sparse(8),
        },
    );
    assert_type_head("substrate_named", &Ty::Substrate("file".to_owned()));
    assert_type_head("seq", &Ty::Seq(Box::new(Ty::Bytes), 2));
    assert_type_head("bytes", &Ty::Bytes);
    assert_type_head("float", &Ty::Float);
    assert_type_head("data_named", &data("List", vec![Ty::Bytes]));
    assert_type_head("var_is_none", &var("A"));
    assert_type_head(
        "fn_is_none",
        &Ty::Fn(Box::new(Ty::Bytes), Box::new(Ty::Bytes)),
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// subst_ty ∘ param_subst (LIVE — `checkty::{param_subst, subst_ty}`): type-var, width-var carrier,
// nested, Fn/Seq, and the unbound-var passthrough.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn typealg_subst_ty_cases() {
    // Type-variable substitution.
    assert_subst("var_to_bytes", &var("A"), &["A"], &[Ty::Bytes]);
    assert_subst(
        "data_arg_var_to_binary",
        &data("List", vec![var("A")]),
        &["A"],
        &[bin(8)],
    );
    assert_subst(
        "fn_both_sides",
        &Ty::Fn(Box::new(var("A")), Box::new(var("B"))),
        &["A", "B"],
        &[Ty::Bytes, Ty::Float],
    );
    assert_subst(
        "seq_elem",
        &Ty::Seq(Box::new(var("E")), 3),
        &["E"],
        &[Ty::Bytes],
    );
    // Width-variable substitution via the DN-42/M-753 carrier convention (`v -> TyBinary(WdLit(n))`).
    assert_subst("binary_width_var", &bin_var("N"), &["N"], &[bin(16)]);
    assert_subst("ternary_width_var", &tern_var("N"), &["N"], &[bin(3)]);
    // Unbound var: `Z` is not in the map, so it is left unchanged (still in scope).
    assert_subst("unbound_var_passthrough", &var("Z"), &["A"], &[Ty::Bytes]);
    // Concrete type is unaffected by any substitution.
    assert_subst(
        "concrete_unchanged",
        &data("Pair", vec![Ty::Bytes, Ty::Float]),
        &["A"],
        &[bin(8)],
    );
    // Multi-param map, deep nesting.
    assert_subst(
        "multi_param_nested",
        &data("Map", vec![var("K"), data("List", vec![var("V")])]),
        &["K", "V"],
        &[Ty::Bytes, bin(32)],
    );
    // PR #1231 review nit: the top-level concrete arms subst_ty leaves untouched (Dense/Vsa/
    // Substrate/concrete-width — no `Var`/width-var to replace).
    assert_subst(
        "dense_unchanged",
        &Ty::Dense(256, Scalar::F32),
        &["A"],
        &[Ty::Bytes],
    );
    assert_subst(
        "vsa_unchanged",
        &Ty::Vsa {
            model: "MAP-I".to_owned(),
            dim: 256,
            sparsity: Sparsity::Dense,
        },
        &["A"],
        &[Ty::Bytes],
    );
    assert_subst(
        "substrate_unchanged",
        &Ty::Substrate("net".to_owned()),
        &["A"],
        &[Ty::Bytes],
    );
    assert_subst("concrete_width_unchanged", &bin(8), &["A"], &[Ty::Bytes]);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// param_subst positional binding (LIVE): the i-th param binds the i-th arg (probe via subst_ty), and
// a length mismatch yields the partial map Rust's `.zip` produces (an over-long param list, an unbound
// tail → passthrough).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn typealg_param_subst_positional_and_mismatch() {
    // Second param binds the second arg.
    assert_subst(
        "probe_second_param",
        &var("B"),
        &["A", "B"],
        &[Ty::Bytes, Ty::Float],
    );
    // More params than args: the trailing param `C` gets no binding → its probe passes through.
    assert_subst(
        "more_params_than_args_bound",
        &var("A"),
        &["A", "B", "C"],
        &[bin(8), Ty::Float],
    );
    assert_subst(
        "more_params_than_args_unbound_tail",
        &var("C"),
        &["A", "B", "C"],
        &[bin(8), Ty::Float],
    );
    // More args than params: the extra arg is dropped (zip stops at the shorter side).
    assert_subst(
        "more_args_than_params",
        &var("A"),
        &["A"],
        &[Ty::Bytes, Ty::Float],
    );
}
