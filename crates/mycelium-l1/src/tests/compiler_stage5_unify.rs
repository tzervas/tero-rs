//! M-740 Stage 5, increment 3 (M-1008; DN-26 §7.3 row 5 / §9 flag-1) — the self-hosted
//! `compiler.semcore` unification + type-resolution layer: the LIVE-ORACLE differential gate for
//! checkty.rs's next type-algebra leaves ported into `lib/compiler/semcore.myc`:
//!   * `unify`      (checkty.rs::unify)      — one-sided unification, accumulating a substitution.
//!   * `resolve_ty` (checkty.rs::resolve_ty) — surface `TypeRef` → checked `Ty` (+ guarantee).
//!   * `tuple_type_name`/`tuple_ctor_name`/`synthetic_tuple_data` (checkty.rs, M-826) — the
//!     synthetic-tuple naming + on-demand `DataInfo`.
//!   * `dec_u32` (FLAG-semcore-15, semcore-local) — the base-10 renderer the tuple names need; its
//!     verdict is asserted against Rust's own `format!("{n}")` (a decimal-string witness).
//!
//! **Live-oracle posture (VR-5).** Every case calls the REAL Rust `checkty` fn on a fixture and
//! asserts the `.myc` port's verdict against THAT computed result — never a hand-derived constant.
//! `unify` mutates a `BTreeMap`; the port threads + returns the substitution (FLAG-semcore-13), so
//! the witness applies `subst_ty(decl, s)` under BOTH the port's and the oracle's resulting map and
//! compares the two `Ty`s via the in-file `ty_eq` (a non-vacuity probe below pins that this witness
//! discriminates — a degenerate always-`True` would pass hollowly). The conflict path (a second,
//! conflicting binding) is asserted `Err` against the oracle's `Err` (the never-silent mismatch,
//! G2/VR-5). All five fns are `pub`/`pub(crate)`, reachable from this in-crate `src/tests/` module
//! (no visibility change to `checkty.rs`; only this module + its one `mod` line were added).
//!
//! M-981 applies as in every prior self-hosted-compiler-scale stage: only the L1-eval leg is
//! exercised (small synthetic fixtures, not a corpus program — the L0/AOT leg's marginal value is
//! low relative to its eval-depth cost, M-987).

use crate::ast::{BaseType, Scalar, Sparsity, Strength, TypeRef, WidthRef};
use crate::checkty::{
    check_nodule, resolve_ty, subst_ty, synthetic_tuple_data, tuple_ctor_name, tuple_type_name,
    unify, CtorInfo, DataInfo, Ty, Width,
};
use crate::elab::build_registry;
use crate::eval::Evaluator;
use crate::mono::monomorphize;
use crate::parse;
use mycelium_core::Payload;
use std::collections::BTreeMap;

/// Extract a `Binary{N}` `CoreValue`'s bits as a `u32` (MSB-first) — the established convention.
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

/// The driver prelude: the `DataInfo`/`CtorInfo` structural-equality probes (support scaffolding
/// faithful to `DataInfo: PartialEq`, mirroring the FLAG-semcore-12 `ty_eq` posture) used only to
/// compare `synthetic_tuple_data`'s output in-language, plus a `Strength`-option equality for the
/// `resolve_ty` guarantee passthrough.
fn driver_prelude() -> String {
    r#"
// names_eq_probe: structural equality over two Vec[Bytes] (parameter-name lists).
fn names_eq_probe(a: Vec[Bytes], b: Vec[Bytes]) => Bool =
  match a {
    Nil => match b { Nil => True, Cons(_, _) => False },
    Cons(ha, ta) => match b { Nil => False, Cons(hb, tb) => and_(beq(ha, hb), names_eq_probe(ta, tb)) }
  };

// ci_eq_probe / ci_list_eq_probe: structural equality over CtorInfo (name + field types via ty_list_eq).
fn ci_eq_probe(a: CtorInfo, b: CtorInfo) => Bool =
  match a { CI(na, fa) => match b { CI(nb, fb) => and_(beq(na, nb), ty_list_eq(fa, fb)) } };

fn ci_list_eq_probe(a: Vec[CtorInfo], b: Vec[CtorInfo]) => Bool =
  match a {
    Nil => match b { Nil => True, Cons(_, _) => False },
    Cons(ha, ta) => match b { Nil => False, Cons(hb, tb) => and_(ci_eq_probe(ha, hb), ci_list_eq_probe(ta, tb)) }
  };

// di_eq_probe: structural equality over DataInfo (name + params + ctors).
fn di_eq_probe(a: DataInfo, b: DataInfo) => Bool =
  match a { DI(na, pa, ca) => match b { DI(nb, pb, cb) =>
    and_(beq(na, nb), and_(names_eq_probe(pa, pb), ci_list_eq_probe(ca, cb))) } };

// strength_eq_probe / opt_strength_eq_probe: the resolve_ty guarantee passthrough witness.
fn strength_eq_probe(a: Strength, b: Strength) => Bool =
  match a {
    GExact => match b { GExact => True, _ => False },
    GProven => match b { GProven => True, _ => False },
    GEmpirical => match b { GEmpirical => True, _ => False },
    GDeclared => match b { GDeclared => True, _ => False }
  };

fn opt_strength_eq_probe(a: Option[Strength], b: Option[Strength]) => Bool =
  match a {
    None => match b { None => True, Some(_) => False },
    Some(sa) => match b { None => False, Some(sb) => strength_eq_probe(sa, sb) }
  };
"#
    .to_owned()
}

fn program(driver: &str) -> String {
    format!("{SEMCORE_SRC}\n{}\n{driver}", driver_prelude())
}

/// L1-eval-only assertion (the M-981 convention): parse → check → monomorphize → build_registry →
/// eval `main` → compare the `Binary{32}` result to the LIVE-ORACLE-derived `expected_u32`.
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

// ── Rust → `.myc` fixture encoders ────────────────────────────────────────────────────────────────

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

fn encode_names(names: &[String]) -> String {
    let mut s = String::from("Nil");
    for n in names.iter().rev() {
        s = format!("Cons({}, {})", encode_bytes(n), s);
    }
    s
}

fn encode_widthref(w: &WidthRef) -> String {
    match w {
        WidthRef::Lit(n) => format!("WLit({})", encode_u32(*n)),
        WidthRef::Name(v) => format!("WName({})", encode_bytes(v)),
    }
}

fn encode_strength(s: Strength) -> &'static str {
    match s {
        Strength::Exact => "GExact",
        Strength::Proven => "GProven",
        Strength::Empirical => "GEmpirical",
        Strength::Declared => "GDeclared",
    }
}

fn encode_opt_strength(g: Option<Strength>) -> String {
    match g {
        Some(s) => format!("Some({})", encode_strength(s)),
        None => "None".to_owned(),
    }
}

fn encode_base(b: &BaseType) -> String {
    match b {
        BaseType::Binary(w) => format!("KwBinary({})", encode_widthref(w)),
        BaseType::Ternary(w) => format!("KwTernary({})", encode_widthref(w)),
        BaseType::Dense(d, s) => format!("KwDense({}, {})", encode_u32(*d), encode_scalar(*s)),
        BaseType::Vsa {
            model,
            dim,
            sparsity,
        } => format!(
            "Vsa({}, {}, {})",
            encode_bytes(model),
            encode_u32(*dim),
            encode_sparsity(sparsity)
        ),
        BaseType::Substrate(t) => format!("KwSubstrate({})", encode_bytes(t)),
        BaseType::Seq { elem, len } => {
            format!("KwSeq({}, {})", encode_typeref(elem), encode_u32(*len))
        }
        BaseType::Bytes => "KwBytes".to_owned(),
        BaseType::Float => "KwFloat".to_owned(),
        BaseType::Named(n, args) => {
            format!("Named({}, {})", encode_bytes(n), encode_typeref_list(args))
        }
        BaseType::Fn(a, b) => format!("FnArrow({}, {})", encode_typeref(a), encode_typeref(b)),
        BaseType::Tuple(elems) => format!("Tuple({})", encode_typeref_list(elems)),
        BaseType::Ambient(_) => {
            unreachable!("Ambient BaseType is not exercised by the resolve_ty differential")
        }
    }
}

fn encode_typeref(t: &TypeRef) -> String {
    format!(
        "TR({}, {})",
        encode_base(&t.base),
        encode_opt_strength(t.guarantee)
    )
}

fn encode_typeref_list(ts: &[TypeRef]) -> String {
    let mut s = String::from("Nil");
    for t in ts.iter().rev() {
        s = format!("Cons({}, {})", encode_typeref(t), s);
    }
    s
}

fn encode_ctor(c: &CtorInfo) -> String {
    format!(
        "CI({}, {})",
        encode_bytes(&c.name),
        encode_ty_list(&c.fields)
    )
}

fn encode_ctor_list(cs: &[CtorInfo]) -> String {
    let mut s = String::from("Nil");
    for c in cs.iter().rev() {
        s = format!("Cons({}, {})", encode_ctor(c), s);
    }
    s
}

fn encode_datainfo(d: &DataInfo) -> String {
    format!(
        "DI({}, {}, {})",
        encode_bytes(&d.name),
        encode_names(&d.params),
        encode_ctor_list(&d.ctors)
    )
}

fn encode_datainfo_list(ds: &[DataInfo]) -> String {
    let mut s = String::from("Nil");
    for d in ds.iter().rev() {
        s = format!("Cons({}, {})", encode_datainfo(d), s);
    }
    s
}

/// A `.myc` substitution assoc-list `Vec[Pair[Bytes, Ty]]` from a `BTreeMap<String, Ty>` seed.
fn encode_subst(s: &BTreeMap<String, Ty>) -> String {
    let mut out = String::from("Nil");
    for (k, v) in s.iter().rev() {
        out = format!("Cons(Pr({}, {}), {})", encode_bytes(k), encode_ty(v), out);
    }
    out
}

// Small fixture constructors keeping the test bodies to `assert over a case`.
fn tref(base: BaseType) -> TypeRef {
    TypeRef {
        base,
        guarantee: None,
    }
}
fn tref_g(base: BaseType, g: Strength) -> TypeRef {
    TypeRef {
        base,
        guarantee: Some(g),
    }
}
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
/// A `DataInfo` fixture: `name<params>` with a single constructor of the given field types.
fn di(name: &str, params: &[&str], ctor: &str, fields: Vec<Ty>) -> DataInfo {
    DataInfo {
        name: name.to_owned(),
        params: params.iter().map(|p| (*p).to_owned()).collect(),
        ctors: vec![CtorInfo {
            name: ctor.to_owned(),
            fields,
        }],
    }
}
fn registry(ds: &[DataInfo]) -> BTreeMap<String, DataInfo> {
    ds.iter().map(|d| (d.name.clone(), d.clone())).collect()
}

// ── per-leaf live-oracle assertions ───────────────────────────────────────────────────────────────

/// `dec_u32`: the `.myc` render must equal Rust's `format!("{n}")` (a decimal-string witness).
fn assert_dec_u32(label: &str, n: u32) {
    let expected = format!("{n}");
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match bytes_eq(dec_u32({}), {}) {{ 0b1 => one32(), _ => zero32() }};\n",
        encode_u32(n),
        encode_bytes(&expected)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `tuple_type_name` / `tuple_ctor_name`: the `.myc` name must equal the live `checkty` name.
fn assert_tuple_names(label: &str, n: u32) {
    let want_ty = tuple_type_name(n as usize);
    let want_ctor = tuple_ctor_name(n as usize);
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match and_(beq(tuple_type_name({}), {}), beq(tuple_ctor_name({}), {})) {{ True => one32(), False => zero32() }};\n",
        encode_u32(n),
        encode_bytes(&want_ty),
        encode_u32(n),
        encode_bytes(&want_ctor)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `synthetic_tuple_data`: the `.myc` `DataInfo` must equal the live `checkty::synthetic_tuple_data`.
fn assert_synthetic_tuple(label: &str, n: u32) {
    let want = synthetic_tuple_data(n as usize);
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match di_eq_probe(synthetic_tuple_data({}), {}) {{ True => one32(), False => zero32() }};\n",
        encode_u32(n),
        encode_datainfo(&want)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `resolve_ty` (accept): the `.myc` resolved `Ty` (and its guarantee) must match the live oracle.
fn assert_resolve_ok(label: &str, ds: &[DataInfo], tyvars: &[&str], t: &TypeRef) {
    let map = registry(ds);
    let tv: Vec<String> = tyvars.iter().map(|s| (*s).to_owned()).collect();
    let (want_ty, want_g) = resolve_ty("test", &map, &tv, t)
        .unwrap_or_else(|e| panic!("{label}: oracle resolve_ty errored: {e}"));
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match resolve_ty({}, {}, {}) {{\n    Ok(pr) => match pr {{ Pr(t, g) => match and_(ty_eq(t, {}), opt_strength_eq_probe(g, {})) {{ True => one32(), False => zero32() }} }},\n    Err(_) => zero32()\n  }};\n",
        encode_datainfo_list(ds),
        encode_names(&tv),
        encode_typeref(t),
        encode_ty(&want_ty),
        encode_opt_strength(want_g)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `resolve_ty` (refuse): the `.myc` `resolve_ty` must return `Err` exactly where the oracle does.
fn assert_resolve_err(label: &str, ds: &[DataInfo], tyvars: &[&str], t: &TypeRef) {
    let map = registry(ds);
    let tv: Vec<String> = tyvars.iter().map(|s| (*s).to_owned()).collect();
    assert!(
        resolve_ty("test", &map, &tv, t).is_err(),
        "{label}: oracle resolve_ty unexpectedly succeeded (fixture should refuse)"
    );
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match resolve_ty({}, {}, {}) {{ Err(_) => one32(), Ok(_) => zero32() }};\n",
        encode_datainfo_list(ds),
        encode_names(&tv),
        encode_typeref(t)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `unify` (accept): run the oracle to build the substitution, apply it to `decl`, and assert the
/// `.myc` `unify`'s own resulting substitution reproduces that same substituted `Ty`.
fn assert_unify_ok(label: &str, decl: &Ty, actual: &Ty, seed: &BTreeMap<String, Ty>) {
    let mut s = seed.clone();
    unify("test", decl, actual, &mut s)
        .unwrap_or_else(|e| panic!("{label}: oracle unify errored: {e}"));
    let expected_ty = subst_ty(decl, &s);
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match unify({}, {}, {}) {{\n    Ok(s) => match ty_eq(subst_ty({}, s), {}) {{ True => one32(), False => zero32() }},\n    Err(_) => zero32()\n  }};\n",
        encode_ty(decl),
        encode_ty(actual),
        encode_subst(seed),
        encode_ty(decl),
        encode_ty(&expected_ty)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `unify` (refuse): the never-silent conflict/mismatch path — the `.myc` `unify` must return `Err`
/// exactly where the oracle does.
fn assert_unify_err(label: &str, decl: &Ty, actual: &Ty, seed: &BTreeMap<String, Ty>) {
    let mut s = seed.clone();
    assert!(
        unify("test", decl, actual, &mut s).is_err(),
        "{label}: oracle unify unexpectedly succeeded (fixture should refuse)"
    );
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match unify({}, {}, {}) {{ Err(_) => one32(), Ok(_) => zero32() }};\n",
        encode_ty(decl),
        encode_ty(actual),
        encode_subst(seed)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Structural gate: `semcore.myc` (with the increment-3 additions) parses and type-checks green.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn semcore_unify_parses_and_checks() {
    let nodule = parse(SEMCORE_SRC).unwrap_or_else(|e| panic!("semcore.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("semcore.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Non-vacuity probe: the subst_ty/ty_eq witness must DISCRIMINATE — a deliberately-wrong expected
// `Ty` yields 0, so the accept-case witnesses above are not hollow always-1 assertions.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn unify_witness_discriminates() {
    // `unify(Var A, Bytes)` binds A→Bytes, so `subst_ty(Var A, s)` = Bytes. Comparing against the
    // WRONG type `Float` must yield 0 (proving the equality witness really tests the value).
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match unify({}, {}, Nil) {{\n    Ok(s) => match ty_eq(subst_ty({}, s), {}) {{ True => one32(), False => zero32() }},\n    Err(_) => zero32()\n  }};\n",
        encode_ty(&var("A")),
        encode_ty(&Ty::Bytes),
        encode_ty(&var("A")),
        encode_ty(&Ty::Float),
    );
    assert_l1_only_u32("wrong_expected_yields_zero", &program(&driver), 0);
    // di_eq_probe must discriminate too: two distinct arities' synthetic data are not equal.
    let d2 = synthetic_tuple_data(2);
    let driver2 = format!(
        "fn main() => Binary{{32}} =\n  match di_eq_probe(synthetic_tuple_data({}), {}) {{ True => one32(), False => zero32() }};\n",
        encode_u32(3),
        encode_datainfo(&d2)
    );
    assert_l1_only_u32("di_eq_discriminates", &program(&driver2), 0);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// dec_u32 (LIVE — Rust `format!("{n}")`): single/multi-digit + boundary values.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn dec_u32_cases() {
    for n in [
        0u32, 1, 2, 9, 10, 11, 99, 100, 256, 1000, 65535, 4096, 1_000_000,
    ] {
        assert_dec_u32(&format!("dec_{n}"), n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// synthetic-tuple naming + on-demand DataInfo (LIVE — checkty::{tuple_type_name,tuple_ctor_name,
// synthetic_tuple_data}).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn tuple_helpers_cases() {
    for n in [2u32, 3, 4, 5, 10] {
        assert_tuple_names(&format!("tuple_names_{n}"), n);
        assert_synthetic_tuple(&format!("tuple_data_{n}"), n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// resolve_ty (LIVE — checkty::resolve_ty): every BaseType arm + the tyvar/arity/unknown refusals.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn resolve_ty_accept_cases() {
    let list = di("List", &["A"], "MkList", vec![var("A")]);
    let pair = di("Pair", &["A", "B"], "MkPair", vec![var("A"), var("B")]);
    let bool_t = di("Bool", &[], "True", vec![]);
    let tuple2 = synthetic_tuple_data(2);
    let ds = [list.clone(), pair.clone(), bool_t.clone(), tuple2.clone()];

    // Primitive reprs.
    assert_resolve_ok(
        "binary_lit",
        &ds,
        &[],
        &tref(BaseType::Binary(WidthRef::Lit(8))),
    );
    assert_resolve_ok(
        "binary_name",
        &ds,
        &["N"],
        &tref(BaseType::Binary(WidthRef::Name("N".to_owned()))),
    );
    assert_resolve_ok(
        "ternary_lit",
        &ds,
        &[],
        &tref(BaseType::Ternary(WidthRef::Lit(6))),
    );
    assert_resolve_ok("dense", &ds, &[], &tref(BaseType::Dense(1024, Scalar::F32)));
    assert_resolve_ok(
        "substrate",
        &ds,
        &[],
        &tref(BaseType::Substrate("file".to_owned())),
    );
    assert_resolve_ok("bytes", &ds, &[], &tref(BaseType::Bytes));
    assert_resolve_ok("float", &ds, &[], &tref(BaseType::Float));
    // VSA: the surface `MAP_I` model id canonicalizes to the kernel `MAP-I` (vsa_kernel_model_id).
    assert_resolve_ok(
        "vsa_canonicalized",
        &ds,
        &[],
        &tref(BaseType::Vsa {
            model: "MAP_I".to_owned(),
            dim: 256,
            sparsity: Sparsity::Dense,
        }),
    );
    // Seq recurses under the same tyvar scope.
    assert_resolve_ok(
        "seq_of_binary",
        &ds,
        &[],
        &tref(BaseType::Seq {
            elem: Box::new(tref(BaseType::Binary(WidthRef::Lit(8)))),
            len: 4,
        }),
    );
    // A bare name that is a tyvar → Ty::Var; a bare name that is a nullary data type → Ty::Data.
    assert_resolve_ok(
        "tyvar",
        &ds,
        &["T"],
        &tref(BaseType::Named("T".to_owned(), vec![])),
    );
    assert_resolve_ok(
        "nullary_data",
        &ds,
        &[],
        &tref(BaseType::Named("Bool".to_owned(), vec![])),
    );
    // Applied data types (arity checked), nested + width-var element.
    assert_resolve_ok(
        "applied_list",
        &ds,
        &[],
        &tref(BaseType::Named(
            "List".to_owned(),
            vec![tref(BaseType::Binary(WidthRef::Lit(8)))],
        )),
    );
    assert_resolve_ok(
        "applied_pair_tyvar_arg",
        &ds,
        &["A"],
        &tref(BaseType::Named(
            "Pair".to_owned(),
            vec![
                tref(BaseType::Named("A".to_owned(), vec![])),
                tref(BaseType::Bytes),
            ],
        )),
    );
    // Function arrow resolves both sides recursively.
    assert_resolve_ok(
        "fn_arrow",
        &ds,
        &[],
        &tref(BaseType::Fn(
            Box::new(tref(BaseType::Bytes)),
            Box::new(tref(BaseType::Binary(WidthRef::Lit(8)))),
        )),
    );
    // Tuple resolves to the synthetic Tuple$2 data type (must be registered).
    assert_resolve_ok(
        "tuple2",
        &ds,
        &[],
        &tref(BaseType::Tuple(vec![
            tref(BaseType::Bytes),
            tref(BaseType::Float),
        ])),
    );
    // The guarantee annotation passes through unchanged.
    assert_resolve_ok(
        "guarantee_passthrough",
        &ds,
        &[],
        &tref_g(BaseType::Binary(WidthRef::Lit(8)), Strength::Proven),
    );
}

#[test]
fn resolve_ty_refuse_cases() {
    let list = di("List", &["A"], "MkList", vec![var("A")]);
    let ds = [list];
    // Unknown type name.
    assert_resolve_err(
        "unknown_type",
        &ds,
        &[],
        &tref(BaseType::Named("Nope".to_owned(), vec![])),
    );
    // Arity mismatch: List takes 1 argument, given 2.
    assert_resolve_err(
        "arity_mismatch",
        &ds,
        &[],
        &tref(BaseType::Named(
            "List".to_owned(),
            vec![tref(BaseType::Bytes), tref(BaseType::Float)],
        )),
    );
    // A tuple whose synthetic type is NOT registered (no Tuple$2 in the registry) → refuse.
    assert_resolve_err(
        "unregistered_tuple",
        &ds,
        &[],
        &tref(BaseType::Tuple(vec![
            tref(BaseType::Bytes),
            tref(BaseType::Float),
        ])),
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// unify (LIVE — checkty::unify): binding, structural descent, width-var carrier, and the
// never-silent conflict/mismatch refusals (G2/VR-5).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn unify_accept_cases() {
    let empty = BTreeMap::new();
    // Type-var binding.
    assert_unify_ok("var_bytes", &var("A"), &Ty::Bytes, &empty);
    // Structural Data descent, binding a nested var.
    assert_unify_ok(
        "data_nested",
        &data("List", vec![var("A")]),
        &data("List", vec![bin(8)]),
        &empty,
    );
    // Fn: param and return unify independently.
    assert_unify_ok(
        "fn_both_sides",
        &Ty::Fn(Box::new(var("A")), Box::new(var("B"))),
        &Ty::Fn(Box::new(Ty::Bytes), Box::new(Ty::Float)),
        &empty,
    );
    // Seq: equal length, element unifies.
    assert_unify_ok(
        "seq_elem",
        &Ty::Seq(Box::new(var("E")), 3),
        &Ty::Seq(Box::new(Ty::Bytes), 3),
        &empty,
    );
    // Width-var carrier (Binary + Ternary), and a var-var width binding.
    assert_unify_ok("binary_width_lit", &bin_var("N"), &bin(16), &empty);
    assert_unify_ok(
        "ternary_width_lit",
        &tern_var("N"),
        &Ty::Ternary(Width::Lit(3)),
        &empty,
    );
    assert_unify_ok("binary_width_var_var", &bin_var("N"), &bin_var("M"), &empty);
    // Two occurrences of the same var with the SAME actual → consistent (no conflict).
    assert_unify_ok(
        "repeated_var_consistent",
        &data("Pair", vec![var("A"), var("A")]),
        &data("Pair", vec![Ty::Bytes, Ty::Bytes]),
        &empty,
    );
    // A concrete pair with no vars: unifies iff already equal (the `_ if decl == actual` arm).
    assert_unify_ok(
        "concrete_equal",
        &data("Box", vec![Ty::Bytes]),
        &data("Box", vec![Ty::Bytes]),
        &empty,
    );
    // A pre-seeded map with a compatible binding is preserved.
    let mut seed = BTreeMap::new();
    seed.insert("A".to_owned(), Ty::Bytes);
    assert_unify_ok("seeded_compatible", &var("A"), &Ty::Bytes, &seed);
}

#[test]
fn unify_refuse_cases() {
    let empty = BTreeMap::new();
    // The conflict path: the same var forced to two distinct types (never-silent — G2/VR-5).
    assert_unify_err(
        "conflicting_binding",
        &data("Pair", vec![var("A"), var("A")]),
        &data("Pair", vec![Ty::Bytes, Ty::Float]),
        &empty,
    );
    // Pre-seeded conflict: A already bound to Bytes, unifying against Float refuses.
    let mut seed = BTreeMap::new();
    seed.insert("A".to_owned(), Ty::Bytes);
    assert_unify_err("seeded_conflict", &var("A"), &Ty::Float, &seed);
    // Head mismatch (distinct data names).
    assert_unify_err(
        "data_name_mismatch",
        &data("List", vec![Ty::Bytes]),
        &data("Vec", vec![Ty::Bytes]),
        &empty,
    );
    // Arity mismatch on Data.
    assert_unify_err(
        "data_arity_mismatch",
        &data("Pair", vec![var("A"), var("B")]),
        &data("Pair", vec![Ty::Bytes]),
        &empty,
    );
    // Seq length mismatch (never a silent length coercion).
    assert_unify_err(
        "seq_len_mismatch",
        &Ty::Seq(Box::new(var("E")), 3),
        &Ty::Seq(Box::new(Ty::Bytes), 4),
        &empty,
    );
    // Cross-paradigm width var (Binary var vs Ternary lit) — falls through to mismatch.
    assert_unify_err(
        "cross_paradigm_width",
        &bin_var("N"),
        &Ty::Ternary(Width::Lit(8)),
        &empty,
    );
    // Width-var conflict: N bound to 8 then forced to 16.
    assert_unify_err(
        "width_conflict",
        &data("Pair", vec![bin_var("N"), bin_var("N")]),
        &data("Pair", vec![bin(8), bin(16)]),
        &empty,
    );
    // Primitive kind mismatch.
    assert_unify_err("prim_mismatch", &Ty::Bytes, &Ty::Float, &empty);
}
