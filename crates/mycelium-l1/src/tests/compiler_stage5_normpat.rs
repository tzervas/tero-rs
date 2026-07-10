//! M-740 Stage 5, increment 6 (M-1011; DN-26 §7.3 row 5 / §9 flag-1) — the self-hosted
//! `compiler.semcore` literal + pattern typing leaves: the LIVE-ORACLE differential gate for
//! checkty.rs's `lit_ty_of`, `literal_key`, and `normalize_pattern` ported into
//! `lib/compiler/semcore.myc`.
//!
//! **Live-oracle posture (VR-5).** Every case calls the REAL Rust `checkty` fn on a fixture and
//! asserts the `.myc` port's verdict against THAT computed result. `lit_ty_of` (`pub(crate)`) is
//! tested directly (Ty via `ty_eq`, refusals as `Err`); `normalize_pattern` (`pub(crate)`) is tested
//! on its returned matrix `Pat` (via a `pat_eq_probe`) AND its binder set (name/type/occurrence via
//! `bind_list_eq_probe`); `literal_key` is module-private, exercised TRANSITIVELY through the
//! `Pat::Lit` arm. The normalized `Pat` is the same shape `useful` (increment 1) consumes, so this
//! tightens that gate transitively. Only this in-crate `src/tests/` module + its one `mod` line were
//! added — no visibility change to `checkty.rs`. `infer_type` is DEFERRED (FLAG-semcore-20: a thin
//! wrapper over the un-ported inference engine, not a leaf).
//!
//! M-981 applies: only the L1-eval leg is exercised (small synthetic fixtures, not a corpus program).

use crate::ast::{Literal, Pattern, Scalar, Sparsity};
use crate::checkty::{
    check_nodule, lit_ty_of, normalize_pattern, synthetic_tuple_data, CtorInfo, DataInfo, Ty, Width,
};
use crate::elab::build_registry;
use crate::eval::Evaluator;
use crate::mono::monomorphize;
use crate::parse;
use crate::usefulness::Pat;
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

/// The driver prelude: structural-equality probes for the matrix `Pat` and the `Bind` set (support
/// scaffolding faithful to the derived `PartialEq`, mirroring the FLAG-semcore-12 `ty_eq` posture).
fn driver_prelude() -> String {
    r#"
fn pat_eq_probe(a: Pat, b: Pat) => Bool =
  match a {
    MpWild => match b { MpWild => True, _ => False },
    MpLit(ka) => match b { MpLit(kb) => beq(ka, kb), _ => False },
    MpCtor(na, sa) => match b { MpCtor(nb, sb) => and_(beq(na, nb), pat_list_eq_probe(sa, sb)), _ => False }
  };

fn pat_list_eq_probe(a: Vec[Pat], b: Vec[Pat]) => Bool =
  match a {
    Nil => match b { Nil => True, Cons(_, _) => False },
    Cons(ha, ta) => match b { Nil => False, Cons(hb, tb) => and_(pat_eq_probe(ha, hb), pat_list_eq_probe(ta, tb)) }
  };

fn u32list_eq_probe(a: Vec[Binary{32}], b: Vec[Binary{32}]) => Bool =
  match a {
    Nil => match b { Nil => True, Cons(_, _) => False },
    Cons(ha, ta) => match b { Nil => False, Cons(hb, tb) => and_(eq_u(ha, hb), u32list_eq_probe(ta, tb)) }
  };

fn bind_eq_probe(a: Bind, b: Bind) => Bool =
  match a { Bnd(na, ta, oa) => match b { Bnd(nb, tb, ob) =>
    and_(beq(na, nb), and_(ty_eq(ta, tb), u32list_eq_probe(oa, ob))) } };

fn bind_list_eq_probe(a: Vec[Bind], b: Vec[Bind]) => Bool =
  match a {
    Nil => match b { Nil => True, Cons(_, _) => False },
    Cons(ha, ta) => match b { Nil => False, Cons(hb, tb) => and_(bind_eq_probe(ha, hb), bind_list_eq_probe(ta, tb)) }
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

fn encode_u64(n: u64) -> String {
    let mut s = String::from("0b");
    for (count, i) in (0..64).rev().enumerate() {
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

fn encode_literal(l: &Literal) -> String {
    match l {
        Literal::Bin(s) => format!("Bin({})", encode_bytes(s)),
        Literal::Trit(s) => format!("Trit({})", encode_bytes(s)),
        Literal::Int(i) => format!("Int({})", encode_u64(*i as u64)),
        Literal::Bytes(s) => format!("LBytes({})", encode_bytes(s)),
        Literal::Str(s) => format!("Str({})", encode_bytes(s)),
        Literal::Float(s) => format!("LFloat({})", encode_bytes(s)),
        Literal::List(elems) => {
            assert!(
                elems.is_empty(),
                "only the empty list literal is exercised (no Expr encoder here)"
            );
            "List(Nil)".to_owned()
        }
        Literal::AmbientInt(_, _) => {
            unreachable!(
                "AmbientInt is not exercised by the lit_ty_of/normalize_pattern differential"
            )
        }
    }
}

fn encode_pattern(p: &Pattern) -> String {
    match p {
        Pattern::Wildcard => "PWildcard".to_owned(),
        Pattern::Lit(l) => format!("PLit({})", encode_literal(l)),
        Pattern::Ctor(n, subs) => {
            format!("PCtor({}, {})", encode_bytes(n), encode_pattern_list(subs))
        }
        Pattern::Ident(n) => format!("PIdent({})", encode_bytes(n)),
        Pattern::Tuple(subs) => format!("PTuple({})", encode_pattern_list(subs)),
        Pattern::Or(subs) => format!("POr({})", encode_pattern_list(subs)),
    }
}

fn encode_pattern_list(ps: &[Pattern]) -> String {
    let mut s = String::from("Nil");
    for p in ps.iter().rev() {
        s = format!("Cons({}, {})", encode_pattern(p), s);
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

fn encode_pat(p: &Pat) -> String {
    match p {
        Pat::Wild => "MpWild".to_owned(),
        Pat::Ctor(n, subs) => {
            let mut s = String::from("Nil");
            for sub in subs.iter().rev() {
                s = format!("Cons({}, {})", encode_pat(sub), s);
            }
            format!("MpCtor({}, {s})", encode_bytes(n))
        }
        Pat::Lit(k) => format!("MpLit({})", encode_bytes(k)),
    }
}

fn encode_occ(occ: &[usize]) -> String {
    let mut s = String::from("Nil");
    for i in occ.iter().rev() {
        s = format!(
            "Cons({}, {})",
            encode_u32(u32::try_from(*i).expect("occ fits u32")),
            s
        );
    }
    s
}

fn encode_binds(binds: &[(String, Ty, Vec<usize>)]) -> String {
    let mut s = String::from("Nil");
    for (name, ty, occ) in binds.iter().rev() {
        s = format!(
            "Cons(Bnd({}, {}, {}), {})",
            encode_bytes(name),
            encode_ty(ty),
            encode_occ(occ),
            s
        );
    }
    s
}

// Small fixture constructors.
fn di(name: &str, params: &[&str], ctors: Vec<(&str, Vec<Ty>)>) -> DataInfo {
    DataInfo {
        name: name.to_owned(),
        params: params.iter().map(|p| (*p).to_owned()).collect(),
        ctors: ctors
            .into_iter()
            .map(|(n, fields)| CtorInfo {
                name: n.to_owned(),
                fields,
            })
            .collect(),
    }
}
fn registry(ds: &[DataInfo]) -> BTreeMap<String, DataInfo> {
    ds.iter().map(|d| (d.name.clone(), d.clone())).collect()
}
fn var(n: &str) -> Ty {
    Ty::Var(n.to_owned())
}
fn data(n: &str, args: Vec<Ty>) -> Ty {
    Ty::Data(n.to_owned(), args)
}
fn bin(n: u32) -> Ty {
    Ty::Binary(Width::Lit(n))
}
fn ctorp(n: &str, subs: Vec<Pattern>) -> Pattern {
    Pattern::Ctor(n.to_owned(), subs)
}
fn identp(n: &str) -> Pattern {
    Pattern::Ident(n.to_owned())
}

/// The standard registry: `Option<A>`, `Pair<A,B>`, `Bool`, and the synthetic `Tuple$2`.
fn std_registry() -> Vec<DataInfo> {
    vec![
        di(
            "Option",
            &["A"],
            vec![("None", vec![]), ("Some", vec![var("A")])],
        ),
        di(
            "Pair",
            &["A", "B"],
            vec![("MkPair", vec![var("A"), var("B")])],
        ),
        di("Bool", &[], vec![("True", vec![]), ("False", vec![])]),
        synthetic_tuple_data(2),
    ]
}

// ── per-leaf live-oracle assertions ───────────────────────────────────────────────────────────────

/// `lit_ty_of` (accept): the `.myc` Ty must equal the live `checkty::lit_ty_of`.
fn assert_lit_ty_ok(label: &str, l: &Literal) {
    let want =
        lit_ty_of("test", l).unwrap_or_else(|e| panic!("{label}: oracle lit_ty_of errored: {e}"));
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match lit_ty_of({}) {{ Ok(t) => match ty_eq(t, {}) {{ True => one32(), False => zero32() }}, Err(_) => zero32() }};\n",
        encode_literal(l),
        encode_ty(&want)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `lit_ty_of` (refuse): the `.myc` `lit_ty_of` must return `Err` exactly where the oracle does.
fn assert_lit_ty_err(label: &str, l: &Literal) {
    assert!(
        lit_ty_of("test", l).is_err(),
        "{label}: oracle lit_ty_of unexpectedly succeeded"
    );
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match lit_ty_of({}) {{ Err(_) => one32(), Ok(_) => zero32() }};\n",
        encode_literal(l)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `normalize_pattern` (accept): the `.myc` matrix `Pat` AND binder set must match the live oracle.
fn assert_normalize_ok(label: &str, ds: &[DataInfo], pat: &Pattern, expected: &Ty) {
    let map = registry(ds);
    let mut binds: Vec<(String, Ty, Vec<usize>)> = Vec::new();
    let want_pat = normalize_pattern(&map, "test", pat, expected, &[], &mut binds)
        .unwrap_or_else(|e| panic!("{label}: oracle normalize_pattern errored: {e}"));
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match normalize_pattern({}, {}, {}, Nil, Nil) {{\n    Ok(pr) => match pr {{ Pr(p, binds) => match and_(pat_eq_probe(p, {}), bind_list_eq_probe(binds, {})) {{ True => one32(), False => zero32() }} }},\n    Err(_) => zero32()\n  }};\n",
        encode_datainfo_list(ds),
        encode_pattern(pat),
        encode_ty(expected),
        encode_pat(&want_pat),
        encode_binds(&binds)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

/// `normalize_pattern` (refuse): the `.myc` `normalize_pattern` must `Err` exactly where the oracle does.
fn assert_normalize_err(label: &str, ds: &[DataInfo], pat: &Pattern, expected: &Ty) {
    let map = registry(ds);
    let mut binds: Vec<(String, Ty, Vec<usize>)> = Vec::new();
    assert!(
        normalize_pattern(&map, "test", pat, expected, &[], &mut binds).is_err(),
        "{label}: oracle normalize_pattern unexpectedly succeeded"
    );
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match normalize_pattern({}, {}, {}, Nil, Nil) {{ Err(_) => one32(), Ok(_) => zero32() }};\n",
        encode_datainfo_list(ds),
        encode_pattern(pat),
        encode_ty(expected)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Structural gate: `semcore.myc` (with the increment-6 additions) parses and type-checks green.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn semcore_normpat_parses_and_checks() {
    let nodule = parse(SEMCORE_SRC).unwrap_or_else(|e| panic!("semcore.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("semcore.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Non-vacuity probe: a wrong expected Pat yields 0 (the pat_eq/bind witness discriminates).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn normpat_witness_discriminates() {
    let ds = std_registry();
    // normalize `_` = Wild; comparing against the wrong `MpLit("x")` must yield 0.
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match normalize_pattern({}, {}, {}, Nil, Nil) {{\n    Ok(pr) => match pr {{ Pr(p, _) => match pat_eq_probe(p, MpLit({})) {{ True => one32(), False => zero32() }} }},\n    Err(_) => zero32()\n  }};\n",
        encode_datainfo_list(&ds),
        encode_pattern(&Pattern::Wildcard),
        encode_ty(&bin(8)),
        encode_bytes("x")
    );
    assert_l1_only_u32("wrong_pat_yields_zero", &program(&driver), 0);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// lit_ty_of (LIVE — checkty::lit_ty_of): every literal kind + the never-silent refusals.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn lit_ty_of_cases() {
    // Binary literal width = count of 0/1 chars (separators skipped).
    assert_lit_ty_ok("bin_1010", &Literal::Bin("1010".to_owned()));
    assert_lit_ty_ok("bin_sep", &Literal::Bin("10_10".to_owned()));
    // Ternary literal width = byte length.
    assert_lit_ty_ok("trit", &Literal::Trit("012".to_owned()));
    // Bytes/Str type as Bytes; Float as Float.
    assert_lit_ty_ok("bytes", &Literal::Bytes("ff".to_owned()));
    assert_lit_ty_ok("str", &Literal::Str("hi".to_owned()));
    assert_lit_ty_ok("float", &Literal::Float("1.5".to_owned()));
    // Refusals: bare int, empty binary/ternary, list-in-pattern.
    assert_lit_ty_err("int", &Literal::Int(42));
    assert_lit_ty_err("empty_bin", &Literal::Bin(String::new()));
    assert_lit_ty_err("empty_trit", &Literal::Trit(String::new()));
    assert_lit_ty_err("list", &Literal::List(vec![]));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// normalize_pattern accept (LIVE): wildcard/ident-bind/nullary-ctor/ctor-recurse/tuple/lit — the
// matrix Pat AND the binder occurrences.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn normalize_pattern_accept_cases() {
    let ds = std_registry();
    // Wildcard → Wild, no binds.
    assert_normalize_ok("wildcard", &ds, &Pattern::Wildcard, &bin(8));
    // Ident that is NOT a ctor → Wild + a binder (name, expected type, occ []).
    assert_normalize_ok("ident_bind", &ds, &identp("x"), &data("Bool", vec![]));
    assert_normalize_ok("ident_bind_binary", &ds, &identp("n"), &bin(16));
    // Ident that IS a nullary ctor → Ctor(name, []), no binds.
    assert_normalize_ok(
        "nullary_ctor_ident",
        &ds,
        &identp("True"),
        &data("Bool", vec![]),
    );
    // Ctor with a field: param_subst instantiates the field type; the sub-binder gets occ [0].
    assert_normalize_ok(
        "ctor_some_binary",
        &ds,
        &ctorp("Some", vec![identp("v")]),
        &data("Option", vec![bin(8)]),
    );
    // Two-field ctor: occ [0] and [1], each field type instantiated.
    assert_normalize_ok(
        "ctor_pair",
        &ds,
        &ctorp("MkPair", vec![identp("a"), identp("b")]),
        &data("Pair", vec![Ty::Bytes, Ty::Float]),
    );
    // Nested ctor: occ [0], then [0,0] and [0,1].
    assert_normalize_ok(
        "ctor_nested",
        &ds,
        &ctorp(
            "Some",
            vec![ctorp("MkPair", vec![identp("a"), identp("b")])],
        ),
        &data("Option", vec![data("Pair", vec![Ty::Bytes, Ty::Float])]),
    );
    // A wildcard sub-pattern inside a ctor binds nothing.
    assert_normalize_ok(
        "ctor_wild_sub",
        &ds,
        &ctorp("Some", vec![Pattern::Wildcard]),
        &data("Option", vec![bin(8)]),
    );
    // Tuple pattern → the synthetic MkTuple$2 ctor over Tuple$2.
    assert_normalize_ok(
        "tuple_pattern",
        &ds,
        &Pattern::Tuple(vec![identp("a"), identp("b")]),
        &data("Tuple$2", vec![Ty::Bytes, Ty::Float]),
    );
    // Literal patterns: the matrix Lit key, type-matched to the scrutinee.
    assert_normalize_ok(
        "lit_binary",
        &ds,
        &Pattern::Lit(Literal::Bin("1010".to_owned())),
        &bin(4),
    );
    assert_normalize_ok(
        "lit_binary_sep",
        &ds,
        &Pattern::Lit(Literal::Bin("10_10".to_owned())),
        &bin(4),
    );
    assert_normalize_ok(
        "lit_bytes",
        &ds,
        &Pattern::Lit(Literal::Bytes("ab".to_owned())),
        &Ty::Bytes,
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// normalize_pattern refuse (LIVE): the never-silent refusals (G2/VR-5).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn normalize_pattern_refuse_cases() {
    let ds = std_registry();
    // A ctor pattern on a non-data scrutinee.
    assert_normalize_err(
        "ctor_on_binary",
        &ds,
        &ctorp("Some", vec![identp("v")]),
        &bin(8),
    );
    // Not a constructor of the scrutinee's type.
    assert_normalize_err(
        "not_a_ctor",
        &ds,
        &ctorp("Nope", vec![identp("v")]),
        &data("Option", vec![bin(8)]),
    );
    // Arity mismatch: Some takes 1 field, given 0.
    assert_normalize_err(
        "ctor_arity",
        &ds,
        &ctorp("Some", vec![]),
        &data("Option", vec![bin(8)]),
    );
    // A nullary-looking Ident that is actually a field-carrying ctor must bind its fields.
    assert_normalize_err(
        "ident_ctor_needs_fields",
        &ds,
        &identp("Some"),
        &data("Option", vec![bin(8)]),
    );
    // Literal pattern type mismatch (Binary4 literal vs a Bytes scrutinee).
    assert_normalize_err(
        "lit_type_mismatch",
        &ds,
        &Pattern::Lit(Literal::Bin("1010".to_owned())),
        &Ty::Bytes,
    );
    // A float literal is not a legal match pattern (ADR-040 FLAG-4).
    assert_normalize_err(
        "float_pattern",
        &ds,
        &Pattern::Lit(Literal::Float("1.5".to_owned())),
        &Ty::Float,
    );
    // An or-pattern must be desugared before normalize_pattern.
    assert_normalize_err(
        "or_pattern",
        &ds,
        &Pattern::Or(vec![identp("a"), identp("b")]),
        &data("Bool", vec![]),
    );
}
