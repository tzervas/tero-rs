//! M-740 Stage 4 (DN-26 §7.3 row 4) — the self-hosted `compiler.ambient` port.
//!
//! `lib/compiler/ambient.myc`'s `resolve`/`resolve_report`/`expand_to_source` vs the live Rust
//! oracle (`mycelium_l1::ambient::{resolve, expand_to_source}`, `crates/mycelium-l1/src/ambient.rs`)
//! over a small, hand-transcribed battery of SYNTHETIC nodules (FLAG-ambient-6 in `ambient.myc`
//! itself): `compiler.ambient` operates on an ALREADY-PARSED `Nodule` value (it depends only on
//! `crate::ast`, never `crate::parse`), so — unlike `compiler.parse` — it cannot differential
//! directly against the `docs/spec/grammar/conformance/*` corpus source files without either a
//! self-hosted parser in the same nodule (unavailable, cross-nodule execution is staged) or a
//! nontrivial Rust-side AST-to-L1-Value serializer (out of this leaf's scope). Each `test_input_N`
//! below is hand-transcribed to be STRUCTURALLY IDENTICAL to its `.myc` counterpart
//! (`lib/compiler/ambient.myc`'s own `test_input_N`).
//!
//! Three legs per fixture, mirroring the established Stage-3 one-eval-per-file economy:
//! (a) **classification parity** — self-hosted `resolve` Ok/Err must agree with the oracle;
//! (b) **AST structural fingerprint** on every file both sides accept (the `fp` module below,
//!     copied verbatim from `compiler_stage3.rs`'s own `mod fp` — same AST shape, same tags);
//! (c) **`expand_to_source` byte-for-byte parity** — the oracle's rendered text is passed in as a
//!     `Bytes` argument (the same "pass a `Bytes` value in" idiom `bytes_value` already
//!     establishes for corpus source text) and compared via `bytes_eq` on the self-hosted side —
//!     no hashing needed, this is an exact string comparison.
//! Plus a fourth leg for the four never-silent refusals (MultipleDefaults/UnresolvedAmbient/
//! ParadigmShapeMismatch/BareDecimalNoEncoding): **error-kind classification parity** (a 5-way
//! code; message TEXT fidelity is out of scope, mirroring FLAG-parse-8/FLAG-ambient-3).
//!
//! M-981 applies as in every prior stage: only the L1-eval leg is exercised (the L0 substitution
//! interpreter is impractical at this scale). M-980's split-match idiom is used throughout
//! `ambient.myc`.

use mycelium_l1::ast::{
    AmbientParams, Ctor, Expr, FnDecl, FnSig, Item, Literal, Nodule, ObjectDecl, Paradigm, Param,
    Path, Pattern, Scalar, Sparsity, TypeRef,
};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{ambient, check_nodule, monomorphize, AmbientError, Evaluator, Vis};

const AMBIENT_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/ambient.myc"
));

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The structural gate: `ambient.myc` parses and type-checks green (no driver needed).
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn ambient_myc_parses_and_checks() {
    let nodule = mycelium_l1::parse(AMBIENT_SRC)
        .unwrap_or_else(|e| panic!("ambient.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("ambient.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The oracle-side AST fingerprint mirror (`fp` module) — copied VERBATIM from
// `compiler_stage3.rs`'s own `mod fp`: same 109-entry tag table, same rotl(7)-xor mixing, same
// per-node field-visitation order. `ambient.myc`'s resolver produces the SAME `Nodule` shape
// `compiler.parse` does, so the identical fingerprint walker applies unchanged.
// ─────────────────────────────────────────────────────────────────────────────────────────────
mod fp {
    use super::*;
    use mycelium_l1::ast::{
        Arm, BaseType, DeriveDecl, ExecutionMode, Hypha, ImplDecl, InherentImplDecl, LowerDecl,
        LowerRhs, Strength, TraitDecl, TraitRef, TypeDecl, TypeParam, UsePath, ViaDecl, WidthRef,
    };

    #[derive(Clone, Copy)]
    pub struct Fp {
        pub hash: u32,
        pub count: u32,
    }

    fn tag(fp: Fp, t: u32) -> Fp {
        Fp {
            hash: fp.hash.rotate_left(7) ^ t,
            count: fp.count + 1,
        }
    }

    fn bytes(fp: Fp, s: &str) -> Fp {
        tag(fp, s.len() as u32)
    }

    fn u32v(fp: Fp, n: u32) -> Fp {
        tag(fp, n)
    }

    fn bool_(fp: Fp, b: bool) -> Fp {
        tag(fp, if b { 25 } else { 26 })
    }

    fn vis(fp: Fp, v: Vis) -> Fp {
        tag(fp, if matches!(v, Vis::Private) { 1 } else { 2 })
    }

    fn paradigm(fp: Fp, p: &Paradigm) -> Fp {
        tag(
            fp,
            match p {
                Paradigm::Binary => 3,
                Paradigm::Ternary => 4,
                Paradigm::Dense => 5,
                Paradigm::Vsa => 6,
            },
        )
    }

    fn scalar(fp: Fp, s: &Scalar) -> Fp {
        tag(
            fp,
            match s {
                Scalar::F16 => 7,
                Scalar::Bf16 => 8,
                Scalar::F32 => 9,
                Scalar::F64 => 10,
            },
        )
    }

    fn strength(fp: Fp, s: &Strength) -> Fp {
        tag(
            fp,
            match s {
                Strength::Exact => 11,
                Strength::Proven => 12,
                Strength::Empirical => 13,
                Strength::Declared => 14,
            },
        )
    }

    fn sparsity(fp: Fp, s: &Sparsity) -> Fp {
        match s {
            Sparsity::Dense => tag(fp, 15),
            Sparsity::Sparse(k) => u32v(tag(fp, 16), *k),
        }
    }

    fn paramkind(fp: Fp, k: &mycelium_l1::ast::ParamKind) -> Fp {
        tag(
            fp,
            if matches!(k, mycelium_l1::ast::ParamKind::Type) {
                17
            } else {
                18
            },
        )
    }

    fn execmode(fp: Fp, e: &ExecutionMode) -> Fp {
        tag(
            fp,
            if matches!(e, ExecutionMode::Interpreted) {
                19
            } else {
                20
            },
        )
    }

    fn widthref(fp: Fp, w: &WidthRef) -> Fp {
        match w {
            WidthRef::Lit(n) => u32v(tag(fp, 21), *n),
            WidthRef::Name(s) => bytes(tag(fp, 22), s),
        }
    }

    fn path(fp: Fp, p: &Path) -> Fp {
        let mut fp = tag(fp, 23);
        for seg in &p.0 {
            fp = bytes(fp, seg);
        }
        fp
    }

    fn usepath(fp: Fp, u: &UsePath) -> Fp {
        let fp = path(tag(fp, 24), &u.path);
        bool_(fp, u.glob)
    }

    fn ambientparams(fp: Fp, a: &AmbientParams) -> Fp {
        match a {
            AmbientParams::Size(n) => u32v(tag(fp, 27), *n),
            AmbientParams::Dense(n, sc) => scalar(u32v(tag(fp, 28), *n), sc),
            AmbientParams::Vsa {
                model,
                dim,
                sparsity: sp,
            } => sparsity(u32v(bytes(tag(fp, 29), model), *dim), sp),
        }
    }

    fn typeref(fp: Fp, t: &TypeRef) -> Fp {
        let fp = basetype(tag(fp, 30), &t.base);
        guarantee_opt(fp, &t.guarantee)
    }

    fn guarantee_opt(fp: Fp, g: &Option<Strength>) -> Fp {
        match g {
            None => tag(fp, 31),
            Some(s) => strength(fp, s),
        }
    }

    fn typeref_list(fp: Fp, xs: &[TypeRef]) -> Fp {
        xs.iter().fold(fp, typeref)
    }

    fn basetype(fp: Fp, b: &BaseType) -> Fp {
        match b {
            BaseType::Binary(w) => widthref(tag(fp, 32), w),
            BaseType::Ternary(w) => widthref(tag(fp, 33), w),
            BaseType::Dense(n, sc) => scalar(u32v(tag(fp, 34), *n), sc),
            BaseType::Vsa {
                model,
                dim,
                sparsity: sp,
            } => sparsity(u32v(bytes(tag(fp, 35), model), *dim), sp),
            BaseType::Substrate(name) => bytes(tag(fp, 36), name),
            BaseType::Seq { elem, len } => u32v(typeref(tag(fp, 37), elem), *len),
            BaseType::Bytes => tag(fp, 38),
            BaseType::Float => tag(fp, 39),
            BaseType::Named(name, args) => typeref_list(bytes(tag(fp, 40), name), args),
            BaseType::Ambient(ap) => ambientparams(tag(fp, 41), ap),
            BaseType::Fn(a, r) => typeref(typeref(tag(fp, 42), a), r),
            BaseType::Tuple(elems) => typeref_list(tag(fp, 43), elems),
        }
    }

    fn traitref(fp: Fp, t: &TraitRef) -> Fp {
        typeref_list(bytes(tag(fp, 44), &t.name), &t.args)
    }

    fn traitref_list(fp: Fp, xs: &[TraitRef]) -> Fp {
        xs.iter().fold(fp, traitref)
    }

    fn typeparam(fp: Fp, t: &TypeParam) -> Fp {
        let fp = paramkind(bytes(tag(fp, 45), &t.name), &t.kind);
        traitref_list(fp, &t.bounds)
    }

    fn typeparam_list(fp: Fp, xs: &[TypeParam]) -> Fp {
        xs.iter().fold(fp, typeparam)
    }

    fn param(fp: Fp, p: &Param) -> Fp {
        typeref(bytes(tag(fp, 46), &p.name), &p.ty)
    }

    fn param_list(fp: Fp, xs: &[Param]) -> Fp {
        xs.iter().fold(fp, param)
    }

    fn bytes_list(fp: Fp, xs: &[String]) -> Fp {
        xs.iter().fold(fp, |fp, s| bytes(fp, s))
    }

    // FLAG-parse-10 mirror: effect budgets are skipped except for the effect NAME.

    fn fnsig(fp: Fp, s: &FnSig) -> Fp {
        let fp = bytes(tag(fp, 47), &s.name);
        let fp = typeparam_list(fp, &s.params);
        let fp = param_list(fp, &s.value_params);
        let fp = typeref(fp, &s.ret);
        bytes_list(fp, &s.effects)
    }

    fn fnsig_list(fp: Fp, xs: &[FnSig]) -> Fp {
        xs.iter().fold(fp, fnsig)
    }

    fn ctor(fp: Fp, c: &Ctor) -> Fp {
        typeref_list(bytes(tag(fp, 48), &c.name), &c.fields)
    }

    fn ctor_list(fp: Fp, xs: &[Ctor]) -> Fp {
        xs.iter().fold(fp, ctor)
    }

    fn typedecl(fp: Fp, t: &TypeDecl) -> Fp {
        let fp = bytes(tag(fp, 49), &t.name);
        let fp = vis(fp, t.vis);
        let fp = bytes_list(fp, &t.params);
        ctor_list(fp, &t.ctors)
    }

    fn traitdecl(fp: Fp, t: &TraitDecl) -> Fp {
        let fp = bytes(tag(fp, 50), &t.name);
        let fp = vis(fp, t.vis);
        let fp = bytes_list(fp, &t.params);
        fnsig_list(fp, &t.sigs)
    }

    fn fndecl(fp: Fp, f: &FnDecl) -> Fp {
        let fp = vis(tag(fp, 51), f.vis);
        let fp = bool_(fp, f.thaw);
        let fp = tier_opt(fp, &f.tier);
        let fp = fnsig(fp, &f.sig);
        expr(fp, &f.body)
    }

    fn tier_opt(fp: Fp, t: &Option<ExecutionMode>) -> Fp {
        match t {
            None => tag(fp, 52),
            Some(m) => execmode(fp, m),
        }
    }

    fn fndecl_list(fp: Fp, xs: &[FnDecl]) -> Fp {
        xs.iter().fold(fp, fndecl)
    }

    fn impldecl(fp: Fp, i: &ImplDecl) -> Fp {
        let fp = bytes(tag(fp, 53), &i.trait_name);
        let fp = typeref_list(fp, &i.trait_args);
        let fp = typeref(fp, &i.for_ty);
        fndecl_list(fp, &i.methods)
    }

    fn viadecl(fp: Fp, v: &ViaDecl) -> Fp {
        let fp = u32v(tag(fp, 54), v.field_idx);
        let fp = bytes(fp, &v.trait_name);
        typeref_list(fp, &v.trait_args)
    }

    fn viadecl_list(fp: Fp, xs: &[ViaDecl]) -> Fp {
        xs.iter().fold(fp, viadecl)
    }

    fn impldecl_list(fp: Fp, xs: &[ImplDecl]) -> Fp {
        xs.iter().fold(fp, impldecl)
    }

    fn objectdecl(fp: Fp, o: &ObjectDecl) -> Fp {
        let fp = bytes(tag(fp, 55), &o.name);
        let fp = vis(fp, o.vis);
        let fp = bytes_list(fp, &o.params);
        let fp = ctor(fp, &o.ctor);
        let fp = viadecl_list(fp, &o.via_decls);
        let fp = impldecl_list(fp, &o.impls);
        fndecl_list(fp, &o.fns)
    }

    fn inherentimpldecl(fp: Fp, i: &InherentImplDecl) -> Fp {
        let fp = typeref(tag(fp, 56), &i.for_ty);
        fndecl_list(fp, &i.methods)
    }

    fn lowerrhs(fp: Fp, r: &LowerRhs) -> Fp {
        match r {
            LowerRhs::Expr(e) => expr(tag(fp, 57), e),
            LowerRhs::Impl(i) => impldecl(tag(fp, 58), i),
        }
    }

    fn lowerdecl(fp: Fp, l: &LowerDecl) -> Fp {
        let fp = bytes(tag(fp, 59), &l.name);
        let fp = bytes_list(fp, &l.params);
        lowerrhs(fp, &l.rhs)
    }

    fn derivedecl(fp: Fp, d: &DeriveDecl) -> Fp {
        let fp = bytes(tag(fp, 60), &d.name);
        typeref(fp, &d.for_ty)
    }

    fn literal(fp: Fp, l: &Literal) -> Fp {
        match l {
            Literal::Bin(s) => bytes(tag(fp, 61), s),
            Literal::Trit(s) => bytes(tag(fp, 62), s),
            Literal::Int(n) => u32v(tag(fp, 63), *n as u32),
            Literal::AmbientInt(par, n) => u32v(paradigm(tag(fp, 64), par), *n as u32),
            Literal::List(elems) => expr_list(tag(fp, 65), elems),
            Literal::Bytes(s) => bytes(tag(fp, 66), s),
            Literal::Str(s) => bytes(tag(fp, 67), s),
            Literal::Float(s) => bytes(tag(fp, 68), s),
            _ => panic!("fp::literal: unhandled non_exhaustive Literal variant"),
        }
    }

    fn pattern(fp: Fp, p: &Pattern) -> Fp {
        match p {
            Pattern::Wildcard => tag(fp, 69),
            Pattern::Lit(l) => literal(tag(fp, 70), l),
            Pattern::Ctor(name, subs) => pattern_list(bytes(tag(fp, 71), name), subs),
            Pattern::Ident(name) => bytes(tag(fp, 72), name),
            Pattern::Tuple(subs) => pattern_list(tag(fp, 73), subs),
            Pattern::Or(alts) => pattern_list(tag(fp, 74), alts),
        }
    }

    fn pattern_list(fp: Fp, xs: &[Pattern]) -> Fp {
        xs.iter().fold(fp, pattern)
    }

    fn arm(fp: Fp, a: &Arm) -> Fp {
        let fp = pattern(tag(fp, 75), &a.pattern);
        expr(fp, &a.body)
    }

    fn arm_list(fp: Fp, xs: &[Arm]) -> Fp {
        xs.iter().fold(fp, arm)
    }

    fn expr_opt(fp: Fp, e: &Option<Box<Expr>>) -> Fp {
        match e {
            None => tag(fp, 76),
            Some(x) => expr(fp, x),
        }
    }

    fn hypha(fp: Fp, h: &Hypha) -> Fp {
        let fp = expr_opt(tag(fp, 77), &h.forage);
        expr(fp, &h.body)
    }

    fn hypha_list(fp: Fp, xs: &[Hypha]) -> Fp {
        xs.iter().fold(fp, hypha)
    }

    fn typeref_opt(fp: Fp, t: &Option<TypeRef>) -> Fp {
        match t {
            None => tag(fp, 78),
            Some(x) => typeref(fp, x),
        }
    }

    fn expr(fp: Fp, e: &Expr) -> Fp {
        match e {
            Expr::Let {
                name,
                ty,
                bound,
                body,
            } => {
                let fp = bytes(tag(fp, 79), name);
                let fp = typeref_opt(fp, ty);
                let fp = expr(fp, bound);
                expr(fp, body)
            }
            Expr::If { cond, conseq, alt } => {
                let fp = expr(tag(fp, 80), cond);
                let fp = expr(fp, conseq);
                expr(fp, alt)
            }
            Expr::Match { scrutinee, arms } => {
                let fp = expr(tag(fp, 81), scrutinee);
                arm_list(fp, arms)
            }
            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => {
                let fp = bytes(tag(fp, 82), x);
                let fp = bytes(fp, acc);
                let fp = expr(fp, xs);
                let fp = expr(fp, init);
                expr(fp, body)
            }
            Expr::Swap {
                value,
                target,
                policy,
            } => {
                let fp = expr(tag(fp, 83), value);
                let fp = typeref(fp, target);
                path(fp, policy)
            }
            Expr::WithParadigm {
                paradigm: par,
                body,
            } => {
                let fp = paradigm(tag(fp, 84), par);
                expr(fp, body)
            }
            Expr::Wild(body) => expr(tag(fp, 85), body),
            Expr::Spore(value) => expr(tag(fp, 86), value),
            Expr::Consume(value) => expr(tag(fp, 87), value),
            Expr::Colony(hyphae) => hypha_list(tag(fp, 88), hyphae),
            Expr::Lambda { params, body } => {
                let fp = param_list(tag(fp, 89), params);
                expr(fp, body)
            }
            Expr::App { head, args } => {
                let fp = expr(tag(fp, 90), head);
                expr_list(fp, args)
            }
            Expr::Fuse { left, right } => {
                let fp = expr(tag(fp, 91), left);
                expr(fp, right)
            }
            Expr::Reclaim { policy, body } => {
                let fp = expr(tag(fp, 92), policy);
                expr(fp, body)
            }
            Expr::Path(p) => path(tag(fp, 93), p),
            Expr::Lit(l) => literal(tag(fp, 94), l),
            Expr::Ascribe(inner, ty) => {
                let fp = expr(tag(fp, 95), inner);
                typeref(fp, ty)
            }
            Expr::TupleLit(elems) => expr_list(tag(fp, 96), elems),
        }
    }

    fn expr_list(fp: Fp, xs: &[Expr]) -> Fp {
        xs.iter().fold(fp, expr)
    }

    fn item(fp: Fp, i: &Item) -> Fp {
        match i {
            Item::Use(u) => usepath(tag(fp, 97), u),
            Item::Default(par) => paradigm(tag(fp, 98), par),
            Item::Type(t) => typedecl(tag(fp, 99), t),
            Item::Trait(t) => traitdecl(tag(fp, 100), t),
            Item::Impl(i2) => impldecl(tag(fp, 101), i2),
            Item::Fn(f) => fndecl(tag(fp, 102), f),
            Item::Object(o) => objectdecl(tag(fp, 103), o),
            Item::Lower(l) => lowerdecl(tag(fp, 104), l),
            Item::Derive(d) => derivedecl(tag(fp, 105), d),
            Item::InherentImpl(i3) => inherentimpldecl(tag(fp, 106), i3),
        }
    }

    fn item_list(fp: Fp, xs: &[Item]) -> Fp {
        xs.iter().fold(fp, item)
    }

    pub fn fingerprint_nodule(n: &Nodule) -> Fp {
        let fp = Fp { hash: 0, count: 0 };
        nodule(fp, n)
    }

    fn nodule(fp: Fp, n: &Nodule) -> Fp {
        let fp = path(tag(fp, 107), &n.path);
        let fp = bool_(fp, n.std_sys);
        item_list(fp, &n.items)
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Value-construction helpers (the `compiler_stage3.rs` `bytes_value`/`b32_value` convention).
// ─────────────────────────────────────────────────────────────────────────────────────────────
fn bytes_value(s: &[u8]) -> mycelium_core::Value {
    mycelium_core::Value::new(
        mycelium_core::Repr::Bytes,
        mycelium_core::Payload::Bytes(s.to_vec()),
        mycelium_core::Meta::exact(mycelium_core::Provenance::Root),
    )
    .expect("a Bytes value is well-formed")
}

fn b32_value(n: u32) -> mycelium_core::Value {
    let bits: Vec<bool> = (0..32).rev().map(|i| (n >> i) & 1 == 1).collect();
    mycelium_core::Value::new(
        mycelium_core::Repr::Binary { width: 32 },
        mycelium_core::Payload::Bits(bits),
        mycelium_core::Meta::exact(mycelium_core::Provenance::Root),
    )
    .expect("a Binary{32} value is well-formed")
}

/// Extract a `Binary{32}` verdict from an `Evaluator::call` result (the established convention).
fn verdict_u32(
    v: mycelium_l1::L1Value,
    mono: &mycelium_l1::Env,
    registry: &mycelium_core::DataRegistry,
) -> u32 {
    let core = v
        .to_core(mono, registry)
        .unwrap_or_else(|| panic!("L1 result is outside the r3 data fragment"));
    let repr_val = core
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr CoreValue, got {core:?}"));
    match repr_val.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("expected a Bits payload, got {other:?}"),
    }
}

/// Run a 0-arg `Binary{32}`-returning self-hosted entry and return its value (mirrors
/// `run_verdict` in `compiler_stage3.rs`, specialized to `ambient.myc`'s nullary drivers).
fn run_u32_entry(env: &mycelium_l1::Env, entry: &str) -> u32 {
    let mono =
        monomorphize(env, entry).unwrap_or_else(|e| panic!("{entry}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{entry}: build_registry failed: {e}"));
    let val = Evaluator::new(&mono)
        .call(entry, vec![])
        .unwrap_or_else(|e| panic!("{entry}: L1-eval failed: {e}"));
    verdict_u32(val, &mono, &registry)
}

/// Run a `Binary{32}`-returning verdict entry taking `(want_ok, want_hash, want_count)`.
fn run_verdict3(
    env: &mycelium_l1::Env,
    entry: &str,
    want_ok: u32,
    want_hash: u32,
    want_count: u32,
) -> u32 {
    let mono =
        monomorphize(env, entry).unwrap_or_else(|e| panic!("{entry}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{entry}: build_registry failed: {e}"));
    let args = vec![
        mycelium_l1::L1Value::Repr(b32_value(want_ok)),
        mycelium_l1::L1Value::Repr(b32_value(want_hash)),
        mycelium_l1::L1Value::Repr(b32_value(want_count)),
    ];
    let val = Evaluator::new(&mono)
        .with_fuel(200_000_000)
        .call(entry, args)
        .unwrap_or_else(|e| panic!("{entry}: L1-eval failed: {e}"));
    verdict_u32(val, &mono, &registry)
}

/// Run a `Binary{32}`-returning verdict entry taking a single `Bytes` argument.
fn run_verdict_bytes(env: &mycelium_l1::Env, entry: &str, want: &[u8]) -> u32 {
    let mono =
        monomorphize(env, entry).unwrap_or_else(|e| panic!("{entry}: monomorphize failed: {e}"));
    let registry =
        build_registry(&mono).unwrap_or_else(|e| panic!("{entry}: build_registry failed: {e}"));
    let args = vec![mycelium_l1::L1Value::Repr(bytes_value(want))];
    let val = Evaluator::new(&mono)
        .with_fuel(200_000_000)
        .call(entry, args)
        .unwrap_or_else(|e| panic!("{entry}: L1-eval failed: {e}"));
    verdict_u32(val, &mono, &registry)
}

fn checked_env() -> mycelium_l1::Env {
    check_nodule(
        &mycelium_l1::parse(AMBIENT_SRC)
            .unwrap_or_else(|e| panic!("ambient.myc: parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("ambient.myc: check failed: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Synthetic fixtures (mirrors `ambient.myc`'s own `test_input_N` literals, FLAG-ambient-6).
// ─────────────────────────────────────────────────────────────────────────────────────────────
fn tr_unguaranteed(b: mycelium_l1::ast::BaseType) -> TypeRef {
    TypeRef::unguaranteed(b)
}

fn named(name: &str) -> mycelium_l1::ast::BaseType {
    mycelium_l1::ast::BaseType::Named(name.to_owned(), vec![])
}

fn simple_fn(name: &str, value_params: Vec<Param>, ret: TypeRef, body: Expr) -> FnDecl {
    FnDecl {
        vis: Vis::Private,
        thaw: false,
        tier: None,
        sig: FnSig {
            name: name.to_owned(),
            params: vec![],
            value_params,
            ret,
            effects: vec![],
            effect_budgets: Default::default(),
        },
        body,
    }
}

// TC1: pure passthrough — no ambient anywhere; `resolve` must be the identity.
fn test_input_1() -> Nodule {
    Nodule {
        path: Path(vec!["t1".into()]),
        std_sys: false,
        items: vec![Item::Fn(simple_fn(
            "f",
            vec![Param {
                name: "x".into(),
                ty: tr_unguaranteed(named("Widget")),
            }],
            tr_unguaranteed(named("Widget")),
            Expr::Path(Path(vec!["x".into()])),
        ))],
    }
}

// TC2: `default paradigm Binary` + a paradigm-less `{8}` return type resolves to `Binary{8}`.
fn test_input_2() -> Nodule {
    Nodule {
        path: Path(vec!["t2".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Binary),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(AmbientParams::Size(8))),
                Expr::Lit(Literal::Int(42)),
            )),
        ],
    }
}

// TC3: `default paradigm Ternary`, same shape as TC2 but Ternary.
fn test_input_3() -> Nodule {
    Nodule {
        path: Path(vec!["t3".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Ternary),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(AmbientParams::Size(5))),
                Expr::Lit(Literal::Int(0)),
            )),
        ],
    }
}

// TC4: `default paradigm Dense` + `{4, F32}` resolves to `Dense{4, F32}`.
fn test_input_4() -> Nodule {
    Nodule {
        path: Path(vec!["t4".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Dense),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(AmbientParams::Dense(
                    4,
                    Scalar::F32,
                ))),
                Expr::Lit(Literal::Float("0.0".into())),
            )),
        ],
    }
}

// TC5: `default paradigm VSA` + `{"hrr", 128, Dense}` resolves to `VSA{"hrr", 128, Dense}`.
fn test_input_5() -> Nodule {
    Nodule {
        path: Path(vec!["t5".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Vsa),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(AmbientParams::Vsa {
                    model: "hrr".into(),
                    dim: 128,
                    sparsity: Sparsity::Dense,
                })),
                Expr::Lit(Literal::Float("0.0".into())),
            )),
        ],
    }
}

// TC6: nested `with paradigm Ternary { … }` locally overrides a `Binary` nodule default.
fn test_input_6() -> Nodule {
    Nodule {
        path: Path(vec!["t6".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Binary),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(named("Unit")),
                Expr::WithParadigm {
                    paradigm: Paradigm::Ternary,
                    body: Box::new(Expr::Lit(Literal::Int(1))),
                },
            )),
        ],
    }
}

// TC7: an `object` declaration with a paradigm-less ctor field, resolved under a Binary default.
fn test_input_7() -> Nodule {
    Nodule {
        path: Path(vec!["t7".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Binary),
            Item::Object(ObjectDecl {
                vis: Vis::Private,
                name: "Cell".into(),
                params: vec![],
                ctor: Ctor {
                    name: "Cell".into(),
                    fields: vec![tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(
                        AmbientParams::Size(16),
                    ))],
                },
                via_decls: vec![],
                impls: vec![],
                fns: vec![],
            }),
        ],
    }
}

// TC8: a mixed-expr body (let/if/match/app/tuple) under a Binary ambient — several bare decimals
// each resolve to `AmbientInt(Binary, _)`.
fn test_input_8() -> Nodule {
    let body = Expr::Let {
        name: "x".into(),
        ty: None,
        bound: Box::new(Expr::Lit(Literal::Int(1))),
        body: Box::new(Expr::If {
            cond: Box::new(Expr::Path(Path(vec!["x".into()]))),
            conseq: Box::new(Expr::Match {
                scrutinee: Box::new(Expr::Path(Path(vec!["x".into()]))),
                arms: vec![mycelium_l1::ast::Arm {
                    pattern: Pattern::Wildcard,
                    body: Expr::TupleLit(vec![
                        Expr::Lit(Literal::Int(0)),
                        Expr::Lit(Literal::Int(1)),
                    ]),
                }],
            }),
            alt: Box::new(Expr::App {
                head: Box::new(Expr::Path(Path(vec!["id".into()]))),
                args: vec![Expr::Lit(Literal::Int(0))],
            }),
        }),
    };
    Nodule {
        path: Path(vec!["t8".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Binary),
            Item::Fn(simple_fn("f", vec![], tr_unguaranteed(named("Unit")), body)),
        ],
    }
}

// ERR1: two `default paradigm` declarations -> MultipleDefaults.
fn test_input_err1() -> Nodule {
    Nodule {
        path: Path(vec!["e1".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Binary),
            Item::Default(Paradigm::Ternary),
        ],
    }
}

// ERR2: a paradigm-less `{8}` with no enclosing ambient anywhere -> UnresolvedAmbient.
fn test_input_err2() -> Nodule {
    Nodule {
        path: Path(vec!["e2".into()]),
        std_sys: false,
        items: vec![Item::Fn(simple_fn(
            "f",
            vec![],
            tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(AmbientParams::Size(8))),
            Expr::Lit(Literal::Float("0.0".into())),
        ))],
    }
}

// ERR3: a Binary ambient but a Dense-shaped `{4, F32}` param -> ParadigmShapeMismatch.
fn test_input_err3() -> Nodule {
    Nodule {
        path: Path(vec!["e3".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Binary),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(mycelium_l1::ast::BaseType::Ambient(AmbientParams::Dense(
                    4,
                    Scalar::F32,
                ))),
                Expr::Lit(Literal::Float("0.0".into())),
            )),
        ],
    }
}

// ERR4: a bare decimal under a Dense ambient -> BareDecimalNoEncoding.
fn test_input_err4() -> Nodule {
    Nodule {
        path: Path(vec!["e4".into()]),
        std_sys: false,
        items: vec![
            Item::Default(Paradigm::Dense),
            Item::Fn(simple_fn(
                "f",
                vec![],
                tr_unguaranteed(named("Unit")),
                Expr::Lit(Literal::Int(1)),
            )),
        ],
    }
}

/// A 5-way classification code mirroring `ambient.myc`'s `ambient_error_kind` (FLAG-ambient-3:
/// message TEXT is not compared, only which refusal fired).
fn ambient_error_kind(e: &AmbientError) -> u32 {
    match e {
        AmbientError::MultipleDefaults { .. } => 1,
        AmbientError::UnresolvedAmbient { .. } => 2,
        AmbientError::ParadigmShapeMismatch { .. } => 3,
        AmbientError::BareDecimalNoEncoding { .. } => 4,
        AmbientError::DepthExceeded { .. } => 5,
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Leg (a) + (b): classification parity + AST fingerprint parity, one eval per fixture.
// ─────────────────────────────────────────────────────────────────────────────────────────────
fn assert_resolve_verdict(env: &mycelium_l1::Env, entry: &str, oracle_input: &Nodule) {
    let (want_ok, want_hash, want_count) = match ambient::resolve(oracle_input) {
        Ok(resolved) => {
            let f = fp::fingerprint_nodule(&resolved);
            (1u32, f.hash, f.count)
        }
        Err(_) => (0u32, 0u32, 0u32),
    };
    let verdict = run_verdict3(env, entry, want_ok, want_hash, want_count);
    assert_eq!(
        verdict, 1,
        "{entry}: Stage-4 resolve differential verdict {verdict} \
         (0 = Ok/Err classification mismatch vs oracle Ok={want_ok}; \
          2 = fingerprint HASH mismatch (oracle {want_hash:#010x}); \
          3 = fingerprint NODE-COUNT mismatch (oracle {want_count}))"
    );
}

#[test]
fn ambient_myc_resolve_matches_oracle_over_synthetic_fixtures() {
    let env = checked_env();
    assert_resolve_verdict(&env, "stage4_verdict_1", &test_input_1());
    assert_resolve_verdict(&env, "stage4_verdict_2", &test_input_2());
    assert_resolve_verdict(&env, "stage4_verdict_3", &test_input_3());
    assert_resolve_verdict(&env, "stage4_verdict_4", &test_input_4());
    assert_resolve_verdict(&env, "stage4_verdict_5", &test_input_5());
    assert_resolve_verdict(&env, "stage4_verdict_6", &test_input_6());
    assert_resolve_verdict(&env, "stage4_verdict_7", &test_input_7());
    assert_resolve_verdict(&env, "stage4_verdict_8", &test_input_8());

    // Every TC1-8 fixture must be oracle-Ok (they are all constructed to be valid ambient usage);
    // assert that explicitly so a future edit that accidentally makes one invalid is caught here,
    // not silently downgraded to a vacuous Err/Err agreement.
    for (label, input) in [
        ("TC1", test_input_1()),
        ("TC2", test_input_2()),
        ("TC3", test_input_3()),
        ("TC4", test_input_4()),
        ("TC5", test_input_5()),
        ("TC6", test_input_6()),
        ("TC7", test_input_7()),
        ("TC8", test_input_8()),
    ] {
        assert!(
            ambient::resolve(&input).is_ok(),
            "{label}: expected the oracle to accept this fixture"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Leg (c): `expand_to_source` byte-for-byte parity (both on the raw fixture and on the resolved
// twin, for the fixtures that have a dedicated `_resolved_` driver in `ambient.myc`).
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn ambient_myc_expand_to_source_matches_oracle() {
    let env = checked_env();

    let want1 = ambient::expand_to_source(&test_input_1());
    let v1 = run_verdict_bytes(&env, "stage4_expand_verdict_1", want1.as_bytes());
    assert_eq!(v1, 1, "TC1 expand_to_source mismatch vs oracle:\n{want1}");

    let want2 = ambient::expand_to_source(&test_input_2());
    let v2 = run_verdict_bytes(&env, "stage4_expand_verdict_2", want2.as_bytes());
    assert_eq!(v2, 1, "TC2 expand_to_source mismatch vs oracle:\n{want2}");

    let resolved2 = ambient::resolve(&test_input_2()).expect("TC2 must resolve Ok");
    let want2r = ambient::expand_to_source(&resolved2);
    let v2r = run_verdict_bytes(&env, "stage4_expand_verdict_resolved_2", want2r.as_bytes());
    assert_eq!(
        v2r, 1,
        "TC2 (resolved) expand_to_source mismatch vs oracle:\n{want2r}"
    );

    let want7 = ambient::expand_to_source(&test_input_7());
    let v7 = run_verdict_bytes(&env, "stage4_expand_verdict_7", want7.as_bytes());
    assert_eq!(v7, 1, "TC7 expand_to_source mismatch vs oracle:\n{want7}");

    let resolved7 = ambient::resolve(&test_input_7()).expect("TC7 must resolve Ok");
    let want7r = ambient::expand_to_source(&resolved7);
    let v7r = run_verdict_bytes(&env, "stage4_expand_verdict_resolved_7", want7r.as_bytes());
    assert_eq!(
        v7r, 1,
        "TC7 (resolved) expand_to_source mismatch vs oracle:\n{want7r}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Leg (d): error-kind classification parity over the four never-silent refusals.
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn ambient_myc_error_kind_matches_oracle_over_refusal_fixtures() {
    let env = checked_env();

    let cases: [(&str, Nodule); 4] = [
        ("stage4_err_kind_1", test_input_err1()),
        ("stage4_err_kind_2", test_input_err2()),
        ("stage4_err_kind_3", test_input_err3()),
        ("stage4_err_kind_4", test_input_err4()),
    ];
    for (entry, input) in cases {
        let want_kind = match ambient::resolve(&input) {
            Ok(_) => panic!("{entry}: expected the oracle to REFUSE this fixture"),
            Err(e) => ambient_error_kind(&e),
        };
        let got_kind = run_u32_entry(&env, entry);
        assert_eq!(
            got_kind, want_kind,
            "{entry}: self-hosted error-kind {got_kind} != oracle error-kind {want_kind}"
        );
    }
}
