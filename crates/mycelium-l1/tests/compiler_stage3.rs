//! M-740 Stage 3b (DN-26 §7.3 row 3) — the self-hosted `compiler.parse` port.
//!
//! `lib/compiler/parse.myc`'s `parse`/`parse_phylum` (source text -> AST) vs the live Rust oracle
//! (`mycelium_l1::{parse, parse_phylum}`, `crates/mycelium-l1/src/parse.rs`) over the full L1
//! conformance corpus (`docs/spec/grammar/conformance/{accept,reject}/`): two legs, per the task
//! brief.
//!
//! (a) **Classification parity** — for every corpus file, self-hosted `parse` Ok/Err must agree
//!     with the ORACLE's Ok/Err on the SAME source (parity with the oracle, not the directory
//!     label — some reject files fail at later pipeline stages than the parser, and the self-hosted
//!     lexer's own narrowings, FLAG-lex-2/3/4, may make a handful of inputs out of scope; any such
//!     divergence is asserted/explained explicitly here, never silently skipped).
//! (b) **AST structural fingerprint** on every file BOTH sides accept — a preorder-walk (rolling
//!     hash via `rotl(7) xor tag` + node count) computed identically on both sides (the self-hosted
//!     walker lives in `parse.myc` itself, `fingerprint_nodule`; the oracle-side mirror is
//!     `fp::fingerprint_nodule` below, hand-kept in lock-step — same 109-entry tag table (tags
//!     108/109 are the `parse_phylum` leg's `Phy` / header-less-path entries), same per-node
//!     field-visitation order). Strong enough to catch a real shape divergence (constructor
//!     kind + argument count + identifier/literal length at every node), not a bare node count.
//!
//! M-981 applies as in every prior stage: only the L1-eval leg is exercised at this scale (the L0
//! substitution interpreter is impractical for a self-hosted PARSER, ~4300 lines counting its
//! embedded lexer+AST+parser — 3x lex.myc's own already-infeasible-at-L0 scale, Stage-1's own
//! finding). M-980's split-match idiom is used throughout every driver fn below.
//!
//! Honest narrowings carried by `parse.myc` itself (full detail in-file, FLAG-parse-1..10): the
//! self-hosted `Tok`'s keyword-shaped constructors are locally `T`-prefixed to avoid a flat-namespace
//! collision with `ast.myc`'s own keyword-shaped constructors once lexer+parser+ast share one
//! self-contained nodule (FLAG-parse-2, a NEW finding this leaf surfaced); decimal-digit-run ->
//! Binary{32}/{64} conversion now lives in the parser, not the lexer (FLAG-parse-3), with an
//! unexercised overflow/narrowing gap mirroring FLAG-parse-4 (FLAG-parse-4/6/9); no generic parser
//! combinator is used (FLAG-parse-5); the depth budget is threaded explicitly, restated
//! value-semantically from `self.depth` (FLAG-parse-7); error POSITION/message fidelity is not
//! compared, only Ok/Err classification (FLAG-parse-8); `parse_phylum` is fully ported (FLAG-parse-9);
//! effect-budget VALUES are not mixed into the fingerprint, only their names (FLAG-parse-10).
//!
//! **Recursion discipline (RFC-0041 §7 W7 amendment 11, PR #1166 review fix):** every
//! SOURCE-LENGTH-bounded loop in parse.myc — item/arm/hypha/ctor/param/segment lists, the embedded
//! lexer's token/lexeme scanners — is written in **accumulator + reverse direct-tail style**
//! (`loop(ts, Cons(item, acc))` tail-calls itself; one final `rev_acc(acc, Nil)` restores source
//! order). Expression/type/pattern NESTING recursion is the other, deliberately different class:
//! recursive descent proper, bounded by the explicit 4096 depth budget (FLAG-parse-7), not by TCO.
//!
//! **FLAG-parse-11, CLOSED by M-994 fix (a):** the depth BENEFIT of the direct-tail shape used to
//! be **dormant** — the L1 evaluator's TCO elided a self-call only when the caller's `InvokePost`
//! frame was directly on top of the CEK stack, and a `match` arm body runs under a
//! `Frame::MatchPop` (scope restore), a `let` body under `Frame::LetPop`, so tail calls from
//! inside match/let were NOT elided (probed: a 10,000-iteration match-arm tail loop tripped
//! `DepthExceeded{4096}` with `tco_trace().total_elided == 1`). `eval.rs::enter_call` now looks
//! THROUGH a run of binder-restoring `MatchPop`/`LetPop` frames (RFC-0041 §4.6), so every
//! terminating loop's `match`-driven tail call elides — the DEPTH cost of list length is gone, so
//! the reviewer's 5,000-item repro (and deeper) no longer refuses on the depth budget. (A SEPARATE
//! throughput wall remains — L1-eval cost is ~n^3 in token count, M-987/fix (b) — so a *large* N
//! is still slow, just no longer depth-bounded; the two concerns are kept distinct in the pins
//! below.) Witnessed by `l1_eval_tco_match_arm_tail_call_is_elided` (micro scale) +
//! `parse_myc_many_item_nodule_depth_semantics` Part 2 (macro scale, reduced 512-frame budget,
//! same modest N as Part 1); the correctness guard `l1_eval_non_tail_self_call_still_refuses_depth`
//! proves a NON-tail self-call still refuses (no over-elision).

use mycelium_l1::ast::{
    AmbientParams, Arm, BaseType, Ctor, DeriveDecl, ExecutionMode, Expr, FnDecl, FnSig, Hypha,
    ImplDecl, InherentImplDecl, Item, Literal, LowerDecl, LowerRhs, ObjectDecl, Paradigm, Param,
    ParamKind, Path, Pattern, Scalar, Sparsity, Strength, TraitDecl, TraitRef, TypeDecl, TypeParam,
    TypeRef, ViaDecl, WidthRef,
};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator, Nodule, UsePath, Vis};

const PARSE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/parse.myc"
));

fn program(driver: &str) -> String {
    format!("{PARSE_SRC}\n{driver}")
}

/// L1-eval-only assertion (the M-981 convention every prior stage uses).
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
    let repr_val = l1_core
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: expected a Repr CoreValue, got {l1_core:?}"));
    let got = match repr_val.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("{label}: expected a Bits payload, got {other:?}"),
    };
    assert_eq!(
        got, expected_u32,
        "{label}: L1-eval result {got} does not match the expected value {expected_u32}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The structural gate: `parse.myc` parses and type-checks green (no driver needed).
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn parse_myc_parses_and_checks() {
    let nodule = parse(PARSE_SRC).unwrap_or_else(|e| panic!("parse.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("parse.myc: check failed: {e}"));
}

#[test]
fn parse_myc_lexes_and_parses_a_trivial_source() {
    assert_l1_only_u32(
        "trivial nodule parses Ok",
        &program(r#"fn main() => Binary{32} = parse_ok_code("nodule d;\n");"#),
        1,
    );
    assert_l1_only_u32(
        "empty source is Err",
        &program(r#"fn main() => Binary{32} = parse_ok_code("");"#),
        0,
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The oracle-side AST fingerprint mirror (`fp` module) — hand-kept in lock-step with parse.myc's
// own `walk_*`/`fp_tag` family: SAME 109-entry tag table (1..109, sequential, no gaps; 108/109 =
// the parse_phylum leg's Phy / header-less-path entries), SAME
// rotl(7)-xor mixing, SAME per-node field-visitation order (occasionally NOT the natural
// declaration order — e.g. `TypeDecl`/`TraitDecl`/`ObjectDecl` mix `name` before `vis` — reproduced
// exactly here since only cross-side CONSISTENCY matters, not "natural" order).
// ─────────────────────────────────────────────────────────────────────────────────────────────
mod fp {
    use super::*;

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

    fn paramkind(fp: Fp, k: &ParamKind) -> Fp {
        tag(fp, if matches!(k, ParamKind::Type) { 17 } else { 18 })
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

    // FLAG-parse-10 mirror: effect budgets are skipped except for the effect NAME (already covered
    // by `bytes_list` over `FnSig::effects`); `effect_budgets` (a BTreeMap) contributes nothing.

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

    // The FLAG-parse-9 `parse_phylum` leg (tags 108 = Phy, 109 = header-less path None; a headed
    // path mixes `path` directly — the same Option-mixing convention `guarantee_opt`/`tier_opt`
    // use, mirrored from parse.myc's `walk_phylum_path_opt`).
    pub fn fingerprint_phylum(ph: &mycelium_l1::Phylum) -> Fp {
        let fp = Fp { hash: 0, count: 0 };
        let fp = tag(fp, 108);
        let fp = match &ph.path {
            None => tag(fp, 109),
            Some(p) => path(fp, p),
        };
        ph.nodules.iter().fold(fp, nodule)
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The Stage-3b gate: classification parity (all files) + AST fingerprint parity (accepted files).
//
// Runtime economy (the DN-26 §7.3 runtime-budget note; the Stage-1 differential took ~213s at 1/3
// this program size): the whole differential runs as ONE `check_nodule` of `parse.myc` plus ONE
// `Evaluator::call("stage3_verdict", ...)` per corpus file — the oracle-computed expectations
// (Ok/Err code, fingerprint hash, node count) travel INTO the self-hosted program as arguments
// (a `Bytes` value + three `Binary{32}`s), and a single packed verdict comes back. No per-file
// re-check/re-monomorphize (the `Evaluator::new(&env).call(entry, args)` pattern
// `tests/enablement.rs` established).
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn bytes_value(s: &str) -> mycelium_core::Value {
    mycelium_core::Value::new(
        mycelium_core::Repr::Bytes,
        mycelium_core::Payload::Bytes(s.as_bytes().to_vec()),
        mycelium_core::Meta::exact(mycelium_core::Provenance::Root),
    )
    .expect("a Bytes value from source text is well-formed")
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

#[test]
fn parse_myc_matches_oracle_over_the_full_conformance_corpus() {
    let started = std::time::Instant::now();
    // ONE parse + check of the self-hosted parser (no driver appended — `stage3_verdict` is
    // defined in parse.myc itself).
    let env =
        check_nodule(&parse(PARSE_SRC).unwrap_or_else(|e| panic!("parse.myc: parse failed: {e}")))
            .unwrap_or_else(|e| panic!("parse.myc: check failed: {e}"));

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dirs = [
        "docs/spec/grammar/conformance/accept",
        "docs/spec/grammar/conformance/reject",
    ];
    let mut total = 0usize;
    let mut accepted = 0usize;
    let mut phylum_accepted = 0usize;
    for dir in dirs {
        let dir_path = root.join(dir);
        let mut files: Vec<_> = std::fs::read_dir(&dir_path)
            .unwrap_or_else(|e| panic!("cannot read {dir_path:?}: {e}"))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "myc"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no .myc files found under {dir_path:?}");
        for path in &files {
            let source = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
            let label = format!("{dir}/{:?}", path.file_name().unwrap());

            // Leg 1 — the `parse` (single-nodule) entry: oracle classification + (on Ok) the
            // mirror fingerprint, one eval of `stage3_verdict`.
            let (want_ok, want_hash, want_count) = match &parse(&source) {
                Ok(n) => {
                    let f = fp::fingerprint_nodule(n);
                    (1u32, f.hash, f.count)
                }
                Err(_) => (0u32, 0u32, 0u32),
            };
            run_verdict(
                &env,
                "stage3_verdict",
                &label,
                &source,
                want_ok,
                want_hash,
                want_count,
            );
            if want_ok == 1 {
                accepted += 1;
            }

            // Leg 2 — the `parse_phylum` entry (FLAG-parse-9): same shape over the phylum AST.
            // This is what covers `accept/19-phylum-cross-nodule.myc` (a phylum-headed fixture
            // the single-nodule `parse` entry REJECTS on both sides — its acceptance lives here).
            let (pwant_ok, pwant_hash, pwant_count) = match &mycelium_l1::parse_phylum(&source) {
                Ok(ph) => {
                    let f = fp::fingerprint_phylum(ph);
                    (1u32, f.hash, f.count)
                }
                Err(_) => (0u32, 0u32, 0u32),
            };
            run_verdict(
                &env,
                "stage3_phylum_verdict",
                &format!("{label} [phylum leg]"),
                &source,
                pwant_ok,
                pwant_hash,
                pwant_count,
            );
            if pwant_ok == 1 {
                phylum_accepted += 1;
            }
            total += 1;
        }
    }
    assert!(
        total >= 57,
        "expected the full accept+reject corpus (~57 files), found {total}"
    );
    // 26 of the 27 accept-corpus files are single-nodule (oracle-`parse`-Ok); the 27th,
    // `19-phylum-cross-nodule.myc`, is phylum-headed — `parse` rejects it ON BOTH SIDES (parity
    // held above) and its acceptance is asserted through the `parse_phylum` leg instead.
    assert_eq!(
        accepted, 26,
        "expected exactly 26 accept-corpus files to be oracle-`parse`-Ok (all but the phylum fixture)"
    );
    assert_eq!(
        phylum_accepted, 27,
        "expected all 27 accept-corpus files to be oracle-`parse_phylum`-Ok (parse_phylum is a strict superset of parse)"
    );
    eprintln!(
        "stage3 corpus differential: {total} files x 2 legs (parse: {accepted} accepted; \
         parse_phylum: {phylum_accepted} accepted; fingerprints compared on every accepted leg, \
         classification parity on all) in {:.1}s",
        started.elapsed().as_secs_f64()
    );
}

/// One self-hosted verdict eval: call `entry(src, want_ok, want_hash, want_count)` in the checked
/// `env` and assert the packed verdict is 1 (full agreement). Runs under the default depth budget
/// (the FLAG-parse-11 reduced-budget pin builds its own `Evaluator` inline).
fn run_verdict(
    env: &mycelium_l1::Env,
    entry: &str,
    label: &str,
    source: &str,
    want_ok: u32,
    want_hash: u32,
    want_count: u32,
) {
    let args = vec![
        mycelium_l1::L1Value::Repr(bytes_value(source)),
        mycelium_l1::L1Value::Repr(b32_value(want_ok)),
        mycelium_l1::L1Value::Repr(b32_value(want_hash)),
        mycelium_l1::L1Value::Repr(b32_value(want_count)),
    ];
    // A raised step budget (default 1M): interpreting the self-hosted parser over a REAL ~100-line
    // lib file costs a few million steps (`lib/std/fmt.myc` exhausted the default) — the budget
    // stays finite (the non-termination guard holds, RFC-0007 §4.5/§4.6), just sized to the
    // workload. The conformance-corpus files fit comfortably under the same ceiling.
    let verdict_val = Evaluator::new(env)
        .with_fuel(200_000_000)
        .call(entry, args)
        .unwrap_or_else(|e| panic!("{label}: L1-eval of {entry} failed: {e}"));
    let repr = verdict_val
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: verdict must be a repr value"));
    let verdict = match repr.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("{label}: expected a Bits verdict payload, got {other:?}"),
    };
    assert_eq!(
        verdict, 1,
        "{label}: Stage-3 differential verdict {verdict} \
         (0 = Ok/Err classification mismatch vs oracle Ok={want_ok}; \
          2 = fingerprint HASH mismatch (oracle {want_hash:#010x}); \
          3 = fingerprint NODE-COUNT mismatch (oracle {want_count}))"
    );
}

/// The DN-26 §7.3 "plus `lib/std/*.myc` + `lib/compiler/*.myc` where runtime permits" leg —
/// REAL self-hosted programs, not conformance fixtures. **Honest runtime narrowing (the Stage-1
/// FLAG precedent):** L1-eval'ing the self-hosted parser over a source file costs time roughly
/// linear in that file's token count; the conformance corpus averages ~10 lines/file, but the lib
/// tree runs to 850+ lines (lex.myc) and 4400+ (parse.myc itself), which would multiply the gate's
/// wall-clock severalfold. This test therefore differentials the SMALLEST six real lib files
/// (every one under 125 lines — result/option/math/fmt/cmp/iter), classification + fingerprint,
/// both legs each — a documented subset, never a silent skip; the full-tree sweep is deliberate
/// follow-on work once the AOT leg (M-981) makes per-file cost negligible.
#[test]
fn parse_myc_matches_oracle_on_a_small_real_lib_subset() {
    let started = std::time::Instant::now();
    let env =
        check_nodule(&parse(PARSE_SRC).unwrap_or_else(|e| panic!("parse.myc: parse failed: {e}")))
            .unwrap_or_else(|e| panic!("parse.myc: check failed: {e}"));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let subset = [
        "lib/std/result.myc",
        "lib/std/option.myc",
        "lib/std/math.myc",
        "lib/std/fmt.myc",
        "lib/std/cmp.myc",
        "lib/std/iter.myc",
    ];
    for rel in subset {
        let source = std::fs::read_to_string(root.join(rel))
            .unwrap_or_else(|e| panic!("cannot read {rel}: {e}"));
        let (want_ok, want_hash, want_count) = match &parse(&source) {
            Ok(n) => {
                let f = fp::fingerprint_nodule(n);
                (1u32, f.hash, f.count)
            }
            Err(_) => (0u32, 0u32, 0u32),
        };
        // Every file in this subset is a real, landed stdlib nodule — the oracle must accept it.
        assert_eq!(
            want_ok, 1,
            "{rel}: expected the oracle to accept a landed stdlib nodule"
        );
        run_verdict(
            &env,
            "stage3_verdict",
            rel,
            &source,
            want_ok,
            want_hash,
            want_count,
        );
    }
    eprintln!(
        "stage3 lib-subset differential: {} real stdlib files, classification + fingerprint, in {:.1}s",
        subset.len(),
        started.elapsed().as_secs_f64()
    );
}

/// PR #1166 HIGH regression pins — the source-length-bounded-recursion finding, in two parts.
/// **Both parts now GREEN under M-994 fix (a)** (were: Part 1 green / Part 2 a known-gap pin).
///
/// **Part 1 (green leg):** a 150-item synthetic nodule (752 tokens — comfortably under the 4096
/// depth budget) parses green AND fingerprint-matches the oracle under the DEFAULT depth budget —
/// the direct-tail rewrite changed recursion SHAPE only, never any AST result. Timing on this
/// machine (debug, one eval): see the eprintln — eval cost is grossly SUPER-LINEAR in token count
/// (~n^2.5-3 over this range; Empirical), which is a SEPARATE throughput concern (M-987, the ~n^3
/// per-reference-copy cost — fix (b)), NOT the depth question this pin pair proves. N stays modest
/// on purpose: the DEPTH fix is proved by the SAME input passing a budget that previously refused
/// it, not by iterating deeper.
///
/// **Part 2 (FLAG-parse-11, CLOSED by M-994 fix (a) — the depth proof):** the SAME 150-item input
/// under a REDUCED 512-frame depth budget now PARSES SUCCESSFULLY (full-agreement verdict) — was:
/// `DepthExceeded{512}`, because the evaluator's TCO did not elide tail calls made from match arms
/// / let bodies (the `MatchPop`/`LetPop` frames blocked the InvokePost-on-top precondition), so
/// list length consumed ~1 frame per item whatever the source shape. `eval.rs::enter_call` now
/// looks through those frames (RFC-0041 §4.6), so the item-list tail loop is flat: only genuine
/// expression NESTING consumes depth, never list LENGTH — the input that refused
/// `DepthExceeded{512}` before now passes the same 512 budget (an 8x reduction from the 4096
/// default). That input-that-refused-now-passes IS the fix-(a) depth proof; raising N to iterate
/// much deeper/faster is a THROUGHPUT question gated on M-987 (the ~n^3 eval cost — fix (b)), not
/// on this depth fix, so it is deliberately not attempted here.
#[test]
fn parse_myc_many_item_nodule_depth_semantics() {
    let env =
        check_nodule(&parse(PARSE_SRC).unwrap_or_else(|e| panic!("parse.myc: parse failed: {e}")))
            .unwrap_or_else(|e| panic!("parse.myc: check failed: {e}"));
    let n = 150usize;
    let mut source = String::from("nodule many.items;\n");
    for i in 0..n {
        source.push_str(&format!("use a.b{i};\n"));
    }
    let oracle = parse(&source)
        .unwrap_or_else(|e| panic!("the oracle must accept the synthetic {n}-item input: {e}"));
    let f = fp::fingerprint_nodule(&oracle);

    // Part 1: green + fingerprint parity under the default depth budget.
    let t = std::time::Instant::now();
    run_verdict(
        &env,
        "stage3_verdict",
        &format!("synthetic {n}-item nodule (default depth budget)"),
        &source,
        1,
        f.hash,
        f.count,
    );
    eprintln!(
        "many-item regression: {n} items (752 tokens) parsed + fingerprint-matched under the \
         default depth budget in {:.1}s",
        t.elapsed().as_secs_f64()
    );

    // Part 2 (M-994 fix (a) flip — the DEPTH proof): the FLAG-parse-11 pin is now a GREEN
    // assertion. The SAME 150-item input that previously refused `DepthExceeded{512}` now PARSES
    // SUCCESSFULLY under the reduced 512-frame budget, because the item-list tail loop is elided
    // through MatchPop/LetPop, so list length no longer consumes a stack frame per item. NOTE: N is
    // deliberately kept equal to Part 1 (not raised) — proving the DEPTH fix only needs the same
    // input to pass a budget that used to refuse it. Iterating much deeper/faster (a large N under
    // a tighter budget) is gated on M-987 (the ~n^3 eval COST wall, fix (b)), a separate throughput
    // fix; this pin proves fix (a) alone.
    let args = vec![
        mycelium_l1::L1Value::Repr(bytes_value(&source)),
        mycelium_l1::L1Value::Repr(b32_value(1)),
        mycelium_l1::L1Value::Repr(b32_value(f.hash)),
        mycelium_l1::L1Value::Repr(b32_value(f.count)),
    ];
    let verdict_val = Evaluator::new(&env)
        .with_fuel(200_000_000)
        .with_depth(512)
        .call("stage3_verdict", args)
        .unwrap_or_else(|e| {
            panic!(
                "M-994 fix (a): an {n}-item input must now parse successfully under a reduced \
                 512-frame budget (the item-list loop is tail-elided; only expression NESTING \
                 should consume depth) -- got an error instead: {e}"
            )
        });
    let repr = verdict_val
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr verdict, got {verdict_val:?}"));
    let verdict = match repr.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("expected a Bits verdict payload, got {other:?}"),
    };
    assert_eq!(
        verdict, 1,
        "Part 2 (reduced 512-frame budget): Stage-3 differential verdict {verdict} \
         (0 = Ok/Err classification mismatch; 2 = fingerprint HASH mismatch; \
          3 = fingerprint NODE-COUNT mismatch) -- expected full agreement (1)"
    );
}

/// The FLAG-parse-11 gap, pinned at MICRO scale (fast — no parse.myc involved) — **now CLOSED by
/// M-994 fix (a)**: the L1 evaluator's TCO used to elide a self-call ONLY from bare-body position
/// (`spin(n) = spin(n)`, the kernel's own witness shape); a tail call from inside a `match` arm was
/// NOT elided (the arm body evaluates under a `Frame::MatchPop` scope-restore frame, so the
/// caller's `InvokePost` was not directly on top). `eval.rs::enter_call` now looks THROUGH a run of
/// binder-restoring `MatchPop`/`LetPop` frames to find the tail-eligible `InvokePost` underneath
/// (RFC-0041 §4.6), so a 10,000-iteration match-arm tail countdown now RUNS TO COMPLETION well past
/// the default 4096 depth budget, with one elision per iteration (was: 1, the bare `main`->`count`
/// hop only). Was `l1_eval_tco_gap_match_arm_tail_call_is_not_elided` (the FLAG-nodule-2 flip
/// precedent); see also `parse_myc_many_item_nodule_depth_semantics` Part 2 for the macro-scale
/// witness.
#[test]
fn l1_eval_tco_match_arm_tail_call_is_elided() {
    let src = format!(
        "nodule d;\nfn count(n: Binary{{32}}) => Binary{{32}} =\n  match eq(n, 0b{z:032b}) {{ 0b1 => n, _ => count(sub_u(n, 0b{o:032b})) }};\nfn main() => Binary{{32}} = count(0b{n:032b});",
        z = 0,
        o = 1,
        n = 10_000u32
    );
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("probe parse failed: {e}")))
        .unwrap_or_else(|e| panic!("probe check failed: {e}"));
    let ev = Evaluator::new(&env).with_fuel(100_000_000);
    let out = ev.call("main", vec![]).unwrap_or_else(|e| {
        panic!(
            "M-994 fix (a): a 10,000-iteration match-arm tail countdown must now run to \
             completion under the widened TCO (tail position seen through MatchPop) -- got an \
             error instead: {e}"
        )
    });
    let repr = out
        .as_repr()
        .unwrap_or_else(|| panic!("expected a Repr result, got {out:?}"));
    let got = match repr.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("expected a Bits payload, got {other:?}"),
    };
    assert_eq!(got, 0, "count(10_000) must terminate at 0, got {got}");
    let elided = ev.tco_trace().total_elided;
    assert!(
        elided >= 10_000,
        "expected >= 10,000 elisions (one per match-arm tail hop, plus the initial main->count \
         hop), got {elided}"
    );
}

/// CORRECTNESS GUARD for M-994 fix (a) (adapted from the M-994 spike's
/// `tests/spike_m994_tco.rs::spike_non_tail_self_call_still_deepens_and_refuses`): a NON-tail
/// self-call must STILL keep its frame -- the fix widens tail position through `MatchPop`/`LetPop`,
/// it must NOT wrongly elide a call whose result is CONSUMED by the caller. Here the recursive call
/// is an ARGUMENT to `add_u` (`add_u(n, sum(sub_u(n, 1)))`), not in tail position -- deep recursion
/// must still trip the depth budget explicitly (`DepthExceeded`), never silently elide and return a
/// wrong answer.
#[test]
fn l1_eval_non_tail_self_call_still_refuses_depth() {
    let src = format!(
        "nodule d;\nfn sum(n: Binary{{32}}) => Binary{{32}} =\n  match eq(n, 0b{z:032b}) {{ 0b1 => 0b{z:032b}, _ => add_u(n, sum(sub_u(n, 0b{o:032b}))) }};\nfn main() => Binary{{32}} = sum(0b{big:032b});",
        z = 0,
        o = 1,
        big = 10_000u32
    );
    let env = check_nodule(&parse(&src).unwrap_or_else(|e| panic!("probe parse failed: {e}")))
        .unwrap_or_else(|e| panic!("probe check failed: {e}"));
    let ev = Evaluator::new(&env).with_fuel(100_000_000);
    let err = ev.call("main", vec![]).expect_err(
        "a non-tail deep self-recursion (the recursive call is an add_u ARGUMENT, not in tail \
         position) must still refuse with DepthExceeded -- the widened TCO must not over-elide",
    );
    assert!(
        matches!(err, mycelium_l1::L1Error::DepthExceeded { limit: 4096 }),
        "expected DepthExceeded(4096) for the non-tail recursion, got: {err}"
    );
}
