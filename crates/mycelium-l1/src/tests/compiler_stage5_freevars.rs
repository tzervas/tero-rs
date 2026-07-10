//! M-740 Stage 5, increment 5 (M-1010; DN-26 §7.3 row 5 / §9 flag-1) — the self-hosted
//! `compiler.semcore` free-variable + binder analysis: the LIVE-ORACLE differential gate for
//! mono.rs's `free_vars`/`free_vars_at` + `pattern_binders`/`pattern_binders_at` ported into
//! `lib/compiler/semcore.myc` as `free_vars` (+ the `fvw`/`fvw_arms` worker and
//! `pattern_binder_names`).
//!
//! **Live-oracle posture (VR-5).** Every case calls the REAL Rust `mono::free_vars` on an `Expr`
//! fixture and asserts the `.myc` port's ordered capture list equals it (order-sensitive
//! `names_list_eq_probe` — first-occurrence order matters). `free_vars` is `pub(crate)`; its worker
//! and BOTH pattern-binder fns are module-private, exercised TRANSITIVELY through `free_vars` (the
//! `match`/`for`/`lambda` shadowing arms drive them). Only this in-crate `src/tests/` module + its
//! one `mod` line were added — no visibility change to `mono.rs`.
//!
//! M-981 applies: only the L1-eval leg is exercised (small synthetic `Expr` fixtures, not a corpus
//! program — the L0/AOT leg's marginal value is low relative to its eval-depth cost, M-987).

use crate::ast::{Arm, Expr, Literal, Paradigm, Param, Path, Pattern, TypeRef};
use crate::checkty::check_nodule;
use crate::elab::build_registry;
use crate::eval::Evaluator;
use crate::mono::{free_vars, monomorphize};
use crate::parse;
use mycelium_core::Payload;
use std::collections::BTreeSet;

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

/// The driver prelude: an order-sensitive `Vec[Bytes]` equality (free_vars' first-occurrence order).
fn driver_prelude() -> String {
    r#"
fn names_list_eq_probe(a: Vec[Bytes], b: Vec[Bytes]) => Bool =
  match a {
    Nil => match b { Nil => True, Cons(_, _) => False },
    Cons(ha, ta) => match b { Nil => False, Cons(hb, tb) => and_(beq(ha, hb), names_list_eq_probe(ta, tb)) }
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

// ── the live oracle: free_vars over a fresh scope, returning the ordered capture list ─────────────
fn oracle_free_vars(e: &Expr) -> Vec<String> {
    let mut bound = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    free_vars(e, &mut bound, &mut seen, &mut out).expect("oracle free_vars within depth budget");
    out
}

// ── Rust → `.myc` fixture encoders ────────────────────────────────────────────────────────────────

fn encode_bytes(s: &str) -> String {
    format!("{s:?}")
}

fn encode_names(names: &[String]) -> String {
    let mut s = String::from("Nil");
    for n in names.iter().rev() {
        s = format!("Cons({}, {})", encode_bytes(n), s);
    }
    s
}

/// A placeholder surface `TypeRef` — free_vars ignores every `TypeRef`/`Path`/`Paradigm` slot, so a
/// trivial `Bytes` type (no guarantee) suffices wherever a fixture needs a well-typed `TypeRef`.
fn placeholder_tref() -> String {
    "TR(KwBytes, None)".to_owned()
}

fn encode_paradigm(p: Paradigm) -> &'static str {
    match p {
        Paradigm::Binary => "PBinary",
        Paradigm::Ternary => "PTernary",
        Paradigm::Dense => "PDense",
        Paradigm::Vsa => "PVsa",
    }
}

fn encode_literal(l: &Literal) -> String {
    match l {
        Literal::List(elems) => format!("List({})", encode_expr_list(elems)),
        Literal::Str(s) => format!("Str({})", encode_bytes(s)),
        Literal::Bin(s) => format!("Bin({})", encode_bytes(s)),
        // Only List/Str/Bin are exercised by the free_vars fixtures (List recurses, the others are
        // ignored leaves); the remaining literals need no encoder here.
        other => unreachable!("literal {other:?} is not exercised by the free_vars differential"),
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

fn encode_arm(a: &Arm) -> String {
    format!(
        "Ar({}, {})",
        encode_pattern(&a.pattern),
        encode_expr(&a.body)
    )
}

fn encode_arm_list(arms: &[Arm]) -> String {
    let mut s = String::from("Nil");
    for a in arms.iter().rev() {
        s = format!("Cons({}, {})", encode_arm(a), s);
    }
    s
}

fn encode_param(p: &Param) -> String {
    // The param type is a placeholder (free_vars ignores it); only the name matters.
    format!("Prm({}, {})", encode_bytes(&p.name), placeholder_tref())
}

fn encode_param_list(ps: &[Param]) -> String {
    let mut s = String::from("Nil");
    for p in ps.iter().rev() {
        s = format!("Cons({}, {})", encode_param(p), s);
    }
    s
}

fn encode_path(p: &Path) -> String {
    let mut segs = String::from("Nil");
    for seg in p.0.iter().rev() {
        segs = format!("Cons({}, {})", encode_bytes(seg), segs);
    }
    format!("Pth({segs})")
}

fn encode_expr(e: &Expr) -> String {
    match e {
        Expr::Let {
            name, bound, body, ..
        } => format!(
            "Let({}, None, {}, {})",
            encode_bytes(name),
            encode_expr(bound),
            encode_expr(body)
        ),
        Expr::If { cond, conseq, alt } => format!(
            "If({}, {}, {})",
            encode_expr(cond),
            encode_expr(conseq),
            encode_expr(alt)
        ),
        Expr::Match { scrutinee, arms } => {
            format!(
                "Match({}, {})",
                encode_expr(scrutinee),
                encode_arm_list(arms)
            )
        }
        Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        } => format!(
            "For({}, {}, {}, {}, {})",
            encode_bytes(x),
            encode_expr(xs),
            encode_bytes(acc),
            encode_expr(init),
            encode_expr(body)
        ),
        Expr::Swap { value, policy, .. } => format!(
            "Swap({}, {}, {})",
            encode_expr(value),
            placeholder_tref(),
            encode_path(policy)
        ),
        Expr::WithParadigm { paradigm, body } => {
            format!(
                "WithParadigm({}, {})",
                encode_paradigm(*paradigm),
                encode_expr(body)
            )
        }
        Expr::Wild(b) => format!("Wild({})", encode_expr(b)),
        Expr::Spore(b) => format!("Spore({})", encode_expr(b)),
        Expr::Consume(b) => format!("Consume({})", encode_expr(b)),
        Expr::Colony(hyphae) => {
            let mut s = String::from("Nil");
            for h in hyphae.iter().rev() {
                // free_vars ignores the forage policy; encode `None` for it.
                s = format!("Cons(Hy(None, {}), {})", encode_expr(&h.body), s);
            }
            format!("Colony({s})")
        }
        Expr::Lambda { params, body } => {
            format!(
                "Lambda({}, {})",
                encode_param_list(params),
                encode_expr(body)
            )
        }
        Expr::App { head, args } => {
            format!("App({}, {})", encode_expr(head), encode_expr_list(args))
        }
        Expr::Fuse { left, right } => {
            format!("Fuse({}, {})", encode_expr(left), encode_expr(right))
        }
        Expr::Reclaim { policy, body } => {
            format!("Reclaim({}, {})", encode_expr(policy), encode_expr(body))
        }
        Expr::Path(p) => format!("Path({})", encode_path(p)),
        Expr::Lit(l) => format!("Lit({})", encode_literal(l)),
        Expr::Ascribe(inner, _) => {
            format!("Ascribe({}, {})", encode_expr(inner), placeholder_tref())
        }
        Expr::TupleLit(elems) => format!("TupleLit({})", encode_expr_list(elems)),
    }
}

fn encode_expr_list(es: &[Expr]) -> String {
    let mut s = String::from("Nil");
    for e in es.iter().rev() {
        s = format!("Cons({}, {})", encode_expr(e), s);
    }
    s
}

// Small fixture constructors keeping the test bodies to `assert over a case`.
fn pathv(name: &str) -> Expr {
    Expr::Path(Path(vec![name.to_owned()]))
}
fn pathv_multi(segs: &[&str]) -> Expr {
    Expr::Path(Path(segs.iter().map(|s| (*s).to_owned()).collect()))
}
fn app(head: Expr, args: Vec<Expr>) -> Expr {
    Expr::App {
        head: Box::new(head),
        args,
    }
}
fn let_(name: &str, bound: Expr, body: Expr) -> Expr {
    Expr::Let {
        name: name.to_owned(),
        ty: None,
        bound: Box::new(bound),
        body: Box::new(body),
    }
}
fn lambda(params: &[&str], body: Expr) -> Expr {
    Expr::Lambda {
        params: params
            .iter()
            .map(|n| Param {
                name: (*n).to_owned(),
                ty: TypeRef {
                    base: crate::ast::BaseType::Bytes,
                    guarantee: None,
                },
            })
            .collect(),
        body: Box::new(body),
    }
}
fn match_(scrutinee: Expr, arms: Vec<Arm>) -> Expr {
    Expr::Match {
        scrutinee: Box::new(scrutinee),
        arms,
    }
}
fn arm(pattern: Pattern, body: Expr) -> Arm {
    Arm { pattern, body }
}
fn for_(x: &str, xs: Expr, acc: &str, init: Expr, body: Expr) -> Expr {
    Expr::For {
        x: x.to_owned(),
        xs: Box::new(xs),
        acc: acc.to_owned(),
        init: Box::new(init),
        body: Box::new(body),
    }
}

/// The witness: assert the `.myc` `free_vars` capture list equals the live `mono::free_vars` list.
fn assert_free_vars(label: &str, e: &Expr) {
    let want = oracle_free_vars(e);
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match free_vars({}) {{\n    Ok(fv) => match names_list_eq_probe(fv, {}) {{ True => one32(), False => zero32() }},\n    Err(_) => zero32()\n  }};\n",
        encode_expr(e),
        encode_names(&want)
    );
    assert_l1_only_u32(label, &program(&driver), 1);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Structural gate: `semcore.myc` (with the increment-5 additions) parses and type-checks green.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn semcore_freevars_parses_and_checks() {
    let nodule = parse(SEMCORE_SRC).unwrap_or_else(|e| panic!("semcore.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("semcore.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Non-vacuity probe: comparing against a WRONG capture list yields 0.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn freevars_witness_discriminates() {
    // free_vars of `f(x)` = ["f", "x"]; comparing against the wrong ["f"] must yield 0.
    let e = app(pathv("f"), vec![pathv("x")]);
    let driver = format!(
        "fn main() => Binary{{32}} =\n  match free_vars({}) {{\n    Ok(fv) => match names_list_eq_probe(fv, {}) {{ True => one32(), False => zero32() }},\n    Err(_) => zero32()\n  }};\n",
        encode_expr(&e),
        encode_names(&["f".to_owned()])
    );
    assert_l1_only_u32("wrong_capture_yields_zero", &program(&driver), 0);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// free_vars (LIVE — mono::free_vars): capture, de-dup + order, and shadowing across every binder.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn freevars_basic_cases() {
    // A single free var; a multi-segment path is NOT a local var (captures nothing).
    assert_free_vars("single_var", &pathv("x"));
    assert_free_vars("multi_seg_path", &pathv_multi(&["mod", "f"]));
    // App captures head + args, first-occurrence order, de-duplicated.
    assert_free_vars(
        "app_head_args",
        &app(pathv("f"), vec![pathv("x"), pathv("y")]),
    );
    assert_free_vars(
        "app_dedup_order",
        &app(pathv("f"), vec![pathv("x"), pathv("x"), pathv("g")]),
    );
    // A literal captures nothing; a list literal walks its elements.
    assert_free_vars("str_literal", &Expr::Lit(Literal::Str("hi".to_owned())));
    assert_free_vars(
        "list_literal",
        &Expr::Lit(Literal::List(vec![pathv("a"), pathv("b")])),
    );
    // Fuse / TupleLit / If / Ascribe / Swap / WithParadigm / Wild / Spore / Consume walk-throughs.
    assert_free_vars(
        "fuse",
        &Expr::Fuse {
            left: Box::new(pathv("a")),
            right: Box::new(pathv("b")),
        },
    );
    assert_free_vars("tuple_lit", &Expr::TupleLit(vec![pathv("a"), pathv("b")]));
    assert_free_vars(
        "if_cond_branches",
        &Expr::If {
            cond: Box::new(pathv("c")),
            conseq: Box::new(pathv("t")),
            alt: Box::new(pathv("e")),
        },
    );
    assert_free_vars(
        "ascribe",
        &Expr::Ascribe(
            Box::new(pathv("z")),
            TypeRef {
                base: crate::ast::BaseType::Bytes,
                guarantee: None,
            },
        ),
    );
    assert_free_vars(
        "swap",
        &Expr::Swap {
            value: Box::new(pathv("v")),
            target: TypeRef {
                base: crate::ast::BaseType::Bytes,
                guarantee: None,
            },
            policy: Path(vec!["p".to_owned()]),
        },
    );
    assert_free_vars(
        "with_paradigm",
        &Expr::WithParadigm {
            paradigm: Paradigm::Binary,
            body: Box::new(pathv("b")),
        },
    );
    assert_free_vars("wild", &Expr::Wild(Box::new(pathv("w"))));
    assert_free_vars("spore", &Expr::Spore(Box::new(pathv("s"))));
    assert_free_vars("consume", &Expr::Consume(Box::new(pathv("r"))));
    assert_free_vars(
        "reclaim",
        &Expr::Reclaim {
            policy: Box::new(pathv("pol")),
            body: Box::new(pathv("bod")),
        },
    );
    assert_free_vars(
        "colony",
        &Expr::Colony(vec![
            crate::ast::Hypha {
                forage: None,
                body: pathv("h1"),
            },
            crate::ast::Hypha {
                forage: None,
                body: app(pathv("f"), vec![pathv("h2")]),
            },
        ]),
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// free_vars binder/shadowing (LIVE): let / lambda / for / match — a bound name is NOT captured, and
// the binder's scope is restored after its body (a use outside the scope IS captured).
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn freevars_binder_cases() {
    // let: the bound name shadows in the body; the bound EXPR sees the outer scope.
    assert_free_vars(
        "let_shadow",
        &let_(
            "x",
            pathv("y"),
            app(pathv("f"), vec![pathv("x"), pathv("z")]),
        ),
    );
    // `x` free in the bound expr, then re-bound: captured once (from the bound expr), not from body.
    assert_free_vars("let_bound_uses_x", &let_("x", pathv("x"), pathv("x")));
    // lambda: params shadow inside the body; a non-param var is captured.
    assert_free_vars(
        "lambda_capture",
        &lambda(&["a"], app(pathv("f"), vec![pathv("a"), pathv("b")])),
    );
    // for: x and acc are bound in the body; xs/init see the outer scope.
    assert_free_vars(
        "for_binders",
        &for_(
            "x",
            pathv("xs"),
            "acc",
            pathv("init"),
            app(pathv("g"), vec![pathv("x"), pathv("acc"), pathv("free")]),
        ),
    );
    // match: a Ctor pattern's binders shadow in that arm; the scrutinee is captured.
    assert_free_vars(
        "match_ctor_binders",
        &match_(
            pathv("scrut"),
            vec![
                arm(
                    Pattern::Ctor(
                        "Cons".to_owned(),
                        vec![
                            Pattern::Ident("h".to_owned()),
                            Pattern::Ident("t".to_owned()),
                        ],
                    ),
                    app(pathv("f"), vec![pathv("h"), pathv("t"), pathv("outer")]),
                ),
                arm(Pattern::Wildcard, pathv("dflt")),
            ],
        ),
    );
    // match: an Ident pattern binds; a tuple pattern binds each element; a Lit pattern binds nothing.
    assert_free_vars(
        "match_ident_tuple_lit",
        &match_(
            pathv("s"),
            vec![
                arm(Pattern::Ident("v".to_owned()), pathv("v")),
                arm(
                    Pattern::Tuple(vec![
                        Pattern::Ident("a".to_owned()),
                        Pattern::Ident("b".to_owned()),
                    ]),
                    app(pathv("a"), vec![pathv("b"), pathv("c")]),
                ),
                arm(
                    Pattern::Lit(Literal::Bin("1010".to_owned())),
                    pathv("lit_body"),
                ),
            ],
        ),
    );
    // Nested: a lambda inside a let inside a match arm — shadowing composes.
    assert_free_vars(
        "nested_scopes",
        &let_(
            "outer",
            pathv("seed"),
            match_(
                pathv("outer"),
                vec![arm(
                    Pattern::Ctor("Some".to_owned(), vec![Pattern::Ident("v".to_owned())]),
                    lambda(
                        &["w"],
                        app(
                            pathv("v"),
                            vec![pathv("w"), pathv("outer"), pathv("global")],
                        ),
                    ),
                )],
            ),
        ),
    );
}
