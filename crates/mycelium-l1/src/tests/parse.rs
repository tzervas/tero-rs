//! Behavioral tests for the M-640 separated-list / keyword `expect` factoring. They drive the
//! private helpers through the public [`parse`] entry on representative grammar so the folded
//! call sites are exercised end-to-end (empty / single / multi lists, the match-arm trailing
//! comma, and that bare lists reject a trailing comma) — pinning byte-identical behavior.
use crate::ast::{BaseType, Expr, Item, Literal, TypeRef};
use crate::parse::*;

fn fn_body(src: &str) -> Expr {
    let n = parse(src).expect("parses");
    n.items
        .into_iter()
        .find_map(|i| match i {
            Item::Fn(fd) => Some(fd.body),
            _ => None,
        })
        .expect("a fn item")
}

// --- RFC-0025 / M-705: operator syntax (infix sugar desugaring to word functions) ----------

/// The body of `fn main() => T = <expr>` for an operator-sugar fixture.
fn op_body(expr: &str) -> Expr {
    fn_body(&format!("nodule d;\nfn main() => Binary{{8}} = {expr};"))
}

#[test]
fn infix_sugar_desugars_to_the_word_call() {
    // `a + b` is *structurally identical* to `add(a, b)` after parsing — the sugar leaves no
    // separate trace (RFC-0025 Q5: the desugared App node is the EXPLAIN record).
    assert_eq!(op_body("a + b"), op_body("add(a, b)"));
    assert_eq!(op_body("a - b"), op_body("sub(a, b)"));
    assert_eq!(op_body("a * b"), op_body("mul(a, b)"));
    assert_eq!(op_body("a ^ b"), op_body("xor(a, b)"));
    assert_eq!(op_body("a / b"), op_body("div(a, b)"));
    assert_eq!(op_body("a % b"), op_body("rem(a, b)"));
    assert_eq!(op_body("a & b"), op_body("band(a, b)"));
    assert_eq!(op_body("a | b"), op_body("bor(a, b)"));
    assert_eq!(op_body("a == b"), op_body("eq(a, b)"));
    assert_eq!(op_body("a != b"), op_body("ne(a, b)"));
    assert_eq!(op_body("a && b"), op_body("and(a, b)"));
    assert_eq!(op_body("a || b"), op_body("or(a, b)"));
    // RFC-0037 D1 + RFC-0025 §4.1 (M-745): the angle/shift glyphs freed by the bracket kind-split
    // (`<`/`>` no longer open a type-arg list). Each desugars to its canonical word call exactly
    // like the arithmetic/bitwise glyphs above. `<=`/`>=` have no glyph — their word forms
    // `lte`/`gte` are ordinary calls (RFC-0037 retired the two-char glyphs).
    assert_eq!(op_body("a < b"), op_body("lt(a, b)"));
    assert_eq!(op_body("a > b"), op_body("gt(a, b)"));
    assert_eq!(op_body("a << b"), op_body("shl(a, b)"));
    assert_eq!(op_body("a >> b"), op_body("shr(a, b)"));
}

/// M-916 inventory check (RFC-0025 §4.2): `lte`/`gte` have no glyph — they are word-canonical
/// already and parse as *ordinary* application (`App { head: Path(["lte"|"gte"]), args: [a, b] }`),
/// structurally identical to any other named call (e.g. `add(a, b)`). No desugaring step applies
/// to them — the parser's `infix_op` table (which drives the glyph desugar) has no `<=`/`>=` entry
/// at all, since no such token exists (confirmed at the lexer level).
#[test]
fn lte_gte_are_ordinary_calls_no_glyph_no_desugar() {
    let lte = op_body("lte(a, b)");
    let Expr::App { head, args } = &lte else {
        panic!("expected an App node, got: {lte:?}");
    };
    assert!(matches!(head.as_ref(), Expr::Path(p) if p.0 == vec!["lte".to_owned()]));
    assert_eq!(args.len(), 2);

    let gte = op_body("gte(a, b)");
    let Expr::App { head, args } = &gte else {
        panic!("expected an App node, got: {gte:?}");
    };
    assert!(matches!(head.as_ref(), Expr::Path(p) if p.0 == vec!["gte".to_owned()]));
    assert_eq!(args.len(), 2);
}

/// M-916 inventory check (RFC-0025 §4.2 / RFC-0037 D1): the retired `<=`/`>=` glyphs are a
/// never-silent parse refusal, not a silent reinterpretation as the old two-char operator or a
/// panic. `a <= b` lexes as `a`, `LAngle`, `Eq`, `b` (confirmed at the lexer level), so the parser
/// reads `a < (= b)`: `<` opens a valid comparison RHS, and `=` cannot start an expression there.
#[test]
fn le_ge_glyphs_are_a_never_silent_parse_refusal() {
    let le = parse("nodule d;\nfn f(a: Binary{8}, b: Binary{8}) => Binary{8} = a <= b;")
        .expect_err("the retired `<=` glyph must not silently parse");
    assert!(
        le.message.contains("expected an expression"),
        "expected an explicit refusal at the `=`, got: {}",
        le.message
    );
    let ge = parse("nodule d;\nfn f(a: Binary{8}, b: Binary{8}) => Binary{8} = a >= b;")
        .expect_err("the retired `>=` glyph must not silently parse");
    assert!(
        ge.message.contains("expected an expression"),
        "expected an explicit refusal at the `=`, got: {}",
        ge.message
    );
}

#[test]
fn prefix_sugar_desugars_to_the_word_call() {
    assert_eq!(op_body("-a"), op_body("neg(a)"));
    assert_eq!(op_body("!a"), op_body("not(a)"));
    // Prefix is right-associative and binds tighter than any binary op.
    assert_eq!(op_body("- -a"), op_body("neg(neg(a))"));
    assert_eq!(op_body("-a + b"), op_body("add(neg(a), b)"));
}

#[test]
fn precedence_follows_the_rust_table() {
    // `*` (70) > `+` (60): `a + b * c` ≡ `add(a, mul(b, c))`.
    assert_eq!(op_body("a + b * c"), op_body("add(a, mul(b, c))"));
    // `&` (50) > `^` (40) > `|` (30): `a | b ^ c & d` ≡ `bor(a, xor(b, band(c, d)))`.
    assert_eq!(
        op_body("a | b ^ c & d"),
        op_body("bor(a, xor(b, band(c, d)))")
    );
    // arithmetic (60) > equality (20) > `&&` (11) > `||` (10).
    assert_eq!(
        op_body("a + b == c && d || e"),
        op_body("or(and(eq(add(a, b), c), d), e)")
    );
}

#[test]
fn comparison_and_shift_tiers_follow_the_rust_table() {
    // RFC-0025 §4.1 (M-745), grounded in Rust's table (the cited source of truth): shift binds
    // *tighter* than the bitwise ops, comparison *looser* than them. This pins the precedence to
    // the ratified §4.1 tiers — NOT RFC-0037 §6's illustrative sketch, which nested shift adjacent
    // to comparison (shift looser than `|`), contradicting §4.1 (see the parse.rs `infix_op` note).
    //
    // Shift (Tier 4, bp 55): tighter than `&`/`^`/`|` (50/40/30), looser than `+`/`-` (60).
    //   `a + b << c & d` ≡ band(shl(add(a, b), c), d).
    assert_eq!(
        op_body("a + b << c & d"),
        op_body("band(shl(add(a, b), c), d)")
    );
    // Comparison (Tier 8, bp 25): looser than `|` (30), tighter than `==`/`!=` (20).
    //   `a | b < c == d` ≡ eq(lt(bor(a, b), c), d).
    assert_eq!(
        op_body("a | b < c == d"),
        op_body("eq(lt(bor(a, b), c), d)")
    );
    // Shift is strictly tighter than comparison (Tier 4 vs Tier 8):
    //   `a << b < c >> d` ≡ lt(shl(a, b), shr(c, d)).
    assert_eq!(
        op_body("a << b < c >> d"),
        op_body("lt(shl(a, b), shr(c, d))")
    );
}

#[test]
fn binary_operators_are_left_associative() {
    // `a - b - c` ≡ `sub(sub(a, b), c)`, NOT `sub(a, sub(b, c))`.
    assert_eq!(op_body("a - b - c"), op_body("sub(sub(a, b), c)"));
    assert_eq!(op_body("a + b + c"), op_body("add(add(a, b), c)"));
    // The M-745 tiers are left-associative too. Note `a < b < c` ≡ `lt(lt(a, b), c)`: RFC-0025
    // §4.1 makes *every* binary operator left-associative, so comparison chains (unlike Rust's
    // non-associative `<`) parse without error rather than being rejected.
    assert_eq!(op_body("a << b << c"), op_body("shl(shl(a, b), c)"));
    assert_eq!(op_body("a < b < c"), op_body("lt(lt(a, b), c)"));
}

#[test]
fn parens_override_precedence() {
    assert_eq!(op_body("(a + b) * c"), op_body("mul(add(a, b), c)"));
}

#[test]
fn deep_operator_nesting_is_refused_not_crashed() {
    // A4-02 / G2: a crafted prefix chain (`!!!…a`) or parenthesized operator nesting must be
    // refused with an explicit depth error, never drive a host-stack overflow. Both the prefix
    // recursion (parse_unary) and the precedence recursion (parse_binexpr) participate in the
    // shared MAX_EXPR_DEPTH budget.
    // RFC-0041 §4.2/§7 (W1): MAX_EXPR_DEPTH raised 256 → 4096, so exceed 4096 to trip the guard.
    let prefix = "!".repeat(5000);
    let src = format!("nodule d;\nfn main() => Binary{{8}} = {prefix}0b0000_0000;");
    let err = parse(&src).expect_err("a 5000-deep prefix chain must be refused");
    assert!(
        err.message.contains("refusing to recurse"),
        "got: {}",
        err.message
    );
    // A flat (non-nested) left-associative chain of the SAME length must still parse — the loop
    // keeps it O(1) deep, so length alone never trips the budget.
    let flat = (0..2000)
        .map(|_| "0b0000_0000")
        .collect::<Vec<_>>()
        .join(" ^ ");
    let ok = format!("nodule d;\nfn main() => Binary{{8}} = {flat};");
    parse(&ok).expect("a long FLAT operator chain parses (loop, not recursion)");
}

#[test]
fn word_form_remains_valid_alongside_sugar() {
    // The sugar is additive: the canonical word call still parses (and is a legal operand of
    // sugar). `add(a, b) * c` ≡ `mul(add(a, b), c)`.
    assert_eq!(op_body("add(a, b) * c"), op_body("mul(add(a, b), c)"));
}

#[test]
fn empty_list_literal_parses_to_no_elems() {
    // `comma_separated_until(RBracket)` empty path.
    let Expr::Lit(Literal::List(elems)) = fn_body("nodule d;\nfn main() => Binary{1} = [];") else {
        panic!("expected a list literal");
    };
    assert!(elems.is_empty());
}

#[test]
fn single_and_multi_element_list_literals() {
    let one = fn_body("nodule d;\nfn main() => Binary{1} = [0b0];");
    let Expr::Lit(Literal::List(e1)) = one else {
        panic!("list")
    };
    assert_eq!(e1.len(), 1);
    let many = fn_body("nodule d;\nfn main() => Binary{1} = [0b0, 0b1, 0b0];");
    let Expr::Lit(Literal::List(e3)) = many else {
        panic!("list")
    };
    assert_eq!(e3.len(), 3);
}

#[test]
fn call_args_empty_single_multi() {
    // `comma_separated_until(RParen)` for application args.
    let zero = fn_body("nodule d;\nfn main() => Binary{1} = f();");
    let Expr::App { args, .. } = zero else {
        panic!("app")
    };
    assert_eq!(args.len(), 0);
    let two = fn_body("nodule d;\nfn main() => Binary{1} = f(0b0, 0b1);");
    let Expr::App { args, .. } = two else {
        panic!("app")
    };
    assert_eq!(args.len(), 2);
}

#[test]
fn ctor_fields_and_type_params_and_args() {
    // Constructor fields (`comma_separated` after `(`), type params/args (`<…>`).
    let n = parse(
        "nodule d;\ntype Pair[A, B] = MkPair(A, B);\nfn id(x: Pair[Binary{1}, Binary{1}]) => Pair[Binary{1}, Binary{1}] = x;",
    )
    .expect("parses");
    let Item::Type(td) = &n.items[0] else {
        panic!("type decl")
    };
    assert_eq!(td.params, vec!["A".to_owned(), "B".to_owned()]);
    assert_eq!(td.ctors.len(), 1);
    assert_eq!(td.ctors[0].fields.len(), 2); // two ctor fields
}

#[test]
fn value_params_empty_and_nonempty() {
    let zero = parse("nodule d;\nfn main() => Binary{1} = 0b0;").expect("parses");
    let Item::Fn(fd) = &zero.items[0] else {
        panic!("fn")
    };
    assert_eq!(fd.sig.value_params.len(), 0);
    let two =
        parse("nodule d;\nfn g(a: Binary{1}, b: Binary{1}) => Binary{1} = a;").expect("parses");
    let Item::Fn(fd) = &two.items[0] else {
        panic!("fn")
    };
    assert_eq!(fd.sig.value_params.len(), 2);
}

#[test]
fn match_arms_tolerate_a_trailing_comma() {
    // `comma_separated(Some(RBrace))` trailing-comma path — same arm count with or without it.
    let with = fn_body(
        "nodule d;\ntype B = F | T;\nfn m(x: B) => Binary{1} = match x { F => 0b0, T => 0b1, };",
    );
    let Expr::Match { arms, .. } = with else {
        panic!("match")
    };
    assert_eq!(arms.len(), 2);
    let without = fn_body(
        "nodule d;\ntype B = F | T;\nfn m(x: B) => Binary{1} = match x { F => 0b0, T => 0b1 };",
    );
    let Expr::Match { arms, .. } = without else {
        panic!("match")
    };
    assert_eq!(arms.len(), 2);
}

#[test]
fn empty_match_is_still_an_explicit_error() {
    // The non-empty invariant of match arms must survive the factoring: `match x { }` parses the
    // first arm and fails on the pattern — never silently an empty arm list.
    let err = parse("nodule d;\ntype B = F;\nfn m(x: B) => Binary{1} = match x { };")
        .expect_err("empty match must be rejected");
    assert!(err.message.contains("a pattern"), "{}", err.message);
}

#[test]
fn a_bare_list_rejects_a_trailing_comma() {
    // Constructor fields take no trailing comma (`comma_separated(None)`): a dangling `,` makes
    // the helper try to parse another field and fail explicitly — behavior unchanged by M-640.
    let err = parse("nodule d;\ntype T = C(Binary{1},)")
        .expect_err("trailing comma in ctor fields must be rejected");
    assert!(err.message.contains("expected a type"), "{}", err.message);
}

#[test]
fn keyword_opener_diagnostic_is_the_backtick_spelling() {
    // `expect_keyword` must reproduce the exact `` `let` `` (etc.) message of the old inline
    // form. A `let` body that is truncated right where a keyword opener is required surfaces it.
    let err = parse("nodule d;\nfn main() => Binary{1} = if 0b0 then 0b1 els 0b0")
        .expect_err("malformed if must be rejected");
    // `els` is an identifier where `else` is required.
    assert!(err.message.contains("`else`"), "{}", err.message);
}

// --- M-659 / RFC-0019 §4.1: `impl` decls + bounded type-params parse ---

#[test]
fn an_impl_decl_parses_with_trait_args_for_type_and_methods() {
    let n = parse(
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };",
    )
    .expect("an impl parses");
    let Item::Impl(id) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Impl(_)))
        .expect("an impl item")
    else {
        panic!("impl");
    };
    assert_eq!(id.trait_name, "Cmp");
    assert_eq!(id.trait_args.len(), 1); // `<Binary{8}>`
    assert_eq!(id.methods.len(), 1);
    assert_eq!(id.methods[0].sig.name, "cmp");
}

#[test]
fn an_impl_without_for_is_an_explicit_error() {
    let err = parse("nodule d;\nimpl Cmp[Binary{8}] Binary{8} { }")
        .expect_err("impl missing `for` must be rejected");
    assert!(err.message.contains("`for`"), "{}", err.message);
}

#[test]
fn a_bounded_fn_type_param_parses_with_a_self_bound_and_a_plus_list() {
    // `[T: Cmp]` (single self-bound) and `[T: A + B[T]]` (a `+`-list with type-args) both parse.
    let n = parse("nodule d;\nfn f[T: Cmp](x: T) => T = x;\nfn g[T: A + B[T]](x: T) => T = x;")
        .expect("bounded type-params parse");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.params.len(), 1);
    assert_eq!(f.sig.params[0].name, "T");
    assert_eq!(f.sig.params[0].bounds.len(), 1);
    assert_eq!(f.sig.params[0].bounds[0].name, "Cmp");
    let Item::Fn(g) = &n.items[1] else {
        panic!("fn")
    };
    assert_eq!(g.sig.params[0].bounds.len(), 2); // A + B[T]
    assert_eq!(g.sig.params[0].bounds[1].name, "B");
    assert_eq!(g.sig.params[0].bounds[1].args.len(), 1); // B[T]
}

#[test]
fn an_unbounded_fn_type_param_still_parses_the_identity_case() {
    // The §11 identity: `[A]` with no bound is `TypeParam { bounds: [] }` — every v0 program
    // that parsed before this extension still parses.
    let n = parse("nodule d;\nfn id[A](x: A) => A = x;").expect("unbounded parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.params.len(), 1);
    assert!(f.sig.params[0].bounds.is_empty());
}

#[test]
fn a_bound_on_a_type_decl_param_is_an_explicit_parse_refusal() {
    // Stage-1: bounds live only on fn type-params. A bound on a `type` param is rejected, never
    // silently dropped (G2). (Conformance reject/15 pins this at the corpus level too.)
    let err = parse("nodule d;\ntype Box[A: Cmp] = Wrap(A)")
        .expect_err("a bound on a type-decl param must be rejected");
    assert!(err.message.contains("deferred"), "{}", err.message);
}

// --- M-660 / RFC-0014 §3.4: effect annotations `!{ … }` on fn signatures parse ---

#[test]
fn an_effect_annotation_parses_into_the_signature_effect_set() {
    // `!{io, time}` after the return type lands as the signature's effect set, in source order.
    let n = parse("nodule d;\nfn a() => Binary{8} !{io, time} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.effects, vec!["io".to_owned(), "time".to_owned()]);
}

#[test]
fn an_unannotated_fn_has_an_empty_effect_set_and_an_explicit_empty_set_too() {
    // Unannotated ⇒ pure (empty set); the explicit written `!{}` is also the empty set — both
    // mean "declares no effects" (RFC-0014 I5).
    let plain = parse("nodule d;\nfn a() => Binary{8} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &plain.items[0] else {
        panic!("fn")
    };
    assert!(f.sig.effects.is_empty());
    let empty = parse("nodule d;\nfn a() => Binary{8} !{} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &empty.items[0] else {
        panic!("fn")
    };
    assert!(f.sig.effects.is_empty());
}

#[test]
fn a_trait_method_requirement_carries_an_effect_annotation() {
    // The effect annotation is part of the shared signature tail, so a trait method requirement
    // (no body) carries it too (the impl-vs-trait effect conformance check consumes it — M-660).
    let n = parse("nodule d;\ntrait T[A] { fn m(x: A) => A !{io}; };").expect("parses");
    let Item::Trait(td) = &n.items[0] else {
        panic!("trait")
    };
    assert_eq!(td.sigs[0].effects, vec!["io".to_owned()]);
}

#[test]
fn a_bare_bang_without_an_effect_brace_is_an_explicit_error() {
    // `!` only ever opens an effect set; a `!` not followed by `{` is a never-silent parse error
    // (v0 has no negation/`not` operator — logical ops are named prims; G2).
    let err = parse("nodule d;\nfn a() => Binary{8} ! = 0b00000000")
        .expect_err("a bare `!` must be rejected");
    assert!(err.message.contains("effect set"), "got: {}", err.message);
}

// --- M-661 / RFC-0016 §8-Q6: the `@std-sys` nodule-header FFI-floor marker parses ---

#[test]
fn the_std_sys_header_marker_sets_the_nodule_flag() {
    // `nodule <path> @std-sys` sets `Nodule.std_sys`; a plain `nodule <path>` leaves it false.
    // The marker is an attribute on the header, parsed after the path (M-661).
    let marked = parse("nodule std.sys.fs @std-sys;\nfn f() => Binary{1} = 0b0;").expect("parses");
    assert!(marked.std_sys, "the @std-sys marker must set std_sys");
    assert_eq!(marked.path.0, vec!["std", "sys", "fs"]);
    let plain = parse("nodule d;\nfn f() => Binary{1} = 0b0;").expect("parses");
    assert!(!plain.std_sys, "an unmarked nodule is not std-sys");
}

#[test]
fn a_std_sys_nodule_parses_a_wild_block_in_a_fn_body() {
    // The marker + a `wild` block parse together (the context gate + effect coverage are CHECKER
    // concerns, not parse concerns — this only pins that the surface admits both).
    let n = parse("nodule std.sys.x @std-sys;\nfn f() => Binary{8} !{ffi} = wild { host_call() };")
        .expect("a @std-sys nodule with a wild block parses");
    assert!(n.std_sys);
    let Item::Fn(fd) = &n.items[0] else {
        panic!("fn")
    };
    assert!(matches!(fd.body, Expr::Wild(_)), "the body is a wild block");
    assert_eq!(fd.sig.effects, vec!["ffi".to_owned()]);
}

// --- M-685 / RFC-0024 §3: function type `A => B` surface + fn-name-as-value ---

/// Helper: extract the `TypeRef` of the first named parameter of the first `fn` item.
fn first_param_ty(src: &str) -> TypeRef {
    let n = parse(src).expect("parses");
    n.items
        .into_iter()
        .find_map(|i| match i {
            Item::Fn(fd) => Some(fd.sig.value_params.into_iter().next()?.ty),
            _ => None,
        })
        .expect("a fn with at least one value parameter")
}

#[test]
fn simple_fn_type_parses_to_basetype_fn() {
    // `f: A => B` in a parameter builds `BaseType::Fn(Named("A"), Named("B"))`.
    // Use a single-param fn so `first_param_ty` finds the fn-typed one directly.
    let ty = first_param_ty("nodule d;\nfn apply[A, B](f: A => B) => B = f;");
    let BaseType::Fn(arg, ret) = ty.base else {
        panic!("expected BaseType::Fn, got {:?}", ty.base);
    };
    assert!(
        matches!(arg.base, BaseType::Named(ref n, _) if n == "A"),
        "arg should be Named(A), got {:?}",
        arg.base
    );
    assert!(
        matches!(ret.base, BaseType::Named(ref n, _) if n == "B"),
        "ret should be Named(B), got {:?}",
        ret.base
    );
    assert!(ty.guarantee.is_none(), "no guarantee on the outer fn type");
}

#[test]
fn fn_type_is_right_associative() {
    // `A => B => C` must parse as `A => (B => C)`.
    let ty = first_param_ty("nodule d;\nfn f[A, B, C](g: A => B => C) => A = g;");
    // Outer is `Fn(A, B => C)`.
    let BaseType::Fn(arg, ret) = ty.base else {
        panic!("expected outer BaseType::Fn");
    };
    assert!(matches!(arg.base, BaseType::Named(ref n, _) if n == "A"));
    // Inner `ret` must itself be `Fn(B, C)`.
    let BaseType::Fn(b, c) = ret.base else {
        panic!(
            "expected inner BaseType::Fn (right-assoc), got {:?}",
            ret.base
        );
    };
    assert!(matches!(b.base, BaseType::Named(ref n, _) if n == "B"));
    assert!(matches!(c.base, BaseType::Named(ref n, _) if n == "C"));
}

#[test]
fn guarantee_binds_tighter_than_arrow() {
    // `A @ Exact => B` must parse as `(A @ Exact) => B`.
    let ty = first_param_ty("nodule d;\nfn f[A, B](g: A @ Exact => B) => B = g;");
    let BaseType::Fn(arg, _ret) = ty.base else {
        panic!("expected BaseType::Fn");
    };
    // The LHS `(A @ Exact)` carries the Exact guarantee; the outer fn type has none.
    assert!(
        matches!(arg.guarantee, Some(crate::ast::Strength::Exact)),
        "arg should carry Exact guarantee, got {:?}",
        arg.guarantee
    );
    assert!(ty.guarantee.is_none(), "outer fn type has no guarantee");
}

#[test]
fn rfc_0024_map_snippet_parses() {
    // RFC-0024 §3's canonical snippet: `fn map[A, B, E](r: Result[A,E], f: A => B) => Result[B,E]`.
    // Structural check: two value params, second has type `BaseType::Fn`.
    let n = parse(
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };",
    )
    .expect("RFC-0024 §3 map snippet parses");
    let Item::Fn(fd) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(_)))
        .expect("fn")
    else {
        panic!("fn");
    };
    assert_eq!(fd.sig.name, "map");
    assert_eq!(fd.sig.value_params.len(), 2);
    let f_ty = &fd.sig.value_params[1].ty;
    assert!(
        matches!(f_ty.base, BaseType::Fn(_, _)),
        "second param `f` should have a function type, got {:?}",
        f_ty.base
    );
}

#[test]
fn bare_fn_name_in_value_position_parses_as_path() {
    // `map(mk_ok(), double)` — `double` in value (non-call) position is `Expr::Path`, not
    // `Expr::App`.  This confirms fn-as-value needs no parser change (RFC-0024 §3).
    let n = parse(
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn double[A](x: A) => A = x;\nfn mk_ok[A](x: A) => Result[A, A] = Ok(x);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn main() => Result[Binary{8}, Binary{8}] = map(mk_ok(0b00000000), double);",
    )
    .expect("parses");
    // Find the `main` fn and inspect its body.
    let Item::Fn(main_fd) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(fd) if fd.sig.name == "main"))
        .expect("main fn")
    else {
        panic!("main fn");
    };
    // Body is `map(mk_ok(0b00000000), double)` → `App { head: Path([map]), args: [App(mk_ok), Path([double])] }`.
    let Expr::App { ref head, ref args } = main_fd.body else {
        panic!("expected App, got {:?}", main_fd.body);
    };
    assert!(matches!(head.as_ref(), Expr::Path(p) if p.0 == vec!["map"]));
    assert_eq!(args.len(), 2);
    // First arg: `mk_ok(0b00000000)` — an App.
    assert!(
        matches!(args[0], Expr::App { .. }),
        "first arg is App (call)"
    );
    // Second arg: `double` — a bare Path (fn-as-value, no call parens).
    assert!(
        matches!(args[1], Expr::Path(ref p) if p.0 == vec!["double"]),
        "second arg `double` should be a bare Path, got {:?}",
        args[1]
    );
}

#[test]
fn malformed_arrow_missing_rhs_is_explicit_error() {
    // `A =>` with no right-hand type must be an explicit `ParseError` — never silently accepted
    // (G2 / house rule #2: never-silent).
    let err = parse("nodule d;\nfn f[A](g: A =>) => A = g")
        .expect_err("a bare `A =>` with no rhs must be rejected");
    // The error should describe what was missing — a type is expected after `=>`.
    assert!(
        err.message.contains("type") || err.message.contains("expected"),
        "error message should mention a missing type: {:?}",
        err.message
    );
}

#[test]
fn fn_type_in_return_position_parses() {
    // A function may return a function type: `fn make_fn[A, B]() => A => B = ...`
    // The `=>` in the return type is also right-associative and fully parsed.
    let n = parse("nodule d;\nfn make_fn[A, B](x: A) => A => B = x;").expect("parses");
    let Item::Fn(fd) = &n.items[0] else {
        panic!("fn")
    };
    assert!(
        matches!(fd.sig.ret.base, BaseType::Fn(_, _)),
        "return type should be BaseType::Fn, got {:?}",
        fd.sig.ret.base
    );
}

/// DN-57: an optional `;` **terminates a component** (top-level item / trait + impl method). A
/// program written with `;` terminators parses to the **same AST** as the newline-delimited form —
/// the `;` is a never-required terminator that adds no AST node, so it enables whitespace-independent
/// / streamable source without changing meaning. (`,` separates siblings; `;` terminates a component.)
#[test]
fn dn57_optional_semicolon_terminates_components_ast_transparent() {
    // Item terminators: two fns on one line, separated only by `;` (no newline) == the newline form.
    let semi = "nodule d;\nfn a() => Binary{8} = 0b0; fn b(x: Binary{8}) => Binary{8} = x;";
    let plain = "nodule d;\nfn a() => Binary{8} = 0b0;\nfn b(x: Binary{8}) => Binary{8} = x;";
    assert_eq!(
        parse(semi).expect("`;`-terminated items parse"),
        parse(plain).expect("newline-delimited items parse"),
        "the optional `;` adds no AST node — same tree either way",
    );
    // A trailing `;` on the final item is accepted (it terminates, it does not separate).
    assert!(parse("nodule d;\nfn a() => Binary{8} = 0b0;").is_ok());
    // Method terminators inside trait + impl bodies parse.
    let methods = "nodule d;\ntrait T { fn f(x: Binary{8}) => Binary{8}; };\nimpl T for Binary{8} { fn f(x: Binary{8}) => Binary{8} = x; };";
    assert!(
        parse(methods).is_ok(),
        "`;`-terminated trait/impl methods parse"
    );
}

// ---- DN-54 / M-812: lower / derive parse -------------------------------------------------------

/// A zero-param `lower` rule parses to `Item::Lower` with the correct name and empty params.
#[test]
fn lower_no_params_parses() {
    let n = parse("nodule d;\nlower Trivial = true;").expect("parses");
    let item = &n.items[0];
    let crate::ast::Item::Lower(ld) = item else {
        panic!("expected Item::Lower, got {item:?}");
    };
    assert_eq!(ld.name, "Trivial");
    assert!(ld.params.is_empty());
}

/// A parametric `lower` rule with two params parses correctly.
#[test]
fn lower_with_params_parses() {
    let n = parse("nodule d;\nlower Pair[A, B] = true;").expect("parses");
    let crate::ast::Item::Lower(ld) = &n.items[0] else {
        panic!("expected Item::Lower");
    };
    assert_eq!(ld.name, "Pair");
    assert_eq!(ld.params, vec!["A".to_owned(), "B".to_owned()]);
}

/// A `derive` application parses to `Item::Derive` with the correct name and target type.
#[test]
fn derive_decl_parses() {
    let n = parse("nodule d;\nderive Trivial for Binary{8};").expect("parses");
    let crate::ast::Item::Derive(dd) = &n.items[0] else {
        panic!("expected Item::Derive");
    };
    assert_eq!(dd.name, "Trivial");
    // `for_ty` base must be Binary{8}
    let crate::ast::BaseType::Binary(w) = &dd.for_ty.base else {
        panic!("expected BaseType::Binary, got {:?}", dd.for_ty.base);
    };
    // Width 8 (the width ref)
    let crate::ast::WidthRef::Lit(8) = w else {
        panic!("expected width 8, got {w:?}");
    };
}

/// `lower` without `=` is an explicit parse error (G2 / DN-54 §3).
#[test]
fn lower_missing_eq_is_refused() {
    let err = parse("nodule d;\nlower Trivial true").unwrap_err();
    assert!(
        err.message.contains("="),
        "expected missing-`=` error, got: {}",
        err.message
    );
}

/// `derive` without `for` is an explicit parse error (G2 / DN-54 §4).
#[test]
fn derive_missing_for_is_refused() {
    let err = parse("nodule d;\nderive Trivial Binary{8}").unwrap_err();
    assert!(
        err.message.contains("for"),
        "expected missing-`for` error, got: {}",
        err.message
    );
}

/// `lower` or `derive` at expression position is an explicit parse error (top-level only, G2).
#[test]
fn lower_at_expr_position_is_refused() {
    let err = parse("nodule d;\nfn f() => Binary{8} = lower Trivial = true").unwrap_err();
    assert!(
        err.message.contains("top-level"),
        "`lower` at expr position must say it's a top-level declaration, got: {}",
        err.message
    );
}

#[test]
fn derive_at_expr_position_is_refused() {
    let err = parse("nodule d;\nfn f() => Binary{8} = derive Trivial for Binary{8}").unwrap_err();
    assert!(
        err.message.contains("top-level"),
        "`derive` at expr position must say it's a top-level declaration, got: {}",
        err.message
    );
}

/// `grow` at item position produces a teaching diagnostic referencing `derive` (DN-38 §8.1).
#[test]
fn grow_at_item_position_points_to_derive() {
    let err = parse("nodule d;\ngrow something").unwrap_err();
    assert!(
        err.message.contains("derive"),
        "`grow` at item position must mention `derive` in the teaching diagnostic, got: {}",
        err.message
    );
    assert!(
        err.message.contains("DN-38"),
        "`grow` diagnostic must cite DN-38, got: {}",
        err.message
    );
}

// --- M-677 / RFC-0014 §3.4 I4: per-effect budget annotations `!{eff(<=N<unit>?)}` parse ---

#[test]
fn effect_with_integer_budget_parses_into_effect_budgets() {
    // `!{retry(<=3)}` — retry bound of 3 units. The effects vec gets the effect name, and
    // effect_budgets maps it to the ceiling (M-677, RFC-0014 §4.5 I4).
    let n = parse("nodule d;\nfn f() => Binary{8} !{retry(<=3)} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.effects, vec!["retry".to_owned()]);
    assert_eq!(f.sig.effect_budgets.get("retry").copied(), Some(3));
}

#[test]
fn effect_with_kib_budget_parses_into_effect_budgets() {
    // `!{alloc(<=64KiB)}` — alloc bound of 64 × 1024 = 65536 bytes (M-677, RFC-0014 §4.5 I4).
    let n =
        parse("nodule d;\nfn f() => Binary{8} !{alloc(<=64KiB)} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.effects, vec!["alloc".to_owned()]);
    assert_eq!(f.sig.effect_budgets.get("alloc").copied(), Some(65536));
}

#[test]
fn effect_with_mib_budget_parses_into_effect_budgets() {
    // `!{alloc(<=1MiB)}` — alloc bound of 1 × 1048576 bytes.
    let n = parse("nodule d;\nfn f() => Binary{8} !{alloc(<=1MiB)} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.effect_budgets.get("alloc").copied(), Some(1_048_576));
}

#[test]
fn mixed_budgeted_and_unbounded_effects_parse_correctly() {
    // `!{retry(<=3), io}` — one budgeted, one unbounded. The effects vec lists both;
    // effect_budgets only maps the budgeted one.
    let n =
        parse("nodule d;\nfn f() => Binary{8} !{retry(<=3), io} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.effects, vec!["retry".to_owned(), "io".to_owned()]);
    assert_eq!(f.sig.effect_budgets.get("retry").copied(), Some(3));
    assert!(
        !f.sig.effect_budgets.contains_key("io"),
        "unbounded `io` must have no budget entry"
    );
}

#[test]
fn effect_budget_zero_is_refused() {
    // A ceiling of zero is a never-silent error (G2): a zero budget would exhaust before any work.
    let err = parse("nodule d;\nfn f() => Binary{8} !{retry(<=0)} = 0b00000000;")
        .expect_err("zero budget must be refused");
    assert!(
        err.message.contains("zero"),
        "expected zero-budget error, got: {}",
        err.message
    );
}

#[test]
fn effect_budget_duplicate_effect_is_refused() {
    // A duplicate effect name in the set is a never-silent error (G2).
    let err = parse("nodule d;\nfn f() => Binary{8} !{retry, retry} = 0b00000000;")
        .expect_err("duplicate effect must be refused");
    assert!(
        err.message.contains("duplicate") || err.message.contains("retry"),
        "expected duplicate-effect error, got: {}",
        err.message
    );
}

#[test]
fn effect_annotation_without_budget_still_parses() {
    // An unbounded `!{io}` still has an empty `effect_budgets` map — backward compat (M-660).
    let n = parse("nodule d;\nfn f() => Binary{8} !{io} = 0b00000000;").expect("parses");
    let Item::Fn(f) = &n.items[0] else {
        panic!("fn")
    };
    assert_eq!(f.sig.effects, vec!["io".to_owned()]);
    assert!(
        f.sig.effect_budgets.is_empty(),
        "unbounded effect must have an empty effect_budgets map"
    );
}

// ---- RFC-0020 §9 / R20-Q3: or-patterns (`A | B => body`) parse --------------------------------

/// A two-alternative or-pattern `A | B => e` parses to a single arm whose pattern is
/// `Pattern::Or([A, B])`. The or-pattern is pure surface sugar: the checker desugars it (KC-3).
#[test]
fn or_pattern_two_alts_parses_to_pattern_or() {
    use crate::ast::{Arm, Pattern};
    // A match arm `Zero | One => body` — both alts are bare idents (nullary-ctor or binder; the
    // checker distinguishes them; the parser sees an identifier pattern in either case).
    let n = parse(
        "nodule d;\ntype Bit = Zero | One;\nfn classify(b: Bit) => Binary{1} = match b { Zero | One => 0b1 };",
    )
    .expect("or-pattern parses");
    let Item::Fn(fd) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(_)))
        .expect("fn")
    else {
        panic!("fn");
    };
    let Expr::Match { arms, .. } = &fd.body else {
        panic!("expected match, got {:?}", fd.body);
    };
    assert_eq!(
        arms.len(),
        1,
        "one surface arm (or-pattern, not expanded yet)"
    );
    let Arm {
        pattern: Pattern::Or(alts),
        ..
    } = &arms[0]
    else {
        panic!("expected Pattern::Or, got {:?}", arms[0].pattern);
    };
    assert_eq!(alts.len(), 2);
    assert!(
        matches!(&alts[0], Pattern::Ident(n) if n == "Zero"),
        "first alt"
    );
    assert!(
        matches!(&alts[1], Pattern::Ident(n) if n == "One"),
        "second alt"
    );
}

/// Three-alternative or-pattern `A | B | C => e` parses to `Pattern::Or([A, B, C])`.
#[test]
fn or_pattern_three_alts_parses_correctly() {
    use crate::ast::Pattern;
    let n = parse(
        "nodule d;\ntype Sign = Neg | Zero | Pos;\nfn any_sign(s: Sign) => Binary{1} = match s { Neg | Zero | Pos => 0b1 };",
    )
    .expect("three-alt or-pattern parses");
    let Item::Fn(fd) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(_)))
        .expect("fn")
    else {
        panic!("fn");
    };
    let Expr::Match { arms, .. } = &fd.body else {
        panic!("expected match");
    };
    assert_eq!(arms.len(), 1);
    let Pattern::Or(alts) = &arms[0].pattern else {
        panic!("expected Pattern::Or");
    };
    assert_eq!(alts.len(), 3);
}

/// A match with an or-pattern arm AND a plain arm both parse correctly.
#[test]
fn or_pattern_mixed_with_plain_arm_parses() {
    use crate::ast::Pattern;
    let n = parse(
        "nodule d;\ntype Sign = Neg | Zero | Pos;\nfn is_nonzero(s: Sign) => Binary{1} = match s { Neg | Pos => 0b1, Zero => 0b0 };",
    )
    .expect("mixed or-pattern + plain arm parses");
    let Item::Fn(fd) = n
        .items
        .iter()
        .find(|i| matches!(i, Item::Fn(_)))
        .expect("fn")
    else {
        panic!("fn");
    };
    let Expr::Match { arms, .. } = &fd.body else {
        panic!("expected match");
    };
    assert_eq!(arms.len(), 2, "two surface arms");
    // First arm has an or-pattern.
    assert!(
        matches!(&arms[0].pattern, Pattern::Or(_)),
        "first arm is or-pattern"
    );
    // Second arm is a plain ident pattern.
    assert!(
        matches!(&arms[1].pattern, Pattern::Ident(_)),
        "second arm is plain ident"
    );
}

/// `a | b` in expression position is NOT parsed as an or-pattern — it desugars to `bor(a, b)`
/// (infix bitwise-or). The disambiguation is by grammar position: `|` after a pattern before `=>`
/// is an or-separator; `|` after an expression is bitwise-or. The two must never mix silently (G2).
#[test]
fn pipe_in_expr_position_is_bitwise_or_not_or_pattern() {
    // `fn f(x: Binary{8}) => Binary{8} = x | 0b0000_0001` — `|` is bitwise-or here.
    let n = parse("nodule d;\nfn f(x: Binary{8}) => Binary{8} = x | 0b0000_0001;")
        .expect("bitwise-or in expr position parses");
    let Item::Fn(fd) = &n.items[0] else {
        panic!("fn");
    };
    // Body must be App(bor, [x, 0b0000_0001]) — desugared by the parser, not a Pattern::Or.
    assert!(
        matches!(&fd.body, Expr::App { head, .. } if matches!(head.as_ref(), Expr::Path(p) if p.0 == vec!["bor"])),
        "pipe in expression position desugars to bor(), not an or-pattern; got {:?}",
        fd.body
    );
}

// -------------------------------------------------------------------------
// RFC-0037 D2-b: short repr-keyword aliases (M-915) — `bin`/`tern`/`emb`/`hvec`
// -------------------------------------------------------------------------

/// Parse a `derive Trivial for <src>;` item and return the resolved `for_ty` base — a compact way
/// to drive [`crate::parse::Parser::parse_base_type`] (private) through the public [`parse`] entry.
fn base_type_of(src: &str) -> BaseType {
    let n = parse(&format!("nodule d;\nderive Trivial for {src};")).expect("parses");
    let Item::Derive(dd) = &n.items[0] else {
        panic!("expected Item::Derive");
    };
    dd.for_ty.base.clone()
}

/// `bin{N}` parses to the exact same `BaseType::Binary` as `Binary{N}` — the short form
/// **elaborates identically** (D2-b): no new AST node distinguishes the spelling used.
#[test]
fn bin_short_alias_elaborates_identically_to_binary() {
    assert_eq!(base_type_of("bin{8}"), base_type_of("Binary{8}"));
    let BaseType::Binary(crate::ast::WidthRef::Lit(8)) = base_type_of("bin{8}") else {
        panic!("expected BaseType::Binary(Lit(8))");
    };
}

/// `tern{N}` parses identically to `Ternary{N}`.
#[test]
fn tern_short_alias_elaborates_identically_to_ternary() {
    assert_eq!(base_type_of("tern{6}"), base_type_of("Ternary{6}"));
    let BaseType::Ternary(crate::ast::WidthRef::Lit(6)) = base_type_of("tern{6}") else {
        panic!("expected BaseType::Ternary(Lit(6))");
    };
}

/// `emb{D,S}` parses identically to `Dense{D,S}`.
#[test]
fn emb_short_alias_elaborates_identically_to_dense() {
    assert_eq!(
        base_type_of("emb{768, F32}"),
        base_type_of("Dense{768, F32}")
    );
    let BaseType::Dense(768, crate::ast::Scalar::F32) = base_type_of("emb{768, F32}") else {
        panic!("expected BaseType::Dense(768, F32)");
    };
}

/// `hvec{model,dim,sparsity}` parses identically to `VSA{model,dim,sparsity}`.
#[test]
fn hvec_short_alias_elaborates_identically_to_vsa() {
    assert_eq!(
        base_type_of("hvec{MAP, 10000, Dense}"),
        base_type_of("VSA{MAP, 10000, Dense}")
    );
    let BaseType::Vsa {
        model,
        dim,
        sparsity,
    } = base_type_of("hvec{MAP, 10000, Sparse{4}}")
    else {
        panic!("expected BaseType::Vsa");
    };
    assert_eq!(model, "MAP");
    assert_eq!(dim, 10000);
    assert_eq!(sparsity, crate::ast::Sparsity::Sparse(4));
}

/// The width-parameter form (`bin{N}` with `N` a name, not a literal — DN-42/M-753 v1) works for
/// the short alias exactly as it does for the long form.
#[test]
fn bin_short_alias_supports_a_width_parameter_name() {
    let BaseType::Binary(crate::ast::WidthRef::Name(n)) = base_type_of("bin{N}") else {
        panic!("expected BaseType::Binary(Name(\"N\"))");
    };
    assert_eq!(n, "N");
}

/// A short-alias repr type composes normally in a full fn signature, mixed with long-form types.
#[test]
fn short_repr_keywords_parse_in_a_fn_signature() {
    let n = parse("nodule d;\nfn widen(x: bin{8}) => Binary{8} = x;").expect("parses");
    let Item::Fn(fd) = &n.items[0] else {
        panic!("expected Item::Fn");
    };
    assert_eq!(fd.sig.value_params.len(), 1);
    let BaseType::Binary(crate::ast::WidthRef::Lit(8)) = &fd.sig.value_params[0].ty.base else {
        panic!("expected the param type to be Binary{{8}} (via `bin`)");
    };
}

/// `vec` was explicitly REJECTED as a short alias (DN-02/RFC-0037 D2-b) — it is not a keyword, so
/// `vec{4, F32}` in type position is a never-silent parse error (never a silent accept as a repr
/// type), not a rewrite (G2).
#[test]
fn vec_is_never_accepted_as_a_repr_keyword() {
    let err = parse("nodule d;\nfn f(x: vec{4, F32}) => Binary{8} = 0b0;")
        .expect_err("`vec{...}` must not parse as a repr type");
    // `vec` lexes as a plain identifier (`BaseType::Named`), so the parser expects a param-list
    // delimiter after it, not `{` — the explicit diagnostic differs from the repr-keyword arms
    // but is still a real, explained parse error (G2), never a silent accept.
    assert!(
        err.message.contains("`)` to close the parameter list"),
        "expected the param-list diagnostic, got: {}",
        err.message
    );
}

/// A malformed short-form repr type (`emb` with no `{…}` descriptor at all) is a never-silent
/// parse error whose message names both spellings, mirroring the long form's diagnostic.
#[test]
fn malformed_emb_short_form_is_an_explicit_parse_error() {
    let err = parse("nodule d;\nfn f() => emb = 0;").expect_err("a bare `emb` must error");
    assert!(
        err.message.contains("Dense") && err.message.contains("emb"),
        "expected the diagnostic to name both `Dense` and `emb`, got: {}",
        err.message
    );
}

/// The `emb{…}` short form still requires the `,` between dim and dtype, exactly like `Dense{…}`
/// (same underlying arm) — a never-silent parse error either way.
#[test]
fn emb_short_form_still_requires_the_comma() {
    let err = parse("nodule d;\nfn f() => emb{4 F32} = 0;")
        .expect_err("missing comma inside emb{...} must error");
    assert!(
        err.message.contains("dim and dtype"),
        "expected the missing-comma diagnostic, got: {}",
        err.message
    );
}
