//! M-740 Stage 4 (DN-26 §7.3 row 4) — the self-hosted `compiler.totality` port.
//!
//! `lib/compiler/totality.myc`'s `classify_all` (RFC-0007 §4.5 structural totality — self- and
//! mutual-recursion descent) vs the live Rust oracle (`mycelium_l1::totality::classify_all`,
//! `crates/mycelium-l1/src/totality.rs`): a **unit differential** over small synthetic `FnDecl`
//! tables, not a corpus sweep — unlike Stage 2/3, this pass's INPUT is a checker-internal data
//! structure (`BTreeMap<String, FnDecl>`), not source text, so there is no `.myc` conformance
//! corpus to sweep. Per the task brief: feed small synthetic `FnDecl` sets exercising each of
//! `classify_all`'s branches (non-recursive, self-descent, self-non-descent, mutual-descent,
//! mutual-non-descent), args-in/verdict-out, one eval per name checked.
//!
//! Each scenario is built TWICE, independently, from the same logical shape: once as real
//! `mycelium_l1::ast` Rust values (the oracle side, `classify_all`'d directly — no parsing
//! involved), and once as literal `.myc` AST-constructor calls embedded in the driver source (the
//! self-hosted side, evaluated through `lib/compiler/totality.myc`'s own `classify_all`). The
//! ORACLE computes the expected classification at test time (never a hand-assumed expectation —
//! the differential principle every prior stage's tests hold to); only the INPUT shape (which
//! functions call which, with what patterns) is a fixed fixture this leaf designed.
//!
//! Honest narrowings carried by `totality.myc` itself (full detail in-file, FLAG-totality-1..5):
//! `BTreeMap`/`BTreeSet` -> sorted `Vec[Pair[Bytes, V]]` assoc-lists / `Vec[Bytes]` key lists, with
//! an unenforced (by the type) sorted-by-name precondition on `classify_all`'s own `fns` parameter
//! (FLAG-totality-1); the shared `walk_expr` HOF becomes two independent specialized traversals
//! (FLAG-totality-2); no `mycelium_stack::with_deep_stack` analogue, relying entirely on the
//! threaded `depth: Binary{32}` budget (FLAG-totality-3); `Pattern::Or` reaching `pattern_binders`
//! (a Rust `panic!` invariant violation) becomes a documented dead `Ok(acc)` fallback
//! (FLAG-totality-4); SCC member order is not preserved and `combos` is computed via an early-exit
//! product (FLAG-totality-5) — both proven order/result-irrelevant in-file.

use std::collections::BTreeMap;

use mycelium_l1::ast::{Arm, BaseType, Expr, FnDecl, FnSig, Param, Path, Pattern, TypeRef, Vis};
use mycelium_l1::totality::{classify_all, Totality};
use mycelium_l1::{check_nodule, monomorphize, parse, Evaluator};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The Rust-side (oracle) fixture builders — plain `mycelium_l1::ast` value construction, no
// parsing. Types never matter to totality (it never inspects them); every param/return type is
// the trivial `Bytes` (`BaseType::Bytes`, unguaranteed).
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn ty_bytes() -> TypeRef {
    TypeRef::unguaranteed(BaseType::Bytes)
}

fn oracle_param(name: &str) -> Param {
    Param {
        name: name.to_string(),
        ty: ty_bytes(),
    }
}

fn oracle_sig(name: &str, params: &[&str]) -> FnSig {
    FnSig {
        name: name.to_string(),
        params: vec![],
        value_params: params.iter().map(|p| oracle_param(p)).collect(),
        ret: ty_bytes(),
        effects: vec![],
        effect_budgets: BTreeMap::new(),
    }
}

fn oracle_fn_decl(name: &str, params: &[&str], body: Expr) -> FnDecl {
    FnDecl {
        vis: Vis::Private,
        thaw: false,
        tier: None,
        sig: oracle_sig(name, params),
        body,
    }
}

fn oracle_path(name: &str) -> Expr {
    Expr::Path(Path(vec![name.to_string()]))
}

fn oracle_call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::App {
        head: Box::new(oracle_path(name)),
        args,
    }
}

fn oracle_arm(pattern: Pattern, body: Expr) -> Arm {
    Arm { pattern, body }
}

fn oracle_match(scrutinee: Expr, arms: Vec<Arm>) -> Expr {
    Expr::Match {
        scrutinee: Box::new(scrutinee),
        arms,
    }
}

fn oracle_ctor_pat(name: &str, subs: Vec<Pattern>) -> Pattern {
    Pattern::Ctor(name.to_string(), subs)
}

fn oracle_ident_pat(name: &str) -> Pattern {
    Pattern::Ident(name.to_string())
}

/// `fn <name>(n) = match n { Z => n, S(m) => <call> }` — the shared shape every self/mutual
/// recursion scenario below uses; `call` is either a self-call or a call to a sibling.
fn peano_match_fn(name: &str, call: Expr) -> FnDecl {
    let body = oracle_match(
        oracle_path("n"),
        vec![
            oracle_arm(oracle_ctor_pat("Z", vec![]), oracle_path("n")),
            oracle_arm(oracle_ctor_pat("S", vec![oracle_ident_pat("m")]), call),
        ],
    );
    oracle_fn_decl(name, &["n"], body)
}

// Scenario 1 — no recursion at all: `fn ident(x) = x`.
fn scenario_non_recursive() -> BTreeMap<String, FnDecl> {
    let mut m = BTreeMap::new();
    m.insert(
        "ident".to_string(),
        oracle_fn_decl("ident", &["x"], oracle_path("x")),
    );
    m
}

// Scenario 2 — self-recursion with a structural descent: `S(m) => countdown(m)`.
fn scenario_self_descent() -> BTreeMap<String, FnDecl> {
    let mut m = BTreeMap::new();
    m.insert(
        "countdown".to_string(),
        peano_match_fn(
            "countdown",
            oracle_call("countdown", vec![oracle_path("m")]),
        ),
    );
    m
}

// Scenario 3 — self-recursion WITHOUT a structural descent: `S(m) => bad(n)` (passes `n`, not the
// smaller `m`).
fn scenario_self_non_descent() -> BTreeMap<String, FnDecl> {
    let mut m = BTreeMap::new();
    m.insert(
        "bad".to_string(),
        peano_match_fn("bad", oracle_call("bad", vec![oracle_path("n")])),
    );
    m
}

// Scenario 4 — mutual recursion WITH a structural descent: `even`/`odd` each call the other on
// the smaller `m`.
fn scenario_mutual_descent() -> BTreeMap<String, FnDecl> {
    let mut m = BTreeMap::new();
    m.insert(
        "even".to_string(),
        peano_match_fn("even", oracle_call("odd", vec![oracle_path("m")])),
    );
    m.insert(
        "odd".to_string(),
        peano_match_fn("odd", oracle_call("even", vec![oracle_path("m")])),
    );
    m
}

// Scenario 5 — mutual recursion WITHOUT a structural descent: `evenb`/`oddb` each call the other
// on `n` (not `m`).
fn scenario_mutual_non_descent() -> BTreeMap<String, FnDecl> {
    let mut m = BTreeMap::new();
    m.insert(
        "evenb".to_string(),
        peano_match_fn("evenb", oracle_call("oddb", vec![oracle_path("n")])),
    );
    m.insert(
        "oddb".to_string(),
        peano_match_fn("oddb", oracle_call("evenb", vec![oracle_path("n")])),
    );
    m
}

fn oracle_totality(fns: &BTreeMap<String, FnDecl>, name: &str) -> Totality {
    let result = classify_all(fns).unwrap_or_else(|e| {
        panic!("oracle classify_all refused on a small synthetic fixture: {e}")
    });
    *result
        .get(name)
        .unwrap_or_else(|| panic!("oracle classify_all result is missing {name:?}"))
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The `.myc`-side (self-hosted) fixture builders — literal `totality.myc` AST-constructor source
// text, mirroring the oracle builders above field-for-field (FD/FS/Prm/TR/KwBytes/Path/Pth/Match/
// Ar/PCtor/PIdent/App/Cons/Nil/Pr — totality.myc's own copied-in ast.myc constructor spellings).
// ─────────────────────────────────────────────────────────────────────────────────────────────

fn b32(n: u32) -> String {
    format!("0b{n:032b}")
}

fn myc_bytes_list(items: &[String]) -> String {
    let mut s = "Nil : Vec[Bytes]".to_string();
    for item in items.iter().rev() {
        s = format!(r#"Cons("{item}", {s})"#);
    }
    s
}

fn myc_path(name: &str) -> String {
    format!("Path(Pth({}))", myc_bytes_list(&[name.to_string()]))
}

fn myc_expr_list(items: &[String]) -> String {
    let mut s = "Nil : Vec[Expr]".to_string();
    for item in items.iter().rev() {
        s = format!("Cons({item}, {s})");
    }
    s
}

fn myc_call(name: &str, args: &[String]) -> String {
    format!("App({}, {})", myc_path(name), myc_expr_list(args))
}

fn myc_pattern_list(items: &[String]) -> String {
    let mut s = "Nil : Vec[Pattern]".to_string();
    for item in items.iter().rev() {
        s = format!("Cons({item}, {s})");
    }
    s
}

fn myc_ctor_pat(name: &str, subs: &[String]) -> String {
    format!(r#"PCtor("{name}", {})"#, myc_pattern_list(subs))
}

fn myc_ident_pat(name: &str) -> String {
    format!(r#"PIdent("{name}")"#)
}

fn myc_arm_list(items: &[String]) -> String {
    let mut s = "Nil : Vec[Arm]".to_string();
    for item in items.iter().rev() {
        s = format!("Cons({item}, {s})");
    }
    s
}

fn myc_match(scrutinee: &str, arms: &[String]) -> String {
    format!("Match({scrutinee}, {})", myc_arm_list(arms))
}

fn myc_param_list(names: &[&str]) -> String {
    let mut s = "Nil : Vec[Param]".to_string();
    for n in names.iter().rev() {
        s = format!(r#"Cons(Prm("{n}", TR(KwBytes, None)), {s})"#);
    }
    s
}

fn myc_fn_decl(name: &str, params: &[&str], body: &str) -> String {
    format!(
        r#"FD(Private, False, None, FS("{name}", Nil : Vec[TypeParam], {params}, TR(KwBytes, None), Nil : Vec[Bytes], Nil : Vec[EffectBudget]), {body})"#,
        params = myc_param_list(params)
    )
}

/// The `.myc` mirror of `peano_match_fn`: `FD(..., Match(Path(n), [Ar(PCtor("Z",[]), Path(n)),
/// Ar(PCtor("S",[PIdent("m")]), call)]))`.
fn myc_peano_match_fn(name: &str, call: &str) -> String {
    let body = myc_match(
        &myc_path("n"),
        &[
            format!("Ar({}, {})", myc_ctor_pat("Z", &[]), myc_path("n")),
            format!("Ar({}, {})", myc_ctor_pat("S", &[myc_ident_pat("m")]), call),
        ],
    );
    myc_fn_decl(name, &["n"], &body)
}

/// A `Vec[Pair[Bytes, FnDecl]]` literal — the caller passes `entries` already sorted ascending by
/// name (FLAG-totality-1's precondition).
fn myc_fn_table(entries: &[(&str, String)]) -> String {
    let mut s = "Nil : Vec[Pair[Bytes, FnDecl]]".to_string();
    for (name, decl) in entries.iter().rev() {
        s = format!(r#"Cons(Pr("{name}", {decl}), {s})"#);
    }
    s
}

const TOTALITY_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../lib/compiler/totality.myc"
));

/// The shared driver prelude: `totality_to_code`/`verdict_for_name` — TEST-ONLY glue kept out of
/// `totality.myc` itself (it is not part of the port, just this differential's harness), mirroring
/// `compiler_stage2.rs::driver_prelude`'s separation.
fn driver_prelude() -> String {
    format!(
        "fn totality_to_code(t: Totality) => Binary{{32}} =\n\
         \x20 match t {{ Total => {one}, Partial => {zero} }};\n\
         fn verdict_for_name(fns: Vec[Pair[Bytes, FnDecl]], name: Bytes, want: Binary{{32}}) => Binary{{32}} =\n\
         \x20 match classify_all(fns) {{\n\
         \x20   Err(_) => {depth_err},\n\
         \x20   Ok(result) => match alist_get(result, name) {{\n\
         \x20     None => {missing},\n\
         \x20     Some(t) => match eq(totality_to_code(t), want) {{ 0b1 => {one}, _ => {zero} }}\n\
         \x20   }}\n\
         \x20 }};\n",
        zero = b32(0),
        one = b32(1),
        depth_err = b32(2),
        missing = b32(3),
    )
}

fn program(entries: &str) -> String {
    format!("{TOTALITY_SRC}\n{}\n{entries}", driver_prelude())
}

/// One L1-eval verdict check (the `run_verdict`/`assert_l1_only_u32` convention every prior stage
/// uses): `entry_name` is a no-arg `fn ... => Binary{32}` already baked with its own `fns`/`name`/
/// `want`; the returned code must be `1` (full agreement) — `0` = classification mismatch, `2` =
/// unexpected `WalkDepthExceeded`, `3` = the name is missing from `classify_all`'s result.
fn assert_verdict(env: &mycelium_l1::Env, label: &str, entry_name: &str) {
    let mono = monomorphize(env, entry_name)
        .unwrap_or_else(|e| panic!("{label}: monomorphize failed: {e}"));
    let l1_val = Evaluator::new(&mono)
        .call(entry_name, vec![])
        .unwrap_or_else(|e| panic!("{label}: L1-eval failed: {e}"));
    let repr = l1_val
        .as_repr()
        .unwrap_or_else(|| panic!("{label}: expected a Repr result"));
    let got = match repr.payload() {
        mycelium_core::Payload::Bits(bits) => {
            bits.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b))
        }
        other => panic!("{label}: expected a Bits payload, got {other:?}"),
    };
    assert_eq!(
        got, 1,
        "{label}: Stage-4 totality differential verdict {got} \
         (0 = classification mismatch vs the Rust oracle; 2 = unexpected WalkDepthExceeded; \
          3 = name missing from classify_all's result)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The structural gate: `totality.myc` parses and type-checks green (no driver needed).
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn totality_myc_parses_and_checks() {
    let nodule = parse(TOTALITY_SRC).unwrap_or_else(|e| panic!("totality.myc: parse failed: {e}"));
    check_nodule(&nodule).unwrap_or_else(|e| panic!("totality.myc: check failed: {e}"));
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The Stage-4 gate: the unit differential over five small synthetic FnDecl tables, one `Evaluator`
// per scenario built from ONE shared `check_nodule` (the `run_verdict`-style check-once/call-many
// economy).
// ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn totality_myc_matches_oracle_non_recursive() {
    let want = oracle_totality(&scenario_non_recursive(), "ident");
    assert_eq!(
        want,
        Totality::Total,
        "oracle sanity: a non-recursive fn is Total"
    );
    let fns = myc_fn_table(&[("ident", myc_fn_decl("ident", &["x"], &myc_path("x")))]);
    let entries = format!(
        "fn verdict_ident() => Binary{{32}} = verdict_for_name({fns}, \"ident\", {});\n",
        b32(totality_code(want))
    );
    let env = check_nodule(
        &parse(&program(&entries)).unwrap_or_else(|e| panic!("scenario 1: parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("scenario 1: check failed: {e}"));
    assert_verdict(&env, "scenario 1 (non-recursive) ident", "verdict_ident");
}

#[test]
fn totality_myc_matches_oracle_self_descent() {
    let want = oracle_totality(&scenario_self_descent(), "countdown");
    assert_eq!(
        want,
        Totality::Total,
        "oracle sanity: a structurally-descending self-call is Total"
    );
    let call = myc_call(
        "countdown",
        &["Path(Pth(Cons(\"m\", Nil : Vec[Bytes])))".to_string()],
    );
    let decl = myc_peano_match_fn("countdown", &call);
    let fns = myc_fn_table(&[("countdown", decl)]);
    let entries = format!(
        "fn verdict_countdown() => Binary{{32}} = verdict_for_name({fns}, \"countdown\", {});\n",
        b32(totality_code(want))
    );
    let env = check_nodule(
        &parse(&program(&entries)).unwrap_or_else(|e| panic!("scenario 2: parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("scenario 2: check failed: {e}"));
    assert_verdict(
        &env,
        "scenario 2 (self descent) countdown",
        "verdict_countdown",
    );
}

#[test]
fn totality_myc_matches_oracle_self_non_descent() {
    let want = oracle_totality(&scenario_self_non_descent(), "bad");
    assert_eq!(
        want,
        Totality::Partial,
        "oracle sanity: a non-descending self-call is Partial"
    );
    let call = myc_call(
        "bad",
        &["Path(Pth(Cons(\"n\", Nil : Vec[Bytes])))".to_string()],
    );
    let decl = myc_peano_match_fn("bad", &call);
    let fns = myc_fn_table(&[("bad", decl)]);
    let entries = format!(
        "fn verdict_bad() => Binary{{32}} = verdict_for_name({fns}, \"bad\", {});\n",
        b32(totality_code(want))
    );
    let env = check_nodule(
        &parse(&program(&entries)).unwrap_or_else(|e| panic!("scenario 3: parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("scenario 3: check failed: {e}"));
    assert_verdict(&env, "scenario 3 (self non-descent) bad", "verdict_bad");
}

#[test]
fn totality_myc_matches_oracle_mutual_descent() {
    let scenario = scenario_mutual_descent();
    let want_even = oracle_totality(&scenario, "even");
    let want_odd = oracle_totality(&scenario, "odd");
    assert_eq!(
        want_even,
        Totality::Total,
        "oracle sanity: mutual descent is Total (even)"
    );
    assert_eq!(
        want_odd,
        Totality::Total,
        "oracle sanity: mutual descent is Total (odd)"
    );

    let even_call = myc_call(
        "odd",
        &["Path(Pth(Cons(\"m\", Nil : Vec[Bytes])))".to_string()],
    );
    let odd_call = myc_call(
        "even",
        &["Path(Pth(Cons(\"m\", Nil : Vec[Bytes])))".to_string()],
    );
    let fns = myc_fn_table(&[
        ("even", myc_peano_match_fn("even", &even_call)),
        ("odd", myc_peano_match_fn("odd", &odd_call)),
    ]);
    let entries = format!(
        "fn verdict_even() => Binary{{32}} = verdict_for_name({fns}, \"even\", {want_even});\n\
         fn verdict_odd() => Binary{{32}} = verdict_for_name({fns}, \"odd\", {want_odd});\n",
        want_even = b32(totality_code(want_even)),
        want_odd = b32(totality_code(want_odd)),
    );
    let env = check_nodule(
        &parse(&program(&entries)).unwrap_or_else(|e| panic!("scenario 4: parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("scenario 4: check failed: {e}"));
    assert_verdict(&env, "scenario 4 (mutual descent) even", "verdict_even");
    assert_verdict(&env, "scenario 4 (mutual descent) odd", "verdict_odd");
}

#[test]
fn totality_myc_matches_oracle_mutual_non_descent() {
    let scenario = scenario_mutual_non_descent();
    let want_evenb = oracle_totality(&scenario, "evenb");
    let want_oddb = oracle_totality(&scenario, "oddb");
    assert_eq!(
        want_evenb,
        Totality::Partial,
        "oracle sanity: non-descending mutual recursion is Partial (evenb)"
    );
    assert_eq!(
        want_oddb,
        Totality::Partial,
        "oracle sanity: non-descending mutual recursion is Partial (oddb)"
    );

    let evenb_call = myc_call(
        "oddb",
        &["Path(Pth(Cons(\"n\", Nil : Vec[Bytes])))".to_string()],
    );
    let oddb_call = myc_call(
        "evenb",
        &["Path(Pth(Cons(\"n\", Nil : Vec[Bytes])))".to_string()],
    );
    let fns = myc_fn_table(&[
        ("evenb", myc_peano_match_fn("evenb", &evenb_call)),
        ("oddb", myc_peano_match_fn("oddb", &oddb_call)),
    ]);
    let entries = format!(
        "fn verdict_evenb() => Binary{{32}} = verdict_for_name({fns}, \"evenb\", {want_evenb});\n\
         fn verdict_oddb() => Binary{{32}} = verdict_for_name({fns}, \"oddb\", {want_oddb});\n",
        want_evenb = b32(totality_code(want_evenb)),
        want_oddb = b32(totality_code(want_oddb)),
    );
    let env = check_nodule(
        &parse(&program(&entries)).unwrap_or_else(|e| panic!("scenario 5: parse failed: {e}")),
    )
    .unwrap_or_else(|e| panic!("scenario 5: check failed: {e}"));
    assert_verdict(
        &env,
        "scenario 5 (mutual non-descent) evenb",
        "verdict_evenb",
    );
    assert_verdict(&env, "scenario 5 (mutual non-descent) oddb", "verdict_oddb");
}

/// The oracle `Totality` mapped to `totality.myc`'s own `totality_to_code` encoding (`Total` = 1,
/// `Partial` = 0) — the differential's shared code, computed from the LIVE oracle result, never
/// hand-guessed.
fn totality_code(t: Totality) -> u32 {
    match t {
        Totality::Total => 1,
        Totality::Partial => 0,
    }
}
