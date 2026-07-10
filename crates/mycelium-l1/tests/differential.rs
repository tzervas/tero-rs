//! The RFC-0007 §4.6 **differential obligation** (NFR-7): on the evaluation-complete fragment,
//! **L1-eval**, **elaborate→L0-interp**, and the **M-150 AOT path** must agree on the observable
//! (`repr + payload + guarantee`) — and every agreeing pair validates through the **M-210 shared
//! TV checker** (`mycelium_cert::check`, the `ObservationalEquiv` instance), the same checker
//! that validates swap certificates and the M-151 interp↔AOT differential.
//!
//! All three paths dispatch through the *same* trusted prim registry and certified swap engine;
//! what this test pins down is that the L1 machinery layered on top — the big-step environment
//! evaluator on one side, inlining elaboration on the other — cannot make "two execution paths
//! mean two semantics".
//!
//! Outside the fragment the obligation is different and also tested here: elaboration must refuse
//! with an explicit `Residual` (never a partial artifact), while the L1 evaluator still runs the
//! program — and a `Partial`-classified unproductive recursion ends in an explicit
//! `FuelExhausted`, never a hang (§4.5).

use mycelium_cert::{
    check, check_core, BinaryTernarySwapEngine, CheckVerdict, Evidence, RefinementRelation,
};
use mycelium_core::{GuaranteeStrength, Payload, Repr, Value};
use mycelium_interp::{Interpreter, PrimRegistry};
use mycelium_l1::elab::build_registry;
use mycelium_l1::{
    check_nodule, check_phylum, elaborate, monomorphize, parse, parse_phylum, ElabError, Evaluator,
    L1Error, L1Value,
};
use mycelium_numerics::Certificate;

type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);

fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}

/// The fragment corpus: checked colonies with a nullary `main` whose bodies inline to
/// `Const/Var/Let/Op/Swap` residue. Each runs on all three paths.
fn corpus() -> Vec<&'static str> {
    vec![
        // bare literal
        "nodule d;\nfn main() => Binary{8} = 0b1011_0010;",
        // let / var
        "nodule d;\nfn main() => Binary{8} = let a = 0b1011_0010 in a;",
        // unary + binary bit ops
        "nodule d;\nfn main() => Binary{8} = not(0b1011_0010);",
        "nodule d;\nfn main() => Binary{8} = xor(0b1011_0010, 0b1111_1111);",
        // balanced-ternary arithmetic (in range — never a silent wrap)
        "nodule d;\nfn main() => Ternary{4} = add(0t00+-, 0t0+0-);",
        "nodule d;\nfn main() => Ternary{4} = mul(0t00+0, 0t00-0);",
        // RFC-0025 / M-705: the SAME programs written with infix/prefix operator sugar. Each
        // desugars (frontend-only, KC-3) to the canonical word call above, so all three paths
        // (L1-eval ≡ L0-interp ≡ AOT) must agree on it exactly as on the word form — pinning the
        // sugar↔word equivalence end-to-end through the trusted prim registry.
        "nodule d;\nfn main() => Binary{8} = 0b1011_0010 ^ 0b1111_1111;",
        "nodule d;\nfn main() => Binary{8} = !0b1011_0010;",
        "nodule d;\nfn main() => Ternary{4} = 0t00+- + 0t0+0-;",
        "nodule d;\nfn main() => Ternary{4} = 0t00+0 * 0t00-0;",
        // precedence: `*` binds tighter than `+`, so `a + b * c` ≡ `add(a, mul(b, c))`.
        "nodule d;\nfn main() => Ternary{4} = 0t000+ + 0t00+0 * 0t00-0;",
        // RFC-0025 §4.1 / M-745: the angle/shift glyphs (`<`/`>`/`<<`/`>>` → lt/gt/shl/shr) are
        // NOT in this three-path corpus on purpose — their word targets have no kernel/stdlib prim
        // yet (they arrive with M-809), so a program using them does not *resolve* end-to-end.
        // Their desugaring is instead pinned at the AST level by the parse-equivalence tests
        // (`src/tests/parse.rs::infix_sugar_desugars_to_the_word_call` — `a < b` ≡ `lt(a, b)`,
        // etc.), which is the honest oracle for a frontend-only rewrite (§4.4: the desugared `App`
        // node IS the record). They join this corpus when their prims land (G2 — never silent on
        // the coverage boundary).
        // the certified binary→ternary swap
        "nodule d;\nfn main() => Ternary{6} = swap(0b1011_0010, to: Ternary{6}, policy: rt);",
        // a call, inlined (acyclic call graph)
        "nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(flip(0b1010_1010));",
        // round-trip swap through a let
        "nodule d;\nfn main() => Binary{8} =\n  let b = 0b0010_1010 in swap(swap(b, to: Ternary{6}, policy: rt), to: Binary{8}, policy: rt);",
        // an op feeding a swap, through a helper
        "nodule d;\nfn widen(x: Binary{8}) => Ternary{6} = swap(not(x), to: Ternary{6}, policy: rt);\nfn main() => Ternary{6} = widen(0b1011_0010);",
        // --- M-666: the `colony { hypha … }` structured-concurrency surface (RFC-0008 §4.7) ---
        // The reference semantics is the RT2 spawn-order sequentialization (RFC-0008 §4.2), so all
        // three execution paths (L1-eval ≡ elaborate→L0-interp ≡ AOT) must agree on it like any
        // other in-fragment program — a single-hypha colony is exactly its body.
        "nodule d;\nfn main() => Binary{8} = colony { hypha not(0b1011_0010) };",
        // A multi-hypha colony: leading hyphae are evaluated for effect (here pure), the observable
        // is the last hypha's value (no v0 product type). Determinism: the value is independent of
        // any scheduling — the sequentialization is the meaning.
        "nodule d;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} =\n  colony { hypha compute(0b0000_1111), hypha compute(0b1010_1010), hypha xor(0b1111_0000, 0b0000_1111) };",
        // --- M-906 (DN-70 D1): `@forage(policy) hypha …` — the D-lite placement-policy surface ---
        // Semantics-free (RT3): a well-formed (non-empty-candidate) `@forage` annotation must not
        // change the observable on any of the three execution paths — placement is a performance
        // concern layered over the unchanged body, exactly like `reclaim`'s policy (DN-58 §B).
        "nodule d;\nfn main() => Binary{8} = colony { @forage(0b1) hypha not(0b1011_0010) };",
        // A multi-bit mask (3 candidate workers) and a multi-hypha colony where only the LAST hypha
        // carries the annotation — pins that the annotation is per-hypha, and that a leading
        // un-annotated hypha is unaffected.
        "nodule d;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} =\n  colony { hypha compute(0b0000_1111), @forage(0b101) hypha compute(0b1010_1010) };",
        // --- RFC-0020 §9 / R20-Q5: list-literal element-type inference from return-type context ---
        // The return type `Seq{Binary{8}, 3}` flows into the list literal `[…]` bidirectionally as
        // the `expected` type in `check_list` — the element type `Binary{8}` is NOT determined
        // bottom-up from the elements (which here are explicit bit-literals anyway), but the return
        // annotation *would* be necessary for bare-decimal elements. All three paths must agree on
        // the `Seq` repr value (L1-eval, elaborate→L0-interp, AOT). The observable is the full
        // `Seq{Binary{8}, 3}` payload.
        "nodule d;\nfn main() => Seq{Binary{8}, 3} = [0b1111_0000, 0b0000_1111, 0b1010_1010];",
    ]
}

#[test]
fn l1_eval_l0_interp_and_aot_agree_on_the_fragment() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    for (i, src) in corpus().iter().enumerate() {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        // Path 1: the L1 fuel-guarded evaluator.
        let l1 = Evaluator::new(&env)
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("program #{i}: L1-eval failed: {e}"));
        let l1 = l1
            .as_repr()
            .unwrap_or_else(|| panic!("program #{i}: fragment result must be a repr value"))
            .clone();

        // Path 2: elaborate to L0, run on the reference interpreter.
        let node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("program #{i}: must be in the fragment: {e}"));
        let l0 = interp
            .eval(&node)
            .unwrap_or_else(|e| panic!("program #{i}: L0-interp failed: {e}"));

        // Path 3: the same L0 term through the AOT path (M-150).
        let aot = mycelium_mlir::run(&node, &prims, &engine)
            .unwrap_or_else(|e| panic!("program #{i}: AOT failed: {e}"));

        assert_eq!(
            observable(&l1),
            observable(&l0),
            "program #{i} diverged: L1-eval vs L0-interp"
        );
        assert_eq!(
            observable(&l0),
            observable(&aot),
            "program #{i} diverged: L0-interp vs AOT"
        );

        // M-210: each agreeing pair validates through the one shared TV checker.
        for (a, b, pair) in [(&l1, &l0, "L1↔interp"), (&l0, &aot, "interp↔AOT")] {
            assert_eq!(
                check(
                    a,
                    b,
                    RefinementRelation::ObservationalEquiv,
                    Certificate::exact(),
                    &Evidence::Observational,
                ),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "program #{i}: the shared checker must validate the {pair} pair"
            );
        }
    }
}

/// The **data + recursion fragment** (RFC-0011 r3/r4): with `Construct`/`Match`/`Lam`/`App`/`Fix` now
/// L0 nodes, a program that builds/matches data and recurses elaborates to a closed L0 term. As of
/// **M-342 (Q5 closed)** the obligation is the full three-way differential **L1-eval ≡
/// elaborate→L0-interp ≡ AOT** on the L0 [`CoreValue`] observable — the AOT `aot::run_core`
/// env-machine now covers the data + recursion fragment (it was repr-only in r3). The L1 evaluator's
/// name-keyed data value is bridged onto the elaborated value's content-addressed `#T#i` identity
/// through the *same* registry (`L1Value::to_core`), so a divergence in any of the three machineries —
/// the big-step `try_match`, the Maranget→flat-`Match` lowering, or the ANF env-machine — is caught.
fn data_corpus() -> Vec<&'static str> {
    vec![
        // a flat data match returning a repr value
        "nodule d;\ntype Sign = Neg | Zero | Pos;\nfn label(s: Sign) => Ternary{1} = match s { Neg => 0t-, Zero => 0t0, _ => 0t+ };\nfn main() => Ternary{1} = label(Zero);",
        // a data RESULT (the program evaluates to a datum)
        "nodule d;\ntype Nat = Z | S(Nat);\nfn main() => Nat = S(S(Z));",
        // nested patterns (Maranget) returning a datum
        "nodule d;\ntype Nat = Z | S(Nat);\nfn pred2(n: Nat) => Nat = match n { Z => Z, S(Z) => Z, S(S(m)) => m };\nfn main() => Nat = pred2(S(S(S(Z))));",
        // a literal-pattern match over a Binary scrutinee
        "nodule d;\nfn classify(b: Binary{4}) => Ternary{1} = match b { 0b0000 => 0t0, 0b1111 => 0t+, _ => 0t- };\nfn main() => Ternary{1} = classify(0b1111);",
        // a data value with a repr field, destructured (binds a field, runs a prim on it)
        "nodule d;\ntype Box = Mk(Binary{8});\nfn flip(x: Box) => Binary{8} = match x { Mk(b) => not(b) };\nfn main() => Binary{8} = flip(Mk(0b1010_1010));",
        // `if` desugaring to a Bool match
        "nodule d;\nfn pick(b: Bool) => Binary{8} = if b then 0b1111_1111 else 0b0000_0000;\nfn main() => Binary{8} = pick(True);",
        // a constructed result carrying a computed repr field
        "nodule d;\ntype Box = Mk(Binary{8});\nfn main() => Box = Mk(not(0b0000_1111));",
        // a multi-field constructor matched with a NESTED wildcard at a non-root occurrence
        // (M-320 Maranget: column ordering over two fields + a `_` at occurrence [1]) — the kind of
        // decision tree the flat Nat cases don't stress; all three paths must still agree
        "nodule d;\ntype Pair = Mk(Bool, Bool);\nfn both(p: Pair) => Bool = match p { Mk(True, b) => b, Mk(False, _) => False };\nfn main() => Bool = both(Mk(True, False));",
        // --- r4: functions + recursion (Lam/App/Fix), now in the fragment ---
        // self-recursion returning a datum (Fix + App + Match)
        "nodule d;\ntype Nat = Z | S(Nat);\nfn drop_(n: Nat) => Nat = match n { Z => Z, S(m) => drop_(m) };\nfn main() => Nat = drop_(S(S(S(Z))));",
        // self-recursion building data on the way back (a recursive copy)
        "nodule d;\ntype Nat = Z | S(Nat);\nfn copy(n: Nat) => Nat = match n { Z => Z, S(m) => S(copy(m)) };\nfn main() => Nat = copy(S(S(Z)));",
        // a `for` fold over a list spine (desugars to a synthesized Fix fold)
        "nodule d;\ntype ByteList = End | More(Binary{8}, ByteList);\nfn checksum(bs: ByteList) => Binary{8} = for b in bs, acc = 0b0000_0000 => xor(acc, b);\nfn main() => Binary{8} = checksum(More(0b1111_0000, More(0b0000_1111, End)));",
        // a recursive helper called by a non-recursive one (inlining + Fix coexist)
        "nodule d;\ntype Nat = Z | S(Nat);\nfn drop_(n: Nat) => Nat = match n { Z => Z, S(m) => drop_(m) };\nfn twice_drop(n: Nat) => Nat = drop_(drop_(n));\nfn main() => Nat = twice_drop(S(S(Z)));",
        // --- r5: mutual recursion (FixGroup), M-343 ---
        // a mutually-recursive pair returning a datum: ping(SS Z) ⟶ pong(S Z) ⟶ ping(Z) ⟶ Z
        "nodule d;\ntype Nat = Z | S(Nat);\nfn ping(n: Nat) => Nat = match n { Z => Z, S(m) => pong(m) };\nfn pong(n: Nat) => Nat = match n { Z => Z, S(m) => ping(m) };\nfn main() => Nat = ping(S(S(Z)));",
        // mutual recursion over a Bool result (even/odd): even(SSS Z) ⟶ odd(SS Z) ⟶ … ⟶ False
        "nodule d;\ntype Nat = Z | S(Nat);\nfn even(n: Nat) => Bool = match n { Z => True, S(m) => odd(m) };\nfn odd(n: Nat) => Bool = match n { Z => False, S(m) => even(m) };\nfn main() => Bool = even(S(S(S(Z))));",
        // mutual recursion that BUILDS data on the way back (constructive through the group):
        // f(SSS Z) ⟶ S(g(SS Z)) ⟶ S(f(S Z)) ⟶ S(S(g(Z))) ⟶ S(S(Z))
        "nodule d;\ntype Nat = Z | S(Nat);\nfn f(n: Nat) => Nat = match n { Z => Z, S(m) => S(g(m)) };\nfn g(n: Nat) => Nat = match n { Z => Z, S(m) => f(m) };\nfn main() => Nat = f(S(S(S(Z))));",
        // a three-function mutual cycle (f → g → h → f) returning a datum
        "nodule d;\ntype Nat = Z | S(Nat);\nfn f3(n: Nat) => Nat = match n { Z => Z, S(m) => g3(m) };\nfn g3(n: Nat) => Nat = match n { Z => Z, S(m) => h3(m) };\nfn h3(n: Nat) => Nat = match n { Z => Z, S(m) => f3(m) };\nfn main() => Nat = f3(S(S(S(S(Z)))));",
        // --- RFC-0020 §9 / R20-Q3: or-patterns ---
        // An or-pattern `Neg | Pos => body` desugars (KC-3) to two arms sharing the same body.
        // All three paths (L1-eval ≡ elaborate→L0-interp ≡ AOT) must agree on the desugared form.
        // The desugar is checker-level only — zero new L0 node (uses the existing Match/Alt).
        "nodule d;\ntype Sign = Neg | Zero | Pos;\nfn classify(s: Sign) => Ternary{1} = match s { Neg | Pos => 0t-, Zero => 0t0 };\nfn main() => Ternary{1} = classify(Neg);",
        // A three-alternative or-pattern — all alts share the same body; exhaustive.
        "nodule d;\ntype Bit = Zero | One;\nfn always_one(b: Bit) => Binary{1} = match b { Zero | One => 0b1 };\nfn main() => Binary{1} = always_one(Zero);",
        // --- M-391 (R7-Q3 surface): two further surface-written mutual-recursion shapes ---
        // a mutual pair returning a REPR (not a datum): hi(SS Z) ⟶ lo(S Z) ⟶ hi(Z) ⟶ 0b1111_1111
        "nodule d;\ntype Nat = Z | S(Nat);\nfn hi(n: Nat) => Binary{8} = match n { Z => 0b1111_1111, S(m) => lo(m) };\nfn lo(n: Nat) => Binary{8} = match n { Z => 0b0000_0000, S(m) => hi(m) };\nfn main() => Binary{8} = hi(S(S(Z)));",
        // a mutual pair destructuring a MULTI-FIELD constructor (Maranget over two fields, inside a
        // FixGroup): shrink(Mk(S Z, S Z)) ⟶ expand(Mk(Z, S Z)) ⟶ shrink(Mk(Z, Z)) ⟶ Z.
        // (`expand`, not `grow` — `grow` is a DN-03 §1 reserved surface keyword as of M-664.)
        "nodule d;\ntype Nat = Z | S(Nat);\ntype Two = Mk(Nat, Nat);\nfn shrink(t: Two) => Nat = match t { Mk(Z, b) => b, Mk(S(a), b) => expand(Mk(a, b)) };\nfn expand(t: Two) => Nat = match t { Mk(a, Z) => a, Mk(a, S(b)) => shrink(Mk(a, b)) };\nfn main() => Nat = shrink(Mk(S(Z), S(Z)));",
    ]
}

#[test]
fn l1_eval_l0_interp_and_aot_agree_on_the_data_and_recursion_fragment() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    for (i, src) in data_corpus().iter().enumerate() {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");
        let registry = build_registry(&env).expect("the data registry builds");

        // Path 1: the L1 fuel-guarded evaluator, projected onto the L0 CoreValue domain.
        let l1 = Evaluator::new(&env)
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("program #{i}: L1-eval failed: {e}"));
        let l1_core = l1
            .to_core(&env, &registry)
            .unwrap_or_else(|| panic!("program #{i}: L1 result is outside the r3 data fragment"));

        // Path 2: elaborate to L0, run on the reference interpreter (eval_core spans repr + data).
        let node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("program #{i}: must be in the r3/r4 fragment: {e}"));
        let l0_core = interp
            .eval_core(&node)
            .unwrap_or_else(|e| panic!("program #{i}: L0-interp failed: {e}"));

        // Path 3: the same L0 term through the AOT env-machine (M-342) — now spans data + recursion.
        let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
            .unwrap_or_else(|e| panic!("program #{i}: AOT run_core failed: {e}"));

        // All three paths must agree on the whole L0 value — constructor identity, fields, and the
        // meet-summary guarantee (for a datum) or repr+payload+guarantee (for a repr value).
        assert_eq!(
            l1_core, l0_core,
            "program #{i} diverged: L1-eval vs elaborate→L0-interp"
        );
        assert_eq!(
            l0_core, aot_core,
            "program #{i} diverged: L0-interp vs AOT env-machine"
        );
        assert_eq!(
            l1_core.guarantee(),
            aot_core.guarantee(),
            "program #{i}: guarantee summaries disagree (L1 vs AOT)"
        );

        // The single shared M-210 checker validates each pair through `check_core` — now over the
        // **whole** `CoreValue` (datum *or* repr), so the data + recursion fragment's *datum* results
        // validate through the same checker the repr fragment uses, closing M-302's "through the
        // M-210 ObservationalEquiv checker" obligation for the full kernel corpus (never a bespoke
        // structural compare; a mislabeled lowering is an explicit `NotValidated`, not a silent pass).
        for (x, y, pair) in [
            (&l1_core, &l0_core, "L1↔interp"),
            (&l0_core, &aot_core, "interp↔AOT"),
        ] {
            assert_eq!(
                check_core(x, y),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "program #{i}: the shared checker must validate the {pair} result pair"
            );
        }
    }
}

/// The arities (member counts) of **every** `FixGroup` in `n`, in pre-order — a small structural probe
/// so the M-391 identity assertion can confirm a surface group lowered to *exactly* the FixGroup(s)
/// expected (count *and* size, not just the first one). Walks the whole term, including inside each
/// `FixGroup`'s member lambdas and body, so a nested or spurious group cannot hide.
fn fixgroup_arities(n: &mycelium_core::Node) -> Vec<usize> {
    use mycelium_core::{Alt, Node};
    match n {
        Node::FixGroup { defs, body } => {
            let mut v = vec![defs.len()];
            for (_, d) in defs {
                v.extend(fixgroup_arities(d));
            }
            v.extend(fixgroup_arities(body));
            v
        }
        Node::Let { bound, body, .. } => {
            let mut v = fixgroup_arities(bound);
            v.extend(fixgroup_arities(body));
            v
        }
        Node::Fix { body, .. } | Node::Lam { body, .. } => fixgroup_arities(body),
        Node::App { func, arg } => {
            let mut v = fixgroup_arities(func);
            v.extend(fixgroup_arities(arg));
            v
        }
        Node::Op { args, .. } | Node::Construct { args, .. } => {
            args.iter().flat_map(fixgroup_arities).collect()
        }
        Node::Swap { src, .. } => fixgroup_arities(src),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            let mut v = fixgroup_arities(scrutinee);
            for alt in alts {
                match alt {
                    Alt::Ctor { body, .. } | Alt::Lit { body, .. } => {
                        v.extend(fixgroup_arities(body))
                    }
                }
            }
            if let Some(d) = default {
                v.extend(fixgroup_arities(d));
            }
            v
        }
        Node::Const(_) | Node::Var(_) => Vec::new(),
    }
}

/// M-391 / ADR-003 (identity-first): a mutually-recursive group written in surface syntax lowers to
/// *the* `FixGroup` the SCC decomposition dictates — deterministically (same source ⟶ byte-equal term,
/// so the content hash is stable) and materialized as that concrete, content-addressed L0 node (the
/// grouping is reified, never a black box; walked here). There is a single `FixGroup` emission path
/// (RP-6 nodule-wide visibility feeds the existing Tarjan→`FixGroup` lowering; DN-13), so
/// "surface-written ≡ programmatic" is pinned here against that canonical path.
#[test]
fn surface_mutual_recursion_lowers_to_the_canonical_fixgroup() {
    let src = "nodule d;\ntype Nat = Z | S(Nat);\nfn ping(n: Nat) => Nat = match n { Z => Z, S(m) => pong(m) };\nfn pong(n: Nat) => Nat = match n { Z => Z, S(m) => ping(m) };\nfn main() => Nat = ping(S(S(Z)));";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");

    // Determinism: the lowering (fresh-name numbering, group member order) is reproducible, so the
    // term — and therefore its content hash — is stable across elaborations of the same source.
    let a = elaborate(&env, "main").expect("elaborates");
    let b = elaborate(&env, "main").expect("elaborates");
    assert_eq!(
        a, b,
        "elaboration must be deterministic (stable content identity)"
    );

    // The surface ping/pong group is materialized as exactly one 2-member `FixGroup` — the concrete,
    // content-addressed L0 node that reifies the grouping (no black box). Walking the whole term must
    // find exactly one `FixGroup`, of arity 2 (uniqueness — not merely "the first one encountered").
    assert_eq!(
        fixgroup_arities(&a),
        vec![2],
        "the surface ping/pong group must lower to exactly one 2-member FixGroup"
    );
}

/// Never-silent (G2): RP-6 makes top-level functions mutually visible, so `ping` may forward-reference
/// `pong` — but a reference to a function that does **not** exist must stay an explicit checker error,
/// never silently absorbed into the mutual group as a phantom member. Here `pongg` is a typo for
/// `pong`; the program must be REJECTED at check time, not elaborated.
#[test]
fn an_undefined_reference_is_an_explicit_error_not_a_silent_mutual_group() {
    let src = "nodule d;\ntype Nat = Z | S(Nat);\nfn ping(n: Nat) => Nat = match n { Z => Z, S(m) => pongg(m) };\nfn pong(n: Nat) => Nat = match n { Z => Z, S(m) => ping(m) };\nfn main() => Nat = ping(S(Z));";
    let nodule = parse(src).expect("parses");
    let err = check_nodule(&nodule).expect_err("an undefined reference must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("pongg"),
        "the error must explicitly name the undefined reference; got: {msg}"
    );
}

/// A **mutant-witness** for the elaboration: a deliberately wrong elaboration must be caught by the
/// differential. We construct a divergence directly — two structurally different data programs whose
/// L0 values must *not* compare equal — confirming the data comparison discriminates (a vacuous
/// `assert_eq!` that always passed would be the bug this guards against).
#[test]
fn the_data_differential_distinguishes_divergent_elaborations() {
    let env = |src| check_nodule(&parse(src).unwrap()).unwrap();
    let reg = |e: &mycelium_l1::Env| build_registry(e).unwrap();
    let e1 = env("nodule d;\ntype Nat = Z | S(Nat);\nfn main() => Nat = S(Z);");
    let e2 = env("nodule d;\ntype Nat = Z | S(Nat);\nfn main() => Nat = S(S(Z));");
    let a = Evaluator::new(&e1)
        .call("main", vec![])
        .unwrap()
        .to_core(&e1, &reg(&e1))
        .unwrap();
    let b = Evaluator::new(&e2)
        .call("main", vec![])
        .unwrap()
        .to_core(&e2, &reg(&e2))
        .unwrap();
    assert_ne!(a, b, "S(Z) and S(S(Z)) must be distinct L0 data values");
    // And the shared M-210 checker must *report* the divergence on the datum pair — a mislabeled
    // lowering is an explicit `NotValidated`, not merely unequal (M-302; NFR-7/VR-4).
    assert!(
        matches!(check_core(&a, &b), CheckVerdict::NotValidated { .. }),
        "the shared checker must reject the divergent datum pair, not silently pass"
    );
}

/// Sanity: the harness discriminates — the shared checker explicitly rejects a genuinely
/// divergent pair, so a passing differential is meaningful, not vacuous.
#[test]
fn the_differential_distinguishes_different_programs() {
    let env = |src| check_nodule(&parse(src).unwrap()).unwrap();
    let e1 = env("nodule d;\nfn main() => Binary{8} = 0b1011_0010;");
    let e2 = env("nodule d;\nfn main() => Binary{8} = 0b1111_1111;");
    let a = Evaluator::new(&e1).call("main", vec![]).unwrap();
    let b = Evaluator::new(&e2).call("main", vec![]).unwrap();
    let verdict = check(
        a.as_repr().unwrap(),
        b.as_repr().unwrap(),
        RefinementRelation::ObservationalEquiv,
        Certificate::exact(),
        &Evidence::Observational,
    );
    assert!(
        matches!(verdict, CheckVerdict::NotValidated { .. }),
        "the checker must reject a divergent pair, got {verdict:?}"
    );
}

/// r4: self-recursion is now **in** the fragment — it elaborates to a `Fix` and agrees with the L1
/// evaluator (the differential corpus exercises this). A `Total`-classified recursion still runs on
/// the L1 evaluator too; the two paths agree on the L0 value.
#[test]
fn self_recursion_elaborates_and_agrees() {
    let src = "nodule d;\ntype Nat = Z | S(Nat);\nfn drop_(n: Nat) => Nat = match n { Z => Z, S(m) => drop_(m) };\nfn main() => Nat = drop_(S(S(Z)));";
    let env = check_nodule(&parse(src).unwrap()).unwrap();
    let registry = build_registry(&env).unwrap();
    assert_eq!(env.totality["drop_"], mycelium_l1::Totality::Total);

    // Recursion now elaborates (no Residual) — r4 retired the §4.6 refusal for self-recursion.
    let node = elaborate(&env, "main").expect("self-recursion elaborates in r4");
    let l0 = Interpreter::default()
        .eval_core(&node)
        .expect("L0-interp runs");
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap()
        .to_core(&env, &registry)
        .unwrap();
    assert_eq!(
        l1, l0,
        "L1-eval and elaborate→L0-interp agree on the recursive result (Z)"
    );
}

/// **M-343 (R7-Q3):** mutual recursion now elaborates to a `FixGroup`, and all three paths agree —
/// L1-eval ≡ elaborate→L0-interp ≡ AOT — on a mutually-recursive program. (Was the r4 boundary where
/// elaboration refused with a `Residual`; M-343 enacts it.) The broader corpus coverage is in
/// `l1_eval_l0_interp_and_aot_agree_on_the_data_and_recursion_fragment`; this pins the named case.
#[test]
fn mutual_recursion_elaborates_and_all_three_paths_agree() {
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let src = "nodule d;\ntype Nat = Z | S(Nat);\nfn ping(n: Nat) => Nat = match n { Z => Z, S(m) => pong(m) };\nfn pong(n: Nat) => Nat = match n { Z => Z, S(m) => ping(m) };\nfn main() => Nat = ping(S(S(Z)));";
    let env = check_nodule(&parse(src).unwrap()).unwrap();
    let registry = build_registry(&env).unwrap();

    // The mutually-recursive group structurally descends on position 0, so the totality checker
    // classifies it `Total` (M-343 / R7-Q3 mutual-descent classification, RFC-0007 §4.5).
    assert_eq!(env.totality["ping"], mycelium_l1::Totality::Total);
    assert_eq!(env.totality["pong"], mycelium_l1::Totality::Total);

    // Mutual recursion now elaborates (no Residual) — it lowers to a FixGroup.
    let node = elaborate(&env, "main").expect("mutual recursion elaborates to a FixGroup (M-343)");

    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap()
        .to_core(&env, &registry)
        .unwrap();
    let l0 = Interpreter::default()
        .eval_core(&node)
        .expect("L0-interp runs the FixGroup");
    let aot = mycelium_mlir::run_core(&node, &prims, &engine).expect("AOT runs the FixGroup");

    assert_eq!(
        l1, l0,
        "L1-eval vs elaborate→L0-interp diverged on mutual recursion"
    );
    assert_eq!(
        l0, aot,
        "L0-interp vs AOT env-machine diverged on mutual recursion"
    );
    // ping(S(S(Z))) ⟶ pong(S(Z)) ⟶ ping(Z) ⟶ Z (a nullary datum).
    assert_eq!(
        l0,
        mycelium_core::CoreValue::Data(mycelium_core::Datum::new(
            registry.ctor_ref("Nat", 0).unwrap(),
            vec![]
        )),
        "the result must be Nat::Z"
    );
}

/// **M-352 (RFC-0014):** an explicit recovery handling site elaborates to an L0 `Match` over a
/// **result sum** — recovery introduces **no new kernel node** (KC-3). `handle e { Ok(v) => v,
/// Err(_) => fallback }` *is* a `Match` on `Result = Ok | Err`, the data+match fragment the three-way
/// differential already covers; this pins the named recovery case: L1-eval ≡ elaborate→L0-interp ≡
/// AOT. (The concrete `handle` spelling is KC-2-gated, RFC-0006; the semantics is this match.)
#[test]
fn recovery_match_over_a_result_sum_agrees_three_ways() {
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    // Written in the existing data+match surface (no new syntax) — the lowering target of a recovery
    // handling site: match the result sum, recover the `Err` case with an explicit fallback.
    let src = "nodule d;\ntype Result = Ok(Binary{8}) | Err(Binary{8});\nfn recover(r: Result) => Binary{8} = match r { Ok(v) => v, Err(e) => 0b0000_0000 };\nfn main() => Binary{8} = recover(Err(0b1111_1111));";
    let env = check_nodule(&parse(src).unwrap()).unwrap();
    let registry = build_registry(&env).unwrap();
    let node = elaborate(&env, "main").expect("a recovery match elaborates (no new kernel node)");

    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .unwrap()
        .to_core(&env, &registry)
        .unwrap();
    let l0 = Interpreter::default()
        .eval_core(&node)
        .expect("L0-interp runs the recovery match");
    let aot = mycelium_mlir::run_core(&node, &prims, &engine).expect("AOT runs the recovery match");

    assert_eq!(
        l1, l0,
        "L1-eval vs elaborate→L0-interp diverged on the recovery match"
    );
    assert_eq!(
        l0, aot,
        "L0-interp vs AOT env-machine diverged on the recovery match"
    );
    // The `Err(_)` arm recovers to the explicit fallback `0b0000_0000`.
    let recovered = l0.as_repr().expect("a Binary result value");
    assert_eq!(recovered.repr(), &Repr::Binary { width: 8 });
    assert_eq!(
        recovered.payload(),
        &Payload::Bits(vec![false; 8]),
        "the recovery fallback must be the zero byte"
    );
}

/// **M-353 (RFC-0014 §4.8):** wiring the recovery `Budgets` ledger into the env-machine must be
/// **meaning-preserving** (NFR-7). The same recovery match, run through the env-machine with an *ample*
/// effect ledger threaded (`run_core_with_effects`), produces the identical observable as the plain
/// `run_core` / L0-interp paths — the §4.8 budget plumbing perturbs *nothing* when budgets suffice; it
/// only adds the explicit, graceful `EffectBudget` refusal at an overrun (tested on the runtime path in
/// `mycelium-mlir`). This pins the L0 touch-point of the integration to the three-way differential.
#[test]
fn the_effect_ledger_is_meaning_preserving_on_the_recovery_match() {
    use mycelium_interp::{Budgets, EffectBudget};
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let src = "nodule d;\ntype Result = Ok(Binary{8}) | Err(Binary{8});\nfn recover(r: Result) => Binary{8} = match r { Ok(v) => v, Err(e) => 0b0000_0000 };\nfn main() => Binary{8} = recover(Err(0b1111_1111));";
    let env = check_nodule(&parse(src).unwrap()).unwrap();
    let node = elaborate(&env, "main").unwrap();

    let plain = mycelium_mlir::run_core(&node, &prims, &engine).unwrap();
    // An ample `alloc` budget — never overruns on this shallow match — must not change the observable.
    let mut budgets = Budgets::new().with(EffectBudget::Bytes(1 << 30));
    let with_ledger = mycelium_mlir::run_core_with_effects(
        &node,
        &prims,
        &engine,
        1_000_000,
        1_000_000,
        &mut budgets,
    )
    .unwrap();
    assert_eq!(
        plain, with_ledger,
        "threading an ample effect ledger must be observable-transparent (NFR-7)"
    );
}

/// A `Partial`-classified unproductive recursion: still runnable, but the clock is the guard —
/// an explicit `FuelExhausted`, never a hang (§4.5).
#[test]
fn a_partial_program_exhausts_fuel_explicitly() {
    let src = "nodule d;\ntype Nat = Z | S(Nat);\nfn spin(n: Nat) => Nat = spin(n);\nfn main() => Nat = spin(Z);";
    let env = check_nodule(&parse(src).unwrap()).unwrap();
    assert_eq!(env.totality["spin"], mycelium_l1::Totality::Partial);
    let err = Evaluator::new(&env)
        .with_fuel(50)
        .call("main", vec![])
        .unwrap_err();
    assert_eq!(err, L1Error::FuelExhausted);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// M-666 (redone): the `colony` **RT2 real-concurrency differential** — concurrent ≡ sequential.
//
// RFC-0008 §4.2/§4.7/RT2: the reference semantics of a deterministic concurrent program is its
// deterministic sequentialization. `mycelium_l1::elaborate` lowers a colony to that sequentialization
// (the oracle); `mycelium_l1::elaborate_colony` lowers it to per-hypha closed L0 programs which
// `mycelium_mlir::run_colony` runs as **real concurrent tasks** (`Scope`/`Colony`, structured
// fork/join, M-357), validating concurrent ≡ sequential. These tests pin that the concurrent
// observable equals the sequential reference every other path already agrees on — Empirically (a
// differential over a corpus + a property, not a `Proven` theorem; VR-5).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// Run `entry`'s colony through `mycelium_mlir::run_colony` (real concurrent execution of the
/// per-hypha L0 programs) and return the colony's observable as a `CoreValue`.
fn run_colony_concurrent(env: &mycelium_l1::Env, entry: &str) -> mycelium_core::CoreValue {
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let hyphae = mycelium_l1::elaborate_colony(env, entry)
        .expect("the colony elaborates to its per-hypha L0 programs");
    mycelium_mlir::run_colony(&hyphae, &prims, &engine, 1_000_000, 1_000_000)
        .expect("the colony runs concurrently and the schedules agree (RT2)")
}

/// **The RT2 real-concurrency differential (M-666).** For each colony in the corpus, the **concurrent**
/// run (`run_colony`, real interleaved tasks) must produce the **identical** observable as the
/// **sequential reference** — both the L1 evaluator's spawn-order run and the `elaborate`→interp
/// sequentialization — and every agreeing pair validates through the shared M-210 checker. This is the
/// faithful RT2 obligation (RFC-0008 §4.6): "concurrent observable ≡ the deterministic reference's
/// observable", now over an executor that genuinely interleaves the hyphae (not a sequential stand-in).
#[test]
fn colony_concurrent_run_equals_the_sequential_reference_rt2() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    // Colonies of varied shapes: a single hypha (degenerate), pure multi-hypha, hyphae that call a
    // helper and a recursive function (exercising the shared recursive-binder prelude per hypha).
    let corpus = [
        "nodule d;\nfn main() => Binary{8} = colony { hypha not(0b1011_0010) };",
        "nodule d;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} =\n  colony { hypha compute(0b0000_1111), hypha compute(0b1010_1010), hypha xor(0b1111_0000, 0b0000_1111) };",
        // a hypha that drives a recursive (Total) function — the per-hypha prelude must carry the `Fix`
        "nodule d;\ntype Nat = Z | S(Nat);\nfn depth(n: Nat) => Binary{8} = match n { Z => 0b0000_0000, S(m) => not(depth(m)) };\nfn main() => Binary{8} =\n  colony { hypha depth(S(S(Z))), hypha not(0b0000_0001), hypha depth(S(Z)) };",
        // a swap reached through a helper call — the colony spans the repr-conversion fragment too.
        // (A hypha body is an `app_expr`, the prior M-666 surface — KEEP; the `swap` keyword form is
        // wrapped in `widen`, exactly the existing differential corpus's `widen` pattern.)
        "nodule d;\nfn widen(x: Binary{8}) => Ternary{6} = swap(x, to: Ternary{6}, policy: rt);\nfn keep(x: Ternary{6}) => Ternary{6} = x;\nfn main() => Ternary{6} =\n  colony { hypha keep(0t00+0-+), hypha widen(0b1011_0010) };",
    ];

    for (i, src) in corpus.iter().enumerate() {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");

        // The sequential reference, two independent ways (the RT2 oracle).
        let l1_seq = Evaluator::new(&env)
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("colony #{i}: L1-eval (sequential reference) failed: {e}"));
        let l1_seq = l1_seq
            .as_repr()
            .unwrap_or_else(|| panic!("colony #{i}: reference result must be a repr value"))
            .clone();
        let seq_node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("colony #{i}: sequentialization must elaborate: {e}"));
        let interp_seq = interp
            .eval(&seq_node)
            .unwrap_or_else(|e| panic!("colony #{i}: elaborate→interp (reference) failed: {e}"));

        // The CONCURRENT run (real interleaved tasks) — the heart of M-666.
        let concurrent = run_colony_concurrent(&env, "main");
        let concurrent = concurrent
            .as_repr()
            .unwrap_or_else(|| panic!("colony #{i}: concurrent result must be a repr value"))
            .clone();

        // RT2: the concurrent observable equals the sequential reference (both ways).
        assert_eq!(
            observable(&concurrent),
            observable(&l1_seq),
            "colony #{i}: concurrent run diverged from the L1 sequential reference (RT2 violated)"
        );
        assert_eq!(
            observable(&concurrent),
            observable(&interp_seq),
            "colony #{i}: concurrent run diverged from the elaborate→interp reference (RT2 violated)"
        );

        // The shared M-210 checker validates the concurrent↔reference pair like any other equivalence.
        assert_eq!(
            check(
                &concurrent,
                &interp_seq,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "colony #{i}: the shared checker must validate the concurrent↔reference pair"
        );
    }
}

/// **Property: the bound on the RT2 differential.** For *any* number `k` of leading hyphae (0..=8), a
/// colony whose hyphae are pure unary ops run **concurrently** (`run_colony`) yields exactly the
/// **last** hypha's value — i.e. the concurrent observable is independent of the `k` leading hyphae and
/// equal to the deterministic sequentialization (RFC-0008 RT2). This is the empirical-confidence
/// breadth behind the `Empirical` determinism tag: many shapes, all concurrent ≡ sequential.
#[test]
fn prop_colony_concurrent_value_is_its_last_hypha_for_any_leading_count() {
    use mycelium_core::{Payload, Repr};
    for k in 0u32..=8 {
        // k leading `not(<i>)` hyphae (evaluated concurrently for effect), then a fixed last hypha
        // whose value (`not(0b0101_0101) = 0b1010_1010`) is the colony's observable for every k.
        let mut hyphae = String::new();
        for i in 0..k {
            let bits = format!("{:08b}", i & 0xFF);
            let bits = format!("{}_{}", &bits[..4], &bits[4..]);
            hyphae.push_str(&format!("hypha not(0b{bits}), "));
        }
        hyphae.push_str("hypha not(0b0101_0101)");
        let src = format!("nodule d;\nfn main() => Binary{{8}} = colony {{ {hyphae} }};");
        let env = check_nodule(&parse(&src).expect("parses")).expect("checks");

        let concurrent = run_colony_concurrent(&env, "main");
        let v = concurrent
            .as_repr()
            .unwrap_or_else(|| panic!("k={k}: concurrent colony result must be a repr value"));
        assert_eq!(v.repr(), &Repr::Binary { width: 8 });
        assert_eq!(
            v.payload(),
            &Payload::Bits(vec![true, false, true, false, true, false, true, false]),
            "k={k}: the CONCURRENT colony's value must equal its LAST hypha (RT2), \
             independent of the {k} leading hyphae"
        );

        // And it equals the sequential reference for this k (the differential, parameterised).
        let seq = Evaluator::new(&env).call("main", vec![]).unwrap();
        assert_eq!(
            observable(seq.as_repr().unwrap()),
            observable(v),
            "k={k}: concurrent ≡ sequential reference (RT2)"
        );
    }
}

/// **A hypha's explicit failure is surfaced, never silently dropped (G2/RT4/I1).** `run_colony`
/// requires every hypha to complete cleanly; a hypha whose L0 evaluation fails (here a deliberate
/// `FuelExhausted` from a starved fuel budget on a recursive hypha) is reported as an explicit
/// `ColonyError::HyphaFailed` carrying its index — never absorbed into a "successful" colony.
#[test]
fn a_failing_hypha_is_an_explicit_colony_error_not_a_silent_drop() {
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    // A Total recursion that needs more than the tiny fuel we give it → an explicit FuelExhausted in
    // that hypha's L0 evaluation; the colony must surface it, not return the last hypha's value.
    let src = "nodule d;\ntype Nat = Z | S(Nat);\nfn depth(n: Nat) => Binary{8} = match n { Z => 0b0000_0000, S(m) => not(depth(m)) };\nfn main() => Binary{8} =\n  colony { hypha depth(S(S(S(S(S(Z)))))), hypha not(0b0000_0001) };";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let hyphae = mycelium_l1::elaborate_colony(&env, "main").expect("elaborates per-hypha");
    // Starve fuel so the recursive hypha #0 cannot finish — an explicit, graceful refusal.
    let err = mycelium_mlir::run_colony(&hyphae, &prims, &engine, 2, 1_000_000).expect_err(
        "a starved recursive hypha must make the colony refuse, never silently succeed",
    );
    match err {
        mycelium_mlir::ColonyError::HyphaFailed { index, outcome } => {
            assert_eq!(index, 0, "the failing hypha is #0 (the recursive one)");
            assert!(
                outcome.contains("Fuel") || outcome.contains("Failed"),
                "the failure is the explicit evaluator refusal; got: {outcome}"
            );
        }
        other => panic!("expected an explicit HyphaFailed, got: {other}"),
    }
}

// --- M-906 (DN-70 D1; RFC-0008 RT3; DN-63 §3.5): `@forage(policy)` D-lite conformance ------------
//
// Three obligations from the D-lite scope split (DN-70 D1's row-by-row DoD): (1) the DN-63 §3.5
// FLAG-14 empty-candidate-set case is an explicit `ForageError::NoCandidates`, agreeing across
// every execution path (never a silent divergence — mirrors `dense_swap_is_an_explicit_residual_
// on_all_paths` above); (2) the RT2 placement differential — two DIFFERENT placements of the same
// deterministic computation must produce the same observable (DN-70 D1 "Conformance obligation");
// (3) the mandatory RFC-0005 §2.2 EXPLAIN trail is genuinely populated, not just documented.

/// **DN-63 §3.5 FLAG-14 — an all-zero `@forage` bitmask is an explicit refusal on every path.**
/// `elaborate` (the sequential-reference path feeding both L0-interp and AOT) refuses with an
/// explicit [`ElabError::Residual`] (`crate::elab::forage_reject_if_empty`); L1-eval refuses with
/// the typed `L1Error::Forage(ForageError::NoCandidates)`. Neither path silently accepts a
/// no-candidate placement (G2/RT4) — classification: Explicit-Residual + Explicit-Kernel-style
/// refusal (`Empirical` — this test), the same shape `dense_swap_is_an_explicit_residual_on_all_
/// paths` pins for the Dense-swap residual.
#[test]
fn forage_no_candidates_is_an_explicit_refusal_on_every_path() {
    let src = "nodule d;\nfn main() => Binary{8} = colony { @forage(0b0) hypha not(0b1011_0010) };";
    let env = check_nodule(&parse(src).expect(
        "an all-zero `@forage` bitmask still checks \
        (the D-lite checker only validates shape: literal + Binary type — the empty-candidate \
        refusal is an elaboration/runtime obligation, DN-63 §3.5 FLAG-14)",
    ))
    .expect("checks");

    // Path A: `elaborate` refuses with an explicit Residual — never a fabricated L0 program.
    let elab_err = elaborate(&env, "main")
        .expect_err("elaborate must refuse an all-zero `@forage` bitmask (DN-63 §3.5 FLAG-14)");
    assert!(
        matches!(elab_err, ElabError::Residual { .. }),
        "the all-zero-bitmask elaborate error must be an explicit Residual; got: {elab_err}"
    );
    assert!(
        elab_err.to_string().contains("FLAG-14") || elab_err.to_string().contains("NoCandidates"),
        "the Residual message must name the DN-63 FLAG-14 empty-candidate-set case; got: {elab_err}"
    );

    // Path B: L1-eval refuses explicitly with the typed `ForageError::NoCandidates`.
    let l1_err = Evaluator::new(&env)
        .call("main", vec![])
        .expect_err("L1-eval must refuse an all-zero `@forage` bitmask explicitly");
    assert!(
        matches!(l1_err, L1Error::Forage(mycelium_l1::ForageError::NoCandidates)),
        "L1-eval's all-zero-bitmask error must be the typed ForageError::NoCandidates; got: {l1_err}"
    );

    // Both paths refuse explicitly and consistently: neither elaborate→{L0-interp, AOT} nor
    // L1-eval ever silently accepts a no-candidate `@forage` (DN-63 §3.5 FLAG-14 → satisfied).
}

/// **The RT2 placement differential (DN-70 D1's explicit conformance obligation).** Two DIFFERENT
/// `@forage` bitmasks — naming disjoint, differently-sized candidate sets, so `mycelium-select`
/// genuinely picks a *different* `NodeRef` for each — must still produce the SAME observable
/// (`repr + payload + guarantee`) for the same deterministic body, on every execution path
/// (L1-eval ≡ elaborate→L0-interp ≡ AOT). This is RT3's "the running node changes performance,
/// never the observable" made executable: placement genuinely varies (asserted below via
/// `forage_decisions()`), the value does not.
#[test]
fn forage_placement_choice_does_not_change_the_observable_rt2() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    // Two same-width masks with the set bit at DIFFERENT string positions (`@forage` candidates
    // are indexed by bitmask digit position — see `eval_hypha_forage`'s doc comment) — `0b10`
    // names only `worker-0`, `0b01` names only `worker-1`; `SelectionPolicy`'s `default_choice: 0`
    // therefore picks a genuinely different `NodeRef` for each, over disjoint candidate sets.
    let masks = ["0b10", "0b01"];
    let mut chosen_nodes = Vec::new();
    let mut repr_observables = Vec::new();

    for mask in masks {
        let src = format!(
            "nodule d;\nfn main() => Binary{{8}} = colony {{ @forage({mask}) hypha not(0b1011_0010) }};"
        );
        let env = check_nodule(&parse(&src).expect("parses")).expect("checks");

        let evaluator = Evaluator::new(&env);
        let l1 = evaluator
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("mask {mask}: L1-eval failed: {e}"));
        let l1 = l1
            .as_repr()
            .unwrap_or_else(|| panic!("mask {mask}: fragment result must be a repr value"))
            .clone();
        let decisions = evaluator.forage_decisions();
        assert_eq!(
            decisions.len(),
            1,
            "mask {mask}: exactly one `@forage` decision must be recorded"
        );
        chosen_nodes.push(decisions[0].explanation.chosen.clone());

        let node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("mask {mask}: must be in the fragment: {e}"));
        let l0 = interp
            .eval(&node)
            .unwrap_or_else(|e| panic!("mask {mask}: L0-interp failed: {e}"));
        let aot = mycelium_mlir::run(&node, &prims, &engine)
            .unwrap_or_else(|e| panic!("mask {mask}: AOT failed: {e}"));

        assert_eq!(
            observable(&l1),
            observable(&l0),
            "mask {mask}: L1-eval vs L0-interp diverged"
        );
        assert_eq!(
            observable(&l0),
            observable(&aot),
            "mask {mask}: L0-interp vs AOT diverged"
        );
        repr_observables.push(l1);
    }

    // The two masks genuinely chose DIFFERENT placement candidates…
    assert_ne!(
        chosen_nodes[0], chosen_nodes[1],
        "the two disjoint bitmasks must choose different `NodeRef` candidates \
         (otherwise this is not exercising RT2 at all)"
    );
    // …yet the two runs' observables are identical — placement is genuinely semantics-free (RT3):
    // the RT2 placement differential DN-70 D1 requires.
    assert_eq!(
        observable(&repr_observables[0]),
        observable(&repr_observables[1]),
        "different `@forage` placements of the SAME deterministic body must agree on the \
         observable (RT2/RT3 — placement is semantics-free)"
    );
}

/// **Mandatory EXPLAIN is genuinely wired (RFC-0005 §2.2; M-906).** A well-formed `@forage`
/// decision is recorded in [`Evaluator::forage_decisions`] with the site, the policy's own
/// content address (`policy_ref`), the full per-candidate cost ranking, and the chosen `NodeRef` —
/// inspectable, never a black box (house rule 2).
#[test]
fn forage_decision_is_recorded_in_the_explain_trail() {
    let src =
        "nodule d;\nfn main() => Binary{8} = colony { @forage(0b101) hypha not(0b1011_0010) };";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let evaluator = Evaluator::new(&env);
    evaluator.call("main", vec![]).expect("runs");

    let decisions = evaluator.forage_decisions();
    assert_eq!(
        decisions.len(),
        1,
        "exactly one `@forage` decision recorded"
    );
    let d = &decisions[0];
    assert_eq!(
        d.site, "main",
        "the decision is attributed to its enclosing fn"
    );
    // `0b101` names candidates `worker-0` and `worker-2` (bits 0 and 2 set); `default_choice: 0`
    // deterministically picks the lowest-index candidate — `worker-0`.
    assert_eq!(
        d.explanation.costs.len(),
        2,
        "the EXPLAIN's cost ranking must list every candidate the bitmask named (2 set bits)"
    );
    assert_eq!(
        d.explanation.chosen,
        mycelium_select::Candidate::Node(mycelium_select::NodeRef("worker-0".to_owned())),
        "the deterministic default arm must choose the lowest-index candidate"
    );
    assert!(
        !d.explanation.overridden,
        "a D-lite decision is never a forced override"
    );
}

// --- M-673: monomorphization — generics + traits to closed L0, three-way differential ----------
//
// After M-673 a generic *instantiation* and a *trait-method call* both elaborate to closed L0 (the
// monomorphization pre-pass — `crate::mono`). The obligation is the SAME three-way differential as
// the data/recursion corpus (L1-eval ≡ elaborate→L0-interp ≡ AOT), but run on the **monomorphized
// env**: the L1 evaluator has no trait-method dispatch (`eval_app` resolves only `env.fns`/ctor/prim),
// so a trait program is only runnable once mono has rewritten its trait calls to direct calls. Running
// the *generic* cases on the mono'd env too keeps the harness uniform (a generic call's head name is
// already in `env.fns`, so L1-eval would also run the source env — but the mono'd env is the honest
// common ground for both kinds).

/// The generic + trait fragment corpus (mirrors `data_corpus`): each program has a nullary `main`
/// whose reachable graph uses generics and/or a trait/impl, and monomorphizes to closed L0.
fn generic_corpus() -> Vec<&'static str> {
    vec![
        // (1) `List<A>` + `first_or` → closed L0 (the M-673 acceptance fixture)
        "nodule d;\ntype List[A] = Nil | Cons(A, List[A]);\nfn first_or[A](xs: List[A], d: A) => A = match xs { Nil => d, Cons(x, _) => x };\nfn main() => Binary{8} = first_or(Cons(0b0000_0001, Nil), 0b0000_0000);",
        // (2) a generic returning a datum (the program evaluates to a `List<Binary{8}>`)
        "nodule d;\ntype List[A] = Nil | Cons(A, List[A]);\nfn main() => List[Binary{8}] = Cons(0b0000_0001, Nil);",
        // (3) a trait + impl, the method called directly (static resolution to a direct call)
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };\nfn main() => Binary{2} = cmp(0b0000_0001, 0b0000_0010);",
        // (4) a bounded generic `use_cmp<T: Cmp>` calling the trait method through its bound, at Binary{8}
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };\nfn use_cmp[T: Cmp](a: T, b: T) => Binary{2} = cmp(a, b);\nfn main() => Binary{2} = use_cmp(0b0000_0001, 0b0000_0010);",
        // (5) fragmentation witness — `first_or` at Binary{8} AND Binary{4} reachable from one main
        "nodule d;\ntype List[A] = Nil | Cons(A, List[A]);\nfn first_or[A](xs: List[A], d: A) => A = match xs { Nil => d, Cons(x, _) => x };\nfn lo() => Binary{4} = first_or(Cons(0b0001, Nil), 0b0000);\nfn hi() => Binary{8} = first_or(Cons(0b0000_0001, Nil), 0b0000_0000);\nfn main() => Binary{8} = let _w = lo() in hi();",
        // (6) a generic recursive fold over a generic spine (Fix over List<Binary{8}>)
        "nodule d;\ntype List[A] = Nil | Cons(A, List[A]);\nfn sum_(xs: List[Binary{8}]) => Binary{8} = match xs { Nil => 0b0000_0000, Cons(x, r) => xor(x, sum_(r)) };\nfn main() => Binary{8} = sum_(Cons(0b0000_1111, Cons(0b1111_0000, Nil)));",
        // (7) a generic instantiated at a USER DATA TYPE as the type arg (not just reprs) — exercises
        //     the repr/data-name mangling boundary end-to-end (the locus of the M-673 injectivity fix)
        "nodule d;\ntype Bit = O | I;\ntype Box[A] = Wrap(A);\nfn unbox(b: Box[Bit]) => Bit = match b { Wrap(x) => x };\nfn main() => Bit = unbox(Wrap(I));",
        // (8) DN-58 §A.5 (M-817): a **Data**-type `fuse` desugars (monomorphization) to the resolved
        //     `Fuse::join` call — an ordinary inlined trait-method call that runs three-way. `Flag`'s
        //     `join` is the absorbing-`On` OR (a commutative/associative/idempotent join-semilattice);
        //     `fuse(On, Off)` = `join(On, Off)` = `On`. This is the user-merge case the brief targets.
        //     `Fuse` is now a **built-in prelude trait** (M-965 F-A1) — no `trait Fuse` declaration
        //     needed — and its `join` here is exhaustively law-checked at `impl` time (M-965 F-A2).
        "nodule d;\ntype Flag = Off | On;\nimpl Fuse[Flag] for Flag { fn join(a: Flag, b: Flag) => Flag = match a { On => On, Off => b }; };\nfn main() => Flag = fuse(On, Off);",
        // --- M-826: tuple/product type round-trip through monomorphization (KC-3 / three-way diff) ---
        // (9) 2-tuple `fst` — checks, monos, and all three paths agree on `Nat` result.
        //     Tuple$2<Nat, Nat> monomorphizes to Tuple$2[Nat] (a closed type); `fst` extracts field 0.
        "nodule d;\ntype Nat = Z | S(Nat);\nfn fst(t: (Nat, Nat)) => Nat = match t { (a, _) => a };\nfn main() => Nat = fst((S(Z), Z));",
        // (10) 2-tuple `snd` — extract field 1 to pin field ordering end-to-end.
        "nodule d;\ntype Nat = Z | S(Nat);\nfn snd(t: (Nat, Nat)) => Nat = match t { (_, b) => b };\nfn main() => Nat = snd((Z, S(Z)));",
        // (11) 3-tuple mid-element — pins the 3-arity synthetic `Tuple$3<Nat, Nat, Nat>`.
        "nodule d;\ntype Nat = Z | S(Nat);\nfn mid(t: (Nat, Nat, Nat)) => Nat = match t { (_, b, _) => b };\nfn main() => Nat = mid((Z, S(Z), Z));",
    ]
}

#[test]
fn l1_eval_l0_interp_and_aot_agree_on_the_monomorphized_generic_and_trait_fragment() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    for (i, src) in generic_corpus().iter().enumerate() {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");
        // Monomorphize: a closed, trait-free, monomorphic env L1-eval can run (it has no trait
        // dispatch). The entry stays `main` (nullary monomorphic ⇒ name unchanged).
        let mono = monomorphize(&env, "main")
            .unwrap_or_else(|e| panic!("program #{i}: must monomorphize: {e}"));
        // The mono'd env has no generics/traits left (the M-673 closure invariant).
        assert!(
            mono.fns.values().all(|fd| fd.sig.params.is_empty())
                && mono.types.values().all(|d| d.params.is_empty())
                && mono.traits.is_empty()
                && mono.instances.is_empty()
                && mono.impls.is_empty(),
            "program #{i}: monomorphized env must be closed (no generics/traits)"
        );
        let registry = build_registry(&mono).expect("the mono'd data registry builds");

        // Path 1: the L1 fuel-guarded evaluator, on the MONOMORPHIZED env (trait calls are now direct).
        let l1 = Evaluator::new(&mono)
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("program #{i}: L1-eval failed: {e}"));
        let l1_core = l1
            .to_core(&mono, &registry)
            .unwrap_or_else(|| panic!("program #{i}: L1 result is outside the r3 data fragment"));

        // Path 2: elaborate to L0 (elaborate monomorphizes internally; on the source env it produces
        // the same closed term), run on the reference interpreter.
        let node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("program #{i}: must elaborate after M-673: {e}"));
        let l0_core = interp
            .eval_core(&node)
            .unwrap_or_else(|e| panic!("program #{i}: L0-interp failed: {e}"));

        // Path 3: the same L0 term through the AOT env-machine.
        let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
            .unwrap_or_else(|e| panic!("program #{i}: AOT run_core failed: {e}"));

        assert_eq!(
            l1_core, l0_core,
            "program #{i} diverged: L1-eval(mono) vs elaborate→L0-interp"
        );
        assert_eq!(
            l0_core, aot_core,
            "program #{i} diverged: L0-interp vs AOT env-machine"
        );
        // The single shared M-210 checker validates each pair (a mislabeled lowering is an explicit
        // NotValidated, never a silent pass).
        for (x, y, pair) in [
            (&l1_core, &l0_core, "L1↔interp"),
            (&l0_core, &aot_core, "interp↔AOT"),
        ] {
            assert_eq!(
                check_core(x, y),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "program #{i}: the shared checker must validate the {pair} pair"
            );
        }
    }
}

/// Determinism across the boundary (M-673): monomorphizing twice yields a byte-equal `Env`, and
/// elaborating the same source twice yields a byte-equal L0 term — the content identity the swarm's
/// hashing relies on. (Identity is *fragmented* per instantiation, but each is stable.)
#[test]
fn monomorphization_and_its_elaboration_are_deterministic() {
    for src in generic_corpus() {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");
        let a = monomorphize(&env, "main").expect("mono a");
        let b = monomorphize(&env, "main").expect("mono b");
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "monomorphization must be deterministic"
        );
        let ea = elaborate(&env, "main").expect("elab a");
        let eb = elaborate(&env, "main").expect("elab b");
        assert_eq!(
            ea, eb,
            "elaboration of a mono'd program must be deterministic"
        );
    }
}

/// A **mutant-witness** for the monomorphized differential: two structurally different trait/generic
/// programs must NOT produce equal L0 values — confirming the comparison discriminates (a vacuous
/// `assert_eq!` would be the bug this guards). Here two impls give the method different bodies.
#[test]
fn the_monomorphized_differential_distinguishes_divergent_instances() {
    let run = |src: &str| {
        let env = check_nodule(&parse(src).unwrap()).unwrap();
        let node = elaborate(&env, "main").unwrap();
        Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(BinaryTernarySwapEngine),
        )
        .eval_core(&node)
        .unwrap()
    };
    // Same trait + call shape, different impl method body (`0b00` vs `0b11`) ⇒ different L0 results.
    let a = run(
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };\nfn main() => Binary{2} = cmp(0b0000_0001, 0b0000_0010);",
    );
    let b = run(
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b11; };\nfn main() => Binary{2} = cmp(0b0000_0001, 0b0000_0010);",
    );
    assert_ne!(
        a, b,
        "different impl bodies must yield different L0 values (the differential discriminates)"
    );
}

// --- M-688: HOF differential — named fns passed to map/and_then/fold over Result ----------------
//
// RFC-0024 §4 (M-685/686/687): a named top-level function is now a first-class value; the
// monomorphizer (mono.rs) specializes the HOF combinator at the call site (defunctionalization),
// yielding closed first-order L0. The obligation is the SAME three-way differential as the
// generic/trait corpus (L1-eval ≡ elaborate→L0-interp ≡ AOT) on the MONOMORPHIZED env — run on
// the mono'd env so L1-eval (which has no HOF dispatch) can run the same program. Differential
// agreement is `Empirical` (trials; VR-5). Contract is `Declared`.
//
// All programs include the std.result combinators inline (inlining from lib/std/result.myc so each
// program is a self-contained source string; the checker sees the full nodule). Named helpers:
//   `not_val(x: Binary{8}) -> Binary{8} = not(x)` — the function value passed to map/and_then
//   `mk_ok_inner(x: Binary{8}) -> Result<Binary{8},Binary{8}> = Ok(not(x))` — for and_then
//   `id_val(x: Binary{8}) -> Binary{8} = x`           — on_ok branch for fold
//   `const_zero(e: Binary{8}) -> Binary{8} = xor(e, e)` — on_err branch for fold (always 0)

/// The HOF corpus: programs using map/and_then/fold with named function arguments, inline.
/// Each must monomorphize to closed L0 — the defunctionalization obligation (RFC-0024 §4).
///
/// Empirical: differential agreement confirmed by the three-way harness below; not a proof.
fn hof_corpus() -> Vec<&'static str> {
    vec![
        // (1) map Ok: map(Ok(0b0000_0001), not_val) → Ok(not(0b0000_0001)) = Ok(0b1111_1110)
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =\n  match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_ok(), not_val);",
        // (2) map Err: map(Err(0b1111_1111), not_val) → Err(0b1111_1111) [Err passes through]
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =\n  match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_err(), not_val);",
        // (3) and_then Ok: and_then(Ok(0b0000_0001), mk_ok_inner) → Ok(not(0b0000_0001)) = Ok(0b1111_1110)
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn and_then[A, B, E](r: Result[A, E], f: A => Result[B, E]) => Result[B, E] =\n  match r { Ok(x) => f(x), Err(e) => Err(e) };\nfn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = and_then(mk_ok(), mk_ok_inner);",
        // (4) and_then Err: and_then(Err(0b1111_1111), mk_ok_inner) → Err(0b1111_1111) [short-circuits]
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn and_then[A, B, E](r: Result[A, E], f: A => Result[B, E]) => Result[B, E] =\n  match r { Ok(x) => f(x), Err(e) => Err(e) };\nfn mk_ok_inner(x: Binary{8}) => Result[Binary{8},Binary{8}] = Ok(not(x));\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_1111);\nfn main() => Result[Binary{8},Binary{8}] = and_then(mk_err(), mk_ok_inner);",
        // (5) fold Ok: fold(Ok(0b1010_1010), id_val, const_zero) → id_val(0b1010_1010) = 0b1010_1010
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn fold[A, E, B](r: Result[A, E], on_ok: A => B, on_err: E => B) => B =\n  match r { Ok(x) => on_ok(x), Err(e) => on_err(e) };\nfn id_val(x: Binary{8}) => Binary{8} = x;\nfn const_zero(e: Binary{8}) => Binary{8} = xor(e, e);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b1010_1010);\nfn main() => Binary{8} = fold(mk_ok(), id_val, const_zero);",
        // (6) fold Err: fold(Err(0b1111_0000), id_val, const_zero) → xor(0b1111_0000,0b1111_0000) = 0b0000_0000
        "nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn fold[A, E, B](r: Result[A, E], on_ok: A => B, on_err: E => B) => B =\n  match r { Ok(x) => on_ok(x), Err(e) => on_err(e) };\nfn id_val(x: Binary{8}) => Binary{8} = x;\nfn const_zero(e: Binary{8}) => Binary{8} = xor(e, e);\nfn mk_err() => Result[Binary{8},Binary{8}] = Err(0b1111_0000);\nfn main() => Binary{8} = fold(mk_err(), id_val, const_zero);",
    ]
}

/// **M-688 (RFC-0024 §4):** HOF programs using named fn arguments to map/and_then/fold over
/// Result run through the three-way differential — L1-eval(mono) ≡ elaborate→L0-interp ≡ AOT —
/// on the MONOMORPHIZED env. This is the end-to-end proof that static defunctionalization (M-687)
/// produces closed first-order L0 that agrees on all three evaluation paths. Mirrors the
/// `l1_eval_l0_interp_and_aot_agree_on_the_monomorphized_generic_and_trait_fragment` harness.
///
/// Empirical: differential agreement is by trial (VR-5 — never Proven). Declared: type contract.
#[test]
fn l1_eval_l0_interp_and_aot_agree_on_hof_via_defunctionalization() {
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    for (i, src) in hof_corpus().iter().enumerate() {
        let env = check_nodule(&parse(src).expect("parses")).expect("checks");
        // Monomorphize: resolves both generic type args AND defunctionalizes the fn-valued params
        // (RFC-0024 §4, M-687). The result is a closed, first-order, trait-free env L1-eval can run.
        let mono = monomorphize(&env, "main").unwrap_or_else(|e| {
            panic!("HOF program #{i}: must monomorphize + defunctionalize: {e}")
        });
        // Closure invariant (M-673 / RFC-0024 §4): no generics, no traits, no fn-typed params.
        assert!(
            mono.fns.values().all(|fd| fd.sig.params.is_empty())
                && mono.types.values().all(|d| d.params.is_empty())
                && mono.traits.is_empty()
                && mono.instances.is_empty()
                && mono.impls.is_empty(),
            "HOF program #{i}: monomorphized+defunctionalized env must be closed (no generics/traits)"
        );
        let registry = build_registry(&mono).expect("the mono'd data registry builds");

        // Path 1: the L1 fuel-guarded evaluator, on the MONOMORPHIZED+DEFUNCTIONALIZED env.
        // (L1-eval has no HOF dispatch — it can only run the defunctionalized, first-order version.)
        let l1 = Evaluator::new(&mono)
            .call("main", vec![])
            .unwrap_or_else(|e| panic!("HOF program #{i}: L1-eval(mono) failed: {e}"));
        let l1_core = l1.to_core(&mono, &registry).unwrap_or_else(|| {
            panic!("HOF program #{i}: L1 result is outside the r3 data fragment")
        });

        // Path 2: elaborate to L0 (elaborate calls monomorphize internally — on the source env),
        // run on the reference interpreter. Empirical: Err arms must pass through, Ok arms transform.
        let node = elaborate(&env, "main").unwrap_or_else(|e| {
            panic!("HOF program #{i}: must elaborate after defunctionalization: {e}")
        });
        let l0_core = interp
            .eval_core(&node)
            .unwrap_or_else(|e| panic!("HOF program #{i}: L0-interp failed: {e}"));

        // Path 3: the same L0 term through the AOT env-machine.
        let aot_core = mycelium_mlir::run_core(&node, &prims, &engine)
            .unwrap_or_else(|e| panic!("HOF program #{i}: AOT run_core failed: {e}"));

        // All three paths must agree — Empirical (differential over HOF corpus; VR-5).
        assert_eq!(
            l1_core, l0_core,
            "HOF program #{i} diverged: L1-eval(mono+defun) vs elaborate→L0-interp"
        );
        assert_eq!(
            l0_core, aot_core,
            "HOF program #{i} diverged: L0-interp vs AOT env-machine"
        );

        // The shared M-210 checker validates each agreeing pair (a mislabeled lowering is an
        // explicit NotValidated, never a silent pass — NFR-7/VR-4/G2).
        for (x, y, pair) in [
            (&l1_core, &l0_core, "L1↔interp"),
            (&l0_core, &aot_core, "interp↔AOT"),
        ] {
            assert_eq!(
                check_core(x, y),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "HOF program #{i}: the shared checker must validate the {pair} pair"
            );
        }
    }
}

/// **Mutant-witness (M-688):** two different named functions passed to `map` produce different L0
/// results — confirming the defunctionalization discriminates. A vacuous differential that always
/// passes regardless of the fn argument would not be caught by the harness above; this closes
/// that gap. `not_val` and `id_val` yield different images on a non-all-ones input (Empirical).
#[test]
fn the_hof_differential_distinguishes_different_named_fn_arguments() {
    let run = |src: &str| {
        let env = check_nodule(&parse(src).unwrap()).unwrap();
        let node = elaborate(&env, "main").unwrap();
        Interpreter::new(
            PrimRegistry::with_builtins(),
            Box::new(BinaryTernarySwapEngine),
        )
        .eval_core(&node)
        .unwrap()
    };
    // Same map call, different fn argument: not_val vs id_val — must give different L0 results on
    // input 0b0000_0001 (not(0b0000_0001) = 0b1111_1110 ≠ 0b0000_0001 = id_val(0b0000_0001)).
    let with_not = run("nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =\n  match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn not_val(x: Binary{8}) => Binary{8} = not(x);\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_ok(), not_val);");
    let with_id = run("nodule d;\ntype Result[A, E] = Ok(A) | Err(E);\nfn map[A, B, E](r: Result[A, E], f: A => B) => Result[B, E] =\n  match r { Ok(x) => Ok(f(x)), Err(e) => Err(e) };\nfn id_val(x: Binary{8}) => Binary{8} = x;\nfn mk_ok() => Result[Binary{8},Binary{8}] = Ok(0b0000_0001);\nfn main() => Result[Binary{8},Binary{8}] = map(mk_ok(), id_val);");
    assert_ne!(
        with_not, with_id,
        "map with not_val vs id_val must yield different L0 values (the HOF differential discriminates)"
    );
}

// --- M-720/M-721: the `wild`/FFI execution floor, three-way differential ------------------------
//
// RFC-0028 §4.2/§4.3: a `wild { name(args…) }` block in a `@std-sys` nodule lowers to a host-dispatch
// `Op{prim:"wild:name"}` (M-720, **no new Core-IR node** — KC-3) and *executes* by dispatching through
// the prim registry — the capability handle. Because all three paths thread the SAME registry, a
// `wild`-backed op resolves identically on L1-eval, L0-interp, and AOT — so the three-way differential
// extends to it (closing DN-14 row 9's staged-execution gap for a deterministic host op, `Empirical`).

/// A deterministic mock host op for the `wild`/FFI differential (M-721): `wild:echo` returns its single
/// argument unchanged. It is *deterministic* (unlike a real syscall), so the three paths can be asserted
/// equal — exactly the recorded `Empirical` basis for the `wild` dispatch mechanism (RFC-0028 §4.6).
/// Real syscall host ops (entropy/clock/io) stay `Declared`: non-deterministic, so not coverable by an
/// equality differential.
fn wild_echo(_prim: &str, args: &[&Value]) -> Result<Value, mycelium_interp::EvalError> {
    match args {
        [v] => Ok((*v).clone()),
        _ => Err(mycelium_interp::EvalError::PrimType {
            prim: "wild:echo".to_owned(),
            why: "the test host op `echo` expects exactly one argument".to_owned(),
        }),
    }
}

/// The trusted built-ins **plus** the `wild:echo` host op registered — a host that *grants* the `echo`
/// capability (RFC-0028 §4.3). The default `with_builtins()` registry grants no `wild:` op.
fn host_registry() -> PrimRegistry {
    let mut prims = PrimRegistry::with_builtins();
    prims.register("wild:echo", wild_echo);
    prims
}

/// **M-720/M-721 — the `wild`/FFI execution floor, three-way.** A `wild` block in a `@std-sys` nodule
/// lowers to `Op{prim:"wild:echo"}` (no new Core-IR node, KC-3) and *executes* by dispatching through
/// the prim registry (the capability handle, §4.3), so L1-eval ≡ elaborate→L0-interp ≡ AOT agree on the
/// `wild`-backed value — the recorded `Empirical` basis closing DN-14 row 9's staged-execution gap.
#[test]
fn wild_ffi_execution_agrees_three_ways() {
    let prims = host_registry();
    let engine = BinaryTernarySwapEngine;
    let interp = Interpreter::new(host_registry(), Box::new(BinaryTernarySwapEngine));

    let src =
        "nodule std.sys.x @std-sys;\nfn main() => Binary{8} !{ffi} = wild { echo(0b1011_0010) };";
    let env = check_nodule(&parse(src).expect("parses")).expect("@std-sys wild checks");

    // M-720: the wild block elaborates to a host-dispatch Op (no Residual), in the `wild:` namespace.
    let node = elaborate(&env, "main").expect("wild elaborates to a host-dispatch Op (M-720)");
    assert!(
        matches!(&node, mycelium_core::Node::Op { prim, .. } if prim == "wild:echo"),
        "wild must lower to Op{{prim:\"wild:echo\"}} — no new Core-IR node (KC-3); got {node:?}"
    );

    // Path 1: L1-eval, with the host registry injected (the capability granted).
    let l1 = Evaluator::new(&env)
        .with_engines(host_registry(), Box::new(BinaryTernarySwapEngine))
        .call("main", vec![])
        .expect("L1-eval runs the wild host op");
    let l1 = l1.as_repr().expect("a repr result").clone();

    // Path 2: L0-interp. Path 3: AOT. Both dispatch the same `wild:echo` through the shared registry.
    let l0 = interp.eval(&node).expect("L0-interp runs the wild host op");
    let aot = mycelium_mlir::run(&node, &prims, &engine).expect("AOT runs the wild host op");

    assert_eq!(
        observable(&l1),
        observable(&l0),
        "wild: L1-eval vs L0-interp diverged"
    );
    assert_eq!(
        observable(&l0),
        observable(&aot),
        "wild: L0-interp vs AOT diverged"
    );

    // The dispatched value is the echoed input byte (deterministic — the differential's premise).
    assert_eq!(l0.repr(), &Repr::Binary { width: 8 });
    assert_eq!(
        l0.payload(),
        &Payload::Bits(vec![true, false, true, true, false, false, true, false]),
        "the host op echoes the input byte 0b1011_0010"
    );

    // The single shared M-210 checker validates each agreeing pair (never a vacuous pass).
    for (a, b, pair) in [(&l1, &l0, "L1↔interp"), (&l0, &aot, "interp↔AOT")] {
        assert_eq!(
            check(
                a,
                b,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "wild: the shared checker must validate the {pair} pair"
        );
    }
}

/// **Never-silent capability (G2; RFC-0028 §4.3).** The *default* registry grants no `wild:` op, so a
/// `wild` program run without the host capability is an explicit refusal — never a silent success or a
/// fabricated value. The refusal names the ungranted host op (both on L0-interp and L1-eval).
#[test]
fn an_ungranted_wild_host_op_is_an_explicit_refusal() {
    let src =
        "nodule std.sys.x @std-sys;\nfn main() => Binary{8} !{ffi} = wild { echo(0b1011_0010) };";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let node = elaborate(&env, "main").expect("elaborates to a host-dispatch Op");

    // L0-interp with the DEFAULT (no host) registry — an explicit, named refusal for the ungranted op.
    let err = Interpreter::default()
        .eval(&node)
        .expect_err("an ungranted wild op must refuse, never fabricate a value");
    let msg = err.to_string();
    assert!(
        msg.contains("echo") && msg.contains("not granted"),
        "the refusal must name the ungranted host capability `echo`; got: {msg}"
    );

    // L1-eval likewise refuses on the default registry — never silently runs.
    let l1_err = Evaluator::new(&env)
        .call("main", vec![])
        .expect_err("L1-eval must also refuse the ungranted host op");
    assert!(
        l1_err.to_string().contains("echo"),
        "L1-eval must surface the ungranted host op `echo`; got: {l1_err}"
    );
}

/// **Never-silent body shape (G2).** A `wild` body that is not a v0 host-call form (here a `let`) is an
/// explicit elaboration `Residual` — never a fabricated lowering (RFC-0028 §4.2). The program still
/// type-checks (the body is the opaque, audited escape — M-661); only its *lowering* refuses.
#[test]
fn a_wild_body_that_is_not_a_host_call_form_is_an_explicit_residual() {
    let src = "nodule std.sys.x @std-sys;\nfn main() => Binary{8} !{ffi} = wild { let a = 0b0000_0000 in a };";
    let env =
        check_nodule(&parse(src).expect("parses")).expect("the opaque body still type-checks");
    let err = elaborate(&env, "main")
        .expect_err("a non-host-call wild body must refuse to lower, never fabricate");
    assert!(
        err.to_string().contains("host-call form"),
        "the refusal must explain the v0 wild-body grammar (RFC-0028 §4.2); got: {err}"
    );
}

// --- DN-52 FLAG-1 / W5 freeze-ledger: Dense swap is an Explicit-Residual on ALL paths -----------
//
// DN-52 census (§5, FLAG-1 → RESOLVED): Dense swap targets are accepted by the checker (RFC-0002 /
// RFC-0005) but the standard three-way harness uses `BinaryTernarySwapEngine`, which only covers
// Binary↔Ternary. The resolution (elab.rs Expr::Swap arm, freeze-ledger W5) is to emit an explicit
// `Residual` in `elaborate` — so the elaborate path is consistent with L1-eval and L0-interp (which
// already refuse explicitly via `EvalError::UnsupportedSwap`). The DN-50 narrow gate holds: a Dense
// swap program is never "accepted but unrunnable silently" — every path is explicit (G2).
//
// Classification (Empirical — by test, not proof): Explicit-Residual.

/// **DN-52 FLAG-1 RESOLVED — Dense swap is an Explicit-Residual on all paths (W5/freeze-ledger).**
///
/// The checker accepts `swap(…, to: Dense{4, F32})` (RFC-0002/RFC-0005). After the elab.rs fix
/// (freeze-ledger W5), `elaborate` emits an explicit `Residual` — never an `Ok(Node::Swap{Dense})`
/// that every downstream runner (L1-eval, L0-interp, AOT) would then refuse with an inconsistent
/// error. L1-eval also refuses explicitly via `EvalError::UnsupportedSwap` (BinaryTernarySwapEngine).
/// Every path is consistent and explicit: no silent accept-but-unrunnable (G2/DN-50/DN-52 §4).
///
/// Classification: Explicit-Residual. Evidence: Empirical (this test).
#[test]
fn dense_swap_is_an_explicit_residual_on_all_paths() {
    // A checker-accepted program: Binary{8} → Dense{4, F32} swap.
    // `Dense{d, s}` is accepted by the checker as a swap target (RFC-0002/RFC-0005 §4.3).
    let src =
        "nodule d;\nfn main() => Dense{4, F32} = swap(0b1011_0010, to: Dense{4, F32}, policy: rt);";

    // The checker accepts this program — it is in the parsable-and-checked domain.
    let env = check_nodule(&parse(src).expect("Dense swap program must parse"))
        .expect("Dense swap must be accepted by the checker (RFC-0002/RFC-0005)");

    // Path A: `elaborate` must now return an explicit Residual (never Ok) — the elab.rs fix.
    // Before the fix, this returned Ok(Node::Swap{target: Repr::Dense{..}}) while every runner
    // refused; after the fix it returns Err(Residual{..}) for consistency (G2/DN-50).
    // Note: `elaborate` returns `Result<Node, ElabError>` (not L1Error) — matched directly.
    let elab_err = elaborate(&env, "main").expect_err(
        "elaborate must refuse Dense swap with an explicit Residual (DN-52 FLAG-1 → RESOLVED)",
    );
    assert!(
        matches!(elab_err, ElabError::Residual { .. }),
        "Dense swap elaborate error must be an explicit Residual, never an UnknownFn or other error; got: {elab_err}"
    );
    let msg = elab_err.to_string();
    assert!(
        msg.contains("Dense") || msg.contains("staged"),
        "the Residual message must mention Dense/staging (DN-52 FLAG-1); got: {msg}"
    );

    // Path B: L1-eval refuses explicitly via the BinaryTernarySwapEngine (UnsupportedSwap).
    let l1_err = Evaluator::new(&env)
        .call("main", vec![])
        .expect_err("L1-eval must refuse Dense swap explicitly (EvalError::UnsupportedSwap)");
    assert!(
        matches!(l1_err, L1Error::Kernel(_)),
        "L1-eval Dense swap error must be a kernel refusal (EvalError::UnsupportedSwap); got: {l1_err}"
    );

    // Both paths refuse explicitly and consistently — the DN-50 narrow gate holds for Dense (G2):
    // elaborate ⇒ Err(Residual) AND L1-eval ⇒ Err(Kernel(UnsupportedSwap)).
    // There is no path that silently accepts or produces a wrong-typed value.
    // Classification: Explicit-Residual (Empirical — this test). DN-52 §5 FLAG-1 → RESOLVED.
}

// --- DN-52 FLAG-2 / W5 freeze-ledger: cross-nodule three-way differential ----------------------
//
// DN-52 census (§5, FLAG-2 → RESOLVED): a two-nodule phylum (A exports a fn, B imports + calls it)
// was check-tested (`phylum.rs`) but NOT differential-tested. The question: does `elaborate` on
// nodule B's env (which already contains A's imported fns — `check_nodule_with` merges them at
// lines 1223-1224 of checkty.rs) run three-way (L1-eval ≡ L0-interp ≡ AOT)?
//
// Finding: YES — cross-nodule elaboration Runs. `PhylumEnv.nodules[i].1` is the merged Env for
// nodule i, with all `use`d functions from other nodules already present in `.fns`. Calling
// `elaborate(env_b, "main")` on nodule B's env therefore finds `helper` (from A) transparently.
//
// Classification (Empirical — by test, not proof): Runs.

/// **DN-52 FLAG-2 RESOLVED — cross-nodule three-way differential runs (W5/freeze-ledger).**
///
/// A two-nodule phylum: nodule `a` exports `pub fn helper`, nodule `b` imports it with `use a::*`
/// and calls it from `main`. `check_phylum` checks both; `PhylumEnv.nodules[1].1` (nodule B's env)
/// contains `helper` in its `.fns` map (merged by `check_nodule_with` — importable fns are part of
/// the merged env, RFC-0006 §4.3 / RFC-0007 §11). `elaborate(env_b, "main")` therefore finds the
/// imported fn and lowers the call, producing a closed L0 term that L0-interp and AOT can run.
/// L1-eval also runs the cross-nodule program directly (it evaluates on the merged env). All three
/// paths agree on the observable — the three-way differential extends to cross-nodule programs.
///
/// Classification: Runs. Evidence: Empirical (this test).
#[test]
fn cross_nodule_program_runs_three_way() {
    // A phylum with two nodules: A exports `helper`, B imports it from A and calls it in `main`.
    // RFC-0006 §4.3: `pub fn` in A is visible to B via `use a.*` (the glob import form).
    let src = "phylum app.cross\nnodule a;\npub fn helper(x: Binary{8}) => Binary{8} = not(x);\nnodule b;\nuse a.*;\nfn main() => Binary{8} = helper(0b1011_0010);";

    let phylum_env = check_phylum(&parse_phylum(src).expect("cross-nodule phylum must parse"))
        .expect("cross-nodule phylum must type-check");

    // Nodule B is the second nodule (index 1). Its env already contains `helper` from A (merged
    // by check_nodule_with — imports.fns merged at lines 1223-1224 of checkty.rs).
    let env_b = &phylum_env.nodules[1].1;
    assert!(
        env_b.fns.contains_key("helper"),
        "nodule B env must contain `helper` from A after check_phylum (RFC-0006 §4.3)"
    );

    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;

    // Path 1: L1-eval on nodule B's (merged) env — finds `helper` from A in env.fns.
    let l1 = Evaluator::new(env_b)
        .call("main", vec![])
        .expect("L1-eval must run the cross-nodule program (helper from A is in env_b.fns)");
    let l1_repr = l1
        .as_repr()
        .expect("L1 result must be a repr value")
        .clone();

    // Path 2: elaborate nodule B's env to L0, then run on the reference interpreter.
    let node = elaborate(env_b, "main")
        .expect("elaborate must run on the cross-nodule merged env (DN-52 FLAG-2 → Runs)");
    let l0 = interp
        .eval(&node)
        .expect("L0-interp must run the elaborated cross-nodule term");

    // Path 3: the same L0 term through the AOT env-machine.
    let aot = mycelium_mlir::run(&node, &prims, &engine)
        .expect("AOT must run the elaborated cross-nodule term");

    // All three paths must agree on the observable (repr + payload + guarantee).
    assert_eq!(
        observable(&l1_repr),
        observable(&l0),
        "cross-nodule: L1-eval vs L0-interp diverged"
    );
    assert_eq!(
        observable(&l0),
        observable(&aot),
        "cross-nodule: L0-interp vs AOT diverged"
    );

    // The shared M-210 checker validates each agreeing pair (never a vacuous pass).
    for (a, b, pair) in [(&l1_repr, &l0, "L1↔interp"), (&l0, &aot, "interp↔AOT")] {
        assert_eq!(
            check(
                a,
                b,
                RefinementRelation::ObservationalEquiv,
                Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "cross-nodule: the shared checker must validate the {pair} pair"
        );
    }

    // The value is not(0b1011_0010) = 0b0100_1101 (one's complement on Binary{8}).
    assert_eq!(l0.repr(), &Repr::Binary { width: 8 });
    assert_eq!(
        l0.payload(),
        &Payload::Bits(vec![false, true, false, false, true, true, false, true]),
        "cross-nodule: helper(0b1011_0010) = not(0b1011_0010) = 0b0100_1101"
    );
    // DN-52 §5 FLAG-2 → RESOLVED: cross-nodule three-way = Runs (Empirical).
}

// --- DN-58 §A/§B (M-817): fuse (repr + data) and reclaim run three-way (closes M-710) ----------
//
// M-817 closes the r4v execution residual. `fuse` now **runs** three-way (L1-eval ≡ L0-interp ≡ AOT):
//   - **repr** `fuse` lowers to the registered `fuse_join:binary` meet prim (bitwise-AND, the `Binary`
//     semilattice greatest-lower-bound) — the *same* prim on all three arms, carrying the canonical
//     `Derived{op:"fuse_join"}` provenance (DN-58 §A.5; RFC-0027 §10.6). [FLAG F-A1 → RESOLVED.]
//   - **data** `fuse` desugars (monomorphization) to the resolved `Fuse::join` call — an ordinary
//     inlined call that runs three-way (DN-58 §A.5), exercised by the `generic_corpus` (case 8) above.
// `reclaim(policy) { body }` lowers to its sequential reference (`Let{_ = policy, body}`) and runs
// three-way; the **real** RT7 supervision — restart cascade + `SupervisionRecord` EXPLAIN trail — is
// the runtime driver `mycelium_mlir::run_reclaim`, validated equal to the reference on success
// (`elaborate_reclaim` supplies the lazy policy/body nodes). [FLAG F-B1 → RESOLVED.]
// Classification: **Empirical** (a differential over these shapes + the M-713 property tests, not a
// mechanized theorem; VR-5). KC-3: no new L0 node — the meet reuses `Op`, the reference reuses `Let`.

/// **DN-58 §A.5 (M-817): `fuse` on `Binary` repr runs three-way (the semilattice meet).**
/// `fuse(0b1011_0010, 0b1100_1111)` = `0b1011_0010 & 0b1100_1111` = `0b1000_0010`. The L1 evaluator,
/// the `elaborate→L0-interp` path, and the AOT env-machine must all produce that value via the same
/// `fuse_join:binary` prim, with the canonical `Derived{op:"fuse_join"}` provenance.
#[test]
fn fuse_repr_differential_three_way_empirical() {
    let src = "nodule d;\nfn main() => Binary{8} = fuse(0b1011_0010, 0b1100_1111);";
    let env = check_nodule(&parse(src).expect("fuse(lit, lit) parses (DN-58 §A)"))
        .expect("fuse on Binary{8} type-checks (DN-58 §A)");

    // The expected meet: 0b1011_0010 & 0b1100_1111 = 0b1000_0010 (MSB-first: bits[0] is the MSB).
    let expected = Payload::Bits(vec![true, false, false, false, false, false, true, false]);

    // Path 1: L1-eval (now via the `fuse_join:binary` meet prim — M-817).
    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .expect("L1-eval runs the Binary fuse meet (DN-58 §A.3; M-817)");
    let l1 = l1.as_repr().expect("a repr fuse result").clone();
    assert_eq!(l1.repr(), &Repr::Binary { width: 8 });
    assert_eq!(
        l1.payload(),
        &expected,
        "fuse meet = 0b1000_0010 (bitwise-AND)"
    );

    // Path 2: elaborate → L0-interp (the registered `fuse_join:binary` prim — FLAG F-A1 resolved).
    let node = elaborate(&env, "main").expect("fuse elaborates to the fuse_join:binary Op (M-817)");
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let l0 = interp
        .eval(&node)
        .expect("L0-interp runs fuse_join:binary (M-817 — FLAG F-A1 resolved)");

    // Path 3: the same L0 term through the AOT env-machine.
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let aot =
        mycelium_mlir::run(&node, &prims, &engine).expect("AOT runs fuse_join:binary (M-817)");

    // All three agree on the observable (repr + payload + guarantee).
    assert_eq!(
        observable(&l0),
        observable(&l1),
        "L0-interp ≡ L1-eval (fuse meet)"
    );
    assert_eq!(
        observable(&aot),
        observable(&l1),
        "AOT ≡ L1-eval (fuse meet)"
    );

    // DN-58 §A.5 / RFC-0027 §10.6: the result carries the canonical `Derived{op:"fuse_join", …}` node
    // (the merge identity the δ-CRDT anti-entropy story reads), not the per-paradigm prim name.
    match l0.meta().provenance() {
        mycelium_core::Provenance::Derived { op, inputs } => {
            assert_eq!(
                *op,
                mycelium_core::operation_hash("fuse_join"),
                "fuse provenance is Derived{{op:\"fuse_join\"}} (DN-58 §A.5)"
            );
            assert_eq!(inputs.len(), 2, "fuse_join derives from both operands");
        }
        other => panic!("fuse result must be Derived{{op:\"fuse_join\"}}, got {other:?}"),
    }
}

/// **DN-58 §A.5 (M-817): a Data-type `fuse` runs three-way as the resolved `Fuse::join` call.**
/// This is the user-merge case the `prm` kickoff targets. `Flag`'s `join` is the absorbing-`On` OR (a
/// commutative/associative/idempotent join-semilattice); `fuse(On, Off)` = `join(On, Off)` = `On`. The
/// L1 evaluator runs on the **monomorphized** env (it has no trait dispatch — `eval_app`); `elaborate`
/// (which monomorphizes internally) and the AOT env-machine run the same resolved call. The broader
/// `generic_corpus` test exercises this shape in the uniform harness; this names the DoD value.
/// `Fuse` is a **built-in prelude trait** (M-965 F-A1 — no `trait Fuse` declaration here), and its
/// three semilattice laws are exhaustively checked at `impl` time over `Flag`'s finite domain
/// (M-965 F-A2) — this fixture's `join` passes all three (it is exactly two-element OR).
#[test]
fn fuse_data_differential_three_way_empirical() {
    let src = "nodule d;\ntype Flag = Off | On;\nimpl Fuse[Flag] for Flag { fn join(a: Flag, b: Flag) => Flag = match a { On => On, Off => b }; };\nfn main() => Flag = fuse(On, Off);";
    let env = check_nodule(&parse(src).expect("data fuse parses")).expect("data fuse type-checks");

    // The mono'd env desugars `fuse(On, Off)` to the resolved `join` call — the form L1-eval runs.
    let mono = monomorphize(&env, "main").expect("the Fuse program monomorphizes (DN-58 §A.5)");
    let registry = build_registry(&mono).expect("the mono'd data registry builds");

    // Path 1: L1-eval on the monomorphized env (the trait call is now a direct call).
    let l1 = Evaluator::new(&mono)
        .call("main", vec![])
        .expect("L1-eval runs the desugared Fuse::join (M-817)");
    let l1_core = l1
        .to_core(&mono, &registry)
        .expect("the Flag result is in the r3 data fragment");

    // Path 2 + 3: elaborate→L0-interp and AOT, both over the resolved-join L0 term.
    let node = elaborate(&env, "main").expect("data fuse elaborates to the resolved join call");
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let l0_core = interp
        .eval_core(&node)
        .expect("L0-interp runs the resolved Fuse::join");
    let aot_core = mycelium_mlir::run_core(
        &node,
        &PrimRegistry::with_builtins(),
        &BinaryTernarySwapEngine,
    )
    .expect("AOT runs the resolved Fuse::join");

    // All three agree on the merged datum.
    assert_eq!(l1_core, l0_core, "data fuse: L1-eval(mono) ≡ L0-interp");
    assert_eq!(l0_core, aot_core, "data fuse: L0-interp ≡ AOT");

    // Oracle: the merge value must be `On` (the absorbing element — join(On, Off) = On, DN-58 §A.1).
    // Compare against a bare `On` over the *same* `Flag` definition (its `#T#i` identity is content-
    // addressed by the type, so an identical `type Flag` yields the identical constructor identity).
    let on_src = "nodule d;\ntype Flag = Off | On;\nfn main() => Flag = On;";
    let on_env = check_nodule(&parse(on_src).expect("parses")).expect("checks");
    let on_node = elaborate(&on_env, "main").expect("a bare `On` elaborates");
    let on_core = interp.eval_core(&on_node).expect("the `On` oracle runs");
    assert_eq!(
        l0_core, on_core,
        "fuse(On, Off) = On (the absorbing element — DN-58 §A.1)"
    );

    // The shared M-210 checker validates the equivalence (a mislabeled lowering is explicit, not silent).
    assert_eq!(
        check_core(&l1_core, &aot_core),
        CheckVerdict::Validated {
            strength: GuaranteeStrength::Exact
        },
        "the shared checker validates the data-fuse L1↔AOT pair"
    );
}

/// **DN-58 §B (M-817): `reclaim(policy) { body }` runs three-way via its sequential reference.**
/// The trusted base lowers `reclaim` to `Let{_ = policy, body}` — evaluate the policy for its effect,
/// then yield the body — so L1-eval, `elaborate→L0-interp`, and the AOT env-machine all produce the
/// body's value. Here `reclaim(0b0000_0001) { not(0b1010_1010) }` yields `not(0b1010_1010)` = `0b0101_0101`.
#[test]
fn reclaim_sequential_reference_runs_three_way() {
    let src = "nodule d;\nfn main() => Binary{8} = reclaim(0b0000_0001) { not(0b1010_1010) };";
    let env = check_nodule(&parse(src).expect("reclaim parses (DN-58 §B)"))
        .expect("reclaim type-checks (the body type is the result type)");

    // not(0b1010_1010) = 0b0101_0101 (MSB-first).
    let expected = Payload::Bits(vec![false, true, false, true, false, true, false, true]);

    let l1 = Evaluator::new(&env)
        .call("main", vec![])
        .expect("L1-eval runs reclaim (policy for effect, then body — DN-58 §B)");
    let l1 = l1.as_repr().expect("a repr reclaim result").clone();
    assert_eq!(
        l1.payload(),
        &expected,
        "reclaim body = not(0b1010_1010) = 0b0101_0101"
    );

    let node =
        elaborate(&env, "main").expect("reclaim elaborates to Let{_ = policy, body} (M-817)");
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let l0 = interp
        .eval(&node)
        .expect("L0-interp runs the reclaim sequential reference");
    let aot = mycelium_mlir::run(
        &node,
        &PrimRegistry::with_builtins(),
        &BinaryTernarySwapEngine,
    )
    .expect("AOT runs the reclaim sequential reference");

    assert_eq!(
        observable(&l0),
        observable(&l1),
        "reclaim: L0-interp ≡ L1-eval"
    );
    assert_eq!(observable(&aot), observable(&l1), "reclaim: AOT ≡ L1-eval");
}

/// **DN-58 §B (M-817): the real RT7 supervision driver (`mycelium_mlir::run_reclaim`).**
/// `elaborate_reclaim` yields the (policy, body) as **lazy** L0 nodes; the driver runs the body under
/// `mycelium-std-runtime::supervise_with_restart`, threading the `SupervisionRecord` EXPLAIN trail.
/// - **Success:** a body that succeeds resolves on the first attempt — the value equals the sequential
///   reference and the EXPLAIN trace is empty (no supervision decision was needed).
/// - **Bounded failure:** a body that deterministically refuses (a fixed-width add overflow) is
///   restarted under the bounded cascade and then **escalates** — an explicit outcome with a recorded
///   trace, never an unbounded restart storm and never a silent drop (RT4/RT7; G2).
#[test]
fn reclaim_real_supervision_driver_dispatches_with_explain() {
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let intensity = mycelium_interp::RestartIntensity {
        max_restarts: 100,
        window_ticks: 1_000,
    };

    // (a) Success: the supervised observable equals the sequential reference, with an empty trace.
    let ok_src = "nodule d;\nfn main() => Binary{8} = reclaim(0b0000_0001) { not(0b1010_1010) };";
    let ok_env = check_nodule(&parse(ok_src).expect("parses")).expect("checks");
    let (policy, body) = mycelium_l1::elaborate_reclaim(&ok_env, "main")
        .expect("reclaim elaborates to its (policy, body) closed L0 programs");
    let run = mycelium_mlir::run_reclaim(
        &policy, &body, intensity, 2, &prims, &engine, 1_000_000, 1_000_000,
    )
    .expect("the supervised body succeeds (first attempt)");
    assert!(
        run.trace.is_empty(),
        "a first-attempt success records no supervision decisions (EXPLAIN)"
    );
    // The supervised value equals the sequential reference (elaborate → Let → L0-interp).
    let ref_node = elaborate(&ok_env, "main").expect("sequential reference elaborates");
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let ref_core = interp.eval_core(&ref_node).expect("the reference runs");
    assert_eq!(
        run.value, ref_core,
        "supervised success ≡ the sequential reference (DN-58 §B)"
    );

    // (b) Bounded failure: a deterministically-refusing body (add overflow) escalates with an EXPLAIN
    // trace — `add_u(0b1111_1111, 0b0000_0001)` overflows `Binary{8}` (never a silent wrap — G2).
    let bad_src =
        "nodule d;\nfn main() => Binary{8} = reclaim(0b0000_0001) { add_u(0b1111_1111, 0b0000_0001) };";
    let bad_env = check_nodule(&parse(bad_src).expect("parses")).expect("checks");
    let (policy, body) = mycelium_l1::elaborate_reclaim(&bad_env, "main")
        .expect("reclaim elaborates to its (policy, body) programs");
    let err = mycelium_mlir::run_reclaim(
        &policy, &body, intensity, 2, &prims, &engine, 1_000_000, 1_000_000,
    )
    .expect_err("a deterministically-failing body escalates under the bounded cascade");
    match err {
        mycelium_mlir::ReclaimError::Supervised { trace, .. } => assert!(
            !trace.is_empty(),
            "an escalation records its restart/escalation decisions (EXPLAIN — no black box)"
        ),
        other => panic!("expected a bounded supervised escalation, got: {other}"),
    }
}

/// **M-677 (RFC-0014 §4.5 I4) differential:** a budgeted fn with an `effect_budgets` ceiling
/// that is never overrun must produce the **same observable** from L1-eval as from
/// elaborate→L0-interp — the budget plumbing is meaning-preserving when under budget (NFR-7).
///
/// Guarantee: `Empirical` (a genuine three-way differential — L1-eval ≡ elaborate→L0-interp ≡
/// AOT env-machine with the effect ledger threaded — not a mechanized proof; VR-5). The AOT arm
/// threads an ample ledger via `run_core_with_effects`, the same observable-transparency basis as
/// the M-353 test above (an under-budget ledger perturbs nothing; the overrun refusal is tested on
/// the `mycelium-mlir` runtime path).
#[test]
fn m677_budgeted_fn_under_budget_is_differential_observable_equivalent() {
    // A straight-line (elaboratable) fn with `!{retry(<=1)}` — the ceiling will never be
    // exceeded because the L1-eval's fresh-ledger-per-call model primes retry=1 and consumes
    // 1 → 0 remaining on this single invocation. Observable: not(0b1011_0010) = 0b0100_1101.
    let src = "nodule d;\nfn main() => Binary{8} !{retry(<=1)} = not(0b1011_0010);";
    let env = check_nodule(&parse(src).expect("parses")).expect("checks");
    let node = elaborate(&env, "main").expect("elaborates — pure straight-line is in the fragment");

    // L1-eval path (with budget ledger wired, M-677).
    let l1_v = Evaluator::new(&env)
        .call("main", vec![])
        .expect("under budget — succeeds");
    let L1Value::Repr(ref l1_repr) = l1_v else {
        panic!("expected repr value from L1-eval")
    };

    // L0-interp path (no budget ledger — the reference semantics).
    let interp = Interpreter::new(
        PrimRegistry::with_builtins(),
        Box::new(BinaryTernarySwapEngine),
    );
    let l0_core = interp.eval_core(&node).expect("L0-interp runs");
    let mycelium_core::CoreValue::Repr(l0_repr) = l0_core else {
        panic!("expected repr from L0-interp")
    };

    assert_eq!(
        observable(l1_repr),
        observable(&l0_repr),
        "M-677: budgeted fn under budget must agree on the observable (L1-eval == L0-interp, NFR-7)"
    );

    // Path 3: the same L0 term through the AOT env-machine with the effect ledger threaded
    // (`run_core_with_effects`, M-353/M-677). An ample ledger never overruns, so threading it is
    // observable-transparent (NFR-7) — the AOT observable must match the other two paths.
    use mycelium_interp::{Budgets, EffectBudget};
    let prims = PrimRegistry::with_builtins();
    let engine = BinaryTernarySwapEngine;
    let mut budgets = Budgets::new().with(EffectBudget::Bytes(1 << 30));
    let aot_core = mycelium_mlir::run_core_with_effects(
        &node,
        &prims,
        &engine,
        1_000_000,
        1_000_000,
        &mut budgets,
    )
    .expect("AOT env-machine runs the budgeted fn under an ample ledger");
    let mycelium_core::CoreValue::Repr(aot_repr) = aot_core else {
        panic!("expected repr from the AOT env-machine")
    };
    assert_eq!(
        observable(&l0_repr),
        observable(&aot_repr),
        "M-677: budgeted fn under budget must agree on the observable (L0-interp == AOT, NFR-7)"
    );
}
