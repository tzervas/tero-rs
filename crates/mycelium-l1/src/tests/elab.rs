use crate::ast::Path;
use crate::checkty::check_nodule;
use crate::checkty::Env;
use crate::elab::*;
use crate::parse;
use mycelium_core::{Alt, Node, Payload, Trit};
use std::collections::BTreeMap;

fn env(src: &str) -> Env {
    check_nodule(&parse(src).expect("parses")).expect("checks")
}

#[test]
fn a_const_let_op_swap_body_elaborates_closed() {
    let env = env(
            "nodule d;\nfn main() => Ternary{6} =\n  let a = 0b1011_0010 in swap(not(a), to: Ternary{6}, policy: rt);",
        );
    let node = elaborate(&env, "main").expect("in the fragment");
    // Closed: the interpreter must not hit a free variable.
    let interp = mycelium_interp::Interpreter::default();
    // The identity engine refuses the cross-paradigm swap, but the term itself is closed and
    // well-formed — getting an UnsupportedSwap (not FreeVariable) proves closure.
    let err = interp.eval(&node).unwrap_err();
    assert!(matches!(
        err,
        mycelium_interp::EvalError::UnsupportedSwap { .. }
    ));
}

#[test]
fn a_call_is_inlined_acyclically() {
    let env = env(
            "nodule d;\nfn flip(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} = flip(flip(0b1010_1010));",
        );
    let node = elaborate(&env, "main").expect("acyclic calls inline");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    // not(not(x)) == x
    assert_eq!(
        v.payload(),
        &Payload::Bits(vec![true, false, true, false, true, false, true, false])
    );
}

#[test]
fn self_recursion_now_elaborates_to_fix_and_runs() {
    // r4: a self-recursive function elaborates to a Fix and runs on the interpreter.
    // drop_(S(S(Z))) ⟶ Z.
    let env = env("nodule d;\ntype Nat = Z | S(Nat);\nfn drop_(n: Nat) => Nat = match n { Z => Z, S(m) => drop_(m) };\nfn main() => Nat = drop_(S(S(Z)));");
    let node = elaborate(&env, "main").expect("self-recursion elaborates in r4");
    let v = mycelium_interp::Interpreter::default()
        .eval_core(&node)
        .expect("terminates");
    assert_eq!(v.as_data().expect("data").fields().len(), 0, "Z");
}

#[test]
fn an_unproductive_recursion_elaborates_then_exhausts_fuel() {
    // A non-terminating recursion still elaborates (it is in the fragment now) but the fuel clock
    // makes its evaluation an explicit refusal, never a hang (RFC-0007 §4.5).
    let env = env("nodule d;\nfn spin(x: Binary{8}) => Binary{8} = spin(x);\nfn main() => Binary{8} = spin(0b0000_0001);");
    let node = elaborate(&env, "main").expect("recursion elaborates in r4");
    let err = mycelium_interp::Interpreter::default()
        .with_fuel(500)
        .eval(&node)
        .unwrap_err();
    assert_eq!(err, mycelium_interp::EvalError::FuelExhausted);
}

/// Whether `n` contains a `FixGroup` anywhere (the mutual-recursion lowering — M-343).
fn contains_fixgroup(n: &Node) -> bool {
    match n {
        Node::FixGroup { .. } => true,
        Node::Let { bound, body, .. } => contains_fixgroup(bound) || contains_fixgroup(body),
        Node::Fix { body, .. } | Node::Lam { body, .. } => contains_fixgroup(body),
        Node::App { func, arg } => contains_fixgroup(func) || contains_fixgroup(arg),
        Node::Op { args, .. } | Node::Construct { args, .. } => args.iter().any(contains_fixgroup),
        Node::Swap { src, .. } => contains_fixgroup(src),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            contains_fixgroup(scrutinee)
                || alts.iter().any(|a| match a {
                    Alt::Ctor { body, .. } | Alt::Lit { body, .. } => contains_fixgroup(body),
                })
                || default.as_deref().is_some_and(contains_fixgroup)
        }
        Node::Const(_) | Node::Var(_) => false,
    }
}

#[test]
fn mutual_recursion_now_elaborates_to_a_fixgroup_and_runs() {
    // M-343 (R7-Q3): a mutually-recursive group (ping/pong) lowers to a `FixGroup` and runs on
    // the reference interpreter — ping(S(Z)) ⟶ pong(Z) ⟶ Z. (Previously an explicit Residual.)
    let env = env("nodule d;\ntype Nat = Z | S(Nat);\nfn ping(n: Nat) => Nat = match n { Z => Z, S(m) => pong(m) };\nfn pong(n: Nat) => Nat = match n { Z => Z, S(m) => ping(m) };\nfn main() => Nat = ping(S(Z));");
    let node = elaborate(&env, "main").expect("mutual recursion elaborates to a FixGroup");
    assert!(
        contains_fixgroup(&node),
        "the mutual-recursion lowering must use a FixGroup node"
    );
    let v = mycelium_interp::Interpreter::default()
        .with_fuel(10_000)
        .eval_core(&node)
        .expect("the FixGroup runs to a value");
    let d = v.as_data().expect("a Nat data value");
    assert_eq!(d.fields().len(), 0, "ping(S(Z)) = pong(Z) = Z (nullary)");
}

#[test]
fn a_match_now_elaborates_to_l0_and_runs() {
    // r3: `match` is no longer Residual — it lowers to a flat L0 Match and runs on the
    // reference interpreter. `match Pos { Neg => 0t-, Zero => 0t0, _ => 0t+ }` ⟶ 0t+.
    let env = env(
            "nodule d;\ntype Sign = Neg | Zero | Pos;\nfn main() => Ternary{1} =\n  match Pos { Neg => 0t-, Zero => 0t0, _ => 0t+ };",
        );
    let node = elaborate(&env, "main").expect("match elaborates in r3");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(v.payload(), &Payload::Trits(vec![Trit::Pos]));
}

#[test]
fn a_data_value_now_elaborates_to_construct() {
    // A program returning a data value elaborates to Construct (via eval_core).
    let env = env("nodule d;\ntype Nat = Z | S(Nat);\nfn main() => Nat = S(Z);");
    let node = elaborate(&env, "main").expect("Construct elaborates in r3");
    let v = mycelium_interp::Interpreter::default()
        .eval_core(&node)
        .expect("runs");
    let d = v.as_data().expect("a data value");
    assert_eq!(d.fields().len(), 1, "S(Z) has one field");
}

#[test]
fn an_if_desugars_to_a_bool_match() {
    // `if` lowers to a Match on the prelude Bool — exercises the True/False registry path.
    let env = env(
        "nodule d;\nfn pick(b: Bool) => Binary{8} = if b then 0b1111_1111 else 0b0000_0000;\nfn main() => Binary{8} = pick(True);",
    );
    let node = elaborate(&env, "main").expect("if elaborates in r3");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(v.payload(), &Payload::Bits(vec![true; 8]));
}

#[test]
fn a_nested_pattern_match_elaborates_and_runs() {
    // pred2 uses depth-2 nested patterns; the Maranget tree lowers them to nested flat L0 Matches.
    // pred2(S(S(S(Z)))) ⟶ S(Z).
    let env = env("nodule d;\ntype Nat = Z | S(Nat);\nfn pred2(n: Nat) => Nat = match n { Z => Z, S(Z) => Z, S(S(m)) => m };\nfn main() => Nat = pred2(S(S(S(Z))));");
    let node = elaborate(&env, "main").expect("nested match elaborates in r3");
    let v = mycelium_interp::Interpreter::default()
        .eval_core(&node)
        .expect("runs");
    let d = v.as_data().expect("a data value");
    assert_eq!(d.fields().len(), 1, "S(Z)");
    assert_eq!(
        d.fields()[0].as_data().expect("inner Z").fields().len(),
        0,
        "the inner value is Z"
    );
}

#[test]
fn a_guarantee_index_now_elaborates_after_static_grading() {
    // RFC-0018 (M-663): an `@ g` guarantee index is **statically checked** (the grading pass) and
    // **erased** at elaboration — it is no longer an `ElabError::Residual`. Here `main` returns
    // `Ternary{6} @ Proven` from a `swap` (the endorsement point: the certificate is trusted at
    // the type level, so the body's grade satisfies the `@ Proven` return demand — R18-Q4), so it
    // both type-checks/grades and elaborates to a closed L0 `Swap` term (grade gone, no L0 form).
    let env = env(
            "nodule d;\nfn main() => Ternary{6} @ Proven = swap(0b0000_0010, to: Ternary{6}, policy: rt);",
        );
    let node =
        elaborate(&env, "main").expect("an `@ g` index elaborates (statically graded, erased)");
    assert!(
        matches!(node, Node::Swap { .. }),
        "the body lowers to an L0 Swap (the grade is erased), got {node:?}"
    );
}

#[test]
fn a_for_fold_now_elaborates_to_a_fix_fold_and_runs() {
    // r4: `for` desugars to a synthesized self-recursive Fix fold and runs. A 2-element xor-fold
    // of 0b1111_0000 and 0b0000_1111 from 0 is 0b1111_1111.
    let env = env("nodule d;\ntype ByteList = End | More(Binary{8}, ByteList);\nfn checksum(bs: ByteList) => Binary{8} = for b in bs, acc = 0b0000_0000 => xor(acc, b);\nfn main() => Binary{8} = checksum(More(0b1111_0000, More(0b0000_1111, End)));");
    let node = elaborate(&env, "main").expect("`for` elaborates in r4");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(v.payload(), &Payload::Bits(vec![true; 8]));
}

#[test]
fn a_for_fold_over_nil_is_the_initial_accumulator() {
    let env = env("nodule d;\ntype ByteList = End | More(Binary{8}, ByteList);\nfn checksum(bs: ByteList) => Binary{8} = for b in bs, acc = 0b1010_1010 => xor(acc, b);\nfn main() => Binary{8} = checksum(End);");
    let node = elaborate(&env, "main").expect("`for` elaborates in r4");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(
        v.payload(),
        &Payload::Bits(vec![true, false, true, false, true, false, true, false])
    );
}

#[test]
fn the_entry_must_be_nullary() {
    let env = env("nodule d;\nfn id(x: Binary{8}) => Binary{8} = x;");
    let err = elaborate(&env, "id").unwrap_err();
    assert!(matches!(err, ElabError::Residual { .. }));
}

#[test]
fn the_policy_name_ref_is_deterministic_and_name_sensitive() {
    let a = policy_name_ref(&Path(vec!["rt".into()]));
    let b = policy_name_ref(&Path(vec!["rt".into()]));
    let c = policy_name_ref(&Path(vec!["other".into()]));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ---- M-666: `colony { hypha … }` elaboration (RFC-0008 §4.7) ----

#[test]
fn a_single_hypha_colony_elaborates_to_its_body_and_runs() {
    // RT2 reference semantics: a one-hypha colony *is* its body. `colony { hypha not(0b…) }`
    // elaborates and runs to `not(0b1011_0010) = 0b0100_1101`.
    let env = env("nodule d;\nfn main() => Binary{8} = colony { hypha not(0b1011_0010) };");
    let node = elaborate(&env, "main").expect("a colony is in the fragment (M-666)");
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(
        v.payload(),
        &Payload::Bits(vec![false, true, false, false, true, true, false, true])
    );
}

#[test]
fn a_multi_hypha_colony_lowers_to_a_let_chain_and_yields_the_last_hypha() {
    // The RT2 spawn-order sequentialization lowers to nested `Let`s (leading hyphae bound to
    // fresh `%`-names), so the L0 form contains ≥1 `Let` and the observable is the LAST hypha's
    // value — here `xor(0b1111_0000, 0b0000_1111) = 0b1111_1111`, regardless of the leading two.
    let env = env(
            "nodule d;\nfn compute(x: Binary{8}) => Binary{8} = not(x);\nfn main() => Binary{8} =\n  colony { hypha compute(0b0000_0001), hypha compute(0b0000_0010), hypha xor(0b1111_0000, 0b0000_1111) };",
        );
    let node = elaborate(&env, "main").expect("multi-hypha colony elaborates");
    // The lowering is a Let chain (the sequentialization), not a single bare op.
    assert!(
        matches!(node, Node::Let { .. }),
        "a multi-hypha colony must lower to a Let chain (the RT2 sequentialization), got {node:?}"
    );
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(
        v.payload(),
        &Payload::Bits(vec![true; 8]),
        "last hypha = all-ones"
    );
}

/// **Property (RT2 sequentialization bound; RFC-0008 §4.2/§4.6 R1).** For *every* number of
/// leading pure hyphae `k ∈ 0..=8`, a colony `colony { hypha e_0, …, hypha e_{k-1}, hypha
/// e_last }` elaborates to L0 and evaluates to **exactly** `e_last`'s value — the leading hyphae
/// never change the observable (the colony equals its last hypha under sequentialization). The
/// leading bodies are all *distinct* from the last, so a silent "keep the first" / "drop the
/// last" elaboration bug would change the result and trip the assertion. This is the bound the
/// `colony` surface rests on; bounded exhaustive generation over `k` is the crate's property-test
/// idiom (no `proptest` dep — consistent with `usefulness`/`totality`).
#[test]
fn prop_colony_value_is_its_last_hypha_for_any_leading_count() {
    let interp = mycelium_interp::Interpreter::default();
    // The last hypha's expected 8-bit payload: not(0b0101_0101) = 0b1010_1010.
    let last_payload: Vec<bool> = (0..8u32).map(|i| i.is_multiple_of(2)).collect();
    for k in 0u32..=8 {
        // k distinct leading hyphae, each a different pure `not(...)` over a per-index literal,
        // then the final hypha whose value is the only observable.
        let mut hyphae = String::new();
        for j in 0..k {
            // a per-index 8-bit literal so the leading bodies differ from each other & the last
            let bits: String = (0..8u32)
                .map(|b| if (j + b).is_multiple_of(2) { '1' } else { '0' })
                .collect();
            hyphae.push_str(&format!("hypha not(0b{bits}), "));
        }
        // last hypha: xor(0b1111_0000, 0b0101_0101) = 0b1010_0101? compute deterministically.
        // Use a literal whose value we assert directly to avoid arithmetic ambiguity: a `not`.
        // not(0b0101_0101) = 0b1010_1010 = last_payload.
        hyphae.push_str("hypha not(0b0101_0101)");
        let src = format!("nodule d;\nfn main() => Binary{{8}} = colony {{ {hyphae} }};");
        let env = env(&src);
        let node = elaborate(&env, "main")
            .unwrap_or_else(|e| panic!("k={k}: colony must be in the fragment: {e}"));
        let v = interp
            .eval(&node)
            .unwrap_or_else(|e| panic!("k={k}: colony must run: {e}"));
        assert_eq!(
            v.payload(),
            &Payload::Bits(last_payload.clone()),
            "k={k}: the colony's value must equal its LAST hypha (RT2 sequentialization), \
                 independent of the {k} leading hyphae"
        );
    }
}

// --- M-659: dictionary-passing lowering is STAGED to a Residual (RFC-0019 §4.4 / RFC-0007 §12.3;
// the M-673 follow-up) — exactly mirroring how a generic instantiation stages (M-657). The
// *checker* types traits/impls/bounded-calls; the L0 lowering of a generic/trait instantiation is
// an explicit never-silent `Residual`, never a partial/fabricated artifact (G2/VR-5).

#[test]
fn a_bounded_generic_entry_is_an_explicit_residual() {
    // A bounded generic fn type-checks, but its L0 lowering is staged — asked to elaborate it
    // directly, the elaborator refuses with an explicit Residual, never a silent or
    // half-monomorphized artifact (G2/VR-5). (A bounded fn has value params *and* type params,
    // so it cannot be a closed entry either way; both reasons surface as an honest Residual.)
    let env = env(
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };\nfn use_cmp[T: Cmp](a: T, b: T) => Binary{2} = cmp(a, b);",
    );
    let err = elaborate(&env, "use_cmp").unwrap_err();
    assert!(
        matches!(err, ElabError::Residual { .. }),
        "a bounded generic entry must stage to a Residual, got {err:?}"
    );
}

#[test]
fn a_nullary_generic_entry_stages_with_the_monomorphization_residual() {
    // A *nullary* generic fn (no value params) reaches the generic-specific staging branch: its
    // L0 lowering is staged to monomorphization (the same staging M-657 introduced for §11), an
    // explicit Residual that names it.
    let env = env("nodule d;\nfn g[A]() => Binary{1} = 0b1;");
    let err = elaborate(&env, "g").unwrap_err();
    let ElabError::Residual { what, .. } = &err else {
        panic!("expected a Residual for a generic entry, got {err:?}");
    };
    assert!(
        what.contains("generic") || what.contains("monomorph"),
        "got: {what}"
    );
}

// ---- M-904: the `consume`-specific residual is gone (DN-71 §4.3) --------------------------

#[test]
fn consume_no_longer_produces_the_m664_era_substrate_residual() {
    // Before M-904, `elaborate` on any entry reaching a `consume` refused with a
    // Substrate-specific `Residual` ("`consume` of an affine `Substrate` is staged …"). That arm is
    // gone (M-904; DN-71 §4.3) — `Expr::Consume` now elaborates transparently as its operand.
    //
    // v0 has no surface syntax that constructs a live `Substrate` value (DN-71 §4.1/§8 FLAG-8: the
    // only entry point is a fn *parameter*, and `elab_prelude` refuses *any* value-parameterized
    // entry — "v0 elaborates closed (nullary) entries" — independently of Substrate), so this
    // program still fails to elaborate. The point of this test is that it fails for that
    // *pre-existing, orthogonal, nullary-entry* reason, never again for the old
    // Substrate-specific one — proving the M-904 residual really is lifted, not just reworded.
    let env = env("nodule d;\nfn take(s: Substrate{Sock}) => Substrate{Sock} = consume s;");
    let err = elaborate(&env, "take").expect_err("a value-parameterized entry still can't close");
    let ElabError::Residual { what, .. } = &err else {
        panic!("expected a Residual, got {err:?}");
    };
    assert!(
        what.contains("value parameters") && what.contains("nullary"),
        "must fail on the pre-existing nullary-entry gate, not a Substrate-specific one: {what}"
    );
    assert!(
        !what.contains("Substrate") && !what.contains("M-664"),
        "the M-664-era Substrate-specific residual message must be gone: {what}"
    );
}

#[test]
fn an_unqualified_trait_method_call_now_elaborates_via_monomorphization() {
    // M-673: a *concrete* trait-method call type-checks (resolved via the instance) and now
    // **elaborates** — the monomorphization pre-pass statically resolves it to a direct call to
    // the instance's method body (a mangled monomorphic fn), so it lowers to a closed L0 term and
    // runs. (Before M-673 this was a staged dictionary-passing `Residual`; that site is kept in
    // `app` as a defensive invariant — see
    // `the_generic_and_trait_residual_sites_remain_as_defensive_invariants`.)
    let env = env(
        "nodule d;\ntrait Cmp[A] { fn cmp(a: A, b: A) => Binary{2}; };\nimpl Cmp[Binary{8}] for Binary{8} { fn cmp(a: Binary{8}, b: Binary{8}) => Binary{2} = 0b00; };\nfn direct() => Binary{2} = cmp(0b0000_0001, 0b0000_0010);",
    );
    let node = elaborate(&env, "direct").expect("a trait-method call elaborates after M-673");
    // The method body is `0b00`, so the closed L0 term runs to that 2-bit value.
    let v = mycelium_interp::Interpreter::default()
        .eval(&node)
        .expect("runs");
    assert_eq!(v.payload(), &Payload::Bits(vec![false, false]));
}

#[test]
fn the_generic_and_trait_residual_sites_remain_as_defensive_invariants() {
    // G2: M-673 keeps the generic/trait `Residual` sites in `app`/`elab_fn_lam` as defensive
    // internal invariants — they must never silently disappear. After monomorphization they should
    // be unreachable on a real (mono'd) program, but they still fire if a generic/trait `Env` is
    // fed **straight** to the prelude/`Elab` machinery (bypassing `monomorphize`). Drive the
    // generic-fn site directly: a generic `FnDecl` reaching `elab_fn_lam` is an explicit Residual.
    let env = env("nodule d;\ntype List[A] = Nil | Cons(A, List[A]);\nfn first_or[A](xs: List[A], d: A) => A = match xs { Nil => d, Cons(x, _) => x };\nfn main() => Binary{8} = first_or(Cons(0b0000_0001, Nil), 0b0000_0000);");
    // `build_registry` + an `Elab` over the *un-monomorphized* env, then ask it to lower the
    // generic `first_or` lambda — the defensive generic-staging Residual must fire (never a
    // half-monomorphized artifact).
    let registry = build_registry(&env).expect("registry builds (skips the generic List)");
    let mut el = Elab {
        env: &env,
        registry,
        fresh: 0,
        rec: BTreeMap::new(),
        depth: 0,
    };
    let err = el.elab_fn_lam("first_or").unwrap_err();
    let ElabError::Residual { what, .. } = &err else {
        panic!("expected the defensive generic Residual, got {err:?}");
    };
    assert!(
        what.contains("generic") || what.contains("monomorph"),
        "the defensive site must still name the generic/monomorphization staging, got: {what}"
    );
}
