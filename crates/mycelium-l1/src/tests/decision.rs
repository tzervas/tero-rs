use crate::checkty::{CtorInfo, DataInfo, Ty, Width};
use crate::decision::*;
use crate::usefulness::Pat;
use std::collections::BTreeMap;

/// Project the sub-value of `value` at `occurrence` (`value` is a concrete [`Pat`] — no `Wild`).
fn project<'a>(value: &'a Pat, occurrence: &[usize]) -> &'a Pat {
    let mut v = value;
    for &j in occurrence {
        match v {
            Pat::Ctor(_, subs) => v = &subs[j],
            _ => panic!("occurrence {occurrence:?} steps into a non-constructor value"),
        }
    }
    v
}

/// Evaluate a decision tree against a concrete value (a `Pat` with no `Wild`), returning the arm to
/// run — the executable semantics that lets a test check the tree against the reference matcher.
/// **Test-only by design:** production does not execute the `Tree` directly — the elaborator emits
/// it as nested L0 `Match` nodes which the L0 interpreter/AOT run. This interpreter exists only to
/// *verify* the compiler against the reference matcher, not to run programs.
fn eval_tree(tree: &Tree, value: &Pat) -> Option<usize> {
    match tree {
        Tree::Leaf(a) => Some(*a),
        Tree::Fail => None,
        Tree::Switch {
            occurrence,
            cases,
            default,
        } => {
            let sub = project(value, occurrence);
            let matched = cases.iter().find(|(h, _)| head_matches(h, sub));
            match matched {
                Some((_, subtree)) => eval_tree(subtree, value),
                None => default.as_deref().and_then(|d| eval_tree(d, value)),
            }
        }
    }
}

fn head_matches(head: &Head, value: &Pat) -> bool {
    match (head, value) {
        (Head::Ctor(n, a), Pat::Ctor(m, subs)) => n == m && *a == subs.len(),
        (Head::Lit(k), Pat::Lit(j)) => k == j,
        _ => false,
    }
}

fn nat_registry() -> BTreeMap<String, DataInfo> {
    let mut m = BTreeMap::new();
    m.insert(
        "Nat".to_owned(),
        DataInfo {
            name: "Nat".to_owned(),
            params: vec![],
            ctors: vec![
                CtorInfo {
                    name: "Z".to_owned(),
                    fields: vec![],
                },
                CtorInfo {
                    name: "S".to_owned(),
                    fields: vec![Ty::Data("Nat".to_owned(), vec![])],
                },
            ],
        },
    );
    m
}

fn ctor(n: &str, subs: Vec<Pat>) -> Pat {
    Pat::Ctor(n.to_owned(), subs)
}

/// The reference matcher: the first arm whose pattern matches `value` (a concrete `Pat`).
fn reference(arms: &[Pat], value: &Pat) -> Option<usize> {
    arms.iter().position(|p| matches_value(p, value))
}
fn matches_value(pat: &Pat, value: &Pat) -> bool {
    match (pat, value) {
        (Pat::Wild, _) => true,
        (Pat::Lit(k), Pat::Lit(j)) => k == j,
        (Pat::Ctor(n, ps), Pat::Ctor(m, vs)) => {
            n == m && ps.len() == vs.len() && ps.iter().zip(vs).all(|(p, v)| matches_value(p, v))
        }
        _ => false,
    }
}

/// Build the Nat value `n` as a concrete `Pat` (Z, S(Z), S(S(Z)), …).
fn nat(n: usize) -> Pat {
    let mut v = ctor("Z", vec![]);
    for _ in 0..n {
        v = ctor("S", vec![v]);
    }
    v
}

/// The tree agrees with the reference matcher on every Nat value up to a depth — the property
/// that earns the compilation (a wrong column choice / specialization would diverge here).
fn assert_agrees(arms: &[Pat], tree: &Tree, depth: usize) {
    for n in 0..=depth {
        let v = nat(n);
        assert_eq!(
            eval_tree(tree, &v),
            reference(arms, &v),
            "tree vs reference diverged on S^{n}(Z)"
        );
    }
}

#[test]
fn flat_nat_match_compiles_and_agrees() {
    let t = nat_registry();
    // Z => 0 | S(_) => 1  (exhaustive, flat).
    let arms = vec![ctor("Z", vec![]), ctor("S", vec![Pat::Wild])];
    let matrix: Vec<Vec<Pat>> = arms.iter().cloned().map(|p| vec![p]).collect();
    let tree = compile(
        &t,
        &matrix,
        &[0, 1],
        &[vec![]],
        &[Ty::Data("Nat".into(), vec![])],
    )
    .expect("compiles within the RFC-0041 recursion budget");
    // Root switches on the whole scrutinee with both constructors covered → no default.
    match &tree {
        Tree::Switch {
            occurrence,
            cases,
            default,
        } => {
            assert_eq!(occurrence, &Vec::<usize>::new());
            assert_eq!(cases.len(), 2);
            assert!(
                default.is_none(),
                "complete data signature needs no default"
            );
        }
        other => panic!("expected a switch, got {other:?}"),
    }
    assert_agrees(&arms, &tree, 4);
}

#[test]
fn nested_nat_match_compiles_and_agrees() {
    let t = nat_registry();
    // Z => 0 | S(Z) => 1 | S(S(_)) => 2  (exhaustive, nested) — the tree must reach arm 2 only
    // for depth ≥ 2, which exercises the S→S child occurrence [0].
    let arms = vec![
        ctor("Z", vec![]),
        ctor("S", vec![ctor("Z", vec![])]),
        ctor("S", vec![ctor("S", vec![Pat::Wild])]),
    ];
    let matrix: Vec<Vec<Pat>> = arms.iter().cloned().map(|p| vec![p]).collect();
    let tree = compile(
        &t,
        &matrix,
        &[0, 1, 2],
        &[vec![]],
        &[Ty::Data("Nat".into(), vec![])],
    )
    .expect("compiles within the RFC-0041 recursion budget");
    assert_agrees(&arms, &tree, 5);
    // Spot-check the arm selection directly.
    assert_eq!(eval_tree(&tree, &nat(0)), Some(0));
    assert_eq!(eval_tree(&tree, &nat(1)), Some(1));
    assert_eq!(eval_tree(&tree, &nat(2)), Some(2));
    assert_eq!(eval_tree(&tree, &nat(3)), Some(2));
}

#[test]
fn first_matching_arm_wins_on_overlap() {
    let t = nat_registry();
    // S(_) => 0 | S(Z) => 1 | Z => 2 : arm 1 is shadowed by the broader arm 0 (redundant), and
    // arm 2 makes the set exhaustive. The tree must still pick arm 0 for S(Z), matching the
    // first-match semantics the reference encodes.
    let arms = vec![
        ctor("S", vec![Pat::Wild]),
        ctor("S", vec![ctor("Z", vec![])]),
        ctor("Z", vec![]),
    ];
    let matrix: Vec<Vec<Pat>> = arms.iter().cloned().map(|p| vec![p]).collect();
    let tree = compile(
        &t,
        &matrix,
        &[0, 1, 2],
        &[vec![]],
        &[Ty::Data("Nat".into(), vec![])],
    )
    .expect("compiles within the RFC-0041 recursion budget");
    assert_eq!(eval_tree(&tree, &nat(1)), Some(0)); // S(Z) → first arm S(_), never the shadowed arm 1
    assert_agrees(&arms, &tree, 4);
}

#[test]
fn literal_match_with_default_compiles_and_switches_with_a_default() {
    let t = nat_registry();
    // 0b0 => 0 | _ => 1 over Binary{1}: a literal switch that always carries a default (the
    // value domain is open).
    let arms = [Pat::Lit("b:0".into()), Pat::Wild];
    let matrix: Vec<Vec<Pat>> = arms.iter().cloned().map(|p| vec![p]).collect();
    let tree = compile(
        &t,
        &matrix,
        &[0, 1],
        &[vec![]],
        &[Ty::Binary(Width::Lit(1))],
    )
    .expect("compiles within the RFC-0041 recursion budget");
    match &tree {
        Tree::Switch { cases, default, .. } => {
            assert_eq!(cases.len(), 1);
            assert!(
                default.is_some(),
                "open literal domain always needs a default"
            );
        }
        other => panic!("expected a switch, got {other:?}"),
    }
    assert_eq!(eval_tree(&tree, &Pat::Lit("b:0".into())), Some(0));
    assert_eq!(eval_tree(&tree, &Pat::Lit("b:1".into())), Some(1)); // falls to default
}

/// A registry with a `Unit` type (one nullary ctor `U`) and a `Wide` type whose sole constructor `W`
/// has `n` `Unit` fields (RFC-0041 §4.7 W6). `Unit` fields are non-wildcard heads, so `compile_rows`
/// must test each column — driving the arity→depth spine (an all-wildcard `W(_,…)` would instead
/// short-circuit to a `Leaf`).
fn wide_unit_registry(n: usize) -> BTreeMap<String, DataInfo> {
    let mut m = nat_registry();
    m.insert(
        "Unit".to_owned(),
        DataInfo {
            name: "Unit".to_owned(),
            params: vec![],
            ctors: vec![CtorInfo {
                name: "U".to_owned(),
                fields: vec![],
            }],
        },
    );
    m.insert(
        "Wide".to_owned(),
        DataInfo {
            name: "Wide".to_owned(),
            params: vec![],
            ctors: vec![CtorInfo {
                name: "W".to_owned(),
                fields: vec![Ty::Data("Unit".to_owned(), vec![]); n],
            }],
        },
    );
    m
}

/// A modest wide-arity constructor compiles fine (well within the depth budget). Pairs with the
/// `#[ignore]`d boundary witness below to show the arity spine is only a problem near the ceiling.
#[test]
fn w6_modest_wide_arity_compiles() {
    let n = 100usize;
    let t = wide_unit_registry(n);
    let arms = vec![vec![Pat::Ctor("W".to_owned(), vec![ctor("U", vec![]); n])]];
    let col = vec![Ty::Data("Wide".to_owned(), vec![])];
    let tree = compile(&t, &arms, &[0], &[vec![]], &col)
        .expect("a 100-field constructor compiles within the RFC-0041 recursion budget");
    assert!(!has_reachable_fail(&tree));
}

/// **RFC-0041 §4.7 (W6): the wide-tuple asymmetry, test-witnessed for the decision-tree twin.** Like
/// `usefulness::useful`, `compile_rows` tests one column per level, so an **arity-N** constructor with
/// non-wildcard fields drives ~N levels of recursion on its width spine. At/over the depth ceiling it
/// false-refuses with a **clean, never-silent** [`mycelium_workstack::BudgetError::DepthExceeded`] —
/// verified *not* a host-stack overflow (run on the production 256 MiB deep stack via
/// [`mycelium_stack::with_deep_stack`]). `#[ignore]`d because the copying specialization is `O(N²)` at
/// the 4095-field boundary (seconds); run deliberately (RFC-0041 §5 census-test convention). A future
/// §4.7 conversion of the width spine to a work-step loop would flip this `Err` to `Ok`.
#[test]
#[ignore = "W6: O(N^2) at the 4095-field arity boundary — documented asymmetry, run deliberately"]
fn w6_wide_arity_compile_refuses() {
    mycelium_stack::with_deep_stack(|| {
        let n = 4095usize;
        let t = wide_unit_registry(n);
        let arms = vec![vec![Pat::Ctor("W".to_owned(), vec![ctor("U", vec![]); n])]];
        let col = vec![Ty::Data("Wide".to_owned(), vec![])];
        let r = compile(&t, &arms, &[0], &[vec![]], &col);
        assert!(
            matches!(
                r,
                Err(mycelium_workstack::BudgetError::DepthExceeded { limit: 4096 })
            ),
            "arity {n} exceeds the depth budget and must refuse never-silently \
             (clean DepthExceeded, not a SIGABRT), got {r:?}"
        );
    });
}
