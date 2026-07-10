use std::collections::BTreeMap;

use crate::ast::{BaseType, Expr, FnDecl, FnSig, Param, Path, TypeRef, Vis, WidthRef};
use crate::totality::*;

/// A `consume(consume(… consume(x) …))` nest `depth` deep — `Expr::Consume` is a bare `Box<Expr>`
/// wrapper (no other fields), so it is the simplest way to build a pathologically-nested `Expr`
/// directly (bypassing the parser's `MAX_EXPR_DEPTH` surface cap — a direct AST is the way to
/// exercise this pass's *own* budget, mirroring `checkty`'s `deep_not` fixture).
fn deep_consume(depth: usize) -> Expr {
    let mut e = Expr::Path(Path(vec!["x".to_string()]));
    for _ in 0..depth {
        e = Expr::Consume(Box::new(e));
    }
    e
}

fn fn_named(name: &str, body: Expr) -> FnDecl {
    FnDecl {
        vis: Vis::Private,
        thaw: false,
        tier: None,
        sig: FnSig {
            name: name.to_string(),
            params: vec![],
            value_params: vec![Param {
                name: "x".to_string(),
                ty: TypeRef {
                    base: BaseType::Binary(WidthRef::Lit(1)),
                    guarantee: None,
                },
            }],
            ret: TypeRef {
                base: BaseType::Binary(WidthRef::Lit(1)),
                guarantee: None,
            },
            effects: vec![],
            effect_budgets: BTreeMap::new(),
        },
        body,
    }
}

#[test]
fn walk_expr_trips_the_depth_budget_cleanly_and_just_under_it_succeeds() {
    // `walk_expr` is the low-level, `pub(crate)` traversal: its production callers
    // (`classify_all`/`checkty`/`elab`) always run it on `mycelium_stack`'s deep worker stack, so a
    // direct unit-test call must do the same — `MAX_WALK_DEPTH` (4096) levels of match-heavy
    // recursion comfortably exceeds a default ~2 MiB test-thread stack even though it is nowhere
    // near the checker's measured ~24,600-level physical ceiling on the deep stack.
    mycelium_stack::with_deep_stack(|| {
        // Just under the budget: the shared traversal completes.
        let ok_body = deep_consume((MAX_WALK_DEPTH - 5) as usize);
        let mut visited = 0usize;
        assert!(
            walk_expr(&ok_body, &mut |_| visited += 1).is_ok(),
            "just under the budget should walk to completion"
        );
        assert!(visited > 0, "the visitor must have run");

        // Past the budget: a clean, explicit refusal — never a host-stack overflow (banked guard 4).
        let bad_body = deep_consume((MAX_WALK_DEPTH + 50) as usize);
        let err = walk_expr(&bad_body, &mut |_| {}).expect_err("past the budget must refuse");
        assert_eq!(err.limit, MAX_WALK_DEPTH);
        assert!(
            err.to_string().contains("recursion-depth budget"),
            "expected the explicit depth-budget refusal, got: {err}"
        );
    });
}

#[test]
fn classify_all_propagates_the_walk_depth_budget_instead_of_overflowing() {
    // `classify_all` runs `collect_calls` (built on `walk_expr`) over every fn body; a
    // pathologically-nested (but otherwise ordinary, non-recursive) body must surface the same
    // explicit `WalkDepthExceeded` refusal, never a host-stack overflow and never a silently wrong
    // `Totality::Partial` verdict manufactured from a resource limit (G2/VR-5).
    let mut fns = BTreeMap::new();
    fns.insert(
        "deep".to_string(),
        fn_named("deep", deep_consume((MAX_WALK_DEPTH + 50) as usize)),
    );
    let err = classify_all(&fns).expect_err("a pathologically-nested body must refuse cleanly");
    assert_eq!(err.limit, MAX_WALK_DEPTH);

    // Control: comfortably under the budget, the same shape classifies normally (no recursion here,
    // so every fn is trivially `Total`).
    let mut ok_fns = BTreeMap::new();
    ok_fns.insert(
        "shallow".to_string(),
        fn_named("shallow", deep_consume((MAX_WALK_DEPTH - 5) as usize)),
    );
    let result = classify_all(&ok_fns).expect("comfortably under the budget must succeed");
    assert_eq!(result.get("shallow"), Some(&Totality::Total));
}

#[test]
fn descend_walk_trips_the_depth_budget_on_a_pathologically_nested_self_recursive_body() {
    // A genuinely self-recursive function (`f` calls `f`) forces `classify_all` into the
    // mutual/self-descent search (`group_descends` → `assignment_descends` → `descend_walk`), which
    // carries its own separate recursive walk over the body — bound it the same way `walk_expr` is
    // bound (M-674): nest the recursive call itself under a pathologically deep `Consume` wrapper so
    // `descend_walk`'s own recursion (not just `collect_calls`'s) is what trips the budget.
    let recursive_call = Expr::App {
        head: Box::new(Expr::Path(Path(vec!["f".to_string()]))),
        args: vec![Expr::Path(Path(vec!["x".to_string()]))],
    };
    let mut body = recursive_call;
    for _ in 0..(MAX_WALK_DEPTH as usize + 50) {
        body = Expr::Consume(Box::new(body));
    }
    let mut fns = BTreeMap::new();
    fns.insert("f".to_string(), fn_named("f", body));
    let err = classify_all(&fns)
        .expect_err("a pathologically-nested self-recursive body must refuse cleanly");
    assert_eq!(err.limit, MAX_WALK_DEPTH);
}
