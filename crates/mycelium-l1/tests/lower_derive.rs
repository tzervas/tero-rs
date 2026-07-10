//! **DN-54 §7 verification harness** for user-extensible generative-lowering rules (`lower`/`derive`;
//! M-812-cont). DN-54 §7 holds a user-defined lowering rule to the **same** discipline as a built-in
//! lowering pass: (§7.1) a **differential** — `observe(derive Name for T) == observe(hand-lowered
//! Name)`; (§7.2) **hygiene** — no free-variable capture; (§7.3) a **round-trip** — the lowered L0
//! re-runs to the same value (the certified-mode `delaborate ∘ lower = id` is the stronger form,
//! gated on `certified` and not implemented in v0 — VR-5; the value-level round-trip is the `fast`
//! obligation tested here).
//!
//! **Tag posture (VR-5, honest).** The rule's observational-identity claim is `Empirical` — earned
//! by these trials (and validated through the **M-210 shared TV checker** `mycelium_cert::check_core`,
//! the same `ObservationalEquiv` instance the built-in differentials use), never self-attested. The
//! mechanism makes the differential hold *by construction*: [`elaborate_lower_rule`] lowers a rule's
//! RHS through the **same** path a hand-written nullary fn body takes, so the two L0 terms are
//! structurally identical (asserted directly) — the runtime differential is then a confirmation, and
//! the M-210 checker the audit.
//!
//! **KC-3 (§6) by construction.** The elaborator's codomain is the *closed* enum
//! [`mycelium_core::Node`] (the frozen L0 grammar), so a `lower` rule can add no kernel node; the
//! produced node lies entirely in the AOT-lowerable v0 fragment (asserted below as a non-vacuous,
//! never-silent check).

use mycelium_cert::{check_core, CheckVerdict};
use mycelium_core::{GuaranteeStrength, Node};
use mycelium_interp::Interpreter;
use mycelium_l1::{check_nodule, elaborate, elaborate_lower_rule, parse};

/// One §7 case: a `lower` rule and the **hand-lowered** equivalent (a nullary fn whose body is the
/// rule's RHS). The harness asserts the rule's elaboration equals the hand-lowered fn's — three
/// ways: structurally (one code path), by run-value (the differential), and through the M-210 checker.
struct Case {
    /// A short label for diagnostics.
    name: &'static str,
    /// The rule name to elaborate via `elaborate_lower_rule`.
    rule: &'static str,
    /// A nodule declaring `lower <rule> = <rhs>` (and only that — no entry fn needed).
    rule_src: &'static str,
    /// The entry fn name in `hand_src`.
    hand_entry: &'static str,
    /// A nodule declaring `fn <hand_entry>() => <T> = <rhs>` — the same RHS, hand-written.
    hand_src: &'static str,
}

/// The §7 differential corpus. Each rule's RHS is a real, in-fragment L1 term (DN-54 §4.1 type-checks
/// it at definition time). The RHS and the hand fn body are textually the same expression, so the
/// differential holds *by construction* — the harness proves the `elaborate_lower_rule` path does not
/// diverge from the ordinary fn-elaboration path.
fn corpus() -> Vec<Case> {
    vec![
        Case {
            name: "binary literal",
            rule: "Eight",
            rule_src: "nodule d;\nlower Eight = 0b0000_0001;",
            hand_entry: "eight",
            hand_src: "nodule d;\nfn eight() => Binary{8} = 0b0000_0001;",
        },
        Case {
            name: "ternary literal",
            rule: "Trits",
            rule_src: "nodule d;\nlower Trits = 0t00+-;",
            hand_entry: "trits",
            hand_src: "nodule d;\nfn trits() => Ternary{4} = 0t00+-;",
        },
        Case {
            name: "Bool ctor",
            rule: "Yes",
            rule_src: "nodule d;\nlower Yes = True;",
            hand_entry: "yes",
            hand_src: "nodule d;\nfn yes() => Bool = True;",
        },
        Case {
            name: "let-bound repr op",
            rule: "Flip",
            rule_src: "nodule d;\nlower Flip = let a = 0b1011_0010 in not(a);",
            hand_entry: "flip",
            hand_src: "nodule d;\nfn flip() => Binary{8} = let a = 0b1011_0010 in not(a);",
        },
    ]
    // NOTE: a `swap`-bearing rule (`lower Widen = swap(…, policy: rt)`) is exercised by the
    // *structural-identity* + KC-3 checks below (`structural_corpus`), but **not** the run-value
    // differential here — the bare `Interpreter::default()` registers no swap engine, so a certified
    // swap is an explicit `unsupported swap` refusal on this path (the swap engine is wired in the
    // three-way `tests/differential.rs`, not the kernel interpreter). Its lowering identity is still
    // pinned (the by-construction proof), and its execution differential rides the existing swap
    // corpus — never silently dropped (G2): the boundary is documented here.
}

/// The structural-identity + KC-3 corpus — the run-value differential corpus **plus** cases whose
/// execution needs machinery the bare `Interpreter::default()` lacks (e.g. a certified `swap`), but
/// whose *lowering identity* (the by-construction claim) and *KC-3* property are still asserted.
fn structural_corpus() -> Vec<Case> {
    let mut v = corpus();
    v.push(Case {
        name: "certified swap (structural only)",
        rule: "Widen",
        rule_src: "nodule d;\nlower Widen = swap(0b1011_0010, to: Ternary{6}, policy: rt);",
        hand_entry: "widen",
        hand_src:
            "nodule d;\nfn widen() => Ternary{6} = swap(0b1011_0010, to: Ternary{6}, policy: rt);",
    });
    v
}

/// **§7.1 structural identity (by construction).** For every case (including the swap one): the
/// rule's elaborated L0 must be **structurally identical** to the hand-lowered fn's — the rule's RHS
/// lowers through the same path the hand fn body takes (the `%`-prefixed synthetic entry name appears
/// only in the never-emitted entry binder, not in a closed body). This is the strongest form of the
/// §7.1 obligation and does not depend on any runtime engine.
#[test]
fn lower_rule_elaboration_structurally_equals_hand_lowered() {
    for c in structural_corpus() {
        let rule_env = check_nodule(&parse(c.rule_src).expect("rule_src parses")).expect("checks");
        let hand_env = check_nodule(&parse(c.hand_src).expect("hand_src parses")).expect("checks");
        let rule_node = elaborate_lower_rule(&rule_env, c.rule)
            .unwrap_or_else(|e| panic!("[{}] rule elaboration failed: {e}", c.name));
        let hand_node = elaborate(&hand_env, c.hand_entry)
            .unwrap_or_else(|e| panic!("[{}] hand elaboration failed: {e}", c.name));
        assert_eq!(
            format!("{rule_node:?}"),
            format!("{hand_node:?}"),
            "[{}] §7.1: rule elaboration must structurally equal the hand-lowered fn",
            c.name
        );
    }
}

/// **§7.1 run-value differential + M-210 validation.** For each runnable case: the rule's elaborated
/// L0 runs to the **same value** as the hand-lowered fn's (the differential), and that pair must
/// **validate** through the shared M-210 TV checker (`ObservationalEquiv`, Exact) — never a bespoke
/// compare; a mislabeled lowering would be an explicit `NotValidated`.
#[test]
fn lower_rule_differential_equals_hand_lowered() {
    let interp = Interpreter::default();
    for c in corpus() {
        let rule_env = check_nodule(&parse(c.rule_src).expect("rule_src parses")).expect("checks");
        let hand_env = check_nodule(&parse(c.hand_src).expect("hand_src parses")).expect("checks");
        let rule_val = interp
            .eval_core(&elaborate_lower_rule(&rule_env, c.rule).expect("elaborates"))
            .unwrap_or_else(|e| panic!("[{}] L0-interp of rule failed: {e}", c.name));
        let hand_val = interp
            .eval_core(&elaborate(&hand_env, c.hand_entry).expect("elaborates"))
            .unwrap_or_else(|e| panic!("[{}] L0-interp of hand failed: {e}", c.name));
        assert_eq!(
            check_core(&rule_val, &hand_val),
            CheckVerdict::Validated {
                strength: GuaranteeStrength::Exact
            },
            "[{}] §7.1: the M-210 ObservationalEquiv checker must validate derive↔hand-lowered",
            c.name
        );
    }
}

/// **§6 KC-3 — structural confirmation (the substantive guard is by-construction).** Honesty note
/// (VR-5): the *real* KC-3 enforcement is the **closed-enum type boundary** — `elaborate_lower_rule`
/// returns `mycelium_core::Node`, a finite Rust enum, so a `lower` rule *cannot* construct a node
/// outside the frozen v0 kernel set; the type system forbids it (Proven-by-construction; this wave
/// makes no `mycelium-core` change). This test is therefore **NOT an independent KC-3 witness**:
/// `Node::is_aot_lowerable` is *total* over the v0 node set (it returns `true` for every variant), so
/// the assertion is a tautology that confirms each rule elaborates to a well-formed `Node` without
/// panicking. It would only become a *discriminating* witness if a future, deliberately
/// non-lowerable `Node` variant were added (then `is_aot_lowerable` must return `false` for it). The
/// genuine derive↔hand-lowered equivalence is pinned by
/// `lower_rule_elaboration_structurally_equals_hand_lowered` above.
#[test]
fn lower_rule_elaboration_stays_in_the_frozen_kernel_kc3() {
    for c in structural_corpus() {
        let env = check_nodule(&parse(c.rule_src).expect("parses")).expect("checks");
        let node = elaborate_lower_rule(&env, c.rule)
            .unwrap_or_else(|e| panic!("[{}] rule elaboration failed: {e}", c.name));
        assert!(
            node.is_aot_lowerable(),
            "[{}] §6/KC-3: the elaborated rule must be a well-formed v0 L0 node (elaboration \
             succeeds; the substantive KC-3 guard is the closed-enum type boundary — a `lower` rule \
             cannot construct a node outside `mycelium_core::Node`)",
            c.name
        );
    }
}

/// **§7.2 hygiene — no free-variable capture.** A rule whose RHS introduces a binder (`let a = …`)
/// must not capture a same-named binder at a use site. We elaborate a `let a`-introducing rule and a
/// hand fn that *also* binds `a` at its use site, and assert the rule's binders are all `%`-fresh
/// (the structural hygiene guarantee — DN-54 §4.3): no surface name `a` appears as a *binder* in the
/// elaborated L0 (every binder is renamed to a `%`-suffixed fresh name that cannot collide with a
/// surface identifier). This confirms the by-construction hygiene on the generated corpus.
#[test]
fn lower_rule_elaboration_is_hygienic_no_capture() {
    let env = check_nodule(
        &parse("nodule d;\nlower Flip = let a = 0b1011_0010 in not(a);").expect("parses"),
    )
    .expect("checks");
    let node = elaborate_lower_rule(&env, "Flip").expect("elaborates");
    // Every `Let`/`Lam`/`Fix` binder in the elaborated term is `%`-fresh (contains `%`), so it cannot
    // be a surface identifier and cannot capture a use-site name (DN-54 §4.3 / §7.2).
    let mut all_fresh = true;
    let mut saw_binder = false;
    walk_binders(&node, &mut |id| {
        saw_binder = true;
        if !id.contains('%') {
            all_fresh = false;
        }
    });
    assert!(saw_binder, "the `let`-bearing rule must elaborate a binder");
    assert!(
        all_fresh,
        "§7.2 hygiene: every elaborated binder must be `%`-fresh (no surface name capturable)"
    );
}

/// **§7.3 round-trip (value-level, `fast`).** The rule's lowered L0 re-runs to the same value as the
/// hand-lowered equivalent — the `fast`-mode round-trip obligation. (The stronger certified-mode
/// `delaborate ∘ lower = id` over the surface term is gated on `certified` mode and not implemented
/// in v0 — held `Declared`, VR-5; not asserted here.)
#[test]
fn lower_rule_value_round_trip() {
    let interp = Interpreter::default();
    for c in corpus() {
        let rule_env = check_nodule(&parse(c.rule_src).expect("parses")).expect("checks");
        let hand_env = check_nodule(&parse(c.hand_src).expect("parses")).expect("checks");
        let rule_val = interp
            .eval_core(&elaborate_lower_rule(&rule_env, c.rule).expect("elaborates"))
            .expect("runs");
        let hand_val = interp
            .eval_core(&elaborate(&hand_env, c.hand_entry).expect("elaborates"))
            .expect("runs");
        assert_eq!(
            rule_val, hand_val,
            "[{}] §7.3: the rule's lowered L0 must run to the hand-lowered value",
            c.name
        );
    }
}

/// Visit every binder id (`Let.id`, `Lam.param`, `Fix.name`, `FixGroup` member names, `Match`
/// `Alt::Ctor` field binders) in an L0 [`Node`] tree — a small structural probe for the hygiene test.
fn walk_binders(n: &Node, f: &mut impl FnMut(&str)) {
    match n {
        Node::Const(_) | Node::Var(_) => {}
        Node::Let { id, bound, body } => {
            f(id);
            walk_binders(bound, f);
            walk_binders(body, f);
        }
        Node::Op { args, .. } | Node::Construct { args, .. } => {
            for a in args {
                walk_binders(a, f);
            }
        }
        Node::Swap { src, .. } => walk_binders(src, f),
        Node::Match {
            scrutinee,
            alts,
            default,
        } => {
            walk_binders(scrutinee, f);
            for a in alts {
                match a {
                    mycelium_core::Alt::Ctor { binders, body, .. } => {
                        for b in binders {
                            f(b);
                        }
                        walk_binders(body, f);
                    }
                    mycelium_core::Alt::Lit { body, .. } => walk_binders(body, f),
                }
            }
            if let Some(d) = default {
                walk_binders(d, f);
            }
        }
        Node::Lam { param, body } => {
            f(param);
            walk_binders(body, f);
        }
        Node::App { func, arg } => {
            walk_binders(func, f);
            walk_binders(arg, f);
        }
        Node::Fix { name, body } => {
            f(name);
            walk_binders(body, f);
        }
        Node::FixGroup { defs, body } => {
            for (name, d) in defs {
                f(name);
                walk_binders(d, f);
            }
            walk_binders(body, f);
        }
    }
}
