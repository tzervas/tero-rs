//! The **CSE** (common-subexpression elimination) pass (M-726; RFC-0029 §7.2) — EXPLAIN-able and
//! never-silent.
//!
//! When two bindings compute the **same pure expression** over the **same operands**, the second is
//! redundant: its uses are redirected to the first, and it becomes dead (removed by
//! [`crate::passes::dce`]). Recomputing a pure value gives the identical result, so the redirect is
//! observably transparent — the `Empirical` differential confirms it.
//!
//! # What is a CSE-able (pure, deterministic) expression here
//! Every L0 primitive is a pure, deterministic function of its operands (`mycelium_core::prim`), a
//! `Const` is a literal, a `Construct` builds a datum from its fields, and a `Swap` is a deterministic
//! function of its source + policy. So `Op` / `Const` / `Construct` / `Swap` are CSE candidates,
//! keyed by their structural shape. **`App` / `Lam` / `Fix` / `FixGroup` / `Match` / `Alias` are
//! conservatively excluded** — a closure value, a recursion, or a branch is not a value-equal
//! redundancy we deduplicate here (and `Alias` is already inlining's job). Two `Const`s are common
//! only when their `repr + payload + guarantee` are identical (the observable identity, NFR-7);
//! `Meta.provenance` is *not* part of value identity, so two literals that differ only in provenance
//! are still merged — a deliberate, recorded choice.
//!
//! # Never-silent
//! Every redirect emits a [`TransformRecord`] naming the redundant site, the canonical binding it was
//! merged into, and why. A binding that is *not* merged is left untouched (a no-op, recorded by its
//! absence). The first occurrence of each expression is always kept (the canonical definition); only
//! later duplicates are redirected — so CSE never removes the value, only the recomputation.

use std::collections::HashMap;

use mycelium_core::lower::Atom;

use super::{render_rhs, Pass, PassBinding, PassRhs, Program, TransformLog, TransformRecord};

/// Run the CSE pass: a pure `Program -> (Program, TransformLog)`. Within each block, the first binding
/// of every pure expression is canonical; a later binding of the *same* expression is redirected to
/// it (its uses point at the canonical name). Recurses into nested blocks. Every redirect is recorded.
#[must_use]
pub fn cse(program: &Program) -> (Program, TransformLog) {
    let mut log = TransformLog::new();
    let out = cse_block(program, &mut log);
    (out, log)
}

fn cse_block(program: &Program, log: &mut TransformLog) -> Program {
    // First recurse into nested blocks (so a closure/arm body is CSE'd on its own terms).
    let recursed: Vec<PassBinding> = program
        .bindings
        .iter()
        .map(|b| PassBinding {
            name: b.name.clone(),
            rhs: cse_rhs(&b.rhs, log),
            layout: b.layout,
        })
        .collect();

    // `canon`: structural key of a pure expression -> the canonical binding name that first computed
    // it. `subst`: redundant binding name -> canonical name (built as we discover duplicates).
    let mut canon: HashMap<String, Atom> = HashMap::new();
    let mut subst: HashMap<Atom, Atom> = HashMap::new();
    let mut out: Vec<PassBinding> = Vec::with_capacity(recursed.len());

    for b in &recursed {
        // Redirect this binding's own operands through any earlier CSE merges first (so a key built
        // from already-merged operands is canonical, e.g. `op f (cse'd-x)` matches `op f x`).
        let redirected = redirect_rhs(&b.rhs, &subst);

        if let Some(key) = cse_key(&redirected) {
            if let Some(existing) = canon.get(&key) {
                // Duplicate of an earlier pure expression — redirect this name to the canonical one.
                subst.insert(b.name.clone(), existing.clone());
                log.record(TransformRecord {
                    pass: Pass::Cse,
                    rule: "cse-merge",
                    site: b.name.render(),
                    before: render_rhs(&redirected),
                    after: format!("→ {} (canonical)", existing.render()),
                    reason: format!(
                        "identical pure expression already computed by {}; uses redirected, \
                         this recomputation is removed by DCE",
                        existing.render()
                    ),
                });
                // Do NOT push the redundant binding — its uses now point at the canonical one. (It is
                // value-equal, so this is observably transparent.) Keep `canon` pointing at the first.
                continue;
            }
            canon.insert(key, b.name.clone());
        }

        out.push(PassBinding {
            name: b.name.clone(),
            rhs: redirected,
            layout: b.layout,
        });
    }

    Program {
        bindings: out,
        result: subst.get(&program.result).cloned().unwrap_or_else(|| {
            // The result might itself name a CSE'd-away binding — redirect it too.
            program.result.clone()
        }),
    }
}

/// Recurse CSE into a RHS's nested blocks. Flat RHSs are returned unchanged (the merging happens at
/// the block level in [`cse_block`]).
fn cse_rhs(rhs: &PassRhs, log: &mut TransformLog) -> PassRhs {
    match rhs {
        PassRhs::Lam { param, body } => PassRhs::Lam {
            param: param.clone(),
            body: cse_block(body, log),
        },
        PassRhs::Fix { name, body } => PassRhs::Fix {
            name: name.clone(),
            body: cse_block(body, log),
        },
        PassRhs::FixGroup { defs, which } => PassRhs::FixGroup {
            defs: defs
                .iter()
                .map(|(n, b)| (n.clone(), cse_block(b, log)))
                .collect(),
            which: which.clone(),
        },
        PassRhs::Match {
            scrutinee,
            alts,
            default,
        } => PassRhs::Match {
            scrutinee: scrutinee.clone(),
            alts: alts
                .iter()
                .map(|alt| match alt {
                    super::PassAlt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => super::PassAlt::Ctor {
                        ctor: ctor.clone(),
                        binders: binders.clone(),
                        body: cse_block(body, log),
                    },
                    super::PassAlt::Lit { value, body } => super::PassAlt::Lit {
                        value: value.clone(),
                        body: cse_block(body, log),
                    },
                })
                .collect(),
            default: default.as_ref().map(|d| cse_block(d, log)),
        },
        other => other.clone(),
    }
}

/// The structural CSE key of a **pure, deterministic** RHS — `None` for an expression we do not
/// deduplicate (closures, recursion, branches, aliases). Two RHSs are common iff their keys are equal.
fn cse_key(rhs: &PassRhs) -> Option<String> {
    match rhs {
        // A literal: keyed by its observable identity (repr + payload + guarantee). Provenance is not
        // part of value identity (NFR-7), so two literals differing only in provenance still merge.
        PassRhs::Const(v) => Some(format!(
            "const|{:?}|{:?}|{:?}",
            v.repr(),
            v.payload(),
            v.meta().guarantee()
        )),
        PassRhs::Op { prim, args } => {
            let a: Vec<String> = args.iter().map(Atom::render).collect();
            Some(format!("op|{prim}|{}", a.join(",")))
        }
        PassRhs::Construct { ctor, args } => {
            let a: Vec<String> = args.iter().map(Atom::render).collect();
            Some(format!("construct|{ctor}|{}", a.join(",")))
        }
        PassRhs::Swap {
            src,
            target,
            policy,
        } => Some(format!(
            "swap|{}|{:?}|{}",
            src.render(),
            target,
            policy.digest()
        )),
        // Excluded from CSE (not a value-equal redundancy we deduplicate here).
        PassRhs::Alias(_)
        | PassRhs::App { .. }
        | PassRhs::Lam { .. }
        | PassRhs::Fix { .. }
        | PassRhs::FixGroup { .. }
        | PassRhs::Match { .. } => None,
    }
}

/// Redirect a RHS's operand atoms through the CSE substitution — including the **free** atoms a
/// nested block (closure/recursion/match body) captures, so a binding merged at the top level is
/// also redirected wherever a nested block reads it. Block-local binders shadow (never rewritten).
fn redirect_rhs(rhs: &PassRhs, subst: &HashMap<Atom, Atom>) -> PassRhs {
    let r = |a: &Atom| subst.get(a).cloned().unwrap_or_else(|| a.clone());
    match rhs {
        PassRhs::Op { prim, args } => PassRhs::Op {
            prim: prim.clone(),
            args: args.iter().map(r).collect(),
        },
        PassRhs::Construct { ctor, args } => PassRhs::Construct {
            ctor: ctor.clone(),
            args: args.iter().map(r).collect(),
        },
        PassRhs::Swap {
            src,
            target,
            policy,
        } => PassRhs::Swap {
            src: r(src),
            target: target.clone(),
            policy: policy.clone(),
        },
        PassRhs::App { func, arg } => PassRhs::App {
            func: r(func),
            arg: r(arg),
        },
        PassRhs::Alias(a) => PassRhs::Alias(r(a)),
        PassRhs::Lam { param, body } => PassRhs::Lam {
            param: param.clone(),
            body: redirect_block(body, subst, &[Atom::Named(param.clone())]),
        },
        PassRhs::Fix { name, body } => PassRhs::Fix {
            name: name.clone(),
            body: redirect_block(body, subst, &[Atom::Named(name.clone())]),
        },
        PassRhs::FixGroup { defs, which } => {
            let shadow: Vec<Atom> = defs.iter().map(|(n, _)| Atom::Named(n.clone())).collect();
            PassRhs::FixGroup {
                defs: defs
                    .iter()
                    .map(|(n, b)| (n.clone(), redirect_block(b, subst, &shadow)))
                    .collect(),
                which: which.clone(),
            }
        }
        PassRhs::Match {
            scrutinee,
            alts,
            default,
        } => PassRhs::Match {
            scrutinee: r(scrutinee),
            alts: alts
                .iter()
                .map(|alt| match alt {
                    super::PassAlt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => {
                        let shadow: Vec<Atom> =
                            binders.iter().map(|x| Atom::Named(x.clone())).collect();
                        super::PassAlt::Ctor {
                            ctor: ctor.clone(),
                            binders: binders.clone(),
                            body: redirect_block(body, subst, &shadow),
                        }
                    }
                    super::PassAlt::Lit { value, body } => super::PassAlt::Lit {
                        value: value.clone(),
                        body: redirect_block(body, subst, &[]),
                    },
                })
                .collect(),
            default: default.as_ref().map(|d| redirect_block(d, subst, &[])),
        },
        PassRhs::Const(_) => rhs.clone(),
    }
}

/// Redirect a nested block through the substitution, dropping any entry shadowed by the block's own
/// binders (so an inner binder of the same name as a merged outer atom is never rewritten).
fn redirect_block(program: &Program, subst: &HashMap<Atom, Atom>, shadowed: &[Atom]) -> Program {
    let local: HashMap<Atom, Atom> = subst
        .iter()
        .filter(|(k, _)| !shadowed.contains(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Program {
        bindings: program
            .bindings
            .iter()
            .map(|b| PassBinding {
                name: b.name.clone(),
                rhs: redirect_rhs(&b.rhs, &local),
                layout: b.layout,
            })
            .collect(),
        result: local
            .get(&program.result)
            .cloned()
            .unwrap_or_else(|| program.result.clone()),
    }
}
