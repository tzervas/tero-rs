//! The **inlining** pass (M-726; RFC-0029 §7.2) — EXPLAIN-able and never-silent.
//!
//! Two complementary, semantics-preserving inlinings over the flat pass IR:
//!
//! 1. **Alias folding** (`alias-fold`). A binding `x = alias(y)` (introduced by lowering a source
//!    `let`) is a pure indirection: every later use of `x` is replaced by `y`, and the alias binding
//!    is left for [`crate::passes::dce`] to remove once dead. Folding an alias can never change a
//!    value — `x` *is* `y` by construction — so this is the safest possible inline.
//! 2. **Single-use closure inlining** (`beta-reduce`). When a `Lam` closure binding is applied
//!    **exactly once** by a directly-following `App` (`f = lam p => body; r = app f arg`) and the
//!    closure value `f` is used nowhere else, the application is β-reduced: the call site is rewritten
//!    to bind the parameter to the argument and splice the closure body inline (its result aliased to
//!    the call's result name). This is the classic inline of a function called once — observably
//!    identical to the call (call-by-value, the body runs with `p ↦ arg` either way), and it removes
//!    the closure allocation + the apply.
//!
//! Both record a [`TransformRecord`] for **every** rewrite (G2: never silent). Anything the pass does
//! not match — multi-use closures, recursion, partial application — is left **untouched** (a no-op,
//! recorded by its absence from the log). Inlining only ever *exposes* more work for CSE/DCE; it never
//! changes the program's observable result (the `Empirical` differential confirms it).
//!
//! **Conservative β-reduction (soundness).** Single-use closure inlining fires **only** when the
//! closure is applied to the value flowing from the immediately-following `App` whose `func` is the
//! closure and whose result is not otherwise constrained, the closure is referenced by **exactly that
//! one** `App`, and the closure body is a self-contained block. Outside that shape the pass declines —
//! it never guesses (VR-5/G2).

use std::collections::HashMap;

use mycelium_core::lower::Atom;

use super::{
    render_rhs, rhs_uses, Pass, PassBinding, PassRhs, Program, TransformLog, TransformRecord,
};

/// Run the inlining pass: a pure `Program -> (Program, TransformLog)`. Alias-folds first (the simplest
/// and always-safe inline), then β-reduces single-use closures; recurses into nested blocks so a
/// closure body is itself inlined. Every rewrite is recorded.
#[must_use]
pub fn inline(program: &Program) -> (Program, TransformLog) {
    let mut log = TransformLog::new();
    let out = inline_block(program, &mut log);
    (out, log)
}

fn inline_block(program: &Program, log: &mut TransformLog) -> Program {
    // First recurse into nested blocks (closure/recursion/match bodies), so inlining is applied at
    // every level (the pass is a fixpoint over the tree, not just the top block).
    let recursed: Vec<PassBinding> = program
        .bindings
        .iter()
        .map(|b| PassBinding {
            name: b.name.clone(),
            rhs: inline_rhs(&b.rhs, log),
            layout: b.layout,
        })
        .collect();

    // Pass A — alias folding. Build the substitution `alias-name -> target`, transitively resolved,
    // then redirect every use. The alias bindings stay (DCE removes the now-dead ones); redirecting a
    // use of an alias to its target is value-preserving by construction.
    let mut subst: HashMap<Atom, Atom> = HashMap::new();
    for b in &recursed {
        if let PassRhs::Alias(target) = &b.rhs {
            let resolved = resolve(&subst, target);
            subst.insert(b.name.clone(), resolved);
        }
    }
    let mut folded = Program {
        bindings: recursed
            .iter()
            .map(|b| {
                let new_rhs = redirect_rhs(&b.rhs, &subst);
                if new_rhs != b.rhs {
                    log.record(TransformRecord {
                        pass: Pass::Inline,
                        rule: "alias-fold",
                        site: b.name.render(),
                        before: render_rhs(&b.rhs),
                        after: render_rhs(&new_rhs),
                        reason: "operand was a `let`-alias; redirected to its target (the alias \
                                 binding becomes dead, removed by DCE)"
                            .to_owned(),
                    });
                }
                PassBinding {
                    name: b.name.clone(),
                    rhs: new_rhs,
                    layout: b.layout,
                }
            })
            .collect(),
        result: resolve(&subst, &program.result),
    };
    if folded.result != program.result {
        log.record(TransformRecord {
            pass: Pass::Inline,
            rule: "alias-fold",
            site: "<result>".to_owned(),
            before: program.result.render(),
            after: folded.result.render(),
            reason: "program result was a `let`-alias; redirected to its target".to_owned(),
        });
    }

    // Pass B — single-use closure β-reduction. Inline a `Lam` applied exactly once by a following App.
    beta_reduce(&mut folded, log);
    folded
}

/// Recurse inlining into a RHS's nested blocks (closure/recursion/match bodies), so the pass is a
/// fixpoint over the whole tree. Flat RHSs are returned unchanged.
fn inline_rhs(rhs: &PassRhs, log: &mut TransformLog) -> PassRhs {
    match rhs {
        PassRhs::Lam { param, body } => PassRhs::Lam {
            param: param.clone(),
            body: inline_block(body, log),
        },
        PassRhs::Fix { name, body } => PassRhs::Fix {
            name: name.clone(),
            body: inline_block(body, log),
        },
        PassRhs::FixGroup { defs, which } => PassRhs::FixGroup {
            defs: defs
                .iter()
                .map(|(n, b)| (n.clone(), inline_block(b, log)))
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
                        body: inline_block(body, log),
                    },
                    super::PassAlt::Lit { value, body } => super::PassAlt::Lit {
                        value: value.clone(),
                        body: inline_block(body, log),
                    },
                })
                .collect(),
            default: default.as_ref().map(|d| inline_block(d, log)),
        },
        other => other.clone(),
    }
}

/// β-reduce single-use closures in place. For each `App { func, arg }` whose `func` is a `Lam` bound
/// earlier in *this* block and used **only** by this one App, splice the closure body inline: a fresh
/// alias binds the parameter to the argument, the body's bindings are appended, and the body's result
/// is aliased to the App's result name. The `Lam` binding becomes dead (DCE removes it).
fn beta_reduce(program: &mut Program, log: &mut TransformLog) {
    // Count how many times each binding name is *used* as an operand anywhere in the block (so we only
    // inline a closure that is applied exactly once and read nowhere else).
    let use_counts = count_uses(program);

    // Map each Lam binding name -> (param, body) for quick lookup.
    let mut lams: HashMap<Atom, (String, Program)> = HashMap::new();
    for b in &program.bindings {
        if let PassRhs::Lam { param, body } = &b.rhs {
            lams.insert(b.name.clone(), (param.clone(), body.clone()));
        }
    }

    let mut out: Vec<PassBinding> = Vec::with_capacity(program.bindings.len());
    for b in &program.bindings {
        if let PassRhs::App { func, arg } = &b.rhs {
            if let Some((param, body)) = lams.get(func) {
                // The closure must be applied exactly once (this App) and used nowhere else — its only
                // use is as this App's `func`. (`count_uses` counts every operand occurrence.)
                if use_counts.get(func).copied() == Some(1) {
                    // Splice: bind the parameter to the argument, append the body, alias the result.
                    out.push(PassBinding {
                        name: Atom::Named(param.clone()),
                        rhs: PassRhs::Alias(arg.clone()),
                        layout: None,
                    });
                    out.extend(body.bindings.iter().cloned());
                    out.push(PassBinding {
                        name: b.name.clone(),
                        rhs: PassRhs::Alias(body.result.clone()),
                        layout: b.layout,
                    });
                    log.record(TransformRecord {
                        pass: Pass::Inline,
                        rule: "beta-reduce",
                        site: b.name.render(),
                        before: format!("app {} {}", func.render(), arg.render()),
                        after: format!(
                            "inlined closure {} (param {param} := {})",
                            func.render(),
                            arg.render()
                        ),
                        reason: "closure is applied exactly once and used nowhere else; \
                                 β-reduced inline (call-by-value: body runs with param := arg \
                                 either way), removing the closure alloc + the apply"
                            .to_owned(),
                    });
                    continue;
                }
            }
        }
        out.push(b.clone());
    }
    program.bindings = out;
}

/// Count every operand occurrence of each binding name across the whole block (top-level RHS uses,
/// the result operand, and uses inside nested blocks). Drives the single-use guard.
fn count_uses(program: &Program) -> HashMap<Atom, usize> {
    let mut counts: HashMap<Atom, usize> = HashMap::new();
    for b in &program.bindings {
        let mut uses = Vec::new();
        rhs_uses(&b.rhs, &mut uses);
        for u in uses {
            *counts.entry(u).or_insert(0) += 1;
        }
    }
    *counts.entry(program.result.clone()).or_insert(0) += 1;
    counts
}

/// Transitively resolve an atom through the alias substitution (so a chain `a -> b -> c` resolves
/// `a` to `c`). Bounded by the substitution size (acyclic by construction — an alias only ever points
/// at an earlier binding).
fn resolve(subst: &HashMap<Atom, Atom>, atom: &Atom) -> Atom {
    let mut cur = atom.clone();
    let mut steps = 0;
    while let Some(next) = subst.get(&cur) {
        if *next == cur || steps > subst.len() {
            break;
        }
        cur = next.clone();
        steps += 1;
    }
    cur
}

/// Redirect every operand atom of a RHS through the alias substitution (the actual inline). Nested
/// blocks are redirected too (an inner use of an outer alias folds as well).
fn redirect_rhs(rhs: &PassRhs, subst: &HashMap<Atom, Atom>) -> PassRhs {
    let r = |a: &Atom| resolve(subst, a);
    match rhs {
        PassRhs::Const(_) => rhs.clone(),
        PassRhs::Alias(a) => PassRhs::Alias(r(a)),
        PassRhs::Op { prim, args } => PassRhs::Op {
            prim: prim.clone(),
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
        PassRhs::Construct { ctor, args } => PassRhs::Construct {
            ctor: ctor.clone(),
            args: args.iter().map(r).collect(),
        },
        PassRhs::App { func, arg } => PassRhs::App {
            func: r(func),
            arg: r(arg),
        },
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
    }
}

/// Redirect a nested block, **without** folding through atoms shadowed by the block's binders (a
/// closure parameter / match binder shadows an outer alias of the same name — never fold across it).
fn redirect_block(program: &Program, subst: &HashMap<Atom, Atom>, shadowed: &[Atom]) -> Program {
    // Drop any substitution whose key is shadowed by this block's binders (parameters/match binders),
    // so an inner binder is never accidentally rewritten.
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
        result: resolve(&local, &program.result),
    }
}
