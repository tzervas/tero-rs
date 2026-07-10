//! The **DCE** (dead-code elimination) pass (M-726; RFC-0029 §7.2) — EXPLAIN-able and never-silent.
//!
//! A binding is **dead** if its name is never read — not by the program result, not by any later
//! binding's operands, and not (transitively) by any binding that is itself live. Removing a dead
//! binding cannot change the observable result: nothing depends on its value. DCE removes exactly the
//! dead set and records every removal.
//!
//! # Liveness (a backward, fixpoint reachability)
//! Seed the live set with the program **result** atom. Then sweep the ordered binding list **right to
//! left**: a binding whose name is live pulls in every atom its RHS reads (its operands and the free
//! atoms its nested blocks capture). One pass reaches the fixpoint, because in ANF an operand always
//! precedes its use — so a live binding's operands are earlier and are marked before the sweep reaches
//! them.
//!
//! # Effects / never-silent
//! This kernel's bindings are **pure** (a primitive is a deterministic function; a `Const`/`Lam` is a
//! value; a `Swap`/`Construct` is pure) and v0 L0 has **no effect node** (KC-3), so a dead binding
//! truly has no observable side effect to preserve — dropping it is sound. (Were an effectful node
//! ever added, DCE would have to keep effectful-but-unused bindings; that is the recorded purity
//! precondition, not silently assumed away — G2.) Every removal emits a [`TransformRecord`]; a binding
//! that survives is kept verbatim (no silent change).

use std::collections::HashSet;

use mycelium_core::lower::Atom;

use super::{
    render_rhs, rhs_uses, Pass, PassAlt, PassBinding, PassRhs, Program, TransformLog,
    TransformRecord,
};

/// Run the DCE pass: a pure `Program -> (Program, TransformLog)`. Computes the live set (backward
/// reachability from the result), keeps the live bindings in order, removes the dead ones, and records
/// each removal. Recurses into the nested blocks of surviving bindings (a live closure body is DCE'd
/// on its own terms).
#[must_use]
pub fn dce(program: &Program) -> (Program, TransformLog) {
    let mut log = TransformLog::new();
    let out = dce_block(program, &mut log);
    (out, log)
}

fn dce_block(program: &Program, log: &mut TransformLog) -> Program {
    let live = live_set(program);

    let mut kept: Vec<PassBinding> = Vec::with_capacity(program.bindings.len());
    for b in &program.bindings {
        if live.contains(&b.name) {
            // Live — keep it, but recurse DCE into its nested blocks (a live closure/arm body may
            // still contain its own dead bindings).
            kept.push(PassBinding {
                name: b.name.clone(),
                rhs: dce_rhs(&b.rhs, log),
                layout: b.layout,
            });
        } else {
            // Dead — remove it, recorded (never silent).
            log.record(TransformRecord {
                pass: Pass::Dce,
                rule: "drop-dead",
                site: b.name.render(),
                before: render_rhs(&b.rhs),
                after: "<removed>".to_owned(),
                reason: "binding's value is never read (not by the result nor any live binding); \
                         pure, so removing it preserves the observable result"
                    .to_owned(),
            });
        }
    }

    Program {
        bindings: kept,
        result: program.result.clone(),
    }
}

/// Compute the live set: every binding name transitively reachable (backward) from the program
/// result. One right-to-left sweep suffices — in ANF an operand always precedes its use, so a live
/// binding's operands are earlier and are marked before the sweep reaches them. `rhs_uses` already
/// includes the free atoms a nested block captures, so a closure that reads an outer binding keeps it
/// alive.
fn live_set(program: &Program) -> HashSet<Atom> {
    let mut live: HashSet<Atom> = HashSet::new();
    live.insert(program.result.clone());
    for b in program.bindings.iter().rev() {
        if live.contains(&b.name) {
            let mut uses = Vec::new();
            rhs_uses(&b.rhs, &mut uses);
            for u in uses {
                live.insert(u);
            }
        }
    }
    live
}

/// Recurse DCE into a RHS's nested blocks (a surviving closure/recursion/match body is DCE'd on its
/// own terms). Flat RHSs are returned unchanged.
fn dce_rhs(rhs: &PassRhs, log: &mut TransformLog) -> PassRhs {
    match rhs {
        PassRhs::Lam { param, body } => PassRhs::Lam {
            param: param.clone(),
            body: dce_block(body, log),
        },
        PassRhs::Fix { name, body } => PassRhs::Fix {
            name: name.clone(),
            body: dce_block(body, log),
        },
        PassRhs::FixGroup { defs, which } => PassRhs::FixGroup {
            defs: defs
                .iter()
                .map(|(n, b)| (n.clone(), dce_block(b, log)))
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
                    PassAlt::Ctor {
                        ctor,
                        binders,
                        body,
                    } => PassAlt::Ctor {
                        ctor: ctor.clone(),
                        binders: binders.clone(),
                        body: dce_block(body, log),
                    },
                    PassAlt::Lit { value, body } => PassAlt::Lit {
                        value: value.clone(),
                        body: dce_block(body, log),
                    },
                })
                .collect(),
            default: default.as_ref().map(|d| dce_block(d, log)),
        },
        other => other.clone(),
    }
}
