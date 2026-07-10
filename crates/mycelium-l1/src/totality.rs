//! The **structural totality checker** (RFC-0007 §4.5; T3.4) — *outside* the trusted kernel: its
//! verdict gates the `matured` privilege, never meaning (a wrong verdict can mis-gate a
//! promotion; semantics stay with the fuel-guarded evaluator).
//!
//! Classification (Foetus-style structural descent, v0):
//! - no (direct or mutual) recursion → **Total**;
//! - self-recursion where *every* recursive call passes, in some fixed argument position, a
//!   variable **structurally smaller** than that parameter (bound by a `Match` alternative on the
//!   parameter or on an already-smaller variable — descent is transitive) → **Total**;
//! - **mutual recursion** (a `FixGroup` / strongly-connected call-graph component, RFC-0001 r5,
//!   R7-Q3) where there is a **mutual structural descent**: a designated argument position `p(f)`
//!   for each member `f` such that *every* call from a member `f` to a member `g` passes, in `g`'s
//!   position `p(g)`, a variable structurally smaller than `f`'s parameter `p(f)` → **Total**.
//!   Self-recursion is the size-1 case. Sound by one well-founded measure: the structural size of
//!   the designated argument strictly decreases at every call along any path through the group, so
//!   no infinite call path exists;
//! - anything else (a non-productive cycle, a group too large to search, or one this structural
//!   criterion cannot witness) → **Partial** — an honest, incomplete classification, not an error.
//!
//! The checker is **sound, not complete**: it never classifies a non-terminating group `Total`
//! (that would mis-grant `matured`), but it may leave a terminating group `Partial`. Widening it
//! (here, from self- to mutual-descent) only ever *adds* `Total` verdicts that the well-founded
//! measure justifies — it never relaxes the bar.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Arm, Expr, FnDecl, Pattern};

/// The divergence bit (RFC-0007 §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Totality {
    /// Checked total: terminates under the reference evaluator for every sufficiently large fuel.
    Total,
    /// Not certified total (may or may not terminate) — honest, not an error.
    Partial,
}

/// A bound on the position-assignment search for a mutual group (∏ of member arities). Beyond it
/// the group stays `Partial` — sound (we never *over*-classify), just incomplete, and well past any
/// realistic hand-written mutual cycle.
const MAX_ASSIGNMENTS: usize = 4096;

/// Explicit depth budget for this module's own AST traversals (M-674 remaining TODO item 2): the
/// shared [`walk_expr`] (also reused by [`crate::elab`]'s call-set collector and
/// [`crate::checkty`]'s lower-rule-edge / effect-coverage collectors — one canonical traversal, DRY,
/// M-641), [`descend_walk`]'s mutual-descent search, and [`pattern_binders`]. This is the compiler
/// **pass's own** recursive-descent recursion — distinct from [`Totality`] itself, which is a
/// semantic verdict *about* a checked program's user-level recursion, not an operational resource
/// limit. Mirrors the checker's `MAX_CHECK_DEPTH` discipline (banked guard 4; A4-02): rather than
/// rely on the host call stack (a resource that is not a semantic limit) to bound the traversal, the
/// pass carries this reified budget and refuses past it with a clean [`WalkDepthExceeded`], never a
/// host-stack overflow. Set comfortably above the parser's `MAX_EXPR_DEPTH` (256) surface-nesting
/// cap, so no parser-produced AST ever approaches it — the ceiling exists as defense-in-depth for a
/// synthetic/API-built tree handed directly to these passes.
///
/// **Grounding (measured, not guessed).** `classify_all` runs on [`mycelium_stack`]'s 256 MiB deep
/// worker stack (below); `walk_expr`'s own frame is far lighter than the checker's `Cx::check`
/// (no per-node type/scope state) — measured empirically, it survives at least 2,000,000 levels of
/// recursion on that stack before overflowing (vs. the checker's ~24,600 at ~10.9 KiB/frame). This
/// budget (`4096`) is therefore a **~500× safety margin** below the measured physical floor, and
/// **16×** above the parser's 256-deep surface cap.
pub const MAX_WALK_DEPTH: u32 = 4096;

/// A never-silent refusal from a pass-internal AST traversal ([`walk_expr`], [`descend_walk`], or
/// [`pattern_binders`]) once its own recursive descent exceeds [`MAX_WALK_DEPTH`] (M-674). Distinct
/// from [`Totality::Partial`]: this is an operational resource refusal about the *pass's* recursion,
/// never a claim about whether the checked program terminates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkDepthExceeded {
    /// The exceeded budget.
    pub limit: u32,
}

impl std::fmt::Display for WalkDepthExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AST nesting exceeds the compiler pass's own recursion-depth budget ({}) — an explicit \
             budget (banked guard 4; M-674), refused cleanly rather than overflowing the host stack",
            self.limit
        )
    }
}

impl std::error::Error for WalkDepthExceeded {}

/// Classify every function in the table.
///
/// Runs on [`mycelium_stack`]'s deep worker stack (M-674), mirroring the checker's/evaluator's
/// discipline: [`MAX_WALK_DEPTH`] — not a host-stack overflow — is always what bounds a
/// pathologically-nested body, regardless of the caller's own thread-stack size (`classify_all` is
/// called both from inside `checkty`'s already-deep-stacked pipeline and directly, e.g. from
/// `mono`'s post-specialization re-classification and from tests — wrapping here, rather than
/// relying on every caller to wrap it, is what makes the budget unconditionally physically backed).
///
/// # Errors
/// [`WalkDepthExceeded`] if the pass's own AST traversal recursion exceeds [`MAX_WALK_DEPTH`] on a
/// pathologically-nested body — an explicit, never-silent refusal (M-674), never a host-stack
/// overflow.
pub fn classify_all(
    fns: &BTreeMap<String, FnDecl>,
) -> Result<BTreeMap<String, Totality>, WalkDepthExceeded> {
    mycelium_stack::with_deep_stack(|| classify_all_inner(fns))
}

fn classify_all_inner(
    fns: &BTreeMap<String, FnDecl>,
) -> Result<BTreeMap<String, Totality>, WalkDepthExceeded> {
    // Call graph.
    let mut calls: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for (name, fd) in fns {
        let mut out = BTreeSet::new();
        collect_calls(&fd.body, fns, &mut out)?;
        calls.insert(name, out);
    }
    let mut result = BTreeMap::new();
    for scc in strongly_connected(fns, &calls) {
        // A component is recursive iff it has > 1 member (necessarily a cycle) or its single member
        // calls itself directly. A non-recursive definition is `Total` with no descent obligation.
        let recursive = scc.len() > 1 || calls[scc[0].as_str()].contains(&scc[0]);
        let total = !recursive || group_descends(&scc, fns)?;
        let t = if total {
            Totality::Total
        } else {
            Totality::Partial
        };
        for name in scc {
            result.insert(name, t);
        }
    }
    Ok(result)
}

/// Partition the functions into strongly-connected components of the call graph (each is a
/// `FixGroup`, RFC-0001 r5). Two functions share a component iff they are mutually reachable;
/// that relation is an equivalence, so a greedy grouping yields the full components. Deterministic
/// (iteration follows the `BTreeMap` key order).
fn strongly_connected(
    fns: &BTreeMap<String, FnDecl>,
    calls: &BTreeMap<&str, BTreeSet<String>>,
) -> Vec<Vec<String>> {
    let mut assigned: BTreeSet<&str> = BTreeSet::new();
    let mut sccs = Vec::new();
    for name in fns.keys() {
        if assigned.contains(name.as_str()) {
            continue;
        }
        let mut group = vec![name.clone()];
        assigned.insert(name);
        for other in fns.keys() {
            if assigned.contains(other.as_str()) {
                continue;
            }
            if reaches(name, other, calls) && reaches(other, name, calls) {
                group.push(other.clone());
                assigned.insert(other);
            }
        }
        sccs.push(group);
    }
    sccs
}

/// Does `from` reach `target` through the call graph (cycle detection for mutual recursion)?
fn reaches(from: &str, target: &str, calls: &BTreeMap<&str, BTreeSet<String>>) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from.to_owned()];
    while let Some(f) = stack.pop() {
        if !seen.insert(f.clone()) {
            continue;
        }
        if let Some(cs) = calls.get(f.as_str()) {
            for c in cs {
                if c == target {
                    return true;
                }
                stack.push(c.clone());
            }
        }
    }
    false
}

fn collect_calls(
    e: &Expr,
    fns: &BTreeMap<String, FnDecl>,
    out: &mut BTreeSet<String>,
) -> Result<(), WalkDepthExceeded> {
    walk_expr(e, &mut |x| {
        if let Expr::App { head, .. } = x {
            if let Expr::Path(p) = head.as_ref() {
                if p.0.len() == 1 && fns.contains_key(&p.0[0]) {
                    out.insert(p.0[0].clone());
                }
            }
        }
    })
}

/// The shared **pre-order `Expr` traversal** (M-641): visit `e` with `f`, then recurse into every
/// sub-expression (calling `f` on each in turn). One canonical structural walk reused by every pass
/// that needs to fold a *stateless* visitor over an expression tree — here totality's `collect_calls`
/// and the elaborator's call-set collector (`crate::elab`). It is the structure only; each caller
/// supplies its own visitor action, so factoring it changes no collected set.
///
/// Passes that thread *context* down the tree (e.g. the totality descent measure in
/// [`descend_walk`], which shadows binders per `Match` arm) are deliberately **not** expressed over
/// this — their per-node state is not a plain `FnMut(&Expr)`, and collapsing them would lose the
/// scoping that keeps the analysis sound (A4-01).
///
/// # Errors
/// [`WalkDepthExceeded`] once the traversal's own recursion exceeds [`MAX_WALK_DEPTH`] (M-674) — a
/// clean, explicit refusal rather than a host-stack overflow on a pathologically-nested `e`.
pub(crate) fn walk_expr(e: &Expr, f: &mut impl FnMut(&Expr)) -> Result<(), WalkDepthExceeded> {
    walk_expr_at(e, f, 0)
}

/// The depth-tracked worker behind [`walk_expr`] (M-674): `depth` counts the live nesting of this
/// traversal's own recursive descent (not any semantic property of `e`), charged on entry and
/// checked against [`MAX_WALK_DEPTH`] before any further recursion.
fn walk_expr_at(e: &Expr, f: &mut impl FnMut(&Expr), depth: u32) -> Result<(), WalkDepthExceeded> {
    let depth = depth + 1;
    if depth > MAX_WALK_DEPTH {
        return Err(WalkDepthExceeded {
            limit: MAX_WALK_DEPTH,
        });
    }
    f(e);
    match e {
        Expr::Let { bound, body, .. } => {
            walk_expr_at(bound, f, depth)?;
            walk_expr_at(body, f, depth)?;
        }
        Expr::If { cond, conseq, alt } => {
            walk_expr_at(cond, f, depth)?;
            walk_expr_at(conseq, f, depth)?;
            walk_expr_at(alt, f, depth)?;
        }
        Expr::Match { scrutinee, arms } => {
            walk_expr_at(scrutinee, f, depth)?;
            for a in arms {
                walk_expr_at(&a.body, f, depth)?;
            }
        }
        // A `for` is bounded by construction (RFC-0007 §4.8) — it adds no recursion of its own;
        // only the calls inside its sub-expressions matter.
        Expr::For { xs, init, body, .. } => {
            walk_expr_at(xs, f, depth)?;
            walk_expr_at(init, f, depth)?;
            walk_expr_at(body, f, depth)?;
        }
        Expr::Swap { value, .. } => walk_expr_at(value, f, depth)?,
        // `with paradigm` is pure surface scoping (stripped by resolution before this runs); recurse
        // transparently into the body in case totality is consulted on an unresolved tree.
        Expr::WithParadigm { body, .. } => walk_expr_at(body, f, depth)?,
        // `wild` is the audited/opaque FFI escape (M-661): its body is trusted foreign code, **not**
        // analyzable Mycelium, so the shared traversal treats it as a LEAF. The `wild` node itself is
        // still visited by `f` above (so effect coverage credits it the `ffi` source — M-661/§8-Q6),
        // but its interior is **never descended**: effects/calls/recursion inside a `wild` body do not
        // leak into the enclosing fn's analysis — consistent with `Cx::check_wild` not recursively
        // checking the body (audited, not verified; VR-5/ADR-014). Execution is staged (`elab` →
        // `Residual`). `spore(value)`, by contrast, wraps a *real* value expression (deferred —
        // E2-5/M-260), so it recurses transparently.
        Expr::Wild(_) => {}
        Expr::Spore(b) => walk_expr_at(b, f, depth)?,
        // M-664: `consume <expr>` wraps a real value expression — walk it transparently so any
        // calls inside the operand are still seen by the call-set/totality collectors.
        Expr::Consume(b) => walk_expr_at(b, f, depth)?,
        // A `lambda` body is deferred (M-704; never executes in v0), but walk it transparently so
        // any calls inside are still seen by the call-set/totality collectors (conservative).
        Expr::Lambda { body, .. } => walk_expr_at(body, f, depth)?,
        // A `colony` block's calls are exactly the calls inside its `hypha` bodies (RFC-0008 §4.7).
        // Each hypha body is walked transparently so the call-set / totality collectors see them.
        Expr::Colony(hyphae) => {
            for h in hyphae {
                walk_expr_at(&h.body, f, depth)?;
            }
        }
        Expr::App { head, args } => {
            walk_expr_at(head, f, depth)?;
            for a in args {
                walk_expr_at(a, f, depth)?;
            }
        }
        Expr::Ascribe(b, _) => walk_expr_at(b, f, depth)?,
        // DN-58 §A/§B (M-667): `fuse(a, b)` and `reclaim(policy) { body }` — walk sub-expressions
        // transparently so the call-set / totality collectors see any recursive calls inside.
        Expr::Fuse { left, right } => {
            walk_expr_at(left, f, depth)?;
            walk_expr_at(right, f, depth)?;
        }
        Expr::Reclaim { policy, body } => {
            walk_expr_at(policy, f, depth)?;
            walk_expr_at(body, f, depth)?;
        }
        // M-826: walk each element for recursive-call detection; the checker rewrites TupleLit
        // to App(MkTuple$N, elems) before totality runs, but handle any surface-form TupleLit
        // that reaches here directly (e.g. in intermediate passes or tests).
        Expr::TupleLit(elems) => {
            for el in elems {
                walk_expr_at(el, f, depth)?;
            }
        }
        Expr::Path(_) | Expr::Lit(_) => {}
    }
    Ok(())
}

/// A mutual group (size ≥ 1) descends iff some assignment of one designated argument position to
/// each member makes *every* inter-member call structural (§4.5). Searches the bounded product of
/// member arities. The size-1 case is exactly self-descent: the one member ranges over its
/// positions, and the only group member it can call is itself.
fn group_descends(
    scc: &[String],
    fns: &BTreeMap<String, FnDecl>,
) -> Result<bool, WalkDepthExceeded> {
    let members: Vec<&FnDecl> = scc.iter().map(|n| &fns[n]).collect();
    let arities: Vec<usize> = members.iter().map(|fd| fd.sig.value_params.len()).collect();
    // A nullary member has no parameter to descend on, so this structural criterion cannot witness
    // the group total — honestly `Partial`.
    if arities.contains(&0) {
        return Ok(false);
    }
    let combos: usize = arities.iter().product();
    if combos > MAX_ASSIGNMENTS {
        return Ok(false);
    }
    // Each candidate is a mixed-radix index over the member arities: digit k chooses the designated
    // position of member k. A `WalkDepthExceeded` from any candidate aborts the whole search
    // immediately (never silently treated as "this candidate doesn't descend" — that would risk a
    // false `Partial` from a resource limit rather than a genuine non-descent, G2).
    for mut rem in 0..combos {
        let mut pos = BTreeMap::new();
        for (fd, &arity) in members.iter().zip(&arities) {
            pos.insert(fd.sig.name.as_str(), rem % arity);
            rem /= arity;
        }
        if assignment_descends(&members, &pos)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check one position assignment: every member's body, walked with that member's designated
/// parameter as the descent measure, makes every call to a group member pass a strictly-smaller
/// argument in the **callee's** designated position.
fn assignment_descends(
    members: &[&FnDecl],
    pos: &BTreeMap<&str, usize>,
) -> Result<bool, WalkDepthExceeded> {
    for fd in members {
        let param = fd.sig.value_params[pos[fd.sig.name.as_str()]].name.as_str();
        let mut ok = true;
        descend_walk(&fd.body, pos, param, &mut BTreeSet::new(), &mut ok, 0)?;
        if !ok {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Walk tracking the set of variables smaller-than the designated parameter; check every call to a
/// group member. `smaller` grows at `Match` alternatives whose scrutinee is the parameter or an
/// already-smaller variable. `pos` maps each group member to the argument position that must
/// receive a smaller variable on a call to it.
///
/// # Errors
/// [`WalkDepthExceeded`] once this traversal's own recursion exceeds [`MAX_WALK_DEPTH`] (M-674) — a
/// clean, explicit refusal rather than a host-stack overflow on a pathologically-nested body.
#[allow(clippy::too_many_arguments)] // the descent search threads its measure + the depth budget
fn descend_walk(
    e: &Expr,
    pos: &BTreeMap<&str, usize>,
    param: &str,
    smaller: &mut BTreeSet<String>,
    ok: &mut bool,
    depth: u32,
) -> Result<(), WalkDepthExceeded> {
    let depth = depth + 1;
    if depth > MAX_WALK_DEPTH {
        return Err(WalkDepthExceeded {
            limit: MAX_WALK_DEPTH,
        });
    }
    match e {
        Expr::App { head, args } => {
            if let Expr::Path(p) = head.as_ref() {
                if p.0.len() == 1 {
                    if let Some(&tpos) = pos.get(p.0[0].as_str()) {
                        // A call to a group member: its designated argument must be a smaller var.
                        let good = args.get(tpos).is_some_and(|a| match a {
                            Expr::Path(v) => v.0.len() == 1 && smaller.contains(&v.0[0]),
                            _ => false,
                        });
                        if !good {
                            *ok = false;
                        }
                    }
                }
            }
            descend_walk(head, pos, param, smaller, ok, depth)?;
            for a in args {
                descend_walk(a, pos, param, smaller, ok, depth)?;
            }
        }
        Expr::Match { scrutinee, arms } => {
            descend_walk(scrutinee, pos, param, smaller, ok, depth)?;
            let scrut_small = match scrutinee.as_ref() {
                Expr::Path(p) if p.0.len() == 1 => p.0[0] == param || smaller.contains(&p.0[0]),
                _ => false,
            };
            for Arm { pattern, body } in arms {
                // Every binder the pattern introduces SHADOWS any outer variable of the same name,
                // so its prior smallness must not leak into the arm body (A4-01: otherwise a binder
                // reusing an outer `smaller` name lets a non-decreasing recursive call look
                // structural). Drop all introduced binders, restore afterwards — mirroring the
                // `Let`/`For` discipline. Only a constructor sub-binder of a *smaller* scrutinee is
                // itself genuinely smaller, so re-add just those.
                let mut introduced = Vec::new();
                pattern_binders(pattern, &mut introduced, 0)?;
                let mut restore = Vec::new();
                for b in &introduced {
                    if smaller.remove(b) {
                        restore.push(b.clone());
                    }
                }
                let mut added = Vec::new();
                if scrut_small {
                    if let Pattern::Ctor(_, subs) = pattern {
                        // Every binder under a constructor of a smaller-or-equal scrutinee is itself
                        // strictly smaller — including binders nested under further constructors
                        // (e.g. `m` in `S(S(m))`), so structural descent works through nested
                        // patterns, not just one level deep.
                        let mut nested = Vec::new();
                        for s in subs {
                            pattern_binders(s, &mut nested, 0)?;
                        }
                        for b in nested {
                            if smaller.insert(b.clone()) {
                                added.push(b);
                            }
                        }
                    }
                }
                descend_walk(body, pos, param, smaller, ok, depth)?;
                for b in added {
                    smaller.remove(&b);
                }
                for b in restore {
                    smaller.insert(b);
                }
            }
        }
        Expr::Let {
            bound, body, name, ..
        } => {
            descend_walk(bound, pos, param, smaller, ok, depth)?;
            // A rebinding shadows; conservatively drop smallness for the shadowed name.
            let was = smaller.remove(name);
            descend_walk(body, pos, param, smaller, ok, depth)?;
            if was {
                smaller.insert(name.clone());
            }
        }
        Expr::If { cond, conseq, alt } => {
            descend_walk(cond, pos, param, smaller, ok, depth)?;
            descend_walk(conseq, pos, param, smaller, ok, depth)?;
            descend_walk(alt, pos, param, smaller, ok, depth)?;
        }
        Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        } => {
            descend_walk(xs, pos, param, smaller, ok, depth)?;
            descend_walk(init, pos, param, smaller, ok, depth)?;
            // The binders shadow; conservatively drop smallness for the shadowed names.
            let had_x = smaller.remove(x);
            let had_acc = smaller.remove(acc);
            descend_walk(body, pos, param, smaller, ok, depth)?;
            if had_x {
                smaller.insert(x.clone());
            }
            if had_acc {
                smaller.insert(acc.clone());
            }
        }
        Expr::Swap { value, .. } => descend_walk(value, pos, param, smaller, ok, depth)?,
        Expr::WithParadigm { body, .. } => descend_walk(body, pos, param, smaller, ok, depth)?,
        // A `wild` body is opaque trusted FFI (M-661) — a leaf here too: a call inside a `wild` block
        // is not analyzable Mycelium recursion (and execution is staged), so it is not subject to the
        // structural-descent check, mirroring `walk_expr` (the opacity invariant is uniform — VR-5).
        Expr::Wild(_) => {}
        Expr::Spore(b) => descend_walk(b, pos, param, smaller, ok, depth)?,
        // M-664: `consume <expr>` introduces no binders; walk the operand transparently so a
        // recursive call inside it is still subject to the structural-descent check.
        Expr::Consume(b) => descend_walk(b, pos, param, smaller, ok, depth)?,
        // A `lambda` introduces its own parameter binders and is a deferred form (M-704); walk its
        // body transparently (it adds no recursive-descent of the enclosing fn's parameter).
        Expr::Lambda { body, .. } => descend_walk(body, pos, param, smaller, ok, depth)?,
        // A `colony`'s hyphae introduce no binders; walk each body transparently so a recursive call
        // inside a hypha is still subject to the structural-descent check (A4-01).
        Expr::Colony(hyphae) => {
            for h in hyphae {
                descend_walk(&h.body, pos, param, smaller, ok, depth)?;
            }
        }
        Expr::Ascribe(b, _) => descend_walk(b, pos, param, smaller, ok, depth)?,
        // DN-58 §A/§B (M-667): `fuse(a, b)` and `reclaim(policy) { body }` — walk sub-expressions
        // transparently so recursive calls inside fuse/reclaim are subject to structural-descent
        // analysis (A4-01; neither `fuse` nor `reclaim` introduces binders, so no shadowing).
        Expr::Fuse { left, right } => {
            descend_walk(left, pos, param, smaller, ok, depth)?;
            descend_walk(right, pos, param, smaller, ok, depth)?;
        }
        Expr::Reclaim { policy, body } => {
            descend_walk(policy, pos, param, smaller, ok, depth)?;
            descend_walk(body, pos, param, smaller, ok, depth)?;
        }
        // M-826: a tuple literal's elements are all value positions; walk each for recursive calls.
        Expr::TupleLit(elems) => {
            for el in elems {
                descend_walk(el, pos, param, smaller, ok, depth)?;
            }
        }
        Expr::Path(_) | Expr::Lit(_) => {}
    }
    Ok(())
}

/// Collect every variable a pattern binds, recursively — so a `Match` arm can shadow them all
/// (A4-01). Wildcards and literals bind nothing.
///
/// # Errors
/// [`WalkDepthExceeded`] once this traversal's own recursion exceeds [`MAX_WALK_DEPTH`] (M-674) — a
/// clean, explicit refusal rather than a host-stack overflow on a pathologically-nested pattern.
fn pattern_binders(
    p: &Pattern,
    out: &mut Vec<String>,
    depth: u32,
) -> Result<(), WalkDepthExceeded> {
    let depth = depth + 1;
    if depth > MAX_WALK_DEPTH {
        return Err(WalkDepthExceeded {
            limit: MAX_WALK_DEPTH,
        });
    }
    match p {
        Pattern::Ident(b) => out.push(b.clone()),
        Pattern::Ctor(_, subs) => {
            for s in subs {
                pattern_binders(s, out, depth)?;
            }
        }
        // M-826: a tuple pattern `(x, y, …)` binds each sub-pattern's variable.
        Pattern::Tuple(subs) => {
            for s in subs {
                pattern_binders(s, out, depth)?;
            }
        }
        Pattern::Wildcard | Pattern::Lit(_) => {}
        // `Pattern::Or` is desugared in `check_match` before totality analysis; reaching here means
        // the program was not checked — a never-silent panic (invariant violation; G2).
        Pattern::Or(_) => {
            panic!(
                "internal: Pattern::Or reached totality::pattern_binders — or-patterns must be \
                 desugared by the checker before any downstream pass (invariant violation)"
            )
        }
    }
    Ok(())
}
