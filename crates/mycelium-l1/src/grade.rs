// nodule: guarantee-grading — RFC-0018 stage-1a static guarantee grading (Design A)
//
//! **Static guarantee grading** (RFC-0018 §4.3, stage 1a; Design A — data-lineage/data-provenance
//! integrity). This pass turns the guarantee index `@ g` from a *dynamically*-checked runtime tag
//! (RFC-0007 §4.3, stage 0) into a **statically**-enforced constraint over the integrity lattice
//! `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` ([`Strength`]). It runs **after** type-checking (a fully
//! type-checked, ambient-resolved body), as a self-contained second walk — keeping the trusted type
//! checker untouched (KC-3, small auditable kernel). On a violation it returns an explicit
//! [`CheckError`] (never a silent pass — G2/VR-5).
//!
//! ## Honesty (VR-5)
//! The whole pass is tagged **`Declared`** — it enforces the *design*; it does **not** discharge the
//! noninterference *theorem* (that stays **Declared-with-argument**, RFC-0018 §11 / `research/09`;
//! mechanization is the future `Proven` basis). The pass can only ever **reject** a program or
//! **over-degrade** a grade (the meet rule) — it never *upgrades* a grade without a written basis.
//!
//! ## The rules (RFC-0018 §4.3, monomorphic stage 1a)
//! - **Grade of a value.** A literal is `Exact` (a written constant is exact by construction — the
//!   G-Const grade of its `Meta`). A variable carries the grade it was bound at (G-Var). A value
//!   built from parts (`let`, constructor, prim `Op`, `match`/`if` branches, `for`-fold) carries the
//!   **meet** of the parts (the pessimistic composition rule, G-Let/G-Con/G-Op).
//! - **`Swap` is the endorsement point** (G-Swap; R18-Q4): a `swap` carries a **certificate
//!   reference** that is *trusted at the type level* — its actual validity is discharged by the
//!   RFC-0002 certificate checker at elaboration/runtime (keeping the proof checker out of the type
//!   checker — KC-3). So a `swap` satisfies any return/argument demand at check time; an invalid
//!   certificate is a never-silent refusal *there*, not here.
//! - **`wild` is the FFI floor** — opaque/untrusted: graded **`Declared`** (the least-trusted grade;
//!   the audited escape can attest nothing — LR-9/S6/VR-5).
//! - **Annotation weakens** (G-Weaken): an `@ g` on a `let` ascription, a value ascription, or a
//!   function's **return** is a *demand* — the inferred grade must be `⊒ g` (else a [`CheckError`]);
//!   the binding then carries the (possibly weaker) annotated `g`.
//! - **Calls check the argument demands** (G-App): each argument's grade must be `⊒` the callee
//!   parameter's declared grade; the call's result grade is the callee's **declared return grade**.
//! - **Design A — `match`/`if` track data, not control.** The scrutinee/condition grade does **not**
//!   degrade the result (no `pc` taint; RFC-0018 §4.5 `G-Match/A`). A *destructured field* binder,
//!   however, inherits the scrutinee's grade (the field's data provenance), so genuine data flow is
//!   still tracked. The result is the meet of the arm/branch **bodies**.
//!
//! ## The unannotated default — modular / bottom (R18-Q5 scoped to *local* inference)
//! A type with **no** written `@ g` (`TypeRef::guarantee == None`) is treated *modularly from the
//! signature*: an unannotated **parameter** demands `Declared` (the weakest — it accepts any
//! argument grade) and binds its body variable at `Declared`; an unannotated **return** advertises
//! `Declared`. So grade-checking only ever *bites* where an `@ g` is explicitly written — it never
//! rejects existing un-annotated code, and a function's advertised grade is **exactly what its
//! signature writes** (S2/LR-6: the grade is part of the observable interface, not silently
//! inferred). Recovering precision is local and verified: write the `@ g` on the return and the
//! checker proves the body supports it. Cross-function return-grade *inference* (an SCC fixpoint) is
//! whole-program — that is **stage 1b** (RFC-0018 §4.7, FlowCaml-style), not 1a (R18-Q5 scopes 1a
//! inference to *within a single expression*), and is deliberately not built here (KC-3).

use crate::ast::{Arm, Expr, FnDecl, Literal, Pattern, Strength};
use crate::checkty::CheckError;
use mycelium_workstack::RecursionBudget;
use std::collections::BTreeMap;

/// The advertised return grade of a callee (G-App result): the written `@ g`, else the modular
/// bottom `Declared` (the signature advertises only what it writes — see the module note).
fn ret_grade(fd: &FnDecl) -> Strength {
    fd.sig.ret.guarantee.unwrap_or(Strength::Declared)
}

/// The grade a function's value parameter **demands** of its argument (G-App premise) and binds its
/// body variable at: the written `@ g`, else the modular bottom `Declared` (an unannotated parameter
/// places no demand and the body may assume nothing stronger than the weakest grade).
fn param_grade(p: &crate::ast::Param) -> Strength {
    p.ty.guarantee.unwrap_or(Strength::Declared)
}

/// **Guarantee-grading pass** (RFC-0018 stage-1a; Pass 3d). Grade-check every **own** top-level
/// function body and every `impl`-method body against the lattice: each body's grade must satisfy its
/// declared return demand, and every call inside it must satisfy its callee's parameter demands.
///
/// `fns` is the **merged, resolved** table (own + imported): a call to an imported `pub fn` resolves
/// to that callee's declared grades, and the own-fn entries hold the **resolved (canonical) bodies**
/// (the checker's `resolve_pattern` already normalized every ctor/binder pattern). The pass therefore
/// walks `fns[name]` for each own name in `own_names` — *not* the raw registered body — and the
/// likewise-resolved `impl_methods`, so grading sees only canonical patterns and needs no type
/// information of its own (M-663 / Copilot review: a global ctor scan over raw patterns was an unsound
/// grade-upgrade). Imported fns were graded in their home nodule (M-662). Every refusal is an explicit
/// [`CheckError`] (G2).
pub(crate) fn check_guarantees(
    fns: &BTreeMap<String, FnDecl>,
    own_names: &BTreeMap<String, FnDecl>,
    impl_methods: &[FnDecl],
) -> Result<(), CheckError> {
    for name in own_names.keys() {
        // `fns` holds the resolved own-fn bodies (the checker overwrote each own entry with its
        // resolved form), so look the canonical body up there rather than walking the raw `own_names`.
        check_fn_grades(fns, &fns[name])?;
    }
    for m in impl_methods {
        check_fn_grades(fns, m)?;
    }
    Ok(())
}

/// Grade-check one function/method body: bind each parameter at its demanded grade, infer the body's
/// grade, and require it to satisfy the declared return demand (G-Weaken at the function boundary).
fn check_fn_grades(fns: &BTreeMap<String, FnDecl>, fd: &FnDecl) -> Result<(), CheckError> {
    let site = &fd.sig.name;
    let mut scope: Vec<(String, Strength)> = fd
        .sig
        .value_params
        .iter()
        .map(|p| (p.name.clone(), param_grade(p)))
        .collect();
    // RFC-0041 §4.7 (W1, RR-29): a fresh per-body recursion budget — the grade walk over a deep
    // resolved body (e.g. a large list literal's desugared `Cons` chain) is charged against it and
    // refused never-silently past the ceiling rather than overflowing the host stack (SIGABRT).
    let budget = RecursionBudget::default();
    let gx = Gx {
        site,
        fns,
        budget: &budget,
    };
    let body = gx.grade(&mut scope, &fd.body)?;
    let demand = ret_grade(fd);
    if !body.satisfies(demand) {
        return Err(CheckError::at(
            site,
            format!(
                "guarantee: `{site}`'s body has grade `{body:?}`, which does not satisfy the \
                 declared return `@ {demand:?}` (RFC-0018 §4.3: a body's grade must be `⊒` the \
                 return demand on the lattice `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` — weaken the \
                 return annotation, or strengthen the body; never silently — G2/VR-5)"
            ),
        ));
    }
    Ok(())
}

/// The grading context for one body: the site (for diagnostics) and the merged function table (for
/// resolving a call's parameter demands + advertised return grade — G-App). No type registry is
/// needed: the checked AST is already canonical (`Cx::resolve_pattern` resolved every ctor/binder
/// pattern), so grading is a pure, type-free computation over the lattice.
struct Gx<'a> {
    site: &'a str,
    fns: &'a BTreeMap<String, FnDecl>,
    /// The per-body recursion budget (RFC-0041 §4.7): every [`Gx::grade`] recursion charges one
    /// depth level, so a deep resolved body is refused never-silently rather than SIGABRTing.
    budget: &'a RecursionBudget,
}

impl Gx<'_> {
    /// Infer the guarantee grade of `e` under `scope` (a lexical stack of `(name, grade)`; shadowing
    /// = later wins, mirroring the type checker's scope). Enforces every `@ g` demand it crosses
    /// (call arguments, ascriptions). The expression is already type-checked + resolved, so this is a
    /// pure grade computation — no type inference, no ambient resolution.
    fn grade(&self, scope: &mut Vec<(String, Strength)>, e: &Expr) -> Result<Strength, CheckError> {
        // RFC-0041 §4.7: charge one level of grade recursion; refuse never-silently past the ceiling
        // (a deep desugared spine — e.g. a large list literal's `Cons` chain — is bounded here, not by
        // a host-stack overflow). The `DepthGuard` releases the level on every exit path.
        let _g = self.budget.try_enter().map_err(|e| {
            CheckError::at(
                self.site,
                format!(
                    "guarantee grading exceeded the recursion budget: {e} — an explicit over-budget \
                     refusal (RFC-0041 §4.7), never a host-stack overflow (G2/VR-5)"
                ),
            )
        })?;
        match e {
            // A written constant is exact by construction (G-Const: the grade of its `Meta`). A list
            // literal is built from its elements, so it carries their meet (G-Con).
            Expr::Lit(Literal::List(es)) => self.meet_all(scope, es),
            Expr::Lit(_) => Ok(Strength::Exact),

            // A variable carries its bound grade (G-Var); any other single name is a nullary
            // constructor / constant — `Exact` (a value built from nothing). Multi-segment paths were
            // already refused by the type checker, so a residual one here is conservatively `Exact`.
            Expr::Path(p) => {
                if p.0.len() == 1 {
                    if let Some((_, g)) = scope.iter().rev().find(|(n, _)| n == &p.0[0]) {
                        return Ok(*g);
                    }
                }
                Ok(Strength::Exact)
            }

            // G-Let: bind `name` at the bound's grade, weakened to the ascribed `@ g` if written
            // (the ascription is a demand — G-Weaken). The let's grade is the meet of the binding's
            // grade and the body's grade (the pessimistic composition rule, RFC-0018 §4.3 G-Let).
            Expr::Let {
                name,
                ty,
                bound,
                body,
            } => {
                let g_bound = self.grade(scope, bound)?;
                let bind = match ty.as_ref().and_then(|t| t.guarantee) {
                    Some(g) => {
                        self.require(g_bound, g, &format!("`let {name}`'s ascription `@ {g:?}`"))?;
                        g
                    }
                    None => g_bound,
                };
                scope.push((name.clone(), bind));
                let g_body = self.grade(scope, body);
                scope.pop();
                Ok(bind.meet(g_body?))
            }

            // Design A: the condition's grade does NOT degrade the result (no `pc` taint — RFC-0018
            // §4.5). The condition is still walked (to enforce any `@ g` demand inside it); the
            // result is the meet of the two branch bodies.
            Expr::If { cond, conseq, alt } => {
                let _ = self.grade(scope, cond)?;
                let t = self.grade(scope, conseq)?;
                let f = self.grade(scope, alt)?;
                Ok(t.meet(f))
            }

            // Design A `G-Match/A`: the scrutinee grade does not appear in the result. A pattern's
            // field binders inherit the scrutinee's grade (data provenance — a destructured field's
            // data did come from the scrutinee), so genuine data flow is tracked; the *control* path
            // (which arm) is not (no `pc`). The result is the meet of the arm bodies.
            Expr::Match { scrutinee, arms } => self.grade_match(scope, scrutinee, arms),

            // A `for`-fold. The element binder carries the spine value's grade. The accumulator's grade
            // across iterations is the *fixpoint* of `meet(g_init, body-output)` — its first value is
            // `g_init`, but the body re-binds it each step, so a later iteration's `acc` may be **weaker**
            // than `g_init`. To stay sound *without* iterating to a fixpoint (stage 1a), the body is
            // graded with `acc` bound at the **bottom** grade `Declared` (the weakest any iteration's
            // accumulator can be): a body that *demands* a strong grade on `acc` is then correctly
            // refused (it would not hold on the second iteration), never silently accepted on the basis
            // of the initial value alone (G2/VR-5 — under-estimating a grade is always sound). The fold's
            // result grade is the conservative meet of the initial accumulator, the spine, and the body
            // (it can only ever over-degrade — honest; precision over folds is stage-1b work).
            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => {
                let g_xs = self.grade(scope, xs)?;
                let g_init = self.grade(scope, init)?;
                scope.push((x.clone(), g_xs));
                scope.push((acc.clone(), Strength::Declared));
                let g_body = self.grade(scope, body);
                scope.pop();
                scope.pop();
                Ok(g_init.meet(g_xs).meet(g_body?))
            }

            // G-Swap (the endorsement point; R18-Q4): the source is walked (to enforce demands within
            // it), but the swap's certificate reference is trusted at the type level — so the result
            // is `Exact` (it satisfies any demand). The certificate's actual validity is discharged by
            // the RFC-0002 checker at elaboration/runtime (KC-3), where an invalid cert is a
            // never-silent refusal (G2).
            Expr::Swap { value, .. } => {
                let _ = self.grade(scope, value)?;
                Ok(Strength::Exact)
            }

            // `wild` is the audited FFI floor (LR-9/S6): opaque and untrusted — it can attest nothing,
            // so it carries the least-trusted grade `Declared` (VR-5). The body is not walked (it is
            // the trusted/opaque escape — not recursively analyzed, matching the type checker).
            Expr::Wild(_) => Ok(Strength::Declared),

            // The colony's observable is its **last** hypha (matching the type rule); leading hyphae
            // are still walked (to enforce demands inside them).
            Expr::Colony(hyphae) => {
                let Some((last, leading)) = hyphae.split_last() else {
                    return Ok(Strength::Exact);
                };
                for h in leading {
                    let _ = self.grade(scope, &h.body)?;
                }
                self.grade(scope, &last.body)
            }

            // G-Weaken: an `@ g` ascription demands the inferred grade be `⊒ g`, and the ascribed
            // expression then carries `g`. A bare type ascription (`: T`, no `@ g`) is grade-transparent.
            Expr::Ascribe(inner, t) => {
                let g_inner = self.grade(scope, inner)?;
                match t.guarantee {
                    Some(g) => {
                        self.require(g_inner, g, &format!("ascription `@ {g:?}`"))?;
                        Ok(g)
                    }
                    None => Ok(g_inner),
                }
            }

            Expr::App { head, args } => self.grade_app(scope, head, args),

            // Staged / resolved-away forms. `spore` is deferred (the type checker already refuses it);
            // `with paradigm` is stripped by the ambient pass before the checker. Defensive,
            // never-reached arms: grade the body conservatively rather than panic.
            Expr::Spore(_) => Ok(Strength::Declared),
            // M-664: `consume <expr>` is a **move** — it transfers the operand value unchanged, so it
            // is grade-**transparent**: the result carries exactly the operand's grade. This both
            // *enforces* the operand's own grade demands (by grading it) and *propagates* its grade —
            // so `consume s` of a `@ Exact` substrate stays `Exact` (returning `Declared` here would
            // false-reject a valid `=> … @ Exact` body). `consume` neither upgrades nor downgrades the
            // attestation (VR-5: no upgrade past a checked basis; the operand's basis is preserved).
            // The single-use affinity it asserts is a *usage* discipline, orthogonal to the value's
            // accuracy grade.
            Expr::Consume(b) => self.grade(scope, b),
            // RFC-0024 §4A (M-704): a `lambda` (closure) is a `Declared`-grade construct — its
            // lowering is a structural rewrite + a type-level contract (the three-way differential is
            // `Empirical`, but the construct itself attests no more than `Declared`; VR-5). Grading
            // runs on the source env (pre-mono), so a `lambda` is reachable here; it carries
            // `Declared` (never upgraded past its basis).
            Expr::Lambda { .. } => Ok(Strength::Declared),
            Expr::WithParadigm { body, .. } => self.grade(scope, body),
            // DN-58 §A/§B (M-667): `fuse(a, b)` — the grade is the *meet* of both operands' grades
            // (RFC-0018 §4.1: composition takes the weakest). This matches how `op_call`/`App` grades
            // binary operations (G-App). Guarantee: `Empirical` (three-way differential, DN-58 §A.5).
            Expr::Fuse { left, right } => {
                let lg = self.grade(scope, left)?;
                let rg = self.grade(scope, right)?;
                Ok(lg.meet(rg))
            }
            // DN-58 §B (M-667): `reclaim(policy) { body }` — the result's grade is the body's grade.
            // The policy expression affects supervision (a runtime concern), not the result's
            // honesty provenance. The policy is still graded to surface any policy-grade violations
            // (never-silent — G2), then the body's grade propagates.
            Expr::Reclaim { policy, body } => {
                let _ = self.grade(scope, policy)?;
                self.grade(scope, body)
            }
            // M-826: a tuple literal `(a, b, …)` is rewritten by the checker to `App(MkTuple$N,
            // elems)` before grading runs on the checked/monomorphized AST. If this arm is reached,
            // grade each element and take the meet (the provenance of a tuple is the weakest element —
            // `Empirical` guarantee). Never-silent: element grading errors propagate (G2).
            Expr::TupleLit(elems) => self.meet_all(scope, elems),
        }
    }

    /// `G-Match/A`: bind each pattern's field binders at the scrutinee's grade (data provenance), grade
    /// each arm body, and take the meet (the scrutinee's *control* grade does not appear — Design A).
    fn grade_match(
        &self,
        scope: &mut Vec<(String, Strength)>,
        scrutinee: &Expr,
        arms: &[Arm],
    ) -> Result<Strength, CheckError> {
        let g_s = self.grade(scope, scrutinee)?;
        let mut acc: Option<Strength> = None;
        for arm in arms {
            let pushed = self.bind_pattern(scope, &arm.pattern, g_s);
            let g_arm = self.grade(scope, &arm.body);
            scope.truncate(scope.len() - pushed);
            let g_arm = g_arm?;
            acc = Some(match acc {
                Some(a) => a.meet(g_arm),
                None => g_arm,
            });
        }
        // A `match` with no arms cannot occur (the parser requires ≥ 1); be conservative if it does.
        Ok(acc.unwrap_or(Strength::Declared))
    }

    /// G-App / G-Con / G-Op. A call to a known user function checks each argument against its
    /// parameter's demanded grade and yields the callee's advertised return grade. Any other
    /// application head (constructor, builtin prim, or unqualified trait method — none of which carry
    /// graded signatures in stage 1a) takes the **conservative meet** of its argument grades
    /// (RFC-0018 §4.6: the ungraded-prim default is grade-preserving = meet of inputs; G-Con is the
    /// meet of the fields). The meet can only over-degrade — honest (VR-5).
    fn grade_app(
        &self,
        scope: &mut Vec<(String, Strength)>,
        head: &Expr,
        args: &[Expr],
    ) -> Result<Strength, CheckError> {
        if let Expr::Path(p) = head {
            if p.0.len() == 1 {
                if let Some(fd) = self.fns.get(&p.0[0]) {
                    // G-App: each argument's grade must satisfy its parameter's demand; the result is
                    // the callee's declared return grade. (Arity was already checked by the type
                    // checker; `zip` is safe.)
                    for (pm, a) in fd.sig.value_params.iter().zip(args) {
                        let g_a = self.grade(scope, a)?;
                        let demand = param_grade(pm);
                        self.require(
                            g_a,
                            demand,
                            &format!("argument `{}` to `{}`", pm.name, p.0[0]),
                        )?;
                    }
                    return Ok(ret_grade(fd));
                }
            }
        }
        // Constructor / prim / trait-method (no graded signature yet) — conservative meet of args.
        self.meet_all(scope, args)
    }

    /// The meet of every expression's grade (`Exact` for an empty list — the identity of the meet).
    fn meet_all(
        &self,
        scope: &mut Vec<(String, Strength)>,
        es: &[Expr],
    ) -> Result<Strength, CheckError> {
        let mut acc = Strength::Exact;
        for e in es {
            acc = acc.meet(self.grade(scope, e)?);
        }
        Ok(acc)
    }

    /// Push every variable a pattern binds onto `scope` at grade `g_s` (the scrutinee's grade — a
    /// destructured field's data provenance is the scrutinee's). Returns how many bindings were pushed,
    /// so the caller can pop exactly that many.
    ///
    /// The checked AST is **canonical** (`Cx::resolve_pattern`): a `Pattern::Ident` is always a true
    /// **binder** and a `Pattern::Ctor` always a constructor — the checker, which alone knows the
    /// *expected scrutinee type*, already resolved the bare-ident ctor/binder ambiguity and rewrote a
    /// nullary-ctor pattern to `Ctor(name, [])`. So a binder enters the grade scope at `g_s`, while a
    /// (nullary or n-ary) `Ctor` binds nothing itself and only recurses into its sub-patterns. This
    /// pass therefore needs **no type information** — resolving the ambiguity *here* (with only the
    /// global type registry, not the scrutinee type) was an unsound grade-upgrade: a binder whose name
    /// collided with a nullary ctor of an *unrelated* type would drop its binding and a later
    /// reference would grade `Exact` instead of `g_s` (M-663 / Copilot review). `Wildcard`/`Lit` bind
    /// nothing.
    fn bind_pattern(
        &self,
        scope: &mut Vec<(String, Strength)>,
        pat: &Pattern,
        g_s: Strength,
    ) -> usize {
        match pat {
            Pattern::Wildcard | Pattern::Lit(_) => 0,
            Pattern::Ident(name) => {
                scope.push((name.clone(), g_s));
                1
            }
            Pattern::Ctor(_, subs) => {
                let mut n = 0;
                for s in subs {
                    n += self.bind_pattern(scope, s, g_s);
                }
                n
            }
            // M-826: a tuple pattern `(x, y, …)` binds each sub-pattern at the scrutinee's grade.
            // The checker rewrites these to `Ctor(MkTuple$N, subs)` during checking, so this arm
            // handles any surface-form pattern that reaches grading directly.
            Pattern::Tuple(subs) => {
                let mut n = 0;
                for s in subs {
                    n += self.bind_pattern(scope, s, g_s);
                }
                n
            }
            // `Pattern::Or` is desugared in `check_match` before grading; reaching here means the
            // program was not checked — a never-silent panic (invariant violation; G2).
            Pattern::Or(_) => {
                panic!(
                    "internal: Pattern::Or reached grade::bind_pattern — or-patterns must be \
                     desugared by the checker before any downstream pass (invariant violation)"
                )
            }
        }
    }

    /// The honesty check `have ⊒ demand` (G-Sub): a never-silent [`CheckError`] naming both grades
    /// and `what` is being constrained when the value is too weak for the demand (VR-5).
    fn require(&self, have: Strength, demand: Strength, what: &str) -> Result<(), CheckError> {
        if have.satisfies(demand) {
            return Ok(());
        }
        Err(CheckError::at(
            self.site,
            format!(
                "guarantee: {what} has grade `{have:?}`, which does not satisfy the demanded \
                 `@ {demand:?}` (RFC-0018 §4.3 — `{have:?} ⊒ {demand:?}` is required on the lattice \
                 `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`; the annotation may only weaken, never \
                 upgrade — VR-5; never silent — G2)"
            ),
        ))
    }
}
