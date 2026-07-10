//! **Monomorphization** (M-673; RFC-0007 §11.3 / §12.3, RFC-0019 §4.4) — the elaboration pre-pass
//! that turns a *checked* generic-and-trait `Env` into a **closed, monomorphic** `Env` the existing
//! [`crate::elab::elaborate`] / [`crate::elab::build_registry`] then lower **unchanged**.
//!
//! # What it does (and what it deliberately does not)
//! [`monomorphize`] re-walks the reachable graph from a nullary monomorphic `entry`, specializing
//! each generic function/data instantiation at its concrete type arguments and **statically
//! resolving** each unqualified trait-method call to the one coherent instance's method body
//! (re-emitted as a direct, mangled function). The result `Env` has **every `params` empty**, **no
//! reachable [`Ty::Var`]**, and **no trait-method calls** — so the L1-eval ≡ L0-interp ≡ AOT
//! differential (NFR-7) runs on a single closed L0 program for generics *and* traits. **No
//! `mycelium-core` change** (KC-3): this is a pure frontend rewrite over the checked `Env`; the
//! kernel/registry path is untouched.
//!
//! It is **not** a tag-changing pass (VR-5 / S1). Totality is **recomputed** over the specialized
//! function set, never fabricated — a specialization's verdict equals its source's because the
//! rewrite is structural. [`subst_ty`] is Swap-free; mono never inserts a `Swap`.
//!
//! ## Per-instantiation guarantee-tag context (M-844 / M-967; RFC-0018 §4 / RFC-0019 §4.4)
//! Static guarantee grading ([`crate::grade::check_guarantees`]) runs **before** mono, over the
//! still-generic, checked `Env` (`checkty.rs`'s Pass 3d) — so the lattice `Exact ⊐ Proven ⊐
//! Empirical ⊐ Declared` is enforced exactly once, against each *declaration's* own written `@ g`
//! annotations, never re-derived or re-validated here (KC-3: no second grading pass). Mono's job
//! w.r.t. tags is narrower and purely custodial: when a declaration is specialized into a mangled
//! copy, that copy's signature (and any inline ascription in its rewritten body) must carry
//! **exactly its own source declaration's** `@ g` annotations — not the emitted-fresh
//! [`TypeRef::unguaranteed`] every reconstructed `Ty → TypeRef` round-trip otherwise produces.
//! Before M-967 every mono'd signature/ascription silently lost its `@ g` this way — a **silent
//! downgrade to "no annotation"**, never an upgrade, but still a transparency violation (a later
//! `EXPLAIN`/audit reader could no longer see what the source actually declared). Two call sites of
//! one generic reaching *different* trait instances (`emit_method`, keyed by `(trait, for_ty)`) is
//! the concrete case DN-64 OQ-S asks about: each instance's method has its **own** `@ g` on its own
//! declaration, so once threaded through, the two mangled specializations naturally carry two
//! **distinct** tags — never merged (each is its own `FnDecl`, keyed by its own mangled name) and
//! never bled into each other (mono never reads one instantiation's tag while emitting another).
//! [`ty_to_ref_tagged`] is the one helper every signature/ascription reconstruction routes through.
//!
//! # Honest identity fragmentation (NOT "one body, one hash")
//! The mangled-name scheme **is** the honest record: `first_or` specialized at `Binary{8}` and at
//! `Binary{4}` become **two distinct** functions `first_or$Binary8` and `first_or$Binary4`, each
//! with its own elaboration and content hash. This is identity *fragmentation*, recorded — not
//! hidden behind a single shared body. (Cross-instantiation sharing of structurally-identical L0
//! terms would be a separate, later content-addressing concern; mono does not claim it.)
//!
//! # Mangling: injective, surface-disjoint (`$` joints, `#` nullary-data tag)
//! Names are mangled with `$` (the joint separator) and a `#` kind-tag on a nullary data type —
//! neither is a surface-identifier character (the lexer never produces them), and the elaborator's
//! fresh variables use `%` ([`crate::elab`]). So a mangled name collides with **neither** a surface
//! name, **nor** a fresh elaboration variable, **nor across the repr/data boundary**: a data type
//! whose name happens to equal a repr mangle (e.g. a type literally named `Binary8`) tags to
//! `Binary8#`, which can never equal the repr `Binary{8}` → `Binary8`. The scheme is therefore
//! **injective** over every input it sees — distinct `(decl, type-args)` (and the repr set) map to
//! distinct names, so two instantiations never silently alias to one body (G2). A unit test pins
//! this, including the adversarial repr-named data type. **Empty type arguments ⇒ the original name,
//! byte-for-byte** (the `#` tag appears only inside a composite name; a monomorphic data type is
//! still registered and referenced under its bare name) — so monomorphic code and non-generic
//! programs pass through unchanged.
//!
//! # Still a `Residual` after M-673 (never-silent — kept explicit)
//! Mono refuses, with [`ElabError::Residual`], anything still outside the fragment: an
//! **undetermined** type parameter (a `Ty::Var` the checker would not let through, defended here too
//! — never guessed), multi-parameter traits / associated types, higher-order (`A -> B`) generics
//! (the surface is first-order — there is no function type), and `wild`/FFI, `spore`, VSA, and
//! `Substrate` (which have no v0 lowering regardless of generics). The generic/trait `Residual` sites
//! in [`crate::elab`] are **kept** as defensive internal invariants (G2): after mono they should be
//! unreachable, but they never silently disappear.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{
    Arm, BaseType, Expr, FnDecl, FnSig, Hypha, Literal, Param, Path, Pattern, Scalar, Sparsity,
    Strength, TypeRef, WidthRef,
};
use crate::checkty::{
    has_var, infer_type, param_subst, resolve_ty, subst_ty, type_head, unify, CtorInfo, DataInfo,
    Env, TraitInfo, Ty, Width,
};
use crate::elab::ElabError;
use crate::totality::{WalkDepthExceeded, MAX_WALK_DEPTH};

/// A reified **instance selection** (RFC-0019 §4.4; house rule #2 — no black boxes). When mono
/// lowers a trait-method call to a direct call, it records *which* instance it picked: the trait, the
/// concrete receiver type, and the mangled name of the emitted method function. The dispatch choice
/// is thus programmatically inspectable (`EXPLAIN`-able), not hidden inside the rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceSelection {
    /// The trait whose method was called.
    pub trait_name: String,
    /// The concrete receiver type the instance is `for` (the full type, not the head — e.g.
    /// `Binary{8}`, never just `Binary`).
    pub for_ty: Ty,
    /// The mangled name of the monomorphic function mono emitted for this instance's method (the
    /// direct callee the trait-method call was rewritten to — e.g. `cmp$Cmp$Binary8`).
    pub impl_mangled: String,
}

/// The **EXPLAIN record** of a monomorphization (M-673): every trait-method dispatch mono resolved,
/// keyed by the mangled callee name (which itself encodes `(method, trait, receiver)`). Populated by
/// [`monomorphize_with_selections`]; queryable so the dictionary-free static resolution is a
/// reified, inspectable record rather than a black box (house rule #2).
///
/// Extended in M-687 (RFC-0024 §4) to also record **HOF defunctionalization specializations**
/// (`hof_specs`): each static HOF specialization — the source fn, its type args, its baked-in
/// function arguments, and the mangled name — is recorded for full inspectability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MonoSelections {
    by_mangled: BTreeMap<String, InstanceSelection>,
    /// HOF defunctionalization records (RFC-0024 §4, M-687): keyed by the mangled HOF
    /// specialization name (e.g. `map$Binary8$Binary8%1:double`).
    pub(crate) hof_specs: BTreeMap<String, HofSpecialization>,
    /// RFC-0024 §4A (M-704; house rule #2): per-closure lowering records, keyed by the generated
    /// constructor name (`Clo$<arrow>$<n>`) — the capture set + generated apply dispatcher.
    pub(crate) closure_specs: BTreeMap<String, ClosureSpecialization>,
}

impl MonoSelections {
    /// The selection mono made for the mangled callee `mangled`, if any. The mangled name is what a
    /// rewritten trait-method call now refers to, so a consumer can map a direct call back to the
    /// trait/instance it came from.
    #[must_use]
    pub fn get(&self, mangled: &str) -> Option<&InstanceSelection> {
        self.by_mangled.get(mangled)
    }

    /// Every recorded selection, in deterministic (mangled-name) order. Additive read accessor.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &InstanceSelection)> {
        self.by_mangled.iter()
    }

    /// How many distinct trait-method instances were resolved (0 for a non-trait program).
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_mangled.len()
    }

    /// Were no trait-method selections recorded? (A non-trait program monomorphizes with an empty
    /// record.)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_mangled.is_empty()
    }

    /// The HOF defunctionalization record for the mangled specialization `mangled`, if any
    /// (RFC-0024 §4, M-687). Returns the source fn, type args, and baked-in function arguments.
    #[must_use]
    pub fn hof(&self, mangled: &str) -> Option<&HofSpecialization> {
        self.hof_specs.get(mangled)
    }

    /// Every recorded HOF specialization, in deterministic (mangled-name) order.
    pub fn hof_iter(&self) -> impl Iterator<Item = (&String, &HofSpecialization)> {
        self.hof_specs.iter()
    }

    /// The closure-lowering record for the generated constructor `ctor_name`, if any (RFC-0024 §4A,
    /// M-704) — the capture set + the generated apply dispatcher (house rule #2).
    #[must_use]
    pub fn closure(&self, ctor_name: &str) -> Option<&ClosureSpecialization> {
        self.closure_specs.get(ctor_name)
    }

    /// Every recorded closure lowering, in deterministic (constructor-name) order.
    pub fn closure_iter(&self) -> impl Iterator<Item = (&String, &ClosureSpecialization)> {
        self.closure_specs.iter()
    }
}

/// Monomorphize a checked `Env` from nullary monomorphic `entry`, returning a closed monomorphic
/// `Env` the existing [`crate::elab::elaborate`] runs unchanged.
///
/// On a program with **no** generics/traits this is a fast **pass-through** (a clone): monomorphic
/// code is mono's identity, so the pre-M-673 differential corpus is observably unchanged (NFR-7).
///
/// # Errors
/// [`ElabError::Residual`] for anything outside the monomorphizable fragment (an undetermined type
/// parameter, a multi-parameter trait, a higher-order generic, …) — never silent, never a guess
/// (G2/VR-5). [`ElabError::UnknownFn`] if `entry` is absent.
pub fn monomorphize(env: &Env, entry: &str) -> Result<Env, ElabError> {
    monomorphize_with_selections(env, entry).map(|(env, _)| env)
}

/// Like [`monomorphize`] but also returns the [`MonoSelections`] EXPLAIN record of every trait-method
/// dispatch resolved (house rule #2 — the static resolution is inspectable, not a black box).
///
/// # Errors
/// See [`monomorphize`].
pub fn monomorphize_with_selections(
    env: &Env,
    entry: &str,
) -> Result<(Env, MonoSelections), ElabError> {
    // Fast pass-through: a fully-monomorphic, non-trait program is mono's identity. Returning a clone
    // keeps the existing monomorphic differential corpus byte-identical (NFR-7) and avoids re-walking
    // a graph that has nothing to specialize.
    if is_already_monomorphic(env) {
        return Ok((env.clone(), MonoSelections::default()));
    }
    let mut m = Mono::new(env);
    m.run(entry)?;
    m.finish()
}

/// Is `env` already fully monomorphic, trait-free, **and** HOF-free? Then mono is the identity
/// (the fast pass-through). True iff **no** function is generic, **no** function has a fn-typed
/// value parameter (which needs defunctionalization — RFC-0024 §4, M-687), **no** data type is
/// generic, and there are **no** (user) traits / instances / retained impls.
///
/// **M-965 note:** every `Env` now carries the built-in `Fuse` prelude trait
/// ([`crate::fuse::TRAIT_NAME`]) even when the program never mentions `fuse`/`Fuse` — mirroring how
/// every `Env` already carries the built-in `Bool` prelude *type* (which doesn't break this check,
/// since `Bool` happens to have empty `params`). A trait registration with **zero** instances/impls
/// contributes nothing to specialize (an unimplemented trait can't be called — `check_fuse`/generic
/// dispatch always requires a resolved instance), so the trait-emptiness test here is *specifically*
/// "no **user-declared** trait" — the always-present builtin is excluded, exactly as `Bool` is
/// excluded from the phylum-wide coherence view (`checkty::check_phylum_inner`'s `name != "Bool"`
/// guard). `env.instances`/`env.impls` being empty is unaffected and still does the real work: a
/// program that actually declares `impl Fuse[T] for T` has a non-empty `instances`/`impls`, which
/// correctly still forces the full (specializing) pass.
fn is_already_monomorphic(env: &Env) -> bool {
    env.fns.values().all(|fd| {
        fd.sig.params.is_empty()
            && fd
                .sig
                .value_params
                .iter()
                .all(|p| !param_has_fn_type(&env.types, &fd.sig.param_names(), &p.ty))
            // RFC-0024 §4A (M-704): a body containing a `lambda`, or a **named fn used as an escaping
            // value** (a single-segment top-level fn name in value position, not as a call head),
            // needs closure lowering — it is **not** mono's identity (the pass-through would leave a
            // construct the elaborator/evaluator cannot run). Detected by a structural walk.
            && !body_has_lambda(&fd.body)
            && !body_has_fn_value(env, &fd.body)
    }) && env.types.values().all(|d| d.params.is_empty())
        && env
            .traits
            .keys()
            .all(|name| name == crate::fuse::TRAIT_NAME)
        && env.instances.is_empty()
        && env.impls.is_empty()
}

/// RFC-0024 §4A.4 (M-704): true iff `e` references a top-level **function name as a value** — a
/// single-segment `Expr::Path` naming a fn, occurring **not** as an application head (an applied fn
/// `f(x)` is an ordinary first-order call, not a fn value). Such a reference needs the closure
/// lowering (a named-fn-as-escaping-value becomes a nullary closure constructor), so the program is
/// **not** mono's fast-path identity. Over-detection is safe — it only ever widens to the full pass.
fn body_has_fn_value(env: &Env, e: &Expr) -> bool {
    match e {
        // A bare path naming a top-level fn, in value position.
        Expr::Path(p) => p.0.len() == 1 && env.fns.contains_key(&p.0[0]),
        Expr::Lit(Literal::List(elems)) => elems.iter().any(|x| body_has_fn_value(env, x)),
        Expr::Lit(_) => false,
        Expr::Let { bound, body, .. } => {
            body_has_fn_value(env, bound) || body_has_fn_value(env, body)
        }
        Expr::If { cond, conseq, alt } => {
            body_has_fn_value(env, cond)
                || body_has_fn_value(env, conseq)
                || body_has_fn_value(env, alt)
        }
        Expr::Match { scrutinee, arms } => {
            body_has_fn_value(env, scrutinee)
                || arms.iter().any(|a| body_has_fn_value(env, &a.body))
        }
        Expr::For { xs, init, body, .. } => {
            body_has_fn_value(env, xs)
                || body_has_fn_value(env, init)
                || body_has_fn_value(env, body)
        }
        Expr::Swap { value, .. } => body_has_fn_value(env, value),
        Expr::WithParadigm { body, .. } => body_has_fn_value(env, body),
        Expr::Wild(b) | Expr::Spore(b) | Expr::Consume(b) => body_has_fn_value(env, b),
        Expr::Colony(hyphae) => hyphae.iter().any(|h| body_has_fn_value(env, &h.body)),
        Expr::Lambda { body, .. } => body_has_fn_value(env, body),
        // The head of an application is a *call target*, not a fn value — do NOT descend into it as a
        // value. Only the arguments are value positions (a fn name passed as an arg is the §4/§4A
        // HOF-argument case, handled by `resolve_fn_args` — but it still means "not the identity").
        Expr::App { head: _, args } => args.iter().any(|x| body_has_fn_value(env, x)),
        Expr::Fuse { left, right } => body_has_fn_value(env, left) || body_has_fn_value(env, right),
        Expr::Reclaim { policy, body } => {
            body_has_fn_value(env, policy) || body_has_fn_value(env, body)
        }
        Expr::Ascribe(inner, _) => body_has_fn_value(env, inner),
        // M-826: a tuple literal is a value-forming expression; its elements may reference fn values.
        Expr::TupleLit(elems) => elems.iter().any(|x| body_has_fn_value(env, x)),
    }
}

/// True iff `e` contains an `Expr::Lambda` anywhere (RFC-0024 §4A, M-704) — the gate that keeps a
/// closure-bearing program out of mono's fast pass-through (it needs the §4A lowering).
fn body_has_lambda(e: &Expr) -> bool {
    match e {
        Expr::Lambda { .. } => true,
        Expr::Lit(Literal::List(elems)) => elems.iter().any(body_has_lambda),
        Expr::Lit(_) | Expr::Path(_) => false,
        Expr::Let { bound, body, .. } => body_has_lambda(bound) || body_has_lambda(body),
        Expr::If { cond, conseq, alt } => {
            body_has_lambda(cond) || body_has_lambda(conseq) || body_has_lambda(alt)
        }
        Expr::Match { scrutinee, arms } => {
            body_has_lambda(scrutinee) || arms.iter().any(|a| body_has_lambda(&a.body))
        }
        Expr::For { xs, init, body, .. } => {
            body_has_lambda(xs) || body_has_lambda(init) || body_has_lambda(body)
        }
        Expr::Swap { value, .. } => body_has_lambda(value),
        Expr::WithParadigm { body, .. } => body_has_lambda(body),
        Expr::Wild(b) | Expr::Spore(b) | Expr::Consume(b) => body_has_lambda(b),
        Expr::Colony(hyphae) => hyphae.iter().any(|h| body_has_lambda(&h.body)),
        Expr::App { head, args } => body_has_lambda(head) || args.iter().any(body_has_lambda),
        Expr::Fuse { left, right } => body_has_lambda(left) || body_has_lambda(right),
        Expr::Reclaim { policy, body } => body_has_lambda(policy) || body_has_lambda(body),
        Expr::Ascribe(inner, _) => body_has_lambda(inner),
        // M-826: a tuple literal may contain lambda expressions in its elements.
        Expr::TupleLit(elems) => elems.iter().any(body_has_lambda),
    }
}

/// True iff the parameter type `t` resolves to (or contains) a `Ty::Fn` — meaning this parameter
/// is a HOF that needs defunctionalization (RFC-0024 §4, M-687). Best-effort: a resolution failure
/// is treated as "not fn-typed" (the full mono pass will catch it with an explicit Residual).
fn param_has_fn_type(
    types: &BTreeMap<String, crate::checkty::DataInfo>,
    tyvars: &[String],
    t: &TypeRef,
) -> bool {
    use crate::ast::BaseType;
    match &t.base {
        BaseType::Fn(_, _) => true,
        BaseType::Named(n, args) => {
            // A type variable or a data type with fn-typed arguments — check args recursively.
            // A data type itself (not a type var) is not fn-typed; a type variable is also not
            // fn-typed at the surface level (it resolves to a concrete type at specialization time,
            // which may or may not be `Ty::Fn`; we conservatively say false here and let the full
            // mono pass handle it with an explicit Residual if it turns out fn-typed).
            if tyvars.contains(n) || types.contains_key(n.as_str()) {
                return false; // bare type var or concrete data type — not a fn type itself
            }
            // Otherwise check args (e.g. `F<A->B>` would have a fn-typed arg — exotic but safe).
            args.iter().any(|a| param_has_fn_type(types, tyvars, a))
        }
        _ => false, // Binary/Ternary/Dense/Substrate — never fn-typed
    }
}

/// A monomorphization work item — the unit of the dedup worklist. Deduplication is by the item's
/// canonical [`item_key`] (a discriminant-tagged mangled string), so a `BTreeSet<String>` of seen
/// keys guarantees each specialization is emitted **once** (dedup ⟹ the recursive walk terminates).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Item {
    /// A function instance: the source fn `name` at concrete type arguments `targs` (empty for a
    /// monomorphic fn — which mangles to `name` unchanged), optionally specialised by resolved
    /// **function-argument** identities (RFC-0024 §4, M-687). `fn_args` carries `(param_index,
    /// callee_mangled_name)` for each value-parameter whose type is `Ty::Fn`; empty means no
    /// higher-order specialization. An `Item::Fn` with non-empty `fn_args` is a defunctionalized
    /// HOF specialization — distinct from the un-specialized (or differently-specialized) version
    /// of the same fn.
    Fn {
        name: String,
        targs: Vec<Ty>,
        /// Resolved width arguments in declaration order (DN-42 / M-753 step-c): one `Width::Lit`
        /// per width parameter of the callee. Baked into the item key so two calls at different
        /// widths produce distinct specializations (never a silent alias — G2/VR-5).
        wargs: Vec<Width>,
        /// `(param_index, callee_mangled_name)` for each fn-typed value parameter, sorted by
        /// param index (deterministic). Baked into the item key so two different function
        /// arguments produce two distinct specializations (never a silent alias — G2).
        fn_args: Vec<(usize, String)>,
        /// RFC-0024 §4A (M-704): `(param_index, arrow_mangle)` for each fn-typed value parameter
        /// that is **kept** as a dynamic closure value (it received a lambda or a dynamically-flowing
        /// fn value — §4's static resolution does not apply). Such a parameter stays in the emitted
        /// signature typed `Fn$<arrow>`; an application of it inside the body lowers to
        /// `apply$<arrow>(f, x)`. Sorted by index (deterministic), baked into the item key so a HOF
        /// specialized dynamically is distinct from its static (or un-)specialization (G2).
        dyn_fns: Vec<(usize, String)>,
    },
    /// A data-type instance: the source type `name` at concrete `targs`.
    Data { name: String, targs: Vec<Ty> },
    /// A trait-method instance: the unqualified method `method` of trait `trait_name`, resolved at the
    /// concrete receiver `for_ty` (the coherent instance's method, emitted as a direct fn).
    Method {
        trait_name: String,
        method: String,
        for_ty: Ty,
    },
}

/// The **EXPLAIN record** of a single HOF defunctionalization (RFC-0024 §4, M-687): which
/// higher-order function was specialized, at which type arguments, with which function arguments
/// baked in. Recorded in [`MonoSelections`] so the static dispatch is inspectable (house rule #2
/// — no black boxes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HofSpecialization {
    /// The source (polymorphic / HOF) function name.
    pub source_fn: String,
    /// The concrete type arguments (empty if the HOF was monomorphic).
    pub targs: Vec<Ty>,
    /// The resolved function argument(s): `(param_index, callee_mangled_name)`, parallel to the
    /// `fn_args` of the [`Item::Fn`] that triggered this specialization.
    pub fn_args: Vec<(usize, String)>,
    /// The mangled name of the emitted closed-first-order specialization.
    pub mangled: String,
}

/// The monomorphization driver: the source (checked, generic) env, the dedup worklist, and the
/// accumulating monomorphic output (`fns`/`types`) plus the EXPLAIN selection record.
struct Mono<'e> {
    src: &'e Env,
    /// Canonical keys of items already enqueued (dedup) — guarantees one emission per specialization
    /// (termination). Keyed by [`item_key`] so `Ty` needs no `Ord` (it is `Eq` only).
    seen: BTreeSet<String>,
    /// The pending worklist (LIFO; order does not affect the result — emission is keyed by mangled
    /// name into `BTreeMap`s).
    work: Vec<Item>,
    /// Emitted monomorphic functions (mangled name → closed `FnDecl`).
    out_fns: BTreeMap<String, FnDecl>,
    /// Emitted monomorphic data types (mangled name → `DataInfo` with empty `params`).
    out_types: BTreeMap<String, DataInfo>,
    /// The reified trait-method dispatch record (house rule #2).
    selections: BTreeMap<String, InstanceSelection>,
    /// HOF defunctionalization specialization records (RFC-0024 §4, M-687; house rule #2).
    hof_specs: BTreeMap<String, HofSpecialization>,
    /// Active fn-parameter substitution during HOF specialization emission (M-687): maps a
    /// value-parameter name whose type is `Ty::Fn` to the mangled name of its resolved callee.
    /// Populated by [`emit_fn`] when `fn_args` is non-empty; cleared after each emission. Only
    /// consulted in [`rewrite_hof_app`].
    fn_param_subst: BTreeMap<String, String>,
    /// RFC-0024 §4A (M-704): the accumulating per-arrow **closure tag-sums**. Keyed by the arrow
    /// mangle (`Fn$<A>$<B>`); each value holds the arrow's domain/codomain and one constructor per
    /// distinct escaping closure of that arrow (the §4A.4 fn-tag sum). Emitted **once per arrow** at
    /// [`Self::finish`] (after the worklist drains — the whole-program closure set is then complete,
    /// §4A.5; no open-world fallback arm). Reuses the existing data + match + call L0 constructs, so
    /// **no `mycelium-core` node is added** (KC-3).
    closures: BTreeMap<String, ClosureSum>,
    /// RFC-0024 §4A (M-704; house rule #2): per-closure EXPLAIN records — the capture set + the
    /// generated constructor + apply dispatcher, keyed by the closure constructor name. The dynamic
    /// dispatch is thus a reified, inspectable record, never a black box.
    closure_specs: BTreeMap<String, ClosureSpecialization>,
    /// Active **dynamic** fn-parameter map during a HOF specialization emission (M-704): maps a
    /// value-parameter name whose type is `Ty::Fn` and which is **kept** as a closure value (it
    /// received a non-statically-known argument — a lambda or a dynamic fn value) to its arrow mangle.
    /// An application `f(x)` of such a parameter rewrites to `apply$<arrow>(f, x)`. Populated by
    /// [`Self::emit_fn`]; cleared after each emission. Disjoint from `fn_param_subst` (static params).
    dyn_fn_param: BTreeMap<String, String>,
}

/// RFC-0024 §4A.4 (M-704): one closure **tag-sum** for a single arrow type `A => B`. Accumulates a
/// constructor per distinct escaping closure of that arrow (deduplicated by a content key), plus the
/// arrow's concrete domain/codomain so [`Mono::finish`] can emit the sum's `DataInfo` and the
/// `apply$A$B` dispatcher (`emit_data`/`emit_fn` — existing emitters; no new L0 node, KC-3).
#[derive(Debug, Clone)]
struct ClosureSum {
    /// The arrow's domain type `A` (concrete).
    arrow_a: Ty,
    /// The arrow's codomain type `B` (concrete).
    arrow_b: Ty,
    /// One constructor per distinct closure, in deterministic registration order.
    ctors: Vec<ClosureCtor>,
    /// Content-key → constructor index, for deduplication (two structurally identical closures of the
    /// same arrow share one constructor — the §4 identity-fragmentation discipline; G2).
    by_key: BTreeMap<String, usize>,
}

/// RFC-0024 §4A.4 (M-704): one constructor of a closure tag-sum — a single escaping closure's
/// captured environment + its (rewritten) body. The dispatcher arm binds the captures, binds the
/// lambda's parameter to the dispatcher's argument, then evaluates the body.
#[derive(Debug, Clone)]
struct ClosureCtor {
    /// The constructor name (`Clo$<arrow>$<n>` — injective via the `$` joints; §4 mangling ground).
    ctor_name: String,
    /// The captured free variables, `(name, concrete type)`, in deterministic (first-occurrence)
    /// order. Empty for a captureless lambda or a bare named fn (a nullary constructor).
    captures: Vec<(String, Ty)>,
    /// The lambda's single parameter name (stage-1 single-arg — §4A.8 gates multi-arg on tuples).
    param_name: String,
    /// The (already-rewritten) lambda body. References the captures (bound by the dispatcher's match
    /// arm) and `param_name` (bound to the dispatcher's argument via a `Let`).
    body: Expr,
}

/// RFC-0024 §4A.6 (M-704; house rule #2): the EXPLAIN record of one closure lowering — which arrow,
/// which captured variables, which generated constructor + dispatcher. Mirrors [`HofSpecialization`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureSpecialization {
    /// The arrow mangle (`Fn$<A>$<B>`) this closure inhabits.
    pub arrow: String,
    /// The generated constructor name (`Clo$<arrow>$<n>`).
    pub ctor_name: String,
    /// The captured free variables, `(name, type-display)`, in capture order.
    pub captures: Vec<(String, String)>,
    /// The generated apply-dispatcher's name (`apply$<A>$<B>`).
    pub apply_fn: String,
}

impl<'e> Mono<'e> {
    fn new(src: &'e Env) -> Self {
        Mono {
            src,
            seen: BTreeSet::new(),
            work: Vec::new(),
            out_fns: BTreeMap::new(),
            out_types: BTreeMap::new(),
            selections: BTreeMap::new(),
            hof_specs: BTreeMap::new(),
            fn_param_subst: BTreeMap::new(),
            closures: BTreeMap::new(),
            closure_specs: BTreeMap::new(),
            dyn_fn_param: BTreeMap::new(),
        }
    }

    /// Enqueue an item if it has not been seen (dedup ⟹ termination).
    fn enqueue(&mut self, item: Item) {
        if self.seen.insert(item_key(&item)) {
            self.work.push(item);
        }
    }

    /// Seed from the nullary monomorphic `entry` and drain the worklist, specializing each item.
    fn run(&mut self, entry: &str) -> Result<(), ElabError> {
        let Some(fd) = self.src.fns.get(entry) else {
            return Err(ElabError::UnknownFn(entry.to_owned()));
        };
        if !fd.sig.params.is_empty() {
            return residual(
                entry,
                "monomorphization entry is generic — elaborate a concrete (nullary, monomorphic) \
                 entry (RFC-0007 §11.3)",
            );
        }
        self.enqueue(Item::Fn {
            name: entry.to_owned(),
            targs: vec![],
            wargs: vec![], // entry is monomorphic (nullary, no width params)
            fn_args: vec![],
            dyn_fns: vec![],
        });
        while let Some(item) = self.work.pop() {
            match item {
                Item::Fn {
                    name,
                    targs,
                    wargs,
                    fn_args,
                    dyn_fns,
                } => self.emit_fn(&name, &targs, &wargs, &fn_args, &dyn_fns)?,
                Item::Data { name, targs } => self.emit_data(&name, &targs)?,
                Item::Method {
                    trait_name,
                    method,
                    for_ty,
                } => self.emit_method(&trait_name, &method, &for_ty)?,
            }
        }
        // RFC-0024 §4A.5/§4A.7 (M-704): the worklist has drained, so the whole-program closure set
        // is now complete (every reachable lambda/dynamic-fn has contributed its constructor). Emit
        // each arrow's tag-sum + `apply` dispatcher — closed first-order L0, no open-world arm.
        self.emit_closures()?;
        Ok(())
    }

    /// Consume the driver into the closed monomorphic [`Env`] plus its [`MonoSelections`] EXPLAIN
    /// record: the emitted fns/types, recomputed totality, and **empty** trait/instance/impl
    /// registries (no generics/traits remain).
    ///
    /// # Errors
    /// [`ElabError::DepthExceeded`] if the totality pass's own AST-traversal recursion exceeds its
    /// explicit depth budget ([`crate::totality::MAX_WALK_DEPTH`]; M-674) on a pathologically-nested
    /// specialized body — a clean, explicit *budget* refusal (never a host-stack overflow, and never
    /// mis-reported as a semantic `Residual`; consistent with `elab::collect_calls`).
    fn finish(self) -> Result<(Env, MonoSelections), ElabError> {
        // Recompute totality over the specialized fn set (a specialization's verdict equals its
        // source's; the SCC/descent machinery is structural — totality.rs). The matured gate and the
        // elaborator's SCC pass then read verdicts by the *mangled* names. Never fabricated (VR-5).
        // A depth-budget trip here is the never-silent `DepthExceeded` (M-674), NOT a `Residual`
        // (which would conflate a resource limit with an "outside-the-fragment" semantic verdict).
        let totality =
            crate::totality::classify_all(&self.out_fns).map_err(|e| ElabError::DepthExceeded {
                site: "<monomorphization>".to_owned(),
                limit: e.limit,
            })?;
        let env = Env {
            types: self.out_types,
            fns: self.out_fns,
            totality,
            traits: BTreeMap::new(),
            instances: BTreeMap::new(),
            impls: BTreeMap::new(),
            // DN-54 / M-812: monomorphized specializations do not carry lower rules (the rule
            // registry is a pre-mono artefact — rules are expanded before/at elaborate time). The
            // same holds for derive-provenance (M-973) and via-provenance (M-966): both are recorded
            // during nodule checking, before monomorphization, so a mono specialization carries none.
            lower_rules: BTreeMap::new(),
            derived_provenance: BTreeMap::new(),
            via_provenance: BTreeMap::new(),
        };
        Ok((
            env,
            MonoSelections {
                by_mangled: self.selections,
                hof_specs: self.hof_specs,
                closure_specs: self.closure_specs,
            },
        ))
    }

    /// Specialize source function `name` at concrete `targs` (and optionally with baked-in
    /// function arguments `fn_args` for HOF defunctionalization — RFC-0024 §4, M-687) and emit
    /// the monomorphic `FnDecl` under its mangled name. Discovers transitive instances by walking
    /// (and rewriting) the body.
    ///
    /// When `fn_args` is non-empty: each `(param_index, callee_mangled)` pair names a
    /// value-parameter whose declared type is `Ty::Fn` and the statically-resolved callee that was
    /// passed at the call site. The specialized body replaces every application of that fn-param
    /// with a direct call to the callee, and the fn-param is **dropped** from the emitted
    /// value-parameter list — producing a closed first-order signature (no `Ty::Fn` in params).
    ///
    /// When `dyn_fns` is non-empty (RFC-0024 §4A, M-704): each `(param_index, arrow_mangle)` pair
    /// names a fn-typed value-parameter **kept** as a closure value — its emitted type becomes the
    /// closure tag-sum `Fn$<arrow>` (an ordinary `Ty::Data`), and every application of it inside the
    /// body is rewritten to `apply$<arrow>(f, x)`. The parameter is **not** dropped (the closure is
    /// passed at the call site).
    fn emit_fn(
        &mut self,
        name: &str,
        targs: &[Ty],
        wargs: &[Width],
        fn_args: &[(usize, String)],
        dyn_fns: &[(usize, String)],
    ) -> Result<(), ElabError> {
        let mangled = mangle_hof_decl(name, targs, wargs, fn_args, dyn_fns);
        // Already emitted? (the worklist dedups, but a defensive check keeps emission idempotent.)
        if self.out_fns.contains_key(&mangled) {
            return Ok(());
        }
        let fd = self
            .src
            .fns
            .get(name)
            .ok_or_else(|| ElabError::UnknownFn(name.to_owned()))?
            .clone();
        let tyvars = fd.sig.param_names();
        if tyvars.len() != targs.len() {
            return residual(
                name,
                format!(
                    "internal: `{name}` has {} type parameter(s) but was queued with {} argument(s)",
                    tyvars.len(),
                    targs.len()
                ),
            );
        }
        let mut subst: BTreeMap<String, Ty> = param_subst(&tyvars, targs);

        // DN-42 / M-753 step-c: inject width-arg carriers into the shared subst map so that
        // `subst_ty` resolves `Width::Var(v)` in parameter/return types to the concrete literal.
        // Carrier convention: `var_name → Ty::Binary(Width::Lit(n))` regardless of paradigm —
        // `subst_ty` extracts the right paradigm (Binary or Ternary) from the carrier. An
        // undetermined width var at emit time is an internal invariant violation (VR-5).
        let wvars = fd.sig.width_param_names();
        if wvars.len() != wargs.len() {
            return residual(
                name,
                format!(
                    "internal: `{name}` has {} width parameter(s) but was queued with {} \
                     width argument(s) — an invariant violation (DN-42 / M-753 step-c)",
                    wvars.len(),
                    wargs.len()
                ),
            );
        }
        for (v, w) in wvars.iter().zip(wargs.iter()) {
            match w {
                Width::Lit(n) => {
                    subst.insert(v.clone(), Ty::Binary(Width::Lit(*n)));
                }
                Width::Var(wv) => {
                    return residual(
                        name,
                        format!(
                            "width param `{v}` of `{name}` is still a variable `{wv}` at emit \
                             — undetermined width is never guessed (DN-42 §4 / VR-5)"
                        ),
                    );
                }
            }
        }

        // Build the fn-parameter substitution map for HOF defunctionalization:
        //   fn_param_name → callee_mangled_name
        // and validate that each fn-arg index names an actual fn-typed param.
        let fn_arg_map: BTreeMap<String, String> = fn_args
            .iter()
            .map(|(idx, callee)| {
                let pname = fd
                    .sig
                    .value_params
                    .get(*idx)
                    .map(|p| p.name.clone())
                    .ok_or_else(|| ElabError::Residual {
                        site: name.to_owned(),
                        what: format!(
                            "HOF fn_arg index {idx} out of bounds for `{name}` (internal)"
                        ),
                    })?;
                Ok((pname, callee.clone()))
            })
            .collect::<Result<_, ElabError>>()?;

        // Set of param indices that are fn-typed and will be dropped from the emitted signature
        // (the §4 static path). The §4A dynamic params are KEPT (typed `Fn$<arrow>`).
        let dropped_indices: BTreeSet<usize> = fn_args.iter().map(|(i, _)| *i).collect();
        // RFC-0024 §4A (M-704): `param_index → arrow_mangle` for kept dynamic closure params.
        let dyn_by_idx: BTreeMap<usize, String> = dyn_fns.iter().cloned().collect();
        // `param_name → arrow_mangle` for kept dynamic closure params (the body-rewrite map).
        let mut dyn_param_map: BTreeMap<String, String> = BTreeMap::new();

        // The concrete value-parameter scope (param name → substituted concrete type), for
        // re-inferring sub-expression types while walking the body. Fn-typed params that are being
        // defunctionalized are added to scope at their `Ty::Fn` type (so re-inference still works),
        // but are **not** emitted in `new_params` (they are dropped from the closed-first-order sig).
        let mut scope: Vec<(String, Ty)> = Vec::with_capacity(fd.sig.value_params.len());
        let mut new_params: Vec<Param> = Vec::with_capacity(fd.sig.value_params.len());
        for (idx, p) in fd.sig.value_params.iter().enumerate() {
            let cty = self.concrete_ty(name, &tyvars, &subst, &p.ty)?;
            // Enqueue any generic data instance the parameter type names, so a type that appears
            // only as a parameter (never destructured in this body) is still emitted (insurance;
            // dedup makes it idempotent). Skip for Ty::Fn — no data enqueuing needed.
            if !matches!(cty, Ty::Fn(_, _)) {
                self.enqueue_tys_in(&cty);
            }
            scope.push((p.name.clone(), cty.clone()));
            if dropped_indices.contains(&idx) {
                // §4 static: defunctionalized away — not emitted.
                continue;
            }
            if let Some(arrow) = dyn_by_idx.get(&idx) {
                // §4A dynamic: kept, but its emitted type is the closure tag-sum `Fn$<arrow>` (an
                // ordinary nullary `Ty::Data`); the worklist already scheduled the sum + `apply`.
                dyn_param_map.insert(p.name.clone(), arrow.clone());
                new_params.push(Param {
                    name: p.name.clone(),
                    // M-967: the closure param's own source `@ g` (if any) still threads through —
                    // only its *base type* becomes the synthetic closure tag-sum.
                    ty: ty_to_ref_tagged(&Ty::Data(arrow.clone(), vec![]), p.ty.guarantee),
                });
                continue;
            }
            // M-967/M-844: this parameter's own `@ g` (from `p.ty`, the source declaration being
            // specialized) is threaded onto the emitted concrete type — the per-instantiation tag
            // context, never a fresh unguaranteed type (VR-5 / DN-64 OQ-S).
            new_params.push(Param {
                name: p.name.clone(),
                ty: ty_to_ref_tagged(&cty, p.ty.guarantee),
            });
        }
        let ret_cty = self.concrete_ty(name, &tyvars, &subst, &fd.sig.ret)?;
        self.enqueue_tys_in(&ret_cty);

        // Install the HOF fn-param substitution maps for the duration of this body rewrite.
        debug_assert!(
            self.fn_param_subst.is_empty() && self.dyn_fn_param.is_empty(),
            "fn_param_subst / dyn_fn_param must be empty before entering emit_fn (invariant)"
        );
        self.fn_param_subst = fn_arg_map;
        self.dyn_fn_param = dyn_param_map;

        // The declared return type drives return-position inference (e.g. a bare nullary generic
        // ctor, or a return-driven trait-method receiver), mirroring the checker's `expected`.
        let body_result = self.rewrite(name, &mut scope, &fd.body, Some(&ret_cty));

        // Always clear the substitution maps — even on error.
        self.fn_param_subst.clear();
        self.dyn_fn_param.clear();

        let new_body = body_result?;
        let new_sig = FnSig {
            name: mangled.clone(),
            params: vec![], // monomorphic: no type parameters remain
            value_params: new_params,
            // M-967/M-844: the source's own declared return `@ g` threads onto this instantiation's
            // return type — this specialization's own tag context, not a fresh unguaranteed one.
            ret: ty_to_ref_tagged(&ret_cty, fd.sig.ret.guarantee),
            effects: fd.sig.effects.clone(),
            effect_budgets: fd.sig.effect_budgets.clone(),
        };
        self.out_fns.insert(
            mangled.clone(),
            FnDecl {
                vis: fd.vis,
                thaw: fd.thaw,
                tier: fd.tier,
                sig: new_sig,
                body: new_body,
            },
        );
        // EXPLAIN: if this was a HOF specialization (static §4 or dynamic §4A), record it (house
        // rule #2 — no black boxes). The `fn_args` capture the statically-baked callees; the dynamic
        // closure dispatch is additionally recorded per-closure in `closure_specs`.
        if !fn_args.is_empty() || !dyn_fns.is_empty() {
            self.hof_specs.insert(
                mangled.clone(),
                HofSpecialization {
                    source_fn: name.to_owned(),
                    targs: targs.to_vec(),
                    fn_args: fn_args.to_vec(),
                    mangled,
                },
            );
        }
        Ok(())
    }

    /// Specialize source data type `name` at concrete `targs` and emit the monomorphic [`DataInfo`]
    /// (empty `params`; fields rewritten to mangled-nullary `Ty::Data`). Constructor names are mangled
    /// so distinct instantiations never collide on a ctor name (the registry/`Env::ctor` key).
    fn emit_data(&mut self, name: &str, targs: &[Ty]) -> Result<(), ElabError> {
        let mangled = mangle_decl(name, targs);
        if self.out_types.contains_key(&mangled) {
            return Ok(());
        }
        let d = self
            .src
            .types
            .get(name)
            .ok_or_else(|| ElabError::Residual {
                site: name.to_owned(),
                what: format!("unknown data type `{name}` during monomorphization"),
            })?
            .clone();
        if d.params.len() != targs.len() {
            return residual(
                name,
                format!(
                    "internal: data `{name}` has {} type parameter(s) but was queued with {}",
                    d.params.len(),
                    targs.len()
                ),
            );
        }
        let subst = param_subst(&d.params, targs);
        let mut ctors: Vec<CtorInfo> = Vec::with_capacity(d.ctors.len());
        for c in &d.ctors {
            let mut fields: Vec<Ty> = Vec::with_capacity(c.fields.len());
            for f in &c.fields {
                let cf = subst_ty(f, &subst);
                if has_var(&cf) {
                    return residual(
                        name,
                        format!(
                            "data `{name}` field stays abstract ({cf}) after substitution — an \
                             undetermined type parameter is never guessed (RFC-0007 §11.3)"
                        ),
                    );
                }
                // RFC-0024 §4A (M-704): a **fn-typed field** (a data type storing a closure — the
                // dynamic-fn-as-field shape) lowers to the closure tag-sum `Fn$<arrow>` (an ordinary
                // nullary `Ty::Data`). Register the arrow so its sum + `apply` are emitted, exactly as
                // a fn-typed parameter does. Non-fn fields take the existing mangled-nullary form.
                if let Ty::Fn(a, b) = &cf {
                    let arrow = self.register_arrow(a, b);
                    fields.push(Ty::Data(arrow, vec![]));
                } else {
                    // Enqueue any data instance the field references, and rewrite it to its
                    // mangled-nullary form so the registry/`field_spec` consumes the already-working
                    // `Ty::Data(n, [])` arm.
                    self.enqueue_tys_in(&cf);
                    fields.push(mangle_ty_in_ty(&cf));
                }
            }
            ctors.push(CtorInfo {
                name: mangle_ctor(&c.name, targs),
                fields,
            });
        }
        self.out_types.insert(
            mangled.clone(),
            DataInfo {
                name: mangled,
                params: vec![],
                ctors,
            },
        );
        Ok(())
    }

    /// Statically resolve trait `trait_name`'s method `method` at concrete receiver `for_ty` and emit
    /// the instance's resolved method body as a direct monomorphic fn under the mangled method name.
    /// Records the [`InstanceSelection`] (EXPLAIN). The instance was confirmed during checking
    /// (`require_instance`), so resolution here is deterministic — never a guess (G2).
    fn emit_method(
        &mut self,
        trait_name: &str,
        method: &str,
        for_ty: &Ty,
    ) -> Result<(), ElabError> {
        let mangled = mangle_method(method, trait_name, for_ty);
        if self.out_fns.contains_key(&mangled) {
            return Ok(());
        }
        let Some(head) = type_head(for_ty) else {
            return residual(
                method,
                format!(
                    "trait-method receiver `{for_ty}` has no concrete instance head — a blanket / \
                     abstract receiver is not a stage-1 instance (RFC-0019 §4.5)"
                ),
            );
        };
        let key = (trait_name.to_owned(), head);
        let methods = self
            .src
            .impls
            .get(&key)
            .ok_or_else(|| ElabError::Residual {
                site: method.to_owned(),
                what: format!(
                "no retained impl methods for `({trait_name}, {for_ty})` — the instance was not \
                 found during monomorphization (RFC-0019 §4.5 / M-673)"
            ),
            })?;
        // Resolution must match the FULL receiver (head-erasure is the coherence key, not the
        // resolution key — a `Binary{8}` instance must not serve a `Binary{4}` call; G2). The retained
        // instance's concrete `for_ty` is on record in `src.instances`.
        if let Some(info) = self.src.instance(trait_name, &key.1) {
            if info.for_ty != *for_ty {
                return residual(
                    method,
                    format!(
                        "the `{trait_name}` instance on this head is for `{}`, not `{for_ty}` — \
                         never a silently reused mismatched instance (RFC-0019 §4.5)",
                        info.for_ty
                    ),
                );
            }
        }
        let md = methods
            .iter()
            .find(|m| m.sig.name == method)
            .ok_or_else(|| ElabError::Residual {
                site: method.to_owned(),
                what: format!("instance `({trait_name}, {for_ty})` has no method `{method}`"),
            })?
            .clone();
        // An impl method over a concrete `for_ty` carries no abstract type-variables (the checker
        // resolved its param/return types concretely), so the empty substitution is correct; we still
        // route through `concrete_ty` to defend the no-`Ty::Var` invariant.
        let empty: BTreeMap<String, Ty> = BTreeMap::new();
        let mut scope: Vec<(String, Ty)> = Vec::with_capacity(md.sig.value_params.len());
        let mut new_params: Vec<Param> = Vec::with_capacity(md.sig.value_params.len());
        for p in &md.sig.value_params {
            let cty = self.concrete_ty(method, &[], &empty, &p.ty)?;
            self.enqueue_tys_in(&cty);
            scope.push((p.name.clone(), cty.clone()));
            // M-967/M-844: this instance method's own declared `@ g` (from its `impl` block, which
            // may differ from another instance of the same trait/method) threads onto its
            // specialization — the per-instance tag context (DN-64 OQ-S), never merged with a
            // sibling instance's tag and never dropped to unguaranteed.
            new_params.push(Param {
                name: p.name.clone(),
                ty: ty_to_ref_tagged(&cty, p.ty.guarantee),
            });
        }
        let ret_cty = self.concrete_ty(method, &[], &empty, &md.sig.ret)?;
        self.enqueue_tys_in(&ret_cty);
        let new_body = self.rewrite(method, &mut scope, &md.body, Some(&ret_cty))?;
        self.out_fns.insert(
            mangled.clone(),
            FnDecl {
                vis: md.vis,
                thaw: md.thaw,
                tier: md.tier,
                sig: FnSig {
                    name: mangled.clone(),
                    params: vec![],
                    value_params: new_params,
                    // M-967/M-844: this instance method's own declared return `@ g` threads onto its
                    // specialization's return type.
                    ret: ty_to_ref_tagged(&ret_cty, md.sig.ret.guarantee),
                    effects: md.sig.effects.clone(),
                    effect_budgets: md.sig.effect_budgets.clone(),
                },
                body: new_body,
            },
        );
        // EXPLAIN: record the resolved selection, keyed by the mangled callee (which encodes
        // method+trait+receiver). Inspectable, not a black box (house rule #2).
        self.selections.insert(
            mangled.clone(),
            InstanceSelection {
                trait_name: trait_name.to_owned(),
                for_ty: for_ty.clone(),
                impl_mangled: mangled,
            },
        );
        Ok(())
    }

    /// RFC-0024 §4A.4/§4A.7 (M-704): emit, once per registered arrow, the closure **tag-sum** data
    /// type (`emit_data`'s node — a `DataInfo`) and the **`apply` dispatcher** fn (`emit_fn`'s node —
    /// an ordinary `FnDecl` whose body is an `Expr::Match`). Both are constructs the trusted
    /// elaborator / `mycelium-core` registry already lower **unchanged** — **no new L0 node** (KC-3).
    /// Run after the worklist drains, so the whole-program closure set is complete (§4A.5): the sum's
    /// constructors are exactly the program's reachable closures of that arrow — closed, no
    /// open-world fallback arm.
    fn emit_closures(&mut self) -> Result<(), ElabError> {
        // Clone the accumulated arrows out so the emission can borrow `self` mutably (the `closures`
        // map is finalized — the worklist has drained, so no new closure can be registered now).
        let arrows: Vec<(String, ClosureSum)> = self
            .closures
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (arrow, sum) in arrows {
            // (1) The tag-sum `DataInfo`: one constructor per distinct closure; fields = capture
            // types (a fn-typed capture is its own closure data type `Fn$<inner-arrow>`).
            let ctors: Vec<CtorInfo> = sum
                .ctors
                .iter()
                .map(|c| CtorInfo {
                    name: c.ctor_name.clone(),
                    fields: c
                        .captures
                        .iter()
                        .map(|(_, t)| closure_field_ty(t))
                        .collect(),
                })
                .collect();
            self.out_types.insert(
                arrow.clone(),
                DataInfo {
                    name: arrow.clone(),
                    params: vec![],
                    ctors,
                },
            );

            // (2) The `apply$A$B(clo: Fn$A$B, %fnarg: A) -> B` dispatcher: `match clo { Clo_i(caps…)
            // => let <param_i> = %fnarg in body_i }`. The arm binds the captures by their original
            // names (so the body's capture references resolve), then binds the lambda's parameter to
            // the dispatcher's argument via a `Let` (robust to shadowing inside the body — no rename).
            let apply_name = apply_fn_name(&arrow);
            let arms: Vec<Arm> = sum
                .ctors
                .iter()
                .map(|c| {
                    let subs: Vec<Pattern> = c
                        .captures
                        .iter()
                        .map(|(n, _)| Pattern::Ident(n.clone()))
                        .collect();
                    let arm_body = Expr::Let {
                        name: c.param_name.clone(),
                        ty: None,
                        bound: Box::new(Expr::Path(Path(vec![APPLY_PARAM.to_owned()]))),
                        body: Box::new(c.body.clone()),
                    };
                    Arm {
                        pattern: Pattern::Ctor(c.ctor_name.clone(), subs),
                        body: arm_body,
                    }
                })
                .collect();
            // A sum with zero reachable closures cannot be applied (no producer ⇒ no consumer reached
            // it); emit an empty `match` so the elaborator's exhaustiveness/usefulness pass governs it
            // (it is unreachable — never a fabricated arm, G2).
            let body = Expr::Match {
                scrutinee: Box::new(Expr::Path(Path(vec!["%clo".to_owned()]))),
                arms,
            };
            let sig = FnSig {
                name: apply_name.clone(),
                params: vec![],
                value_params: vec![
                    Param {
                        name: "%clo".to_owned(),
                        ty: ty_to_ref(&Ty::Data(arrow.clone(), vec![])),
                    },
                    Param {
                        name: APPLY_PARAM.to_owned(),
                        ty: closure_param_ref(&sum.arrow_a),
                    },
                ],
                ret: closure_param_ref(&sum.arrow_b),
                effects: vec![],
                effect_budgets: std::collections::BTreeMap::new(),
            };
            self.out_fns.insert(
                apply_name.clone(),
                FnDecl {
                    vis: crate::ast::Vis::Private,
                    thaw: false,
                    tier: None,
                    sig,
                    body,
                },
            );
        }
        Ok(())
    }

    /// Resolve a declared [`TypeRef`] (with the decl's type-params as vars) to its **concrete** [`Ty`]
    /// under `subst`, refusing if a `Ty::Var` survives (an undetermined parameter — never guessed).
    fn concrete_ty(
        &self,
        site: &str,
        tyvars: &[String],
        subst: &BTreeMap<String, Ty>,
        t: &TypeRef,
    ) -> Result<Ty, ElabError> {
        let abstract_ty =
            resolve_ty(site, &self.src.types, tyvars, t).map_err(|e| ElabError::Residual {
                site: site.to_owned(),
                what: format!("could not resolve a type during monomorphization: {e}"),
            })?;
        let c = subst_ty(&abstract_ty.0, subst);
        if has_var(&c) {
            return residual(
                site,
                format!(
                    "type `{c}` stays abstract after substitution — an undetermined type parameter \
                     is never guessed (RFC-0007 §11.3 / S1)"
                ),
            );
        }
        // The concrete type may itself name a generic data instance to enqueue (e.g. `List<Binary{8}>`
        // as a parameter type).
        Ok(c)
    }

    /// Enqueue every generic **data** instance mentioned in a concrete `Ty` (recursing into
    /// arguments), so a type used only inside another type/field is still emitted.
    fn enqueue_tys_in(&mut self, ty: &Ty) {
        if let Ty::Data(n, args) = ty {
            for a in args {
                self.enqueue_tys_in(a);
            }
            // A monomorphic (nullary) data type still needs registering if it is reachable; enqueue it
            // either way (empty targs mangle to the original name, so it is byte-identical).
            if self.src.types.contains_key(n) {
                self.enqueue(Item::Data {
                    name: n.clone(),
                    targs: args.clone(),
                });
            }
        }
    }

    // ----- the body rewriter -------------------------------------------------------------------

    /// Rewrite (and walk) an expression under a **concrete** value scope, threading the bidirectional
    /// `expected` type. Mirrors every [`Expr`] arm: rewrites `App`/`Path`/`Pattern` names to their
    /// mangled monomorphic forms, discovers transitive instances, and refuses anything outside the
    /// monomorphizable fragment with an explicit [`ElabError::Residual`] (never silent — G2).
    ///
    /// `expected` matters where the checker's bidirectional pass used it: a bare nullary generic ctor
    /// (`Nil`) and a return-driven trait-method receiver both take their type from context.
    fn rewrite(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        e: &Expr,
        expected: Option<&Ty>,
    ) -> Result<Expr, ElabError> {
        match e {
            Expr::Lit(l) => Ok(Expr::Lit(l.clone())),
            Expr::Path(p) => self.rewrite_path(site, scope, p, expected),
            Expr::App { head, args } => self.rewrite_app(site, scope, head, args, expected),
            Expr::Let {
                name,
                ty,
                bound,
                body,
            } => {
                // The bound's expected is its ascription (if any), resolved concretely; the body's is
                // the enclosing `expected`. The binder's concrete type comes from re-inference.
                let want = match ty {
                    Some(t) => Some(self.concrete_ty(site, &[], &BTreeMap::new(), t)?),
                    None => None,
                };
                let bound2 = self.rewrite(site, scope, bound, want.as_ref())?;
                let bty = self.infer(site, scope, bound)?;
                scope.push((name.clone(), bty));
                let body2 = self.rewrite(site, scope, body, expected);
                scope.pop();
                let body2 = body2?;
                Ok(Expr::Let {
                    name: name.clone(),
                    // The ascription, if present, is now concrete (mono erases type params); keep it
                    // for fidelity (the elaborator ignores the type part — it re-infers). M-967: its
                    // own `@ g` (from the source `ty`) threads onto the rewritten ascription too —
                    // never silently dropped.
                    ty: want
                        .as_ref()
                        .map(|w| ty_to_ref_tagged(w, ty.as_ref().and_then(|t| t.guarantee))),
                    bound: Box::new(bound2),
                    body: Box::new(body2),
                })
            }
            Expr::If { cond, conseq, alt } => {
                let bool_ty = Ty::Data("Bool".to_owned(), vec![]);
                let cond2 = self.rewrite(site, scope, cond, Some(&bool_ty))?;
                let conseq2 = self.rewrite(site, scope, conseq, expected)?;
                // The else-branch may borrow the then-branch's type as its expected (bare-decimal
                // width sharing), mirroring `check_if`.
                let then_ty = self.infer(site, scope, conseq)?;
                let alt2 = self.rewrite(site, scope, alt, expected.or(Some(&then_ty)))?;
                Ok(Expr::If {
                    cond: Box::new(cond2),
                    conseq: Box::new(conseq2),
                    alt: Box::new(alt2),
                })
            }
            Expr::Match { scrutinee, arms } => {
                self.rewrite_match(site, scope, scrutinee, arms, expected)
            }
            Expr::For {
                x,
                xs,
                acc,
                init,
                body,
            } => self.rewrite_for(site, scope, x, xs, acc, init, body),
            Expr::Swap {
                value,
                target,
                policy,
            } => {
                // `swap` is never silent; mono does not touch its certificate. The target is a concrete
                // repr (no type params), kept verbatim; only the value is rewritten.
                let value2 = self.rewrite(site, scope, value, None)?;
                Ok(Expr::Swap {
                    value: Box::new(value2),
                    target: target.clone(),
                    policy: policy.clone(),
                })
            }
            Expr::Ascribe(inner, t) => {
                let want = self.concrete_ty(site, &[], &BTreeMap::new(), t)?;
                let inner2 = self.rewrite(site, scope, inner, Some(&want))?;
                // M-967: the ascription's own `@ g` threads onto the rewritten concrete type.
                Ok(Expr::Ascribe(
                    Box::new(inner2),
                    ty_to_ref_tagged(&want, t.guarantee),
                ))
            }
            Expr::Colony(hyphae) => {
                let mut out = Vec::with_capacity(hyphae.len());
                for h in hyphae {
                    // M-906 (DN-70 D1): rewrite the optional `@forage(policy)` literal through
                    // monomorphization too (mirrors `body`; a literal bitmask carries no type
                    // variables, but the rewrite keeps the pass total over every hypha field).
                    let forage = match &h.forage {
                        Some(p) => Some(Box::new(self.rewrite(site, scope, p, None)?)),
                        None => None,
                    };
                    out.push(Hypha {
                        forage,
                        body: self.rewrite(site, scope, &h.body, None)?,
                    });
                }
                Ok(Expr::Colony(out))
            }
            // DN-58 §A/§B (M-667): `fuse(a, b)` and `reclaim(policy) { body }` — rewrite both
            // operands/policy/body through monomorphization. These constructs are type-concrete
            // (the checker verified homogeneity); any lingering type-variable inside an operand
            // is a monomorphization concern handled transparently here.
            Expr::Fuse { left, right } => {
                // DN-58 §A.5 (M-817): a **Data**-type `fuse(a, b)` desugars to the resolved
                // `Fuse::join` trait-method call — exactly the form the L1 evaluator dispatches
                // (`eval.rs` builds `join(left, right)`), and the form that makes the user merge
                // **run** three-way (the coherent instance method is emitted as a direct fn and
                // inlined by `elab`). A **repr**-type `fuse` has no user `join`; its meet is a
                // built-in (the `Binary` meet is `fuse_join:binary`), so it stays an `Expr::Fuse`
                // for `elab` to lower to the meet prim. The checker (`check_fuse`) has already
                // verified a coherent `Fuse` instance exists for the Data case, so the resolution
                // below cannot be a guess (G2/VR-5).
                let lty = self.infer(site, scope, left)?;
                let is_repr = matches!(
                    &lty,
                    Ty::Binary(_) | Ty::Ternary(_) | Ty::Dense(_, _) | Ty::Bytes | Ty::Seq(_, _)
                );
                if is_repr {
                    let left2 = self.rewrite(site, scope, left, None)?;
                    let right2 = self.rewrite(site, scope, right, None)?;
                    Ok(Expr::Fuse {
                        left: Box::new(left2),
                        right: Box::new(right2),
                    })
                } else {
                    // `fuse(a, b) ≡ join(a, b)` (left ↦ `self`, right ↦ `other` — DN-58 §A.2
                    // canonical `Fuse::join`). Route through the trait-method resolver so the
                    // coherent instance is *selected and recorded* (EXPLAIN — house rule #2), never
                    // guessed. The expected type seeds return-driven receiver inference; the operand
                    // types pin it regardless.
                    let join_args = [left.as_ref().clone(), right.as_ref().clone()];
                    self.rewrite_trait_method_call(site, scope, "join", &join_args, expected)
                }
            }
            Expr::Reclaim { policy, body } => {
                let policy2 = self.rewrite(site, scope, policy, None)?;
                let body2 = self.rewrite(site, scope, body, expected)?;
                Ok(Expr::Reclaim {
                    policy: Box::new(policy2),
                    body: Box::new(body2),
                })
            }
            // M-826: `TupleLit` nodes are rewritten to `App { head: Path(MkTuple$N), args }` by
            // the checker (`check_tuple_lit`), so this arm should never be reached on a well-checked
            // AST. Treat a surviving `TupleLit` as a residual (defense in depth — G2: never silent).
            Expr::TupleLit(_) => residual(
                site,
                "internal: TupleLit survived to monomorphization — the checker should have \
                 rewritten it to a constructor App (M-826; never silent, G2)",
            ),
            // Constructs with no v0 lowering regardless of generics — kept as explicit residuals so the
            // elaborator's own refusal still fires (defense in depth; never a fabricated artifact).
            Expr::Wild(_) => residual(
                site,
                "wild/FFI has no L0 form in v0 — monomorphization does not change that (M-661)",
            ),
            Expr::Spore(_) => residual(site, "`spore` is deferred (E2-5/M-260)"),
            // M-904 (DN-71 Model S §4.3): `consume`'s L0 form is the identity of its operand (the
            // affine move is a checker-level fact, discharged statically at check time — DN-71 §4.2;
            // `crate::grade` already treats `consume` as move-transparent). `Substrate{tag}` carries
            // no type parameters (LR-8), so there is nothing here for mono to specialize — rewrite the
            // operand and reconstruct `Consume`, mirroring the `Ascribe` transparent-wrapper case
            // above. This lifts the former M-664 residual: the same `consume` type rule, now honestly
            // passed through rather than staged (G2/VR-5) — matching `elab.rs`'s own M-904 arm.
            Expr::Consume(operand) => {
                let operand2 = self.rewrite(site, scope, operand, expected)?;
                Ok(Expr::Consume(Box::new(operand2)))
            }
            // RFC-0024 §4A.4 (M-704): a `lambda` lowers to a **closure-constructor application** — its
            // captured environment, snapshotted by value at this definition site (value-semantics).
            // The closure tag-sum + the `apply$<arrow>` dispatcher are emitted once per arrow at
            // `finish()`. No new L0 node (the result is an ordinary `Expr::App` of a data ctor; KC-3).
            Expr::Lambda { params, body } => {
                self.rewrite_lambda(site, scope, params, body, expected)
            }
            Expr::WithParadigm { .. } => residual(
                site,
                "internal: a `with paradigm` block reached monomorphization — the ambient \
                 resolution pass strips it before checking (RFC-0012 §4.4)",
            ),
        }
    }

    /// Rewrite a path/variable. A local binder passes through; a recursive-fn reference or a nullary
    /// constructor is rewritten to its mangled monomorphic name (and its instance enqueued).
    fn rewrite_path(
        &mut self,
        site: &str,
        scope: &[(String, Ty)],
        p: &Path,
        expected: Option<&Ty>,
    ) -> Result<Expr, ElabError> {
        if p.0.len() != 1 {
            return residual(site, format!("dotted path `{}`", p.0.join(".")));
        }
        let name = &p.0[0];
        // A value binder in scope is left as-is.
        if scope.iter().any(|(n, _)| n == name) {
            return Ok(Expr::Path(p.clone()));
        }
        // A nullary data constructor (Nil, Z, True, …). Its type — hence its data instance — comes from
        // `expected` for a generic type (mirroring `check_path`); a monomorphic one needs no context.
        if let Some((d, i)) = self.src.ctor(name) {
            if d.ctors[i].fields.is_empty() {
                let (dname, targs) = self.ctor_data_instance(site, &d.name, expected)?;
                self.enqueue(Item::Data {
                    name: dname.clone(),
                    targs: targs.clone(),
                });
                return Ok(Expr::Path(Path(vec![mangle_ctor(name, &targs)])));
            }
            // A non-nullary ctor referenced bare is unsaturated — the checker already refused it; keep
            // an explicit residual as defense in depth.
            return residual(
                site,
                format!("constructor `{name}` referenced without saturation (W6)"),
            );
        }
        // A bare reference to a (recursive) function. `rewrite_path` is reached only for a path in
        // **value position** (a call head goes through `rewrite_app`; a statically-resolved HOF
        // argument is consumed by `resolve_fn_args` and never rewritten here) — so a fn name here is
        // a fn **value**.
        if let Some(fd) = self.src.fns.get(name).cloned() {
            if !fd.sig.params.is_empty() {
                return residual(
                    site,
                    format!(
                        "generic function `{name}` referenced as a bare value — the surface is \
                         first-order (no function values); apply it (RFC-0007 §11.3)"
                    ),
                );
            }
            // RFC-0024 §4A.4 (M-704): a **named monomorphic fn used as an escaping value** (e.g.
            // `let f = negate in f(x)`) becomes a **nullary closure constructor** of its arrow type
            // (`apply` then calls the named fn). This is the "a bare named fn becomes a nullary
            // constructor" case (§4A.4) — the same lowering as a captureless lambda. A single-value-
            // parameter monomorphic fn has a concrete arrow `A => B`; anything else (nullary / multi-
            // param) cannot be a single-arg fn value and is a never-silent refusal (G2; partial
            // application is tuple-gated — §4A.8).
            if fd.sig.value_params.len() == 1 {
                let a =
                    self.concrete_ty(site, &[], &BTreeMap::new(), &fd.sig.value_params[0].ty)?;
                let b = self.concrete_ty(site, &[], &BTreeMap::new(), &fd.sig.ret)?;
                return self.wrap_named_fn_as_closure(name, &a, &b);
            }
            // Multi-parameter fn used as a first-class value (M-822 / RFC-0024 §4A.5): desugar to
            // a curried lambda wrapper `lambda(p1: A) => lambda(p2: B) => … => name(p1, …, pN)`.
            // This is the mono-side mirror of `check_path`'s multi-param currying. Only
            // monomorphic (no type params) fns are supported here — generic multi-param fns still
            // produce a residual (need full type-arg inference machinery — never silent, G2).
            if fd.sig.value_params.len() > 1 {
                let vparams = fd.sig.value_params.clone();
                let ret_ty_ref = fd.sig.ret.clone();
                // Resolve concrete types for each param and the return.
                let mut param_tys: Vec<Ty> = Vec::with_capacity(vparams.len());
                for p in &vparams {
                    param_tys.push(self.concrete_ty(site, &[], &BTreeMap::new(), &p.ty)?);
                }
                let ret_ty = self.concrete_ty(site, &[], &BTreeMap::new(), &ret_ty_ref)?;
                // Build the innermost call: `name(p1, p2, …, pN)`.
                let call = Expr::App {
                    head: Box::new(Expr::Path(p.clone())),
                    args: vparams
                        .iter()
                        .map(|p| Expr::Path(Path(vec![p.name.clone()])))
                        .collect(),
                };
                // Build curried nested lambdas (inner-first): lambda(pN) => … => lambda(p1) => call.
                let mut body: Expr = call;
                for (vp, _ty) in vparams.iter().zip(param_tys.iter()).rev() {
                    body = Expr::Lambda {
                        params: vec![vp.clone()],
                        body: Box::new(body),
                    };
                }
                // Build the curried arrow type A -> (B -> (… -> Z)) and lower via rewrite_lambda.
                let curried_ty = param_tys
                    .iter()
                    .rev()
                    .fold(ret_ty, |acc, t| Ty::Fn(Box::new(t.clone()), Box::new(acc)));
                // Rewrite the outer lambda (which recurses into the inner) — a mut scope is needed.
                let mut empty_scope: Vec<(String, Ty)> = Vec::new();
                return self.rewrite(site, &mut empty_scope, &body, Some(&curried_ty));
            }
            // Nullary fn referenced as a bare value: not a function value — never-silent (G2).
            return residual(
                site,
                format!(
                    "function `{name}` has 0 value parameters and cannot be used as a function \
                     value — a nullary fn must be applied directly, not used as a value \
                     (RFC-0024 §4A, never a silent coercion — G2)"
                ),
            );
        }
        // Unresolved here means a free name; the checker would have refused it. Keep it verbatim so the
        // elaborator's own "unresolved name" residual fires (never silently dropped).
        Ok(Expr::Path(p.clone()))
    }

    /// Rewrite an application head + arguments. Dispatches exactly as the checker's `check_app`:
    /// user fn (monomorphic or generic), constructor (monomorphic or generic), unqualified
    /// trait-method, or prim — rewriting names to mangled forms and enqueuing instances.
    fn rewrite_app(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        args: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<Expr, ElabError> {
        // M-826 Part 2 — lift the first-order restriction for chained HOF application `f(x)(y)`:
        // when the head is not a Path (e.g. `App{head: apply, args: [succ]}`), infer its type;
        // if it has a function type `A -> B`, rewrite via the dynamic closure dispatcher (the same
        // `apply$<arrow>` path dynamic HOF values use — §4A.5). Never-silent (G2): a non-function
        // head is an explicit Residual.
        if !matches!(head, Expr::Path(_)) {
            let hty = self.infer(site, scope, head)?;
            let Ty::Fn(param_ty, _) = &hty else {
                return residual(
                    site,
                    format!(
                        "application head is not a function — the expression has type `{hty}`, \
                         which is not callable (M-826 §Part2 — never silent, G2)"
                    ),
                );
            };
            if args.len() != 1 {
                return residual(
                    site,
                    format!(
                        "higher-order application requires exactly 1 argument in stage-1; \
                         got {} — partial application / multi-arg HOF is deferred (RFC-0024 §5, G2)",
                        args.len()
                    ),
                );
            }
            // Rewrite the head (e.g. the inner App), then route through the `apply$<arrow>`
            // dispatcher — the same dynamic closure application path as closure values.
            let head2 = self.rewrite(site, scope, head, None)?;
            let arg2 = self.rewrite(site, scope, &args[0], Some(param_ty))?;
            let arrow = mangle_arrow(param_ty, &{
                let Ty::Fn(_, ret) = &hty else { unreachable!() };
                ret.as_ref().clone()
            });
            let dispatcher = format!("apply${arrow}");
            return Ok(Expr::App {
                head: Box::new(Expr::Path(Path(vec![dispatcher]))),
                args: vec![head2, arg2],
            });
        }
        let Expr::Path(p) = head else {
            unreachable!("non-Path head handled above")
        };
        if p.0.len() != 1 {
            return residual(site, format!("dotted call `{}`", p.0.join(".")));
        }
        let name = &p.0[0];

        // (0) HOF parameter application: `f(x)` where `f` is a fn-typed value parameter being
        // defunctionalized. The fn-param substitution map maps `f` to its resolved callee's mangled
        // name — rewrite to a direct call (RFC-0024 §4, M-687). The callee was already enqueued
        // when the HOF specialization was enqueued at the outer call site.
        if let Some(callee_mangled) = self.fn_param_subst.get(name).cloned() {
            // The HOF parameter `f: A -> B` is single-argument (RFC-0024 §3/§5 — multi-arg is a
            // staged Residual). Validate the argument count to stay never-silent (G2).
            if args.len() != 1 {
                return residual(
                    site,
                    format!(
                        "HOF parameter `{name}` applied to {} argument(s); only 1 is supported in \
                         stage-1 (RFC-0024 §5 — partial application / multi-arg HOF is deferred)",
                        args.len()
                    ),
                );
            }
            // Re-infer the arg type from scope to thread the right `expected` (mirrors the
            // checker's `Ty::Fn` arm in `check_app`). The callee must already be in `out_fns`
            // (it was enqueued by `rewrite_app` at the outer HOF call site and emitted before the
            // HOF body is walked — if not, an `emit_fn` is triggered now via the worklist; since
            // the worklist drains recursively the callee is present). For re-inference we can use
            // `None` as `expected` (the arg type is concrete from scope).
            let arg2 = self.rewrite(site, scope, &args[0], None)?;
            return Ok(Expr::App {
                head: Box::new(Expr::Path(Path(vec![callee_mangled]))),
                args: vec![arg2],
            });
        }

        // (0b) RFC-0024 §4A (M-704): **dynamic closure application** — `f(x)` where `f` is a closure
        // VALUE (either a kept dynamic HOF parameter, or any in-scope binder of arrow type `A => B`:
        // a `let`-bound lambda, a fn value out of a `match`/field/return). Rewrite to a call to the
        // generated dispatcher `apply$<arrow>(f, x)`. The dispatcher's `match` over the closed
        // whole-program constructor set IS the dynamic dispatch (§4A.5). Single-argument only in
        // stage-1 (multi-arg/partial is tuple-gated, §4A.8) — never silent (G2).
        let dyn_arrow: Option<String> = self.dyn_fn_param.get(name).cloned().or_else(|| {
            scope
                .iter()
                .rev()
                .find(|(n, _)| n == name)
                .and_then(|(_, t)| {
                    if let Ty::Fn(a, b) = t {
                        Some(mangle_arrow(a, b))
                    } else {
                        None
                    }
                })
        });
        if let Some(arrow) = dyn_arrow {
            if args.len() != 1 {
                return residual(
                    site,
                    format!(
                        "closure value `{name}` applied to {} argument(s); only 1 is supported in \
                         stage-1 (RFC-0024 §4A.8 — multi-arg / partial application is tuple-gated, \
                         never a silent coercion)",
                        args.len()
                    ),
                );
            }
            // Register the arrow so the sum + `apply` dispatcher are scheduled (idempotent). For a
            // scope-binder closure the arrow types come from the binder's `Ty::Fn`; for a kept
            // dynamic param the arrow was already registered at the outer call site.
            if let Some((_, Ty::Fn(a, b))) = scope.iter().rev().find(|(n, _)| n == name) {
                let (a, b) = (a.as_ref().clone(), b.as_ref().clone());
                let _ = self.register_arrow(&a, &b);
            }
            let apply_name = apply_fn_name(&arrow);
            let arg2 = self.rewrite(site, scope, &args[0], None)?;
            return Ok(Expr::App {
                head: Box::new(Expr::Path(Path(vec![apply_name]))),
                args: vec![Expr::Path(Path(vec![name.clone()])), arg2],
            });
        }

        // (1) User function call (the head name is in scope as a fn). Clone the decl so the immutable
        // borrow of `self.src` does not outlive the `&mut self` calls below.
        if let Some(fd) = self.src.fns.get(name).cloned() {
            // DN-42 / M-753 step-c: call infer_fn_targs if the function has either
            // type params OR width params. Both return types are bundled together.
            let (targs, wargs) =
                if fd.sig.params.is_empty() && fd.sig.width_param_names().is_empty() {
                    (vec![], vec![])
                } else {
                    self.infer_fn_targs(site, scope, name, &fd, args)?
                };
            // Detect fn-typed value parameters and classify each actual argument: §4 static
            // (M-687 — baked + dropped) or §4A dynamic (M-704 — kept as a closure value).
            let (fn_args, dyn_fns) = self.resolve_fn_args(site, scope, name, &fd, &targs, args)?;
            let want_tys = self.fn_value_param_tys(site, &fd, &targs)?;
            // Static fn-args are dropped from the call (baked into the key); dynamic fn-args are
            // KEPT (the closure value is passed). Build the index sets for each.
            let static_indices: BTreeSet<usize> = fn_args.iter().map(|(i, _)| *i).collect();
            let dyn_indices: BTreeSet<usize> = dyn_fns.iter().map(|(i, _)| *i).collect();
            let mut args2 = Vec::with_capacity(args.len());
            for (idx, (a, exp)) in args.iter().zip(want_tys.iter()).enumerate() {
                if static_indices.contains(&idx) {
                    // §4 static: defunctionalized away (baked into the key, not passed). Skip.
                    continue;
                }
                if dyn_indices.contains(&idx) {
                    // §4A dynamic: the argument is a closure value of `exp` (a `Ty::Fn`). Rewriting
                    // it lowers a lambda to a closure-constructor application, or threads a closure
                    // binder / dynamic fn value through unchanged. Pass it at the call site.
                    args2.push(self.rewrite(site, scope, a, Some(exp))?);
                    continue;
                }
                args2.push(self.rewrite(site, scope, a, Some(exp))?);
            }
            let mangled = mangle_hof_decl(name, &targs, &wargs, &fn_args, &dyn_fns);
            self.enqueue(Item::Fn {
                name: name.clone(),
                targs: targs.clone(),
                wargs: wargs.clone(),
                fn_args,
                dyn_fns,
            });
            return Ok(Expr::App {
                head: Box::new(Expr::Path(Path(vec![mangled]))),
                args: args2,
            });
        }

        // (2) Saturated constructor application.
        if let Some((d, _)) = self.src.ctor(name) {
            let dname = d.name.clone();
            // The concrete data instance of this constructor application — `infer_type` types the whole
            // app to `Ty::Data(dname, targs)` (it solves the data targs from the field args + expected).
            // `app_ctor_data_instance` resolves only via the `n == dname` arm, so its data name is
            // always `dname`; keep just the solved type args (the owner name is already known).
            let (_di, targs) =
                self.app_ctor_data_instance(site, scope, head, args, &dname, expected)?;
            // Rewrite each field argument under its concrete field-type expected.
            let field_tys = self.ctor_field_tys(site, &dname, name, &targs)?;
            let args2 = self.rewrite_call_args(site, scope, field_tys, args)?;
            self.enqueue(Item::Data {
                name: dname,
                targs: targs.clone(),
            });
            return Ok(Expr::App {
                head: Box::new(Expr::Path(Path(vec![mangle_ctor(name, &targs)]))),
                args: args2,
            });
        }

        // (3) Unqualified trait-method call (resolved to a direct call to the instance method).
        if self.is_trait_method(name) {
            return self.rewrite_trait_method_call(site, scope, name, args, expected);
        }

        // (4) A prim (or an unknown name the elaborator will refuse). Rewrite arguments and keep the
        //     head verbatim — prims have no type parameters. A bare-decimal arg is already resolved by
        //     the checker, so each arg infers concretely.
        let mut args2 = Vec::with_capacity(args.len());
        for a in args {
            args2.push(self.rewrite(site, scope, a, None)?);
        }
        Ok(Expr::App {
            head: Box::new(head.clone()),
            args: args2,
        })
    }

    /// Solve a generic **function** call's type arguments by unifying the callee's declared parameter
    /// types (abstract over its type-params) against the actual argument types (re-inferred concretely
    /// in the current scope) — exactly the checker's `check_app_generic_fn` inference. An undetermined
    /// parameter is an explicit residual (never guessed — G2/VR-5).
    fn infer_fn_targs(
        &self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        name: &str,
        fd: &FnDecl,
        args: &[Expr],
    ) -> Result<(Vec<Ty>, Vec<Width>), ElabError> {
        if fd.sig.value_params.len() != args.len() {
            return residual(
                site,
                format!(
                    "`{name}` takes {} argument(s), got {}",
                    fd.sig.value_params.len(),
                    args.len()
                ),
            );
        }
        let callee_vars = fd.sig.param_names();
        let mut subst: BTreeMap<String, Ty> = BTreeMap::new();
        for (pm, a) in fd.sig.value_params.iter().zip(args) {
            let want = resolve_ty(site, &self.src.types, &callee_vars, &pm.ty)
                .map_err(|e| res_err(site, e))?
                .0;
            let want_now = subst_ty(&want, &subst);
            let got = self.infer(site, scope, a)?;
            unify_into(site, &want_now, &got, &mut subst)?;
        }
        let mut targs = Vec::with_capacity(callee_vars.len());
        for v in &callee_vars {
            match subst.get(v) {
                Some(t) if !has_var(t) => targs.push(t.clone()),
                _ => {
                    return residual(
                        site,
                        format!(
                            "`{name}` is generic over `{v}`, but this call does not determine it — \
                             never a guessed default (RFC-0007 §11.3 / VR-5)"
                        ),
                    )
                }
            }
        }
        // DN-42 / M-753 step-c: also collect resolved width arguments (carrier convention —
        // width var `N` was bound as `Ty::Binary(Width::Lit(n))` by unify). An unresolved width
        // parameter is an explicit residual — never a guessed default (VR-5/G2).
        let callee_wvars = fd.sig.width_param_names();
        let mut wargs = Vec::with_capacity(callee_wvars.len());
        for v in &callee_wvars {
            match subst.get(v) {
                Some(Ty::Binary(Width::Lit(n))) => wargs.push(Width::Lit(*n)),
                _ => {
                    return residual(
                        site,
                        format!(
                            "`{name}` is width-generic over `{v}`, but this call does not \
                             determine the width — undetermined width is never guessed (DN-42 §4 / VR-5)"
                        ),
                    )
                }
            }
        }
        Ok((targs, wargs))
    }

    /// The concrete data instance `(dname, targs)` of a **nullary** constructor used as a value — from
    /// `expected` for a generic type (mirroring `check_path`). A monomorphic type needs no context.
    fn ctor_data_instance(
        &self,
        site: &str,
        dname: &str,
        expected: Option<&Ty>,
    ) -> Result<(String, Vec<Ty>), ElabError> {
        let d = self
            .src
            .types
            .get(dname)
            .ok_or_else(|| ElabError::Residual {
                site: site.to_owned(),
                what: format!("unknown data type `{dname}`"),
            })?;
        if d.params.is_empty() {
            return Ok((dname.to_owned(), vec![]));
        }
        match expected {
            Some(Ty::Data(en, eargs)) if en == dname && eargs.len() == d.params.len() => {
                for a in eargs {
                    if has_var(a) {
                        return residual(
                            site,
                            format!("nullary constructor of `{dname}<…>` resolved to abstract {a}"),
                        );
                    }
                }
                Ok((dname.to_owned(), eargs.clone()))
            }
            _ => residual(
                site,
                format!(
                    "constructor of generic `{dname}<…>` needs its type argument(s) from context — \
                     never a guess (RFC-0007 §11.3)"
                ),
            ),
        }
    }

    /// The concrete data instance of a **saturated** constructor application — `infer_type` types the
    /// whole application to `Ty::Data(dname, targs)`, solving the data type arguments from the field
    /// arguments (and `expected`). The returned name is the source data name; `targs` are concrete.
    fn app_ctor_data_instance(
        &self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        head: &Expr,
        args: &[Expr],
        dname: &str,
        expected: Option<&Ty>,
    ) -> Result<(String, Vec<Ty>), ElabError> {
        let app = Expr::App {
            head: Box::new(head.clone()),
            args: args.to_vec(),
        };
        // Re-infer against `expected` so a bare nullary generic sub-ctor (`Nil`) in a field is pinned.
        let ty = self.infer_against(site, scope, &app, expected)?;
        match ty {
            Ty::Data(n, targs) if n == dname => {
                for a in &targs {
                    if has_var(a) {
                        return residual(
                            site,
                            format!("constructor `{dname}` left type argument {a} undetermined"),
                        );
                    }
                }
                Ok((n, targs))
            }
            other => residual(
                site,
                format!("constructor application did not type to `{dname}<…>` (got {other})"),
            ),
        }
    }

    /// The (substituted, concrete) value-parameter types of fn `fd` at `targs` — the per-argument
    /// `expected` types for rewriting a generic/monomorphic function call's arguments.
    fn fn_value_param_tys(
        &self,
        site: &str,
        fd: &FnDecl,
        targs: &[Ty],
    ) -> Result<Vec<Ty>, ElabError> {
        let tyvars = fd.sig.param_names();
        let subst = param_subst(&tyvars, targs);
        let mut out = Vec::with_capacity(fd.sig.value_params.len());
        for p in &fd.sig.value_params {
            let (abstract_ty, _) =
                resolve_ty(site, &self.src.types, &tyvars, &p.ty).map_err(|e| res_err(site, e))?;
            out.push(subst_ty(&abstract_ty, &subst));
        }
        Ok(out)
    }

    /// The (substituted, concrete) field types of constructor `cname` of data `dname` at `targs` —
    /// the per-argument `expected` types for rewriting the field arguments.
    fn ctor_field_tys(
        &self,
        site: &str,
        dname: &str,
        cname: &str,
        targs: &[Ty],
    ) -> Result<Vec<Ty>, ElabError> {
        let d = self
            .src
            .types
            .get(dname)
            .ok_or_else(|| ElabError::Residual {
                site: site.to_owned(),
                what: format!("unknown data type `{dname}`"),
            })?;
        let c = d
            .ctors
            .iter()
            .find(|c| c.name == cname)
            .ok_or_else(|| ElabError::Residual {
                site: site.to_owned(),
                what: format!("`{dname}` has no constructor `{cname}`"),
            })?;
        let subst = param_subst(&d.params, targs);
        Ok(c.fields.iter().map(|f| subst_ty(f, &subst)).collect())
    }

    /// Rewrite each call argument under its concrete `expected` field/parameter type (so a bare
    /// nullary generic ctor argument is pinned). `want_tys` is parallel to `args`.
    fn rewrite_call_args(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        want_tys: Vec<Ty>,
        args: &[Expr],
    ) -> Result<Vec<Expr>, ElabError> {
        let mut out = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let exp = want_tys.get(i);
            out.push(self.rewrite(site, scope, a, exp)?);
        }
        Ok(out)
    }

    /// Resolve and rewrite an **unqualified trait-method call** to a direct call to the coherent
    /// instance's (mangled) method (RFC-0019 §4.4). Mirrors `check_trait_method_call`: find the single
    /// owning trait, solve its parameter by unifying the method signature against the arguments
    /// (seeded from `expected`), look up the instance, enqueue + emit the method, and record the
    /// EXPLAIN selection. Refuses (never guesses) on ambiguity, a multi-parameter trait, an
    /// undetermined receiver, or a missing instance.
    fn rewrite_trait_method_call(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        name: &str,
        args: &[Expr],
        expected: Option<&Ty>,
    ) -> Result<Expr, ElabError> {
        let owners: Vec<&TraitInfo> = self
            .src
            .traits
            .values()
            .filter(|tr| tr.sigs.iter().any(|s| s.name == name))
            .collect();
        let tr = match owners.as_slice() {
            [one] => *one,
            [] => {
                return residual(site, format!("`{name}` is not a trait method (internal)"));
            }
            many => {
                let names: Vec<&str> = many.iter().map(|t| t.name.as_str()).collect();
                return residual(
                    site,
                    format!(
                        "ambiguous trait-method call `{name}` — declared by multiple traits ({}) — \
                         an explicit refusal, never a guess (RFC-0019 §4.4)",
                        names.join(", ")
                    ),
                );
            }
        };
        if tr.params.len() != 1 {
            return residual(
                site,
                format!(
                    "trait-method resolution for `{name}` needs a single-parameter trait \
                     (multi-parameter traits are v2 — RFC-0019 §10)"
                ),
            );
        }
        let sig = tr
            .sigs
            .iter()
            .find(|s| s.name == name)
            .expect("owner has the method");
        if sig.value_params.len() != args.len() {
            return residual(
                site,
                format!(
                    "trait method `{}::{name}` takes {} argument(s), got {}",
                    tr.name,
                    sig.value_params.len(),
                    args.len()
                ),
            );
        }
        let tparam = &tr.params[0];
        let trait_vars = std::slice::from_ref(tparam);
        let mut subst: BTreeMap<String, Ty> = BTreeMap::new();
        // Seed from `expected` against the (abstract) return type — return-driven receiver inference
        // (mirrors `check_trait_method_call`).
        if let Some(exp) = expected {
            if let Ok((ret_abs, _)) = resolve_ty(site, &self.src.types, trait_vars, &sig.ret) {
                let _ = unify_into(site, &ret_abs, exp, &mut subst);
            }
        }
        for (pm, a) in sig.value_params.iter().zip(args) {
            let want = resolve_ty(site, &self.src.types, trait_vars, &pm.ty)
                .map_err(|e| res_err(site, e))?
                .0;
            let want_now = subst_ty(&want, &subst);
            let got = self.infer(site, scope, a)?;
            unify_into(site, &want_now, &got, &mut subst)?;
        }
        let Some(receiver) = subst.get(tparam).cloned() else {
            return residual(
                site,
                format!(
                    "trait-method call `{name}` does not determine trait `{}`'s parameter `{tparam}` \
                     — never a guess (RFC-0019 §4.4)",
                    tr.name
                ),
            );
        };
        if has_var(&receiver) {
            return residual(
                site,
                format!(
                    "trait-method call `{name}` left receiver `{receiver}` abstract — an \
                     undetermined trait parameter is never guessed (RFC-0019 §4.4 / VR-5)"
                ),
            );
        }
        // Rewrite the arguments under the instance method's concrete parameter types.
        let want_tys: Vec<Ty> = sig
            .value_params
            .iter()
            .map(|pm| {
                resolve_ty(site, &self.src.types, trait_vars, &pm.ty)
                    .map(|(t, _)| subst_ty(&t, &subst))
                    .map_err(|e| res_err(site, e))
            })
            .collect::<Result<_, _>>()?;
        let args2 = self.rewrite_call_args(site, scope, want_tys, args)?;
        let mangled = mangle_method(name, &tr.name, &receiver);
        self.enqueue(Item::Method {
            trait_name: tr.name.clone(),
            method: name.to_owned(),
            for_ty: receiver.clone(),
        });
        Ok(Expr::App {
            head: Box::new(Expr::Path(Path(vec![mangled]))),
            args: args2,
        })
    }

    /// RFC-0024 §4A.4 (M-704): lower a **named monomorphic fn used as an escaping value** to a
    /// **nullary** closure constructor of its arrow `a => b` — the dispatcher arm calls the named fn
    /// directly (`callee_mangled(%fnarg)`). This is the "a bare named fn becomes a nullary
    /// constructor" case (§4A.4). The named fn is enqueued so it is emitted; the ctor is deduplicated
    /// by content key (two `let f = double` sites share one constructor — identity fragmentation, G2).
    fn wrap_named_fn_as_closure(
        &mut self,
        fn_name: &str,
        a: &Ty,
        b: &Ty,
    ) -> Result<Expr, ElabError> {
        let arrow = self.register_arrow(a, b);
        let callee_mangled = mangle_decl(fn_name, &[]);
        self.enqueue(Item::Fn {
            name: fn_name.to_owned(),
            targs: vec![],
            wargs: vec![],
            fn_args: vec![],
            dyn_fns: vec![],
        });
        // The dispatcher body for this constructor is a direct call to the named fn on the apply
        // argument: `callee_mangled(<param>)`. Use the canonical apply-param name as the lambda
        // parameter so the `Let` the dispatcher wraps (`let <param> = %fnarg in body`) is the
        // identity binding `let %fnarg = %fnarg in callee(%fnarg)` — well-formed and inert.
        let body = Expr::App {
            head: Box::new(Expr::Path(Path(vec![callee_mangled]))),
            args: vec![Expr::Path(Path(vec![APPLY_PARAM.to_owned()]))],
        };
        let content_key = format!("{arrow}|named:{fn_name}");
        let sum = self
            .closures
            .get_mut(&arrow)
            .expect("arrow registered above");
        let ctor_name = if let Some(&idx) = sum.by_key.get(&content_key) {
            sum.ctors[idx].ctor_name.clone()
        } else {
            let n = sum.ctors.len();
            let ctor_name = format!("Clo${arrow}${n}");
            sum.ctors.push(ClosureCtor {
                ctor_name: ctor_name.clone(),
                captures: Vec::new(),
                param_name: APPLY_PARAM.to_owned(),
                body,
            });
            sum.by_key.insert(content_key, n);
            self.closure_specs.insert(
                ctor_name.clone(),
                ClosureSpecialization {
                    arrow: arrow.clone(),
                    ctor_name: ctor_name.clone(),
                    captures: Vec::new(),
                    apply_fn: apply_fn_name(&arrow),
                },
            );
            ctor_name
        };
        // A nullary constructor value is a bare `Path` (no fields to apply).
        Ok(Expr::Path(Path(vec![ctor_name])))
    }

    /// RFC-0024 §4A (M-704): register a closure tag-sum for the arrow `a => b` (idempotently), and
    /// schedule its `apply$<arrow>` dispatcher fn so `finish()` emits both. Returns the arrow mangle.
    /// The sum's `DataInfo` and the dispatcher are built at `finish()` (the whole-program closure set
    /// is complete only after the worklist drains — §4A.5).
    fn register_arrow(&mut self, a: &Ty, b: &Ty) -> String {
        let arrow = mangle_arrow(a, b);
        self.closures
            .entry(arrow.clone())
            .or_insert_with(|| ClosureSum {
                arrow_a: a.clone(),
                arrow_b: b.clone(),
                ctors: Vec::new(),
                by_key: BTreeMap::new(),
            });
        arrow
    }

    /// RFC-0024 §4A.3/§4A.4 (M-704): lower a `lambda(p: A) => body` at its **definition site** to a
    /// closure-constructor application of its captured environment. Computes the capture set (free
    /// variables of `body`, bound in the enclosing `scope`, minus the lambda's parameter and all
    /// top-level names — §4A.3), registers (deduplicated by content key) one constructor of the
    /// arrow's tag-sum whose fields are the captured types, rewrites the body under
    /// `captures ∪ {param}`, and returns `Clo$<arrow>$<n>(cap1, …, capk)` — the value-snapshot
    /// capture binding (§4A.4). The closure value is an ordinary `Expr::App` of a data constructor
    /// (no new L0 node — KC-3).
    fn rewrite_lambda(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        params: &[Param],
        body: &Expr,
        expected: Option<&Ty>,
    ) -> Result<Expr, ElabError> {
        // Zero-parameter lambda: never-silent refusal (G2) — mirrors `check_lambda` (the checker
        // already refused; this keeps mono never-silent as a defensive guard).
        if params.is_empty() {
            return residual(
                site,
                "a `lambda` requires at least 1 parameter — a zero-argument lambda has no type \
                 without a unit/nullary type (never a silent accept — G2)"
                    .to_owned(),
            );
        }
        // Multi-argument currying (M-822 / RFC-0024 §4A.5/§4A.8): desugar `lambda(p1, p2, …) =>
        // body` to `lambda(p1) => lambda(p2) => … => body` before lowering. The checker already
        // transformed these into nested single-param lambdas; this guard ensures mono never
        // silently accepts a multi-param lambda that slips through (G2 — never-silent).
        if params.len() > 1 {
            let (first, rest) = params.split_first().expect("len > 1");
            let inner_body = Expr::Lambda {
                params: rest.to_vec(),
                body: Box::new(body.clone()),
            };
            return self.rewrite_lambda(
                site,
                scope,
                std::slice::from_ref(first),
                &inner_body,
                expected,
            );
        }
        // Exactly one parameter — the base case.
        let [param] = params else {
            unreachable!("len == 1 after the multi-arg and zero-arg branches above")
        };
        // The concrete parameter type (lambda params are always ascribed). Re-infer the body type
        // under `scope ∪ {param}` to pin the codomain — mirrors `check_lambda`. An `expected` arrow
        // pins the codomain when the body's type is context-driven (e.g. a bare nullary ctor).
        let param_ty = self.concrete_ty(site, &[], &BTreeMap::new(), &param.ty)?;
        let expected_ret: Option<Ty> = match expected {
            Some(Ty::Fn(_, er)) => Some(er.as_ref().clone()),
            _ => None,
        };
        scope.push((param.name.clone(), param_ty.clone()));
        let body_ty_res = self.infer_against(site, scope, body, expected_ret.as_ref());
        scope.pop();
        let body_ty = body_ty_res?;
        let arrow = self.register_arrow(&param_ty, &body_ty);

        // §4A.3 capture set: free variables of `body` (single-segment paths not bound inside `body`)
        // that are **locals in the enclosing `scope`** (so not top-level names) and not the param.
        // Order = first-occurrence (a total deterministic order — §4A.3 / G2). Each capture's type
        // comes from the enclosing scope (mono runs post-check, so every binder type is known).
        let mut bound_in_body: BTreeSet<String> = BTreeSet::new();
        bound_in_body.insert(param.name.clone());
        let mut free: Vec<String> = Vec::new();
        let mut seen_free: BTreeSet<String> = BTreeSet::new();
        // A `WalkDepthExceeded` from the free-variable walk (M-866) is the same never-silent
        // operational-resource refusal `finish()` surfaces from `totality::classify_all` below —
        // reuse `ElabError::DepthExceeded` rather than inventing a second depth-budget error shape.
        free_vars(body, &mut bound_in_body, &mut seen_free, &mut free).map_err(|e| {
            ElabError::DepthExceeded {
                site: site.to_owned(),
                limit: e.limit,
            }
        })?;
        // Keep only those that are locals in the enclosing scope (captured); a name resolving to a
        // top-level fn/ctor/prim is NOT captured (§4A.3 — handled by `rewrite_path`/§4). A HOF
        // value-parameter that was statically specialized (`fn_param_subst`, e.g. `f→negate` — M-687/
        // M-715) is also NOT captured: it is a **compile-time-baked constant** dropped from the emitted
        // signature, with no runtime value (it lingers in `scope` only for inference). `rewrite_path`
        // resolves such a name to the baked callee inside the body, so capturing it would build a
        // closure ctor referencing a param with no value (a Stuck `unresolved name`, or a silent
        // wrong-entity if a ctor/fn shared the name — G2). Exclude it here.
        let mut captures: Vec<(String, Ty)> = Vec::new();
        for v in &free {
            if self.fn_param_subst.contains_key(v) {
                continue;
            }
            if let Some((_, ty)) = scope.iter().rev().find(|(n, _)| n == v) {
                captures.push((v.clone(), ty.clone()));
            }
        }

        // Rewrite the body under a fresh scope of exactly the captures + the parameter (so captured
        // locals stay `Path`, the param stays `Path`, and top-level names resolve via `rewrite_path`).
        let mut body_scope: Vec<(String, Ty)> = captures.clone();
        body_scope.push((param.name.clone(), param_ty.clone()));
        let rewritten_body = self.rewrite(
            site,
            &mut body_scope,
            body,
            expected_ret.as_ref().or(Some(&body_ty)),
        )?;

        // Content key for dedup: arrow + capture names/types + the param name + the rewritten body.
        // Two structurally identical closures of one arrow share a constructor (§4 identity
        // fragmentation; G2). The body's `Debug` is a stable structural fingerprint.
        let cap_sig: String = captures
            .iter()
            .map(|(n, t)| format!("{n}:{}", mangle_ty_or_fn(t)))
            .collect::<Vec<_>>()
            .join(",");
        let content_key = format!("{}|{}|{}|{:?}", arrow, cap_sig, param.name, rewritten_body);

        // Register the constructor (idempotent by content key).
        let sum = self
            .closures
            .get_mut(&arrow)
            .expect("arrow registered above");
        let ctor_name = if let Some(&idx) = sum.by_key.get(&content_key) {
            sum.ctors[idx].ctor_name.clone()
        } else {
            let n = sum.ctors.len();
            let ctor_name = format!("Clo${arrow}${n}");
            sum.ctors.push(ClosureCtor {
                ctor_name: ctor_name.clone(),
                captures: captures.clone(),
                param_name: param.name.clone(),
                body: rewritten_body,
            });
            sum.by_key.insert(content_key, n);
            // EXPLAIN (house rule #2): record the closure lowering.
            self.closure_specs.insert(
                ctor_name.clone(),
                ClosureSpecialization {
                    arrow: arrow.clone(),
                    ctor_name: ctor_name.clone(),
                    captures: captures
                        .iter()
                        .map(|(n, t)| (n.clone(), format!("{t}")))
                        .collect(),
                    apply_fn: apply_fn_name(&arrow),
                },
            );
            ctor_name
        };

        // Enqueue any generic data instance a capture type names (so a type captured but not otherwise
        // reachable is still emitted; dedup makes it idempotent). Skip `Ty::Fn` (closure-typed captures
        // are themselves lowered to their own `Fn$<arrow>` data type, registered when that arrow is).
        for (_, ty) in &captures {
            if !matches!(ty, Ty::Fn(_, _)) {
                self.enqueue_tys_in(ty);
            }
        }

        // The capture binding: `Clo$arrow$n(cap1, …, capk)` — the captured *current* values by name
        // (their bindings in the enclosing scope). A captureless lambda is a nullary constructor.
        let cap_args: Vec<Expr> = captures
            .iter()
            .map(|(n, _)| Expr::Path(Path(vec![n.clone()])))
            .collect();
        if cap_args.is_empty() {
            Ok(Expr::Path(Path(vec![ctor_name])))
        } else {
            Ok(Expr::App {
                head: Box::new(Expr::Path(Path(vec![ctor_name]))),
                args: cap_args,
            })
        }
    }

    /// Resolve the **function-argument identities** for a call to `name` at `targs`, for any
    /// value-parameter whose (substituted) type is `Ty::Fn` (RFC-0024 §4/§4A). For each such
    /// parameter, classify the actual argument into the §4 **static** path or the §4A **dynamic**
    /// (closure) path (the hybrid — §4A.2, "try §4 first"):
    /// - an `Expr::Path` to a statically-known **monomorphic** top-level function → **static**
    ///   (M-687): bake its identity, drop the param, direct-call inside the body.
    /// - a HOF value-parameter already bound to a static specialization (`fn_param_subst`) → **static**
    ///   (M-715 recursive re-pass).
    /// - a `lambda`, a dynamically-flowing fn value (match/field/return), or a closure binder in scope
    ///   → **dynamic** (M-704, RFC-0024 §4A): the param is **kept** as a `Fn$<arrow>` closure value;
    ///   the argument lowers to a closure value; an application of the param inside the body becomes
    ///   `apply$<arrow>(f, x)`.
    /// - a still-generic top-level function as a value → deferred (FLAG — RFC-0024 §5).
    ///
    /// Returns `(static_fn_args, dyn_fns)`: `static_fn_args` are `(idx, callee_mangled)` (dropped,
    /// baked); `dyn_fns` are `(idx, arrow_mangle)` (kept, lowered to a closure). Both sorted by index
    /// (deterministic). Enqueues each resolved static callee so it is emitted even if otherwise
    /// unreachable. The dynamic arrow is registered (and its `apply`/sum scheduled) lazily.
    #[allow(clippy::type_complexity)]
    fn resolve_fn_args(
        &mut self,
        site: &str,
        scope: &[(String, Ty)], // read below to classify a fn-typed arg as a scope-bound closure
        callee_name: &str,
        fd: &FnDecl,
        targs: &[Ty],
        args: &[Expr],
    ) -> Result<(Vec<(usize, String)>, Vec<(usize, String)>), ElabError> {
        let tyvars = fd.sig.param_names();
        let subst = param_subst(&tyvars, targs);
        let mut fn_args: Vec<(usize, String)> = Vec::new();
        let mut dyn_fns: Vec<(usize, String)> = Vec::new();
        for (idx, (pm, actual)) in fd.sig.value_params.iter().zip(args).enumerate() {
            let (abstract_ty, _) =
                resolve_ty(site, &self.src.types, &tyvars, &pm.ty).map_err(|e| res_err(site, e))?;
            let cty = subst_ty(&abstract_ty, &subst);
            let Ty::Fn(arr_a, arr_b) = &cty else {
                continue; // not a fn-typed parameter — nothing to defunctionalize
            };
            // §4 static path: a bare top-level monomorphic function name, or a re-passed static HOF
            // parameter. Anything else routes to the §4A dynamic (closure) path.
            if let Expr::Path(p) = actual {
                if p.0.len() == 1 {
                    let fn_name = &p.0[0];
                    // M-715 recursive re-pass: a static HOF value-parameter already pinned.
                    if let Some(mangled) = self.fn_param_subst.get(fn_name) {
                        fn_args.push((idx, mangled.clone()));
                        continue;
                    }
                    // M-704: a *dynamic* (closure) HOF value-parameter re-passed — thread it through
                    // as the dynamic closure value of the same arrow (do NOT statically resolve it).
                    if let Some(arrow) = self.dyn_fn_param.get(fn_name).cloned() {
                        dyn_fns.push((idx, arrow));
                        continue;
                    }
                    // A closure binder in scope (a `let f = lambda… in map(xs, f)`) — dynamic.
                    let is_scope_closure = scope
                        .iter()
                        .rev()
                        .find(|(n, _)| n == fn_name)
                        .is_some_and(|(_, t)| matches!(t, Ty::Fn(_, _)));
                    if !is_scope_closure {
                        if let Some(callee_fd) = self.src.fns.get(fn_name) {
                            // A still-generic top-level function as a value: deferred (FLAG).
                            if !callee_fd.sig.params.is_empty() {
                                return residual(
                                    site,
                                    format!(
                                        "function-valued argument `{fn_name}` for parameter `{}` of \
                                         `{callee_name}` is still generic (has type parameters) — a \
                                         generic fn as a value requires type-argument context to \
                                         defunctionalize; this case is deferred (RFC-0024 §5, FLAG: \
                                         generic-fn-as-arg — never a silent guess)",
                                        pm.name
                                    ),
                                );
                            }
                            // §4 static: a monomorphic top-level function — bake its identity, drop it.
                            let callee_mangled = mangle_decl(fn_name, &[]);
                            self.enqueue(Item::Fn {
                                name: fn_name.clone(),
                                targs: vec![],
                                wargs: vec![],
                                fn_args: vec![],
                                dyn_fns: vec![],
                            });
                            fn_args.push((idx, callee_mangled));
                            continue;
                        }
                        // An unbound fn-valued name — never silent (G2).
                        return residual(
                            site,
                            format!(
                                "function-valued argument `{fn_name}` for parameter `{}` of \
                                 `{callee_name}` is not a top-level function nor a closure binder in \
                                 scope (RFC-0024 §3/§4A — never a silent coercion)",
                                pm.name
                            ),
                        );
                    }
                    // else: a closure binder — fall through to the dynamic path below.
                }
            }
            // §4A dynamic path: a lambda, a dynamically-flowing fn value, or a closure binder — the
            // parameter is kept as a closure value of this arrow. Register the arrow so its sum +
            // `apply` are emitted at `finish()`.
            let arrow = self.register_arrow(arr_a, arr_b);
            dyn_fns.push((idx, arrow));
        }
        fn_args.sort_by_key(|(i, _)| *i);
        dyn_fns.sort_by_key(|(i, _)| *i);
        Ok((fn_args, dyn_fns))
    }

    /// Rewrite a `match` — re-infer the (concrete) scrutinee type, rewrite the scrutinee, then each
    /// arm with its pattern's constructor names mangled and its binders bound at their concrete types.
    fn rewrite_match(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        scrutinee: &Expr,
        arms: &[Arm],
        expected: Option<&Ty>,
    ) -> Result<Expr, ElabError> {
        let sty = self.infer(site, scope, scrutinee)?;
        let scrut2 = self.rewrite(site, scope, scrutinee, None)?;
        let mut out_arms = Vec::with_capacity(arms.len());
        for arm in arms {
            // Bind the pattern's variables at their concrete types (from the scrutinee type), rewrite
            // ctor names, then rewrite the arm body under the extended scope.
            let mut arm_scope = scope.clone();
            let pat2 = self.rewrite_pattern(site, &arm.pattern, &sty, &mut arm_scope)?;
            let body2 = self.rewrite(site, &mut arm_scope, &arm.body, expected)?;
            out_arms.push(Arm {
                pattern: pat2,
                body: body2,
            });
        }
        Ok(Expr::Match {
            scrutinee: Box::new(scrut2),
            arms: out_arms,
        })
    }

    /// Rewrite a pattern against the (concrete) scrutinee type `sty`: mangle each constructor name to
    /// its monomorphic form, recurse into sub-patterns at the constructor's substituted field types,
    /// and push every binder onto `scope` at its concrete type. Enqueues the data instance the pattern
    /// matches (so a pattern-only-used type is still emitted).
    fn rewrite_pattern(
        &mut self,
        site: &str,
        pat: &Pattern,
        sty: &Ty,
        scope: &mut Vec<(String, Ty)>,
    ) -> Result<Pattern, ElabError> {
        match pat {
            Pattern::Wildcard => Ok(Pattern::Wildcard),
            Pattern::Lit(l) => Ok(Pattern::Lit(l.clone())),
            Pattern::Ident(b) => {
                // A bare identifier is a binder (a nullary ctor would have been normalized to
                // `Ctor(b, [])` by the checker's `normalize_pattern` before elaboration; but `match`
                // bodies in the *source* `Env` are the resolved bodies, so a nullary ctor may appear as
                // `Ctor` already — here treat a bare ident as a binder of the scrutinee type).
                scope.push((b.clone(), sty.clone()));
                Ok(Pattern::Ident(b.clone()))
            }
            Pattern::Ctor(cname, subs) => {
                let (dname, targs) = match sty {
                    Ty::Data(n, a) => (n.clone(), a.clone()),
                    other => {
                        return residual(
                            site,
                            format!(
                                "a constructor pattern `{cname}` against non-data type {other}"
                            ),
                        )
                    }
                };
                self.enqueue(Item::Data {
                    name: dname.clone(),
                    targs: targs.clone(),
                });
                let field_tys = self.ctor_field_tys(site, &dname, cname, &targs)?;
                if field_tys.len() != subs.len() {
                    return residual(
                        site,
                        format!(
                            "constructor pattern `{cname}` binds {} field(s), the type has {}",
                            subs.len(),
                            field_tys.len()
                        ),
                    );
                }
                let mut subs2 = Vec::with_capacity(subs.len());
                for (sub, fty) in subs.iter().zip(&field_tys) {
                    subs2.push(self.rewrite_pattern(site, sub, fty, scope)?);
                }
                Ok(Pattern::Ctor(mangle_ctor(cname, &targs), subs2))
            }
            // M-826: a tuple pattern `(x, y, …)` is rewritten by the checker to
            // `Pattern::Ctor(MkTuple$N, subs)` before mono runs. A surviving `Pattern::Tuple`
            // here is rewritten to the equivalent Ctor pattern and delegated — never-silent (G2).
            Pattern::Tuple(subs) => {
                let n = subs.len();
                let ctor_name = crate::checkty::tuple_ctor_name(n);
                self.rewrite_pattern(site, &Pattern::Ctor(ctor_name, subs.clone()), sty, scope)
            }
            // `Pattern::Or` is desugared in `check_match` before monomorphization; reaching here
            // means the program was not checked — a never-silent explicit error (G2).
            Pattern::Or(_) => Err(ElabError::Residual {
                site: site.to_owned(),
                what: "internal: Pattern::Or reached monomorphization — or-patterns must be \
                       desugared by the checker (invariant violation — report this)"
                    .to_owned(),
            }),
        }
    }

    /// Rewrite a `for x in xs, acc = init => body` — re-infer the (concrete) spine + accumulator
    /// types, bind `x`/`acc`, and rewrite each part. The element type is the spine's element type.
    #[allow(clippy::too_many_arguments)]
    fn rewrite_for(
        &mut self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        x: &str,
        xs: &Expr,
        acc: &str,
        init: &Expr,
        body: &Expr,
    ) -> Result<Expr, ElabError> {
        let sty = self.infer(site, scope, xs)?;
        let Ty::Data(tname, targs) = &sty else {
            return residual(site, format!("`for` spine is not a data type: {sty}"));
        };
        self.enqueue(Item::Data {
            name: tname.clone(),
            targs: targs.clone(),
        });
        let elem_ty = self.for_elem_ty(site, tname, targs)?;
        let aty = self.infer(site, scope, init)?;
        let xs2 = self.rewrite(site, scope, xs, None)?;
        let init2 = self.rewrite(site, scope, init, None)?;
        let mut body_scope = scope.clone();
        body_scope.push((x.to_owned(), elem_ty));
        body_scope.push((acc.to_owned(), aty));
        let body2 = self.rewrite(site, &mut body_scope, body, None)?;
        Ok(Expr::For {
            x: x.to_owned(),
            xs: Box::new(xs2),
            acc: acc.to_owned(),
            init: Box::new(init2),
            body: Box::new(body2),
        })
    }

    /// The element type of a linear-recursive spine type `tname` at `targs` — the single non-spine
    /// field of its cons constructor, with the type arguments substituted in.
    fn for_elem_ty(&self, site: &str, tname: &str, targs: &[Ty]) -> Result<Ty, ElabError> {
        let d = self
            .src
            .types
            .get(tname)
            .ok_or_else(|| ElabError::Residual {
                site: site.to_owned(),
                what: format!("unknown type `{tname}`"),
            })?;
        let subst = param_subst(&d.params, targs);
        for c in &d.ctors {
            if c.fields.is_empty() {
                continue;
            }
            let elem = c
                .fields
                .iter()
                .find(|f| !matches!(f, Ty::Data(n, _) if n == tname));
            if let Some(e) = elem {
                return Ok(subst_ty(e, &subst));
            }
        }
        residual(site, format!("`for` type `{tname}` has no element field"))
    }

    // ----- re-inference helpers ----------------------------------------------------------------

    /// Is `name` a method of some registered trait (the trait-method dispatch gate)?
    fn is_trait_method(&self, name: &str) -> bool {
        self.src
            .traits
            .values()
            .any(|tr| tr.sigs.iter().any(|s| s.name == name))
    }

    /// Re-infer the concrete type of `e` under the concrete `scope`, using the checker's re-inference
    /// (`infer_type`) over the *source* env. A failure is an explicit residual (never silent).
    fn infer(&self, site: &str, scope: &mut Vec<(String, Ty)>, e: &Expr) -> Result<Ty, ElabError> {
        infer_type(self.src, scope, e).map_err(|err| ElabError::Residual {
            site: site.to_owned(),
            what: format!("could not re-infer a type during monomorphization: {err}"),
        })
    }

    /// Re-infer `e` against an `expected` type (bidirectional) — needed where a bare nullary generic
    /// ctor or a return-driven receiver takes its type from context. Falls back to `infer` when there
    /// is no expected type. Uses the public bidirectional check via a temporary ascription so the
    /// `expected` is threaded without exposing the checker's private `Cx`.
    fn infer_against(
        &self,
        site: &str,
        scope: &mut Vec<(String, Ty)>,
        e: &Expr,
        expected: Option<&Ty>,
    ) -> Result<Ty, ElabError> {
        match expected {
            None => self.infer(site, scope, e),
            Some(exp) => {
                // Thread `expected` by ascribing `e : exp` and inferring that — `check_ascribe` runs the
                // bidirectional check against `exp` (so a bare `Nil` field is pinned), then returns the
                // ascribed type. `exp` is the **source-named** concrete type (re-inference resolves names
                // against the source env), and it is concrete, so the ascription is exact (never a
                // coercion — S1).
                let ascribed = Expr::Ascribe(Box::new(e.clone()), ty_to_source_ref(exp));
                self.infer(site, scope, &ascribed)
            }
        }
    }
}

// ----- free helpers ----------------------------------------------------------------------------

/// RFC-0024 §4A.2 (M-704): the canonical name of the generated `apply` dispatcher's value argument
/// (the `x` in `apply(clo, x)`). Uses the `%` fresh-variable character (never a surface-identifier
/// char — `crate::elab`), so it can never collide with a captured variable or the lambda's parameter.
const APPLY_PARAM: &str = "%fnarg";

/// RFC-0024 §4A.4 (M-704): the **field type** of a captured variable inside a closure tag-sum's
/// constructor. A fn-typed capture (a closure capturing a closure) becomes its own arrow tag-sum
/// `Fn$<inner>` (a nullary `Ty::Data`); everything else takes the existing mangled-nullary form
/// ([`mangle_ty_in_ty`]) the registry/`field_spec` already consume.
fn closure_field_ty(t: &Ty) -> Ty {
    match t {
        Ty::Fn(a, b) => Ty::Data(mangle_arrow(a, b), vec![]),
        _ => mangle_ty_in_ty(t),
    }
}

/// RFC-0024 §4A.2 (M-704): the surface [`TypeRef`] for an `apply` dispatcher's param/return type. A
/// fn-typed position (a higher-order arrow `(B => C) => D` etc.) is the closure data type
/// `Fn$<inner>`; everything else round-trips via [`ty_to_ref`].
fn closure_param_ref(t: &Ty) -> TypeRef {
    match t {
        Ty::Fn(a, b) => ty_to_ref(&Ty::Data(mangle_arrow(a, b), vec![])),
        _ => ty_to_ref(t),
    }
}

/// RFC-0024 §4A.3 (M-704): the **free-variable walk** for closure capture analysis. Collects the
/// single-segment `Expr::Path` names occurring in `e` that are **not** bound by an enclosing binder
/// *within* `e` (`Let`, `Match` arm patterns, `For`, inner `Lambda` params) — appended to `out` in
/// **first-occurrence order** (a total deterministic order — §4A.3 / G2), each once (`seen` dedups).
/// `bound` carries the names currently in scope inside `e` (seeded with the lambda's own parameter
/// by the caller); a name in `bound` is local, not free. Whether a *free* name is actually
/// *captured* (a local of the enclosing scope) vs. a top-level reference is decided by the caller
/// against the enclosing scope (`rewrite_lambda`).
///
/// This is a pure structural walk — never silent, never a guess (G2): every binder the AST exposes
/// is respected, so `freevars` is invariant under α-renaming of bound variables (the §4A.9 property).
///
/// # Errors
/// [`WalkDepthExceeded`] once this traversal's own recursion exceeds [`MAX_WALK_DEPTH`] (M-866,
/// mirroring the totality/checker/elaborator `M-674` discipline — see [`free_vars_at`]) — a clean,
/// explicit refusal rather than a host-stack overflow on a pathologically-nested `e` (G2).
pub(crate) fn free_vars(
    e: &Expr,
    bound: &mut BTreeSet<String>,
    seen: &mut BTreeSet<String>,
    out: &mut Vec<String>,
) -> Result<(), WalkDepthExceeded> {
    free_vars_at(e, bound, seen, out, 0)
}

/// The depth-tracked worker behind [`free_vars`] (M-866): `depth` counts the live nesting of this
/// traversal's own recursive descent (not any semantic property of `e`), charged on entry and
/// checked against [`MAX_WALK_DEPTH`] before any further recursion — mirrors
/// [`crate::totality::walk_expr_at`]'s M-674 discipline (same budget, same reified-counter
/// mechanism, DRY): rather than rely on the host call stack (a resource, not a semantic limit) to
/// bound the walk, the pass carries this explicit budget and refuses past it with a clean
/// [`WalkDepthExceeded`], never a host-stack overflow.
fn free_vars_at(
    e: &Expr,
    bound: &mut BTreeSet<String>,
    seen: &mut BTreeSet<String>,
    out: &mut Vec<String>,
    depth: u32,
) -> Result<(), WalkDepthExceeded> {
    let depth = depth + 1;
    if depth > MAX_WALK_DEPTH {
        return Err(WalkDepthExceeded {
            limit: MAX_WALK_DEPTH,
        });
    }
    match e {
        Expr::Path(p) => {
            if p.0.len() == 1 {
                let n = &p.0[0];
                if !bound.contains(n) && seen.insert(n.clone()) {
                    out.push(n.clone());
                }
            }
        }
        Expr::Lit(Literal::List(elems)) => {
            for el in elems {
                free_vars_at(el, bound, seen, out, depth)?;
            }
        }
        Expr::Lit(_) => {}
        Expr::Let {
            name,
            bound: b,
            body,
            ..
        } => {
            free_vars_at(b, bound, seen, out, depth)?;
            // `name` is bound in `body` only (let is non-recursive at the surface). Respect shadowing.
            let was = bound.insert(name.clone());
            free_vars_at(body, bound, seen, out, depth)?;
            if was {
                bound.remove(name);
            }
        }
        Expr::If { cond, conseq, alt } => {
            free_vars_at(cond, bound, seen, out, depth)?;
            free_vars_at(conseq, bound, seen, out, depth)?;
            free_vars_at(alt, bound, seen, out, depth)?;
        }
        Expr::Match { scrutinee, arms } => {
            free_vars_at(scrutinee, bound, seen, out, depth)?;
            for arm in arms {
                let mut added: Vec<String> = Vec::new();
                pattern_binders(&arm.pattern, bound, &mut added)?;
                free_vars_at(&arm.body, bound, seen, out, depth)?;
                for n in added {
                    bound.remove(&n);
                }
            }
        }
        Expr::For {
            x,
            xs,
            acc,
            init,
            body,
        } => {
            free_vars_at(xs, bound, seen, out, depth)?;
            free_vars_at(init, bound, seen, out, depth)?;
            let ax = bound.insert(x.clone());
            let aacc = bound.insert(acc.clone());
            free_vars_at(body, bound, seen, out, depth)?;
            if aacc {
                bound.remove(acc);
            }
            if ax {
                bound.remove(x);
            }
        }
        Expr::Swap { value, .. } => free_vars_at(value, bound, seen, out, depth)?,
        Expr::WithParadigm { body, .. } => free_vars_at(body, bound, seen, out, depth)?,
        Expr::Wild(b) | Expr::Spore(b) | Expr::Consume(b) => {
            free_vars_at(b, bound, seen, out, depth)?;
        }
        Expr::Colony(hyphae) => {
            for h in hyphae {
                free_vars_at(&h.body, bound, seen, out, depth)?;
            }
        }
        Expr::Lambda { params, body } => {
            // An inner lambda's params shadow inside its body (nested closures — §4A.3).
            let mut added: Vec<String> = Vec::new();
            for p in params {
                if bound.insert(p.name.clone()) {
                    added.push(p.name.clone());
                }
            }
            free_vars_at(body, bound, seen, out, depth)?;
            for n in added {
                bound.remove(&n);
            }
        }
        Expr::App { head, args } => {
            free_vars_at(head, bound, seen, out, depth)?;
            for a in args {
                free_vars_at(a, bound, seen, out, depth)?;
            }
        }
        Expr::Fuse { left, right } => {
            free_vars_at(left, bound, seen, out, depth)?;
            free_vars_at(right, bound, seen, out, depth)?;
        }
        Expr::Reclaim { policy, body } => {
            free_vars_at(policy, bound, seen, out, depth)?;
            free_vars_at(body, bound, seen, out, depth)?;
        }
        Expr::Ascribe(inner, _) => free_vars_at(inner, bound, seen, out, depth)?,
        // M-826: a tuple literal's elements are all value positions; walk each for free variables.
        Expr::TupleLit(elems) => {
            for el in elems {
                free_vars_at(el, bound, seen, out, depth)?;
            }
        }
    }
    Ok(())
}

/// Collect a pattern's binders into `bound` (inserting each newly-bound name), recording the
/// newly-added names in `added` so the caller can pop them after the arm body. A `Pattern::Ident` is
/// a binder; a `Pattern::Ctor` recurses into sub-patterns; `Wildcard`/`Lit` bind nothing. (A nullary
/// constructor written as a bare `Ident` is conservatively treated as a binder here — over-binding
/// only ever *removes* a name from the capture set, never adds a spurious capture, so it is safe for
/// the free-variable analysis; the real ctor/binder distinction is the checker's, already done.)
///
/// # Errors
/// [`WalkDepthExceeded`] once this traversal's own recursion exceeds [`MAX_WALK_DEPTH`] (M-866) — a
/// clean, explicit refusal rather than a host-stack overflow on a pathologically-nested pattern.
/// Mirrors [`crate::totality::pattern_binders`]'s own M-674 depth-budget discipline (same value,
/// same reified-counter mechanism, DRY) — the two walk distinct pass concerns (free-variable capture
/// here vs. structural-descent binder tracking there) so the code is not literally shared, but the
/// budget and refusal shape are kept identical for one crate-wide "AST pass depth" guarantee.
fn pattern_binders(
    pat: &Pattern,
    bound: &mut BTreeSet<String>,
    added: &mut Vec<String>,
) -> Result<(), WalkDepthExceeded> {
    pattern_binders_at(pat, bound, added, 0)
}

/// The depth-tracked worker behind [`pattern_binders`] (M-866) — see [`free_vars_at`] for the
/// shared discipline this mirrors.
fn pattern_binders_at(
    pat: &Pattern,
    bound: &mut BTreeSet<String>,
    added: &mut Vec<String>,
    depth: u32,
) -> Result<(), WalkDepthExceeded> {
    let depth = depth + 1;
    if depth > MAX_WALK_DEPTH {
        return Err(WalkDepthExceeded {
            limit: MAX_WALK_DEPTH,
        });
    }
    match pat {
        Pattern::Wildcard | Pattern::Lit(_) => {}
        Pattern::Ident(n) => {
            if bound.insert(n.clone()) {
                added.push(n.clone());
            }
        }
        Pattern::Ctor(_, subs) => {
            for s in subs {
                pattern_binders_at(s, bound, added, depth)?;
            }
        }
        // M-826: a tuple pattern `(x, y, …)` binds each sub-pattern element.
        Pattern::Tuple(subs) => {
            for s in subs {
                pattern_binders_at(s, bound, added, depth)?;
            }
        }
        // `Pattern::Or` is desugared in `check_match` before monomorphization; reaching here
        // means the program was not checked — an invariant violation (G2: never silent).
        Pattern::Or(_) => panic!(
            "internal: Pattern::Or reached mono::pattern_binders — or-patterns must be \
             desugared by the checker before any downstream pass \
             (invariant violation — report this)"
        ),
    }
    Ok(())
}

/// The canonical dedup key of a work item — a kind-tagged string so a function and a data type that
/// happen to mangle to the same name never alias, and `Ty` needs no `Ord` (just its `Display`).
fn item_key(item: &Item) -> String {
    match item {
        Item::Fn {
            name,
            targs,
            wargs,
            fn_args,
            dyn_fns,
        } => format!(
            "fn:{}",
            mangle_hof_decl(name, targs, wargs, fn_args, dyn_fns)
        ),
        Item::Data { name, targs } => format!("data:{}", mangle_decl(name, targs)),
        Item::Method {
            trait_name,
            method,
            for_ty,
        } => format!("method:{}", mangle_method(method, trait_name, for_ty)),
    }
}

/// Mangle a HOF-specialization declaration name at concrete type arguments **and** fn arguments
/// (RFC-0024 §4, M-687). Extends [`mangle_decl`]: after the type-argument segments (`$`-joined),
/// appends fn-argument segments as `%{param_index}:{callee_mangled}` per baked-in fn parameter.
///
/// The `%` separator is the elaborator's fresh-variable character (never a surface-identifier
/// character), so a HOF-specialization mangled name is **disjoint** from:
/// - surface names (no `$`/`#`/`%` in the Mycelium lexer)
/// - trait-method mangled names (`method$Trait$ForTy` — no `%`)
/// - type-only specializations (`name$TyArg…` — no `%`)
/// - data-repr names (no `%`)
///
/// This preserves the overall injective, surface-disjoint property of the mangling scheme (G2).
///
/// **Empty `fn_args` delegates to [`mangle_decl`]** — so a fn with no HOF params produces the
/// exact same mangled name as before M-687 (backward-compatible with the existing corpus).
pub(crate) fn mangle_hof_decl(
    name: &str,
    targs: &[Ty],
    wargs: &[Width],
    fn_args: &[(usize, String)],
    dyn_fns: &[(usize, String)],
) -> String {
    // DN-42 / M-753 step-c: include width arguments in the mangled name so two calls at different
    // widths produce distinct specializations (identity fragmentation; G2 / never-silent).
    // Width args are appended after type args using the same `$` joint; Width::Lit(n) becomes
    // `Binary{n}` via mangle_ty (consistent with type-arg mangling). Width::Var should never
    // reach here (mono refuses undetermined params first).
    let base = mangle_decl_with_wargs(name, targs, wargs);
    if fn_args.is_empty() && dyn_fns.is_empty() {
        return base;
    }
    let mut s = base;
    for (idx, callee) in fn_args {
        s.push('%');
        s.push_str(&idx.to_string());
        s.push(':');
        s.push_str(callee);
    }
    // RFC-0024 §4A (M-704): dynamic (kept-as-closure) fn parameters get a distinct `~` joint so a
    // dynamically-specialized HOF is never confused with a statically-specialized one (`%`) or a
    // plain specialization (neither). `~` is not a surface-identifier character (the lexer never
    // produces it), preserving the injective, surface-disjoint mangling property (G2).
    for (idx, arrow) in dyn_fns {
        s.push('~');
        s.push_str(&idx.to_string());
        s.push(':');
        s.push_str(arrow);
    }
    s
}

/// Mangle a declaration name at concrete type arguments **and** width arguments (DN-42 / M-753
/// step-c). Width args are appended after type args using `$` joints:
/// `add<N>` at N=8 → `add$Binary8`. Width::Var should never reach here.
fn mangle_decl_with_wargs(name: &str, targs: &[Ty], wargs: &[Width]) -> String {
    let mut s = mangle_decl(name, targs);
    for w in wargs {
        s.push('$');
        match w {
            Width::Lit(n) => s.push_str(&format!("Binary{n}")),
            Width::Var(v) => s.push_str(&format!("WVAR_{v}")), // should not reach here (VR-5)
        }
    }
    s
}

fn residual<T>(site: &str, what: impl Into<String>) -> Result<T, ElabError> {
    Err(ElabError::Residual {
        site: site.to_owned(),
        what: what.into(),
    })
}

/// Wrap a checker [`crate::checkty::CheckError`] as an elaboration [`ElabError::Residual`] (the
/// re-inference primitives return `CheckError`; mono surfaces them as residuals — never silent).
fn res_err(site: &str, e: crate::checkty::CheckError) -> ElabError {
    ElabError::Residual {
        site: site.to_owned(),
        what: format!("monomorphization re-inference: {e}"),
    }
}

/// One-sided unification (the checker's [`crate::checkty::unify`]) surfacing its failure as a
/// residual. Binds the abstract `decl`'s type-vars from the concrete `actual`.
fn unify_into(
    site: &str,
    decl: &Ty,
    actual: &Ty,
    s: &mut BTreeMap<String, Ty>,
) -> Result<(), ElabError> {
    unify(site, decl, actual, s).map_err(|e| res_err(site, e))
}

/// Mangle a type to a flat identifier-suffix fragment (injective; `$`-free for primitives, `$`-joined
/// for applied data). `Binary{8}`→`Binary8`, `Ternary{6}`→`Ternary6`, `Dense{16,F32}`→`Dense16F32`,
/// `Data("List",[Binary8])`→`List$Binary8`, nullary `Data("Bool",[])`→`Bool`.
pub(crate) fn mangle_ty(t: &Ty) -> String {
    match t {
        Ty::Binary(Width::Lit(n)) => format!("Binary{n}"),
        Ty::Binary(Width::Var(v)) => format!("BinaryVAR_{v}"),
        Ty::Ternary(Width::Lit(m)) => format!("Ternary{m}"),
        Ty::Ternary(Width::Var(v)) => format!("TernaryVAR_{v}"),
        Ty::Dense(d, s) => format!("Dense{d}{}", scalar_tag(*s)),
        // RFC-0003 §3 (M-892): `VSA{model, dim, sparsity}` mangles like `Seq` (the `$` separates
        // the shape fragment from the model id, whose `-` — not an identifier char — maps to `_`;
        // injective over the kernel model-id alphabet). `VSA{MAP-I, 256, Dense}` → `Vsa256Dn$MAP_I`.
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => {
            let sp = match sparsity {
                Sparsity::Dense => "Dn".to_owned(),
                Sparsity::Sparse(k) => format!("Sp{k}"),
            };
            format!("Vsa{dim}{sp}${}", model.replace('-', "_"))
        }
        Ty::Substrate(tag) => format!("Substrate{tag}"),
        // RFC-0032 D3/D4: `Seq{T, N}` mangles to `SeqN$<elem>` (injective — the `$` separates the
        // length from the recursively-mangled element); `Bytes` is nullary.
        Ty::Seq(elem, n) => format!("Seq{n}${}", mangle_ty(elem)),
        Ty::Bytes => "Bytes".to_owned(),
        // ADR-040 (M-897): the nullary scalar-float repr mangles like `Bytes` (a data type named
        // `Float` mangles to `Float#` via the `#` tag below — no collision, injectivity holds).
        Ty::Float => "Float".to_owned(),
        // A nullary data type tags its name with `#` (not a surface-identifier char — the lexer
        // never produces it), so a data type whose name happens to equal a repr mangle (e.g. a type
        // literally named `Binary8`) becomes `Binary8#` and can NEVER collide with the repr
        // `Binary{8}` → `Binary8`. This keeps `mangle_ty`/`mangle_decl`/`item_key` injective across
        // the repr/data boundary, so two distinct instantiations never alias to one mangled name (no
        // silent drop — G2). The `#` appears only inside a composite name; a monomorphic data type is
        // still registered and referenced under its bare name (`mangle_ty_in_ty` clones a nullary
        // `Data` directly), so monomorphic passthrough is unaffected.
        Ty::Data(n, args) if args.is_empty() => format!("{n}#"),
        Ty::Data(n, args) => {
            let mut s = n.clone();
            for a in args {
                s.push('$');
                s.push_str(&mangle_ty(a));
            }
            s
        }
        // A `Ty::Var` must never reach mangling (mono refuses an undetermined parameter first); a
        // distinctive marker keeps a hypothetical leak observable rather than silently collidable.
        Ty::Var(v) => format!("VAR_{v}"),
        // RFC-0024 §4 / M-687: function-type parameters are defunctionalized in M-687.
        // A `Ty::Fn` reaching mangling before M-687 is a bug — use a distinctive, non-collidable
        // marker so the leak surfaces loudly (never silently — G2/VR-5).
        Ty::Fn(a, r) => format!("HOF_FN_{}__TO__{}", mangle_ty(a), mangle_ty(r)),
    }
}

/// The scalar tag used inside [`mangle_ty`] (`F16`/`BF16`/`F32`/`F64`).
fn scalar_tag(s: Scalar) -> &'static str {
    match s {
        Scalar::F16 => "F16",
        Scalar::Bf16 => "BF16",
        Scalar::F32 => "F32",
        Scalar::F64 => "F64",
    }
}

/// Mangle a declaration name (fn or data type) at concrete type arguments: `name` + `"$" + mangle_ty`
/// per argument. **Empty `targs` ⇒ the original name, byte-for-byte** — so monomorphic code and
/// non-generic programs are untouched.
pub(crate) fn mangle_decl(name: &str, targs: &[Ty]) -> String {
    if targs.is_empty() {
        return name.to_owned();
    }
    let mut s = name.to_owned();
    for t in targs {
        s.push('$');
        s.push_str(&mangle_ty(t));
    }
    s
}

/// Mangle a **constructor** name at its data type's concrete arguments — same scheme as
/// [`mangle_decl`] (empty `targs` ⇒ unchanged). Distinct instantiations get distinct ctor names so the
/// registry / [`Env::ctor`] key stays globally unique across mono'd data types.
pub(crate) fn mangle_ctor(name: &str, targs: &[Ty]) -> String {
    mangle_decl(name, targs)
}

/// Mangle a trait method to the direct monomorphic fn name `method$Trait$ForTy` — e.g.
/// `cmp$Cmp$Binary8`. The receiver is mangled with [`mangle_ty`]; the name encodes (method, trait,
/// receiver), which is the honest queryable identity of the resolved dispatch.
pub(crate) fn mangle_method(method: &str, trait_name: &str, for_ty: &Ty) -> String {
    format!("{method}${trait_name}${}", mangle_ty(for_ty))
}

/// RFC-0024 §4A.4 (M-704): mangle a closure arrow type `A => B` to the tag-sum data name
/// `Fn$<A>$<B>` — the same injective, surface-disjoint scheme as [`mangle_decl`] (`$` joints, the
/// `#` nullary-data tag inside [`mangle_ty`]). A nested arrow recurses ([`mangle_ty_or_fn`]), so a
/// closure-capturing-closure's arrow names its inner arrow's tag-sum. Distinct arrows ⇒ distinct
/// names (no silent alias — G2).
pub(crate) fn mangle_arrow(a: &Ty, b: &Ty) -> String {
    format!("Fn${}${}", mangle_ty_or_fn(a), mangle_ty_or_fn(b))
}

/// The generated dispatcher fn name for an arrow mangle `Fn$A$B` → `apply$A$B` (RFC-0024 §4A.2). The
/// `Fn$`-prefix is stripped so the dispatcher and its sum share the `A$B` suffix (queryable identity).
pub(crate) fn apply_fn_name(arrow: &str) -> String {
    let suffix = arrow.strip_prefix("Fn$").unwrap_or(arrow);
    format!("apply${suffix}")
}

/// Like [`mangle_ty`] but mangles a `Ty::Fn` to its arrow tag-sum name (`Fn$A$B`) rather than the
/// loud `HOF_FN_…` leak marker — used inside closure mangling where a fn-typed capture/codomain is a
/// real, lowered closure type (RFC-0024 §4A). Non-fn types delegate to [`mangle_ty`].
pub(crate) fn mangle_ty_or_fn(t: &Ty) -> String {
    match t {
        Ty::Fn(a, b) => mangle_arrow(a, b),
        _ => mangle_ty(t),
    }
}

/// Rewrite a concrete `Ty` so every applied data type becomes its **mangled-nullary** form
/// (`Data("List$Binary8", [])`), the shape `build_registry`/`field_spec` already handle. Primitive
/// reprs pass through unchanged.
fn mangle_ty_in_ty(t: &Ty) -> Ty {
    match t {
        Ty::Binary(_)
        | Ty::Ternary(_)
        | Ty::Dense(_, _)
        // M-892: the VSA repr is a primitive — passes through unchanged.
        | Ty::Vsa { .. }
        | Ty::Substrate(_)
        | Ty::Bytes
        | Ty::Float => t.clone(),
        // RFC-0032 D3: mangle the element type (it may carry a mono'd applied data type), keeping the
        // sequence structure; primitive element reprs pass through unchanged.
        Ty::Seq(elem, n) => Ty::Seq(Box::new(mangle_ty_in_ty(elem)), *n),
        Ty::Data(_, args) if args.is_empty() => t.clone(),
        Ty::Data(_, _) => Ty::Data(mangle_ty(t), vec![]),
        Ty::Var(v) => Ty::Var(v.clone()), // defended against earlier; pass through if it ever appears
        // RFC-0024 §4 / M-687: function types pass through un-mangled; the defunctionalization
        // rewrite in M-687 will eliminate them before any fn mangle/registry step.
        Ty::Fn(_, _) => t.clone(),
    }
}

/// Convert a concrete checked [`Ty`] back to a **source-named** surface [`TypeRef`] (no guarantee
/// index) — an applied data type keeps its original name and recurses into its arguments
/// (`List<Binary{8}>` → `Named("List", [Binary{8}])`). Used to thread an `expected` type into
/// re-inference (`infer_type`), which resolves names against the **source** env. (Contrast
/// [`ty_to_ref`], which produces the *mangled-nullary* output form for the emitted env.)
fn ty_to_source_ref(t: &Ty) -> TypeRef {
    let base = match t {
        Ty::Binary(Width::Lit(n)) => BaseType::Binary(WidthRef::Lit(*n)),
        Ty::Binary(Width::Var(v)) => BaseType::Binary(WidthRef::Name(v.clone())),
        Ty::Ternary(Width::Lit(m)) => BaseType::Ternary(WidthRef::Lit(*m)),
        Ty::Ternary(Width::Var(v)) => BaseType::Ternary(WidthRef::Name(v.clone())),
        Ty::Dense(d, s) => BaseType::Dense(*d, *s),
        // M-892: round-trip the VSA repr. The checked model is the canonical kernel id
        // (`MAP-I`); `resolve_ty`'s canonicalization is idempotent on kernel ids, so threading
        // it back through re-inference is stable (the parser itself can only produce the
        // underscore surface spelling — this ref is checker-internal).
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => BaseType::Vsa {
            model: model.clone(),
            dim: *dim,
            sparsity: sparsity.clone(),
        },
        Ty::Substrate(tag) => BaseType::Substrate(tag.clone()),
        // RFC-0032 D3/D4: round-trip the sequence/byte-string reprs to their surface forms.
        Ty::Seq(elem, n) => BaseType::Seq {
            elem: Box::new(ty_to_source_ref(elem)),
            len: *n,
        },
        Ty::Bytes => BaseType::Bytes,
        // ADR-040 (M-897): the nullary scalar-float repr round-trips like `Bytes`.
        Ty::Float => BaseType::Float,
        Ty::Data(n, args) => {
            BaseType::Named(n.clone(), args.iter().map(ty_to_source_ref).collect())
        }
        Ty::Var(v) => BaseType::Named(v.clone(), vec![]),
        // RFC-0024 §4 / M-687: function types round-trip as `BaseType::Fn`. Used only for re-inference
        // context threading; defunctionalization (M-687) rewrites them before any registry step.
        Ty::Fn(a, r) => BaseType::Fn(Box::new(ty_to_source_ref(a)), Box::new(ty_to_source_ref(r))),
    };
    TypeRef::unguaranteed(base)
}

/// Convert a concrete checked [`Ty`] back to a surface [`TypeRef`] (no guarantee index) so a rewritten
/// `FnDecl`/`Param`/`Ascribe` carries a concrete surface type. Mono erases type variables and bakes a
/// data type's arguments into its **mangled-nullary** name, so an applied `Ty::Data(_, args!=[])`
/// becomes the `Named` of its mangled name; a `Ty::Var` would be an internal error, surfaced as a
/// distinctive `Named` so a leak is never silent (rather than a panic).
fn ty_to_ref(t: &Ty) -> TypeRef {
    let base = match t {
        Ty::Binary(Width::Lit(n)) => BaseType::Binary(WidthRef::Lit(*n)),
        Ty::Binary(Width::Var(v)) => BaseType::Binary(WidthRef::Name(v.clone())),
        Ty::Ternary(Width::Lit(m)) => BaseType::Ternary(WidthRef::Lit(*m)),
        Ty::Ternary(Width::Var(v)) => BaseType::Ternary(WidthRef::Name(v.clone())),
        Ty::Dense(d, s) => BaseType::Dense(*d, *s),
        // M-892: round-trip the VSA repr (kernel model id — idempotent under re-resolution;
        // see `ty_to_source_ref`).
        Ty::Vsa {
            model,
            dim,
            sparsity,
        } => BaseType::Vsa {
            model: model.clone(),
            dim: *dim,
            sparsity: sparsity.clone(),
        },
        Ty::Substrate(tag) => BaseType::Substrate(tag.clone()),
        // RFC-0032 D3/D4: round-trip the sequence/byte-string reprs (the element type is mono'd to a
        // concrete surface form via the same `ty_to_ref`).
        Ty::Seq(elem, n) => BaseType::Seq {
            elem: Box::new(ty_to_ref(elem)),
            len: *n,
        },
        Ty::Bytes => BaseType::Bytes,
        // ADR-040 (M-897): the nullary scalar-float repr round-trips like `Bytes`.
        Ty::Float => BaseType::Float,
        // A mono'd data type is nullary (its arguments are baked into its mangled name).
        Ty::Data(n, args) if args.is_empty() => BaseType::Named(n.clone(), vec![]),
        Ty::Data(_, _) => BaseType::Named(mangle_ty(t), vec![]),
        Ty::Var(v) => BaseType::Named(format!("VAR_{v}"), vec![]),
        // RFC-0024 §4 / M-687: function types in rewritten fn-decl positions; defunctionalization
        // in M-687 will eliminate these. Preserve as `BaseType::Fn` so the AST stays structurally
        // sound (never a silent drop or panic — G2/VR-5).
        Ty::Fn(a, r) => BaseType::Fn(Box::new(ty_to_ref(a)), Box::new(ty_to_ref(r))),
    };
    TypeRef::unguaranteed(base)
}

/// Like [`ty_to_ref`], but attaches `guarantee` — the **source declaration's own** `@ g` (M-844 /
/// M-967; RFC-0018 §4). Every call site passes the guarantee straight off the *original* (still
/// abstract/generic) [`TypeRef`] being specialized (e.g. `p.ty.guarantee`, `fd.sig.ret.guarantee`)
/// — never derived from the concrete `Ty` (which carries none) and never merged/averaged across
/// instantiations (each call builds one fresh `TypeRef` for one specialization). This is the sole
/// per-instantiation guarantee-tag threading point: a monomorphized copy's signature/ascription
/// keeps exactly what its own source wrote — no silent loss (the pre-M-967 `ty_to_ref` blanking),
/// no silent merge across instantiations, and no upgrade past the source's own annotation (VR-5).
fn ty_to_ref_tagged(t: &Ty, guarantee: Option<Strength>) -> TypeRef {
    let mut r = ty_to_ref(t);
    r.guarantee = guarantee;
    r
}
