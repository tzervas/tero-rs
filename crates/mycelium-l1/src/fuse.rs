//! DN-58 §A (M-965; F-A1/F-A2) — the `Fuse` **prelude surface** + the **semilattice-law checker**.
//!
//! **F-A1 (prelude).** `Fuse` is now a **built-in trait** — the trait analogue of the built-in
//! `Bool` data type ([`crate::checkty::prelude`]): a nodule never has to hand-write
//! `trait Fuse[T] { fn join(a: T, b: T) => T; }` before writing `impl Fuse[T] for T { … }`.
//! [`prelude`] is the single source of the built-in [`TraitInfo`] that
//! [`crate::checkty::register_nodule_decls`] seeds every nodule's trait registry with (mirroring
//! how `register_nodule_decls` seeds `Bool` into the type registry).
//!
//! DN-58 §A.2 proposed the surface `trait Fuse { fn join(self: Self, other: Self) => Self; }` —
//! but this stage-1 trait model (RFC-0019 §4.1) has no implicit `Self` slot; every trait is
//! parameterized by an **explicit** type-variable name, and "generic over the implementing type"
//! is already the established idiom of repeating that name at the impl site
//! (`impl Tr[T] for T { … }` — see `crates/mycelium-l1/src/tests/checkty.rs`). `Fuse` reuses that
//! idiom verbatim (`impl Fuse[T] for T`) rather than inventing a new trait-model feature; this is
//! the concrete resolution of DN-58 §A.6's "trait name is open" flag (`Fuse` — matches the
//! keyword) plus the `Self`-vs-explicit-parameter question DN-58 left implicit.
//!
//! **F-A2 (law checker).** [`check_fuse_laws`] is the [`crate::checkty::check_nodule_with`]
//! post-pass that the `checkty.rs` `check_fuse` doc comment used to flag as "not yet wired". It
//! empirically verifies a declared `Fuse` instance's `join` obeys the three semilattice laws
//! (RFC-0008 RT6; DN-58 §A.1): **idempotence** (`join(a, a) = a`), **commutativity**
//! (`join(a, b) = join(b, a)`), and **associativity**
//! (`join(join(a, b), c) = join(a, join(b, c))`). A violation is an explicit, never-silent
//! [`CheckError`] naming the failed law, a concrete counterexample, and the `impl`'s site (G2) —
//! refused **here, at definition time**, never reaching production (RFC-0008 RT6's convergence
//! guarantee is only as good as the laws it assumes; this is the "caught at definition, not
//! production" checker the DN-58 §A.4 obligation calls for).
//!
//! **Honest scope (VR-5) — exhaustive, not general.** The check is **exhaustive** over a finite,
//! enumerable `for_ty` — in v0, a **nullary-constructor `Data` type** (every constructor takes zero
//! fields; the same shape as the prelude `Bool`, or a user `type Sign = Neg | Zero | Pos;`). For
//! any other `for_ty` (one with fields, a parametric/recursive type, a repr type — `Binary`/
//! `Ternary`/… are handled by `checkty::check_fuse`'s separate always-fusible repr path, never by
//! an `impl`) the domain is not finitely enumerable in v0, so [`check_fuse_laws`] **skips** the
//! check rather than sampling or guessing — it never silently claims a law holds without having
//! checked it. Widening this to a sampled/property-tested domain for composite types is the DN-58
//! §A.6 F-A3 follow-on: deferred, and named here rather than silently absent.
//!
//! Guarantee: the exhaustive-domain law verification here is `Empirical` (every value of the
//! domain is tried; this is a complete case-enumeration for a *finite* domain, not a mechanized
//! proof over an inductive definition — RFC-0008 RT6's Isabelle/HOL basis is a proof about the
//! *general* semilattice-merge construction, not automatically re-derived for each concrete
//! instance). Never `Proven` without that mechanized basis (VR-5).

use std::collections::BTreeMap;

use crate::ast::{BaseType, FnDecl, FnSig, Param, TypeRef, Vis};
use crate::checkty::{CheckError, DataInfo, Env, InstanceInfo, TraitInfo, Ty};
use crate::eval::Evaluator;

/// The synthetic top-level-fn name a `Fuse` instance's `join` method is probed under while the law
/// checker evaluates it. `#` cannot appear in a surface identifier (the lexer never produces it),
/// so this can never collide with a real program's function name.
const PROBE_FN: &str = "#fuse_law_probe#";

/// F-A1 — the built-in `Fuse` trait (DN-58 §A.2): `trait Fuse[T] { fn join(a: T, b: T) => T; }`.
/// Hand-built in Rust (mirrors [`crate::checkty::prelude`]'s `Bool` `DataInfo`) rather than parsed
/// from surface syntax, so the parameter name `T` (standing in for DN-58's proposed `Self`) is
/// just an ordinary trait type-variable — no new trait-model feature.
#[must_use]
pub(crate) fn prelude() -> TraitInfo {
    let t = |name: &str| TypeRef::unguaranteed(BaseType::Named(name.to_owned(), vec![]));
    TraitInfo {
        name: "Fuse".to_owned(),
        params: vec!["T".to_owned()],
        sigs: vec![FnSig {
            name: "join".to_owned(),
            params: vec![],
            value_params: vec![
                Param {
                    name: "a".to_owned(),
                    ty: t("T"),
                },
                Param {
                    name: "b".to_owned(),
                    ty: t("T"),
                },
            ],
            ret: t("T"),
            effects: vec![],
            effect_budgets: BTreeMap::new(),
        }],
    }
}

/// This trait's name — the one string every registration/lookup site must agree on (Law of
/// Demeter — a single named constant beats a scattered literal `"Fuse"`).
pub(crate) const TRAIT_NAME: &str = "Fuse";

/// If `ty` is a **finite, enumerable** domain in v0's sense (a registered `Data` type every one of
/// whose constructors takes zero fields — the `Bool`-shape), return every value of that domain;
/// `None` for anything else (an unregistered name, a type with fielded constructors, a repr type,
/// an uninhabited type) — never a guess at a partial enumeration (G2).
fn enumerate_finite_domain(
    ty: &Ty,
    types: &BTreeMap<String, DataInfo>,
) -> Option<Vec<crate::eval::L1Value>> {
    let Ty::Data(name, _args) = ty else {
        return None;
    };
    let info = types.get(name)?;
    if info.ctors.is_empty() || info.ctors.iter().any(|c| !c.fields.is_empty()) {
        return None;
    }
    Some(
        info.ctors
            .iter()
            .map(|c| crate::eval::L1Value::Data {
                ty: name.clone(),
                ctor: c.name.clone(),
                fields: std::sync::Arc::new(vec![]),
            })
            .collect(),
    )
}

/// F-A2 — the semilattice-law checker. Walks every registered instance of the [`TRAIT_NAME`]
/// trait; for each whose `for_ty` is a [`enumerate_finite_domain`] domain, exhaustively checks
/// idempotence, commutativity, and associativity by evaluating `join` (via a scratch
/// [`Evaluator`]) over every value/pair/triple of that domain. The first law violation found is
/// returned as a never-silent [`CheckError`] naming the law, a witness, and the `impl`'s site
/// (G2); a non-enumerable `for_ty` is silently **skipped** — not silently *accepted as lawful*,
/// just left unchecked (documented above, VR-5).
///
/// # Errors
/// A [`CheckError`] naming the violated law + witness for the first unlawful `Fuse` instance
/// found (site order = `BTreeMap` iteration order over `(trait, head)`, i.e. deterministic).
pub(crate) fn check_fuse_laws(
    types: &BTreeMap<String, DataInfo>,
    fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    instances: &BTreeMap<(String, String), InstanceInfo>,
    impls: &BTreeMap<(String, String), Vec<FnDecl>>,
) -> Result<(), CheckError> {
    for ((trait_name, head), methods) in impls {
        if trait_name != TRAIT_NAME {
            continue;
        }
        let Some(inst) = instances.get(&(trait_name.clone(), head.clone())) else {
            // Invariant: every `impls` entry has a matching `instances` entry (both are populated
            // from the same registered `impl` set — `register_instances` then `check_impl_methods`
            // over the same items). Never reachable; skip rather than panic (G2 in spirit — this
            // function stays total even if that invariant is ever loosened elsewhere).
            continue;
        };
        // `check_impl_method_set` (registration) already enforced the impl's method set matches
        // the trait's requirement **exactly**, so a `Fuse` impl always has exactly one method named
        // `join`; this `find` cannot miss in practice, but we don't `expect` — a future trait-shape
        // change should fail closed (skip), never panic.
        let Some(join) = methods.iter().find(|m| m.sig.name == "join") else {
            continue;
        };
        let Some(domain) = enumerate_finite_domain(&inst.for_ty, types) else {
            continue;
        };
        check_laws_over_domain(
            &inst.for_ty,
            join,
            &domain,
            types,
            fns,
            traits,
            instances,
            impls,
        )?;
    }
    Ok(())
}

/// Exhaustively check the three semilattice laws for `join` over every value of `domain`
/// (`for_ty`'s enumerated values), evaluating through a scratch [`Env`] that carries every
/// already-checked registry (so `join`'s body may call ordinary top-level fns / other trait
/// methods exactly as it would in the real program) plus one synthetic entry — `join` itself,
/// under [`PROBE_FN`] — so [`Evaluator::call`] can invoke it directly without going through the
/// generic trait-method call-resolution surface (unnecessary here: `for_ty` is already concrete,
/// so there is exactly one candidate method, no ambiguity to resolve).
#[allow(clippy::too_many_arguments)]
fn check_laws_over_domain(
    for_ty: &Ty,
    join: &FnDecl,
    domain: &[crate::eval::L1Value],
    types: &BTreeMap<String, DataInfo>,
    fns: &BTreeMap<String, FnDecl>,
    traits: &BTreeMap<String, TraitInfo>,
    instances: &BTreeMap<(String, String), InstanceInfo>,
    impls: &BTreeMap<(String, String), Vec<FnDecl>>,
) -> Result<(), CheckError> {
    let mut probe_fns = fns.clone();
    probe_fns.insert(
        PROBE_FN.to_owned(),
        FnDecl {
            vis: Vis::Private,
            thaw: false,
            tier: None,
            sig: join.sig.clone(),
            body: join.body.clone(),
        },
    );
    let scratch = Env {
        types: types.clone(),
        fns: probe_fns,
        totality: BTreeMap::new(),
        traits: traits.clone(),
        instances: instances.clone(),
        impls: impls.clone(),
        lower_rules: BTreeMap::new(),
        // M-973 (DN-54 §10) added `derived_provenance` to `Env` after this leaf branched; the
        // law-probe scratch env has no derives, so an empty map is correct (mirrors mono.rs). M-966
        // added `via_provenance` similarly — the scratch env has no `via` delegation either.
        derived_provenance: BTreeMap::new(),
        via_provenance: BTreeMap::new(),
    };
    let evaluator = Evaluator::new(&scratch);
    let site = format!("impl Fuse[{for_ty}] for {for_ty}");
    let call = |a: &crate::eval::L1Value,
                b: &crate::eval::L1Value|
     -> Result<crate::eval::L1Value, CheckError> {
        evaluator
            .call(PROBE_FN, vec![a.clone(), b.clone()])
            .map_err(|e| {
                CheckError::new(
                    &site,
                    format!(
                        "could not evaluate `Fuse::join` for `{for_ty}` while checking the \
                         semilattice laws (DN-58 §A.4; RFC-0008 RT6): {e} — a `join` that cannot \
                         be evaluated over its own declared domain cannot be certified lawful \
                         (never-silent — G2)"
                    ),
                )
            })
    };

    // Idempotence: join(a, a) = a, for every a in the domain.
    for a in domain {
        let got = call(a, a)?;
        if &got != a {
            return Err(CheckError::new(
                &site,
                format!(
                    "`Fuse::join` for `{for_ty}` violates idempotence (RFC-0008 RT6 / DN-58 §A.1): \
                     join({a:?}, {a:?}) = {got:?}, expected {a:?} — a semilattice-law violation is \
                     refused at definition, never a silent accept (G2/M-965)"
                ),
            ));
        }
    }

    // Commutativity: join(a, b) = join(b, a), for every a, b in the domain.
    for a in domain {
        for b in domain {
            let ab = call(a, b)?;
            let ba = call(b, a)?;
            if ab != ba {
                return Err(CheckError::new(
                    &site,
                    format!(
                        "`Fuse::join` for `{for_ty}` violates commutativity (RFC-0008 RT6 / DN-58 \
                         §A.1): join({a:?}, {b:?}) = {ab:?} but join({b:?}, {a:?}) = {ba:?} — a \
                         semilattice-law violation is refused at definition, never a silent accept \
                         (G2/M-965)"
                    ),
                ));
            }
        }
    }

    // Associativity: join(join(a, b), c) = join(a, join(b, c)), for every a, b, c in the domain.
    for a in domain {
        for b in domain {
            for c in domain {
                let ab = call(a, b)?;
                let ab_c = call(&ab, c)?;
                let bc = call(b, c)?;
                let a_bc = call(a, &bc)?;
                if ab_c != a_bc {
                    return Err(CheckError::new(
                        &site,
                        format!(
                            "`Fuse::join` for `{for_ty}` violates associativity (RFC-0008 RT6 / \
                             DN-58 §A.1): join(join({a:?}, {b:?}), {c:?}) = {ab_c:?} but \
                             join({a:?}, join({b:?}, {c:?})) = {a_bc:?} — a semilattice-law \
                             violation is refused at definition, never a silent accept (G2/M-965)"
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}
