//! **Maranget usefulness** for L1 `match` — the checked exhaustiveness/redundancy core that makes
//! W7 hold for **nested** patterns (RFC-0007 §4.4/§4.7; Maranget 2007, *Warnings for pattern
//! matching*). It is the analysis half of the Maranget pipeline; the *decision-tree compilation* to
//! the flat kernel `Match` (Maranget 2008) is the elaborator's job and lands with full L1-in-Core-IR.
//!
//! The algorithm is the standard `U(P, q)`: given a pattern **matrix** `P` (one row per prior arm,
//! one column per scrutinee position) and a pattern **vector** `q`, decide whether some value matches
//! `q` but no row of `P` — and, when so, return a **witness** value (as a pattern) demonstrating it.
//! Two derived checks drive the typechecker:
//! - **Exhaustiveness:** the match covers everything iff `U(P, [_])` is *not* useful — a witness is a
//!   concrete missing pattern (e.g. `Cons(_, Nil)`), reported verbatim (coverage is *checked*, never
//!   assumed — W7).
//! - **Redundancy:** arm `i` is reachable iff `U(P₀..ᵢ, rowᵢ)` *is* useful; an unreachable arm is a
//!   redundancy error (subsumes the M-320 duplicate-literal check).
//!
//! The column **type** decides the constructor signature: a data type's signature is its finite
//! constructor set (so a column is *complete* once every constructor appears), while `Binary{n}` /
//! `Ternary{m}` have a value domain that is **never** enumerated — their signature is open, so a
//! literal column is complete only via a `_`/binder default (matching the M-320 rule). Recursion in
//! the data registry is handled lazily: a constructor's field column types are looked up only when
//! that constructor is actually expanded, so a recursive type (`Nat = Z | S(Nat)`) terminates.

use std::collections::{BTreeMap, BTreeSet};

use mycelium_workstack::{BudgetError, RecursionBudget};

use crate::checkty::{DataInfo, Ty};

/// A normalized pattern for the usefulness matrix. The typechecker lowers `ast::Pattern` to this:
/// binders and `_` both become [`Pat::Wild`] (they do not refine coverage), a nullary constructor or
/// constructor application becomes [`Pat::Ctor`], and a `Binary`/`Ternary` literal becomes
/// [`Pat::Lit`] keyed by its canonical form (arity 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Pat {
    /// `_` or a binder — matches anything.
    Wild,
    /// A constructor pattern `Name(sub, …)` (nullary ⇒ empty subs).
    Ctor(String, Vec<Pat>),
    /// A `Binary`/`Ternary` literal, keyed canonically (see `checkty::literal_key`).
    Lit(String),
}

/// The finite constructor signature of `ty`, or `None` if its value domain is open (`Binary`/
/// `Ternary` — never a complete signature, so a literal column always needs a default).
fn signature<'a>(ty: &Ty, types: &'a BTreeMap<String, DataInfo>) -> Option<&'a DataInfo> {
    match ty {
        Ty::Data(n, _) => types.get(n),
        _ => None,
    }
}

/// The field types of constructor `c` in data type `ty` (empty if not found — the caller has already
/// type-checked the pattern, so a miss cannot happen on a well-typed matrix).
fn ctor_fields(ty: &Ty, c: &str, types: &BTreeMap<String, DataInfo>) -> Vec<Ty> {
    signature(ty, types)
        .and_then(|d| d.ctors.iter().find(|ci| ci.name == c))
        .map(|ci| ci.fields.clone())
        .unwrap_or_default()
}

/// A **matrix row** that can be specialized (Maranget `S`/default): it exposes its pattern columns
/// and can rebuild itself with a new column vector, carrying any *non-pattern* payload through
/// unchanged. Implemented by the bare `Vec<Pat>` row the usefulness analysis uses and by the
/// arm-tagged `Row` the decision-tree compiler uses (`crate::decision`), so the specialization is
/// written **once** over both (M-641). `with_columns` is the only place a row's payload is
/// preserved, keeping every implementor's identity (e.g. the surface arm index) intact.
pub(crate) trait SpecializeRow {
    /// This row's pattern columns (always non-empty when specialized).
    fn columns(&self) -> &[Pat];
    /// Rebuild a row of the same kind with `columns` as its new column vector, preserving payload.
    fn with_columns(&self, columns: Vec<Pat>) -> Self;
}

impl SpecializeRow for Vec<Pat> {
    fn columns(&self) -> &[Pat] {
        self
    }
    fn with_columns(&self, columns: Vec<Pat>) -> Self {
        columns
    }
}

/// Specialize the matrix on a constructor head `c` of arity `a`: keep rows whose first pattern is `c`
/// (expanding its sub-patterns into the new leading columns) or a wildcard (expanding to `a`
/// wildcards), dropping rows headed by a different constructor. Generic over the row type so the
/// usefulness matrix (`Vec<Pat>`) and the decision-tree matrix (`Row`) share one implementation.
pub(crate) fn specialize_ctor<R: SpecializeRow>(matrix: &[R], c: &str, a: usize) -> Vec<R> {
    let mut out = Vec::new();
    for row in matrix {
        let (first, rest) = row.columns().split_first().expect("non-empty row");
        match first {
            Pat::Ctor(n, subs) if n == c => {
                let mut r = subs.clone();
                r.extend_from_slice(rest);
                out.push(row.with_columns(r));
            }
            Pat::Wild => {
                let mut r = vec![Pat::Wild; a];
                r.extend_from_slice(rest);
                out.push(row.with_columns(r));
            }
            _ => {} // different constructor / a literal head: drop
        }
    }
    out
}

/// Specialize the matrix on a literal head `k` (arity 0): keep rows headed by that exact literal or a
/// wildcard, dropping the leading column. Generic over the row type (see [`specialize_ctor`]).
pub(crate) fn specialize_lit<R: SpecializeRow>(matrix: &[R], k: &str) -> Vec<R> {
    let mut out = Vec::new();
    for row in matrix {
        let (first, rest) = row.columns().split_first().expect("non-empty row");
        match first {
            Pat::Lit(j) if j == k => out.push(row.with_columns(rest.to_vec())),
            Pat::Wild => out.push(row.with_columns(rest.to_vec())),
            _ => {}
        }
    }
    out
}

/// The default matrix `D(P)`: rows headed by a wildcard, with the leading column dropped.
fn default_matrix(matrix: &[Vec<Pat>]) -> Vec<Vec<Pat>> {
    matrix
        .iter()
        .filter_map(|row| {
            let (first, rest) = row.split_first().expect("non-empty row");
            matches!(first, Pat::Wild).then(|| rest.to_vec())
        })
        .collect()
}

/// The set of constructor names appearing in the matrix's first column.
fn head_ctors(matrix: &[Vec<Pat>]) -> BTreeSet<String> {
    matrix
        .iter()
        .filter_map(|row| match &row[0] {
            Pat::Ctor(n, _) => Some(n.clone()),
            _ => None,
        })
        .collect()
}

/// `U(P, q)` — is `q` useful w.r.t. matrix `P` (some value matches `q` but no row of `P`)? Returns a
/// witness value (as a pattern vector of the same width) when useful, else `None`. `col_types` gives
/// the type of each column (parallel to `q`); it drives the complete-signature test and the lazy
/// field-type expansion.
///
/// **RFC-0041 §4.7 (W1, RR-29):** the Maranget recursion is charged against a per-query
/// [`RecursionBudget`] — a wide-arity constructor / deeply-nested pattern that would otherwise drive
/// this walk into an unbounded host-stack overflow (SIGABRT) is now refused **never-silently** with a
/// [`BudgetError::DepthExceeded`] at the [`RecursionBudget::DEFAULT_DEPTH_LIMIT`] ceiling. The caller
/// ([`crate::checkty::Cx::check_match`]) maps that into its [`crate::checkty::CheckError`] surface.
///
/// **RFC-0041 §4.7 (W6): the wide-tuple asymmetry — DOCUMENTED, not converted (Empirical, VR-5).**
/// This walk consumes the query's columns left-to-right, holding one live [`DepthGuard`] per column so
/// the reassembled witness (`rebuild_ctor`/`prepend` on return) has the query's original width. An
/// N-field tuple/ctor therefore recurses ~N deep on its **arity** spine — data-shaped width, **not**
/// genuine control nesting (which the parser's `MAX_EXPR_DEPTH`/checker's `MAX_CHECK_DEPTH` already
/// bound to 4096). Tuple/ctor arity is *not* charged against those nesting caps (grammar siblings
/// enter/leave the parser's depth budget; `comma_separated` imposes no field-count cap), so a
/// wide-arity pattern is **surface-reachable end-to-end**: a real `object W(f0, …, f_{N-1})` matched
/// by `W(x0, …, x_{N-1})` reaches this walk with width N. **Measured boundary:** at **N ≥ 4095** the
/// arity spine exhausts the depth budget and this returns [`BudgetError::DepthExceeded`] — a *false*
/// refusal (the pattern is shallow, only wide). W1's structural twin `check_list` **converted** the
/// analogous large-**list-literal** spine to a flat work-step loop (`checkty::check_list`), because a
/// large list literal is a *mundane, realistic* input (lookup tables, embedded data). A **4095-field
/// product type** is by contrast *pathological* — absent from any realistic corpus — and the W6 plan
/// (§7) conditions this twin's conversion on "**residual frontend conversion if profiling demands**,"
/// which it does not. Crucially the current behavior is **already safe and never-silent** (a clean
/// `DepthExceeded`, verified not a SIGABRT on the production 256 MiB deep stack), so the DoD's "no
/// input SIGABRTs any pass" bar holds; the residual is a *precision* defect (a shallow-but-wide
/// pattern refused as if deep), not a safety one. Per §4.7's explicit fork we therefore **document the
/// wide-tuple asymmetry** rather than force a high-risk byte-identical iterative rewrite of this
/// trusted branching Maranget walk (KISS/YAGNI/KC-3). The boundary is test-witnessed
/// (`tests::usefulness::w6_wide_arity_*`); a future conversion would flip the refusing case to accept.
/// **FLAG (W6 → orchestrator/maintainer):** if 4095-arity is deemed "realistic enough" to warrant the
/// conversion, this is the seam — convert the width spine to iteration charging `charge_steps`.
pub(crate) fn useful(
    types: &BTreeMap<String, DataInfo>,
    matrix: &[Vec<Pat>],
    q: &[Pat],
    col_types: &[Ty],
) -> Result<Option<Vec<Pat>>, BudgetError> {
    // A fresh per-query budget: each top-level `U(P, q)` is an independent walk, so its depth resets
    // to zero (the guards release on return). The default depth ceiling (4096) is the §4.0 metric.
    let budget = RecursionBudget::default();
    useful_budgeted(&budget, types, matrix, q, col_types)
}

/// The budget-charged Maranget `U(P, q)` recursion (RFC-0041 §4.7). Charges one depth level per
/// recursion point via [`RecursionBudget::try_enter`]; the [`mycelium_workstack::DepthGuard`] releases
/// it on every exit path.
fn useful_budgeted(
    budget: &RecursionBudget,
    types: &BTreeMap<String, DataInfo>,
    matrix: &[Vec<Pat>],
    q: &[Pat],
    col_types: &[Ty],
) -> Result<Option<Vec<Pat>>, BudgetError> {
    // Charge one level of Maranget recursion; refuse never-silently past the ceiling (§4.7).
    let _g = budget.try_enter()?;
    // Base case (no columns): useful iff no row remains (every prior row already "matched"); the
    // witness is the empty value vector.
    if q.is_empty() {
        return Ok(matrix.is_empty().then(Vec::new));
    }
    let head_ty = &col_types[0];
    match &q[0] {
        Pat::Ctor(c, subs) => {
            let a = subs.len();
            let m2 = specialize_ctor(matrix, c, a);
            let mut q2 = subs.clone();
            q2.extend_from_slice(&q[1..]);
            let mut ct2 = ctor_fields(head_ty, c, types);
            ct2.extend_from_slice(&col_types[1..]);
            Ok(useful_budgeted(budget, types, &m2, &q2, &ct2)?.map(|w| rebuild_ctor(c, a, w)))
        }
        Pat::Lit(k) => {
            let m2 = specialize_lit(matrix, k);
            let q2 = q[1..].to_vec();
            let ct2 = col_types[1..].to_vec();
            Ok(useful_budgeted(budget, types, &m2, &q2, &ct2)?
                .map(|w| prepend(Pat::Lit(k.clone()), w)))
        }
        Pat::Wild => match signature(head_ty, types) {
            // Finite (data) signature: complete once every constructor appears in column 0.
            Some(d) => {
                let d = d.clone();
                let present = head_ctors(matrix);
                if d.ctors.iter().all(|ci| present.contains(&ci.name)) {
                    // Complete: useful iff useful under *some* constructor specialization.
                    for ci in &d.ctors {
                        let a = ci.fields.len();
                        let m2 = specialize_ctor(matrix, &ci.name, a);
                        let mut q2 = vec![Pat::Wild; a];
                        q2.extend_from_slice(&q[1..]);
                        let mut ct2 = ci.fields.clone();
                        ct2.extend_from_slice(&col_types[1..]);
                        if let Some(w) = useful_budgeted(budget, types, &m2, &q2, &ct2)? {
                            return Ok(Some(rebuild_ctor(&ci.name, a, w)));
                        }
                    }
                    Ok(None)
                } else {
                    // Incomplete: recurse on the default; the witness head is a *missing* constructor.
                    let m2 = default_matrix(matrix);
                    Ok(
                        useful_budgeted(budget, types, &m2, &q[1..], &col_types[1..])?.map(|w| {
                            let missing = d.ctors.iter().find(|ci| !present.contains(&ci.name));
                            let head = missing.map_or(Pat::Wild, |ci| {
                                Pat::Ctor(ci.name.clone(), vec![Pat::Wild; ci.fields.len()])
                            });
                            prepend(head, w)
                        }),
                    )
                }
            }
            // Open (Binary/Ternary) domain: never complete — recurse on the default, witness `_`.
            None => {
                let m2 = default_matrix(matrix);
                Ok(
                    useful_budgeted(budget, types, &m2, &q[1..], &col_types[1..])?
                        .map(|w| prepend(Pat::Wild, w)),
                )
            }
        },
    }
}

/// Re-fold a witness whose first `a` elements are constructor `c`'s sub-witnesses.
fn rebuild_ctor(c: &str, a: usize, mut w: Vec<Pat>) -> Vec<Pat> {
    let rest = w.split_off(a);
    let head = Pat::Ctor(c.to_owned(), w);
    prepend(head, rest)
}

fn prepend(head: Pat, rest: Vec<Pat>) -> Vec<Pat> {
    let mut v = Vec::with_capacity(rest.len() + 1);
    v.push(head);
    v.extend(rest);
    v
}

/// Render a witness pattern for a diagnostic (`Cons(_, Nil)`, `0b1010`, `<+0->`, `_`). Literal keys
/// carry a `b:`/`t:` tag (from `checkty::literal_key`) that is rewritten back to surface syntax.
pub(crate) fn render(p: &Pat) -> String {
    match p {
        Pat::Wild => "_".to_owned(),
        Pat::Lit(k) => match k.split_once(':') {
            Some(("b", bits)) => format!("0b{bits}"),
            Some(("t", trits)) => format!("<{trits}>"),
            _ => k.clone(),
        },
        Pat::Ctor(n, subs) if subs.is_empty() => n.clone(),
        Pat::Ctor(n, subs) => {
            let inner: Vec<String> = subs.iter().map(render).collect();
            format!("{n}({})", inner.join(", "))
        }
    }
}
