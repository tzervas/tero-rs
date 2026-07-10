//! **Maranget decision-tree compilation** for L1 `match` (M-320; RFC-0007 §3/§4.4; Maranget 2008,
//! *Compiling pattern matching to good decision trees*) — the **codegen half** of the Maranget
//! pipeline whose analysis half is [`crate::usefulness`].
//!
//! Where usefulness answers "is this match exhaustive / are any arms redundant?", this pass answers
//! "in what order do we test the scrutinee to reach the right arm?" — it lowers a (checked,
//! exhaustive) nested-pattern match into a [`Tree`] of `switch`/`leaf` nodes over **occurrences**
//! (paths into the scrutinee). This is exactly what RFC-0007 §3 means by patterns being "compiled
//! away by the elaborator": the surface keeps nested patterns; the tree is flat tests.
//!
//! **Scope / honesty (VR-5).** This builds and *verifies* the decision tree (the tests evaluate it
//! against the reference matcher), and — since RFC-0011 r3 enacted the flat L0 `Match` node
//! (RFC-0001 r3) — the elaborator **emits** it: [`crate::elab`]'s `lower_tree` walks each `Switch`
//! into a nested L0 `Match` and each `Leaf` into the surface arm body, the wiring this module's
//! `Tree` was designed for (RFC-0007 §4.6 / RFC-0011 §4.4). The tree stays the *untrusted,
//! inspectable* compilation artifact **above** the kernel: the trusted node is the flat `Match`, and
//! the three-way differential (`tests/differential.rs`) checks the emitted lowering — L1-eval ≡
//! L0-interp ≡ AOT — so a wrong column choice or specialization is caught, never rubber-stamped. The
//! tree's own `eval_tree` remains a *test-only* reference (it verifies the compiler; it does not
//! run programs). No accuracy guarantee is touched by the compilation — it is a meaning-preserving
//! rewrite, witnessed by the differential.
//!
//! The compiler operates on the same normalized [`Pat`] matrix usefulness uses; a *value* is just a
//! [`Pat`] with no [`Pat::Wild`] (a fully concrete constructor/literal tree), which is what the
//! verification tests feed both the tree and the reference matcher.

use std::collections::BTreeMap;

use mycelium_workstack::{BudgetError, RecursionBudget};

use crate::checkty::{DataInfo, Ty};
use crate::usefulness::{specialize_ctor, specialize_lit, Pat};

/// An **occurrence**: the path of field indices from the scrutinee root to a sub-value (`[]` is the
/// whole scrutinee, `[1]` is its second constructor field, `[1, 0]` the first field of that, …).
pub(crate) type Occurrence = Vec<usize>;

/// The head a [`Tree::Switch`] case tests for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Head {
    /// A constructor by name + arity.
    Ctor(String, usize),
    /// A `Binary`/`Ternary` literal, keyed as in [`crate::usefulness`] (`b:…` / `t:…`).
    Lit(String),
}

/// A compiled match **decision tree** (Maranget 2008). Leaves carry the **surface arm index** to run;
/// `Fail` is only reachable for a non-exhaustive match (the checker rejects those before compilation,
/// so a verified-exhaustive match never produces a reachable `Fail`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Tree {
    /// Run surface arm `usize` (the first matrix row that matched).
    Leaf(usize),
    /// No arm matches.
    Fail,
    /// Test the value at `occurrence` against each `(head, subtree)` case in turn; if none matches,
    /// fall through to `default` (present exactly when the column's signature is incomplete or its
    /// domain is open — `Binary`/`Ternary`).
    Switch {
        /// Which sub-value of the scrutinee this node tests.
        occurrence: Occurrence,
        /// The constructor/literal cases, in signature order (data) or first-seen order (literals).
        cases: Vec<(Head, Tree)>,
        /// The catch-all branch, when the cases do not cover the column's signature.
        default: Option<Box<Tree>>,
    },
}

/// One row of the working matrix: the per-column patterns and the surface arm it came from.
#[derive(Clone)]
struct Row {
    pats: Vec<Pat>,
    arm: usize,
}

/// A decision-tree row specializes exactly like a bare pattern vector, but must **carry its surface
/// arm index** through unchanged (that index is what a `Leaf` ultimately runs). Implementing
/// [`SpecializeRow`](crate::usefulness::SpecializeRow) lets the Maranget `S` specialization be the
/// single shared one in `crate::usefulness` (M-641) rather than a row-for-row duplicate here.
impl crate::usefulness::SpecializeRow for Row {
    fn columns(&self) -> &[Pat] {
        &self.pats
    }
    fn with_columns(&self, columns: Vec<Pat>) -> Self {
        Row {
            pats: columns,
            arm: self.arm,
        }
    }
}

/// Compile a checked match into a decision tree. `matrix` is the per-arm normalized pattern rows (one
/// `Pat` per column), `arms` the parallel surface arm indices, `occ` the occurrence of each column
/// (initially `[[]]` for the single scrutinee), and `tys` each column's type (drives the
/// complete-signature test + the field-type expansion). Assumes the match has already passed
/// exhaustiveness/redundancy (so the first all-wildcard row is a real catch-all).
///
/// **RFC-0041 §4.7 (W1, RR-29):** the tree-compilation recursion is charged against a
/// [`RecursionBudget`]; a wide-arity constructor that would drive [`compile_rows`] into an unbounded
/// host-stack overflow (SIGABRT) is instead refused never-silently with a [`BudgetError::DepthExceeded`]
/// at the default ceiling. The caller ([`crate::checkty::Cx::check_match`]) maps it into a `CheckError`.
///
/// **RFC-0041 §4.7 (W6): the wide-tuple asymmetry — DOCUMENTED, not converted (Empirical, VR-5).**
/// [`compile_rows`] tests one column per recursion level (specialize → recurse → assemble a `Switch`),
/// so — like its analysis twin [`crate::usefulness::useful`] — a constructor whose fields carry
/// non-wildcard heads drives ~N levels of recursion on the tuple/ctor **arity** spine (a wide ctor
/// with *all-wildcard* fields short-circuits to a `Leaf` and does not). This width is data-shaped, not
/// genuine control nesting; **measured boundary N ≥ 4095** false-refuses with
/// [`BudgetError::DepthExceeded`]. It is surface-reachable but pathological (a 4095-field product
/// type), already **safe and never-silent** (a clean refusal, not a SIGABRT, on the production deep
/// stack), and the W6 plan (§7) conditions the twin conversion on "residual frontend conversion **if
/// profiling demands**" — which it does not. See the fuller rationale on
/// [`crate::usefulness::useful`]: per §4.7's fork we **document** the asymmetry rather than force a
/// high-risk byte-identical iterative rewrite of this trusted branching Maranget compiler
/// (KISS/YAGNI/KC-3). Boundary test-witnessed (`tests::decision::w6_wide_arity_compile_refuses`).
/// **FLAG (W6 → orchestrator/maintainer):** the conversion seam is the per-column recursion — a work
/// stack charging `charge_steps` per column — if 4095-arity is deemed realistic enough to warrant it.
pub(crate) fn compile(
    types: &BTreeMap<String, DataInfo>,
    matrix: &[Vec<Pat>],
    arms: &[usize],
    occ: &[Occurrence],
    tys: &[Ty],
) -> Result<Tree, BudgetError> {
    let rows: Vec<Row> = matrix
        .iter()
        .zip(arms)
        .map(|(pats, &arm)| Row {
            pats: pats.clone(),
            arm,
        })
        .collect();
    // A fresh per-compilation budget: this walk is independent of the usefulness query's budget, so
    // its depth resets to zero here (guards release on return). Default depth ceiling (4096).
    let budget = RecursionBudget::default();
    compile_rows(&budget, types, &rows, occ, tys)
}

fn compile_rows(
    budget: &RecursionBudget,
    types: &BTreeMap<String, DataInfo>,
    rows: &[Row],
    occ: &[Occurrence],
    tys: &[Ty],
) -> Result<Tree, BudgetError> {
    // Charge one level of decision-tree recursion; refuse never-silently past the ceiling (§4.7).
    let _g = budget.try_enter()?;
    // No row can match → failure (unreachable for an exhaustive match).
    let Some(first) = rows.first() else {
        return Ok(Tree::Fail);
    };
    // The first row is all wildcards (or there are no columns) → it matches everything here: run it.
    if first.pats.iter().all(|p| matches!(p, Pat::Wild)) {
        return Ok(Tree::Leaf(first.arm));
    }
    // Pick the first column with a non-wildcard head in some row (Maranget's left-to-right heuristic),
    // and rotate it to the front so the specialization helpers can work on column 0.
    let col = (0..occ.len())
        .find(|&i| rows.iter().any(|r| !matches!(r.pats[i], Pat::Wild)))
        .expect("first row is non-wildcard, so some column has a constructor/literal head");
    let (rows, occ, tys) = rotate_to_front(rows, occ, tys, col);
    let occ0 = occ[0].clone();
    let ty0 = tys[0].clone();

    // Gather the heads present in column 0 (constructors with arity, or literal keys).
    let mut ctor_heads: Vec<(String, usize)> = Vec::new();
    let mut lit_heads: Vec<String> = Vec::new();
    for r in &rows {
        match &r.pats[0] {
            Pat::Ctor(n, subs) => {
                if !ctor_heads.iter().any(|(m, _)| m == n) {
                    ctor_heads.push((n.clone(), subs.len()));
                }
            }
            Pat::Lit(k) => {
                if !lit_heads.iter().any(|j| j == k) {
                    lit_heads.push(k.clone());
                }
            }
            Pat::Wild => {}
        }
    }

    let mut cases: Vec<(Head, Tree)> = Vec::new();
    // Whether the cases cover the column's whole signature (so no default is needed).
    let complete = match &ty0 {
        Ty::Data(n, _) => types.get(n).is_some_and(|d| {
            // Iterate constructors in signature order for a stable, complete switch.
            d.ctors
                .iter()
                .all(|ci| ctor_heads.iter().any(|(m, _)| *m == ci.name))
        }),
        // Binary/Ternary value domains are never enumerated — always need a default.
        _ => false,
    };

    if let Ty::Data(dn, _) = &ty0 {
        if let Some(d) = types.get(dn) {
            let d = d.clone();
            for ci in &d.ctors {
                if ctor_heads.iter().any(|(m, _)| *m == ci.name) {
                    let a = ci.fields.len();
                    let sub = compile_rows(
                        budget,
                        types,
                        &specialize_ctor(&rows, &ci.name, a),
                        &child_occ(&occ, &occ0, a),
                        &child_tys(&tys, &ci.fields),
                    )?;
                    cases.push((Head::Ctor(ci.name.clone(), a), sub));
                }
            }
        }
    }
    for k in &lit_heads {
        let sub = compile_rows(
            budget,
            types,
            &specialize_lit(&rows, k),
            &occ[1..],
            &tys[1..],
        )?;
        cases.push((Head::Lit(k.clone()), sub));
    }

    let default = if complete {
        None
    } else {
        Some(Box::new(compile_rows(
            budget,
            types,
            &default_rows(&rows),
            &occ[1..],
            &tys[1..],
        )?))
    };

    Ok(Tree::Switch {
        occurrence: occ0,
        cases,
        default,
    })
}

/// Swap column `i` to the front of the rows + the parallel occurrence/type vectors (an occurrence is
/// an intrinsic path, so reordering columns does not change leaf arms or any occurrence).
fn rotate_to_front(
    rows: &[Row],
    occ: &[Occurrence],
    tys: &[Ty],
    i: usize,
) -> (Vec<Row>, Vec<Occurrence>, Vec<Ty>) {
    let mut occ = occ.to_vec();
    let mut tys = tys.to_vec();
    occ.swap(0, i);
    tys.swap(0, i);
    let rows = rows
        .iter()
        .map(|r| {
            let mut pats = r.pats.clone();
            pats.swap(0, i);
            Row { pats, arm: r.arm }
        })
        .collect();
    (rows, occ, tys)
}

/// The occurrences of the columns after specializing column 0 on a constructor of arity `a`: the `a`
/// child occurrences `occ0.j` followed by the remaining columns.
fn child_occ(occ: &[Occurrence], occ0: &Occurrence, a: usize) -> Vec<Occurrence> {
    let mut out: Vec<Occurrence> = (0..a)
        .map(|j| {
            let mut o = occ0.clone();
            o.push(j);
            o
        })
        .collect();
    out.extend_from_slice(&occ[1..]);
    out
}

/// The column types after specializing column 0 on a constructor: its field types then the rest.
fn child_tys(tys: &[Ty], fields: &[Ty]) -> Vec<Ty> {
    let mut out = fields.to_vec();
    out.extend_from_slice(&tys[1..]);
    out
}

// Maranget's `S` specialization (constructor and literal heads) is shared with the usefulness
// analysis via `crate::usefulness::{specialize_ctor, specialize_lit}` over the `SpecializeRow`
// trait `Row` implements above (M-641) — the decision-tree-specific part is only carrying the arm
// index through, which the trait's `with_columns` does. `default_rows` (the `D(P)` matrix) stays
// local: it is a distinct operation, not part of that shared specialization.

/// The default rows `D(P)`: rows headed by a wildcard, leading column dropped.
fn default_rows(rows: &[Row]) -> Vec<Row> {
    rows.iter()
        .filter_map(|r| {
            let (first, rest) = r.pats.split_first().expect("non-empty row");
            matches!(first, Pat::Wild).then(|| Row {
                pats: rest.to_vec(),
                arm: r.arm,
            })
        })
        .collect()
}

/// Whether the tree contains a reachable [`Tree::Fail`]. Every branch a compiled tree emits is
/// reachable (each case head can occur; a `default` is present only when needed), so "contains a
/// `Fail`" is "has a reachable `Fail`". The checker uses this to confirm an **exhaustive** match
/// compiled to total coverage (defense in depth: usefulness and the Maranget compiler must agree).
pub(crate) fn has_reachable_fail(tree: &Tree) -> bool {
    match tree {
        Tree::Fail => true,
        Tree::Leaf(_) => false,
        Tree::Switch { cases, default, .. } => {
            cases.iter().any(|(_, t)| has_reachable_fail(t))
                || default.as_deref().is_some_and(has_reachable_fail)
        }
    }
}
