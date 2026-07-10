//! `std.iter` — iterator / fold / transducer combinators over the kernel `for` fold (M-526).
//!
//! The ergonomic combinator layer — `map`, `filter`, `fold`, `scan`, `zip`, `take`, `count`,
//! `any_with_witness`/`all_with_witness`, `find`, `position`, `chain`, `skip`, `step_by`,
//! `enumerate`, `flat_map`, `reduce`, `transduce`, and an explicitly-named lazy surface
//! (`lazy_unfold` via [`Lazy`], `lazy_take`) — expressed over the one iteration primitive the
//! kernel has: the RFC-0007 §4.8 `for` fold, a bounded walk of a linearly-recursive value that
//! the §4.5 totality checker classifies **`Total` with zero extension** (iteration is bounded
//! *by construction* because a value is finite and acyclic, §4.7).
//!
//! # Honesty crux: totality preservation
//!
//! Every eager combinator in this module is **total and terminating** because it lowers to (or
//! composes) a single RFC-0007 §4.8 `for` fold over a finite source. Termination is
//! **inherited** from the kernel's `Total`-by-construction fold — it is not re-proved here
//! (KC-3: this module adds no trusted code). The guarantee matrix ([`guarantee_matrix::MATRIX`])
//! records this per combinator; the sole exception is [`Lazy::unfold`], which is honestly tagged
//! [`GuaranteeStrength::Declared`] and typed as a distinct [`Lazy<E>`] (no silent thunk).
//!
//! # Short-circuit combinators (`any`/`all`/`find`) — FLAG Q3 / RFC-0007 §4.8
//!
//! RFC-0007 §4.8 excludes `break`/`loop`/`while`; early exit is described there as "a later,
//! explicit form (fold-to-`Option`)". Accordingly, [`any_with_witness`], [`all_with_witness`],
//! and [`find`] in this module are implemented as **done-flag folds** — the accumulator carries
//! a done-flag and the step function stops updating once the condition fires, but the fold
//! **still walks the full spine**. This is **total** (termination is guaranteed) but **not
//! short-circuiting in the cost sense** (extra work is done past the first match).
//!
//! **FLAG (RFC-0007 §4.8 / Q3 in spec §7):** whether a true early-termination fold primitive
//! (a `for` that may stop before the spine end, while remaining `Total`) is a worthwhile kernel
//! extension is an **open question** for RFC-0007, NOT decided here.
//!
//! # `Foldable<E>` — monomorphic placeholder (FLAG Q2 / RFC-0007 r4 deferred traits)
//!
//! The spec (§3) writes combinators as ranging over `Foldable<E>` — a linear-recursive value of
//! element type `E` (RFC-0007 §4.8 shape: `nil | cons(E, T)`). Whether `Foldable` is a *trait*
//! or per-type monomorphic depends on the trait/typeclass story, which RFC-0007 r4 **defers**
//! (traits/LR-2 are NOT ratified).
//!
//! **FLAG (RFC-0007 r4 / Q2 in spec §7):** this implementation uses a newtype
//! [`Foldable<E>`] backed by `Vec<E>` as a concrete stand-in for the spine shape. When the
//! RFC-0007 trait story lands, `Foldable` must be generalised (or replaced by a trait bound).
//! The combinator implementations must not change; only the abstraction boundary does.
//!
//! # C1 — never-silent
//!
//! - Partial-in-result reductions (`reduce`, `find`, `position`) return `Option` (never a
//!   sentinel).
//! - `step_by(0)` returns `Err(`[`ZeroStep`]`)` — never a silent step-of-1.
//! - `zip` truncates to the shorter spine; the truncation point is *reportable* via
//!   [`ZipOutcome`] (C3 EXPLAIN; Q1 in spec §7).
//! - `zip_exact` returns `Err(`[`error::ZipLengthMismatch`]`)` on length mismatch.
//! - The lazy source is explicitly typed as [`Lazy<E>`] and the `Declared` tag is embedded in
//!   the guarantee matrix.
//!
//! # C5 — above the small kernel (KC-3)
//!
//! No `unsafe`, no FFI (ADR-014). All combinators are pure and consume `mycelium-core` types
//! only for the guarantee lattice. The `Foldable` spine is a thin `Vec<E>` stand-in (Q2).
//!
//! # Design spec
//! `docs/spec/stdlib/iter.md` (M-526, #167); contract: RFC-0016 §4.1 (C1–C6).
//!
//! # Open questions (FLAGs carried from spec §7)
//!
//! - **(Q1)** `zip` length-mismatch policy — truncate + `ZipOutcome` + `zip_exact` as floor.
//! - **(Q2)** `Foldable` trait vs monomorphic — FLAG for RFC-0007 trait story.
//! - **(Q3)** Short-circuit under a `Total` fold — done-flag fold; FLAG to RFC-0007 §4.8.
//! - **(Q4)** `Lazy` surface in `iter` vs own module — kept here, type-segregated.
//! - **(Q5)** Transducer fusion law — `Empirical` property test; NOT `Proven` (VR-5).
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Iterators are representation-neutral combinators —
//! element representation passes through unchanged. A `map` over a `Seq` of `Binary{32}` values
//! produces `Binary{32}` values (or whatever the mapping function returns); no representation
//! is coerced by the combinator itself. Cross-representation element transformation requires an
//! explicit swap in the mapping function.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/iter.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; the same-named `lib/std/iter.myc` prototype is a narrower, structurally distinct surface (DN-66 S3.1) — the D6 retirement trigger has not fired, so no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

pub mod error;
pub mod foldable;
pub mod guarantee_matrix;
pub mod lazy;
pub mod transducer;
pub mod zip_outcome;

pub use error::ZeroStep;
pub use foldable::Foldable;
pub use lazy::Lazy;
pub use mycelium_core::GuaranteeStrength;
pub use transducer::Transducer;
pub use zip_outcome::ZipOutcome;

// ─── Transforms: Foldable in, Foldable out ────────────────────────────────────

/// Apply `f` to every element, producing a new `Foldable<F>`.
///
/// # Guarantee tag: `Exact` (inherited — one `for` fold over a finite source; RFC-0007 §4.8)
/// Totality is inherited from the kernel's `Total`-by-construction fold.
/// The closure's guarantee is the closure's tag; `map` itself is exact.
///
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn map<E, F>(source: Foldable<E>, f: impl FnMut(E) -> F) -> Foldable<F> {
    Foldable::from_vec(source.into_vec().into_iter().map(f).collect())
}

/// Keep only elements for which `pred` returns `true`.
///
/// # Guarantee tag: `Exact` (inherited — one `for` fold)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn filter<E>(source: Foldable<E>, pred: impl FnMut(&E) -> bool) -> Foldable<E> {
    Foldable::from_vec(source.into_vec().into_iter().filter(pred).collect())
}

/// Running accumulator fold — length-preserving (one output element per input element).
///
/// `scan(source, init, f)` produces `[f(init, e₀), f(f(init, e₀), e₁), …]`.
///
/// # Guarantee tag: `Exact` (inherited — one `for` fold; length-preserving by construction)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn scan<E: Clone, A: Clone>(
    source: Foldable<E>,
    init: A,
    mut f: impl FnMut(A, E) -> A,
) -> Foldable<A> {
    let mut acc = init;
    Foldable::from_vec(
        source
            .into_vec()
            .into_iter()
            .map(|e| {
                acc = f(acc.clone(), e);
                acc.clone()
            })
            .collect(),
    )
}

/// Pair each element with its zero-based index.
///
/// # Guarantee tag: `Exact` (inherited — one `for` fold)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn enumerate<E>(source: Foldable<E>) -> Foldable<(usize, E)> {
    Foldable::from_vec(source.into_vec().into_iter().enumerate().collect())
}

/// Map each element to a `Foldable<F>` and flatten — finite-of-finite is finite (§4.7).
///
/// # Guarantee tag: `Exact` (inherited — outer fold + inner spine walks both bounded)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn flat_map<E, F>(source: Foldable<E>, mut f: impl FnMut(E) -> Foldable<F>) -> Foldable<F> {
    Foldable::from_vec(
        source
            .into_vec()
            .into_iter()
            .flat_map(|e| f(e).into_vec())
            .collect(),
    )
}

// ─── Reductions: Foldable in, value out ───────────────────────────────────────

/// The §4.8 `for` fold, surfaced directly.
///
/// Walks the full spine of `source`, applying `f` left-to-right with accumulator `init`.
/// This IS the RFC-0007 §4.8 `for` fold primitive.
///
/// # Guarantee tag: `Exact` (the primitive — total by kernel construction; RFC-0007 §4.8)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn fold<E, A>(source: Foldable<E>, init: A, f: impl FnMut(A, E) -> A) -> A {
    source.into_vec().into_iter().fold(init, f)
}

/// Reduce a non-empty `Foldable` with a combining function, returning `None` on empty input.
///
/// # Guarantee tag: `Exact` (inherited — total, but partial-in-result)
/// # Fallibility: `None` on empty input (C1 — never a sentinel)
/// # Effects: none
#[must_use]
pub fn reduce<E>(source: Foldable<E>, f: impl FnMut(E, E) -> E) -> Option<E> {
    let mut iter = source.into_vec().into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, f))
}

/// Count the number of elements in `source`.
///
/// # Guarantee tag: `Exact` (inherited — one `for` fold)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn count<E>(source: Foldable<E>) -> usize {
    source.into_vec().len()
}

/// Return `true` if any element satisfies `pred`, together with an [`AnyAllWitness`].
///
/// # Guarantee tag: `Exact` (inherited — done-flag `for` fold; total, full-spine walk)
///
/// # Short-circuit note (FLAG — RFC-0007 §4.8 / Q3)
/// Implemented as a **done-flag fold**: once `pred` fires the done-flag is set but the fold
/// continues to the end of the spine. **Total** (guaranteed termination) but NOT
/// short-circuiting in the cost sense. Whether a true early-termination primitive is worthwhile
/// is OPEN for RFC-0007 §4.8 — see FLAG Q3. Do not resolve here.
///
/// # EXPLAIN: the done-flag decision is reified
/// The returned [`AnyAllWitness`] carries the index of the first matching element so the caller
/// can inspect *where* the done-flag fired (C3).
///
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn any_with_witness<E>(
    source: Foldable<E>,
    pred: impl Fn(&E) -> bool,
) -> (bool, AnyAllWitness) {
    let mut first_match: Option<usize> = None;
    let mut idx: usize = 0;
    let result = fold(source, false, |done, e| {
        let fired = !done && pred(&e);
        if fired {
            first_match = Some(idx);
        }
        idx = idx.saturating_add(1);
        done || fired
    });
    (result, AnyAllWitness { first_match })
}

/// Return `true` if all elements satisfy `pred`, together with an [`AnyAllWitness`].
///
/// # Guarantee tag: `Exact` (inherited — done-flag `for` fold; total, full-spine walk)
///
/// # Short-circuit note (FLAG — RFC-0007 §4.8 / Q3)
/// Done-flag fold: once an element fails `pred` the done-flag is set but the fold continues.
/// Whether a true early-termination primitive is worthwhile is OPEN for RFC-0007 §4.8 — FLAG Q3.
///
/// # EXPLAIN: the done-flag decision is reified
/// The returned [`AnyAllWitness`] carries the index of the first element that *failed* `pred`.
///
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn all_with_witness<E>(
    source: Foldable<E>,
    pred: impl Fn(&E) -> bool,
) -> (bool, AnyAllWitness) {
    let mut first_fail: Option<usize> = None;
    let mut idx: usize = 0;
    let result = fold(source, true, |still_all, e| {
        let failed = still_all && !pred(&e);
        if failed {
            first_fail = Some(idx);
        }
        idx = idx.saturating_add(1);
        still_all && !failed
    });
    (
        result,
        AnyAllWitness {
            first_match: first_fail,
        },
    )
}

/// Return the first element satisfying `pred`, or `None` if no element matches.
///
/// # Guarantee tag: `Exact` (inherited — done-flag `for` fold; total, full-spine walk)
///
/// # Short-circuit note (FLAG — RFC-0007 §4.8 / Q3)
/// Done-flag fold: the accumulator is `Option<E>`; once a match is found it is preserved
/// unchanged for the remainder of the fold. The fold walks the full spine. FLAG Q3.
///
/// # Fallibility: `None` when no match (C1 — never a sentinel)
/// # Effects: none
#[must_use]
pub fn find<E: Clone>(source: Foldable<E>, pred: impl Fn(&E) -> bool) -> Option<E> {
    fold(source, None, |found, e| {
        if found.is_none() && pred(&e) {
            Some(e)
        } else {
            found
        }
    })
}

/// Return the zero-based index of the first element satisfying `pred`, or `None` if none.
///
/// # Guarantee tag: `Exact` (inherited — done-flag `for` fold; total)
/// # Fallibility: `None` when no match (C1)
/// # Effects: none
#[must_use]
pub fn position<E>(source: Foldable<E>, pred: impl Fn(&E) -> bool) -> Option<usize> {
    fold(source, (None::<usize>, 0usize), |(found, idx), e| {
        // `saturating_add` matches the `any_with_witness`/`all_with_witness` counters: a
        // `usize::MAX`-length spine must not panic on overflow (release has overflow-checks=true).
        if found.is_none() && pred(&e) {
            (Some(idx), idx.saturating_add(1))
        } else {
            (found, idx.saturating_add(1))
        }
    })
    .0
}

// ─── Pair / merge combinators ─────────────────────────────────────────────────

/// Pair elements from two `Foldable`s, truncating to the shorter spine.
///
/// Returns the paired `Foldable<(E, F)>` together with a [`ZipOutcome`] that records the
/// truncation point (C3 EXPLAIN — the truncation is never silent, C1).
///
/// # Guarantee tag: `Exact` (inherited — total; explicit length policy)
/// # Fallibility: total (truncation is not an error — it is reportable via `ZipOutcome`)
/// # Effects: none
/// # EXPLAIN: [`ZipOutcome`] records the lengths and which side was truncated (Q1)
#[must_use]
pub fn zip<E, F>(left: Foldable<E>, right: Foldable<F>) -> (Foldable<(E, F)>, ZipOutcome) {
    let left_len = left.len();
    let right_len = right.len();
    let paired: Vec<(E, F)> = left.into_vec().into_iter().zip(right.into_vec()).collect();
    let result_len = paired.len();
    let outcome = ZipOutcome::new(left_len, right_len, result_len);
    (Foldable::from_vec(paired), outcome)
}

/// Pair elements from two `Foldable`s; return `Err(ZipLengthMismatch)` if lengths differ.
///
/// The fallible, exact-length variant (spec §7-Q1 proposal). On success the lengths match
/// exactly and no truncation occurs; on failure the error carries both lengths (C1/C3).
///
/// # Guarantee tag: `Exact`
/// # Fallibility: `Err(ZipLengthMismatch)` when lengths differ (C1 — never truncates silently)
/// # Effects: none
pub fn zip_exact<E, F>(
    left: Foldable<E>,
    right: Foldable<F>,
) -> Result<Foldable<(E, F)>, error::ZipLengthMismatch> {
    let left_len = left.len();
    let right_len = right.len();
    if left_len != right_len {
        return Err(error::ZipLengthMismatch {
            left_len,
            right_len,
        });
    }
    let paired: Vec<(E, F)> = left.into_vec().into_iter().zip(right.into_vec()).collect();
    Ok(Foldable::from_vec(paired))
}

/// Append all elements of `right` after `left` — two finite spines remain finite.
///
/// # Guarantee tag: `Exact` (inherited — one conceptual `for` fold over the concatenated spine)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn chain<E>(left: Foldable<E>, right: Foldable<E>) -> Foldable<E> {
    let mut v = left.into_vec();
    v.extend(right.into_vec());
    Foldable::from_vec(v)
}

// ─── Bounded slicing ──────────────────────────────────────────────────────────

/// Keep at most the first `n` elements. If `n ≥ len(source)`, returns the entire source.
///
/// # Guarantee tag: `Exact` (inherited — the bound is a `Nat` value, not a promise)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn take<E>(source: Foldable<E>, n: usize) -> Foldable<E> {
    Foldable::from_vec(source.into_vec().into_iter().take(n).collect())
}

/// Drop the first `n` elements, returning the remainder. If `n ≥ len(source)`, returns empty.
///
/// # Guarantee tag: `Exact` (inherited)
/// # Fallibility: total
/// # Effects: none
#[must_use]
pub fn skip<E>(source: Foldable<E>, n: usize) -> Foldable<E> {
    Foldable::from_vec(source.into_vec().into_iter().skip(n).collect())
}

/// Keep every `k`-th element (0-indexed). Returns `Err(ZeroStep)` when `k = 0`.
///
/// `step_by(source, 1)` is the identity; `step_by(source, 2)` keeps indices 0, 2, 4, …
///
/// # Guarantee tag: `Exact` (inherited)
/// # Fallibility: `Err(ZeroStep)` when `k = 0` (C1 — no silent step-of-1)
/// # Effects: none
pub fn step_by<E>(source: Foldable<E>, k: usize) -> Result<Foldable<E>, ZeroStep> {
    if k == 0 {
        return Err(ZeroStep);
    }
    Ok(Foldable::from_vec(
        source.into_vec().into_iter().step_by(k).collect(),
    ))
}

// ─── Transducer surface ───────────────────────────────────────────────────────

/// Apply a [`Transducer<E, F>`] to `source`, reducing into `init` with `f`.
///
/// A transducer is a source-independent step transformer that **fuses into a single `for` fold**
/// — one pass over the spine. The pipeline is inspectable via [`Transducer::describe`] (C3).
///
/// # Guarantee tag: `Exact` (inherited — fuses to one `for` fold)
/// # Fallibility: total
/// # Effects: none
/// # EXPLAIN (C3): [`Transducer::describe`] exposes the fused step pipeline
///
/// # Property obligation (spec §7-Q5 / VR-5)
/// The fusion law is an `Empirical` property checked in tests — NOT tagged `Proven` (VR-5).
#[must_use]
pub fn transduce<E: 'static, F: 'static, A>(
    source: Foldable<E>,
    xf: &Transducer<E, F>,
    init: A,
    f: impl FnMut(A, F) -> A,
) -> A {
    let transformed: Vec<F> = xf.apply(source.into_vec());
    transformed.into_iter().fold(init, f)
}

// ─── Lazy surface ─────────────────────────────────────────────────────────────

/// Convert a [`Lazy<E>`] back into a bounded, total [`Foldable<E>`] by applying a `Nat` bound.
///
/// This is the **only** sanctioned way to turn an unbounded `Lazy` source into a total, foldable
/// value — the unbounded→bounded transition is an explicit call, not an implicit cutoff.
///
/// # Guarantee tag: `Exact` — total *given the bound* (the `Nat` bound makes it terminating)
/// # Fallibility: total
/// # Effects: none
/// # EXPLAIN: the bound applied is the `n` parameter — always visible to the caller
#[must_use]
pub fn lazy_take<E: 'static>(source: Lazy<E>, n: usize) -> Foldable<E> {
    source.take(n)
}

// ─── EXPLAIN artifacts ────────────────────────────────────────────────────────

/// The reified done-flag witness for [`any_with_witness`] and [`all_with_witness`] (C3).
///
/// Records the zero-based index of the first element that triggered the done-flag:
/// - for `any_with_witness`: the index of the first element satisfying the predicate.
/// - for `all_with_witness`: the index of the first element *failing* the predicate.
///
/// `None` means the done-flag was never fired. This makes the done-flag decision inspectable
/// without coupling the result type to a specific value shape (C3 / spec §5 C3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyAllWitness {
    /// Zero-based index of the element that fired the done-flag, or `None` if none did.
    pub first_match: Option<usize>,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ──────────────────────────────────────────────────────────────

    fn fvec<E: Clone>(v: &[E]) -> Foldable<E> {
        Foldable::from_vec(v.to_vec())
    }

    fn nums(n: usize) -> Foldable<usize> {
        Foldable::from_vec((0..n).collect())
    }

    // ─── guarantee matrix checks (RFC-0016 §4.5) ──────────────────────────────

    #[test]
    fn matrix_row_count_matches_expected() {
        assert_eq!(
            guarantee_matrix::MATRIX.len(),
            21,
            "matrix must have 21 rows (18 spec rows + 2 split rows + zip_exact)"
        );
    }

    #[test]
    fn all_eager_rows_are_exact_and_effect_free() {
        for row in guarantee_matrix::MATRIX {
            if row.op != "lazy_unfold" {
                assert_eq!(
                    row.tag,
                    GuaranteeStrength::Exact,
                    "op '{}' should be Exact (inherited from kernel `for` fold)",
                    row.op
                );
            }
            assert_eq!(row.effects, "none", "op '{}' must be effect-free", row.op);
        }
    }

    #[test]
    fn lazy_unfold_is_declared() {
        let row = guarantee_matrix::MATRIX
            .iter()
            .find(|r| r.op == "lazy_unfold")
            .expect("lazy_unfold must be in the matrix");
        assert_eq!(
            row.tag,
            GuaranteeStrength::Declared,
            "lazy_unfold must be tagged Declared (not total — VR-5)"
        );
        assert!(
            !row.totality_preserving,
            "lazy_unfold is NOT totality-preserving"
        );
    }

    #[test]
    fn all_eager_rows_are_totality_preserving() {
        for row in guarantee_matrix::MATRIX {
            if row.op != "lazy_unfold" {
                assert!(
                    row.totality_preserving,
                    "op '{}' must be totality-preserving",
                    row.op
                );
            }
        }
    }

    #[test]
    fn explainable_ops_include_decision_bearing_ones() {
        let explainable: Vec<&str> = guarantee_matrix::MATRIX
            .iter()
            .filter(|r| r.explainable)
            .map(|r| r.op)
            .collect();
        for op in &[
            "zip",
            "any",
            "all",
            "find",
            "transduce",
            "lazy_unfold",
            "lazy_take",
        ] {
            assert!(explainable.contains(op), "op '{op}' should be EXPLAIN-able");
        }
        for op in &["map", "filter", "fold", "count", "chain", "skip", "take"] {
            assert!(
                !explainable.contains(op),
                "op '{op}' should NOT be EXPLAIN-able"
            );
        }
    }

    // ─── map ──────────────────────────────────────────────────────────────────

    #[test]
    fn map_applies_f_to_each_element() {
        let out = map(fvec(&[1u32, 2, 3]), |x| x * 2);
        assert_eq!(out.as_slice(), &[2u32, 4, 6]);
    }

    #[test]
    fn map_over_empty_is_empty() {
        let out = map(Foldable::<u32>::empty(), |x| x + 1);
        assert_eq!(out.len(), 0);
    }

    /// Property: map preserves length (totality-preserving bound).
    #[test]
    fn map_preserves_length_property() {
        for n in 0usize..=64 {
            assert_eq!(
                map(nums(n), |x| x + 1).len(),
                n,
                "map must preserve length for n={n}"
            );
        }
    }

    // ─── filter ───────────────────────────────────────────────────────────────

    #[test]
    fn filter_keeps_matching_elements() {
        let out = filter(fvec(&[1u32, 2, 3, 4, 5]), |x| x % 2 == 0);
        assert_eq!(out.as_slice(), &[2u32, 4]);
    }

    #[test]
    fn filter_all_false_is_empty() {
        let out = filter(fvec(&[1u32, 3, 5]), |_| false);
        assert_eq!(out.len(), 0);
    }

    /// Property: filter result length ≤ source length.
    #[test]
    fn filter_output_le_input_property() {
        for n in 0usize..=64 {
            let out = filter(nums(n), |x| x % 2 == 0);
            assert!(
                out.len() <= n,
                "filter output {olen} must be ≤ {n}",
                olen = out.len()
            );
        }
    }

    // ─── scan ─────────────────────────────────────────────────────────────────

    #[test]
    fn scan_running_sum() {
        let out = scan(fvec(&[1u32, 2, 3, 4]), 0u32, |acc, e| acc + e);
        assert_eq!(out.as_slice(), &[1u32, 3, 6, 10]);
    }

    /// Property: scan preserves length (spec §3 "length-preserving fold").
    #[test]
    fn scan_preserves_length_property() {
        for n in 0usize..=64 {
            let out = scan(nums(n), 0usize, |acc, e| acc + e);
            assert_eq!(out.len(), n, "scan is length-preserving for n={n}");
        }
    }

    // ─── enumerate ────────────────────────────────────────────────────────────

    #[test]
    fn enumerate_attaches_indices() {
        let out = enumerate(fvec(&["a", "b", "c"]));
        assert_eq!(out.as_slice(), &[(0, "a"), (1, "b"), (2, "c")]);
    }

    /// Property: enumerate preserves length and indices are 0..n.
    #[test]
    fn enumerate_length_and_indices_property() {
        for n in 0usize..=64 {
            let out = enumerate(nums(n));
            assert_eq!(out.len(), n);
            for (i, (idx, val)) in out.as_slice().iter().enumerate() {
                assert_eq!(*idx, i);
                assert_eq!(*val, i);
            }
        }
    }

    // ─── flat_map ─────────────────────────────────────────────────────────────

    #[test]
    fn flat_map_flattens_finite_of_finite() {
        let out = flat_map(fvec(&[1u32, 2, 3]), |x| fvec(&[x, x * 10]));
        assert_eq!(out.as_slice(), &[1u32, 10, 2, 20, 3, 30]);
    }

    #[test]
    fn flat_map_empty_inner_is_empty_output() {
        let out = flat_map(fvec(&[1u32, 2, 3]), |_| Foldable::<u32>::empty());
        assert_eq!(out.len(), 0);
    }

    // ─── fold ─────────────────────────────────────────────────────────────────

    #[test]
    fn fold_sums_correctly() {
        let total = fold(fvec(&[1u32, 2, 3, 4, 5]), 0u32, |acc, e| acc + e);
        assert_eq!(total, 15);
    }

    #[test]
    fn fold_over_empty_returns_init() {
        let result = fold(Foldable::<u32>::empty(), 42u32, |acc, e| acc + e);
        assert_eq!(result, 42);
    }

    /// Property: fold left-sum = Σ(0..n).
    #[test]
    fn fold_left_sum_property() {
        for n in 0usize..=64 {
            let sum = fold(nums(n), 0usize, |a, e| a + e);
            let expected: usize = (0..n).sum();
            assert_eq!(sum, expected, "fold sum must equal Σ(0..n) for n={n}");
        }
    }

    // ─── reduce ───────────────────────────────────────────────────────────────

    #[test]
    fn reduce_none_on_empty() {
        assert_eq!(reduce(Foldable::<u32>::empty(), |a, b| a + b), None);
    }

    #[test]
    fn reduce_identity_on_singleton() {
        assert_eq!(reduce(fvec(&[42u32]), |a, b| a + b), Some(42));
    }

    #[test]
    fn reduce_sum_on_multiple() {
        assert_eq!(reduce(fvec(&[1u32, 2, 3, 4]), |a, b| a + b), Some(10));
    }

    // ─── count ────────────────────────────────────────────────────────────────

    #[test]
    fn count_returns_length() {
        assert_eq!(count(fvec(&[1u32, 2, 3])), 3);
        assert_eq!(count(Foldable::<u32>::empty()), 0);
    }

    /// Property: count = len.
    #[test]
    fn count_equals_len_property() {
        for n in 0usize..=128 {
            assert_eq!(count(nums(n)), n);
        }
    }

    // ─── any_with_witness ─────────────────────────────────────────────────────

    #[test]
    fn any_true_when_pred_holds() {
        let (result, witness) = any_with_witness(fvec(&[1u32, 2, 3, 4]), |x| *x == 3);
        assert!(result);
        assert_eq!(witness.first_match, Some(2));
    }

    #[test]
    fn any_false_when_pred_never_holds() {
        let (result, witness) = any_with_witness(fvec(&[1u32, 3, 5]), |x| x % 2 == 0);
        assert!(!result);
        assert_eq!(witness.first_match, None);
    }

    #[test]
    fn any_false_on_empty() {
        let (result, witness) = any_with_witness(Foldable::<u32>::empty(), |_| true);
        assert!(!result);
        assert_eq!(witness.first_match, None);
    }

    /// Property: when pred fires at index 0, first_match is Some(0) for all source lengths.
    #[test]
    fn any_done_flag_first_match_at_zero_property() {
        for n in 1usize..=32 {
            let (result, witness) = any_with_witness(nums(n), |x| *x == 0);
            assert!(result, "n={n}: pred fires at 0");
            assert_eq!(witness.first_match, Some(0), "n={n}: first_match must be 0");
        }
    }

    // ─── all_with_witness ─────────────────────────────────────────────────────

    #[test]
    fn all_true_when_pred_holds_for_all() {
        let (result, witness) = all_with_witness(fvec(&[2u32, 4, 6]), |x| x % 2 == 0);
        assert!(result);
        assert_eq!(witness.first_match, None);
    }

    #[test]
    fn all_false_on_first_failure() {
        let (result, witness) = all_with_witness(fvec(&[2u32, 3, 4]), |x| x % 2 == 0);
        assert!(!result);
        assert_eq!(witness.first_match, Some(1));
    }

    #[test]
    fn all_true_on_empty() {
        let (result, _) = all_with_witness(Foldable::<u32>::empty(), |_| false);
        assert!(result);
    }

    /// Property: all_with_witness reports first failure index correctly.
    #[test]
    fn all_witness_first_failure_property() {
        for n in 2usize..=32 {
            let (result, witness) = all_with_witness(nums(n), |x| *x % 2 == 0);
            // element 1 (value 1) fails the even predicate.
            assert!(!result, "n={n}: element 1 fails pred");
            assert_eq!(
                witness.first_match,
                Some(1),
                "n={n}: first failure at index 1"
            );
        }
    }

    // ─── find / position ──────────────────────────────────────────────────────

    #[test]
    fn find_returns_first_match() {
        assert_eq!(find(fvec(&[1u32, 2, 3, 2, 1]), |x| *x == 2), Some(2u32));
    }

    #[test]
    fn find_none_when_absent() {
        assert_eq!(find(fvec(&[1u32, 3, 5]), |x| *x == 2), None);
    }

    #[test]
    fn position_returns_first_index() {
        assert_eq!(position(fvec(&[10u32, 20, 30, 20]), |x| *x == 20), Some(1));
    }

    #[test]
    fn position_none_when_absent() {
        assert_eq!(position(fvec(&[1u32, 3, 5]), |x| *x == 2), None);
    }

    /// Property: position(pred) matches the expected index for a known element.
    #[test]
    fn position_correct_index_property() {
        let data: Foldable<u32> = Foldable::from_vec((0..32).collect());
        assert_eq!(position(data, |x| *x == 17), Some(17));
    }

    // ─── zip ──────────────────────────────────────────────────────────────────

    #[test]
    fn zip_equal_lengths() {
        let (out, outcome) = zip(fvec(&[1u32, 2, 3]), fvec(&[10u32, 20, 30]));
        assert_eq!(out.as_slice(), &[(1, 10), (2, 20), (3, 30)]);
        assert!(!outcome.was_truncated());
    }

    #[test]
    fn zip_truncates_to_shorter() {
        let (out, outcome) = zip(fvec(&[1u32, 2, 3, 4]), fvec(&[10u32, 20]));
        assert_eq!(out.as_slice(), &[(1, 10), (2, 20)]);
        assert!(outcome.was_truncated());
        assert_eq!(outcome.left_len(), 4);
        assert_eq!(outcome.right_len(), 2);
        assert_eq!(outcome.result_len(), 2);
        assert_eq!(outcome.left_excess(), 2);
        assert_eq!(outcome.right_excess(), 0);
    }

    #[test]
    fn zip_exact_ok_on_equal_lengths() {
        assert!(zip_exact(fvec(&[1u32, 2]), fvec(&[10u32, 20])).is_ok());
    }

    #[test]
    fn zip_exact_err_on_unequal_lengths() {
        let err = zip_exact(fvec(&[1u32, 2, 3]), fvec(&[10u32, 20])).unwrap_err();
        assert_eq!(err.left_len, 3);
        assert_eq!(err.right_len, 2);
    }

    /// Property: zip result length = min(left, right).
    #[test]
    fn zip_result_length_is_min_property() {
        for (l, r) in [(0, 0), (3, 3), (5, 2), (0, 5), (10, 10), (7, 0)] {
            let (out, _) = zip(nums(l), nums(r));
            assert_eq!(out.len(), l.min(r));
        }
    }

    // ─── chain ────────────────────────────────────────────────────────────────

    #[test]
    fn chain_appends_right_after_left() {
        let out = chain(fvec(&[1u32, 2]), fvec(&[3u32, 4, 5]));
        assert_eq!(out.as_slice(), &[1u32, 2, 3, 4, 5]);
    }

    /// Property: chain length = left + right.
    #[test]
    fn chain_length_is_sum_property() {
        for (l, r) in [(0, 0), (3, 3), (5, 2), (0, 5)] {
            assert_eq!(chain(nums(l), nums(r)).len(), l + r);
        }
    }

    // ─── take ─────────────────────────────────────────────────────────────────

    #[test]
    fn take_keeps_first_n() {
        let out = take(fvec(&[1u32, 2, 3, 4, 5]), 3);
        assert_eq!(out.as_slice(), &[1u32, 2, 3]);
    }

    #[test]
    fn take_beyond_length_returns_all() {
        let out = take(fvec(&[1u32, 2]), 10);
        assert_eq!(out.as_slice(), &[1u32, 2]);
    }

    /// Property: take(n) result length = min(n, source.len()).
    #[test]
    fn take_length_is_min_property() {
        for src_len in 0usize..=32 {
            for n in 0usize..=(src_len + 5) {
                assert_eq!(
                    take(nums(src_len), n).len(),
                    src_len.min(n),
                    "take({n}) on len={src_len}"
                );
            }
        }
    }

    // ─── skip ─────────────────────────────────────────────────────────────────

    #[test]
    fn skip_drops_first_n() {
        let out = skip(fvec(&[1u32, 2, 3, 4, 5]), 2);
        assert_eq!(out.as_slice(), &[3u32, 4, 5]);
    }

    #[test]
    fn skip_beyond_length_is_empty() {
        assert_eq!(skip(fvec(&[1u32, 2]), 10).len(), 0);
    }

    /// Property: skip(n) result length = max(0, source.len() - n).
    #[test]
    fn skip_length_property() {
        for src_len in 0usize..=32 {
            for n in 0usize..=(src_len + 5) {
                assert_eq!(
                    skip(nums(src_len), n).len(),
                    src_len.saturating_sub(n),
                    "skip({n}) on len={src_len}"
                );
            }
        }
    }

    // ─── step_by ──────────────────────────────────────────────────────────────

    #[test]
    fn step_by_zero_is_err() {
        assert!(step_by(fvec(&[1u32, 2, 3]), 0).is_err());
    }

    #[test]
    fn step_by_one_is_identity() {
        let out = step_by(fvec(&[1u32, 2, 3, 4, 5]), 1).unwrap();
        assert_eq!(out.as_slice(), &[1u32, 2, 3, 4, 5]);
    }

    #[test]
    fn step_by_two_keeps_even_indices() {
        let out = step_by(fvec(&[0u32, 1, 2, 3, 4, 5]), 2).unwrap();
        assert_eq!(out.as_slice(), &[0u32, 2, 4]);
    }

    /// Property: step_by(k) result length = ceil(source.len() / k) for k ≥ 1.
    #[test]
    fn step_by_length_property() {
        for src_len in 0usize..=32 {
            for k in 1usize..=8 {
                assert_eq!(
                    step_by(nums(src_len), k).unwrap().len(),
                    src_len.div_ceil(k),
                    "step_by({k}) on len={src_len}"
                );
            }
        }
    }

    // ─── transduce ────────────────────────────────────────────────────────────

    #[test]
    fn transduce_map_only() {
        let xf = Transducer::map(|x: u32| x * 2);
        let result = transduce(fvec(&[1u32, 2, 3]), &xf, vec![], |mut acc, e| {
            acc.push(e);
            acc
        });
        assert_eq!(result, vec![2u32, 4, 6]);
    }

    #[test]
    fn transduce_filter_only() {
        let xf = Transducer::filter(|x: &u32| (*x).is_multiple_of(2));
        let result = transduce(fvec(&[1u32, 2, 3, 4]), &xf, vec![], |mut acc, e| {
            acc.push(e);
            acc
        });
        assert_eq!(result, vec![2u32, 4]);
    }

    #[test]
    fn transduce_composed_filter_then_map() {
        // filter(x > 2) then map(x * 2): [1,2,3,4] → [3,4] → [6,8]
        let xf = Transducer::filter(|x: &u32| *x > 2).compose(Transducer::map(|x: u32| x * 2));
        let result = transduce(fvec(&[1u32, 2, 3, 4]), &xf, vec![], |mut acc, e| {
            acc.push(e);
            acc
        });
        assert_eq!(result, vec![6u32, 8]);
    }

    #[test]
    fn transducer_describe_is_non_empty() {
        let xf = Transducer::filter(|x: &u32| *x > 2).compose(Transducer::map(|x: u32| x * 2));
        assert!(!xf.describe().is_empty());
    }

    /// Property (spec §7-Q5 / VR-5 — `Empirical`): transduce(compose(t1, t2)) = sequential.
    #[test]
    fn transduce_fusion_law_empirical() {
        let inputs = fvec(&[1u32, 2, 3, 4, 5, 6]);
        let composed =
            Transducer::<u32, u32>::filter(|x| *x > 2).compose(Transducer::map(|x: u32| x * 10));
        let composed_result = transduce(inputs.clone(), &composed, vec![], |mut acc, e| {
            acc.push(e);
            acc
        });
        let seq: Vec<u32> = inputs
            .into_vec()
            .into_iter()
            .filter(|x| *x > 2)
            .map(|x| x * 10)
            .collect();
        assert_eq!(
            composed_result, seq,
            "transduce(compose) must equal sequential application (Empirical)"
        );
    }

    // ─── Lazy surface ─────────────────────────────────────────────────────────

    #[test]
    fn lazy_unfold_and_lazy_take() {
        let lazy = Lazy::unfold(0u32, |s| Some((s, s + 1)));
        let foldable = lazy_take(lazy, 5);
        assert_eq!(foldable.as_slice(), &[0u32, 1, 2, 3, 4]);
    }

    #[test]
    fn lazy_take_zero_is_empty() {
        let lazy = Lazy::unfold(0u32, |s| Some((s, s + 1)));
        assert_eq!(lazy_take(lazy, 0).len(), 0);
    }

    #[test]
    fn lazy_take_terminates_on_infinite_source() {
        let lazy = Lazy::unfold(0u64, |s| Some((s, s + 1)));
        let foldable = lazy_take(lazy, 100);
        assert_eq!(foldable.len(), 100);
        assert_eq!(foldable.as_slice()[99], 99u64);
    }

    #[test]
    fn lazy_terminates_at_bounded_source() {
        let lazy = Lazy::unfold(0u32, |s| if s < 3 { Some((s, s + 1)) } else { None });
        assert_eq!(lazy_take(lazy, 100).as_slice(), &[0u32, 1, 2]);
    }

    // ─── C1 never-silent roundup ──────────────────────────────────────────────

    #[test]
    fn c1_never_silent_roundup() {
        assert_eq!(reduce(Foldable::<u32>::empty(), |a, b| a + b), None);
        assert_eq!(find(Foldable::<u32>::empty(), |_| true), None);
        assert_eq!(position(Foldable::<u32>::empty(), |_| true), None);
        assert!(step_by(fvec(&[1u32]), 0).is_err());
        assert!(zip_exact(fvec(&[1u32]), fvec(&[1u32, 2])).is_err());
    }

    // ─── C4: value-semantic ───────────────────────────────────────────────────

    #[test]
    fn combinators_are_pure_same_input_same_output() {
        let src = fvec(&[1u32, 2, 3]);
        assert_eq!(
            map(src.clone(), |x| x * 3).as_slice(),
            map(src, |x| x * 3).as_slice()
        );
    }

    // ─── composition: chain of combinators stays total ────────────────────────

    #[test]
    fn composed_combinators_stay_total() {
        // skip(5) → [5..20]; take(10) → [5..15]; filter(even) → [6,8,10,12,14]; map(x²)
        let result: Vec<usize> = map(filter(take(skip(nums(20), 5), 10), |x| x % 2 == 0), |x| {
            x * x
        })
        .into_vec();
        assert_eq!(result, vec![36usize, 64, 100, 144, 196]);
    }

    // ─── ZipOutcome EXPLAIN ───────────────────────────────────────────────────

    #[test]
    fn zip_outcome_explain_is_complete() {
        let (_, outcome) = zip(nums(7), nums(3));
        assert!(outcome.was_truncated());
        assert_eq!(outcome.left_len(), 7);
        assert_eq!(outcome.right_len(), 3);
        assert_eq!(outcome.result_len(), 3);
        assert_eq!(outcome.left_excess(), 4);
        assert_eq!(outcome.right_excess(), 0);
    }
}
