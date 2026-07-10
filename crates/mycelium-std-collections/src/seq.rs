//! `Seq<E>` — immutable persistent indexed sequence (spec §3, RFC-0016 §4.4).
//!
//! # Guarantee tag: `Exact` throughout (spec §4 / RFC-0016 C2)
//! No `Seq` op carries accuracy, precision, or probability semantics. Every operation is a
//! deterministic structural fact — `Exact` is the honest floor (RFC-0016 C2 "an op with no
//! accuracy semantics … is simply `Exact`").
//!
//! # Value semantics (C4 / ADR-003)
//! `Seq<E>` is an **immutable value**. Every "mutating" op (`push`, `pop`, `update`,
//! `concat`, `slice`) returns a *new* `Seq<E>`; the receiver is never modified.
//! Structural sharing is an implementation detail: it is invisible to identity.
//!
//! # Never-silent (C1 / G2)
//! - `get` / `first` return `Option` — `None` when out of range / empty. Never a default.
//! - `pop` returns `Option<(Seq<E>, E)>` — `None` on empty. Never a silent no-op.
//! - `update` / `slice` return `Result<_, CollErr>` — `Err(IndexOOB)` on bad bounds. Never
//!   a clamp or a sentinel.
//!
//! # Iteration order (documented — the honesty crux)
//! `foldable` walks elements in **index order** (0, 1, …, len-1). This is a first-class
//! promise: the order is stable across `push`/`pop`/`update` and depends only on the
//! sequence's contents, never on internal implementation choices (no silent reorder).
//!
//! # EXPLAIN (C3)
//! `update` and `slice` produce `Err(IndexOOB)` refusal records whose fields (`index`,
//! `len`, `context`) are inspectable — the caller can read *why* the op was refused
//! without parsing strings (RFC-0013 structured diagnostic).

use crate::error::CollErr;

/// An immutable persistent indexed sequence (value-semantic; spec §3).
///
/// Iteration order is **index order** (0, 1, …, len-1) — a documented, stable property
/// of the type (the no-silent-reorder crux from RFC-0016 §4.4).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Seq<E> {
    /// Internal storage. `Arc`-based sharing is deferred (spec is design-phase);
    /// a plain `Vec` gives the correct value semantics and passes the guarantee matrix.
    /// Structural sharing is an *implementation* property invisible to identity (C4).
    inner: Vec<E>,
}

impl<E: Clone> Seq<E> {
    // ─── Constructors ─────────────────────────────────────────────────────────

    /// An empty `Seq`.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn empty() -> Self {
        Seq { inner: Vec::new() }
    }

    /// Construct a `Seq` from a slice of elements (cloned).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn from_slice(elems: &[E]) -> Self {
        Seq {
            inner: elems.to_vec(),
        }
    }

    // ─── Queries ──────────────────────────────────────────────────────────────

    /// The number of elements.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` iff the sequence is empty.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The element at index `i`, or `None` when `i >= len` (C1 — never a default).
    ///
    /// Guarantee: `Exact`. Fallibility: `None` when `i >= len`.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&E> {
        self.inner.get(i)
    }

    /// The first element, or `None` on empty (C1 — never a default).
    ///
    /// Guarantee: `Exact`. Fallibility: `None` on empty.
    #[must_use]
    pub fn first(&self) -> Option<&E> {
        self.inner.first()
    }

    // ─── Mutators (all return a NEW value — value semantics, C4) ─────────────

    /// Append `x` to the back, returning a new `Seq`.
    ///
    /// The input `Seq` is not modified (value semantics, C4).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn push(&self, x: E) -> Self {
        let mut new_inner = self.inner.clone();
        new_inner.push(x);
        Seq { inner: new_inner }
    }

    /// Remove the last element, returning `Some((new_seq, element))`,
    /// or `None` on empty (C1 — never a silent no-op).
    ///
    /// Guarantee: `Exact`. Fallibility: `None` on empty.
    #[must_use]
    pub fn pop(&self) -> Option<(Self, E)> {
        if self.inner.is_empty() {
            None
        } else {
            let mut new_inner = self.inner.clone();
            let elem = new_inner.pop().expect("inner is non-empty");
            Some((Seq { inner: new_inner }, elem))
        }
    }

    /// Replace element at index `i` with `x`, returning a new `Seq`.
    ///
    /// Returns `Err(IndexOOB)` when `i >= len` (C1 — never a silent clamp or default).
    ///
    /// # EXPLAIN (C3)
    /// The `Err(IndexOOB { index, len, context })` is the reified refusal record
    /// (RFC-0013 structured diagnostic): `index`, `len`, and `context` are inspectable.
    ///
    /// Guarantee: `Exact`. Fallibility: `Err(IndexOOB)` when `i >= len`.
    pub fn update(&self, i: usize, x: E) -> Result<Self, CollErr> {
        let len = self.inner.len();
        if i >= len {
            return Err(CollErr::index_oob(i, len, "i >= len"));
        }
        let mut new_inner = self.inner.clone();
        new_inner[i] = x;
        Ok(Seq { inner: new_inner })
    }

    /// Concatenate `self` (then `other`), returning a new `Seq`.
    ///
    /// Order: `self` elements first, then `other` elements (documented).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn concat(&self, other: &Self) -> Self {
        let mut new_inner = self.inner.clone();
        new_inner.extend_from_slice(&other.inner);
        Seq { inner: new_inner }
    }

    /// Return the sub-sequence `[lo, hi)`.
    ///
    /// Returns `Err(IndexOOB)` when `lo > hi` or `hi > len` — never silently clamps
    /// (C1 — the "no silent clamp" guarantee).
    ///
    /// # EXPLAIN (C3)
    /// The refusal record names the violated constraint in `context`:
    /// `"lo > hi"` or `"hi > len"`.
    ///
    /// Guarantee: `Exact`. Fallibility: `Err(IndexOOB)` on `lo > hi` or `hi > len`.
    pub fn slice(&self, lo: usize, hi: usize) -> Result<Self, CollErr> {
        let len = self.inner.len();
        if lo > hi {
            return Err(CollErr::index_oob(lo, hi, "lo > hi"));
        }
        if hi > len {
            return Err(CollErr::index_oob(hi, len, "hi > len"));
        }
        Ok(Seq {
            inner: self.inner[lo..hi].to_vec(),
        })
    }

    // ─── Foldable (for std.iter integration) ─────────────────────────────────

    /// Iterate elements in **index order** (0, 1, …, len-1).
    ///
    /// This is the documented-order promise (the no-silent-reorder crux). The order is
    /// a property of the type, not of the build or insertion history.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn foldable(&self) -> &[E] {
        // Returning a slice is the simplest inspectable Foldable at this layer.
        // When std.iter (M-526) lands it will consume this as a sequence source.
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CollErr;

    // ─── Helpers ──────────────────────────────────────────────────────────────

    fn seq123() -> Seq<i32> {
        Seq::from_slice(&[1, 2, 3])
    }

    // ─── guarantee matrix spot-checks ──────────────────────────────────────

    /// `len` / `is_empty` are `Exact` and total.
    #[test]
    fn len_and_is_empty_are_total() {
        let s: Seq<i32> = Seq::empty();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        let s2 = s.push(42);
        assert_eq!(s2.len(), 1);
        assert!(!s2.is_empty());
    }

    /// `get` returns `None` for out-of-range, never panics (C1).
    #[test]
    fn get_returns_none_when_out_of_range() {
        let s = seq123();
        assert_eq!(s.get(0), Some(&1));
        assert_eq!(s.get(2), Some(&3));
        // C1: index == len is out of range → None, never a silent default.
        assert_eq!(s.get(3), None);
        assert_eq!(s.get(usize::MAX), None);
    }

    /// `first` returns `None` on empty (C1).
    #[test]
    fn first_returns_none_on_empty() {
        let s: Seq<i32> = Seq::empty();
        assert_eq!(s.first(), None); // C1: never a sentinel
        let s2 = s.push(7);
        assert_eq!(s2.first(), Some(&7));
    }

    // ─── value semantics (C4): every mutator returns a NEW value ──────────

    /// `push` returns a new `Seq`; the original is unchanged (C4).
    #[test]
    fn push_returns_new_seq_original_unchanged() {
        let s = seq123();
        let s2 = s.push(4);
        assert_eq!(s.len(), 3, "original must be unchanged after push (C4)");
        assert_eq!(s2.len(), 4);
        assert_eq!(s2.get(3), Some(&4));
    }

    /// `pop` returns a new `Seq`; the original is unchanged (C4).
    #[test]
    fn pop_returns_new_seq_original_unchanged() {
        let s = seq123();
        let (s2, elem) = s.pop().expect("non-empty");
        assert_eq!(s.len(), 3, "original must be unchanged after pop (C4)");
        assert_eq!(s2.len(), 2);
        assert_eq!(elem, 3);
    }

    /// `pop` on empty returns `None` — never a silent no-op (C1).
    #[test]
    fn pop_returns_none_on_empty() {
        let s: Seq<i32> = Seq::empty();
        assert_eq!(s.pop(), None); // C1: never a sentinel or side-effect
    }

    /// `update` returns a new `Seq`; the original is unchanged (C4).
    #[test]
    fn update_returns_new_seq_original_unchanged() {
        let s = seq123();
        let s2 = s.update(1, 99).expect("valid index");
        assert_eq!(s.get(1), Some(&2), "original must be unchanged (C4)");
        assert_eq!(s2.get(1), Some(&99));
    }

    /// `update` returns `Err(IndexOOB)` when `i >= len` (C1).
    #[test]
    fn update_returns_err_when_oob() {
        let s = seq123();
        let err = s.update(3, 99).unwrap_err();
        // C1: Err carries the refusal record (C3/RFC-0013).
        assert!(
            matches!(
                err,
                CollErr::IndexOOB {
                    index: 3,
                    len: 3,
                    ..
                }
            ),
            "must be IndexOOB with index=3 len=3, got {err:?}"
        );
    }

    // ─── property test: update refusal bound (C1) ──────────────────────────

    /// Property: `update(i, x)` succeeds iff `i < len`; fails with `IndexOOB` iff `i >= len`.
    /// Guard: a clamping update (returning the unmodified seq instead of Err) makes this fail.
    #[test]
    fn update_succeeds_iff_i_lt_len() {
        // One representative sample per boundary:
        //   i == 0 (first valid), i == len-1 (last valid), i == len (first invalid),
        //   i == len+1 (further out), i == usize::MAX.
        let s = seq123();
        let len = s.len();
        for i in 0..len {
            assert!(
                s.update(i, 0).is_ok(),
                "update({i}) must succeed when i < len={len}"
            );
        }
        for i in [len, len + 1, usize::MAX] {
            let r = s.update(i, 0);
            assert!(
                r.is_err(),
                "update({i}) must fail when i >= len={len} (C1 never-silent)"
            );
        }
    }

    // ─── concat + slice ──────────────────────────────────────────────────────

    /// `concat` preserves order: `a ‖ b` — not `b ‖ a` (documented order).
    #[test]
    fn concat_order_is_a_then_b() {
        let a = Seq::from_slice(&[1, 2]);
        let b = Seq::from_slice(&[3, 4]);
        let ab = a.concat(&b);
        assert_eq!(ab.foldable(), &[1, 2, 3, 4]);
        // Reverse should be [3,4,1,2]
        let ba = b.concat(&a);
        assert_eq!(ba.foldable(), &[3, 4, 1, 2]);
        assert_ne!(ab, ba, "order must matter (documented concat order)");
    }

    /// `slice(lo, hi)` with `lo == hi` gives an empty sub-sequence.
    #[test]
    fn slice_empty_range_returns_empty() {
        let s = seq123();
        let empty = s.slice(1, 1).expect("valid empty range");
        assert!(empty.is_empty());
    }

    /// `slice` returns `Err(IndexOOB)` when `lo > hi` (C1 — no silent clamp).
    #[test]
    fn slice_err_when_lo_gt_hi() {
        let s = seq123();
        let err = s.slice(2, 1).unwrap_err();
        assert!(
            matches!(
                err,
                CollErr::IndexOOB {
                    context: "lo > hi",
                    ..
                }
            ),
            "must be IndexOOB with context='lo > hi', got {err:?}"
        );
    }

    /// `slice` returns `Err(IndexOOB)` when `hi > len` (C1 — no silent clamp).
    #[test]
    fn slice_err_when_hi_gt_len() {
        let s = seq123();
        let err = s.slice(0, 4).unwrap_err();
        assert!(
            matches!(
                err,
                CollErr::IndexOOB {
                    context: "hi > len",
                    ..
                }
            ),
            "must be IndexOOB with context='hi > len', got {err:?}"
        );
    }

    // ─── property test: slice bound invariants ───────────────────────────────

    /// Property: `slice(lo, hi)` succeeds iff `lo <= hi <= len`.
    /// Guard: a silently-clamped slice makes this fail.
    #[test]
    fn slice_succeeds_iff_valid_bounds() {
        let s = seq123();
        let len = s.len();
        // Valid: every (lo, hi) with lo <= hi <= len
        for lo in 0..=len {
            for hi in lo..=len {
                assert!(
                    s.slice(lo, hi).is_ok(),
                    "slice({lo},{hi}) must succeed with lo<=hi<=len={len}"
                );
            }
        }
        // Invalid: lo > hi
        assert!(s.slice(2, 1).is_err(), "slice(2,1) must fail: lo > hi (C1)");
        // Invalid: hi > len
        assert!(
            s.slice(0, len + 1).is_err(),
            "slice(0, len+1) must fail: hi > len (C1)"
        );
    }

    // ─── foldable iteration order ────────────────────────────────────────────

    /// `foldable` iterates in index order (documented order, no silent reorder).
    #[test]
    fn foldable_walks_in_index_order() {
        let s = seq123();
        let elems: Vec<i32> = s.foldable().to_vec();
        assert_eq!(
            elems,
            [1, 2, 3],
            "foldable must walk in index order (spec §3)"
        );
    }

    /// `foldable` of an empty seq is empty.
    #[test]
    fn foldable_of_empty_is_empty() {
        let s: Seq<i32> = Seq::empty();
        assert!(s.foldable().is_empty());
    }

    // ─── property test: no-silent-reorder invariant (spec §4 / RFC-0016 §4.4) ─

    /// Property: two `Seq`s built with the same elements in the same order are equal
    /// and their `foldable` slices are equal — insertion history with the same final
    /// contents yields the same observable sequence.
    ///
    /// Guard: any non-determinism in `push` ordering makes this fail.
    ///
    /// Note: this property is tagged `Declared` in the spec (§4/§7-Q2 — to be promoted
    /// to `Empirical` once code lands). It is checked here as a deterministic test
    /// (fixed elements, no randomness), which satisfies the property-test obligation
    /// (VR-5: do not claim `Proven` without a theorem).
    #[test]
    fn equal_contents_yield_equal_foldable_order() {
        // Path 1: push 1, 2, 3 sequentially.
        let s1 = Seq::empty().push(1).push(2).push(3);
        // Path 2: build from slice.
        let s2 = Seq::from_slice(&[1, 2, 3]);
        // Path 3: concat two sub-sequences.
        let s3 = Seq::from_slice(&[1]).concat(&Seq::from_slice(&[2, 3]));
        assert_eq!(
            s1.foldable(),
            s2.foldable(),
            "different construction paths with same contents must yield same foldable order"
        );
        assert_eq!(
            s1.foldable(),
            s3.foldable(),
            "concat path must yield same foldable order (no silent reorder)"
        );
    }

    // ─── round-trip test (push/pop) ───────────────────────────────────────────

    /// Property: push then pop round-trips the element.
    /// Guard: a push or pop that discards the element makes this fail.
    #[test]
    fn push_pop_round_trip() {
        let s = seq123();
        let s2 = s.push(99);
        let (s3, elem) = s2.pop().expect("non-empty after push");
        assert_eq!(elem, 99, "pop must return the last pushed element");
        assert_eq!(s3.len(), s.len(), "round-trip must restore original length");
        assert_eq!(
            s3.foldable(),
            s.foldable(),
            "round-trip must restore original contents"
        );
    }
}
