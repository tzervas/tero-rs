//! `Set<E>` — immutable persistent membership set (spec §3, RFC-0016 §4.4).
//!
//! # Guarantee tag: `Exact` throughout (spec §4 / RFC-0016 C2)
//! No `Set` op carries accuracy, precision, or probability semantics. Every operation is
//! a deterministic structural fact — `Exact` is the honest floor.
//!
//! # Value semantics (C4 / ADR-003)
//! `Set<E>` is an **immutable value**. Every "mutating" op (`insert`, `remove`,
//! `union`, `intersection`, `difference`) returns a *new* `Set<E>`; the receiver is
//! never modified. Structural sharing is an implementation detail invisible to identity.
//!
//! # Never-silent (C1 / G2)
//! - `contains` is total and returns `bool`.
//! - `insert` is *idempotent*: inserting an already-present element returns a new set
//!   with the same contents (no error; the op is total).
//! - `remove` is *idempotent on absent elements*: removing an absent element returns a
//!   new set with the same contents — **never** an error or a silent no-op on the
//!   original (the "no-op-returning-new" guarantee from spec §3).
//!
//! # Iteration order (documented — the honesty crux; RFC-0016 §4.4)
//! `Set` uses **insertion order**: `foldable()` yields elements in the order they were
//! first inserted. `union`, `intersection`, and `difference` each document their output
//! order (see per-op docs). The order is a property of the type, never an exposed
//! hash-bucket order (the no-silent-reorder crux).
//!
//! Note on the §7-Q1 open question: spec §7-Q1 leaves the ordered-vs-bucketed default
//! to M-501. This implementation fixes insertion order (the simplest deterministic
//! choice that satisfies no-silent-reorder). FLAGGED below.
//!
//! # EXPLAIN (C3)
//! `union` / `intersection` / `difference` each declare their result order in their
//! docs — the documented order is the inspectable artifact (spec §4).

use std::collections::HashMap;

/// An immutable persistent membership set (value-semantic; spec §3).
///
/// Iteration order is **insertion order** (first insertion of each element determines
/// position). This is a documented, stable property — never an exposed hash-bucket order.
///
/// # FLAG: §7-Q1
/// Insertion-order is chosen as the deterministic floor; M-501 may ratify a different
/// default. The property tests check the order-stability invariant and will catch regressions.
#[derive(Debug, Clone)]
pub struct Set<E> {
    /// Elements in insertion order.
    order: Vec<E>,
    /// Element → index in `order` (non-identity bucketing hash — not std.content).
    index: HashMap<E, usize>,
}

// Manual PartialEq/Eq based on the `order` field only (the index is derived from it).
// Two sets are equal iff they have the same elements in the same documented insertion order.
// This is the correct value-semantic equality: same observable contents = same value (C4).
impl<E: Clone + Eq + std::hash::Hash> PartialEq for Set<E> {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}
impl<E: Clone + Eq + std::hash::Hash> Eq for Set<E> {}

impl<E: Clone + Eq + std::hash::Hash> Set<E> {
    // ─── Constructors ─────────────────────────────────────────────────────────

    /// An empty `Set`.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn empty() -> Self {
        Set {
            order: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Construct a `Set` from a slice of elements (insertion order = slice order;
    /// duplicates take the first occurrence's position).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn from_slice(elems: &[E]) -> Self {
        let mut s = Set::empty();
        for e in elems {
            s = s.insert(e.clone());
        }
        s
    }

    // ─── Queries ──────────────────────────────────────────────────────────────

    /// The number of elements.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// `true` iff the set is empty.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// `true` iff `x` is in the set (C1 — total, never a partial result).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn contains(&self, x: &E) -> bool {
        self.index.contains_key(x)
    }

    // ─── Mutators (all return a NEW value — value semantics, C4) ─────────────

    /// Insert `x`, returning a **new** `Set`.
    ///
    /// If `x` is already present, the new set has the same contents (idempotent).
    /// The receiver is not modified (C4).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn insert(&self, x: E) -> Self {
        if self.index.contains_key(&x) {
            // Idempotent: already present — return a new set with the same contents (C4).
            self.clone()
        } else {
            let mut new_set = self.clone();
            let i = new_set.order.len();
            new_set.order.push(x.clone());
            new_set.index.insert(x, i);
            new_set
        }
    }

    /// Remove `x`, returning a **new** `Set`.
    ///
    /// If `x` is absent, the new set has the same contents ("no-op-returning-new"
    /// guarantee from spec §3). The receiver is not modified (C4). No error is returned
    /// for a missing element (C1 — the absence of a non-member is not an error condition).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn remove(&self, x: &E) -> Self {
        if let Some(&i) = self.index.get(x) {
            // Rebuild without the removed element.
            let mut new_order: Vec<E> = Vec::with_capacity(self.order.len() - 1);
            let mut new_index: HashMap<E, usize> = HashMap::with_capacity(self.order.len() - 1);
            for (j, e) in self.order.iter().enumerate() {
                if j != i {
                    let new_i = new_order.len();
                    new_order.push(e.clone());
                    new_index.insert(e.clone(), new_i);
                }
            }
            Set {
                order: new_order,
                index: new_index,
            }
        } else {
            // Absent: return a new set with the same contents (no-op-returning-new, C4).
            self.clone()
        }
    }

    // ─── Set operations ───────────────────────────────────────────────────────

    /// Union: all elements from `self` then any new elements from `other`.
    ///
    /// Order: `self` elements first (in their insertion order), then elements that are in
    /// `other` but not in `self` (in `other`'s insertion order). This is documented and
    /// deterministic — never an exposed hash-bucket order.
    ///
    /// # EXPLAIN (C3)
    /// The result order (self first, then other-only) is the inspectable artifact.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for e in &other.order {
            result = result.insert(e.clone());
        }
        result
    }

    /// Intersection: elements present in **both** `self` and `other`, in `self`'s
    /// insertion order.
    ///
    /// Order: the order of elements in `self` is preserved for those that are also in
    /// `other`. Documented and deterministic.
    ///
    /// # EXPLAIN (C3)
    /// The result order (self's order, filtered) is the inspectable artifact.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = Set::empty();
        for e in &self.order {
            if other.contains(e) {
                result = result.insert(e.clone());
            }
        }
        result
    }

    /// Difference: elements in `self` that are **not** in `other`, in `self`'s insertion
    /// order.
    ///
    /// Order: self's insertion order, filtered to elements absent from `other`. Documented
    /// and deterministic.
    ///
    /// # EXPLAIN (C3)
    /// The result order (self's order, filtered) is the inspectable artifact.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut result = Set::empty();
        for e in &self.order {
            if !other.contains(e) {
                result = result.insert(e.clone());
            }
        }
        result
    }

    // ─── Foldable (for std.iter integration) ─────────────────────────────────

    /// Iterate elements in **insertion order** (documented order, no silent reorder).
    ///
    /// This is the documented-order promise (the no-silent-reorder crux). The order is
    /// a property of the type, not of the build or hash seed.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn foldable(&self) -> &[E] {
        &self.order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Helpers ──────────────────────────────────────────────────────────────

    fn set_abc() -> Set<&'static str> {
        Set::from_slice(&["a", "b", "c"])
    }

    // ─── value semantics (C4) ─────────────────────────────────────────────────

    /// `insert` returns a new `Set`; original is unchanged (C4).
    #[test]
    fn insert_returns_new_set_original_unchanged() {
        let s = Set::empty();
        let s2 = s.insert("x");
        assert!(s.is_empty(), "original must be unchanged after insert (C4)");
        assert_eq!(s2.len(), 1);
    }

    /// `remove` returns a new `Set`; original is unchanged (C4).
    #[test]
    fn remove_returns_new_set_original_unchanged() {
        let s = set_abc();
        let s2 = s.remove(&"b");
        assert_eq!(s.len(), 3, "original must be unchanged after remove (C4)");
        assert_eq!(s2.len(), 2);
    }

    // ─── never-silent / idempotent (C1) ───────────────────────────────────────

    /// `insert` is idempotent: re-inserting a present element returns the same contents.
    #[test]
    fn insert_is_idempotent() {
        let s = set_abc();
        let s2 = s.insert("a"); // already present
        assert_eq!(s2.len(), 3, "idempotent insert must not grow the set");
        assert_eq!(
            s2.foldable(),
            s.foldable(),
            "idempotent insert must not change order"
        );
    }

    /// `remove` is idempotent on absent elements: returns new set with same contents (C1).
    #[test]
    fn remove_absent_element_returns_same_contents() {
        let s = set_abc();
        let s2 = s.remove(&"z"); // absent
        assert_eq!(s2.len(), 3, "absent remove must not shrink the set");
        assert_eq!(s2.foldable(), s.foldable());
    }

    /// `contains` is total and never misses present elements.
    #[test]
    fn contains_is_total_and_accurate() {
        let s = set_abc();
        assert!(s.contains(&"a"));
        assert!(s.contains(&"b"));
        assert!(s.contains(&"c"));
        assert!(!s.contains(&"z"), "C1: honest absence");
    }

    // ─── insertion-order guarantee (the honesty crux — no silent reorder) ────

    /// `foldable` walks in insertion order.
    #[test]
    fn foldable_walks_in_insertion_order() {
        let s = set_abc();
        assert_eq!(
            s.foldable(),
            &["a", "b", "c"],
            "foldable must walk in insertion order (spec §3)"
        );
    }

    /// Inserting elements in a non-sorted order preserves that order in `foldable`.
    #[test]
    fn insertion_order_preserved_regardless_of_element_hash() {
        let s = Set::from_slice(&["c", "a", "b"]);
        assert_eq!(
            s.foldable(),
            &["c", "a", "b"],
            "foldable must walk in INSERTION order, not hash/alpha order (no silent reorder)"
        );
    }

    // ─── set operations + their documented orders ─────────────────────────────

    /// `union`: self-elements first, then other-only elements.
    #[test]
    fn union_order_self_first_then_other_only() {
        let a = Set::from_slice(&[1, 2, 3]);
        let b = Set::from_slice(&[3, 4, 5]);
        let u = a.union(&b);
        // 1, 2, 3 from a; then 4, 5 from b (3 already present).
        assert_eq!(u.foldable(), &[1, 2, 3, 4, 5]);
    }

    /// `intersection`: self's order, filtered to elements also in other.
    #[test]
    fn intersection_order_is_self_filtered() {
        let a = Set::from_slice(&[1, 2, 3, 4]);
        let b = Set::from_slice(&[4, 3, 5]);
        let i = a.intersection(&b);
        // 3, 4 from a (in a's order), since those are in b.
        assert_eq!(i.foldable(), &[3, 4]);
    }

    /// `difference`: self's order, filtered to elements not in other.
    #[test]
    fn difference_order_is_self_filtered() {
        let a = Set::from_slice(&[1, 2, 3, 4]);
        let b = Set::from_slice(&[3, 4, 5]);
        let d = a.difference(&b);
        // 1, 2 from a (in a's order), since those are not in b.
        assert_eq!(d.foldable(), &[1, 2]);
    }

    /// union is commutative in *content* but NOT necessarily in *order*.
    #[test]
    fn union_content_is_commutative_but_order_may_differ() {
        let a = Set::from_slice(&[1, 2]);
        let b = Set::from_slice(&[2, 3]);
        let u_ab = a.union(&b);
        let u_ba = b.union(&a);
        // Both contain {1, 2, 3} — content equal.
        assert!(u_ab.contains(&1));
        assert!(u_ab.contains(&2));
        assert!(u_ab.contains(&3));
        assert!(u_ba.contains(&1));
        assert!(u_ba.contains(&2));
        assert!(u_ba.contains(&3));
        // But order differs: a∪b = [1,2,3], b∪a = [2,3,1].
        assert_eq!(u_ab.foldable(), &[1, 2, 3]);
        assert_eq!(u_ba.foldable(), &[2, 3, 1]);
    }

    // ─── property test: no-silent-reorder invariant (spec §4 / RFC-0016 §4.4) ─

    /// Property: two sets with the same elements inserted in the same order yield the
    /// same `foldable` sequence.
    ///
    /// Spec §4 §7-Q2: `Declared` invariant → promoted to `Empirical` by this test (VR-5).
    #[test]
    fn equal_contents_same_order_yield_same_foldable() {
        let s1 = Set::from_slice(&[10, 20, 30]);
        let s2 = Set::from_slice(&[10, 20, 30]);
        assert_eq!(
            s1.foldable(),
            s2.foldable(),
            "same contents + same insertion order must yield same foldable (no silent reorder)"
        );
    }

    /// Property: after remove, the remaining elements keep their relative insertion order.
    #[test]
    fn remove_preserves_relative_order_of_remaining_elements() {
        let s = set_abc();
        let s2 = s.remove(&"b");
        assert_eq!(
            s2.foldable(),
            &["a", "c"],
            "remove must preserve relative insertion order of remaining elements (no silent reorder)"
        );
    }

    // ─── round-trip (insert/remove) ───────────────────────────────────────────

    /// Property: insert then remove round-trips (element absent after removal).
    #[test]
    fn insert_remove_round_trip() {
        let s = set_abc();
        let s2 = s.insert("d");
        assert!(s2.contains(&"d"));
        let s3 = s2.remove(&"d");
        assert!(!s3.contains(&"d"));
        assert_eq!(s3.len(), s.len(), "round-trip must restore original length");
        assert_eq!(
            s3.foldable(),
            s.foldable(),
            "round-trip must restore original order"
        );
    }
}
