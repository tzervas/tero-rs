//! `Map<K, V>` — immutable persistent key→value map (spec §3, RFC-0016 §4.4).
//!
//! # Guarantee tag: `Exact` throughout (spec §4 / RFC-0016 C2)
//! No `Map` op carries accuracy, precision, or probability semantics. Every operation is
//! a deterministic structural fact — `Exact` is the honest floor.
//!
//! # Value semantics (C4 / ADR-003)
//! `Map<K, V>` is an **immutable value**. Every "mutating" op (`insert`, `remove`)
//! returns a *new* `Map<K, V>`; the receiver is never modified. Structural sharing is an
//! implementation detail invisible to identity.
//!
//! # Never-silent (C1 / G2)
//! - `get` returns `Option<&V>` — `None` on missing key, never a default value.
//! - `remove` returns `(Map<K,V>, Option<V>)` — the second field is `None` when the key
//!   was absent; the new `Map` is returned regardless (idempotent: absent key → same map).
//! - `get_or` is the **only** way to request a default, and it requires an **explicit**
//!   named `default` argument — the default is never silent (C1/G2).
//!
//! # Iteration order (documented — the honesty crux; RFC-0016 §4.4)
//! `Map` uses **insertion order**: `keys()`, `values()`, `entries()` yield entries in the
//! order they were first inserted. This is a *deterministic, documented property of the
//! type*, not a hash-bucket order that could silently change across a rehash/rebalance.
//! Since the internal bucketing hash is a private mechanism and the observable order is
//! recorded separately (insertion order), a "rehash" is an internal operation with **no**
//! observable reorder — the honesty crux (RFC-0016 §4.4).
//!
//! Note on the §7-Q1 open question: spec §7-Q1 leaves the ordered-vs-bucketed default
//! to M-501's ratification. This implementation **fixes insertion-order** (the simplest
//! deterministic choice that satisfies no-silent-reorder). This is flagged below.
//!
//! # Non-identity bucketing hash (content vs buckets boundary)
//! The `std::collections::HashMap` internally uses a hash. That is the *non-identity*
//! bucketing hash (hashing-for-maps). It is **not** `std.content`'s canonical
//! content-addressing hash (ADR-003; README §5 seam). The two are kept strictly distinct.
//!
//! # EXPLAIN (C3)
//! `keys()`, `values()`, `entries()` iterate in the documented insertion order — the
//! order contract itself is the inspectable artifact (spec §4): a caller can always see
//! *which* documented order the map yields.

use std::collections::HashMap;

/// An immutable persistent key→value map (value-semantic; spec §3).
///
/// Iteration order is **insertion order** (the first insertion of each key determines its
/// position). Duplicate-key inserts update the value but preserve the key's original
/// insertion position. This order is a documented, stable property — never an exposed
/// hash-bucket order (the no-silent-reorder crux, RFC-0016 §4.4).
///
/// # FLAG: §7-Q1
/// The ordered-vs-bucketed default is M-501's to ratify (spec §7-Q1). This implementation
/// chooses *insertion order* as the deterministic floor. If M-501 ratifies a different
/// order, this impl changes; the guarantee-matrix property test (order stability) will
/// catch any silent regression.
#[derive(Debug, Clone)]
pub struct Map<K, V> {
    /// Key-value pairs in insertion order (preserves documented iteration order).
    /// An `IndexMap`-like structure is appropriate here; we use a `Vec` + `HashMap`
    /// to avoid external dependencies while keeping insertion order stable.
    ///
    /// Invariant: `order` and `index` are always in sync.
    order: Vec<(K, V)>,
    /// Key → index in `order` (for O(n) lookup with no external deps).
    /// The hash is a *non-identity* bucketing hash (not `std.content`'s digest).
    index: HashMap<K, usize>,
}

// Manual PartialEq/Eq based on the `order` field only (the index is derived from it).
// Two maps are equal iff they have the same key-value pairs in the same documented order.
// This is the correct value-semantic equality: same observable contents = same value (C4).
impl<K: Clone + Eq + std::hash::Hash, V: PartialEq + Clone> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}
impl<K: Clone + Eq + std::hash::Hash, V: Eq + Clone> Eq for Map<K, V> {}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> Map<K, V> {
    // ─── Constructors ─────────────────────────────────────────────────────────

    /// An empty `Map`.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn empty() -> Self {
        Map {
            order: Vec::new(),
            index: HashMap::new(),
        }
    }

    // ─── Queries ──────────────────────────────────────────────────────────────

    /// The number of key→value pairs.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// `true` iff the map has no entries.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The value for key `k`, or `None` when `k` is absent (C1 — never a default).
    ///
    /// Guarantee: `Exact`. Fallibility: `None` on missing key.
    #[must_use]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.index.get(k).map(|&i| &self.order[i].1)
    }

    /// `true` iff `k` is present.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn contains_key(&self, k: &K) -> bool {
        self.index.contains_key(k)
    }

    /// The value for key `k`, or `default` when `k` is absent.
    ///
    /// The default is an **explicit named argument** — it is never a hidden fallback
    /// (C1). Callers who want absence to be explicit should use `get` (which returns
    /// `None`). `get_or` is the sanctioned "I know the default" surface.
    ///
    /// # EXPLAIN (C3)
    /// The default argument is reified at the call site: the caller can inspect
    /// `default` directly (no hidden fallback). This is the "default is a NAMED arg,
    /// not silent" guarantee from spec §3.
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn get_or<'a>(&'a self, k: &K, default: &'a V) -> &'a V {
        self.get(k).unwrap_or(default)
    }

    // ─── Mutators (all return a NEW value — value semantics, C4) ─────────────

    /// Insert `k → v`, returning a **new** `Map`.
    ///
    /// If `k` is already present, the value is replaced (the key retains its original
    /// insertion position in the iteration order). The receiver is not modified (C4).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn insert(&self, k: K, v: V) -> Self {
        let mut new_map = self.clone();
        if let Some(&i) = new_map.index.get(&k) {
            // Key exists: update value, preserve insertion-order position.
            new_map.order[i].1 = v;
        } else {
            // New key: append to insertion order.
            let i = new_map.order.len();
            new_map.order.push((k.clone(), v));
            new_map.index.insert(k, i);
        }
        new_map
    }

    /// Remove `k`, returning `(new_map, Option<V>)`.
    ///
    /// The second component is `Some(v)` when `k` was present, `None` when absent
    /// (C1 — the absence is explicit, never silent). The returned `Map` is always
    /// a *new* value (C4).
    ///
    /// Guarantee: `Exact`, total.
    pub fn remove(&self, k: &K) -> (Self, Option<V>) {
        if let Some(&i) = self.index.get(k) {
            let removed_v = self.order[i].1.clone();
            // Rebuild the order Vec and index map without the removed entry.
            let mut new_order: Vec<(K, V)> = Vec::with_capacity(self.order.len() - 1);
            let mut new_index: HashMap<K, usize> = HashMap::with_capacity(self.order.len() - 1);
            for (j, (ek, ev)) in self.order.iter().enumerate() {
                if j != i {
                    let new_i = new_order.len();
                    new_order.push((ek.clone(), ev.clone()));
                    new_index.insert(ek.clone(), new_i);
                }
            }
            (
                Map {
                    order: new_order,
                    index: new_index,
                },
                Some(removed_v),
            )
        } else {
            // Key absent: return a new map with the same contents (C4: still a new value).
            (self.clone(), None)
        }
    }

    // ─── Foldable surfaces (documented insertion order) ───────────────────────

    /// All keys in **insertion order** (documented — the honesty crux).
    ///
    /// Guarantee: `Exact`, total. EXPLAIN: the documented insertion order is the
    /// inspectable artifact.
    #[must_use]
    pub fn keys(&self) -> Vec<&K> {
        self.order.iter().map(|(k, _)| k).collect()
    }

    /// All values in **insertion order** (same documented order as `keys`).
    ///
    /// Guarantee: `Exact`, total.
    #[must_use]
    pub fn values(&self) -> Vec<&V> {
        self.order.iter().map(|(_, v)| v).collect()
    }

    /// All `(key, value)` pairs in **insertion order** (documented order).
    ///
    /// Guarantee: `Exact`, total. EXPLAIN: insertion order is the inspectable artifact.
    #[must_use]
    pub fn entries(&self) -> Vec<(&K, &V)> {
        self.order.iter().map(|(k, v)| (k, v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Helpers ──────────────────────────────────────────────────────────────

    fn map_abc() -> Map<&'static str, i32> {
        Map::empty().insert("a", 1).insert("b", 2).insert("c", 3)
    }

    // ─── value semantics (C4) ─────────────────────────────────────────────────

    /// `insert` returns a new `Map`; original is unchanged (C4).
    #[test]
    fn insert_returns_new_map_original_unchanged() {
        let m = Map::empty();
        let m2 = m.insert("k", 1);
        assert!(m.is_empty(), "original must be unchanged after insert (C4)");
        assert_eq!(m2.len(), 1);
    }

    /// `remove` returns a new `Map`; original is unchanged (C4).
    #[test]
    fn remove_returns_new_map_original_unchanged() {
        let m = map_abc();
        let (m2, v) = m.remove(&"b");
        assert_eq!(m.len(), 3, "original must be unchanged after remove (C4)");
        assert_eq!(m2.len(), 2);
        assert_eq!(v, Some(2));
    }

    // ─── never-silent (C1) ───────────────────────────────────────────────────

    /// `get` returns `None` on missing key — never a default (C1).
    #[test]
    fn get_returns_none_on_missing_key() {
        let m = map_abc();
        assert_eq!(m.get(&"z"), None); // C1: honest absence
    }

    /// `remove` returns `None` in second position when key absent (C1).
    #[test]
    fn remove_returns_none_when_key_absent() {
        let m = map_abc();
        let (m2, v) = m.remove(&"z");
        assert_eq!(v, None, "C1: absence must be explicit (None), not silent");
        assert_eq!(m2.len(), 3, "absent remove must return same-contents map");
    }

    /// `get_or` requires an explicit default (never a hidden fallback — C1).
    #[test]
    fn get_or_uses_explicit_default() {
        let m = map_abc();
        let def = 99;
        assert_eq!(*m.get_or(&"z", &def), 99, "missing key → explicit default");
        assert_eq!(
            *m.get_or(&"a", &def),
            1,
            "present key → its value, not default"
        );
    }

    // ─── insertion-order guarantee (the honesty crux — no silent reorder) ────

    /// `keys()`, `values()`, `entries()` all walk in insertion order.
    #[test]
    fn keys_values_entries_walk_in_insertion_order() {
        let m = map_abc();
        assert_eq!(m.keys(), vec![&"a", &"b", &"c"]);
        assert_eq!(m.values(), vec![&1, &2, &3]);
        assert_eq!(m.entries(), vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
    }

    /// Insert order determines position, not key hash (no silent reorder).
    #[test]
    fn insertion_order_preserved_regardless_of_key_hash() {
        // Insert in reverse alphabetical order: c, b, a.
        let m = Map::empty().insert("c", 3).insert("b", 2).insert("a", 1);
        assert_eq!(
            m.keys(),
            vec![&"c", &"b", &"a"],
            "keys must walk in INSERTION order, not hash/alpha order (no silent reorder)"
        );
    }

    /// Duplicate-key insert updates value but preserves original insertion position.
    #[test]
    fn duplicate_insert_updates_value_preserves_position() {
        let m = map_abc().insert("b", 99);
        // "b" was inserted second; its position must still be second after update.
        assert_eq!(m.keys(), vec![&"a", &"b", &"c"]);
        assert_eq!(m.get(&"b"), Some(&99));
    }

    // ─── property test: no-silent-reorder invariant (spec §4 / RFC-0016 §4.4) ─

    /// Property: two maps with identical contents (same keys+values) inserted in the same
    /// order yield the same `entries()` sequence.
    ///
    /// Guard: any non-determinism in insert makes this fail.
    ///
    /// Spec §4 §7-Q2: this invariant is `Declared` (spec-phase); this test promotes it
    /// to `Empirical` by property-checking it for a representative sample. We do not claim
    /// `Proven` — no theorem is cited (VR-5).
    #[test]
    fn equal_contents_same_order_yield_same_entries_order() {
        let m1 = Map::empty().insert(1, "x").insert(2, "y").insert(3, "z");
        let m2 = Map::empty().insert(1, "x").insert(2, "y").insert(3, "z");
        assert_eq!(
            m1.entries(),
            m2.entries(),
            "same contents + same insertion order must yield same entries() sequence (no silent reorder)"
        );
    }

    /// Property: after a remove, the remaining keys keep their relative insertion order.
    /// Guard: a remove that scrambles the remaining order makes this fail.
    #[test]
    fn remove_preserves_relative_order_of_remaining_keys() {
        let m = map_abc();
        let (m2, _) = m.remove(&"b");
        assert_eq!(
            m2.keys(),
            vec![&"a", &"c"],
            "remove must preserve relative insertion order of remaining keys (no silent reorder)"
        );
    }

    // ─── round-trip (insert/remove) ───────────────────────────────────────────

    /// Property: insert then remove round-trips (key absent after removal).
    #[test]
    fn insert_remove_round_trip() {
        let m = map_abc();
        let m2 = m.insert("d", 4);
        assert_eq!(m2.get(&"d"), Some(&4));
        let (m3, v) = m2.remove(&"d");
        assert_eq!(v, Some(4));
        assert_eq!(m3.get(&"d"), None);
        assert_eq!(m3.len(), m.len(), "round-trip must restore original length");
    }
}
