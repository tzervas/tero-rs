//! `std.collections` — Ring 2 / Tier B value-semantic persistent collections (M-511).
//!
//! The ordinary collection surface: an immutable persistent **`Seq`** (indexed sequence),
//! an immutable persistent **`Map`** (key→value), and an immutable persistent **`Set`**
//! (membership), each made value-semantic, held to the §4.1 contract (C1–C6), and shipped
//! with a per-op guarantee matrix (RFC-0016 §4.5; [`guarantee_matrix::MATRIX`]).
//!
//! # Honesty crux (two structural halves, spec §1)
//!
//! 1. **No silent reorder.** Each structure iterates in a *documented, stable order* that
//!    is a property of the type, never an exposed hash-bucket order. A rehash/rebalance is
//!    an internal operation with **no** observable reorder (RFC-0016 §4.4). Iteration order
//!    is:
//!    - `Seq` — **index order** (0, 1, …, len-1).
//!    - `Map` / `Set` — **insertion order** (first insertion of each key/element).
//!
//! 2. **Out-of-bounds / missing-key is explicit.** `Seq::get`/`first`, `Map::get` → `None`;
//!    `Seq::pop` → `None` on empty; `Seq::update`/`slice` → `Err(IndexOOB)`. Never a
//!    silent default, a clamp, or a sentinel (C1 / G2).
//!
//! # Module boundary (spec §2)
//!
//! - **In scope:** `Seq<E>`, `Map<K,V>`, `Set<E>`, their value-semantic op surfaces, and
//!   the *non-identity* hashing-for-buckets used internally by `Map`/`Set`. The order
//!   contract is a first-class promise.
//!
//! - **Out of scope (and who owns it):**
//!   - *Identity / content-addressing hashing* — `std.content` (M-523). The bucketing hash
//!     used internally is **not** `std.content`'s canonical `ContentHash` (ADR-003; README §5
//!     seam). The two are kept strictly distinct.
//!   - *Iteration / fold combinators* (`map`/`filter`/`fold`/…) — `std.iter` (M-526). This
//!     crate exposes each structure as a `foldable()` slice/sequence in documented order;
//!     `iter` owns the traversal vocabulary.
//!   - *Ordering / equality semantics* — `std.cmp` (M-532). `collections` parameterises over
//!     `Hash`/`Eq` for its bucket-based structures but does not define comparison.
//!   - *Representation change* — `std.swap` (M-516). No op silently changes a `Repr`.
//!
//! # Guarantee matrix
//! Every exported op has a row in [`guarantee_matrix::MATRIX`] (RFC-0016 §4.5 / spec §4).
//! All ops are `Exact` (no accuracy semantics — RFC-0016 C2), pure (`effects: none[*]`),
//! and either total or honestly fallible. The matrix is asserted in tests, not prose-only
//! (C2 / VR-5).
//!
//! # C1 — never-silent
//! - `Seq::get` / `first` / `pop` / `Map::get` → `None` for absence.
//! - `Seq::update` / `slice` → `Err(`[`CollErr::IndexOOB`]`)` for bad bounds.
//! - `Map::remove` → `(new_map, None)` when key absent — explicit, never a silent no-op.
//! - No op silently reorders elements (C1 on order — spec §5 C1).
//!
//! # C4 — content-addressed, value-semantic
//! Every structure is an immutable value; every "update" returns a new value. Structural
//! sharing (when present) is an implementation detail invisible to identity. The
//! *bucketing* hash used by `Map`/`Set` internally is **not** content identity (ADR-003).
//!
//! # C6 — pure
//! Every op declares `effects: none` or `none*` (bounded allocation for a new persistent
//! node — intrinsic, budget-free at this layer). No IO, no clock, no randomness.
//!
//! # Open questions (FLAGs carried from spec §7)
//!
//! - **(Q1) The ordered-vs-bucketed default for `Map`/`Set` — M-501's to ratify.**
//!   This implementation uses **insertion order**, which satisfies the no-silent-reorder
//!   crux. If M-501 ratifies a different order (e.g. `cmp`-sorted), this impl changes;
//!   the property tests check order stability and will catch any regression.
//!
//! - **(Q2) The no-silent-reorder invariant's tag.**
//!   The matrix records the documented-order property as a spec-`Declared` invariant
//!   that is promoted to `Empirical` by the property tests in each module. It is **not**
//!   claimed `Proven` (no checked theorem — VR-5).
//!
//! - **(Q3) Whole-collection identity via `std.content`.**
//!   C4 relies on a collection's identity being the content-addressed digest of its
//!   normalized structure (`std.content`, M-523). That integration is deferred until
//!   `std.content`'s `hash_of_value` entry point for composite values is ratified
//!   (content §7-Q2). Collections currently satisfy C4 via structural equality (`PartialEq`);
//!   the content-hash integration is a follow-on.
//!
//! - **(Q4) Bucketing-hash seed.**
//!   The `HashMap` / `HashSet` internals use Rust's default hasher (deterministic in a
//!   single process). For cross-process determinism or HashDoS mitigation, a declared-seed
//!   (RT3 / RFC-0014) bucketer would replace the default. FLAGGED: any seeded hash must
//!   declare its seed acquisition as an explicit effect (C6) — never ambient entropy.
//!
//! - **(FLAG: iter dependency)**
//!   `std.iter` (M-526) is a sibling P5-B crate that is not yet implemented. `collections`
//!   exposes `foldable()` as a slice/vector (the simplest inspectable sequence source) so
//!   `iter` can consume it once it lands. No `mycelium-std-iter` dep is added here; the
//!   integration is a follow-on.
//!
//! - **(FLAG: workspace Cargo.toml)**
//!   The orchestrator scaffold did not include `mycelium-std-collections` in the workspace
//!   `Cargo.toml`; this leaf added it. FLAG for the orchestrator to reconcile.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Collections are representation-agnostic containers —
//! a `Seq<E>` stores elements of whatever `Repr` the caller provides; no element representation
//! is coerced by the container. The bucketing hash used by `Map`/`Set` internally is not a
//! representation operation (ADR-003 — it is not `ContentHash`); no `Repr` is changed or inferred
//! by collection operations. Representation change of elements requires an explicit `std.swap`.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/collections.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; the same-named `lib/std/collections.myc` prototype is a narrower, structurally distinct surface (DN-66 S3.1) — the D6 retirement trigger has not fired, so no item here is `#[deprecated]`.

#![forbid(unsafe_code)]

pub mod error;
pub mod guarantee_matrix;
pub mod map;
pub mod seq;
pub mod set;

pub use error::CollErr;
pub use map::Map;
pub use seq::Seq;
pub use set::Set;

#[cfg(test)]
mod integration_tests {
    //! Integration-level tests that cross the `Seq`/`Map`/`Set` module boundaries.
    //!
    //! Per-module unit tests live in each `seq.rs` / `map.rs` / `set.rs`. These tests
    //! check the guarantee matrix itself and a few cross-module properties.

    use super::*;
    use guarantee_matrix::{Explainable, Fallibility, MATRIX};

    // ─── Guarantee matrix cross-checks ───────────────────────────────────────

    /// Every row in the guarantee matrix is `Exact` and effect-free.
    /// Guard: any upgrade to Proven/Empirical/Declared makes this fail.
    #[test]
    fn guarantee_matrix_all_rows_are_exact_and_pure() {
        assert!(
            !MATRIX.is_empty(),
            "guarantee matrix must be non-empty (spec §4 lists 25 rows)"
        );
        for row in MATRIX {
            assert_eq!(
                row.guarantee, "Exact",
                "op {:?} must be Exact — no accuracy semantics (RFC-0016 C2)",
                row.op
            );
            assert!(
                row.effects == "none" || row.effects == "none*",
                "op {:?} must be effect-free (C6), got {:?}",
                row.op,
                row.effects
            );
        }
    }

    /// Every fallible op in the matrix has a non-empty error set (C1).
    #[test]
    fn guarantee_matrix_fallible_ops_name_their_error_set() {
        for row in MATRIX {
            if row.fallibility == Fallibility::Fallible {
                assert!(
                    !row.error_set.is_empty(),
                    "fallible op {:?} must name its error set (C1)",
                    row.op
                );
            }
        }
    }

    /// EXPLAIN-able ops are the decision-bearing ones (refusal records, documented order).
    #[test]
    fn guarantee_matrix_explainable_count_matches_spec() {
        let count = MATRIX
            .iter()
            .filter(|r| r.explainable == Explainable::Yes)
            .count();
        // Spec §4: update(1), slice(1), get_or(1), keys/values/entries(3),
        //          union/intersection/difference(3) = 9 EXPLAIN-able ops.
        assert_eq!(
            count, 9,
            "expected 9 EXPLAIN-able ops (spec §4); got {count}"
        );
    }

    // ─── Cross-structure value-semantics check ───────────────────────────────

    /// A `Seq` of `Map`s round-trips: push a map, pop it, verify identity.
    /// Demonstrates that `collections` types compose cleanly (KC-3 composition).
    #[test]
    fn seq_of_maps_composes() {
        let m1: Map<&str, i32> = Map::empty().insert("x", 1);
        let m2: Map<&str, i32> = Map::empty().insert("y", 2);
        let s: Seq<Map<&str, i32>> = Seq::empty().push(m1.clone()).push(m2.clone());
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), Some(&m1));
        assert_eq!(s.get(1), Some(&m2));
        let (s2, popped) = s.pop().expect("non-empty");
        assert_eq!(popped, m2, "pop must return m2 (last pushed)");
        assert_eq!(s2.len(), 1);
    }

    /// A `Set` of `Seq`s: demonstrates elements can be complex values (KC-3).
    #[test]
    fn set_of_seqs_composes() {
        let s1: Seq<i32> = Seq::from_slice(&[1, 2]);
        let s2: Seq<i32> = Seq::from_slice(&[3, 4]);
        let set = Set::empty().insert(s1.clone()).insert(s2.clone());
        assert!(set.contains(&s1));
        assert!(set.contains(&s2));
        assert!(!set.contains(&Seq::from_slice(&[99])));
    }
}
