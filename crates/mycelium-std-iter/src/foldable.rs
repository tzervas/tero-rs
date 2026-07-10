//! [`Foldable<E>`] — the concrete finite linear-recursive value type for `std.iter`.
//!
//! This is the stand-in for the RFC-0007 §4.8 spine shape (`nil | cons(E, T)`) until the
//! trait/typeclass story lands (RFC-0007 r4 deferred). Backed by `Vec<E>`, it is:
//! - **Finite and acyclic** by construction (a `Vec` is always bounded).
//! - **Value-semantic**: `clone()` produces an independent copy (C4).
//! - **`#[forbid(unsafe_code)]` clean**.
//!
//! # FLAG (spec §7-Q2 / RFC-0007 r4 deferred traits)
//! When the RFC-0007 trait story lands, `Foldable` must be generalised to a trait or replaced
//! by a trait bound. The combinator implementations in `lib.rs` must not change; only this
//! abstraction boundary does.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Foldable<E> {
    inner: Vec<E>,
}

impl<E> Foldable<E> {
    /// Construct a `Foldable` from a `Vec<E>`.
    #[must_use]
    pub fn from_vec(v: Vec<E>) -> Self {
        Foldable { inner: v }
    }

    /// An empty `Foldable` (the `nil` spine).
    #[must_use]
    pub fn empty() -> Self {
        Foldable { inner: Vec::new() }
    }

    /// The number of elements (the spine length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` iff the spine is `nil`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrow the elements as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[E] {
        &self.inner
    }

    /// Consume `self` and return the backing `Vec<E>`.
    #[must_use]
    pub fn into_vec(self) -> Vec<E> {
        self.inner
    }
}
