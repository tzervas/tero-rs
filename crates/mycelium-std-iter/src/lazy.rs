//! [`Lazy<E>`] — the explicitly-named, type-segregated demand-driven surface (spec §3/§4/Q4).
//!
//! A `Lazy<E>` is a potentially-unbounded source. It is **NOT total** and is tagged
//! [`Declared`](mycelium_core::GuaranteeStrength::Declared) — the unboundedness is *asserted
//! and flagged*, never proven away. It is a distinct named type; there is no implicit coercion
//! from [`Foldable`] to `Lazy<E>` (C1 / C4).
//!
//! The only sanctioned way to turn a `Lazy<E>` back into a total, foldable value is
//! [`lazy_take`](crate::lazy_take) — an explicit `Nat` bound that re-establishes the totality
//! guarantee (spec §3, the bridge sentence; spec §4 `lazy_take` row).
//!
//! # FLAG (spec §7-Q4)
//! Whether the `Lazy` surface belongs in `iter` or its own module is an open question. It is
//! kept here, type- and name-segregated, with a `Declared` tag. Revisit if the surface grows.
//!
//! # Guarantee tag: `Declared` (not total — the source may be unbounded)
//! The `Declared` tag is the honest floor (VR-5): we assert the source *may not terminate*
//! without a bound, but we cannot prove it terminates for a given step function `f`. This is
//! the module's single place where totality is not preserved, and it is named, typed, and
//! tagged accordingly (spec §4 "Tag justification").

use crate::Foldable;

/// A demand-driven, potentially-unbounded sequence.
///
/// Constructed via [`Lazy::unfold`]. To obtain a finite, total [`Foldable<E>`], use
/// [`lazy_take`](crate::lazy_take).
///
/// # Guarantee tag: `Declared` — NOT total (VR-5 downgrade; spec §4)
pub struct Lazy<E> {
    /// The step function: `State → Option<(Element, NewState)>`. `None` signals the source is
    /// exhausted; otherwise the next element and state are returned. The step function is stored
    /// as a boxed closure so the state type is erased (avoiding a generic type parameter on
    /// `Lazy` that would complicate the public API).
    ///
    /// # Design note
    /// The state erasure is the only place allocation is introduced by the `Lazy` type. The
    /// combinator implementations (map, filter, …) are all over `Foldable` and do not touch
    /// this box. No `unsafe` is used.
    #[allow(clippy::type_complexity)]
    step: Box<dyn FnMut() -> Option<E>>,
}

impl<E: 'static> Lazy<E> {
    /// Construct a `Lazy<E>` from an initial state `s` and a step function `f`.
    ///
    /// `f(state)` returns `Some((element, next_state))` to yield an element and advance the
    /// state, or `None` to signal exhaustion.
    ///
    /// # Guarantee tag: `Declared` — not total
    /// The resulting `Lazy` may produce infinitely many elements if `f` never returns `None`.
    /// The `Declared` tag is the honest floor (spec §4; VR-5).
    ///
    /// # EXPLAIN: the lazy/declared artifact
    /// Every `Lazy<E>` value carries an implicit "lazy / may-not-terminate" `Declared`
    /// artifact visible in the type: it is a `Lazy<E>`, not a `Foldable<E>`. The type name
    /// itself is the EXPLAIN record (C3 / spec §5 C3).
    pub fn unfold<S: 'static>(init: S, mut f: impl FnMut(S) -> Option<(E, S)> + 'static) -> Self {
        let mut state = Some(init);
        Lazy {
            step: Box::new(move || {
                let s = state.take()?;
                match f(s) {
                    Some((elem, next_s)) => {
                        state = Some(next_s);
                        Some(elem)
                    }
                    None => None,
                }
            }),
        }
    }

    /// Consume `self`, driving the source for up to `n` elements.
    ///
    /// This is called by [`lazy_take`](crate::lazy_take) and is the **only** sanctioned way to
    /// re-establish totality over a `Lazy` source (the `Nat` bound makes it terminating).
    ///
    /// # Guarantee tag: `Exact` — total given the bound (spec §4 `lazy_take` row)
    #[must_use]
    pub fn take(mut self, n: usize) -> Foldable<E> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            match (self.step)() {
                Some(e) => result.push(e),
                None => break,
            }
        }
        Foldable::from_vec(result)
    }
}

// `Lazy<E>` is intentionally NOT `Clone` or `Copy`: it owns a stateful step function.
// (This is by design — demand-driven steps are not generally idempotent.)
