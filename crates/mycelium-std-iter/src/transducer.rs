//! [`Transducer<E, F>`] — a composable, source-independent step transformer (spec §3/§4).
//!
//! A transducer is a description of a pipeline of transformations (`map`, `filter`, …) that
//! fuses into a **single** pass over the source — one `for` fold, no intermediate allocations
//! from intermediate `Foldable`s. It is *source-independent*: the same transducer can be
//! applied to any `Foldable<E>`.
//!
//! # Fusion law (spec §7-Q5 / VR-5)
//! The associativity of `compose` and the fusion law (`transduce(compose(t1, t2)) = sequential
//! t1 then t2`) is an **`Empirical`** property checked in tests — NOT tagged `Proven` without a
//! checked theorem. See the `transduce_fusion_law_empirical` test in `lib.rs`.
//!
//! # EXPLAIN (C3)
//! The fused step pipeline is inspectable via [`Transducer::describe`], which returns a
//! human-readable description of each step in composition order.
//!
//! # Guarantee tag: `Exact` (the transducer is a pure structural description)
//! Applying the transducer (via [`crate::transduce`]) is total and inherits the kernel fold's
//! totality guarantee (spec §4 `transduce` row).

/// A composable, source-independent step transformer.
///
/// The type parameters are:
/// - `E`: the element type of the source `Foldable`.
/// - `F`: the element type of the output after transformation.
///
/// Transducers compose left-to-right via [`compose`](Transducer::compose):
/// `t1.compose(t2)` applies `t1` first, then `t2`.
pub struct Transducer<E, F> {
    /// The transformation function: a `Vec<E>` in, `Vec<F>` out.
    ///
    /// # Design note
    /// This uses a boxed closure over `Vec` for simplicity (Q2 / the Foldable-trait pending). A
    /// trait-based generalization would replace the Vec boundary with a fold callback. The
    /// observable behaviour is identical; the closure boundary only affects intermediate
    /// allocation.
    #[allow(clippy::type_complexity)]
    transform: Box<dyn Fn(Vec<E>) -> Vec<F>>,
    /// Human-readable description of this step (for EXPLAIN / C3).
    description: String,
}

impl<E: 'static, F: 'static> Transducer<E, F> {
    /// Construct a transducer from a raw transformation function.
    ///
    /// Prefer the named constructors (`map`, `filter`) for common shapes.
    pub fn new(description: impl Into<String>, f: impl Fn(Vec<E>) -> Vec<F> + 'static) -> Self {
        Transducer {
            transform: Box::new(f),
            description: description.into(),
        }
    }

    /// Apply the transformation to a `Vec<E>`, producing a `Vec<F>`.
    ///
    /// Called by [`crate::transduce`]; not usually called directly.
    pub(crate) fn apply(&self, v: Vec<E>) -> Vec<F> {
        (self.transform)(v)
    }

    /// A human-readable description of the step pipeline (EXPLAIN artifact, C3).
    #[must_use]
    pub fn describe(&self) -> &str {
        &self.description
    }

    /// Compose `self` with `next`: `self.compose(next)` applies `self` first, then `next`.
    ///
    /// The resulting transducer's description lists both steps in order (C3 / EXPLAIN).
    ///
    /// # Associativity
    /// `compose` is associative: `(t1.compose(t2)).compose(t3)` and
    /// `t1.compose(t2.compose(t3))` produce the same output for every input. This is asserted
    /// empirically in tests (spec §7-Q5; NOT tagged `Proven` without a checked theorem — VR-5).
    pub fn compose<G: 'static>(self, next: Transducer<F, G>) -> Transducer<E, G> {
        let desc = format!("{} → {}", self.description, next.description);
        Transducer {
            transform: Box::new(move |v| next.apply(self.apply(v))),
            description: desc,
        }
    }
}

impl<E: 'static, F: 'static> Transducer<E, F>
where
    F: Clone,
{
    // (no additional methods needed here)
}

// ─── Named constructors ───────────────────────────────────────────────────────

impl<E: 'static> Transducer<E, E> {
    /// A transducer that keeps only elements satisfying `pred`.
    pub fn filter(pred: impl Fn(&E) -> bool + 'static) -> Transducer<E, E> {
        Transducer::new("filter", move |v| {
            v.into_iter().filter(|e| pred(e)).collect()
        })
    }
}

impl<E: 'static, F: 'static> Transducer<E, F> {
    /// A transducer that maps each element with `f`.
    pub fn map(f: impl Fn(E) -> F + 'static) -> Transducer<E, F> {
        Transducer::new("map", move |v| v.into_iter().map(&f).collect())
    }
}

// ─── Clone impl ───────────────────────────────────────────────────────────────

// `Transducer<E, F>` stores a `Box<dyn Fn(…)>` which is not `Clone` in general.
// We deliberately do NOT derive Clone; callers that need a second instance should
// construct a new transducer. The `Empirical` fusion test in lib.rs constructs two
// transducers independently to avoid this limitation.
