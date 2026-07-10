//! **Diagnostic routes → RFC-0008 observability sinks** (RFC-0013 §8, closed v0 set; RFC-0008 §4.8
//! "the diagnostic stream lives in RFC-0008").
//!
//! RFC-0013 left `route` a free-form string and deferred *which* targets exist and *how* they compose
//! with the runtime's observability sinks to "the RFC-0008 integration". This module closes that: a
//! **closed v0 [`Route`] vocabulary**, each bound to an RFC-0008 sink with an **honest delivery
//! guarantee** tagged on the lattice (RT5) — *never* an unguaranteed "fire and forget" claimed as
//! reliable, and *never* upgraded without a checked basis (VR-5).
//!
//! Two invariants this layer must not cross:
//! - **I1 (never-silent).** A route says *where a presentation goes*, never *whether the error
//!   propagates*. Resolution lives **outside** [`super::record::present`] (which already returns the
//!   error unchanged), so no route — not even [`Route::Null`] — can gate propagation. An *unknown*
//!   route is an explicit [`UnknownRoute`] error (surfaced, never silently dropped), not a missed
//!   delivery that hides the error.
//! - **RT5 (honest sink guarantees).** Each sink's [`Delivery`] carries the strongest claim that is
//!   *honest for v0*: in-process/durable/best-effort delivery is `Declared` (asserted by construction,
//!   not `Proven` absent a checked theorem); the mesh sink is *probabilistic* and carries a declared
//!   [`ProbabilityBound`](mycelium_core::BoundKind::Probability) δ; the null sink honestly reports
//!   **not delivered**. None claims reliability it has not earned.
//!
//! **Tooling layer only** — no kernel logging dependency (KC-3): this binds *names to honest
//! semantics*; the actual sink transports are the RFC-0008 runtime's, consumed through this contract.

use std::fmt;

use mycelium_core::{Bound, BoundBasis, BoundKind, GuaranteeStrength};

/// The **closed v0 set** of diagnostic routes (RFC-0013 §8). A `route` string resolves to exactly one
/// of these or is an explicit [`UnknownRoute`]; there is no open-ended free-form target in v0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Route {
    /// The in-process **diagnostic stream** — the default observability feed RFC-0008 hosts
    /// (RFC-0013 §4.8 Feeds). Synchronous in-process hand-off.
    Stream,
    /// The durable **representation-crossing audit view** (RFC-0013 §4.6) — persisted for later
    /// inspection / EXPLAIN.
    Audit,
    /// A best-effort textual **log** sink (may be redirected/dropped by the io layer).
    Log,
    /// The **null** sink: the presentation is intentionally discarded. Honest — it reports *not
    /// delivered*; it still does **not** gate propagation (I1: the error bubbles regardless).
    Null,
    /// The **mesh** sink: probabilistic gossip/pub-sub delivery across the runtime overlay
    /// (RFC-0008 §4.3, RT5). Carries a [`Delivery::Probabilistic`] δ; never claimed reliable.
    Mesh,
}

impl Route {
    /// The canonical route string (the on-the-wire/`PolicyFile` projection name).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Route::Stream => "stream",
            Route::Audit => "audit",
            Route::Log => "log",
            Route::Null => "null",
            Route::Mesh => "mesh",
        }
    }

    /// The closed v0 set, in declaration order (for enumeration / exhaustive tests).
    #[must_use]
    pub fn all() -> [Route; 5] {
        [
            Route::Stream,
            Route::Audit,
            Route::Log,
            Route::Null,
            Route::Mesh,
        ]
    }

    /// Resolve a `route` string to its [`Route`] — **checked against the closed v0 set** (the §4.5 X1
    /// "looked up, never evaluated" discipline applied to routes). An unrecognised route is an explicit
    /// [`UnknownRoute`], never a silent misroute.
    ///
    /// # Errors
    /// Returns [`UnknownRoute`] when `s` is not one of the closed v0 routes.
    pub fn resolve(s: &str) -> Result<Route, UnknownRoute> {
        Route::all()
            .into_iter()
            .find(|r| r.as_str() == s)
            .ok_or_else(|| UnknownRoute {
                route: s.to_owned(),
            })
    }

    /// The RFC-0008 sink this route binds to, with its **honest delivery guarantee** (RT5).
    #[must_use]
    pub fn binding(self) -> SinkBinding {
        let (sink, delivery) = match self {
            // Synchronous in-process hand-off to the diagnostic stream: delivered before `present`'s
            // consumer returns. Single process, bounded buffer with explicit backpressure — no loss
            // path in v0. Honest strength `Declared` (asserted; not `Proven` absent a checked basis).
            Route::Stream => ("rfc0008.diagnostic-stream", Delivery::Synchronous),
            // Durable append to the audit store: delivered + persisted. `Declared` (v0 store assumed
            // durable; a checked durability basis would upgrade it).
            Route::Audit => ("rfc0008.audit-store", Delivery::Durable),
            // Best-effort textual log: the io layer may redirect/drop it. `Declared`, explicitly
            // best-effort — never claimed reliable.
            Route::Log => ("rfc0008.text-log", Delivery::BestEffort),
            // The null sink: intentionally discarded. Honestly *not delivered* — never a "fire and
            // forget" masquerading as reliable (RT5). Propagation is unaffected (I1).
            Route::Null => ("rfc0008.null", Delivery::Discarded),
            // The mesh overlay: probabilistic gossip delivery (RT5). v0 carries a *declared* δ — no
            // measured convergence yet, so `Declared`, never upgraded (VR-5). A deployed epidemic
            // protocol upgrades the basis to Empirical/Proven per T4.2 (RFC-0008 §4.3).
            Route::Mesh => (
                "rfc0008.mesh",
                Delivery::Probabilistic {
                    // A declared failure-probability placeholder; honest basis = UserDeclared (no
                    // measured trials in v0). The structure is the real `ProbabilityBound`.
                    bound: Bound {
                        kind: BoundKind::Probability { delta: 0.01 },
                        basis: BoundBasis::UserDeclared,
                    },
                },
            ),
        };
        SinkBinding {
            route: self,
            sink,
            delivery,
        }
    }
}

impl fmt::Display for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A `route` string that is not in the closed v0 [`Route`] set — an explicit configuration error
/// (never silently misrouted; I1 — and the error it would have presented still propagates).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownRoute {
    /// The unrecognised route string.
    pub route: String,
}

impl fmt::Display for UnknownRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let known: Vec<&str> = Route::all().iter().map(|r| r.as_str()).collect();
        write!(
            f,
            "unknown diagnostic route {:?}: not in the closed v0 set {{{}}} (RFC-0013 §8) — an explicit \
             configuration error, never a silent misroute",
            self.route,
            known.join(", ")
        )
    }
}

impl std::error::Error for UnknownRoute {}

/// The **honest delivery semantics** of a sink (RT5). The lattice tag a delivery may *claim* is bounded
/// by what is honest for v0; [`guarantee`](Delivery::guarantee) reads it off, never upgraded (VR-5).
#[derive(Debug, Clone, PartialEq)]
pub enum Delivery {
    /// In-process, synchronous: handed to the sink before the presenting call returns. No loss path in
    /// v0. Honest strength `Declared`.
    Synchronous,
    /// Durable append (delivered + persisted). Honest strength `Declared`.
    Durable,
    /// Best-effort: may be dropped/redirected by the io layer. Honest strength `Declared`, explicitly
    /// best-effort.
    BestEffort,
    /// Intentionally discarded (the null sink): **not delivered**, and honestly so.
    Discarded,
    /// Probabilistic gossip delivery (mesh): carries a declared [`ProbabilityBound`] δ.
    ///
    /// [`ProbabilityBound`]: mycelium_core::BoundKind::Probability
    Probabilistic {
        /// The declared failure-probability bound (v0: `UserDeclared` basis — no measured convergence).
        bound: Bound,
    },
}

impl Delivery {
    /// Whether the sink actually **delivers** the presentation. The null sink does not — and says so;
    /// callers that need a delivery must not route to it expecting one (RT5).
    #[must_use]
    pub fn delivers(&self) -> bool {
        !matches!(self, Delivery::Discarded)
    }

    /// The honest **delivery guarantee** on the lattice (RT5/VR-5): `None` for the null sink (nothing
    /// delivered, so no delivery claim), `Some(strength)` otherwise. v0 never exceeds `Declared` — a
    /// stronger claim requires a *checked* basis (a proven no-loss property, or measured convergence).
    #[must_use]
    pub fn guarantee(&self) -> Option<GuaranteeStrength> {
        match self {
            Delivery::Discarded => None,
            Delivery::Synchronous
            | Delivery::Durable
            | Delivery::BestEffort
            | Delivery::Probabilistic { .. } => Some(GuaranteeStrength::Declared),
        }
    }

    /// The probabilistic delivery bound, if this is a probabilistic sink (the mesh δ; RT5).
    #[must_use]
    pub fn probability_bound(&self) -> Option<&Bound> {
        match self {
            Delivery::Probabilistic { bound } => Some(bound),
            _ => None,
        }
    }
}

/// A resolved binding of a [`Route`] to its RFC-0008 sink and the sink's honest [`Delivery`] guarantee.
#[derive(Debug, Clone, PartialEq)]
pub struct SinkBinding {
    /// The route.
    pub route: Route,
    /// The RFC-0008 sink identifier this route delivers to.
    pub sink: &'static str,
    /// The sink's honest delivery semantics (RT5).
    pub delivery: Delivery,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_v0_route_set_is_closed_and_round_trips() {
        for r in Route::all() {
            assert_eq!(Route::resolve(r.as_str()).unwrap(), r);
        }
        // Anything outside the closed set is an explicit error, never a silent misroute (I1).
        let err = Route::resolve("diagnostics_channel").unwrap_err();
        assert_eq!(err.route, "diagnostics_channel");
        assert!(err.to_string().contains("closed v0 set"));
    }

    #[test]
    fn every_sink_guarantee_is_honest_and_never_upgraded() {
        // RT5/VR-5: no sink claims more than `Declared` in v0; the null sink claims *nothing* (it does
        // not deliver, and says so); the mesh sink is probabilistic and carries a real δ bound.
        for r in Route::all() {
            let b = r.binding();
            match b.delivery.guarantee() {
                Some(g) => assert_eq!(
                    g,
                    GuaranteeStrength::Declared,
                    "{r} must not over-claim its delivery guarantee (VR-5)"
                ),
                None => assert!(
                    !b.delivery.delivers(),
                    "only a non-delivering sink may have no guarantee"
                ),
            }
        }
        // The mesh sink carries a well-formed, declared ProbabilityBound (RT5).
        let mesh = Route::Mesh.binding();
        let bound = mesh
            .delivery
            .probability_bound()
            .expect("mesh is probabilistic");
        assert!(bound.well_formed());
        assert!(matches!(bound.kind, BoundKind::Probability { .. }));
        // The null sink honestly reports non-delivery.
        assert!(!Route::Null.binding().delivery.delivers());
    }
}
