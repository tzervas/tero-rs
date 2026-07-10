//! The never-silent R2 residual ledger and refusal surface (M-963; DN-78 §3 B-3 / §4; G2).
//!
//! Every M-828-tail item that DN-78 §4 defers to a research spike or Phase II has exactly one
//! row here — construct, why deferred (the unmet prerequisite), and the tracker — plus a
//! [`require`] entry point that refuses with an explicit typed error. The deferral is thereby
//! **mechanized**: inspectable data with a regression guard (a [`DeferredR2`] variant without a
//! ledger row fails the completeness test in `src/tests/r2_residual.rs`), not prose.
//!
//! # Naming note (ADR-020 §5)
//!
//! The RFC-0008 §4.5 reserved vocabulary stays out of public **operation names** (the
//! guarantee-matrix test enforces this). This ledger *names the constructs it refuses* in its
//! **data** — a refusal must name what it refuses to be never-silent (G2) — which is the same
//! posture as the L1 parse-time teaching diagnostics. Nothing here activates a reserved
//! construct or claims one is available.
//!
//! # Guarantee tags (VR-5)
//!
//! - Refusal totality ([`require`] on every deferred item returns an explicit error, in
//!   Phase I): **`Exact`** — by construction ([`require`] is a total match; tested).
//! - Ledger completeness (one row per [`DeferredR2`] variant, tracker + basis non-empty):
//!   **`Exact`** — enforced by test over [`DeferredR2::ALL`].
//! - The *deferral decisions themselves* are `Declared` (DN-78 §4/§5 — grounded in DN-63's
//!   prerequisite analysis, no formal dependency proof).

/// A deferred M-828-tail item (DN-78 §4). Each variant carries exactly one [`ResidualRow`] in
/// [`RESIDUALS`] (completeness is tested, not assumed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferredR2 {
    /// R-1 — the gossip/pub-sub overlay (`mesh`).
    MeshOverlay,
    /// R-2 — the external-capability contract (`graft`).
    GraftCapability,
    /// R-3 — explicit cross-node value movement (`xloc`).
    XlocMovement,
    /// R-4 — the content-addressed checkpoint (`cyst`).
    CystCheckpoint,
    /// R-5ʹ — the L1 surface syntax for the capture/set surface built in
    /// [`crate::policy_mech`] (the runtime-side machinery is active; the language surface is
    /// not).
    CaptureSetL1Surface,
    /// R-6 — `forage`/`backbone` maturity: multi-node candidate sets, the full node-signal
    /// inventory (DN-63 FLAG-13), real transport paths and the promotion mechanism
    /// (DN-63 FLAG-16).
    MultiNodePlacement,
}

impl DeferredR2 {
    /// Every deferred item, for exhaustive iteration in tests and tooling.
    pub const ALL: [DeferredR2; 6] = [
        DeferredR2::MeshOverlay,
        DeferredR2::GraftCapability,
        DeferredR2::XlocMovement,
        DeferredR2::CystCheckpoint,
        DeferredR2::CaptureSetL1Surface,
        DeferredR2::MultiNodePlacement,
    ];
}

/// One ledger row: what is deferred, why, and where it is tracked (DN-78 §4; G2 — the residual
/// is explicit data, not a silent gap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualRow {
    /// The deferred item.
    pub item: DeferredR2,
    /// The reserved construct(s) or surface the row refuses (names the RFC-0008 §4.5
    /// vocabulary — see the module-level naming note).
    pub construct: &'static str,
    /// The unmet prerequisite that makes building it now a guess (G2/VR-5).
    pub why_deferred: &'static str,
    /// The tracking task id(s) — the residual is tracked, never dropped.
    pub tracker: &'static str,
    /// The decision basis (DN-78 §4 row; DN-63 §).
    pub basis: &'static str,
}

/// The residual ledger — one row per [`DeferredR2`] variant, in [`DeferredR2::ALL`] order
/// (completeness and order are tested).
pub static RESIDUALS: &[ResidualRow] = &[
    ResidualRow {
        item: DeferredR2::MeshOverlay,
        construct: "mesh",
        why_deferred: "needs the DN-61 B.1 (clock) + B.2 (Byzantine) research passes, the v0 \
                       gossip-protocol choice (DN-63 FLAG-4), and a committed ProbabilityBound",
        tracker: "M-913 (research) + M-828 (remainder)",
        basis: "DN-78 §4 R-1; DN-63 §3.2/§4/§5",
    },
    ResidualRow {
        item: DeferredR2::GraftCapability,
        construct: "graft",
        why_deferred: "needs the RFC-0028 §7 capability follow-on (DN-63 FLAG-10/11/12 open)",
        tracker: "M-828",
        basis: "DN-78 §4 R-2; DN-63 §3.4",
    },
    ResidualRow {
        item: DeferredR2::XlocMovement,
        construct: "xloc",
        why_deferred: "needs mesh (carrier) and graft (capability check) at least Accepted, \
                       plus the wire-format swap story (DN-63 FLAG-1/2/3)",
        tracker: "M-828",
        basis: "DN-78 §4 R-3; DN-63 §3.1/§4",
    },
    ResidualRow {
        item: DeferredR2::CystCheckpoint,
        construct: "cyst",
        why_deferred: "needs xloc (mobility) and the RFC-0027 OQ-3 reclamation-in-dormancy \
                       resolution (DN-63 FLAG-7/8/9)",
        tracker: "M-828",
        basis: "DN-78 §4 R-4; DN-63 §3.3",
    },
    ResidualRow {
        item: DeferredR2::CaptureSetL1Surface,
        construct: "capture/set surface syntax (rides forage per DN-63 §3.5)",
        why_deferred: "needs the serial l1 lane and an implementation-RFC vehicle (RFC-0008 \
                       §4.5 status rule); the runtime-side machinery IS active (policy_mech)",
        tracker: "M-828",
        basis: "DN-78 §4 R-5ʹ / §3; DN-70 R-5",
    },
    ResidualRow {
        item: DeferredR2::MultiNodePlacement,
        construct: "forage/backbone maturity (multi-node)",
        why_deferred: "the multi-node candidate set comes from the mesh overlay (deferred R-1); \
                       the promotion mechanism belongs to the backbone implementation RFC \
                       (M-825 resolution; DN-63 FLAG-13/FLAG-16 open)",
        tracker: "M-828",
        basis: "DN-78 §4 R-6; DN-63 §3.5/§3.6",
    },
];

/// The ledger row for `item`. Total by construction (a match over the variant), and the
/// row's `item` field round-trips (tested).
#[must_use]
pub fn residual_for(item: DeferredR2) -> &'static ResidualRow {
    let idx = match item {
        DeferredR2::MeshOverlay => 0,
        DeferredR2::GraftCapability => 1,
        DeferredR2::XlocMovement => 2,
        DeferredR2::CystCheckpoint => 3,
        DeferredR2::CaptureSetL1Surface => 4,
        DeferredR2::MultiNodePlacement => 5,
    };
    &RESIDUALS[idx]
}

/// The explicit refusal a deferred item's [`require`] returns (G2: typed, inspectable, and
/// teaching — it names the construct, the unmet prerequisite, and the tracker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R2DeferredError {
    /// The item that was required.
    pub item: DeferredR2,
    /// Its ledger row (why + tracker + basis).
    pub row: &'static ResidualRow,
}

impl core::fmt::Display for R2DeferredError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "'{}' is not built in Phase I: {} — tracked as {} ({})",
            self.row.construct, self.row.why_deferred, self.row.tracker, self.row.basis
        )
    }
}

impl std::error::Error for R2DeferredError {}

/// The refusal entry point: a runtime path that would need a deferred construct calls this and
/// gets an explicit typed error — never a silent no-op or fallback (G2).
///
/// In Phase I this refuses for **every** [`DeferredR2`] item (guarantee: **`Exact`** — total by
/// construction; tested over [`DeferredR2::ALL`]). The signature is the stable contract: when a
/// construct activates through its own vehicle (DN-78 §4 trackers), its arm returns `Ok(())`
/// and its ledger row is retired append-only.
pub fn require(item: DeferredR2) -> Result<(), R2DeferredError> {
    Err(R2DeferredError {
        item,
        row: residual_for(item),
    })
}
