//! Per-operation guarantee matrix for `std.runtime` (ADR-020 v0 / SC-2 / VR-5).
//!
//! Every exported operation has an entry here. The matrix is asserted in tests, not
//! prose-only: any tag upgrade requires a checked theorem (VR-5) and a test update.

use mycelium_core::GuaranteeStrength;

/// One row in the guarantee matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaugeRow {
    pub operation: &'static str,
    pub strength: GuaranteeStrength,
    pub basis: &'static str,
}

/// Per-operation guarantee matrix for `std.runtime` v0.
///
/// Grounding: ADR-020 §4; RFC-0008 RT2 (sequentialization + Kahn-determinism differentials).
pub static MATRIX: &[GaugeRow] = &[
    GaugeRow {
        operation: "Scope::new",
        strength: GuaranteeStrength::Exact,
        basis: "constructor is trivially correct (no approximation)",
    },
    GaugeRow {
        operation: "Scope join semantics (all tasks complete before exit)",
        strength: GuaranteeStrength::Empirical,
        basis: "RT2 sequentialization differential; Kahn-determinism not yet Proven (ADR-020 §4)",
    },
    GaugeRow {
        operation: "Colony::new",
        strength: GuaranteeStrength::Exact,
        basis: "constructor is trivially correct",
    },
    GaugeRow {
        operation: "Colony Kahn-determinism (channel-mediated communication)",
        strength: GuaranteeStrength::Empirical,
        basis: "RT2 Kahn-determinism differential; formal proof pending (ADR-020 §4 FLAG)",
    },
    GaugeRow {
        operation: "Task purity contract",
        strength: GuaranteeStrength::Declared,
        basis: "asserted by caller; type system cannot enforce (VR-5: not upgraded without a checked basis)",
    },
    GaugeRow {
        operation: "TaskCtx::is_cancelled",
        strength: GuaranteeStrength::Exact,
        basis: "reads a boolean flag set by scope cancellation",
    },
    GaugeRow {
        operation: "Poll",
        strength: GuaranteeStrength::Exact,
        basis: "enum variant is the exact poll result",
    },
    GaugeRow {
        operation: "SweepOrder determinism",
        strength: GuaranteeStrength::Exact,
        basis: "sweep order is deterministic given the same queue state",
    },
    GaugeRow {
        operation: "Deadlock detection (DAG channels)",
        strength: GuaranteeStrength::Empirical,
        basis: "complete for DAG channel graphs; cyclic graphs FLAG (ADR-020 §7)",
    },
    GaugeRow {
        operation: "Sender::try_send / single-channel FIFO",
        strength: GuaranteeStrength::Exact,
        basis: "FIFO ordering within one channel is exact by construction",
    },
    GaugeRow {
        operation: "Receiver::try_recv / single-channel FIFO",
        strength: GuaranteeStrength::Exact,
        basis: "FIFO ordering within one channel is exact by construction",
    },
    GaugeRow {
        operation: "Network Kahn-determinism (cross-channel)",
        strength: GuaranteeStrength::Empirical,
        basis: "RT2 Kahn-determinism differential; formal proof pending (ADR-020 §4)",
    },
    // ── Channel construction ops (added with the real channel implementation) ──
    GaugeRow {
        operation: "Network::channel (construction)",
        strength: GuaranteeStrength::Exact,
        basis: "constructor is trivially correct; backed by Arc<Mutex<VecDeque>> (ADR-020 §4)",
    },
    GaugeRow {
        operation: "Network::channel zero-capacity check",
        strength: GuaranteeStrength::Exact,
        basis: "fail-closed: ZeroCapacity is returned deterministically when capacity==0 (G2/ADR-020 §4)",
    },
    GaugeRow {
        operation: "Sender::try_send FIFO (bounded channel)",
        strength: GuaranteeStrength::Exact,
        basis: "push to VecDeque tail; FIFO ordering exact by construction (ADR-020 §4)",
    },
    GaugeRow {
        operation: "Receiver::try_recv FIFO (bounded channel)",
        strength: GuaranteeStrength::Exact,
        basis: "pop from VecDeque head; FIFO ordering exact by construction (ADR-020 §4)",
    },
    // ── E12-1 execution maturity (M-709 scheduler / M-711 deadlock / M-713 supervision) ──
    GaugeRow {
        operation: "Scheduler RT2 sequentialization differential (OS threads)",
        strength: GuaranteeStrength::Empirical,
        basis: "parallel run equals sequential reference by RT1; property-tested, not Proven (M-709)",
    },
    GaugeRow {
        operation: "Scheduler backpressure bound (bounded ready queue)",
        strength: GuaranteeStrength::Exact,
        basis: "ready queue ≤ capacity by construction (enqueue only while len<capacity); G2 (M-709)",
    },
    GaugeRow {
        operation: "Scheduler liveness (each job runs exactly once)",
        strength: GuaranteeStrength::Empirical,
        basis: "property-tested over random job sets; not Proven (M-709)",
    },
    GaugeRow {
        operation: "Deadlock-freedom sweep (run_dataflow no-progress)",
        strength: GuaranteeStrength::Empirical,
        basis: "no-progress sweep ⇒ explicit Deadlock (never a hang, G2); complete for DAG graphs (M-711)",
    },
    GaugeRow {
        operation: "Supervision cancellation propagation (structured scope)",
        strength: GuaranteeStrength::Empirical,
        basis: "cooperative cancel cascades to every child; explicit outcome per child; property-tested (M-713)",
    },
    GaugeRow {
        operation: "Supervision restart bound (bounded cascade)",
        strength: GuaranteeStrength::Exact,
        basis: "rate + total restart bounds enforced structurally; inherited from M-356 Supervisor (M-713)",
    },
    // ── M-861: per-worker-deque work-stealing scheduler ──
    GaugeRow {
        operation: "Scheduler RT2 sequentialization differential (work-stealing)",
        strength: GuaranteeStrength::Empirical,
        basis: "parallel run equals sequential reference under stealing; result-order-only claim, unaffected by execution reordering; property-tested over randomized worker/steal configurations, not Proven (M-861)",
    },
    GaugeRow {
        operation: "Scheduler backpressure bound (total pending across per-worker deques)",
        strength: GuaranteeStrength::Exact,
        basis: "total pending ≤ capacity by construction under one lock guarding every deque together; G2 (M-861)",
    },
    GaugeRow {
        operation: "Scheduler liveness under stealing (each job runs exactly once)",
        strength: GuaranteeStrength::Empirical,
        basis: "property-tested over random job sets and random worker/steal configurations; not Proven (M-861)",
    },
    GaugeRow {
        operation: "Steal-victim-selection policy determinism (RT3 EXPLAIN)",
        strength: GuaranteeStrength::Exact,
        basis: "StealPolicy::select_victim is a total, deterministic function of its inputs; every decision is an inspectable StealDecision record (M-861 / RFC-0008 RT3)",
    },
    // ── M-963: mechanized SelectionPolicy capture/set + the R2 residual ledger (DN-78 §3) ──
    GaugeRow {
        operation: "PolicySlot::set transition record (reified setter)",
        strength: GuaranteeStrength::Exact,
        basis: "every set appends exactly one PolicySetRecord with a per-slot monotonic seq and the outgoing policy's ref; by construction, never a silent override (G2 / M-963 / DN-78 B-2)",
    },
    GaugeRow {
        operation: "PolicySlot::select without an active policy (explicit refusal)",
        strength: GuaranteeStrength::Exact,
        basis: "fail-closed: NoActivePolicy is returned deterministically when no policy is set — no built-in silent default (G2 / ADR-006 / M-963)",
    },
    GaugeRow {
        operation: "Policy capture resolution (unknown ref is an explicit error)",
        strength: GuaranteeStrength::Exact,
        basis: "fail-closed by construction: UnknownPolicyRef/RefMismatch are explicit; a returned capture satisfies policy_ref() == requested, checked not assumed (G2 / ADR-006 / M-963)",
    },
    GaugeRow {
        operation: "Policy capture replay reaches the recorded decision",
        strength: GuaranteeStrength::Empirical,
        basis: "record-vs-replay differential property-tested over randomized policies/inputs; RFC-0005 select determinism has no mechanized theorem, so not Proven (VR-5 / M-963 / M-964)",
    },
    GaugeRow {
        operation: "Deferred-construct refusal (R2 residual ledger)",
        strength: GuaranteeStrength::Exact,
        basis: "require() on every deferred item returns an explicit typed error naming construct + prerequisite + tracker; one ledger row per item, enforced by test (G2 / M-963 / DN-78 B-3)",
    },
];
