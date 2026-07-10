//! Reclamation EXPLAIN/audit record — RFC-0027 §9 / MEM-1.
//!
//! Every reclamation event MUST be observable as a structured EXPLAIN record (RFC-0027 §9,
//! lane-B R-1, G2). This module is the **never-silent observability foundation** for the memory
//! model; the live trigger wiring (rc→0 / scope-exit / channel-close) is downstream (MEM-2/MEM-3).
//!
//! # Design placement decision (`Declared`)
//!
//! `ReclamationRecord` lives in `mycelium-std-runtime` (not `mycelium-core`) because:
//! - It references runtime concepts: `SweepOrder` (scope-tree sweep epoch), scope identity, and
//!   channel identity — all of which are runtime-tier concerns.
//! - `mycelium-core` is the value-level kernel (RFC-0001 §4.6); adding runtime bookkeeping there
//!   would violate SoC and grow the trusted base (KC-3).
//! - The `value_meta_hash` field uses `mycelium-core::ContentHash` (already a dep), which is the
//!   only value-level type needed here; the rest is runtime-local.
//!
//! # Guarantee tags
//!
//! The record **structure and field set** are `Declared` per RFC-0027 §8 and the build plan (MEM-1):
//! the design is normatively specified (RFC-0027 §9) but no property test or proof of trigger
//! *completeness* exists in-repo yet (that lands in MEM-2/MEM-3). The never-silent *contract*
//! (every constructed reclamation path routes through `ReclamationSink::emit`) is
//! **enforced-by-construction** in this module — `Exact` within the scope of what is built here.
//!
//! # FLAG — downstream trigger wiring (MEM-2/MEM-3)
//!
//! This module lands the RECORD TYPE + its construction + the TRIGGER ENUM + the NEVER-SILENT
//! EMIT CONTRACT (the `ReclamationSink` trait) + the EXPLAIN-inspectability interface.
//!
//! The **actual trigger wiring** — wiring `ReclamationSink::emit` into live rc-decrement / scope-exit
//! / channel-close paths — is **MEM-2 (RC cell + rc→0 reclamation) and MEM-3 (region/scope
//! batched reclamation)**. No claim is made that reclamation is "implemented" here; only that the
//! observability foundation is in place for the downstream wiring to emit into.
//!
//! FLAG: `ScopeId`, `ChannelId`, and `SweepEpoch` are `u64`-backed placeholder types. The
//! canonical scope-tree identity type (MEM-3) and channel identity type (network RFC follow-on)
//! will replace them; the field set and trait contract are stable.

use mycelium_core::ContentHash;

// ── Scope and channel identity ────────────────────────────────────────────────

/// A stable, opaque identifier for a runtime scope (RT7 scope tree node).
///
/// Guarantee: `Declared` — the identifier is a `u64` counter placeholder; the canonical
/// scope-tree identity from the runtime tier (RFC-0008 RT7) will be the settled type once the
/// region/scope machinery (MEM-3) lands. The `u64` is monotonic and unique within a process
/// lifetime, providing enough stability to anchor the audit record.
///
/// FLAG: replace with the canonical scope-tree identity type in MEM-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub u64);

/// A stable, opaque identifier for a channel (RFC-0027 §9, `channel_id`).
///
/// Guarantee: `Declared` — `u64` placeholder; the canonical network-tier channel identity
/// will be the settled type once the channel close wiring (MEM-3 / network follow-on) lands.
///
/// FLAG: replace with the canonical channel identity type from the network tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u64);

// ── Sweep epoch ───────────────────────────────────────────────────────────────

/// A monotonic epoch counter from the `SweepOrder` model (RFC-0008 §4.3).
///
/// Ties a reclamation event to the scheduling model's deterministic audit anchor.
///
/// Guarantee: `Declared` — the epoch counter is a `u64` monotonic placeholder;
/// the settled sweep-epoch type from the scheduler (MEM-3 integration) will replace it.
/// Monotonicity is asserted by construction: `ReclamationRecord::new` takes the epoch as a
/// caller-supplied value — the caller must ensure monotonicity; no global state here (KC-3).
///
/// FLAG: integrate with `SweepOrder` / the scheduler's epoch counter in MEM-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SweepEpoch(pub u64);

// ── Trigger enum (exhaustive — G2) ───────────────────────────────────────────

/// The reason a reclamation event fired — exhaustive over the three structural triggers
/// (RFC-0027 §9, `trigger ∈ {RcZero, ScopeExit, ChannelClose}`).
///
/// Exhaustiveness is the G2 ("never-silent") anchor for the *why* of reclamation:
/// every event names its cause; adding a new cause requires a new variant (the compiler
/// enforces exhaustive matching). The `#[non_exhaustive]` attribute is deliberately ABSENT
/// here: the enum IS exhaustive over the RFC-0027 §9 trigger set; callers must match all
/// three arms (G2 — no hidden, unmatched triggers).
///
/// Guarantee: `Declared` — the variant set is normatively specified (RFC-0027 §9).
/// The live trigger-wiring (match-arms emitting records on rc→0 / scope-exit / channel-close)
/// is downstream (MEM-2/MEM-3); the enum shape here is structural, not operational.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReclamationTrigger {
    /// The reference count for this value reached zero (deferred to scope exit, then freed).
    ///
    /// Structural anchor: the RC dec that took rc from 1 → 0 (RFC-0027 §9 / §10.1).
    /// Wiring: MEM-2 (the RC cell + rc→0 path).
    RcZero,

    /// A scope-tree node (RT7) exited, triggering batched reclamation of its owned values.
    ///
    /// Structural anchor: the scope-exit event in the RT7 scope tree (RFC-0027 §10.3).
    /// Wiring: MEM-3 (region/scope batched reclamation).
    ScopeExit,

    /// A channel was closed (disconnected), releasing ownership of values in transit.
    ///
    /// Structural anchor: the `ChannelClose` event from the affine channel protocol
    /// (RFC-0027 §7.3 — cross-hypha transfer rides the affine channel protocol).
    /// Wiring: MEM-3 / network-RFC follow-on.
    ChannelClose,
}

// ── ReclamationRecord — the §9 field set ─────────────────────────────────────

/// The EXPLAIN/audit record for a single reclamation event (RFC-0027 §9).
///
/// This is the **never-silent** observability artifact: a reclamation event that does NOT
/// yield a `ReclamationRecord` is a G2 violation. The `ReclamationSink` trait (below) enforces
/// this at the architecture level — every reclamation path must call `ReclamationSink::emit`.
///
/// ## Fields (RFC-0027 §9 field set — all five present)
///
/// - `scope_id`: which RT7 scope triggered reclamation.
/// - `sweep_epoch`: the `SweepOrder` epoch (RFC-0008 §4.3) — the deterministic audit anchor.
/// - `trigger`: `RcZero | ScopeExit | ChannelClose` (exhaustive, G2 — the record knows *why*).
/// - `value_meta_hash`: `mycelium-core::ContentHash` — the content identity of the reclaimed
///   value (ties the event to the value's provenance/guarantee history, RFC-0001 §4.6).
/// - `channel_id`: `Option<ChannelId>` — present for `ChannelClose` events; absent otherwise.
///
/// ## EXPLAIN-ability (RFC-0005)
///
/// The record is fully inspectable via `explain()` — which returns a human-readable summary
/// of all five fields. This extends the RFC-0005 EXPLAIN contract to reclamation events,
/// reusing the same "inspectable record → EXPLAIN output" pattern (KC-3 / DRY):
/// - *inputs considered*: `scope_id` + `sweep_epoch` (which scope, which epoch)
/// - *chosen option*: `trigger` (why reclamation fired — drop / scope-exit / channel-close)
/// - *content identity*: `value_meta_hash` (what was reclaimed)
/// - *channel boundary*: `channel_id` (for cross-boundary events)
///
/// ## Guarantee tags
///
/// The record structure is `Exact` by construction (all fields are present and typed; the
/// compiler enforces the invariant). The *operational* coverage — that every live reclamation
/// event actually emits a record — is `Declared`: it depends on the trigger wiring in
/// MEM-2/MEM-3, which does not exist yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReclamationRecord {
    /// RT7 scope whose exit or RC-zero triggered reclamation (RFC-0027 §9).
    pub scope_id: ScopeId,

    /// Monotonic epoch from the `SweepOrder` model (RFC-0008 §4.3) — the deterministic
    /// audit anchor tying reclamation to the scheduling model.
    pub sweep_epoch: SweepEpoch,

    /// The structural cause of reclamation — exhaustive over `{RcZero, ScopeExit, ChannelClose}`
    /// (RFC-0027 §9, G2 / never-silent).
    pub trigger: ReclamationTrigger,

    /// Content identity of the reclaimed value: `mycelium-core::ContentHash` tying the event
    /// to the value's provenance/guarantee history (RFC-0001 §4.6, RFC-0027 §9).
    pub value_meta_hash: ContentHash,

    /// Present for `ChannelClose` events; identifies which channel boundary the value crossed
    /// (RFC-0027 §9). `None` for `RcZero` and `ScopeExit`.
    pub channel_id: Option<ChannelId>,
}

impl ReclamationRecord {
    /// Construct a `ReclamationRecord` for a `RcZero` or `ScopeExit` event (no channel).
    ///
    /// The absence of `channel_id` is enforced by the API: callers of this constructor cannot
    /// accidentally supply a channel for a non-channel-close event. Use
    /// `ReclamationRecord::for_channel_close` for `ChannelClose` events.
    ///
    /// Guarantee: `Exact` (by construction — all fields are supplied; no approximation).
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `trigger == ChannelClose` (use `for_channel_close` instead).
    #[must_use]
    pub fn new(
        scope_id: ScopeId,
        sweep_epoch: SweepEpoch,
        trigger: ReclamationTrigger,
        value_meta_hash: ContentHash,
    ) -> Self {
        debug_assert!(
            !matches!(trigger, ReclamationTrigger::ChannelClose),
            "ChannelClose events must carry a channel_id — use ReclamationRecord::for_channel_close"
        );
        ReclamationRecord {
            scope_id,
            sweep_epoch,
            trigger,
            value_meta_hash,
            channel_id: None,
        }
    }

    /// Construct a `ReclamationRecord` for a `ChannelClose` event.
    ///
    /// The `channel_id` is mandatory for channel-close reclamation (RFC-0027 §9).
    ///
    /// Guarantee: `Exact` (by construction).
    #[must_use]
    pub fn for_channel_close(
        scope_id: ScopeId,
        sweep_epoch: SweepEpoch,
        value_meta_hash: ContentHash,
        channel_id: ChannelId,
    ) -> Self {
        ReclamationRecord {
            scope_id,
            sweep_epoch,
            trigger: ReclamationTrigger::ChannelClose,
            value_meta_hash,
            channel_id: Some(channel_id),
        }
    }

    /// EXPLAIN-ability (RFC-0005): return a human-readable summary of all five fields.
    ///
    /// Extends the RFC-0005 EXPLAIN contract to reclamation events:
    /// - *inputs considered*: `scope_id` + `sweep_epoch`
    /// - *chosen option* (trigger): `RcZero | ScopeExit | ChannelClose`
    /// - *content identity*: `value_meta_hash`
    /// - *channel boundary*: `channel_id` (for cross-boundary events, else absent)
    ///
    /// Guarantee: `Exact` — the explanation is a deterministic function of the record fields.
    /// The explanation format is `Declared` (unspecified by RFC-0005 for runtime records —
    /// the machine-readable fields are the normative interface; this is the human-readable aid).
    #[must_use]
    pub fn explain(&self) -> ExplainRecord {
        ExplainRecord {
            scope_id: self.scope_id,
            sweep_epoch: self.sweep_epoch,
            trigger: self.trigger.clone(),
            value_meta_hash: self.value_meta_hash.clone(),
            channel_id: self.channel_id,
        }
    }

    /// Return a reference to the trigger — the G2 "why" of this reclamation event.
    #[must_use]
    pub fn trigger(&self) -> &ReclamationTrigger {
        &self.trigger
    }

    /// Return the content-identity hash of the reclaimed value (RFC-0027 §9 / RFC-0001 §4.6).
    #[must_use]
    pub fn value_meta_hash(&self) -> &ContentHash {
        &self.value_meta_hash
    }
}

// ── EXPLAIN record (RFC-0005 extension) ──────────────────────────────────────

/// The EXPLAIN output for a `ReclamationRecord` — the RFC-0005 EXPLAIN contract applied
/// to reclamation events (RFC-0027 §9).
///
/// This is a *copy* of the record fields in an inspectable, displayable form; it is the
/// "inspectable trace" the RFC-0005 contract requires (§2 — "every automatic selection emits
/// an inspectable record `{inputs considered, cost of each candidate, chosen option, …}`").
/// For reclamation the "selection" is the trigger decision; the "cost" is elided (reclamation
/// has no candidate set — it is a deterministic structural event).
///
/// Guarantee: `Declared` — the display format is not normatively specified; the fields are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainRecord {
    /// The scope that owned the value at reclamation time.
    pub scope_id: ScopeId,
    /// The sweep epoch at reclamation time.
    pub sweep_epoch: SweepEpoch,
    /// Why reclamation fired.
    pub trigger: ReclamationTrigger,
    /// Content identity of the reclaimed value.
    pub value_meta_hash: ContentHash,
    /// Channel (if this was a `ChannelClose` event).
    pub channel_id: Option<ChannelId>,
}

impl std::fmt::Display for ExplainRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let trigger = match &self.trigger {
            ReclamationTrigger::RcZero => "RcZero",
            ReclamationTrigger::ScopeExit => "ScopeExit",
            ReclamationTrigger::ChannelClose => "ChannelClose",
        };
        write!(
            f,
            "ReclamationExplain {{ scope={}, epoch={}, trigger={}, value={}",
            self.scope_id.0,
            self.sweep_epoch.0,
            trigger,
            self.value_meta_hash.as_str(),
        )?;
        if let Some(ch) = self.channel_id {
            write!(f, ", channel={}", ch.0)?;
        }
        write!(f, " }}")
    }
}

// ── ReclamationSink — the never-silent emit contract ─────────────────────────

/// The never-silent emit contract (G2 / RFC-0027 §9).
///
/// Every reclamation path MUST route through a `ReclamationSink::emit` call. An
/// implementation that drops a value without calling `emit` is a G2 violation — "exactly
/// as silently dropping a value" (RFC-0027 §9, lane-B R-1).
///
/// ## Architecture note (`Declared`)
///
/// The trait is the structural barrier that makes silent reclamation impossible:
/// - Reclamation code is required to hold a `&mut dyn ReclamationSink` (or equivalent)
///   and call `emit` before or at reclamation time.
/// - The `CollectingSink` (below) absorbs records for tests and audit-log consumers; a
///   future production sink routes to the supervision policy's observability sink
///   (RFC-0013 §8 / RFC-0027 §9 "routed to the supervision policy's observability sink").
///
/// ## Wiring: MEM-2 / MEM-3 (downstream FLAG)
///
/// The actual injection of `ReclamationSink` into rc-decrement / scope-exit / channel-close
/// call sites is MEM-2 (RC cell) and MEM-3 (region/batched reclamation). This trait defines
/// the contract; MEM-2/MEM-3 implement the call sites.
///
/// Guarantee: `Declared` — the trait contract is specified here; completeness (every live
/// reclamation path actually calls emit) depends on MEM-2/MEM-3 wiring.
pub trait ReclamationSink {
    /// Emit a reclamation record — called exactly once per reclamation event.
    ///
    /// Implementations MUST NOT silently discard the record without side-effect (G2).
    /// Discarding is only valid in tests; production sinks must surface the record.
    fn emit(&mut self, record: ReclamationRecord);
}

/// A `ReclamationSink` implementation that collects all emitted records into a `Vec`.
///
/// Useful for tests and for supervision policies that want to inspect the reclamation audit log.
/// In production, records would be routed to the supervision observability sink (RFC-0013 §8).
#[derive(Debug, Default)]
pub struct CollectingSink {
    /// All records emitted to this sink in emission order.
    pub records: Vec<ReclamationRecord>,
}

impl CollectingSink {
    /// Create a new empty collecting sink.
    #[must_use]
    pub fn new() -> Self {
        CollectingSink {
            records: Vec::new(),
        }
    }

    /// Number of records collected so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records have been collected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Drain all collected records, leaving the sink empty.
    pub fn drain(&mut self) -> Vec<ReclamationRecord> {
        std::mem::take(&mut self.records)
    }
}

impl ReclamationSink for CollectingSink {
    fn emit(&mut self, record: ReclamationRecord) {
        self.records.push(record);
    }
}

// Tests for this module live in `src/tests/reclamation.rs` per the M-797 in-crate test layout
// (`#[cfg(test)] mod tests;` in `lib.rs` → `src/tests/mod.rs` → `src/tests/reclamation.rs`).
