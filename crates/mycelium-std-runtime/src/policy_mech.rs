//! Mechanized `SelectionPolicy` capture and setting (M-963; DN-78 §3 B-1/B-2; the M-828
//! capture-and-set tail, decided under the 2026-07-02 delegation recorded in DN-78 §1).
//!
//! Two surfaces, both riding the existing RFC-0005 machinery in `mycelium-select` — **no new
//! selection mechanism** (KC-3; DN-63 §3.5 "the third application of the existing one"):
//!
//! - **Capture (B-1):** [`capture`] materializes the policy that decided a recorded
//!   [`Explanation`] back into a nameable, diffable, inspectable [`SelectionPolicy`] value via
//!   the [`PolicyRegistry`] — never an opaque handle (ADR-006). [`replay`] re-runs the recorded
//!   inputs (honoring the recorded override state) and requires the same decision; divergence
//!   is an explicit [`ReplayError::Diverged`], never a silent pass (G2).
//! - **Setting (B-2):** [`PolicySlot`] binds the active policy for one RFC-0005 site
//!   ([`PolicySite`]); every [`PolicySlot::set`] appends a [`PolicySetRecord`] to an
//!   append-only transition log — a mechanized set is never a silent override
//!   (research/27-dn64-ergonomics-rnd-RECORD.md §2.2), and EXPLAIN stays answerable afterward
//!   (the slot holds the full policy value). Selection through the slot records the mandatory
//!   [`Explanation`] into an extractable trace (the "runtime records which policy it applied"
//!   half of mechanized capture).
//!
//! # Guarantee tags (VR-5; rows in [`crate::guarantee_matrix::MATRIX`])
//!
//! - Transition-record append (one record per `set`, monotonic `seq`): **`Exact`** —
//!   by construction.
//! - Selection without an active policy is an explicit [`SlotError::NoActivePolicy`]: **`Exact`**
//!   — fail-closed by construction (G2).
//! - Capture resolution of an unknown `policy_ref` is an explicit
//!   [`CaptureError::UnknownPolicyRef`]: **`Exact`** — fail-closed by construction (G2).
//! - Replay-reaches-the-recorded-decision: **`Empirical`** — the record-vs-replay differential
//!   is property-tested (`src/tests/policy_mech.rs`); determinism grounds in RFC-0005 `select`
//!   purity but carries no mechanized theorem, so it is not `Proven` (M-964 audit, DN-78
//!   appendix).

use mycelium_core::ContentHash;
use mycelium_select::{
    select, Candidate, Explanation, PolicyRegistry, SelectError, SelectionInputs, SelectionPolicy,
};

/// The RFC-0005 policy sites (§4: swap-target, packing; RFC-0008 RT3 adds placement as the
/// third). A [`PolicySlot`] is keyed by site so a set/select is always attributed to the site
/// it governs (provenance, RFC-0001 §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySite {
    /// The RFC-0002 swap-target site.
    SwapTarget,
    /// The RFC-0004 §5 packing site.
    Packing,
    /// The RFC-0008 RT3 placement site (single-node in Phase I — DN-78 §3; the multi-node
    /// candidate set is deferred, see [`crate::r2_residual`]).
    Placement,
}

impl core::fmt::Display for PolicySite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolicySite::SwapTarget => write!(f, "swap-target"),
            PolicySite::Packing => write!(f, "packing"),
            PolicySite::Placement => write!(f, "placement"),
        }
    }
}

/// A reified policy-set transition record (G2: a mechanized set is never a silent override —
/// the transition itself is inspectable). Guarantee: **`Exact`** — every [`PolicySlot::set`]
/// appends exactly one record, with a per-slot monotonic sequence number, by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySetRecord {
    /// The site whose active policy changed.
    pub site: PolicySite,
    /// Per-slot monotonic sequence number (0 for the first set).
    pub seq: u64,
    /// The previous active policy's content address, `None` on the first set.
    pub previous: Option<ContentHash>,
    /// The new active policy's content address ([`SelectionPolicy::policy_ref`]).
    pub new_policy: ContentHash,
    /// The new policy's display name (for the EXPLAIN/teaching surface).
    pub new_policy_name: String,
}

/// Why a slot operation failed — always explicit (G2), never a silent default choice.
#[derive(Debug, Clone, PartialEq)]
pub enum SlotError {
    /// Selection was requested but no policy has been set for this site. Guarantee: **`Exact`**
    /// — fail-closed by construction; there is no built-in fallback policy (a silent default
    /// would be a black box, ADR-006).
    NoActivePolicy {
        /// The site that has no active policy.
        site: PolicySite,
    },
    /// The underlying RFC-0005 selection refused (e.g. an out-of-range override).
    Select(SelectError),
}

impl core::fmt::Display for SlotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SlotError::NoActivePolicy { site } => {
                write!(
                    f,
                    "no active policy is set for the {site} site — set one explicitly \
                     (PolicySlot::set); there is no silent default (G2/ADR-006)"
                )
            }
            SlotError::Select(e) => write!(f, "selection refused: {e}"),
        }
    }
}

impl std::error::Error for SlotError {}

impl From<SelectError> for SlotError {
    fn from(e: SelectError) -> Self {
        SlotError::Select(e)
    }
}

/// A runtime slot binding the **active** [`SelectionPolicy`] for one RFC-0005 site, with an
/// append-only transition log and a selection trace (DN-78 §3 B-2).
///
/// The slot is the mechanized *setter surface*: `set` swaps the active policy and records the
/// transition; `select` decides through the active policy and records the mandatory
/// [`Explanation`]. Both logs are read-only views (`transitions`/`trace`) — append-only by
/// construction (no public mutation besides the appending operations).
#[derive(Debug)]
pub struct PolicySlot {
    site: PolicySite,
    active: Option<SelectionPolicy>,
    transitions: Vec<PolicySetRecord>,
    trace: Vec<Explanation>,
}

impl PolicySlot {
    /// An empty slot for `site` — no active policy, no transitions, no trace.
    #[must_use]
    pub fn new(site: PolicySite) -> Self {
        PolicySlot {
            site,
            active: None,
            transitions: Vec::new(),
            trace: Vec::new(),
        }
    }

    /// The site this slot governs.
    #[must_use]
    pub fn site(&self) -> PolicySite {
        self.site
    }

    /// Set the active policy, appending a [`PolicySetRecord`] (returned by reference).
    ///
    /// Guarantee: **`Exact`** — exactly one record is appended per call, `seq` is the per-slot
    /// monotonic count, and `previous` is the outgoing policy's content address (`None` on the
    /// first set). The transition is never silent (G2).
    pub fn set(&mut self, policy: SelectionPolicy) -> &PolicySetRecord {
        let record = PolicySetRecord {
            site: self.site,
            seq: self.transitions.len() as u64,
            previous: self.active.as_ref().map(SelectionPolicy::policy_ref),
            new_policy: policy.policy_ref(),
            new_policy_name: policy.name().to_owned(),
        };
        self.active = Some(policy);
        self.transitions.push(record);
        self.transitions
            .last()
            .expect("push in the line above guarantees a last element")
    }

    /// The active policy, if one has been set. `None` is not a fallback state — selection
    /// through an unset slot refuses explicitly ([`SlotError::NoActivePolicy`]).
    #[must_use]
    pub fn active(&self) -> Option<&SelectionPolicy> {
        self.active.as_ref()
    }

    /// The append-only transition log (every `set`, in order).
    #[must_use]
    pub fn transitions(&self) -> &[PolicySetRecord] {
        &self.transitions
    }

    /// The selection trace: the mandatory [`Explanation`] of every selection made through this
    /// slot, in order — extractable for capture/diffing (DN-78 §3 B-2).
    #[must_use]
    pub fn trace(&self) -> &[Explanation] {
        &self.trace
    }

    /// Decide through the active policy (RFC-0005 `select`), recording the mandatory
    /// [`Explanation`] into the slot's trace.
    ///
    /// Errors are explicit: an unset slot is [`SlotError::NoActivePolicy`] (never a silent
    /// default — G2/ADR-006); an underlying selection refusal passes through as
    /// [`SlotError::Select`].
    pub fn select(
        &mut self,
        inputs: &SelectionInputs,
        forced: Option<usize>,
    ) -> Result<(Candidate, Explanation), SlotError> {
        let site = self.site;
        let policy = self
            .active
            .as_ref()
            .ok_or(SlotError::NoActivePolicy { site })?;
        let (candidate, explanation) = select(policy, inputs, forced)?;
        self.trace.push(explanation.clone());
        Ok((candidate, explanation))
    }
}

/// A captured policy: the RFC-0005-conformant [`SelectionPolicy`] value that decided a recorded
/// [`Explanation`], materialized for reuse/diffing (DN-78 §3 B-1). Not an opaque handle — the
/// full policy value is inspectable (ADR-006).
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedPolicy {
    /// The policy's content address (equal to `policy.policy_ref()` — verified at capture).
    pub policy_ref: ContentHash,
    /// The policy value itself.
    pub policy: SelectionPolicy,
}

/// Why a capture failed — always explicit (G2), never a silent reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// The recorded `policy_ref` resolves to nothing in the given registry. Guarantee:
    /// **`Exact`** — fail-closed by construction; capture never fabricates a policy.
    UnknownPolicyRef {
        /// The unresolvable content address.
        policy_ref: ContentHash,
    },
    /// The registry returned a policy whose own content address differs from the requested one
    /// (a corrupted registry). Never silently accepted.
    RefMismatch {
        /// The content address the capture asked for.
        requested: ContentHash,
        /// The content address the resolved policy actually hashes to.
        resolved: ContentHash,
    },
}

impl core::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CaptureError::UnknownPolicyRef { policy_ref } => {
                write!(
                    f,
                    "policy_ref {policy_ref:?} is not in the registry — capture refuses rather \
                     than reconstructing a policy (G2/ADR-006)"
                )
            }
            CaptureError::RefMismatch {
                requested,
                resolved,
            } => {
                write!(
                    f,
                    "registry corruption: requested {requested:?} but the stored policy hashes \
                     to {resolved:?}"
                )
            }
        }
    }
}

impl std::error::Error for CaptureError {}

/// Materialize the policy that decided `explanation` from `registry` (DN-78 §3 B-1).
///
/// Guarantee: **`Exact`** for the resolution contract — an unknown ref is an explicit
/// [`CaptureError::UnknownPolicyRef`] and a hash mismatch an explicit
/// [`CaptureError::RefMismatch`]; a returned [`CapturedPolicy`] always satisfies
/// `policy.policy_ref() == policy_ref` (checked here, not assumed).
pub fn capture(
    registry: &PolicyRegistry,
    explanation: &Explanation,
) -> Result<CapturedPolicy, CaptureError> {
    let requested = explanation.policy.clone();
    let policy = registry
        .get(&requested)
        .ok_or_else(|| CaptureError::UnknownPolicyRef {
            policy_ref: requested.clone(),
        })?;
    let resolved = policy.policy_ref();
    if resolved != requested {
        return Err(CaptureError::RefMismatch {
            requested,
            resolved,
        });
    }
    Ok(CapturedPolicy {
        policy_ref: requested,
        policy: policy.clone(),
    })
}

/// Why a replay failed — always explicit (G2), never a silent pass.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    /// The captured policy is not the one the record claims decided it (`policy_ref`
    /// mismatch) — replaying against the wrong policy would be a silent apples-to-oranges
    /// comparison, so it refuses up front.
    PolicyMismatch {
        /// The record's policy content address.
        recorded: ContentHash,
        /// The captured policy's content address.
        captured: ContentHash,
    },
    /// Re-running the recorded inputs refused (e.g. a recorded override index that no longer
    /// fits the policy — impossible for a faithful capture, but never silently swallowed).
    Select(SelectError),
    /// The replayed decision differs from the recorded one. With a validated policy and
    /// identical inputs this indicates non-determinism or a record from different code —
    /// either way it is surfaced, never absorbed.
    Diverged {
        /// The original record.
        recorded: Box<Explanation>,
        /// The replayed record that differs from it.
        replayed: Box<Explanation>,
    },
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReplayError::PolicyMismatch { recorded, captured } => write!(
                f,
                "replay refused: the record was decided by {recorded:?} but the captured \
                 policy is {captured:?}"
            ),
            ReplayError::Select(e) => write!(f, "replay selection refused: {e}"),
            ReplayError::Diverged { recorded, replayed } => write!(
                f,
                "replay diverged: recorded chose index {} (rule {:?}), replay chose index {} \
                 (rule {:?})",
                recorded.chosen_index,
                recorded.matched_rule,
                replayed.chosen_index,
                replayed.matched_rule
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<SelectError> for ReplayError {
    fn from(e: SelectError) -> Self {
        ReplayError::Select(e)
    }
}

/// Replay a recorded decision against its captured policy (DN-78 §3 B-1): re-run the recorded
/// inputs — honoring the recorded override state — and require the identical [`Explanation`].
///
/// Guarantee: **`Empirical`** for "replay reaches the recorded decision" — the record-vs-replay
/// differential is property-tested over randomized policies/inputs; RFC-0005 `select` is
/// deterministic (same `(policy, inputs, forced)` → same result) but that determinism carries
/// no mechanized theorem, so the claim is not `Proven` (VR-5; M-964 audit). A divergence is an
/// explicit [`ReplayError::Diverged`] carrying both records for inspection (G2).
pub fn replay(
    captured: &CapturedPolicy,
    recorded: &Explanation,
) -> Result<Explanation, ReplayError> {
    if captured.policy_ref != recorded.policy {
        return Err(ReplayError::PolicyMismatch {
            recorded: recorded.policy.clone(),
            captured: captured.policy_ref.clone(),
        });
    }
    let forced = recorded.overridden.then_some(recorded.chosen_index);
    let (_, replayed) = select(&captured.policy, &recorded.inputs, forced)?;
    if replayed != *recorded {
        return Err(ReplayError::Diverged {
            recorded: Box::new(recorded.clone()),
            replayed: Box::new(replayed),
        });
    }
    Ok(replayed)
}
