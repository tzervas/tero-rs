//! The `Substrate` v0 value form (M-902; DN-71 **Model S** §4.1, maintainer-accepted 2026-07-02).
//!
//! A `Substrate{tag}` value is an **interpreter-level opaque affine handle** — a value-world citizen
//! of the L1 evaluator (the [`crate::eval::L1Value::Substrate`] variant), **not** a kernel value:
//!
//! - **not a kernel node and not a [`mycelium_core::Repr`]** — it names an *external resource*
//!   (RFC-0006 LR-8), not a value representation, so it grows no L0 `Node` and no `Repr` arm
//!   (KC-3; DN-71 §4.1; the elaborator already states the ground truth: `Substrate` "is not a
//!   representation type").
//! - **not content-addressed data** (ADR-003 identity is for values; a handle's identity is *not*
//!   its content — two acquisitions of the same external resource are two distinct handles).
//! - **creatable only through the acquiring surface** — [`SubstrateHandle::acquire`] records the
//!   acquisition provenance; there is **no literal** and no free/`Default` constructor in safe
//!   surface code.
//! - **inspectable, never a black box** (house rule 2): the `tag`, the opaque identity, and the
//!   acquisition provenance are [`SubstrateHandle::explain`]-visible; the resource *contents* stay
//!   opaque (they are the host's, not the value world's).
//!
//! # Scope of M-902 (creation · passage · inspection)
//! This module makes `Substrate` values **exist**, be **passed** (they flow through the evaluator's
//! ordinary value-binding machinery — `let`, argument passing, whole-value pattern binders), and be
//! **inspected**. The invalid states are unrepresentable or explicit errors — never a silent default
//! or a panic (G2/VR-5).
//!
//! # The M-903 use-once transition (this module's runtime half)
//! **M-903** (DN-71 §4.2) lands the affine use-once transition: [`SubstrateHandle::try_consume`] is
//! now the real checked move (Live → Consumed), backed by a `consumed` flag **shared across every
//! clone of the same identity** (`Clone` is passage, not re-acquisition — see the type doc below).
//! The **primary** enforcement is the *static* pass (`crate::affine`, run by
//! [`crate::checkty::check_nodule`]): a well-typed, checker-accepted program never calls
//! `try_consume` twice on the same identity. `try_consume`'s runtime check is the **backstop** DN-71
//! §4.2 asks for — under a correct static pass it is unreachable from checked code, so a tripped
//! backstop is an internal invariant surfaced loudly (G2), never silent corruption. It is also the
//! net that catches what the *lexically single-use* static pass cannot see: a `Substrate` captured
//! by a closure or a `for`-loop body that runs more than once at runtime (`crate::affine`'s module
//! docs name this limitation explicitly).
//!
//! # M-904 — the `consume` lowering, and the drop-without-consume v0 posture
//! **M-904** (DN-71 §4.3) wires [`SubstrateHandle::try_consume`] into real evaluator execution: the
//! surface `consume <expr>` now performs the checked move rather than refusing (see `crate::eval`'s
//! `Expr::Consume` arm). M-904 also lands the **drop-without-consume v0 posture** (DN-71 §8
//! FLAG-4 — the maintainer-delegated recommendation, accepted 2026-07-02): a live `Substrate`
//! binding whose lexical scope ends without an explicit `consume` and without escaping into that
//! scope's own result is **deterministically released** ([`SubstrateHandle::release`]) and the
//! release is **recorded** as a [`ReleaseEvent`] — never a silent leak (G2). This is a v0 posture,
//! not RFC-0027 OQ-5's full drop *protocol* (still open, pending the future `graft` RFC).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Process-unique source of opaque handle identities. A handle's identity is the *external resource*
/// it names, **not** its content (ADR-003 does not apply), so every [`SubstrateHandle::acquire`]
/// mints a fresh id — two acquisitions of the "same" resource are two distinct handles. Starts at 1
/// so `0` is never a valid handle id (a cheap never-silent sentinel if one is ever needed).
static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// How a [`SubstrateHandle`] was acquired — acquisition-provenance metadata (DN-71 §4.1; FLAG-9's
/// recommended v0 posture: carry the provenance, since it is what makes a handle *inspectable*
/// rather than aspirational). EXPLAIN-visible (house rule 2 — no black boxes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstrateProvenance {
    /// The acquiring operation — e.g. the `wild:<op>` host-call key of the quarantined `@std-sys`
    /// FFI floor today (RFC-0031 D1), or `graft(cap)` when R2 activates (`runtime.md` §API).
    pub acquired_via: String,
    /// A human-readable acquisition site/context for the EXPLAIN trail (e.g. the function name or a
    /// source description). Free-form; carried verbatim, never interpreted.
    pub site: String,
}

impl SubstrateProvenance {
    /// A provenance record naming the acquiring op and the acquisition site.
    #[must_use]
    pub fn new(acquired_via: impl Into<String>, site: impl Into<String>) -> Self {
        SubstrateProvenance {
            acquired_via: acquired_via.into(),
            site: site.into(),
        }
    }
}

/// An opaque, runtime-only **affine `Substrate` handle** (DN-71 Model S §4.1; M-902). It carries its
/// affine-resource `tag` (LR-8), an opaque host-handle **identity** (distinct per acquisition), and
/// its acquisition [`SubstrateProvenance`]. It is the payload of the
/// [`crate::eval::L1Value::Substrate`] value form.
///
/// **On `Clone` and affinity.** This type is `Clone` so the handle can ride the evaluator's ordinary
/// value-passing machinery (binding a `let`, passing an argument, a whole-value pattern binder all
/// clone the bound `L1Value`). Cloning preserves the **same identity** (`id`) — a clone is the *same*
/// resource, i.e. surface *passage*, not a second resource. Affinity (use-once) is **not** enforced
/// by making this Rust type non-`Clone`; the **primary** enforcement is a **checker** property — the
/// static affine pass (`crate::affine`, M-903) ensures no surface program moves a `Substrate`
/// binding more than once along any path (DN-71 §4.2; DN-33 §8.1 Q4 — the known-affine binding is
/// owned-unique in the checker, not in the value type). The Rust-level `Clone` is the passage
/// mechanism; [`Self::try_consume`]'s shared `consumed` flag (M-903) is the **runtime backstop** for
/// what the static pass cannot see (a closure/`for`-body capture that runs more than once).
#[derive(Debug, Clone)]
pub struct SubstrateHandle {
    tag: String,
    id: u64,
    provenance: SubstrateProvenance,
    /// The use-once **runtime backstop** (M-903; DN-71 §4.2) — `false` (live) until
    /// [`Self::try_consume`] transitions it. Shared (`Arc`) across every `Clone` of this identity:
    /// cloning is passage, not re-acquisition, so consuming *any* clone must be visible through
    /// *every* clone (never a backstop a naive re-clone could dodge).
    consumed: Arc<AtomicBool>,
}

impl PartialEq for SubstrateHandle {
    /// Identity equality, matching [`Self::id`]'s documented contract ("two handles are the same
    /// resource iff their ids are equal") — `provenance` is invariant per id (set once at
    /// [`Self::acquire`]) and `consumed` is shared *state* over one identity, not part of it, so
    /// neither needs comparing once `id` matches.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SubstrateHandle {}

impl SubstrateHandle {
    /// **Acquire** a fresh `Substrate` handle for `tag`, recording how it was acquired. This is the
    /// *only* way a handle comes to exist (DN-71 §4.1: no literal, no `Default`, no safe surface
    /// constructor). Each call mints a **fresh opaque identity**, so two acquisitions of the same
    /// external resource are two distinct handles (identity is the resource, not its content).
    #[must_use]
    pub fn acquire(tag: impl Into<String>, provenance: SubstrateProvenance) -> Self {
        SubstrateHandle {
            tag: tag.into(),
            id: NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed),
            provenance,
            consumed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The affine-resource `tag` (the `Substrate{tag}` name — RFC-0006 LR-8).
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// The opaque host-handle **identity** — distinct per [`acquire`](Self::acquire), and **not**
    /// content-derived. Two handles are the same resource iff their ids are equal.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The acquisition [`SubstrateProvenance`] (EXPLAIN-visible; DN-71 §4.1 / FLAG-9).
    #[must_use]
    pub fn provenance(&self) -> &SubstrateProvenance {
        &self.provenance
    }

    /// A never-silent, one-line EXPLAIN description of this handle (house rule 2 — no black boxes):
    /// the `tag`, the opaque identity, and the acquisition provenance are shown. The resource
    /// *contents* stay opaque (they are the host's, not the value world's), so this describes the
    /// handle without pretending to expose what it names.
    #[must_use]
    pub fn explain(&self) -> String {
        format!(
            "Substrate{{{}}} #{} acquired via `{}` at {}",
            self.tag, self.id, self.provenance.acquired_via, self.provenance.site
        )
    }

    /// Whether this handle has already been consumed (the [`Self::try_consume`] runtime backstop's
    /// current state) — inspectable, never a black box (house rule 2). `true` reflects a move made
    /// through *any* clone of this identity (the flag is shared, not per-clone).
    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.consumed.load(Ordering::Acquire)
    }

    /// The **consume/move transition** for the affine construct (DN-71 Model S §4.2; M-903): the
    /// checked Live → Consumed move, backed by the shared `consumed` flag.
    ///
    /// This is the **runtime backstop**, not the primary enforcement — the *static* affine pass
    /// (`crate::affine`, run during [`crate::checkty::check_nodule`]) is what a well-typed program is
    /// checked against, so a checker-accepted program never reaches a second `try_consume` on the
    /// same identity. This method exists so a double-consume that somehow slips past the static pass
    /// (the closure/loop-body multiplicity gap `crate::affine`'s docs name) still **traps explicitly**
    /// — never a silent second move, never corrupted state (G2/VR-5).
    ///
    /// `Ok` on the first call for this identity (across all its clones): the flag flips to consumed
    /// and the moved handle (same identity, now-consumed) is returned. `Err(SubstrateError::
    /// AlreadyConsumed)` on any subsequent call, naming the `tag` and `id` of the violated handle —
    /// never a silent no-op or a fabricated second move.
    ///
    /// M-904 (DN-71 §4.3) wires the surface `consume <expr>` into real evaluator execution
    /// (`crate::eval`'s `Expr::Consume` arm): this method is the checked primitive that lowering
    /// calls.
    pub fn try_consume(&self) -> Result<Self, SubstrateError> {
        // The only legal transition is `false -> true`; a `false` result (the AtomicBool's own
        // *prior* value) means WE won the race and made the move — `Ordering::AcqRel` on success
        // (visible to any later `Acquire` load/exchange on this same Arc, e.g. `is_consumed`) and
        // `Ordering::Acquire` on failure (nothing published, just observing that it's already gone).
        match self
            .consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_was_live) => Ok(self.clone()),
            Err(_already_consumed) => Err(SubstrateError::AlreadyConsumed {
                tag: self.tag.clone(),
                id: self.id,
            }),
        }
    }

    /// The **deterministic scope-exit release** — M-904's drop-without-consume v0 posture (DN-71
    /// §8 FLAG-4, maintainer-delegated acceptance 2026-07-02). This is the *other* terminal
    /// Live→terminal transition alongside [`Self::try_consume`]: both end the handle's life, and
    /// there is deliberately **no third Rust-level state** to track (KC-3) — a release reuses the
    /// same shared `consumed` flag, so a handle that is released can never subsequently be
    /// `try_consume`d (or released again) without tripping the same never-silent backstop.
    ///
    /// The evaluator (`crate::eval`) calls this exactly when a live `Substrate` binding's lexical
    /// scope ends **without** an explicit `consume` and **without** the value escaping (directly, or
    /// nested in a constructed value) into what that scope returns — see
    /// `crate::eval::Evaluator::release_events`. `site` is a free-form description of where the
    /// release happened (a binding name, mirroring [`SubstrateProvenance::site`]'s honesty — this
    /// evaluator has no source spans, so nothing more precise is fabricated).
    ///
    /// Returns `Some(ReleaseEvent)` on the *first* terminal transition for this identity — a real
    /// release happened, recorded, never silent (G2). Returns `None` if the handle was already
    /// terminal (already `consume`d, or already released through another clone of the same
    /// identity) — nothing new happened, so no duplicate/fabricated event is manufactured.
    #[must_use]
    pub fn release(&self, site: impl Into<String>) -> Option<ReleaseEvent> {
        match self
            .consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_was_live) => Some(ReleaseEvent {
                tag: self.tag.clone(),
                id: self.id,
                site: site.into(),
            }),
            Err(_already_terminal) => None,
        }
    }
}

/// A recorded **scope-exit release** — the DN-71 §8 FLAG-4 v0 drop-without-consume posture (M-904).
/// Produced by [`SubstrateHandle::release`] exactly when a live handle is released at scope exit;
/// never fabricated for a handle that was explicitly `consume`d or already released (G2). Inspect
/// the accumulated log via `crate::eval::Evaluator::release_events` — never a silent leak (house
/// rule 2: no black boxes).
///
/// This is the v0 posture only: RFC-0027 OQ-5's full drop *protocol* (a possible future
/// `graft`-RFC-driven alternative reclamation policy) stays open; DN-71 does not re-decide it here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseEvent {
    /// The tag of the released handle.
    pub tag: String,
    /// The opaque identity ([`SubstrateHandle::id`]) of the released handle.
    pub id: u64,
    /// Where the release happened (a binding name / free-form site description — no fabricated
    /// line/column; this evaluator has no source spans, VR-5).
    pub site: String,
}

impl core::fmt::Display for ReleaseEvent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "released: `Substrate{{{}}}` #{} was never `consume`d — deterministically released at \
             scope exit (`{}`), per DN-71 §8 FLAG-4's v0 drop-without-consume posture (never a \
             silent leak — G2/M-904)",
            self.tag, self.id, self.site
        )
    }
}

/// Why a `Substrate` operation was refused — always explicit (never-silent; G2/VR-5). The variant
/// set is closed: v0 supports only the create/pass/inspect/consume surface, so the sole refusal is
/// the runtime use-once backstop (`crate::affine`'s static pass is the primary enforcement, and
/// refuses at check time — never reaching this Rust-level `Result` at all for a checked program).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstrateError {
    /// The affine use-once **move** (Live → Consumed) was attempted on an identity that was already
    /// consumed — the [`SubstrateHandle::try_consume`] runtime backstop (M-903; DN-71 Model S §4.2)
    /// tripped. Under a correct static pass this is unreachable from a checked program; a real
    /// occurrence is an internal invariant break, surfaced loudly (G2) rather than as silent
    /// corruption. Names the `tag` and `id` of the violated handle.
    AlreadyConsumed {
        /// The `tag` of the handle whose consume/move was refused.
        tag: String,
        /// The opaque identity ([`SubstrateHandle::id`]) of the already-consumed handle.
        id: u64,
    },
}

impl core::fmt::Display for SubstrateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SubstrateError::AlreadyConsumed { tag, id } => write!(
                f,
                "double-consume: `Substrate{{{tag}}}` #{id} was already consumed — this is the \
                 M-903 runtime use-once backstop (DN-71 Model S §4.2) tripping on a move that the \
                 static affine pass should have refused at check time; a checked program never \
                 reaches this at runtime, so seeing it means either an unchecked call path or a \
                 closure/loop-body capture the static pass cannot see (`crate::affine` docs) — \
                 never silent corruption (G2/VR-5)"
            ),
        }
    }
}

impl std::error::Error for SubstrateError {}
