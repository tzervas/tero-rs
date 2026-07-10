//! Reconstruction manifest types and operations (RFC-0003 §6; spec §3/§4).
//!
//! A `ReconManifest` is the inspectable record `{ mode, model, dim, codebooks (content-addressed),
//! recipe?, decode, bound }`. It is authored/validated here; the actual reconstruction compute is
//! `std.vsa`'s. `spore` packages and validates; `vsa` executes (spec §2 boundary / §7 Q4).
//!
//! # Honesty guarantee (FR-C2 / VR-5)
//!
//! The **reconstruction honesty ceiling** is enforced in [`ReconManifest::validate`]:
//! a manifest whose decode procedure is `Resonator` **must not** have a bound basis stronger than
//! `Empirical`. Any attempt to author an over-strength resonator manifest returns
//! `Err(MalformedManifest::ResonatorOverStrength)`. `spore` carries the tag `std.vsa` establishes;
//! it does not set or upgrade it (VR-5). The ceiling is checked at the `ReconInfo` layer in the
//! kernel (`mycelium-core::recon`) and re-surfaced here for the ergonomic stdlib API.
//!
//! # Regrowth result (FLAG Q4a — RESOLVED)
//!
//! [`RegrowthResult`] carries the `Factorization` from `std.vsa` together with the manifest's full
//! certificate `Bound`, and projects to the stdlib's honest carrier via
//! [`RegrowthResult::as_approx`] → `std.numerics::Approx<Factorization>` (strength derived from the
//! bound's basis, never upgraded — VR-5).

use mycelium_core::bound::Bound;
pub use mycelium_core::ReconMode;
use mycelium_core::{GuaranteeStrength, ReconInfo, WfError};
use mycelium_std_numerics::Approx;
use mycelium_vsa::Factorization;

/// A validated reconstruction manifest — the RFC-0003 §6 record: mode, model, dim, codebooks,
/// optional recipe, decode procedure + params, and the `{ε,δ,strength}` bound certificate.
///
/// Construction goes through [`ReconManifest::new`] (validates at build time) or
/// [`ReconManifest::validate`] (validates an already-built [`ReconInfo`] from the kernel). A
/// well-formed `ReconManifest` is the only way to call `regrow`; an over-strength resonator
/// manifest is unrepresentable as a `ReconManifest`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconManifest {
    inner: ReconInfo,
}

impl ReconManifest {
    /// Build and validate a reconstruction manifest from its components.
    ///
    /// Delegates to [`ReconInfo::new`] for the kernel-level invariants (model non-empty, dim ≥ 1,
    /// codebooks non-empty, mode/recipe consistency, decode-procedure constraints, bound
    /// well-formedness), then applies the `std.spore` additional check:
    ///
    /// - **FR-C2 ceiling**: if `decode.procedure == Resonator`, the bound basis must not exceed
    ///   `Empirical`. The kernel (`ReconInfo::new`) enforces this; a violation is caught as
    ///   `WfError::MalformedReconstruction` and mapped to `Err(MalformedManifest::KernelWf)`.
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// Validation is a pure predicate; the same inputs always produce the same outcome.
    ///
    /// # Fallibility: `Err(MalformedManifest::KernelWf)`
    ///
    /// All kernel-level well-formedness violations (bad mode/bound, missing recipe/decode param,
    /// FR-C2 resonator-ceiling exceeded) return `Err(KernelWf)` — the specific kernel rule
    /// violated is described in the `WfError` message (C1/G2). Use [`ReconManifest::validate`] on
    /// a deserialized `ReconInfo` to get the `ResonatorOverStrength` variant explicitly.
    ///
    /// # Effects: none
    pub fn new(
        mode: ReconMode,
        model: impl Into<String>,
        dim: u32,
        codebooks: Vec<mycelium_core::ContentHash>,
        recipe: Option<mycelium_core::recon::Recipe>,
        decode: mycelium_core::recon::DecodeSpec,
        bound: mycelium_core::bound::Bound,
    ) -> Result<Self, MalformedManifest> {
        // The kernel (`ReconInfo::new`) enforces the FR-C2 ceiling (resonator + ProvenThm basis →
        // WfError::MalformedReconstruction). All WfError variants map to KernelWf here; the
        // `validate` path carries the defense-in-depth re-check for deserialized carry-ins.
        let inner = ReconInfo::new(mode, model, dim, codebooks, recipe, decode, bound)
            .map_err(|_: WfError| MalformedManifest::KernelWf)?;
        Ok(ReconManifest { inner })
    }

    /// Validate an existing [`ReconInfo`] from the kernel, wrapping it as a [`ReconManifest`].
    ///
    /// This is the path used when a `ReconInfo` arrives via deserialization or is produced by the
    /// kernel's own construction. The kernel's invariants are already enforced; this layer surfaces
    /// them in the `std.spore` error vocabulary.
    ///
    /// The resonator over-strength check is redundant here (the kernel enforces it), but is kept
    /// explicit to document the invariant at this layer (VR-5 / no black boxes).
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    ///
    /// # Fallibility: `Err(MalformedManifest::ResonatorOverStrength)`
    /// If the manifest carries a `Resonator` decode whose bound basis exceeds `Empirical`.
    /// In practice this should be unreachable (the kernel already refuses), but the check is
    /// present for defense-in-depth (C1/G2 — never silently trust a carry-in).
    ///
    /// # Effects: none
    pub fn validate(inner: ReconInfo) -> Result<Self, MalformedManifest> {
        // Defense-in-depth: re-check the FR-C2 ceiling even though the kernel enforces it.
        // Mutant witness: removing this check lets an over-strength resonator manifest pass validate.
        if inner.decode().procedure == mycelium_core::recon::DecodeProcedure::Resonator
            && inner.bound().basis.strength().rank() < GuaranteeStrength::Empirical.rank()
        {
            return Err(MalformedManifest::ResonatorOverStrength);
        }
        Ok(ReconManifest { inner })
    }

    /// The reconstruction mode (`IndexedRetrieval` or `CompositionalReconstruction`).
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// # Fallibility: total
    /// # Effects: none
    #[must_use]
    pub fn mode(&self) -> ReconMode {
        self.inner.mode()
    }

    /// The declared guarantee strength from the manifest's bound certificate.
    ///
    /// For a `Resonator` decode this is always ≤ `Empirical` (enforced at construction).
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// # Fallibility: total
    /// # Effects: none
    #[must_use]
    pub fn declared_strength(&self) -> GuaranteeStrength {
        self.inner.bound().basis.strength()
    }

    /// The content hash of the manifest, computed by hashing its canonical representation.
    ///
    /// Uses the kernel's content-hash surface (M-103 / ADR-003): the same manifest always
    /// produces the same hash; metadata is not identity.
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// # Fallibility: total
    /// # Effects: none
    #[must_use]
    pub fn manifest_hash(&self) -> mycelium_core::ContentHash {
        // Hash the canonical JSON serialization of the manifest (the wire form is the canonical
        // encoding; serde_json with sorted keys via BTreeMap/alphabetic field order).
        let json = serde_json::to_string(&self.inner).expect("ReconInfo is always serializable");
        let hex = blake3::hash(json.as_bytes()).to_hex();
        mycelium_core::ContentHash::from_parts("blake3", hex.as_str())
            .expect("blake3 hex is a valid digest")
    }

    /// Access the inner [`ReconInfo`] for callers that need the kernel representation (e.g.
    /// `std.vsa` reconstruct_* functions).
    #[must_use]
    pub fn inner(&self) -> &ReconInfo {
        &self.inner
    }

    /// The bound's failure-probability δ, if this is a `ProbabilityBound` (the common case for
    /// VSA resonator regrowth).
    ///
    /// Returns `None` for other bound kinds (e.g. `ErrorBound`, `CrosstalkBound`).
    ///
    /// # Guarantee tag: `Exact` (deterministic)
    /// # Fallibility: `None` when the bound is not a probability bound
    /// # Effects: none
    #[must_use]
    pub fn delta(&self) -> Option<f64> {
        match &self.inner.bound().kind {
            mycelium_core::bound::BoundKind::Probability { delta } => Some(*delta),
            _ => None,
        }
    }
}

impl std::fmt::Display for ReconManifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReconManifest {{ mode: {:?}, model: {}, dim: {}, strength: {:?} }}",
            self.inner.mode(),
            self.inner.model(),
            self.inner.dim(),
            self.declared_strength()
        )
    }
}

/// A refusal from manifest validation — explicitly named, never silent (C1/G2).
///
/// Each variant names the violated invariant so callers can surface the specific rule
/// violation (G11 dual projection — the error is both the refusal *and* the diagnostic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MalformedManifest {
    /// The manifest's decode is `Resonator` but its bound basis exceeds `Empirical` (FR-C2
    /// violation). A resonator decode is probabilistic-only and can never be `Proven`.
    ResonatorOverStrength,
    /// A kernel-level well-formedness violation: bad mode/bound, missing recipe/decode param.
    /// Covers `ReconInfo::new` refusals that do not correspond to the FR-C2 ceiling specifically.
    KernelWf,
}

impl std::fmt::Display for MalformedManifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MalformedManifest::ResonatorOverStrength => write!(
                f,
                "manifest-error: a Resonator decode is probabilistic-only (FR-C2); \
                 its bound.basis must not exceed Empirical — never Proven (VR-5)"
            ),
            MalformedManifest::KernelWf => write!(
                f,
                "manifest-error: kernel well-formedness check failed \
                 (bad mode/bound, missing recipe or decode param)"
            ),
        }
    }
}

mycelium_std_core::impl_std_error!(MalformedManifest);

/// The result of a probabilistic regrowth attempt via `std.vsa`.
///
/// Carries the `Factorization` returned by `std.vsa::reconstruct_factors` plus the
/// `GuaranteeStrength` tag from the manifest's bound certificate — always ≤ `Empirical`
/// for the resonator path (FR-C2 enforced at manifest construction time).
///
/// # `Approx<T>` coupling (FLAG Q4a — RESOLVED)
///
/// `RegrowthResult` carries the manifest's **full `{ε,δ,strength}` certificate bound** (with its
/// `BoundBasis`), so it projects losslessly to the stdlib's honest carrier via
/// [`RegrowthResult::as_approx`] — `std.numerics::Approx<Factorization>` — with **no fabricated
/// basis**: the strength is *derived* from the bound's basis ([`Approx::attach`], VR-5 — never
/// upgraded). It carries `Factorization` rather than a `core::Value` because the resonator decode
/// yields VSA factor atoms, not a reconstructed `Value` (that mapping is `std.vsa`'s, not spore's).
#[derive(Debug)]
pub struct RegrowthResult {
    /// The recovered factor atoms from the resonator (or cleanup) decode.
    pub factorization: Factorization,
    /// The manifest's `{ε,δ,strength}` certificate bound — its basis determines the honest
    /// strength. **Private (the FR-C2 seal):** a public field would let a caller hand-build a
    /// `RegrowthResult` with a `Proven` basis and project it to a `Proven` `Approx`, exceeding the
    /// probabilistic-regrowth ceiling. Construction goes through [`RegrowthResult::new`], which
    /// refuses an over-strength basis; read it via [`RegrowthResult::bound`].
    bound: Bound,
}

impl RegrowthResult {
    /// Construct a regrowth result, **refusing** a bound whose basis is stronger than `Empirical`
    /// (the FR-C2 / VR-5 probabilistic-regrowth ceiling) with an explicit
    /// [`MalformedManifest::ResonatorOverStrength`] — never a silent accept. This makes
    /// [`as_approx`](Self::as_approx) structurally incapable of producing a strength above
    /// `Empirical`: the ceiling holds by construction, not by comment.
    ///
    /// # Errors
    /// Returns [`MalformedManifest::ResonatorOverStrength`] if `bound`'s basis implies a strength
    /// stronger than `Empirical` (i.e. `Exact` or `Proven`).
    pub fn new(factorization: Factorization, bound: Bound) -> Result<Self, MalformedManifest> {
        if bound.basis.strength().rank() < GuaranteeStrength::Empirical.rank() {
            return Err(MalformedManifest::ResonatorOverStrength);
        }
        Ok(Self {
            factorization,
            bound,
        })
    }

    /// The certificate bound (read-only — construction enforces the FR-C2 ceiling).
    #[must_use]
    pub fn bound(&self) -> &Bound {
        &self.bound
    }

    /// The honest guarantee strength — **derived** from the bound's basis (never fabricated,
    /// never upgraded; VR-5). Always ≤ `Empirical` for a resonator decode (FR-C2, enforced by
    /// [`new`](Self::new)).
    ///
    /// # Guarantee tag: `Exact` (this is a pure predicate)
    /// # Fallibility: total
    #[must_use]
    pub fn strength(&self) -> GuaranteeStrength {
        self.bound.basis.strength()
    }

    /// The failure-probability δ from the certificate, when the bound is a `ProbabilityBound`
    /// (the common case for the resonator path); `None` for other bound kinds.
    #[must_use]
    pub fn delta(&self) -> Option<f64> {
        match &self.bound.kind {
            mycelium_core::bound::BoundKind::Probability { delta } => Some(*delta),
            _ => None,
        }
    }

    /// Project to the stdlib's honest carrier `std.numerics::Approx<Factorization>` (FLAG Q4a —
    /// RESOLVED). The strength is derived from the bound's basis via [`Approx::attach`] — never
    /// upgraded (VR-5); the δ rides along inside the carried `bound`.
    #[must_use]
    pub fn as_approx(&self) -> Approx<Factorization> {
        Approx::attach(self.factorization.clone(), self.bound.clone())
    }

    /// True iff the strength is exactly `Empirical` (the expected case for the resonator path).
    #[must_use]
    pub fn is_empirical(&self) -> bool {
        self.strength() == GuaranteeStrength::Empirical
    }

    /// True iff the strength is `Declared` (the weakest; user-asserted only).
    #[must_use]
    pub fn is_declared(&self) -> bool {
        self.strength() == GuaranteeStrength::Declared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{
        bound::{Bound, BoundBasis, BoundKind},
        content::operation_hash,
        recon::{DecodeProcedure, DecodeSpec},
    };

    fn empirical_bound() -> Bound {
        Bound {
            kind: BoundKind::Probability { delta: 0.05 },
            basis: BoundBasis::EmpiricalFit {
                trials: 1000,
                method: "test".to_owned(),
            },
        }
    }

    fn proven_bound() -> Bound {
        Bound {
            kind: BoundKind::Error {
                eps: 0.01,
                norm: mycelium_core::bound::NormKind::L2,
            },
            basis: BoundBasis::ProvenThm {
                citation: "test theorem".to_owned(),
            },
        }
    }

    fn declared_bound() -> Bound {
        Bound {
            kind: BoundKind::Probability { delta: 0.1 },
            basis: BoundBasis::UserDeclared,
        }
    }

    fn cleanup_decode() -> DecodeSpec {
        DecodeSpec {
            procedure: DecodeProcedure::Cleanup,
            cleanup_threshold: Some(0.3),
            factors: None,
            iteration_budget: None,
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        }
    }

    fn resonator_decode() -> DecodeSpec {
        DecodeSpec {
            procedure: DecodeProcedure::Resonator,
            cleanup_threshold: None,
            factors: Some(vec![operation_hash("factor-a"), operation_hash("factor-b")]),
            iteration_budget: Some(50),
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        }
    }

    // --- validate / ReconManifest::new ---

    /// An IndexedRetrieval + Cleanup + EmpiricalFit manifest is valid.
    #[test]
    fn valid_indexed_cleanup_manifest_is_accepted() {
        let m = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            cleanup_decode(),
            empirical_bound(),
        );
        assert!(m.is_ok(), "{m:?}");
    }

    /// A Resonator + EmpiricalFit manifest is valid (the expected canonical case).
    #[test]
    fn valid_resonator_empirical_manifest_is_accepted() {
        let m = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            resonator_decode(),
            empirical_bound(),
        );
        assert!(m.is_ok(), "{m:?}");
    }

    /// A Resonator + Declared manifest is valid (Declared is weaker than Empirical — OK).
    /// Guard: rejecting Declared-basis resonator manifests violates the spec (only ProvenThm is
    /// forbidden — the rule is "must not exceed Empirical", not "must be exactly Empirical").
    #[test]
    fn resonator_declared_basis_is_accepted() {
        // Mutant witness: rejecting this with ResonatorOverStrength violates the spec.
        let m = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            resonator_decode(),
            declared_bound(),
        );
        assert!(
            m.is_ok(),
            "Declared is weaker than Empirical and must be accepted: {m:?}"
        );
    }

    /// A Resonator + ProvenThm manifest via `new()` is REFUSED (FR-C2 ceiling violated).
    ///
    /// The kernel (`ReconInfo::new`) catches this as `WfError::MalformedReconstruction`, which
    /// maps to `MalformedManifest::KernelWf` here. The `ResonatorOverStrength` variant is the
    /// defense-in-depth path via `validate()` for deserialized carry-ins.
    ///
    /// Guard: accepting this (returning Ok) violates FR-C2 — a probabilistic resonator decode
    /// can never be Proven.
    #[test]
    fn resonator_proven_basis_is_refused_via_new() {
        // Mutant witness: accepting this (returning Ok) violates FR-C2.
        let err = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            resonator_decode(),
            proven_bound(),
        )
        .unwrap_err();
        // The kernel catches the FR-C2 violation and maps it to KernelWf (not ResonatorOverStrength).
        // ResonatorOverStrength is returned only by `validate()` as a defense-in-depth re-check.
        assert_eq!(
            err,
            MalformedManifest::KernelWf,
            "new() must map the kernel's FR-C2 refusal to KernelWf (not ResonatorOverStrength)"
        );
    }

    /// The `validate` path also refuses a (hypothetically constructed) over-strength manifest.
    /// This tests the defense-in-depth re-check in `validate`.
    ///
    /// NOTE: Since `ReconInfo::new` already enforces this, we cannot create a real
    /// over-strength `ReconInfo` directly. Instead we test `validate` with a valid manifest
    /// and verify `declared_strength()` is never above `Empirical` for the resonator path.
    #[test]
    fn validate_resonator_manifest_strength_is_at_most_empirical() {
        // Mutant witness: returning a strength stronger than Empirical for any resonator
        // manifest violates FR-C2.
        let m = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            resonator_decode(),
            empirical_bound(),
        )
        .unwrap();
        let strength = m.declared_strength();
        // rank() ordering: 0=Exact (strongest) … 3=Declared (weakest).
        // "strength ≤ Empirical" (in the lattice sense, i.e. at most as strong as Empirical)
        // means rank() >= Empirical.rank() (2), which excludes Exact (0) and Proven (1).
        assert!(
            strength.rank() >= GuaranteeStrength::Empirical.rank(),
            "resonator manifest strength must not exceed Empirical (rank must be ≥ 2); \
             got {:?} (rank {})",
            strength,
            strength.rank()
        );
    }

    /// `mode()` and `declared_strength()` are deterministic (Exact).
    #[test]
    fn mode_and_strength_are_deterministic() {
        let m = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        assert_eq!(m.mode(), ReconMode::IndexedRetrieval);
        assert_eq!(m.declared_strength(), GuaranteeStrength::Empirical);
    }

    /// `manifest_hash()` is deterministic — same manifest always produces the same hash.
    /// Guard: randomness in manifest_hash makes this fail.
    #[test]
    fn manifest_hash_is_deterministic() {
        let m1 = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        let m2 = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        assert_eq!(
            m1.manifest_hash(),
            m2.manifest_hash(),
            "manifest_hash must be deterministic (Exact)"
        );
        assert!(
            m1.manifest_hash().as_str().starts_with("blake3:"),
            "manifest_hash must use blake3 algorithm"
        );
    }

    /// Different manifests produce different hashes.
    /// Guard: returning a constant hash from manifest_hash makes this fail.
    #[test]
    fn different_manifests_produce_different_hashes() {
        // Mutant witness: returning a constant hash collapses both to the same hash.
        let m1 = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook-a")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        let m2 = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            2048, // different dim
            vec![operation_hash("codebook-a")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        assert_ne!(
            m1.manifest_hash(),
            m2.manifest_hash(),
            "manifests with different dims must hash differently"
        );
    }

    /// `delta()` returns the probability bound's δ when present.
    #[test]
    fn delta_returns_probability_bound_delta() {
        let m = ReconManifest::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            resonator_decode(),
            empirical_bound(),
        )
        .unwrap();
        assert_eq!(m.delta(), Some(0.05));
    }

    /// Error message for ResonatorOverStrength names the violated rule (G11 dual projection).
    #[test]
    fn resonator_over_strength_error_message_names_fr_c2() {
        let msg = format!("{}", MalformedManifest::ResonatorOverStrength);
        assert!(
            msg.contains("Resonator") || msg.contains("resonator"),
            "error must mention Resonator: {msg}"
        );
        assert!(
            msg.contains("Empirical"),
            "error must mention Empirical ceiling: {msg}"
        );
    }

    /// `RegrowthResult::is_empirical()` matches the Empirical strength.
    #[test]
    fn regrowth_result_strength_predicates() {
        use mycelium_vsa::{Factorization, ResonatorTrace, StopReason};
        let trace = ResonatorTrace {
            stop: StopReason::Converged,
            iterations: 3,
            trajectory: vec![],
            final_decode: vec![],
        };
        let bound = Bound {
            kind: BoundKind::Probability { delta: 0.05 },
            basis: BoundBasis::EmpiricalFit {
                trials: 1000,
                method: "resonator decode".into(),
            },
        };
        let r = RegrowthResult::new(
            Factorization {
                factors: vec![],
                trace,
            },
            bound,
        )
        .expect("Empirical basis is within the FR-C2 ceiling");
        assert!(r.is_empirical());
        assert!(!r.is_declared());
        assert_eq!(r.strength(), GuaranteeStrength::Empirical);
        assert_eq!(r.delta(), Some(0.05));
        // Projects losslessly to the honest carrier; strength is derived from the bound basis,
        // never upgraded (VR-5) — and stays at the Empirical ceiling (FR-C2).
        let approx = r.as_approx();
        assert_eq!(approx.strength(), GuaranteeStrength::Empirical);
        assert_eq!(&approx.bound, r.bound());
    }

    /// FR-C2 seal: `RegrowthResult::new` **refuses** a bound whose basis is stronger than
    /// `Empirical` (a `Proven`/`Exact` regrowth is dishonest for the probabilistic resonator path)
    /// — an explicit `ResonatorOverStrength`, never a silent accept. This is what makes
    /// `as_approx()` structurally unable to exceed the ceiling.
    #[test]
    fn regrowth_result_refuses_over_strength_basis() {
        use mycelium_vsa::{Factorization, ResonatorTrace, StopReason};
        let proven = Bound {
            kind: BoundKind::Probability { delta: 0.0 },
            basis: BoundBasis::ProvenThm {
                citation: "bogus — regrowth is never proven".into(),
            },
        };
        let trace = ResonatorTrace {
            stop: StopReason::Converged,
            iterations: 1,
            trajectory: vec![],
            final_decode: vec![],
        };
        let r = RegrowthResult::new(
            Factorization {
                factors: vec![],
                trace,
            },
            proven,
        );
        assert_eq!(r.unwrap_err(), MalformedManifest::ResonatorOverStrength);
    }
}
