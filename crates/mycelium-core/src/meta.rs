//! Runtime metadata that travels with every value (RFC-0001 §4.3; `meta.schema.json`).
//!
//! [`Meta`] enforces the schema invariants **M-I1…M-I4** *by construction* — a [`Meta`] cannot be
//! built with an inconsistent guarantee/bound pairing (the honesty rule, mechanically). Its `serde`
//! form is `meta.schema.json` (M-104): `bound` is modelled by **presence** (absent for `Exact`,
//! per M-I1), and `Deserialize` re-runs the M-I1…M-I4 invariants through [`Meta::new`] so a
//! malformed wire `Meta` is rejected, never silently trusted.

use serde::{Deserialize, Serialize};

use crate::bound::{Bound, BoundBasis};
use crate::cert_mode::CertMode;
use crate::id::ContentHash;
use crate::recon::ReconInfo;
use crate::wrapping::WrappingOpt;
use crate::{GuaranteeStrength, WfError};

/// Provenance: an acyclic derivation DAG (RFC-0001 §4.6). Not part of code identity. The `serde`
/// form is tagged on `kind` (`Root|Derived`), matching `provenance.schema.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Provenance {
    /// A primitive/constant origin.
    Root,
    /// Derived from inputs by the operation at content hash `op`.
    Derived {
        /// Content hash of the producing operation/definition.
        op: ContentHash,
        /// Content hashes of the inputs.
        inputs: Vec<ContentHash>,
    },
}

/// Measured (dynamic) sparsity — distinct from the declared [`crate::repr::SparsityClass`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SparsityObs {
    /// Number of active components.
    pub active: u64,
    /// Density in `[0, 1]`.
    pub density: f64,
}

/// Lossless physical packing schemes (extensible registry; RFC-0001 §4.3; DN-01). The `serde`
/// renderings match `physical-layout.schema.json`'s `PackScheme` enum (`I2S|TL1|TL2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackScheme {
    /// Unpacked.
    Unpacked,
    /// Two bits per trit.
    TwoBitPerTrit,
    /// Five trits per byte.
    FiveTritPerByte,
    /// bitnet.cpp I2_S.
    I2S,
    /// bitnet.cpp TL1.
    #[serde(rename = "TL1")]
    Tl1,
    /// bitnet.cpp TL2.
    #[serde(rename = "TL2")]
    Tl2,
}

/// The recorded schedule-staged packing (RFC-0001 §4.3; RFC-0004 §5). A *record*, not the decision.
/// The `serde` form is tagged on `layout`, matching `physical-layout.schema.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "layout")]
pub enum PhysicalLayout {
    /// Binary words.
    BinaryWords,
    /// Packed trits under `scheme`.
    TritPacked {
        /// The packing scheme used.
        scheme: PackScheme,
    },
    /// Dense scalar array.
    DenseArray,
    /// VSA storage (sparse or dense).
    VsaStore {
        /// Whether storage is sparse.
        sparse: bool,
    },
}

/// Runtime, queryable metadata (RFC-0001 §4.3). Fields are private; the only way to build a `Meta`
/// is [`Meta::new`], which enforces M-I1…M-I4 — so an inconsistent `Meta` is unrepresentable.
///
/// `reconstruction` (RFC-0003 §6; M-260) is attached via [`Meta::with_reconstruction`] — its own
/// invariants are enforced by [`ReconInfo::new`], and it does not interact with M-I1…M-I4.
#[derive(Debug, Clone, PartialEq)]
pub struct Meta {
    provenance: Provenance,
    guarantee: GuaranteeStrength,
    bound: Option<Bound>,
    sparsity: Option<SparsityObs>,
    physical: Option<PhysicalLayout>,
    reconstruction: Option<Box<ReconInfo>>,
    policy_used: Option<ContentHash>,
    /// The certification mode this value was produced under (RFC-0034 §3.1; M-786). A **never-silent**
    /// tag — every `Meta` carries one — defaulting to [`CertMode::Fast`] (RFC-0034 §5). Like all of
    /// `Meta`, it is **excluded from the content hash** (RFC-0001 §4.6; ADR-003 — switching modes never
    /// perturbs identity); the exclusion is by construction (the hasher never reads `Meta`).
    cert_mode: CertMode,
    /// The explicit Axis-B `wrapping` opt-out, if the producing operation was given a named,
    /// explicit `wrapping` annotation (RFC-0034 §10; M-791). **Absent by default** — never-silent
    /// failability (out-of-range → `Option`/`Result`/`SwapError`) is the default in every mode.
    /// Presence means the developer *explicitly* opted in to wraparound at the use site; it is
    /// never an ambient default and does **not** silence Axis-A guarantee/cert_mode tags (VR-5).
    /// Excluded from the content hash (rides `Meta`; RFC-0001 §4.6).
    wrapping_opt: Option<WrappingOpt>,
}

impl Meta {
    /// Build a `Meta`, enforcing the guarantee/bound invariants:
    ///
    /// - **M-I1** `guarantee == Exact ⟺ bound == None`,
    /// - **M-I2** `Proven ⟹ basis ProvenThm`, **M-I3** `Empirical ⟹ EmpiricalFit`,
    ///   **M-I4** `Declared ⟹ UserDeclared`,
    ///
    /// plus numeric well-formedness of any bound. Returns [`WfError`] on violation.
    pub fn new(
        provenance: Provenance,
        guarantee: GuaranteeStrength,
        bound: Option<Bound>,
        sparsity: Option<SparsityObs>,
        physical: Option<PhysicalLayout>,
        policy_used: Option<ContentHash>,
    ) -> Result<Self, WfError> {
        check_guarantee_bound(guarantee, bound.as_ref())?;
        if let Some(b) = &bound {
            if !b.well_formed() {
                return Err(WfError::MalformedBound);
            }
        }
        if let Some(s) = &sparsity {
            if !(0.0..=1.0).contains(&s.density) {
                // A6-08: a sparsity observation is a measurement, not a guarantee bound — so an
                // out-of-range density is `MalformedSparsity`, not the misleading `MalformedBound`.
                return Err(WfError::MalformedSparsity);
            }
        }
        Ok(Meta {
            provenance,
            guarantee,
            bound,
            sparsity,
            physical,
            reconstruction: None,
            policy_used,
            cert_mode: CertMode::default(),
            wrapping_opt: None,
        })
    }

    /// Attach a reconstruction manifest (RFC-0003 §6; M-260). The manifest's own schema
    /// invariants are already enforced by [`ReconInfo::new`]; it is independent of M-I1…M-I4,
    /// so this cannot invalidate an existing `Meta`.
    #[must_use]
    pub fn with_reconstruction(mut self, reconstruction: ReconInfo) -> Self {
        self.reconstruction = Some(Box::new(reconstruction));
        self
    }

    /// Record the schedule-staged packing chosen at a lowering stage (RFC-0004 §5; DN-01;
    /// M-250). This is the **inspectable record** of the layout decision, not the decision locus
    /// (the selector is [`mycelium-select`](https://docs.rs/mycelium-select); RFC-0005 §4).
    ///
    /// **M-I5 (lossless `physical`).** The layout is a lossless re-encoding of the same `payload`:
    /// it touches only the `physical` field and leaves the guarantee, bound, and value untouched
    /// — so recording (or *re*-recording) it can never change the value's type or its guarantee
    /// (RFC-0001 §4.3; `physical-layout.schema.json`). M-I1…M-I4 are therefore preserved by
    /// construction.
    #[must_use]
    pub fn with_physical(mut self, physical: PhysicalLayout) -> Self {
        self.physical = Some(physical);
        self
    }

    /// Record the certification mode this value was produced under (RFC-0034 §3.1; M-786). Touches
    /// only the `cert_mode` tag — it leaves the guarantee, bound, and value untouched, so it can
    /// never change a value's strength or identity (the mode rides `Meta`, excluded from the content
    /// hash; RFC-0001 §4.6 / ADR-003). A stronger mode is **not** a stronger guarantee (VR-5).
    #[must_use]
    pub fn with_cert_mode(mut self, cert_mode: CertMode) -> Self {
        self.cert_mode = cert_mode;
        self
    }

    /// Attach the explicit named Axis-B `wrapping` opt-out marker (RFC-0034 §10; M-791).
    ///
    /// Touches only `wrapping_opt` — leaves the guarantee, bound, cert_mode, and value untouched.
    /// The opt-out is **never-silent**: the developer explicitly called this method at the use site,
    /// making the intent visible and grep-auditable. It does **not** silence Axis-A guarantee tags
    /// or the certification mode; those axes are orthogonal (VR-5 — opting out of one axis never
    /// upgrades another).
    ///
    /// **Absent by default:** not calling this method means Axis-B never-silent failability is
    /// active (the safe default). Presence signals "the caller explicitly requested wraparound."
    ///
    /// **FLAG (M-791):** the op-layer wiring that *honors* this marker by actually electing
    /// wraparound instead of returning `Option`/`Result`/`SwapError` is a downstream task
    /// (M-788 onward); this method records the intent at the metadata level.
    #[must_use]
    pub fn with_wrapping(mut self, opt: WrappingOpt) -> Self {
        self.wrapping_opt = Some(opt);
        self
    }

    /// The common `Exact` metadata with no bound (M-I1).
    #[must_use]
    pub fn exact(provenance: Provenance) -> Self {
        Meta {
            provenance,
            guarantee: GuaranteeStrength::Exact,
            bound: None,
            sparsity: None,
            physical: None,
            reconstruction: None,
            policy_used: None,
            cert_mode: CertMode::default(),
            wrapping_opt: None,
        }
    }

    /// The value's provenance.
    #[must_use]
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    /// The disclosed guarantee strength.
    #[must_use]
    pub fn guarantee(&self) -> GuaranteeStrength {
        self.guarantee
    }
    /// The bound, if approximate.
    #[must_use]
    pub fn bound(&self) -> Option<&Bound> {
        self.bound.as_ref()
    }
    /// Measured sparsity, if recorded.
    #[must_use]
    pub fn sparsity(&self) -> Option<SparsityObs> {
        self.sparsity
    }
    /// The recorded physical layout, if any.
    #[must_use]
    pub fn physical(&self) -> Option<PhysicalLayout> {
        self.physical
    }
    /// The reconstruction manifest, if attached (RFC-0003 §6).
    #[must_use]
    pub fn reconstruction(&self) -> Option<&ReconInfo> {
        self.reconstruction.as_deref()
    }
    /// The policy that produced this value (set iff produced by a swap).
    #[must_use]
    pub fn policy_used(&self) -> Option<&ContentHash> {
        self.policy_used.as_ref()
    }
    /// The certification mode this value was produced under (RFC-0034 §3.1; M-786). Always present —
    /// a never-silent tag — defaulting to [`CertMode::Fast`].
    #[must_use]
    pub fn cert_mode(&self) -> CertMode {
        self.cert_mode
    }

    /// The explicit Axis-B `wrapping` opt-out marker, if present (RFC-0034 §10; M-791).
    ///
    /// Returns `None` (the default) when Axis-B never-silent failability is active — the safe
    /// path requires no annotation. Returns `Some(WrappingOpt)` when the producing operation
    /// was explicitly annotated with a `wrapping` opt-out at the use site.
    ///
    /// Presence does **not** imply a weaker guarantee or a lower cert_mode; those axes are
    /// orthogonal (VR-5). Use this accessor to inspect or propagate the wraparound intent.
    #[must_use]
    pub fn wrapping_opt(&self) -> Option<WrappingOpt> {
        self.wrapping_opt
    }
}

/// The wire projection of [`Meta`] (`meta.schema.json`). Optional fields are omitted when absent
/// (so `Exact` emits no `bound`, satisfying M-I1's presence model); on the way back in, `null` and
/// absent both decode to `None`. `reconstruction` (RFC-0003 §6) **is** carried (serialized when
/// present, re-validated on the way in). `deny_unknown_fields` makes the schema's
/// `additionalProperties: false` a real contract — an unknown wire field is rejected, not silently
/// dropped (A6-02/B2-03).
///
/// **`cert_mode` is not carried on the wire (M-786, deferred — not silent).** The certification mode
/// (RFC-0034 §3.1) is a *runtime* tag resolved from the active `@certification` scope (M-790), not an
/// intrinsic property of a persisted value, so it is intentionally **not** part of this projection
/// (keeping `meta.schema.json` / `additionalProperties: false` unchanged until the schema migration
/// lands). A deserialized `Meta` therefore resolves to [`CertMode::Fast`] — the *weakest* mode, which
/// never over-claims a stronger one (the VR-5 floor: a loaded value is never silently `Certified`).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetaWire {
    provenance: Provenance,
    guarantee: GuaranteeStrength,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bound: Option<Bound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sparsity: Option<SparsityObs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    physical: Option<PhysicalLayout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reconstruction: Option<Box<ReconInfo>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    policy_used: Option<ContentHash>,
}

impl Serialize for Meta {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        MetaWire {
            provenance: self.provenance.clone(),
            guarantee: self.guarantee,
            bound: self.bound.clone(),
            sparsity: self.sparsity,
            physical: self.physical,
            reconstruction: self.reconstruction.clone(),
            policy_used: self.policy_used.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Meta {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = MetaWire::deserialize(deserializer)?;
        // Re-run M-I1…M-I4 (+ numeric well-formedness): wire data is never silently trusted.
        // (`ReconInfo`'s own `Deserialize` has already re-run its schema invariants.)
        let meta = Meta::new(
            w.provenance,
            w.guarantee,
            w.bound,
            w.sparsity,
            w.physical,
            w.policy_used,
        )
        .map_err(serde::de::Error::custom)?;
        Ok(match w.reconstruction {
            Some(r) => meta.with_reconstruction(*r),
            None => meta,
        })
    }
}

/// The M-I1…M-I4 guarantee/bound consistency check.
fn check_guarantee_bound(g: GuaranteeStrength, bound: Option<&Bound>) -> Result<(), WfError> {
    use GuaranteeStrength::{Declared, Empirical, Exact, Proven};
    let basis_ok =
        |b: Option<&Bound>, want_proven: bool, want_empirical: bool| match b.map(|b| &b.basis) {
            Some(BoundBasis::ProvenThm { .. }) => want_proven,
            Some(BoundBasis::EmpiricalFit { .. }) => want_empirical,
            Some(BoundBasis::UserDeclared) => !want_proven && !want_empirical,
            None => false,
        };
    match g {
        Exact => {
            if bound.is_none() {
                Ok(())
            } else {
                Err(WfError::GuaranteeBoundMismatch) // M-I1
            }
        }
        Proven if basis_ok(bound, true, false) => Ok(()), // M-I2
        Empirical if basis_ok(bound, false, true) => Ok(()), // M-I3
        Declared if basis_ok(bound, false, false) => Ok(()), // M-I4
        _ => Err(WfError::GuaranteeBoundMismatch),
    }
}
