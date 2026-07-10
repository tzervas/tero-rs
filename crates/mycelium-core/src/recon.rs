//! The **reconstruction manifest** (`ReconInfo`) — `Meta.reconstruction` (M-260; RFC-0001 §4.3;
//! RFC-0003 §6, normative; `reconstruction-manifest.schema.json`, the ratified name).
//!
//! Explicitly and inspectably distinguishes (RFC-0003 §6):
//! - **Indexed retrieval** — codebook + similarity + threshold; returns a *stored atom*;
//!   bounded-lossy. NOT holographic reconstruction.
//! - **True compositional reconstruction** — requires the structural **recipe / role schema**
//!   (which ops combined which slots) + algebraic inverse operations; can recover *novel*
//!   combinations never stored — VSA's defining capability over a hash table.
//!
//! The kernel carries only this *data type* (RFC-0003 §2 — "its metadata fields" stay in core);
//! constructing manifests and executing decode procedures is the VSA submodule's business
//! (ADR-008). The wire form is exactly the ratified schema; `Deserialize` re-runs
//! [`ReconInfo::new`]'s invariants, so a malformed manifest is rejected, never silently trusted
//! (the M-104 discipline).
//!
//! Per the schema's ratified comment, a **resonator** decode is Phase-3 exploratory and
//! **probabilistic-only** (FR-C2): its bound basis must not exceed `Empirical` — enforced here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bound::Bound;
use crate::id::ContentHash;
use crate::{GuaranteeStrength, WfError};

/// Which capability the manifest supports (RFC-0003 §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReconMode {
    /// Codebook + similarity + threshold; returns a stored atom (bounded-lossy).
    IndexedRetrieval,
    /// Recipe + algebraic inverses; can recover novel combinations.
    CompositionalReconstruction,
}

/// The compositional recipe / role schema: which ops combined which slots. `structure` maps each
/// role name to the content hash of its role atom (an inspectable object, per the schema).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    /// The role names.
    pub roles: Vec<String>,
    /// Role name → content hash of the role atom.
    pub structure: BTreeMap<String, ContentHash>,
}

/// The decoding procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodeProcedure {
    /// Nearest-atom cleanup against the codebook(s).
    Cleanup,
    /// Resonator factorization (Phase-3 exploratory; probabilistic-only — FR-C2).
    Resonator,
}

/// The per-slot cleanup projection a resonator decode uses (RFC-0003 §6.1; RFC-0009 §3/§9 Q2).
/// A metadata-only manifest field — the submodule implements the projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupShape {
    /// Softmax-weighted superposition over the codebook (the standard resonator cleanup).
    Softmax,
    /// Winner-take-all: the single arg-max atom.
    ArgMax,
}

/// The resonator initialisation strategy (RFC-0003 §6.1; RFC-0009 §9 Q1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitStrategy {
    /// Equal-weight superposition of all codebook atoms per slot (the Frady "uniform" start).
    UniformSuperposition,
    /// A single seeded codebook atom per slot.
    SeededGuess,
}

/// Decoding procedure + parameters: a cleanup threshold (indexed/cleanup) or a resonator factor
/// structure + iteration budget (RFC-0003 §6). Optional fields are omitted from the wire form
/// when absent, matching the schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeSpec {
    /// The procedure.
    pub procedure: DecodeProcedure,
    /// Minimum acceptable cleanup confidence in `[0, 1]` (required for `Cleanup`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup_threshold: Option<f64>,
    /// Per-factor codebook references (required for `Resonator`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factors: Option<Vec<ContentHash>>,
    /// Resonator iteration budget (required for `Resonator`; ≥ 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration_budget: Option<u64>,
    /// Resonator per-slot cleanup projection (RFC-0003 §6.1; `Resonator` only). Default `Softmax`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup: Option<CleanupShape>,
    /// Softmax inverse-temperature `β > 0` (RFC-0003 §6.1; meaningful when `cleanup == Softmax`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f64>,
    /// Per-slot top-similarity lock threshold `∈ [0, 1]` for the convergence verdict (RFC-0003 §6.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau_lock: Option<f64>,
    /// Resonator initialisation strategy (RFC-0003 §6.1). Default `UniformSuperposition`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<InitStrategy>,
    /// Initialisation seed for reproducibility (RFC-0003 §6.1; RFC-0009 §6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// The reconstruction manifest. Fields are private; the only constructor, [`ReconInfo::new`],
/// enforces the schema invariants — a malformed manifest is unrepresentable.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconInfo {
    mode: ReconMode,
    model: String,
    dim: u32,
    codebooks: Vec<ContentHash>,
    recipe: Option<Recipe>,
    decode: DecodeSpec,
    bound: Bound,
}

impl ReconInfo {
    /// Build a manifest, enforcing the schema invariants (RFC-0003 §6;
    /// `reconstruction-manifest.schema.json`):
    ///
    /// - `model` non-empty, `dim ≥ 1`, `codebooks` non-empty (content-addressed references);
    /// - `CompositionalReconstruction` **requires** a recipe; `IndexedRetrieval` must not carry
    ///   one (absent on the wire);
    /// - `Cleanup` requires `cleanup_threshold ∈ [0, 1]`; `Resonator` requires non-empty
    ///   `factors` + `iteration_budget ≥ 1` **and** a bound basis no stronger than
    ///   `EmpiricalFit` (probabilistic-only, FR-C2);
    /// - the attached `{ε, δ, strength}` bound must be numerically well-formed.
    pub fn new(
        mode: ReconMode,
        model: impl Into<String>,
        dim: u32,
        codebooks: Vec<ContentHash>,
        recipe: Option<Recipe>,
        decode: DecodeSpec,
        bound: Bound,
    ) -> Result<Self, WfError> {
        let model = model.into();
        if model.is_empty() || dim == 0 || codebooks.is_empty() {
            return Err(WfError::MalformedReconstruction);
        }
        match mode {
            ReconMode::CompositionalReconstruction if recipe.is_none() => {
                return Err(WfError::MalformedReconstruction)
            }
            ReconMode::IndexedRetrieval if recipe.is_some() => {
                return Err(WfError::MalformedReconstruction)
            }
            _ => {}
        }
        match decode.procedure {
            DecodeProcedure::Cleanup => match decode.cleanup_threshold {
                Some(t) if (0.0..=1.0).contains(&t) => {}
                _ => return Err(WfError::MalformedReconstruction),
            },
            DecodeProcedure::Resonator => {
                let factors_ok = decode.factors.as_ref().is_some_and(|f| !f.is_empty());
                let budget_ok = decode.iteration_budget.is_some_and(|b| b >= 1);
                if !factors_ok || !budget_ok {
                    return Err(WfError::MalformedReconstruction);
                }
                // Probabilistic-only (FR-C2): a resonator decode's basis must not *exceed*
                // `Empirical`. Expressed via the lattice rank, not `matches!(ProvenThm)`, so the
                // rule stays correct if a stronger basis variant is ever added (A1-04). A weaker
                // basis (`UserDeclared`/`Declared`, rank ≥ Empirical's) is allowed; only a
                // stronger one (smaller rank, e.g. `ProvenThm`) is rejected.
                if bound.basis.strength().rank() < GuaranteeStrength::Empirical.rank() {
                    return Err(WfError::MalformedReconstruction);
                }
                // A6-06: `cleanup_threshold` is not required on a Resonator path, but the schema
                // bounds it to `[0, 1]` *whenever present*. A stray out-of-range value must be an
                // explicit rejection here too — never silently accepted (the never-silent rule) —
                // so the Rust contract matches the schema's range constraint on the optional field.
                if let Some(t) = decode.cleanup_threshold {
                    if !(0.0..=1.0).contains(&t) {
                        return Err(WfError::MalformedReconstruction);
                    }
                }
                // r4 (RFC-0003 §6.1 / RFC-0009 §4): the resonator decode params are optional and
                // additive, but range-checked *whenever present* — an out-of-range value is an
                // explicit rejection, never silently accepted (G2). `cleanup`/`init` are enums, so
                // the type already constrains them; only the numeric knobs need bounding.
                if let Some(beta) = decode.beta {
                    // Softmax inverse-temperature must be finite and strictly positive.
                    if !beta.is_finite() || beta <= 0.0 {
                        return Err(WfError::MalformedReconstruction);
                    }
                }
                if let Some(tau) = decode.tau_lock {
                    if !(0.0..=1.0).contains(&tau) {
                        return Err(WfError::MalformedReconstruction);
                    }
                }
            }
        }
        if !bound.well_formed() {
            return Err(WfError::MalformedBound);
        }
        Ok(ReconInfo {
            mode,
            model,
            dim,
            codebooks,
            recipe,
            decode,
            bound,
        })
    }

    /// Which capability this manifest supports.
    #[must_use]
    pub fn mode(&self) -> ReconMode {
        self.mode
    }
    /// The VSA model id (matches the producing `Repr.model`).
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
    /// Hypervector dimensionality.
    #[must_use]
    pub fn dim(&self) -> u32 {
        self.dim
    }
    /// The content-addressed codebook references.
    #[must_use]
    pub fn codebooks(&self) -> &[ContentHash] {
        &self.codebooks
    }
    /// The compositional recipe, if this manifest is compositional.
    #[must_use]
    pub fn recipe(&self) -> Option<&Recipe> {
        self.recipe.as_ref()
    }
    /// The decode procedure + parameters.
    #[must_use]
    pub fn decode(&self) -> &DecodeSpec {
        &self.decode
    }
    /// The attached `{ε, δ, strength}` bound certificate.
    #[must_use]
    pub fn bound(&self) -> &Bound {
        &self.bound
    }
}

/// The wire projection (`reconstruction-manifest.schema.json`): `recipe` is omitted when absent
/// (the `IndexedRetrieval` form); `Deserialize` re-runs the invariants. `deny_unknown_fields`
/// enforces the schema's `additionalProperties: false` (A6-02).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReconWire {
    mode: ReconMode,
    model: String,
    dim: u32,
    codebooks: Vec<ContentHash>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recipe: Option<Recipe>,
    decode: DecodeSpec,
    bound: Bound,
}

impl Serialize for ReconInfo {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ReconWire {
            mode: self.mode,
            model: self.model.clone(),
            dim: self.dim,
            codebooks: self.codebooks.clone(),
            recipe: self.recipe.clone(),
            decode: self.decode.clone(),
            bound: self.bound.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ReconInfo {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = ReconWire::deserialize(deserializer)?;
        // Wire data is never silently trusted (the M-104 discipline).
        ReconInfo::new(
            w.mode,
            w.model,
            w.dim,
            w.codebooks,
            w.recipe,
            w.decode,
            w.bound,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bound::{BoundBasis, BoundKind, NormKind};
    use crate::content::operation_hash;

    fn empirical_bound() -> Bound {
        Bound {
            kind: BoundKind::Probability { delta: 0.01 },
            basis: BoundBasis::EmpiricalFit {
                trials: 10_000,
                method: "test".to_owned(),
            },
        }
    }

    fn cleanup_decode() -> DecodeSpec {
        DecodeSpec {
            procedure: DecodeProcedure::Cleanup,
            cleanup_threshold: Some(0.2),
            factors: None,
            iteration_budget: None,
            cleanup: None,
            beta: None,
            tau_lock: None,
            init: None,
            seed: None,
        }
    }

    #[test]
    fn compositional_requires_a_recipe() {
        let err = ReconInfo::new(
            ReconMode::CompositionalReconstruction,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            cleanup_decode(),
            empirical_bound(),
        );
        assert_eq!(err.unwrap_err(), WfError::MalformedReconstruction);
    }

    #[test]
    fn indexed_must_not_carry_a_recipe() {
        let recipe = Recipe {
            roles: vec!["color".to_owned()],
            structure: BTreeMap::from([("color".to_owned(), operation_hash("role"))]),
        };
        let err = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            Some(recipe),
            cleanup_decode(),
            empirical_bound(),
        );
        assert_eq!(err.unwrap_err(), WfError::MalformedReconstruction);
    }

    #[test]
    fn resonator_is_probabilistic_only() {
        let proven = Bound {
            kind: BoundKind::Error {
                eps: 0.1,
                norm: NormKind::L2,
            },
            basis: BoundBasis::ProvenThm {
                citation: "nope".to_owned(),
            },
        };
        let err = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "FHRR",
            1024,
            vec![operation_hash("codebook")],
            None,
            DecodeSpec {
                procedure: DecodeProcedure::Resonator,
                cleanup_threshold: None,
                factors: Some(vec![operation_hash("factor")]),
                iteration_budget: Some(100),
                cleanup: None,
                beta: None,
                tau_lock: None,
                init: None,
                seed: None,
            },
            proven,
        );
        assert_eq!(err.unwrap_err(), WfError::MalformedReconstruction);
    }

    #[test]
    fn resonator_allows_basis_weaker_than_empirical() {
        // A1-04 mutant-witness: the resonator rule is "basis must not *exceed* Empirical", encoded
        // via the lattice rank — so a *weaker* basis (`UserDeclared`/`Declared`) is allowed, only a
        // stronger one (`ProvenThm`) is rejected. If the rule were `matches!(ProvenThm)` only this
        // would still pass; the companion `resonator_is_probabilistic_only` pins the rejection side.
        let declared = Bound {
            kind: BoundKind::Probability { delta: 0.05 },
            basis: BoundBasis::UserDeclared,
        };
        let ok = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "FHRR",
            1024,
            vec![operation_hash("codebook")],
            None,
            DecodeSpec {
                procedure: DecodeProcedure::Resonator,
                cleanup_threshold: None,
                factors: Some(vec![operation_hash("factor")]),
                iteration_budget: Some(100),
                cleanup: None,
                beta: None,
                tau_lock: None,
                init: None,
                seed: None,
            },
            declared,
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn resonator_range_checks_a_stray_cleanup_threshold() {
        // A6-06 mutant-witness: `cleanup_threshold` is optional on a Resonator path, but the schema
        // bounds it to [0, 1] *whenever present*. An out-of-range stray value must be an explicit
        // rejection (never silently accepted), matching the schema's constraint on the field.
        let bad = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "FHRR",
            1024,
            vec![operation_hash("codebook")],
            None,
            DecodeSpec {
                procedure: DecodeProcedure::Resonator,
                cleanup_threshold: Some(1.5), // out of [0, 1]
                factors: Some(vec![operation_hash("factor")]),
                iteration_budget: Some(100),
                cleanup: None,
                beta: None,
                tau_lock: None,
                init: None,
                seed: None,
            },
            empirical_bound(),
        );
        assert_eq!(bad.unwrap_err(), WfError::MalformedReconstruction);
        // An in-range stray threshold is still accepted (it is merely ignored by the procedure).
        let ok = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "FHRR",
            1024,
            vec![operation_hash("codebook")],
            None,
            DecodeSpec {
                procedure: DecodeProcedure::Resonator,
                cleanup_threshold: Some(0.4),
                factors: Some(vec![operation_hash("factor")]),
                iteration_budget: Some(100),
                cleanup: None,
                beta: None,
                tau_lock: None,
                init: None,
                seed: None,
            },
            empirical_bound(),
        );
        assert!(ok.is_ok());
    }

    /// r4 (RFC-0003 §6.1): the optional resonator params are range-checked whenever present —
    /// `beta > 0` (finite) and `tau_lock ∈ [0, 1]`. An out-of-range value is an explicit rejection,
    /// never silently accepted (G2); an in-range set (and the all-`None` default) is accepted.
    #[test]
    fn resonator_range_checks_optional_params() {
        let spec = |cleanup, beta, tau_lock, init, seed| DecodeSpec {
            procedure: DecodeProcedure::Resonator,
            cleanup_threshold: None,
            factors: Some(vec![operation_hash("factor")]),
            iteration_budget: Some(100),
            cleanup,
            beta,
            tau_lock,
            init,
            seed,
        };
        let build = |d| {
            ReconInfo::new(
                ReconMode::IndexedRetrieval,
                "MAP-I",
                1024,
                vec![operation_hash("codebook")],
                None,
                d,
                empirical_bound(),
            )
        };
        // A fully-specified, in-range param set is accepted.
        let ok = build(spec(
            Some(CleanupShape::Softmax),
            Some(4.0),
            Some(0.9),
            Some(InitStrategy::UniformSuperposition),
            Some(7),
        ));
        assert!(ok.is_ok(), "{ok:?}");
        // The all-None default (params omitted) is accepted — the fields are additive.
        assert!(build(spec(None, None, None, None, None)).is_ok());
        // β must be finite and strictly positive.
        for bad_beta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                build(spec(None, Some(bad_beta), None, None, None)).unwrap_err(),
                WfError::MalformedReconstruction,
                "beta={bad_beta} must be rejected"
            );
        }
        // τ_lock must lie in [0, 1].
        for bad_tau in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                build(spec(None, None, Some(bad_tau), None, None)).unwrap_err(),
                WfError::MalformedReconstruction,
                "tau_lock={bad_tau} must be rejected"
            );
        }
    }

    /// The new resonator params survive the wire round-trip and are omitted when absent (matching
    /// the schema's optional fields).
    #[test]
    fn resonator_params_round_trip_on_the_wire() {
        let info = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            DecodeSpec {
                procedure: DecodeProcedure::Resonator,
                cleanup_threshold: None,
                factors: Some(vec![operation_hash("factor")]),
                iteration_budget: Some(50),
                cleanup: Some(CleanupShape::Softmax),
                beta: Some(3.5),
                tau_lock: Some(0.85),
                init: Some(InitStrategy::UniformSuperposition),
                seed: Some(42),
            },
            empirical_bound(),
        )
        .unwrap();
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["decode"]["cleanup"], "Softmax");
        assert_eq!(json["decode"]["beta"], 3.5);
        assert_eq!(json["decode"]["tau_lock"], 0.85);
        assert_eq!(json["decode"]["init"], "UniformSuperposition");
        assert_eq!(json["decode"]["seed"], 42);
        let back: ReconInfo = serde_json::from_value(json).unwrap();
        assert_eq!(back, info);
        // Absent params are omitted from the wire form.
        let bare = serde_json::to_value(
            ReconInfo::new(
                ReconMode::IndexedRetrieval,
                "MAP-I",
                1024,
                vec![operation_hash("codebook")],
                None,
                DecodeSpec {
                    procedure: DecodeProcedure::Resonator,
                    cleanup_threshold: None,
                    factors: Some(vec![operation_hash("factor")]),
                    iteration_budget: Some(50),
                    cleanup: None,
                    beta: None,
                    tau_lock: None,
                    init: None,
                    seed: None,
                },
                empirical_bound(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(bare["decode"].get("beta").is_none(), "absent beta omitted");
        assert!(bare["decode"].get("seed").is_none(), "absent seed omitted");
    }

    #[test]
    fn wire_round_trips_and_rejects_malformed() {
        let info = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "MAP-I",
            1024,
            vec![operation_hash("codebook")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        let json = serde_json::to_value(&info).unwrap();
        // The ratified field names, exactly (reconstruction-manifest.schema.json).
        assert_eq!(json["mode"], "IndexedRetrieval");
        assert_eq!(json["model"], "MAP-I");
        assert_eq!(json["dim"], 1024);
        assert!(json["codebooks"].is_array());
        assert!(json.get("recipe").is_none(), "absent recipe is omitted");
        assert_eq!(json["decode"]["procedure"], "Cleanup");
        assert!(json["bound"]["delta"].is_number());
        let back: ReconInfo = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back, info);
        // A tampered wire manifest (compositional without a recipe) is rejected.
        let mut bad = json;
        bad["mode"] = "CompositionalReconstruction".into();
        assert!(serde_json::from_value::<ReconInfo>(bad).is_err());
    }

    // Mutant-witnesses for the accessor methods (recon.rs:221, 226, 231, 236):
    // - ReconInfo::model() returns "" or "xyzzy" when mutated
    // - ReconInfo::dim() returns 0 or 1 when mutated
    // - ReconInfo::codebooks() returns [] when mutated
    // - ReconInfo::recipe() returns None when mutated (for CompositionalReconstruction)
    // Each test directly asserts the accessor returns the value passed to new().
    #[test]
    fn accessors_return_the_constructed_values() {
        let codebook_ref = operation_hash("codebook-abc");
        let recipe = Recipe {
            roles: vec!["color".to_owned(), "shape".to_owned()],
            structure: BTreeMap::from([
                ("color".to_owned(), operation_hash("role-color")),
                ("shape".to_owned(), operation_hash("role-shape")),
            ]),
        };
        // Use CompositionalReconstruction to exercise the recipe() path.
        let info = ReconInfo::new(
            ReconMode::CompositionalReconstruction,
            "FHRR",
            4096,
            vec![codebook_ref.clone()],
            Some(recipe.clone()),
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();

        // model() must return the exact string, not "" or "xyzzy".
        assert_eq!(info.model(), "FHRR");
        // dim() must return 4096, not 0 or 1.
        assert_eq!(info.dim(), 4096);
        // codebooks() must return the actual slice, not [].
        assert_eq!(info.codebooks().len(), 1);
        assert_eq!(info.codebooks()[0], codebook_ref);
        // recipe() must return Some(&recipe) for CompositionalReconstruction.
        assert!(
            info.recipe().is_some(),
            "recipe() must be Some for CompositionalReconstruction"
        );
        assert_eq!(info.recipe().unwrap(), &recipe);
        // mode() and bound() correctness (not mutated, but pin the accessors).
        assert_eq!(info.mode(), ReconMode::CompositionalReconstruction);

        // For IndexedRetrieval, recipe() must be None.
        let indexed = ReconInfo::new(
            ReconMode::IndexedRetrieval,
            "MAP-B",
            512,
            vec![operation_hash("cb")],
            None,
            cleanup_decode(),
            empirical_bound(),
        )
        .unwrap();
        assert_eq!(indexed.model(), "MAP-B");
        assert_eq!(indexed.dim(), 512);
        assert!(indexed.recipe().is_none());
    }
}
