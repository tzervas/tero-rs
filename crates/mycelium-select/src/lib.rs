//! `mycelium-select` — the **selection-policy language** (E2-6; RFC-0005; ADR-006; G2):
//! automatic, metadata-driven representation selection that is *analyzable by construction,
//! never opaque*.
//!
//! The design is RFC-0005 §2's, verbatim:
//!
//! 1. **Form** — a [`SelectionPolicy`] is an ordered **decision table**: `(predicate over
//!    queryable [`Meta`]) → candidate` rules over a **finite** candidate set, with an explicit
//!    [`CostModel`] and a mandatory default arm. Predicates are a small, non-Turing-complete
//!    structural language ([`Predicate`]) — evaluation is structural recursion on finite data, so
//!    every policy is **total and terminating by construction** (the expressiveness ceiling is the
//!    feature).
//! 2. **Mandatory EXPLAIN** — every selection emits an [`Explanation`] `{inputs considered, cost
//!    of each candidate, matched rule, chosen option, override state}` ([`select`]/[`explain`];
//!    M-221). No selection happens without one.
//! 3. **Determinism** — same [`SelectionInputs`] in → same choice out; ties in [`Action::Cheapest`]
//!    break to the lowest candidate index; rule conflicts resolve by **fixed declared precedence**
//!    (table order, first match wins).
//! 4. **Override** — a forced candidate is first-class and deterministic ([`select`]'s `forced`
//!    argument), recorded in the EXPLAIN trace.
//! 5. **Exact statistics** — the only inputs are the kernel's *exact* metadata (bounds, `dtype`,
//!    sparsity, guarantee), never sampled estimates — the principal source of optimizer opacity
//!    does not arise (RFC-0005 §2.5).
//!
//! A policy is **content-addressed** ([`SelectionPolicy::policy_ref`], RFC-0005 §3): the
//! `PolicyRef` recorded in `Meta.policy_used` is the hash of the policy's canonical serialization,
//! so "which policy chose this, and what does that policy do?" is always answerable.
//!
//! **One mechanism, two sites** (RFC-0005 §4): [`select_swap_target`] (RFC-0002 swap targets) and
//! [`select_packing`] (RFC-0004 §5 packing schedules, consumed by E2-7/M-250) are thin adapters
//! over the single [`select`] — no parallel mechanisms.
//!
//! This crate is deliberately its own crate, outside the trusted kernel (KC-3 / SoC; the
//! `phase-2.md` §5 sequencing decision): it depends on `mycelium-core` only.
//!
//! **Trusted-base discipline (ADR-014 / DN-21 §5 F-1):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use mycelium_core::{
    operation_hash, Bound, BoundKind, ContentHash, GuaranteeStrength, Meta, PackScheme,
    PhysicalLayout, Repr, ScalarKind, SparsityClass, SparsityObs, Value,
};

/// The paradigm kinds, as a predicate-level discriminator (RFC-0001 §4.1; RFC-0032 D3 added `Seq`
/// and D4 added `Bytes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParadigmKind {
    /// `Repr::Binary`.
    Binary,
    /// `Repr::Ternary`.
    Ternary,
    /// `Repr::Dense`.
    Dense,
    /// `Repr::Vsa`.
    Vsa,
    /// `Repr::Seq` — the indexed homogeneous sequence (RFC-0032 D3; M-749).
    Seq,
    /// `Repr::Bytes` — the byte string (RFC-0032 D4; M-750).
    Bytes,
    /// `Repr::Float` — the scalar IEEE-754 float (ADR-040; M-896).
    Float,
}

fn kind_of(repr: &Repr) -> ParadigmKind {
    match repr {
        Repr::Binary { .. } => ParadigmKind::Binary,
        Repr::Ternary { .. } => ParadigmKind::Ternary,
        Repr::Dense { .. } => ParadigmKind::Dense,
        Repr::Vsa { .. } => ParadigmKind::Vsa,
        Repr::Seq { .. } => ParadigmKind::Seq,
        Repr::Bytes => ParadigmKind::Bytes,
        Repr::Float { .. } => ParadigmKind::Float,
    }
}

/// The **exact decode facts** the RFC-0010 decode site queries — generic integers/booleans about a
/// factorization request, *not* VSA types (this crate stays `mycelium-core`-only). All are exact: the
/// factor count, the operational capacity `∏ᵢ kᵢ` (saturating), the dimension, and whether the request
/// sits inside the resonator's validated regime (the `mycelium-vsa` adapter precomputes `in_regime`
/// from `ResonatorProfile::check` — a fact about the inputs, like sparsity, not a sampled estimate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeFacts {
    /// The factor count `F` (number of codebooks).
    pub factors: u32,
    /// The operational capacity `∏ᵢ kᵢ` (saturating product of the codebook sizes).
    pub capacity: u128,
    /// The hypervector dimension `d`.
    pub dim: u32,
    /// Whether the request is inside the resonator's validated `{F, ∏kᵢ, d}` regime.
    pub in_regime: bool,
}

/// The **queryable inputs** a policy may inspect — drawn from a value's [`Repr`] + [`Meta`]
/// (RFC-0005 §2: bounds, `dtype`, sparsity class, guarantee; *exact* metadata, never sampled
/// estimates), plus the optional [`DecodeFacts`] for the RFC-0010 decode site. Serializable so an
/// [`Explanation`] can carry exactly what was considered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionInputs {
    /// The source representation.
    pub src: Repr,
    /// The disclosed guarantee strength.
    pub guarantee: GuaranteeStrength,
    /// The bound, if the value is approximate.
    pub bound: Option<Bound>,
    /// Measured sparsity, if recorded.
    pub sparsity: Option<SparsityObs>,
    /// The decode-site facts (RFC-0010), absent for the swap/packing sites.
    #[serde(default)]
    pub decode: Option<DecodeFacts>,
}

impl SelectionInputs {
    /// The queryable projection of a `(Repr, Meta)` pair (no decode facts — swap/packing sites).
    #[must_use]
    pub fn from_meta(src: Repr, meta: &Meta) -> Self {
        SelectionInputs {
            src,
            guarantee: meta.guarantee(),
            bound: meta.bound().cloned(),
            sparsity: meta.sparsity(),
            decode: None,
        }
    }

    /// The queryable projection of a [`Value`].
    #[must_use]
    pub fn of_value(v: &Value) -> Self {
        Self::from_meta(v.repr().clone(), v.meta())
    }

    /// Attach decode-site facts (RFC-0010) for the [`select_decode_method`] adapter.
    #[must_use]
    pub fn with_decode(mut self, facts: DecodeFacts) -> Self {
        self.decode = Some(facts);
        self
    }
}

/// The predicate language — small, closed, **not Turing-complete**: no loops, no recursion in the
/// language (only finite structural nesting), no arithmetic beyond comparison against literals.
/// Evaluation ([`Predicate::eval`]) is structural recursion on finite data → total and terminating
/// on every input (RFC-0005 §2.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    /// Always true — the explicit catch-all form.
    Always,
    /// The source paradigm kind is exactly this.
    SrcKindIs(ParadigmKind),
    /// The source is `Dense` with exactly this element dtype.
    DtypeIs(ScalarKind),
    /// The disclosed guarantee is at least this strong (lattice rank ≤).
    GuaranteeAtLeast(GuaranteeStrength),
    /// The value carries an `Error` bound with `eps ≤` this. `false` when there is no bound or
    /// the bound is of another kind (the predicate asks for *checked* evidence, not its absence).
    ErrorEpsAtMost(f64),
    /// The source is a `Vsa` with a declared `Sparse` class.
    DeclaredSparse,
    /// (Decode site, RFC-0010) the operational capacity `∏ᵢ kᵢ` is at most this — the enumeration
    /// budget that makes brute force tractable. `false` when there are no decode facts.
    CapacityAtMost(u128),
    /// (Decode site, RFC-0010) the factor count `F` is at most this. `false` when no decode facts.
    FactorsAtMost(u32),
    /// (Decode site, RFC-0010) the request is inside the resonator's validated regime. `false` when
    /// there are no decode facts (the predicate asks for *present* in-regime evidence).
    InResonatorRegime,
    /// Conjunction.
    All(Vec<Predicate>),
    /// Disjunction.
    Any(Vec<Predicate>),
    /// Negation.
    Not(Box<Predicate>),
}

impl Predicate {
    /// Evaluate against the queryable inputs — total: every predicate yields a boolean on every
    /// input, no partiality, no side effects.
    #[must_use]
    pub fn eval(&self, inputs: &SelectionInputs) -> bool {
        match self {
            Predicate::Always => true,
            Predicate::SrcKindIs(k) => kind_of(&inputs.src) == *k,
            Predicate::DtypeIs(dt) => {
                matches!(&inputs.src, Repr::Dense { dtype, .. } if dtype == dt)
            }
            Predicate::GuaranteeAtLeast(g) => inputs.guarantee.rank() <= g.rank(),
            Predicate::ErrorEpsAtMost(x) => matches!(
                inputs.bound.as_ref().map(|b| &b.kind),
                Some(BoundKind::Error { eps, .. }) if eps <= x
            ),
            Predicate::DeclaredSparse => matches!(
                &inputs.src,
                Repr::Vsa {
                    sparsity: SparsityClass::Sparse { .. },
                    ..
                }
            ),
            Predicate::CapacityAtMost(c) => {
                matches!(inputs.decode, Some(d) if d.capacity <= *c)
            }
            Predicate::FactorsAtMost(f) => {
                matches!(inputs.decode, Some(d) if d.factors <= *f)
            }
            Predicate::InResonatorRegime => matches!(inputs.decode, Some(d) if d.in_regime),
            Predicate::All(ps) => ps.iter().all(|p| p.eval(inputs)),
            Predicate::Any(ps) => ps.iter().any(|p| p.eval(inputs)),
            Predicate::Not(p) => !p.eval(inputs),
        }
    }

    /// True iff every floating-point literal in the predicate tree is finite (A5-01/B2-02). A
    /// non-finite `ErrorEpsAtMost` literal serializes to JSON `null`, so two materially different
    /// policies (e.g. `eps ≤ NaN`, which never matches, vs `eps ≤ +∞`, which always matches) would
    /// hash to the **same** content-addressed `policy_ref` — breaking the audit anchor recorded in
    /// `Meta.policy_used` (RFC-0005 §3). [`SelectionPolicy::new`] rejects a policy that violates this.
    #[must_use]
    pub fn literals_finite(&self) -> bool {
        match self {
            Predicate::ErrorEpsAtMost(x) => x.is_finite(),
            Predicate::All(ps) | Predicate::Any(ps) => ps.iter().all(Predicate::literals_finite),
            Predicate::Not(p) => p.literals_finite(),
            Predicate::Always
            | Predicate::SrcKindIs(_)
            | Predicate::DtypeIs(_)
            | Predicate::GuaranteeAtLeast(_)
            | Predicate::DeclaredSparse
            | Predicate::CapacityAtMost(_)
            | Predicate::FactorsAtMost(_)
            | Predicate::InResonatorRegime => true,
        }
    }
}

/// A decode methodology — the **third** RFC-0005 §4 site (RFC-0010): how a value is decoded back to
/// its constituents. An abstract tag (no VSA types leak into this core-only crate); the
/// `mycelium-vsa` executor maps it to the concrete decode (brute-force enumeration / resonator loop /
/// refusal) and reads the guarantee tag off the chosen arm (RFC-0010 §4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodeMethod {
    /// Exhaustive enumeration of all `∏ᵢ kᵢ` codebook combinations — `Exact` (it *is* the
    /// brute-force oracle), tractable only when the operational capacity is small.
    BruteForceExact,
    /// The iterative resonator loop (RFC-0009) — `Empirical`, only inside its validated regime.
    Resonator,
    /// Neither applies (too big to enumerate *and* outside the resonator regime): an explicit
    /// refusal, never a silent best-effort guess (G2).
    Refuse,
}

/// A placement candidate — the identity of one member of a `forage` candidate set (RFC-0008 RT3;
/// DN-63 §3.5; D-lite, M-906). v0/D-lite is **single-node**: a `NodeRef` names a local
/// worker/execution slot, never a network node — multi-node placement is the DN-70 §5 R-6 H2
/// residual. Content-neutral (a display-only identity); no affinity/capability data yet (DN-63
/// FLAG-13 stays open for the full node-level signal set).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeRef(pub String);

/// A selectable candidate — the RFC-0005 §4 sites share one vocabulary (one mechanism). `Node`
/// (M-906, DN-63 §3.5) is the **fourth** site, added the same additive way [`Candidate::Decode`]
/// was: no change to [`select`]'s engine, just a new candidate/site pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Candidate {
    /// A swap-target representation (the RFC-0002 site).
    Repr(Repr),
    /// A packing scheme (the RFC-0004 §5 site; consumed by E2-7/M-250).
    Packing(PackScheme),
    /// A decode methodology (the RFC-0010 site; executed by `mycelium-vsa`).
    Decode(DecodeMethod),
    /// A placement target (the `forage` site; RFC-0008 RT3; DN-63 §3.5; M-906 D-lite).
    Node(NodeRef),
}

/// What a matched rule does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Choose the candidate at this index in the policy's candidate list.
    Choose(usize),
    /// Choose the candidate minimizing the explicit [`CostModel`]; ties break deterministically
    /// to the lowest index.
    Cheapest,
}

/// One row of the decision table: `when` (a [`Predicate`]) → `action`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    /// The guard over the queryable inputs.
    pub when: Predicate,
    /// What to do when the guard holds (first matching row wins — fixed declared precedence).
    pub action: Action,
}

/// The **explicit cost function** (RFC-0005 §2.1): cost = `storage_weight ×` the candidate's
/// storage footprint in **bits** — a real, declared unit, not "arbitrary internal units detached
/// from hardware" (the documented optimizer failure mode RFC-0005 §2 warns about). Footprints:
///
/// - `Repr` candidates: total payload bits — `Binary{w}` = `w`; `Ternary{t}` = `2t` (the DN-01
///   two-bit-per-trit reference packing); `Dense{dim, dtype}` = `dim × dtype bits`;
///   `Vsa` dense = `dim × 64` (f64 components as stored), declared-`Sparse{max_active}` =
///   `max_active × 96` (a 32-bit index + an f64 value per active component).
/// - `Packing` candidates: bits/element × the source's element count — `Unpacked` = 8,
///   `TwoBitPerTrit`/`I2S`/`TL1` = 2.0, `FiveTritPerByte` = 1.6, `TL2` = 1.67 (the RFC-0004 §5 /
///   DN-01 figures).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostModel {
    /// Multiplier on the storage-bits footprint (must be finite and `> 0`).
    pub storage_weight: f64,
}

fn dtype_bits(dt: ScalarKind) -> f64 {
    match dt {
        ScalarKind::F16 | ScalarKind::Bf16 => 16.0,
        ScalarKind::F32 => 32.0,
        ScalarKind::F64 => 64.0,
    }
}

fn repr_storage_bits(repr: &Repr) -> f64 {
    match repr {
        Repr::Binary { width } => f64::from(*width),
        Repr::Ternary { trits } => 2.0 * f64::from(*trits),
        Repr::Dense { dim, dtype } => f64::from(*dim) * dtype_bits(*dtype),
        Repr::Vsa { dim, sparsity, .. } => match sparsity {
            SparsityClass::Dense => f64::from(*dim) * 64.0,
            SparsityClass::Sparse { max_active } => f64::from(*max_active) * 96.0,
        },
        // RFC-0032 D3 (M-749): a homogeneous sequence stores `len` copies of its element repr's
        // footprint — recursing through the same exact storage model (a total, finite cost).
        Repr::Seq { elem, len } => f64::from(*len) * repr_storage_bits(elem),
        // RFC-0032 D4 (M-750): a byte string carries no static length in its `Repr` (the byte count
        // rides the payload), so its footprint is not statically derivable here — `0.0` is the
        // honest static lower bound (the real cost is payload-dependent, not a Repr property). The
        // selection table does not currently produce `Bytes` candidates, so this only fills an
        // EXPLAIN cost line if ever queried.
        Repr::Bytes => 0.0,
        // ADR-040 §2.1 (M-896): a scalar float's static footprint is exactly its frozen width —
        // binary64 = 64 bits (the only registered width; a future width adds its own arm here by
        // construction, never a silent default).
        Repr::Float { width } => match width {
            mycelium_core::FloatWidth::F64 => 64.0,
        },
    }
}

fn packing_bits_per_element(scheme: PackScheme) -> f64 {
    match scheme {
        PackScheme::Unpacked => 8.0,
        PackScheme::TwoBitPerTrit | PackScheme::I2S | PackScheme::Tl1 => 2.0,
        PackScheme::FiveTritPerByte => 1.6,
        // A5-08 — RESOLVED (M-360 real-layout increment): this 1.67 b/w now *matches* the codec.
        // `mycelium-mlir::pack` realizes TL2 as the true bitnet.cpp layout (3 trits → a 5-bit
        // LUT-index, bit-packed ⇒ 5/3 ≈ 1.67 b/w), distinct from `FiveTritPerByte`'s 1.6-b/w base-3
        // reference; the prior placeholder that made them coincide is retired. The native TL2 dot
        // kernel decodes this layout. See the matching note in `mycelium-mlir/src/pack.rs`.
        PackScheme::Tl2 => 1.67,
    }
}

fn src_elements(repr: &Repr) -> f64 {
    match repr {
        Repr::Binary { width } => f64::from(*width),
        Repr::Ternary { trits } => f64::from(*trits),
        Repr::Dense { dim, .. } | Repr::Vsa { dim, .. } => f64::from(*dim),
        // RFC-0032 D3 (M-749): the element count of a sequence is its declared `len`.
        Repr::Seq { len, .. } => f64::from(*len),
        // RFC-0032 D4 (M-750): a byte string's element count is not in its `Repr` (it rides the
        // payload) — `0.0` is the honest static lower bound, like its storage footprint above.
        Repr::Bytes => 0.0,
        // ADR-040 (M-896): a scalar float is exactly one element.
        Repr::Float { .. } => 1.0,
    }
}

impl CostModel {
    /// The deterministic cost of `candidate` given `inputs` — total, finite for every well-formed
    /// policy/input pair.
    #[must_use]
    pub fn cost(&self, candidate: &Candidate, inputs: &SelectionInputs) -> f64 {
        let bits = match candidate {
            Candidate::Repr(r) => repr_storage_bits(r),
            Candidate::Packing(s) => packing_bits_per_element(*s) * src_elements(&inputs.src),
            // Decode site (RFC-0010 §4.3): the cost is the real operation count, not storage bits.
            // Brute force enumerates all `∏ᵢ kᵢ` combinations (its cost *grows* with capacity — why it
            // is chosen only when capacity is small); the resonator is iterative and
            // capacity-independent, and `Refuse` does no work — both 0. The decode table decides by
            // explicit predicate arms (not `Cheapest`), so this only fills the EXPLAIN cost lines.
            Candidate::Decode(m) => match m {
                DecodeMethod::BruteForceExact => inputs.decode.map_or(0.0, |d| d.capacity as f64),
                DecodeMethod::Resonator | DecodeMethod::Refuse => 0.0,
            },
            // The `forage` site (M-906 D-lite): placement has no storage-bits meaning, so this is
            // `0.0` for every `Node` candidate — the D-lite policy always decides via
            // `Action::Choose`/the default arm (never `Cheapest`), so this line only fills the
            // mandatory EXPLAIN's cost column honestly ("not costed"). A real placement cost
            // function (network distance, occupancy-weighted…) is DN-63 FLAG-13/DN-70 §5 R-6 — H2.
            Candidate::Node(_) => 0.0,
        };
        self.storage_weight * bits
    }
}

/// Why a policy could not be constructed — validated up front so every constructed policy is
/// total by construction (no dangling rule indices, no empty candidate set, no degenerate cost).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// The candidate set is empty — a selection over nothing is not total.
    NoCandidates,
    /// A rule's `Choose(i)` or the default arm points outside the candidate list.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
    },
    /// The cost weight is non-finite or non-positive.
    BadCost,
    /// A rule predicate carries a non-finite `f64` literal (e.g. `ErrorEpsAtMost(NaN/∞)`), which
    /// would collide distinct policies under content addressing (A5-01).
    BadPredicateLiteral,
}

impl core::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolicyError::NoCandidates => write!(f, "a policy needs at least one candidate"),
            PolicyError::IndexOutOfRange { index } => {
                write!(
                    f,
                    "rule/default index {index} is outside the candidate list"
                )
            }
            PolicyError::BadCost => write!(f, "cost weight must be finite and > 0"),
            PolicyError::BadPredicateLiteral => {
                write!(
                    f,
                    "predicate float literals must be finite (else policy refs collide)"
                )
            }
        }
    }
}

impl std::error::Error for PolicyError {}

/// A **reified selection policy** (ADR-006; RFC-0005 §2/§3): an ordered decision table over a
/// finite candidate set with an explicit cost model and a mandatory default arm. First-class,
/// inspectable, diffable, and content-addressed ([`Self::policy_ref`]).
///
/// Fields are private; the only constructor, [`Self::new`], validates every index so a constructed
/// policy is total by construction, and `Deserialize` re-validates (wire data is never silently
/// trusted — the Phase-1 house pattern).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionPolicy {
    name: String,
    candidates: Vec<Candidate>,
    rules: Vec<Rule>,
    default_choice: usize,
    cost: CostModel,
}

#[derive(Deserialize)]
struct PolicyWire {
    name: String,
    candidates: Vec<Candidate>,
    rules: Vec<Rule>,
    default_choice: usize,
    cost: CostModel,
}

impl<'de> Deserialize<'de> for SelectionPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = PolicyWire::deserialize(deserializer)?;
        SelectionPolicy::new(w.name, w.candidates, w.rules, w.default_choice, w.cost)
            .map_err(serde::de::Error::custom)
    }
}

impl SelectionPolicy {
    /// Build a policy, validating totality up front: at least one candidate, every `Choose(i)`
    /// and the default arm in range, a sane cost weight.
    pub fn new(
        name: impl Into<String>,
        candidates: Vec<Candidate>,
        rules: Vec<Rule>,
        default_choice: usize,
        cost: CostModel,
    ) -> Result<Self, PolicyError> {
        if candidates.is_empty() {
            return Err(PolicyError::NoCandidates);
        }
        if !(cost.storage_weight.is_finite() && cost.storage_weight > 0.0) {
            return Err(PolicyError::BadCost);
        }
        let in_range = |i: usize| i < candidates.len();
        if !in_range(default_choice) {
            return Err(PolicyError::IndexOutOfRange {
                index: default_choice,
            });
        }
        for r in &rules {
            if !r.when.literals_finite() {
                return Err(PolicyError::BadPredicateLiteral);
            }
            if let Action::Choose(i) = r.action {
                if !in_range(i) {
                    return Err(PolicyError::IndexOutOfRange { index: i });
                }
            }
        }
        Ok(SelectionPolicy {
            name: name.into(),
            candidates,
            rules,
            default_choice,
            cost,
        })
    }

    /// The policy's display name (not part of selection semantics, but part of its identity).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// The finite candidate set.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }
    /// The ordered decision table.
    #[must_use]
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
    /// The mandatory default arm (totality).
    #[must_use]
    pub fn default_choice(&self) -> usize {
        self.default_choice
    }
    /// The explicit cost model.
    #[must_use]
    pub fn cost_model(&self) -> CostModel {
        self.cost
    }

    /// The **content address** of this policy (RFC-0005 §3; RFC-0001 §4.6): the hash of its
    /// canonical (serde-canonical JSON, fixed field order, domain-prefixed) serialization. This is
    /// the `PolicyRef` a swap records in `Meta.policy_used` — "which policy chose this?" is always
    /// answerable by hash.
    #[must_use]
    pub fn policy_ref(&self) -> ContentHash {
        let canonical = serde_json::to_string(self)
            .expect("a validated policy serializes (no non-string map keys)");
        operation_hash(&format!("selection-policy.v1:{canonical}"))
    }
}

/// The per-candidate cost line of an [`Explanation`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateCost {
    /// The candidate.
    pub candidate: Candidate,
    /// Its explicit cost under the policy's [`CostModel`].
    pub cost: f64,
}

/// The **mandatory EXPLAIN record** (M-221; RFC-0005 §2.2/§4): emitted on *every* selection —
/// inputs considered, the cost of each candidate, which rule matched, what was chosen, and the
/// override state. Serializable, deterministic, and re-derivable from `(policy, inputs)` alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Explanation {
    /// The content address of the policy that decided ([`SelectionPolicy::policy_ref`]).
    pub policy: ContentHash,
    /// The policy's display name.
    pub policy_name: String,
    /// The queryable inputs that were considered.
    pub inputs: SelectionInputs,
    /// Every candidate with its explicit cost (the full ranking, not just the winner).
    pub costs: Vec<CandidateCost>,
    /// Index of the decision-table rule that fired; `None` when the default arm decided or an
    /// override bypassed the table.
    pub matched_rule: Option<usize>,
    /// Index of the chosen candidate.
    pub chosen_index: usize,
    /// The chosen candidate.
    pub chosen: Candidate,
    /// Whether a first-class override forced the choice (the deterministic override hook).
    pub overridden: bool,
}

/// Why a selection call failed — always explicit (G2), never a silent fallback choice.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectError {
    /// The forced override index is outside the candidate list.
    OverrideOutOfRange {
        /// The forced index.
        index: usize,
        /// The candidate-list length.
        candidates: usize,
    },
    /// The chosen candidate does not fit the call site (e.g. a `Packing` candidate at the
    /// swap-target site) — a site adapter refuses it rather than coercing.
    WrongSiteKind {
        /// The candidate the policy chose.
        chosen: Candidate,
        /// The site that refused it.
        site: &'static str,
    },
    /// A trit-packed layout was requested for a non-ternary source `Repr` (A5-02). A
    /// [`PhysicalLayout::TritPacked`] record only describes how *trits* sit in bytes (RFC-0004 §5;
    /// DN-01); recording it onto a `Binary`/`Dense`/`Vsa` value's `Meta` would be a silent, latent
    /// mis-tag (a layout that contradicts its own representation). The packing site refuses it
    /// rather than producing the unsound record.
    NonTernarySource {
        /// The source `Repr` that does not admit a trit packing.
        src: Repr,
    },
}

impl core::fmt::Display for SelectError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SelectError::OverrideOutOfRange { index, candidates } => {
                write!(
                    f,
                    "override index {index} out of range ({candidates} candidates)"
                )
            }
            SelectError::WrongSiteKind { chosen, site } => {
                write!(f, "candidate {chosen:?} does not fit the {site} site")
            }
            SelectError::NonTernarySource { src } => {
                write!(
                    f,
                    "a trit-packed layout cannot describe a non-ternary source {src:?}"
                )
            }
        }
    }
}

impl std::error::Error for SelectError {}

/// The **single selection entry point** (RFC-0005 §2; one mechanism for both §4 sites): evaluate
/// the decision table over `inputs` (or honor a first-class `forced` override) and return the
/// chosen candidate **with its mandatory [`Explanation`]** — there is no selection without an
/// EXPLAIN. Deterministic: same `(policy, inputs, forced)` → same result.
pub fn select(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced: Option<usize>,
) -> Result<(Candidate, Explanation), SelectError> {
    let costs: Vec<CandidateCost> = policy
        .candidates
        .iter()
        .map(|c| CandidateCost {
            candidate: c.clone(),
            cost: policy.cost.cost(c, inputs),
        })
        .collect();
    let (chosen_index, matched_rule, overridden) = if let Some(index) = forced {
        if index >= policy.candidates.len() {
            return Err(SelectError::OverrideOutOfRange {
                index,
                candidates: policy.candidates.len(),
            });
        }
        (index, None, true)
    } else {
        match policy
            .rules
            .iter()
            .enumerate()
            .find(|(_, r)| r.when.eval(inputs))
        {
            Some((ri, rule)) => {
                let idx = match rule.action {
                    Action::Choose(i) => i, // in range by construction (Self::new)
                    Action::Cheapest => cheapest(&costs),
                };
                (idx, Some(ri), false)
            }
            None => (policy.default_choice, None, false),
        }
    };
    let chosen = policy.candidates[chosen_index].clone();
    let explanation = Explanation {
        // A5-08 (perf nit): `policy_ref()` recomputes the policy's content address — a full
        // serialize + hash — on *every* `select` call, even though the policy is immutable for the
        // life of a `&SelectionPolicy`. Acceptable at the current scale (a ~5-candidate table), but
        // a memoized/cached ref (compute once at construction) is the obvious win if `select` ever
        // lands on a hot path. Behavior-neutral; left as a deliberate note, not changed here.
        policy: policy.policy_ref(),
        policy_name: policy.name.clone(),
        inputs: inputs.clone(),
        costs,
        matched_rule,
        chosen_index,
        chosen: chosen.clone(),
        overridden,
    };
    Ok((chosen, explanation))
}

/// Index of the minimum-cost candidate; ties break to the lowest index (deterministic).
fn cheapest(costs: &[CandidateCost]) -> usize {
    let mut best = 0;
    for (i, c) in costs.iter().enumerate().skip(1) {
        if c.cost < costs[best].cost {
            best = i;
        }
    }
    best
}

/// `explain(policy, meta) → trace` (RFC-0005 §4): the mandatory EXPLAIN, **total and
/// deterministic** — un-overridden selection cannot fail on a validated policy, so this returns
/// the bare record.
#[must_use]
pub fn explain(policy: &SelectionPolicy, inputs: &SelectionInputs) -> Explanation {
    let (_, explanation) = select(policy, inputs, None)
        .expect("un-overridden selection on a validated policy is total");
    explanation
}

/// Swap-target site adapter (RFC-0005 §4 site 1; RFC-0002): the chosen candidate must be a
/// [`Repr`] — anything else is an explicit [`SelectError::WrongSiteKind`].
pub fn select_swap_target(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced: Option<usize>,
) -> Result<(Repr, Explanation), SelectError> {
    let (chosen, explanation) = select(policy, inputs, forced)?;
    match chosen {
        Candidate::Repr(r) => Ok((r, explanation)),
        other => Err(SelectError::WrongSiteKind {
            chosen: other,
            site: "swap-target",
        }),
    }
}

/// Packing-schedule site adapter (RFC-0005 §4 site 2; RFC-0004 §5 — consumed by E2-7/M-250): the
/// chosen candidate must be a [`PackScheme`].
pub fn select_packing(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced: Option<usize>,
) -> Result<(PackScheme, Explanation), SelectError> {
    let (chosen, explanation) = select(policy, inputs, forced)?;
    match chosen {
        Candidate::Packing(s) => Ok((s, explanation)),
        other => Err(SelectError::WrongSiteKind {
            chosen: other,
            site: "packing",
        }),
    }
}

/// Decode-method site adapter (RFC-0005 §4 site 3; RFC-0010): the chosen candidate must be a
/// [`DecodeMethod`] — anything else is an explicit [`SelectError::WrongSiteKind`]. The `mycelium-vsa`
/// executor consumes the returned method (and the mandatory [`Explanation`]) and runs the matching
/// decode, reading the guarantee tag off the chosen arm (RFC-0010 §4.4).
pub fn select_decode_method(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced: Option<usize>,
) -> Result<(DecodeMethod, Explanation), SelectError> {
    let (chosen, explanation) = select(policy, inputs, forced)?;
    match chosen {
        Candidate::Decode(m) => Ok((m, explanation)),
        other => Err(SelectError::WrongSiteKind {
            chosen: other,
            site: "decode-method",
        }),
    }
}

/// Placement site adapter (the fourth RFC-0005 §4 site, added additively alongside swap-target/
/// packing/decode; RFC-0008 RT3; DN-63 §3.5; M-906 D-lite): the chosen candidate must be a
/// [`NodeRef`] — anything else is an explicit [`SelectError::WrongSiteKind`]. `forage`'s D-lite
/// caller builds a **single-node** candidate list (DN-70 D1: "degenerate but real") and gets back
/// the chosen [`NodeRef`] with its mandatory [`Explanation`] — the same total/deterministic engine
/// as every other site, no new mechanism (KC-3). Multi-node candidate sets (real network
/// placement) are the DN-70 §5 R-6 H2 residual; this adapter does not change to support them.
pub fn select_placement(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced: Option<usize>,
) -> Result<(NodeRef, Explanation), SelectError> {
    let (chosen, explanation) = select(policy, inputs, forced)?;
    match chosen {
        Candidate::Node(n) => Ok((n, explanation)),
        other => Err(SelectError::WrongSiteKind {
            chosen: other,
            site: "placement",
        }),
    }
}

/// A registry resolving a recorded `PolicyRef` back to the policy that decided — the operational
/// form of RFC-0005 §3's guarantee ("which policy chose this, and what does that policy do?").
/// Tooling (the LSP EXPLAIN surface, M-221) consults it to re-derive explanations.
#[derive(Debug, Default)]
pub struct PolicyRegistry {
    by_hash: BTreeMap<String, SelectionPolicy>,
}

impl PolicyRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a policy under its content address; returns the `PolicyRef`.
    pub fn register(&mut self, policy: SelectionPolicy) -> ContentHash {
        let r = policy.policy_ref();
        self.by_hash.insert(r.as_str().to_owned(), policy);
        r
    }

    /// Resolve a `PolicyRef` to its policy, if registered.
    #[must_use]
    pub fn get(&self, policy_ref: &ContentHash) -> Option<&SelectionPolicy> {
        self.by_hash.get(policy_ref.as_str())
    }

    /// Number of registered policies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }
}

// ===================================================================================================
// M-250 — the schedule-staged packing selector (E2-7; RFC-0004 §5; DN-01 Resolved; RFC-0005 §4).
// ===================================================================================================
//
// Packing is a **schedule concern**, not a type distinction (DN-01 §2/§6): a lossless physical
// re-encoding of the same trits. The *type* stays packing-agnostic; the layout is chosen here at a
// lowering stage by a **cost model evaluated exhaustively over a fixed, enumerable candidate set**
// — emphatically *not* a Halide-class autoscheduler (RFC-0004 §5; T1.4 confirms the small ≈5-scheme
// set is materially easier than Halide's exponential search). The choice is then recorded as the
// inspectable [`PhysicalLayout`] on `Meta.physical` (M-I5 lossless: [`Meta::with_physical`]).
//
// This **reuses the one selection mechanism** (RFC-0005 §4: one mechanism, two sites): it is a thin
// wrapper over [`select_packing`], adding only the `PackScheme → PhysicalLayout` record mapping. No
// parallel selector exists.

/// The fixed **bitnet.cpp** ternary packing candidate set (RFC-0004 §5; Wang et al.): `I2_S`
/// (2.0 b/w, lossless multiply-add, the default), `TL1` (2.0 b/w, 4-bit LUT, ARM/NEON), and `TL2`
/// (1.67 b/w, x86/AVX2, memory-bound). Small and enumerable (T1.4), so the selector evaluates the
/// cost model over *all three* — exhaustive, not heuristic search.
pub const BITNET_PACKINGS: [PackScheme; 3] = [PackScheme::I2S, PackScheme::Tl1, PackScheme::Tl2];

/// Map a chosen ternary [`PackScheme`] to the [`PhysicalLayout`] recorded on `Meta.physical`. A
/// ternary value's packing is always `TritPacked` (RFC-0001 §4.3; mirrors `lower::schedule`); the
/// scheme is the only degree of freedom, so this is total and lossless (M-I5).
#[must_use]
pub fn layout_of(scheme: PackScheme) -> PhysicalLayout {
    PhysicalLayout::TritPacked { scheme }
}

/// Build the **default schedule-staged packing policy** (M-250): the three [`BITNET_PACKINGS`]
/// candidates with a single `Always → Cheapest` rule over the bits/element [`CostModel`]. Because
/// the cost is exact storage bits (`I2_S`/`TL1` = 2.0, `TL2` = 1.67 b/w; RFC-0004 §5 / DN-01), the
/// exhaustive cheapest is `TL2`, deterministically — and an override can force `I2_S` (the lossless
/// multiply-add default) or `TL1` (the ARM LUT) at a call site that knows its target.
///
/// The `storage_weight` is `1.0` (cost = raw bits/element × element count; the unit is real bits,
/// not "arbitrary internal units" — RFC-0005 §2). Returns the validated policy.
#[must_use]
pub fn bitnet_packing_policy() -> SelectionPolicy {
    let candidates = BITNET_PACKINGS
        .iter()
        .map(|s| Candidate::Packing(*s))
        .collect();
    SelectionPolicy::new(
        "schedule-staged.bitnet.v1",
        candidates,
        vec![Rule {
            when: Predicate::Always,
            action: Action::Cheapest,
        }],
        0, // unreachable default (Always matches); valid index by construction.
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect("the fixed bitnet packing policy is well-formed by construction")
}

/// The **packing-schedule selector** (M-250; RFC-0004 §5; one mechanism — RFC-0005 §4): evaluate
/// the cost model exhaustively over the policy's fixed packing candidate set via [`select_packing`]
/// and return the chosen [`PhysicalLayout`] to record on `Meta.physical` (M-I5 lossless), together
/// with the **mandatory EXPLAIN** trace (M-221 — there is no selection without one).
///
/// Deterministic: same `(policy, inputs, forced)` → same layout. A first-class `forced` override
/// picks a candidate by index (e.g. `Some(0)` forces `I2_S`); out of range is an explicit
/// [`SelectError::OverrideOutOfRange`]. A non-`Packing` candidate at this site is the explicit
/// [`SelectError::WrongSiteKind`] (`select_packing`'s contract) — never a coercion.
///
/// The produced layout is always a [`PhysicalLayout::TritPacked`], which only describes a *ternary*
/// value's byte layout (RFC-0004 §5; DN-01). Recording it for a non-`Ternary` source would be a
/// silent latent mis-tag, so a non-ternary `inputs.src` is the explicit
/// [`SelectError::NonTernarySource`] (A5-02 well-formedness) — checked before selection, never a
/// layout that contradicts its own representation.
pub fn select_layout(
    policy: &SelectionPolicy,
    inputs: &SelectionInputs,
    forced: Option<usize>,
) -> Result<(PhysicalLayout, Explanation), SelectError> {
    if !matches!(inputs.src, Repr::Ternary { .. }) {
        return Err(SelectError::NonTernarySource {
            src: inputs.src.clone(),
        });
    }
    let (scheme, explanation) = select_packing(policy, inputs, forced)?;
    Ok((layout_of(scheme), explanation))
}

/// One-call convenience: select the packing layout for a value's `(Repr, Meta)` and **record it**
/// onto a returned `Meta` (M-I5 lossless via [`Meta::with_physical`]), returning the updated `Meta`
/// and the EXPLAIN trace. The src `Repr` drives the cost model's element count; the layout record
/// is the schedule artifact, not a type change.
///
/// A non-`Ternary` `src` is the explicit [`SelectError::NonTernarySource`] (A5-02): a trit-packed
/// layout cannot honestly describe a non-ternary value, so no mis-tagged `Meta` is ever produced.
pub fn record_packing_layout(
    policy: &SelectionPolicy,
    src: &Repr,
    meta: &Meta,
    forced: Option<usize>,
) -> Result<(Meta, Explanation), SelectError> {
    let inputs = SelectionInputs::from_meta(src.clone(), meta);
    let (layout, explanation) = select_layout(policy, &inputs, forced)?;
    Ok((meta.clone().with_physical(layout), explanation))
}
