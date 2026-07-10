//! `std.swap` — Ring 1 / Tier A certified, never-silent representation-change surface (M-516).
//!
//! # Summary
//!
//! This is the library form of RFC-0002: every swap yields a target value **and** an inspectable
//! [`SwapCertificate`] describing the conversion, or an explicit error. A swap is **never silent**
//! (C1/G2): unsupported pair → [`SwapError`]; out-of-range → `Err`; no statable bound →
//! [`SwapError::InsufficientCapacity`]. Never a clamp, a re-round, or a sentinel.
//!
//! # Architecture
//!
//! Ring 1 consumer over `mycelium-cert`'s swap engines (KC-3 — this crate adds no trusted code,
//! defines no new legal pairs, and carries no `unsafe`). The single M-210 shared checker lives
//! in `mycelium-cert::check`; this module surfaces it through [`check_swap`].
//!
//! # Guarantee matrix (RFC-0016 §4.5)
//!
//! Encoded as data in [`GUARANTEE_MATRIX`] and asserted in the test suite — never prose-only.
//!
//! # Contract conformance (RFC-0016 §4.1 C1–C6)
//!
//! - **C1 never-silent:** every fallible op returns `Result`; no sentinel, no clamp.
//! - **C2 honest tag:** each op is tagged on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` per the
//!   landed cert; the tag is *derived* from the cert's basis, never asserted (VR-5).
//! - **C3 no black boxes / EXPLAIN:** every swap returns its [`SwapCertificate`]; [`explain`]
//!   projects it to an [`ExplainRecord`] (G11 dual human/machine form).
//! - **C4 value-semantic:** all ops are pure functions of their inputs (+ `PolicyRef` where
//!   policy-dependent); results are immutable values.
//! - **C5 above the kernel:** consumes `mycelium-cert`; no `unsafe`, no FFI, no new trusted code.
//! - **C6 declared bounded effects:** every op is pure save for bounded `alloc` of the target
//!   value + certificate.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 1, Tier A):** Swap operations are the primary site where representation
//! changes are reified. Every swap returns a [`SwapCertificate`] (the EXPLAIN artifact — ADR-003)
//! describing the conversion; no swap is ever silent (S1/G2). The representation change is not
//! just inspectable after the fact — it is structurally impossible to obtain a converted value
//! without its certificate.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/swap.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

// Re-export the types consumers need — surfaced from the landed crates, not redefined here (C5).
pub use mycelium_cert::{
    check, CheckVerdict, Evidence, Fallback, NotValidatedReason, RefinementRelation,
    SwapCertificate, SwapError,
};
pub use mycelium_cert::{BF16_MIN_NORMAL, BF16_REL_EPS, DENSE_VSA_EMP_DELTA, DENSE_VSA_MODEL};
pub use mycelium_core::{
    Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, NormKind, Repr, Value,
};

/// A PolicyRef is a [`ContentHash`] that records which swap policy was applied (RFC-0005; ADR-006).
/// This surface does not define policies; it only records the one it receives.
pub type PolicyRef = ContentHash;

// ──────────────────────────────────────────────────────────────────────────────
// § 1. Core return type: Swapped
// ──────────────────────────────────────────────────────────────────────────────

/// The result of a successful swap: the target **value** together with its inspectable
/// **certificate** — never the value alone (C1/C3). Value-semantic and immutable.
///
/// The certificate is the EXPLAIN artifact (C3); call [`Swapped::explain`] to project it to a
/// dual human/machine form (G11).
#[derive(Debug, Clone, PartialEq)]
pub struct Swapped {
    /// The converted value in the target representation.
    pub value: Value,
    /// The swap's certificate: what the conversion cost and why it is justified.
    pub cert: SwapCertificate,
}

impl Swapped {
    /// Build a `Swapped` from the `(value, cert)` pair the underlying cert crate returns.
    fn from_pair((value, cert): (Value, SwapCertificate)) -> Self {
        Swapped { value, cert }
    }

    /// Project the certificate to a human/machine dual EXPLAIN record (G11; C3).
    ///
    /// This function is total — every certificate explains.
    #[must_use]
    pub fn explain(&self) -> ExplainRecord {
        explain(&self.cert)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 2. CheckError — the check op's explicit error set
// ──────────────────────────────────────────────────────────────────────────────

/// Why a certificate check did not produce a `Validated` verdict (RFC-0002 §2).
///
/// [`NotValidated`](CheckError::NotValidated) is the TV-incompleteness arm: a correct swap may
/// fail to validate. It is **never** a silent pass — callers must take the explicit fallback
/// (RFC-0002 §2).
#[derive(Debug, Clone, PartialEq)]
pub enum CheckError {
    /// The checker found a concrete counterexample (the swap is *wrong*).
    Refuted {
        /// Human-readable detail.
        detail: String,
    },
    /// Translation-validation incompleteness or certificate mismatch: this instance could not be
    /// decided — not a proof of correctness. The caller **must** route to the enclosed [`Fallback`].
    NotValidated {
        /// Why the checker could not validate.
        reason: NotValidatedReason,
        /// The explicit fallback path — always [`Fallback::UseReference`].
        fallback: Fallback,
    },
}

impl core::fmt::Display for CheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CheckError::Refuted { detail } => write!(f, "refuted: {detail}"),
            CheckError::NotValidated { reason, .. } => {
                write!(
                    f,
                    "not validated (TV incompleteness / mismatch): {reason:?}"
                )
            }
        }
    }
}

impl std::error::Error for CheckError {}

// ──────────────────────────────────────────────────────────────────────────────
// § 3. EXPLAIN record (C3 / G11)
// ──────────────────────────────────────────────────────────────────────────────

/// A dual human/machine projection of a [`SwapCertificate`] (G11; C3).
///
/// Produced by [`explain`]. Every certificate explains; this is total.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplainRecord {
    /// Source representation.
    pub src: Repr,
    /// Target representation.
    pub target: Repr,
    /// The policy that justified the swap (RFC-0005).
    pub policy_used: ContentHash,
    /// Which certificate class: `"Bijective"` or `"Bounded"`.
    pub cert_kind: &'static str,
    /// For `Bijective` certs: content hash of the round-trip lemma (RFC-0002 §4; M-121).
    pub lemma_ref: Option<ContentHash>,
    /// For `Bounded` certs: the error/probability bound.
    pub bound: Option<Bound>,
    /// The guarantee strength implied by the certificate's basis (derived, never asserted; VR-5).
    pub strength: GuaranteeStrength,
    /// Human-readable one-line summary (the G11 human projection).
    pub summary: String,
}

/// Project a [`SwapCertificate`] to an [`ExplainRecord`] — total, never fails (C3; G11).
///
/// Bijective certificates carry `strength = Exact` and the M-121 `lemma_ref`.
/// Bounded certificates carry `strength` derived from their basis (never asserted) and their bound.
#[must_use]
pub fn explain(cert: &SwapCertificate) -> ExplainRecord {
    match cert {
        SwapCertificate::Bijective {
            src,
            target,
            policy_used,
            lemma_ref,
            params,
        } => ExplainRecord {
            src: src.clone(),
            target: target.clone(),
            policy_used: policy_used.clone(),
            cert_kind: "Bijective",
            lemma_ref: Some(lemma_ref.clone()),
            bound: None,
            strength: GuaranteeStrength::Exact,
            summary: format!(
                "Exact bijective swap {src:?} \u{2192} {target:?} \
                 (width={w}, trits={t}); lemma {lemma}; Exact within range",
                w = params.width,
                t = params.trits,
                lemma = lemma_ref.as_str(),
            ),
        },
        SwapCertificate::Bounded {
            src,
            target,
            policy_used,
            bound,
        } => {
            // Strength is derived from the basis — never asserted (VR-5).
            let strength = bound.basis.strength();
            let bound_desc = match &bound.kind {
                BoundKind::Error { eps, norm } => format!("\u{03b5}={eps:.2e} ({norm:?})"),
                BoundKind::Probability { delta } => format!("\u{03b4}={delta:.2e}"),
                _ => format!("{:?}", bound.kind),
            };
            let basis_desc = match &bound.basis {
                BoundBasis::ProvenThm { citation } => format!("ProvenThm: {citation}"),
                BoundBasis::EmpiricalFit { trials, method } => {
                    format!("EmpiricalFit ({trials} trials): {method}")
                }
                BoundBasis::UserDeclared => "UserDeclared (unverified)".to_owned(),
            };
            ExplainRecord {
                src: src.clone(),
                target: target.clone(),
                policy_used: policy_used.clone(),
                cert_kind: "Bounded",
                lemma_ref: None,
                bound: Some(bound.clone()),
                strength,
                summary: format!(
                    "{strength:?} bounded swap {src:?} \u{2192} {target:?}; \
                     bound={bound_desc}; basis={basis_desc}"
                ),
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 4. Named swap operations (the exported op surface)
// ──────────────────────────────────────────────────────────────────────────────

/// `bin_to_tern` — encode an `n`-bit two's-complement [`Value`] into `m` balanced trits.
///
/// **Guarantee tag: `Exact`-within-range** (`LosslessWithinRange`; RFC-0002 §4).
/// The bijection is SMT-dischargeable for fixed `(n, m)` legal pairs; the lemma reference appears
/// in the returned certificate. Out-of-range / illegal pair → explicit `Err`, never a clamp (C1).
///
/// **Certificate:** [`SwapCertificate::Bijective`] referencing the M-121 round-trip lemma.
///
/// # Errors
///
/// - [`SwapError::WrongSource`] — `src` is not a `Binary` value.
/// - [`SwapError::IllegalPair`] — `(width, trits_width)` is not a legal pair (`B_n ⊄ T_m`).
/// - [`SwapError::Wf`] — a well-formedness violation constructing the result value.
pub fn bin_to_tern(
    src: &Value,
    trits_width: u32,
    policy: &PolicyRef,
) -> Result<Swapped, SwapError> {
    mycelium_cert::binary_to_ternary(src, trits_width, policy).map(Swapped::from_pair)
}

/// `tern_to_bin` — decode `m` balanced trits back into an `n`-bit two's-complement [`Value`].
///
/// **Guarantee tag: `Exact`-within-range** (partial right-inverse on the image; RFC-0002 §4).
/// A total bijection is impossible at fixed widths (2ⁿ = 3ᵐ only trivially), so values outside
/// the binary range yield an explicit `Err(OutOfRange)` — never coerced (C1).
///
/// **Certificate:** [`SwapCertificate::Bijective`] on success.
///
/// # Errors
///
/// - [`SwapError::WrongSource`] — `src` is not a `Ternary` value.
/// - [`SwapError::IllegalPair`] — `(binary_width, trits)` is not a legal pair.
/// - [`SwapError::OutOfRange`] — the ternary value lies outside the binary range.
/// - [`SwapError::Wf`] — a well-formedness violation.
pub fn tern_to_bin(
    src: &Value,
    binary_width: u32,
    policy: &PolicyRef,
) -> Result<Swapped, SwapError> {
    mycelium_cert::ternary_to_binary(src, binary_width, policy).map(Swapped::from_pair)
}

/// `f32_to_bf16` — round a `Dense{F32}` value to `Dense{BF16}` under round-to-nearest (M-211).
///
/// **Guarantee tag:** `Proven` (ε = 2⁻⁸, Rel) **iff** the certificate's basis is `ProvenThm`
/// with all per-element side-conditions checked (finite, exact `f32`, zero-or-normal, no overflow
/// on rounding — ADR-010 §1; Higham 2002, Thm 2.2). The cert crate derives the tag; this surface
/// does not re-assert it. If any side-condition fails, the swap refuses with an explicit error
/// rather than emitting a bound the theorem does not cover (VR-5).
///
/// **Certificate:** [`SwapCertificate::Bounded`] (ε = `BF16_REL_EPS`, `ProvenThm` basis).
///
/// # Errors
///
/// - [`SwapError::WrongSource`] — `src` is not `Dense{F32}`.
/// - [`SwapError::ApproximateSource`] — `src`'s guarantee is not `Exact` (no composition rule yet).
/// - [`SwapError::NonFinite`] — an element is NaN/±Inf.
/// - [`SwapError::NotAnF32`] — an element is not exactly an `f32` value.
/// - [`SwapError::SubnormalUnsupported`] — an element is subnormal (outside the proven bound's
///   side-conditions; v1 scope).
/// - [`SwapError::RoundOverflow`] — an element overflows BF16's finite range on rounding.
/// - [`SwapError::Wf`] — a well-formedness violation.
pub fn f32_to_bf16(src: &Value, policy: &PolicyRef) -> Result<Swapped, SwapError> {
    mycelium_cert::dense_f32_to_bf16(src, policy).map(Swapped::from_pair)
}

/// `dense_to_vsa` — encode a bipolar `Dense{n, F32}` value into a `Vsa{MAP-I, vsa_dim}`
/// superposition (M-231; RFC-0002 §5; RFC-0003/T0.2).
///
/// **Guarantee tag: `Empirical` (δ) by default** — the VSA capacity result is probabilistic.
/// Rises to `Proven` only when the cited capacity theorem's side-conditions are checked
/// (`vsa_dim ≥ requiredDim(n, delta)` — Clarkson-Ubaru-Yang 2023, Thm 6). The cert crate
/// derives the tag; this surface surfaces it without upgrade (VR-5, C2).
///
/// Only bipolar Dense vectors (`{-1, +1}ⁿ`) are accepted — the capacity theorem covers bundles
/// of bipolar atoms; a weighted-superposition bound is not in the corpus (M-231 v1 scope).
///
/// **Certificate:** [`SwapCertificate::Bounded`] (δ, `ProvenThm` or `EmpiricalFit` basis).
///
/// # Errors
///
/// - [`SwapError::WrongSource`] — not `Dense{F32}`.
/// - [`SwapError::ApproximateSource`] — source is not `Exact`.
/// - [`SwapError::NotBipolar`] — a component is not `±1`.
/// - [`SwapError::InsufficientCapacity`] — no basis covers this `(n, vsa_dim, delta)` instance
///   (a type error, not a `Declared` gamble — RFC-0002 §5).
/// - [`SwapError::Wf`] — a well-formedness violation.
pub fn dense_to_vsa(
    src: &Value,
    vsa_dim: u32,
    delta: f64,
    policy: &PolicyRef,
) -> Result<Swapped, SwapError> {
    mycelium_cert::dense_to_vsa(src, vsa_dim, delta, policy).map(Swapped::from_pair)
}

/// `vsa_to_dense` — decode a `swap.dense_vsa.enc.v1` product back to a bipolar `Dense{F32}` value
/// by signed correlation against the same versioned codebook (M-231).
///
/// **Guarantee tag: `Empirical` (δ) by default**; `Proven` when the capacity theorem's
/// side-conditions check (same as [`dense_to_vsa`]). The cert derives the tag (VR-5, C2).
///
/// The δ describes retrieval from the *enc.v1* encoding **only** — provenance-gated. A VSA value
/// not produced by [`dense_to_vsa`] yields an explicit `Err(NotDenseVsaEncoding)` (the bound
/// would describe nothing for a foreign VSA value).
///
/// **Certificate:** [`SwapCertificate::Bounded`] (δ).
///
/// # Errors
///
/// - [`SwapError::WrongSource`] — not `Vsa{MAP-I}`.
/// - [`SwapError::NotDenseVsaEncoding`] — source was not produced by `dense_to_vsa` (`enc.v1`).
/// - [`SwapError::InsufficientCapacity`] — no basis covers this instance.
/// - [`SwapError::AmbiguousDecode`] — a correlation vanished (sign undefined, never arbitrary).
/// - [`SwapError::Wf`] — a well-formedness violation.
pub fn vsa_to_dense(
    src: &Value,
    components: u32,
    delta: f64,
    policy: &PolicyRef,
) -> Result<Swapped, SwapError> {
    mycelium_cert::vsa_to_dense(src, components, delta, policy).map(Swapped::from_pair)
}

// ──────────────────────────────────────────────────────────────────────────────
// § 5. Certificate check surface (the M-210 shared checker — KC-3, C5)
// ──────────────────────────────────────────────────────────────────────────────

/// Validate that value `b` refines value `a` under the swap described by `cert` (M-210).
///
/// **Guarantee tag: `Exact`** — this is a verdict, not an approximation.
///
/// - `Ok(strength)` — validated at `strength` (derived from the cert's basis; VR-5).
/// - `Err(CheckError::Refuted)` — concrete counterexample (the swap is wrong).
/// - `Err(CheckError::NotValidated)` — TV-incompleteness or certificate mismatch; **not** a proof
///   of correctness, **never** a silent pass (RFC-0002 §2). The caller **must** take the enclosed
///   [`Fallback`] path (always [`Fallback::UseReference`]).
///
/// The relation is inferred from the certificate kind (`Bijective` → `Bijection`; `Bounded` →
/// `BoundedSimilarity`). This surface delegates to the one M-210 checker in `mycelium-cert::check`
/// — it does not enlarge the trusted base (KC-3, C5).
///
/// # Errors
///
/// Returns [`CheckError`] when the verdict is not `Validated`.
pub fn check_swap(
    a: &Value,
    b: &Value,
    cert: &SwapCertificate,
) -> Result<GuaranteeStrength, CheckError> {
    use mycelium_numerics::Certificate;

    let (relation, claimed) = match cert {
        SwapCertificate::Bijective { .. } => (RefinementRelation::Bijection, Certificate::exact()),
        SwapCertificate::Bounded { bound, .. } => {
            // The strength is derived from the basis, never asserted (VR-5).
            let strength = bound.basis.strength();
            let claimed = match &bound.kind {
                BoundKind::Error { eps, norm: _ } => {
                    // The cert was produced by mycelium-cert so the values are well-formed; still
                    // guard against a corrupt deserialized cert.
                    Certificate::new(*eps, 0.0, strength).ok_or_else(|| {
                        CheckError::NotValidated {
                            reason: NotValidatedReason::CertificateMismatch {
                                detail: "certificate ε is not a well-formed bound".to_owned(),
                            },
                            fallback: Fallback::UseReference,
                        }
                    })?
                }
                BoundKind::Probability { delta } => Certificate::new(0.0, *delta, strength)
                    .ok_or_else(|| CheckError::NotValidated {
                        reason: NotValidatedReason::CertificateMismatch {
                            detail: "certificate δ is not a well-formed probability".to_owned(),
                        },
                        fallback: Fallback::UseReference,
                    })?,
                // FLAG: CrosstalkBound/CapacityBound are not in the current legal-pair surface
                // (M-231 v1 scope). Surface as NotValidated/Incomplete rather than panic — the
                // caller takes the UseReference fallback (RFC-0002 §2).
                _ => {
                    return Err(CheckError::NotValidated {
                        reason: NotValidatedReason::Incomplete {
                            detail: "bound kind not checkable at this checker version \
                                     (only ε and δ certificates; FLAG: M-231 v1 scope)"
                                .to_owned(),
                        },
                        fallback: Fallback::UseReference,
                    });
                }
            };
            (RefinementRelation::BoundedSimilarity, claimed)
        }
    };
    let evidence = Evidence::Swap(cert);
    match check(a, b, relation, claimed, &evidence) {
        CheckVerdict::Validated { strength } => Ok(strength),
        CheckVerdict::NotValidated {
            reason: NotValidatedReason::Diverged { detail },
            ..
        } => Err(CheckError::Refuted { detail }),
        CheckVerdict::NotValidated { reason, fallback } => {
            Err(CheckError::NotValidated { reason, fallback })
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 6. Guarantee matrix (RFC-0016 §4.5) — encoded as data, asserted in tests
// ──────────────────────────────────────────────────────────────────────────────

/// One row of the guarantee matrix (RFC-0016 §4.5; swap.md §4).
///
/// Encoded as data so tests can assert invariants rather than prose-only (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixRow {
    /// The exported operation's name.
    pub op: &'static str,
    /// The honest guarantee tag on the `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` lattice.
    ///
    /// For bounded ops this is the *default* strength; the cert may reach `Proven` when the
    /// cited theorem's side-conditions check (see each op's documentation).
    pub guarantee: GuaranteeStrength,
    /// Whether the op is fallible (returns `Result`).
    pub fallible: bool,
    /// Whether the op emits or **is** a certificate artifact (EXPLAIN-able; C3).
    pub cert_carrying: bool,
    /// The certificate kind emitted (`"Bijective"`, `"Bounded"`, or `None` for non-swap ops).
    pub cert_kind: Option<&'static str>,
}

/// The guarantee matrix for `std.swap` (RFC-0016 §4.5; swap.md §4).
///
/// Every invariant is asserted in the test suite via [`assert_matrix_invariants`] — never
/// prose-only (RFC-0016 §4.5, "encoded as data, asserted in tests").
pub const GUARANTEE_MATRIX: &[MatrixRow] = &[
    MatrixRow {
        op: "bin_to_tern",
        // Exact within range: LosslessWithinRange bijection (RFC-0002 §4, T2.1);
        // SMT-dischargeable for fixed (n, m) legal pairs; out-of-range is an explicit error.
        guarantee: GuaranteeStrength::Exact,
        fallible: true,
        cert_carrying: true,
        cert_kind: Some("Bijective"),
    },
    MatrixRow {
        op: "tern_to_bin",
        // Exact within range: partial right-inverse on the image; total bijection impossible
        // at fixed widths (2^n = 3^m only trivially) — inverse is Result-typed off the image.
        guarantee: GuaranteeStrength::Exact,
        fallible: true,
        cert_carrying: true,
        cert_kind: Some("Bijective"),
    },
    MatrixRow {
        op: "f32_to_bf16",
        // Proven (ε) when ProvenThm side-conditions check (per-element: finite, exact f32,
        // zero-or-normal, no overflow). The cert crate derives the tag; never upgraded here.
        // Default is Proven because the cert crate checks and emits ProvenThm for normal inputs.
        guarantee: GuaranteeStrength::Proven,
        fallible: true,
        cert_carrying: true,
        cert_kind: Some("Bounded"),
    },
    MatrixRow {
        op: "dense_to_vsa",
        // Empirical (δ) by default — VSA capacity result is probabilistic (RFC-0003/T0.2).
        // Proven only when the capacity theorem's side-conditions check at cert-build time.
        guarantee: GuaranteeStrength::Empirical,
        fallible: true,
        cert_carrying: true,
        cert_kind: Some("Bounded"),
    },
    MatrixRow {
        op: "vsa_to_dense",
        // Empirical (δ) by default; same reasoning as dense_to_vsa.
        guarantee: GuaranteeStrength::Empirical,
        fallible: true,
        cert_carrying: true,
        cert_kind: Some("Bounded"),
    },
    MatrixRow {
        op: "check_swap",
        // Exact: a verdict is not an approximation. The accuracy lives in the swap the cert
        // describes, not in the verdict itself (C2 — an op with no accuracy semantics is Exact).
        guarantee: GuaranteeStrength::Exact,
        fallible: true,
        cert_carrying: false,
        cert_kind: None,
    },
    MatrixRow {
        op: "explain",
        // Exact: a faithful projection of the certificate (G11). No accuracy semantics.
        guarantee: GuaranteeStrength::Exact,
        fallible: false,
        cert_carrying: true,
        cert_kind: None,
    },
];

/// Assert the structural invariants of the guarantee matrix — called from tests.
///
/// This discharges the RFC-0016 §4.5 obligation: "encoded as data, asserted in tests, never
/// prose-only." Panics with a descriptive message on any violation.
pub fn assert_matrix_invariants() {
    for row in GUARANTEE_MATRIX {
        // Non-empty op name.
        assert!(!row.op.is_empty(), "matrix row has empty op name");

        // cert_kind, when present, must be a known kind.
        if let Some(kind) = row.cert_kind {
            assert!(
                kind == "Bijective" || kind == "Bounded",
                "op {}: cert_kind must be 'Bijective' or 'Bounded', got {kind:?}",
                row.op
            );
        }

        // Bijective certs are Exact — the only genuinely bijective/exact swap class (RFC-0002 §4).
        if row.cert_kind == Some("Bijective") {
            assert_eq!(
                row.guarantee,
                GuaranteeStrength::Exact,
                "op {}: Bijective cert implies Exact guarantee (RFC-0002 §4)",
                row.op
            );
        }

        // Bounded certs are never Exact — Exact means no bound (M-I1; RFC-0001 §4.3).
        if row.cert_kind == Some("Bounded") {
            assert_ne!(
                row.guarantee,
                GuaranteeStrength::Exact,
                "op {}: Bounded cert cannot be Exact (Exact means no bound; M-I1)",
                row.op
            );
        }

        // Non-fallible ops must not carry a cert_kind (they don't emit swap certs).
        // explain is the one non-fallible, cert_carrying op — but it projects rather than emits.
        if !row.fallible && row.cert_kind.is_some() {
            panic!(
                "op {}: non-fallible op should not emit a swap cert_kind",
                row.op
            );
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 7. Convenience re-exports (legality probe + constants)
// ──────────────────────────────────────────────────────────────────────────────

/// Whether `(n, m)` is a legal binary↔ternary pair (`B_n ⊆ T_m`; RFC-0002 §5).
///
/// Lets consumers probe legality without calling into the swap op (useful for batched pipelines).
/// The definitive check is `legal_pair(width, trits)` from `mycelium-cert`.
pub use mycelium_cert::legal_pair;

/// The content hash of the M-121 round-trip lemma — the `lemma_ref` every bijective certificate
/// references (RFC-0002 §4). Useful for validating a deserialized certificate.
pub use mycelium_cert::roundtrip_lemma_ref;

// ──────────────────────────────────────────────────────────────────────────────
// § 8. Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::{operation_hash, Payload, ScalarKind};

    /// A canonical policy hash used across all tests — stands in for a real RFC-0005 policy.
    fn test_policy() -> PolicyRef {
        operation_hash("test.policy.std_swap.v1")
    }

    /// Build a `Binary{n}` value from a signed integer.
    fn make_binary(value: i64, width: u32) -> Value {
        use mycelium_core::{binary, GuaranteeStrength, Meta, Provenance};
        let bits = binary::int_to_bits(value, width)
            .unwrap_or_else(|| panic!("value {value} does not fit in {width} bits"));
        let meta = Meta::new(
            Provenance::Root,
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .expect("meta is well-formed");
        Value::new(Repr::Binary { width }, Payload::Bits(bits), meta)
            .expect("binary value is well-formed")
    }

    /// Build a `Ternary{m}` value from a signed integer.
    fn make_ternary(value: i64, trits: u32) -> Value {
        use mycelium_core::{GuaranteeStrength, Meta, Provenance};
        let trit_vec = mycelium_core::ternary::int_to_trits(value, trits)
            .unwrap_or_else(|| panic!("value {value} does not fit in {trits} trits"));
        let meta = Meta::new(
            Provenance::Root,
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .expect("meta is well-formed");
        Value::new(Repr::Ternary { trits }, Payload::Trits(trit_vec), meta)
            .expect("ternary value is well-formed")
    }

    /// Build a `Dense{n, F32}` value from a slice of f64 scalars.
    fn make_dense_f32(xs: &[f64]) -> Value {
        use mycelium_core::{GuaranteeStrength, Meta, Provenance};
        let meta = Meta::new(
            Provenance::Root,
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .expect("meta is well-formed");
        Value::new(
            Repr::Dense {
                dim: xs.len() as u32,
                dtype: ScalarKind::F32,
            },
            Payload::Scalars(xs.to_vec()),
            meta,
        )
        .expect("dense F32 value is well-formed")
    }

    // ── Guarantee matrix invariants ──────────────────────────────────────────

    /// The guarantee matrix is internally consistent (RFC-0016 §4.5).
    /// Mutation witness: set cert_kind=Some("Bijective") on f32_to_bf16 → assertion fires.
    #[test]
    fn guarantee_matrix_invariants_hold() {
        assert_matrix_invariants();
    }

    /// All expected ops appear in the matrix exactly once.
    /// Mutation witness: remove a row from GUARANTEE_MATRIX → count == 0.
    #[test]
    fn matrix_contains_all_ops_exactly_once() {
        let expected = [
            "bin_to_tern",
            "tern_to_bin",
            "f32_to_bf16",
            "dense_to_vsa",
            "vsa_to_dense",
            "check_swap",
            "explain",
        ];
        for op in &expected {
            let count = GUARANTEE_MATRIX.iter().filter(|r| r.op == *op).count();
            assert_eq!(count, 1, "op '{op}' must appear exactly once in the matrix");
        }
    }

    /// Bijective-cert ops are tagged Exact; Bounded-cert ops are not Exact.
    #[test]
    fn matrix_tag_cert_kind_consistency() {
        for row in GUARANTEE_MATRIX {
            match row.cert_kind {
                Some("Bijective") => assert_eq!(
                    row.guarantee,
                    GuaranteeStrength::Exact,
                    "op {}: Bijective implies Exact",
                    row.op
                ),
                Some("Bounded") => assert_ne!(
                    row.guarantee,
                    GuaranteeStrength::Exact,
                    "op {}: Bounded cannot be Exact",
                    row.op
                ),
                _ => {}
            }
        }
    }

    // ── bin_to_tern / tern_to_bin ────────────────────────────────────────────

    /// `dec(enc x) == Ok(x)` for every value in the n=8 corpus on the legal (8, 6) pair.
    ///
    /// This is the property test for the `LosslessWithinRange` guarantee (RFC-0002 §4).
    /// Mutation witness: change trits to 5 (illegal pair) → enc returns Err(IllegalPair).
    #[test]
    fn bin_tern_round_trip_full_n8_corpus() {
        let policy = test_policy();
        let (n, m) = (8u32, 6u32);
        assert!(legal_pair(n, m), "test pair (n=8, m=6) must be legal");

        // Full corpus for n=8: all 256 values in [-128, 127].
        for v in i8::MIN..=i8::MAX {
            let src = make_binary(i64::from(v), n);
            let encoded = bin_to_tern(&src, m, &policy)
                .unwrap_or_else(|e| panic!("bin_to_tern({v}) failed: {e}"));

            assert!(
                matches!(encoded.cert, SwapCertificate::Bijective { .. }),
                "enc({v}): cert must be Bijective"
            );

            let decoded = tern_to_bin(&encoded.value, n, &policy)
                .unwrap_or_else(|e| panic!("tern_to_bin(enc({v})) failed: {e}"));

            assert_eq!(
                decoded.value.payload(),
                src.payload(),
                "round-trip failed for v={v}"
            );
        }
    }

    /// Out-of-range ternary → explicit `Err(OutOfRange)`, never silent coercion (C1).
    ///
    /// Mutation witness: use v=7 (in range for n=4) → Ok instead of Err.
    #[test]
    fn tern_to_bin_out_of_range_explicit_error() {
        let policy = test_policy();
        // (n=4, m=3): max ternary = (3^3 - 1)/2 = 13; max Binary<4> signed = 7.
        // Value 10 is in the ternary range but outside the binary range.
        let (n, m) = (4u32, 3u32);
        assert!(legal_pair(n, m), "pair (4,3) must be legal");
        let out_of_range = make_ternary(10, m);
        let result = tern_to_bin(&out_of_range, n, &policy);
        assert!(
            matches!(result, Err(SwapError::OutOfRange)),
            "expected Err(OutOfRange), got {result:?}"
        );
    }

    /// Illegal pair → explicit `Err(IllegalPair)` (C1, RFC-0002 §5).
    ///
    /// Mutation witness: change trits to 6 (legal) → Ok.
    #[test]
    fn bin_to_tern_illegal_pair_explicit_error() {
        let policy = test_policy();
        // (n=8, m=4): 2^7=128 > (3^4-1)/2=40 → illegal.
        let src = make_binary(0, 8);
        let result = bin_to_tern(&src, 4, &policy);
        assert!(
            matches!(result, Err(SwapError::IllegalPair { width: 8, trits: 4 })),
            "expected Err(IllegalPair {{8,4}}), got {result:?}"
        );
    }

    /// Wrong source type → explicit `Err(WrongSource)` (C1).
    ///
    /// Mutation witness: use a Binary value → Ok.
    #[test]
    fn bin_to_tern_wrong_source_explicit_error() {
        let policy = test_policy();
        let src = make_ternary(0, 3); // Ternary, not Binary
        let result = bin_to_tern(&src, 5, &policy);
        assert!(
            matches!(result, Err(SwapError::WrongSource { .. })),
            "expected Err(WrongSource), got {result:?}"
        );
    }

    /// The bijective certificate carries the M-121 round-trip lemma reference.
    ///
    /// This is C3 (certificate is inspectable and carries the proof reference).
    #[test]
    fn bijective_cert_carries_m121_lemma_ref() {
        let policy = test_policy();
        let src = make_binary(42, 8);
        let s = bin_to_tern(&src, 6, &policy).expect("legal pair, in range");
        let SwapCertificate::Bijective { lemma_ref, .. } = &s.cert else {
            panic!("expected Bijective cert");
        };
        assert_eq!(
            *lemma_ref,
            roundtrip_lemma_ref(),
            "lemma_ref must equal the M-121 round-trip lemma"
        );
    }

    // ── f32_to_bf16 ──────────────────────────────────────────────────────────

    /// Normal finite f32 values succeed and the cert's ε ≤ BF16_REL_EPS (2⁻⁸).
    ///
    /// Mutation witness: set eps threshold to 0.0 → assertion fails.
    #[test]
    fn f32_to_bf16_normal_values_bounded_cert() {
        let policy = test_policy();
        let xs: &[f64] = &[1.0, -1.0, 2.0, 0.5, 100.0];
        let src = make_dense_f32(xs);
        let s = f32_to_bf16(&src, &policy).expect("normal f32 values should succeed");

        let SwapCertificate::Bounded { bound, .. } = &s.cert else {
            panic!("expected Bounded cert, got {:?}", s.cert);
        };
        let BoundKind::Error { eps, norm: _ } = bound.kind else {
            panic!("expected ErrorBound kind");
        };
        assert!(
            eps <= BF16_REL_EPS + 1e-15,
            "cert ε must not exceed 2⁻⁸ (BF16_REL_EPS={BF16_REL_EPS}); got {eps}"
        );
        assert!(
            matches!(&bound.basis, BoundBasis::ProvenThm { .. }),
            "expected ProvenThm basis for normal inputs"
        );
    }

    /// NaN input → explicit `Err(NonFinite{index:0})`, never silent (C1).
    ///
    /// Mutation witness: replace NaN with 1.0 → Ok.
    #[test]
    fn f32_to_bf16_nan_is_non_finite_error() {
        let policy = test_policy();
        let src = make_dense_f32(&[f64::NAN]);
        let result = f32_to_bf16(&src, &policy);
        assert!(
            matches!(result, Err(SwapError::NonFinite { index: 0 })),
            "expected Err(NonFinite{{0}}), got {result:?}"
        );
    }

    /// +Inf input → explicit `Err(NonFinite{index:0})`.
    ///
    /// Mutation witness: replace Inf with 1.0 → Ok.
    #[test]
    fn f32_to_bf16_inf_is_non_finite_error() {
        let policy = test_policy();
        let src = make_dense_f32(&[f64::INFINITY]);
        let result = f32_to_bf16(&src, &policy);
        assert!(
            matches!(result, Err(SwapError::NonFinite { index: 0 })),
            "expected Err(NonFinite{{0}}), got {result:?}"
        );
    }

    /// Subnormal input → explicit `Err(SubnormalUnsupported{index:0})` (v1 scope; VR-5).
    ///
    /// Mutation witness: replace the subnormal with 1.0 → Ok.
    #[test]
    fn f32_to_bf16_subnormal_is_explicit_error() {
        let policy = test_policy();
        // A value below f32::MIN_POSITIVE (the minimum normal f32/bf16).
        let subnormal = f64::from(f32::MIN_POSITIVE) * 0.5;
        let src = make_dense_f32(&[subnormal]);
        let result = f32_to_bf16(&src, &policy);
        assert!(
            matches!(result, Err(SwapError::SubnormalUnsupported { index: 0 })),
            "expected Err(SubnormalUnsupported{{0}}), got {result:?}"
        );
    }

    // ── dense_to_vsa / vsa_to_dense ──────────────────────────────────────────

    /// `vsa_to_dense(dense_to_vsa(x)) == Ok(x)` for all 2^n bipolar vectors (n=4).
    ///
    /// Property test over the full corpus of 16 bipolar vectors; generous dimension (512)
    /// so the proven capacity theorem applies (ProvenThm basis expected).
    /// Mutation witness: set vsa_dim=1 → Err(InsufficientCapacity).
    #[test]
    fn dense_vsa_round_trip_all_4bit_bipolar_vectors() {
        let policy = test_policy();
        let n = 4u32;
        let vsa_dim = 512u32; // well above requiredDim(4, 0.05)
        let delta = 0.05;

        for mask in 0u32..(1 << n) {
            let xs: Vec<f64> = (0..n)
                .map(|i| if (mask >> i) & 1 == 0 { -1.0 } else { 1.0 })
                .collect();
            let src = make_dense_f32(&xs);
            let encoded = dense_to_vsa(&src, vsa_dim, delta, &policy)
                .unwrap_or_else(|e| panic!("dense_to_vsa failed for mask={mask:#06b}: {e}"));

            assert!(
                matches!(encoded.cert, SwapCertificate::Bounded { .. }),
                "dense_to_vsa cert must be Bounded (mask={mask:#06b})"
            );

            let decoded = vsa_to_dense(&encoded.value, n, delta, &policy)
                .unwrap_or_else(|e| panic!("vsa_to_dense failed for mask={mask:#06b}: {e}"));
            assert_eq!(
                decoded.value.payload(),
                src.payload(),
                "round-trip failed for mask={mask:#06b} (xs={xs:?})"
            );
        }
    }

    /// Insufficient capacity → explicit `Err(InsufficientCapacity)`, never a Declared gamble
    /// (RFC-0002 §5).
    ///
    /// Mutation witness: raise vsa_dim to 512 → Ok.
    #[test]
    fn dense_to_vsa_no_basis_is_explicit_error() {
        let policy = test_policy();
        // 100 bipolar components into dim=1 — no basis covers this.
        let xs = vec![1.0f64; 100];
        let src = make_dense_f32(&xs);
        let result = dense_to_vsa(&src, 1, 0.05, &policy);
        assert!(
            matches!(result, Err(SwapError::InsufficientCapacity { .. })),
            "expected Err(InsufficientCapacity), got {result:?}"
        );
    }

    /// Non-bipolar component → explicit `Err(NotBipolar{index:1})` (M-231 v1 scope).
    ///
    /// Mutation witness: change 0.5 to 1.0 → Ok.
    #[test]
    fn dense_to_vsa_non_bipolar_explicit_error() {
        let policy = test_policy();
        let src = make_dense_f32(&[1.0, 0.5, -1.0]); // 0.5 is not ±1
        let result = dense_to_vsa(&src, 512, 0.05, &policy);
        assert!(
            matches!(result, Err(SwapError::NotBipolar { index: 1 })),
            "expected Err(NotBipolar{{1}}), got {result:?}"
        );
    }

    /// Decoding a VSA value not produced by enc.v1 → explicit `Err(NotDenseVsaEncoding)`.
    ///
    /// Mutation witness: use the output of dense_to_vsa → Ok.
    #[test]
    fn vsa_to_dense_non_encoding_explicit_error() {
        use mycelium_core::{GuaranteeStrength, Meta, Provenance, SparsityClass};
        let policy = test_policy();

        // Construct a bare VSA value with Root provenance (not enc.v1 Derived).
        let meta = Meta::new(
            Provenance::Root,
            GuaranteeStrength::Exact,
            None,
            None,
            None,
            None,
        )
        .expect("meta well-formed");
        let hv = vec![1.0f64; 64];
        let vsa = Value::new(
            Repr::Vsa {
                model: DENSE_VSA_MODEL.to_owned(),
                dim: 64,
                sparsity: SparsityClass::Dense,
            },
            Payload::Hypervector(hv),
            meta,
        )
        .expect("vsa value well-formed");

        let result = vsa_to_dense(&vsa, 2, 0.05, &policy);
        assert!(
            matches!(result, Err(SwapError::NotDenseVsaEncoding)),
            "expected Err(NotDenseVsaEncoding), got {result:?}"
        );
    }

    // ── check_swap (M-210 shared checker) ───────────────────────────────────

    /// `check_swap(a, enc(a), cert)` returns `Ok(Exact)` for a correct bijective swap.
    #[test]
    fn check_swap_validates_bijective_correct() {
        let policy = test_policy();
        let src = make_binary(7, 8);
        let s = bin_to_tern(&src, 6, &policy).expect("valid swap");
        let verdict = check_swap(&src, &s.value, &s.cert);
        assert!(
            matches!(verdict, Ok(GuaranteeStrength::Exact)),
            "expected Ok(Exact), got {verdict:?}"
        );
    }

    /// `check_swap` refutes a tampered target — never a silent pass (RFC-0002 §2).
    ///
    /// Mutation witness: use the correct target → Ok.
    #[test]
    fn check_swap_refutes_tampered_target() {
        let policy = test_policy();
        let src = make_binary(7, 8);
        let s = bin_to_tern(&src, 6, &policy).expect("valid swap");
        // Cert says enc(7) but the target is enc(42) — divergence.
        let wrong_src = make_binary(42, 8);
        let wrong_s = bin_to_tern(&wrong_src, 6, &policy).expect("valid swap");
        let verdict = check_swap(&src, &wrong_s.value, &s.cert);
        assert!(
            matches!(verdict, Err(CheckError::Refuted { .. })),
            "expected Err(Refuted) for tampered target, got {verdict:?}"
        );
    }

    /// `check_swap` validates a bounded (f32→bf16) swap at `Proven` strength.
    #[test]
    fn check_swap_validates_bounded_f32_bf16() {
        let policy = test_policy();
        let src = make_dense_f32(&[1.0, -1.0, 2.0]);
        let s = f32_to_bf16(&src, &policy).expect("valid f32→bf16 swap");
        let verdict = check_swap(&src, &s.value, &s.cert);
        assert!(
            matches!(verdict, Ok(GuaranteeStrength::Proven)),
            "expected Ok(Proven), got {verdict:?}"
        );
    }

    // ── explain (EXPLAIN record; C3/G11) ────────────────────────────────────

    /// `explain` is total for a Bijective cert and yields the expected fields.
    #[test]
    fn explain_bijective_total_and_correct() {
        let policy = test_policy();
        let src = make_binary(1, 8);
        let s = bin_to_tern(&src, 6, &policy).expect("valid swap");
        let rec = s.explain();
        assert!(!rec.summary.is_empty(), "summary must be non-empty");
        assert_eq!(rec.cert_kind, "Bijective");
        assert_eq!(rec.strength, GuaranteeStrength::Exact);
        assert!(
            rec.lemma_ref.is_some(),
            "Bijective cert must carry lemma_ref"
        );
        assert!(rec.bound.is_none(), "Bijective cert must not carry a bound");
        assert_eq!(rec.lemma_ref.as_ref().unwrap(), &roundtrip_lemma_ref());
    }

    /// `explain` is total for a Bounded cert and yields the expected fields.
    #[test]
    fn explain_bounded_total_and_correct() {
        let policy = test_policy();
        let src = make_dense_f32(&[1.0, -1.0]);
        let s = f32_to_bf16(&src, &policy).expect("valid swap");
        let rec = s.explain();
        assert!(!rec.summary.is_empty(), "summary must be non-empty");
        assert_eq!(rec.cert_kind, "Bounded");
        assert_eq!(rec.strength, GuaranteeStrength::Proven);
        assert!(rec.bound.is_some(), "Bounded cert must carry a bound");
        assert!(
            rec.lemma_ref.is_none(),
            "Bounded cert must not carry lemma_ref"
        );
    }

    /// `explain` summary for a Bounded cert mentions the bound kind and basis.
    #[test]
    fn explain_bounded_summary_contains_basis_info() {
        let policy = test_policy();
        let src = make_dense_f32(&[1.0]);
        let s = f32_to_bf16(&src, &policy).expect("valid swap");
        let rec = s.explain();
        // Should mention ProvenThm (the basis name).
        assert!(
            rec.summary.contains("ProvenThm"),
            "summary should mention ProvenThm; got: {}",
            rec.summary
        );
    }

    // ── legal_pair helper ────────────────────────────────────────────────────

    /// `legal_pair` agrees with the spec table (RFC-0002 §5 / binary-ternary.md §2).
    ///
    /// 2^(n-1) ≤ (3^m - 1) / 2  ⇔  legal.
    #[test]
    fn legal_pair_spec_table_sample() {
        // Legal pairs:
        assert!(legal_pair(1, 1)); // 1 ≤ 1 ✓
        assert!(legal_pair(2, 2)); // 2 ≤ 4 ✓
        assert!(legal_pair(4, 3)); // 8 ≤ 13 ✓
        assert!(legal_pair(8, 6)); // 128 ≤ 364 ✓
                                   // Illegal pairs:
        assert!(!legal_pair(8, 4)); // 128 > 40 ✗
        assert!(!legal_pair(8, 5)); // 128 > 121 ✗
        assert!(!legal_pair(0, 0)); // width=0 is a degenerate case
    }
}
