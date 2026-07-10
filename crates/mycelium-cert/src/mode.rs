//! **Mode-gated swap-certificate emission + checking** (M-788; RFC-0034 §4/§5/§7).
//!
//! The certificate machinery in this crate ([`SwapCertificate`] emission and the M-210
//! [`check()`](crate::check()) checker) is *unchanged*; this module gates **whether it runs**, by the
//! active [`CertMode`]:
//!
//! - **[`Fast`](CertMode::Fast)** — the cert machinery does **not** run: no certificate is emitted
//!   and nothing is checked. The result's `Meta` is reconciled through
//!   [`CertMode::gate_result`](mycelium_core::CertMode::gate_result): a would-be `Proven`/`Empirical`
//!   tag floors to `Declared` and its computed bound's basis is relabelled to `UserDeclared`
//!   ("computed, asserted-not-verified in fast"; M-I1…M-I4 stay consistent — see that method).
//! - **[`Balanced`](CertMode::Balanced)** — the certificate **is emitted** but **not checked**; the
//!   honest tags propagate unchanged (the cert machinery's emit half runs, the check half does not).
//! - **[`Certified`](CertMode::Certified)** — emit **and** check (today's full behaviour, unchanged):
//!   the emitted certificate is validated through the one shared [`check()`](crate::check()), and a
//!   check that does not validate is surfaced **never-silently** as a [`CheckVerdict::NotValidated`]
//!   (with its [`Fallback`](crate::Fallback)) on the [`GatedSwap`].
//!
//! **Axis-B is not gated here** (RFC-0034 §4): an out-of-range / illegal / refused swap stays an
//! explicit [`SwapError`] in *every* mode — the raw swap is run first and its error propagates
//! before any mode policy applies. The mode only tunes *certification*, never *fallibility* (G2).
//!
//! This is the policy layer; a future `@certification` scope (M-790) resolves the active mode and
//! feeds it here. Until then a [`ModeGatedSwapEngine`] carries an explicit mode (default
//! [`Fast`](CertMode::Fast)).

use mycelium_core::{BoundKind, CertMode, ContentHash, Meta, Repr, Value};
use mycelium_interp::{EvalError, SwapEngine};
use mycelium_numerics::Certificate;

use crate::{check, CheckVerdict, Evidence, RefinementRelation, SwapCertificate, SwapError};

/// The outcome of a **mode-gated** swap: the converted value (with its mode-reconciled `Meta`), the
/// certificate **iff** the mode emits one, and the check verdict **iff** the mode checks it. The
/// triple makes the mode's effect inspectable (no black box; RFC-0034 §3.1): in `Fast` both options
/// are `None`; in `Balanced` `certificate` is `Some` and `check` is `None`; in `Certified` both are
/// `Some`.
#[derive(Debug, Clone, PartialEq)]
pub struct GatedSwap {
    /// The converted value. Its `Meta` carries the [`CertMode`] tag and, in `Fast`, the
    /// `gate_result`-reconciled `(guarantee, bound)` pair.
    pub value: Value,
    /// The emitted certificate, or `None` when the mode does not emit one (`Fast`).
    pub certificate: Option<SwapCertificate>,
    /// The check verdict, or `None` when the mode does not check (`Fast`, `Balanced`).
    pub check: Option<CheckVerdict>,
}

impl GatedSwap {
    /// `true` iff this swap was certified-and-validated — i.e. the mode checked the certificate and
    /// it validated. `Fast`/`Balanced` (which do not check) are never `validated` (they make no
    /// validation claim — VR-5: absence of a check is not a pass).
    #[must_use]
    pub fn validated(&self) -> bool {
        matches!(self.check, Some(CheckVerdict::Validated { .. }))
    }
}

/// The [`RefinementRelation`] a certificate discharges under, and the `claimed` certificate to
/// present to the checker. Derived from the certificate itself so emission and checking never
/// disagree on the relation (DRY): a [`SwapCertificate::Bijective`] cert is the exact
/// [`RefinementRelation::Bijection`] relation with the `{0,0,Exact}` claim; a
/// [`SwapCertificate::Bounded`] cert is [`RefinementRelation::BoundedSimilarity`]
/// with the claim lifted from its own `{ε|δ, basis-strength}` (so the check re-derives the *same*
/// bound the cert states — never a tighter claim, VR-5).
fn relation_and_claim(cert: &SwapCertificate) -> Option<(RefinementRelation, Certificate)> {
    match cert {
        SwapCertificate::Bijective { .. } => {
            Some((RefinementRelation::Bijection, Certificate::exact()))
        }
        SwapCertificate::Bounded { bound, .. } => {
            let strength = mycelium_numerics::basis_strength(&bound.basis);
            let claimed = match bound.kind {
                BoundKind::Error { eps, .. } => Certificate::new(eps, 0.0, strength)?,
                BoundKind::Probability { delta } => Certificate::new(0.0, delta, strength)?,
                // Crosstalk/Capacity bounds are not (yet) checkable instances; no claim to present.
                BoundKind::Crosstalk { .. } | BoundKind::Capacity { .. } => return None,
            };
            Some((RefinementRelation::BoundedSimilarity, claimed))
        }
    }
}

/// Rebuild a value's `Meta` with the `Fast`-reconciled `(guarantee, bound)` pair (the mode's
/// `gate_result`), preserving every other field and tagging the mode. Only called for `Fast`; the
/// gated pair is guaranteed Meta-constructible (the `gate_result` contract), so the `WfError` arm is
/// defensive (it would only fire if a *non-Fast-gated* pair were passed — it never is here).
fn reconcile_fast_meta(value: &Value) -> Result<Value, SwapError> {
    let m = value.meta();
    let (guarantee, bound) = CertMode::Fast.gate_result(m.guarantee(), m.bound().cloned());
    let mut meta = Meta::new(
        m.provenance().clone(),
        guarantee,
        bound,
        m.sparsity(),
        m.physical(),
        m.policy_used().cloned(),
    )
    .map_err(SwapError::Wf)?;
    meta = meta.with_cert_mode(CertMode::Fast);
    Value::new(value.repr().clone(), value.payload().clone(), meta).map_err(SwapError::Wf)
}

/// Tag an already-honest value with a mode (no `(guarantee, bound)` change — `Balanced`/`Certified`
/// pass the pair through, so only the mode tag is recorded).
fn tag_mode(value: &Value, mode: CertMode) -> Result<Value, SwapError> {
    let meta = value.meta().clone().with_cert_mode(mode);
    Value::new(value.repr().clone(), value.payload().clone(), meta).map_err(SwapError::Wf)
}

/// Apply the [`CertMode`] policy to a **raw** swap result `(value, cert)` produced by one of the
/// crate's certificate-emitting swap functions, with `src` the swap's source value (the checker's
/// reference `A`). Returns the [`GatedSwap`].
///
/// `Fast` reconciles the value's `Meta` and drops the certificate (no emit, no check); `Balanced`
/// keeps the certificate, no check; `Certified` keeps the certificate and runs the M-210 check.
///
/// The raw swap must already have succeeded (Axis-B is not gated — RFC-0034 §4); any `SwapError`
/// from the swap itself is surfaced by the caller before this is reached.
pub fn gate_swap(
    src: &Value,
    value: Value,
    cert: SwapCertificate,
    mode: CertMode,
) -> Result<GatedSwap, SwapError> {
    match mode {
        CertMode::Fast => Ok(GatedSwap {
            value: reconcile_fast_meta(&value)?,
            certificate: None,
            check: None,
        }),
        CertMode::Balanced => Ok(GatedSwap {
            value: tag_mode(&value, CertMode::Balanced)?,
            certificate: Some(cert),
            check: None,
        }),
        CertMode::Certified => {
            // Emit + check (today's full behaviour). The relation/claim are derived from the
            // certificate so the check validates exactly what was emitted (VR-5: never tighter).
            let verdict = relation_and_claim(&cert).map(|(relation, claimed)| {
                check(src, &value, relation, claimed, &Evidence::Swap(&cert))
            });
            Ok(GatedSwap {
                value: tag_mode(&value, CertMode::Certified)?,
                certificate: Some(cert),
                check: verdict,
            })
        }
    }
}

/// A [`SwapEngine`] that wraps the [`CertifiedSwapEngine`](crate::CertifiedSwapEngine) surface and
/// applies the [`CertMode`] policy to every swap. The mode is **explicit** on the engine (default
/// [`Fast`](CertMode::Fast)) until the `@certification` scope (M-790) resolves it from context.
///
/// The trait method [`swap`](SwapEngine::swap) returns only the [`Value`] (the trait's contract);
/// the full [`GatedSwap`] (certificate + check verdict) is available from
/// [`swap_gated`](ModeGatedSwapEngine::swap_gated). In
/// `Certified` mode a check that does **not** validate is surfaced as an [`EvalError`] — never a
/// silent acceptance of an unvalidated swap (SC-3; RFC-0002 §2 fallback).
#[derive(Debug, Clone, Copy)]
pub struct ModeGatedSwapEngine {
    mode: CertMode,
}

impl Default for ModeGatedSwapEngine {
    fn default() -> Self {
        // The project default mode (RFC-0034 §5).
        Self {
            mode: CertMode::Fast,
        }
    }
}

impl ModeGatedSwapEngine {
    /// A mode-gated engine in the given [`CertMode`].
    #[must_use]
    pub fn new(mode: CertMode) -> Self {
        Self { mode }
    }

    /// The active [`CertMode`].
    #[must_use]
    pub fn mode(&self) -> CertMode {
        self.mode
    }

    /// Perform the swap and return the **full** [`GatedSwap`] (value + certificate + check verdict
    /// under the active mode). The raw swap runs first (Axis-B ungated): an unsupported/out-of-range
    /// swap is an explicit [`EvalError`] in every mode.
    pub fn swap_gated(
        &self,
        src: &Value,
        target: &Repr,
        policy: &ContentHash,
    ) -> Result<GatedSwap, EvalError> {
        let (value, cert) = raw_swap(src, target, policy)?;
        Ok(gate_swap(src, value, cert, self.mode)?)
    }
}

impl SwapEngine for ModeGatedSwapEngine {
    fn swap(&self, src: &Value, target: &Repr, policy: &ContentHash) -> Result<Value, EvalError> {
        let gated = self.swap_gated(src, target, policy)?;
        // Never-silent: in Certified, a non-validating check must not yield a value as if validated.
        if let Some(CheckVerdict::NotValidated { reason, .. }) = &gated.check {
            return Err(EvalError::Swap(format!(
                "certified swap did not validate: {reason:?} (fallback: keep the reference value)"
            )));
        }
        Ok(gated.value)
    }
}

/// Run the certificate-emitting swap for `(src.repr(), target)`, returning the raw
/// `(value, certificate)` before any mode policy. Same dispatch surface as
/// [`CertifiedSwapEngine`](crate::CertifiedSwapEngine), but it keeps the certificate (the trait's
/// `swap` discards it). Identity (same-`Repr`) swaps have no certificate object and are therefore
/// **not** gated through this module — they go through the plain engine.
fn raw_swap(
    src: &Value,
    target: &Repr,
    policy: &ContentHash,
) -> Result<(Value, SwapCertificate), EvalError> {
    use mycelium_core::ScalarKind;
    match (src.repr(), target) {
        (Repr::Binary { .. }, Repr::Ternary { trits }) => {
            Ok(crate::binary_to_ternary(src, *trits, policy)?)
        }
        (Repr::Ternary { .. }, Repr::Binary { width }) => {
            Ok(crate::ternary_to_binary(src, *width, policy)?)
        }
        (
            Repr::Dense {
                dim: src_dim,
                dtype: ScalarKind::F32,
            },
            Repr::Dense {
                dim: target_dim,
                dtype: ScalarKind::Bf16,
            },
        ) if src_dim == target_dim => Ok(crate::dense::dense_f32_to_bf16(src, policy)?),
        (
            Repr::Dense {
                dtype: ScalarKind::F32,
                ..
            },
            Repr::Vsa {
                model,
                dim,
                sparsity: mycelium_core::SparsityClass::Dense,
            },
        ) if model == crate::dense_vsa::DENSE_VSA_MODEL => Ok(crate::dense_vsa::dense_to_vsa(
            src,
            *dim,
            crate::DENSE_VSA_DEFAULT_DELTA,
            policy,
        )?),
        (
            Repr::Vsa { model, .. },
            Repr::Dense {
                dim,
                dtype: ScalarKind::F32,
            },
        ) if model == crate::dense_vsa::DENSE_VSA_MODEL => Ok(crate::dense_vsa::vsa_to_dense(
            src,
            *dim,
            crate::DENSE_VSA_DEFAULT_DELTA,
            policy,
        )?),
        // No certificate-emitting class matched: an unsupported swap is an explicit error
        // (identity is handled by the plain engine, which has no certificate to gate).
        (a, b) => Err(EvalError::UnsupportedSwap {
            from: a.clone(),
            to: b.clone(),
        }),
    }
}
