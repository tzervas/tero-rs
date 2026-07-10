//! The first **Bounded/lossy swap**: Dense `F32 → BF16` rounding (M-211; RFC-0002 §3/§5 — the
//! legal-pair table's "Dense `F32` → `BF16`, Bounded (ε), rounding-error theory"; ADR-010 §1).
//!
//! Establishes the *split regime* (ADR-002) alongside the bijective binary↔ternary class: this
//! swap is genuinely lossy, so it emits a [`SwapCertificate::Bounded`] carrying an ε [`Bound`]
//! whose `basis` is **derived from how the bound was obtained, never asserted** (RFC-0002 §3):
//! the standard round-to-nearest relative-error theorem with its side-conditions *checked per
//! element* — every input finite, exactly an `f32`, zero or **normal** in the bfloat16 range, and
//! not overflowing on rounding. When a side-condition fails the swap **refuses with an explicit
//! [`SwapError`]** rather than emitting a bound the theorem does not cover (the honesty rule;
//! VR-5; "a pair with no statable bound is a type error, not a `Declared` gamble", RFC-0002 §5).
//!
//! The emitted certificate validates through the M-210 shared checker
//! ([`mod@crate::check`]) under [`RefinementRelation::BoundedSimilarity`](crate::RefinementRelation).
//!
//! **Honest scope (v1).** Subnormal elements and approximate (non-`Exact`) sources are refused
//! explicitly: the subnormal range is outside the cited theorem's side-conditions (its absolute
//! half-spacing bound is future work), and composing the rounding ε with an input's own bound is
//! the E2-1 Dense-numerics rule that does not exist yet — refusal, never fabrication.

use mycelium_core::{
    operation_hash, Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, NormKind,
    Payload, Provenance, Repr, ScalarKind, Value,
};

use crate::{SwapCertificate, SwapError};

/// The proven per-element relative rounding bound: the unit roundoff `u = β^(1−p)/2 = 2^(1−8)/2 =
/// 2^−8` for bfloat16's `p = 8` significand bits (7 stored + 1 implicit) under round-to-nearest.
pub const BF16_REL_EPS: f64 = 0.003_906_25; // 2⁻⁸, exact in f64

/// Smallest positive *normal* bfloat16 (same exponent range as f32): `2^−126`. Below this the
/// relative-error theorem's side-condition fails (subnormal spacing is absolute, not relative).
pub const BF16_MIN_NORMAL: f64 = f32::MIN_POSITIVE as f64;

/// The cited theorem behind the `ProvenThm` basis. Its side-conditions are the ones
/// [`dense_f32_to_bf16`] checks per element; the citation is accepted axiomatically and only the
/// instantiation is checked (RFC-0002 §7; ADR-010 §1).
const BF16_ROUNDING_CITATION: &str = "round-to-nearest relative error ≤ u = β^(1−p)/2 = 2^−8 for \
     bfloat16 (β=2, p=8) — Higham, Accuracy and Stability of Numerical Algorithms (2002), Thm 2.2; \
     side-conditions checked per element: finite, exact f32, zero-or-normal, no overflow on rounding";

/// Round an `f32` to the nearest bfloat16 (ties to even), returning the result widened back to
/// `f32` bit-exactly (bf16 is the top 16 bits of the f32 format). Standard trick: add `0x7FFF +
/// lsb` so the carry performs round-to-nearest-even on the truncated 16 bits. Caller has excluded
/// NaN/Inf, so the addition cannot wrap.
fn round_f32_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16)
}

/// Round one element under the theorem's checked side-conditions; any violation is an explicit
/// [`SwapError`] carrying the element index — never a silent coercion (SC-3; G2).
fn round_element(x: f64, index: usize) -> Result<f64, SwapError> {
    if !x.is_finite() {
        return Err(SwapError::NonFinite { index });
    }
    #[allow(clippy::cast_possible_truncation)] // checked just below: refuse if it rounded
    let xf = x as f32;
    if f64::from(xf) != x {
        return Err(SwapError::NotAnF32 { index });
    }
    if x != 0.0 && x.abs() < BF16_MIN_NORMAL {
        return Err(SwapError::SubnormalUnsupported { index });
    }
    let rounded = round_f32_to_bf16(xf);
    if !rounded.is_finite() {
        return Err(SwapError::RoundOverflow { index });
    }
    Ok(f64::from(rounded))
}

/// The Dense `F32 → BF16` rounding swap: returns the converted value and a
/// [`SwapCertificate::Bounded`] whose ε bound (`Rel`, `2^−8`) carries a `ProvenThm` basis — the
/// guarantee strength is derived from the basis, never asserted (RFC-0002 §3; ADR-011).
///
/// Refusals (all explicit, RFC-0002 §5): a non-`Dense{F32}` source, an approximate source (no
/// composition rule yet — E2-1), and any element that is NaN/±Inf, not exactly an `f32`,
/// subnormal, or overflowing bf16's finite range on rounding.
pub fn dense_f32_to_bf16(
    src: &Value,
    policy: &ContentHash,
) -> Result<(Value, SwapCertificate), SwapError> {
    let Repr::Dense {
        dim,
        dtype: ScalarKind::F32,
    } = *src.repr()
    else {
        return Err(SwapError::WrongSource {
            expected: "Dense{F32}",
        });
    };
    let Payload::Scalars(xs) = src.payload() else {
        return Err(SwapError::WrongSource {
            expected: "Dense{F32}",
        });
    };
    if src.meta().guarantee() != GuaranteeStrength::Exact {
        return Err(SwapError::ApproximateSource);
    }
    let mut out = Vec::with_capacity(xs.len());
    for (index, &x) in xs.iter().enumerate() {
        out.push(round_element(x, index)?);
    }
    let bound = Bound {
        kind: BoundKind::Error {
            eps: BF16_REL_EPS,
            norm: NormKind::Rel,
        },
        basis: BoundBasis::ProvenThm {
            citation: BF16_ROUNDING_CITATION.to_owned(),
        },
    };
    // The result honestly discloses what it is: Proven (basis-matched, M-I2), bound attached,
    // provenance over the source, policy recorded (ADR-006).
    let meta = Meta::new(
        Provenance::Derived {
            op: operation_hash("swap.dense.f32_bf16"),
            inputs: vec![src.content_hash()],
        },
        GuaranteeStrength::Proven,
        Some(bound.clone()),
        None,
        None,
        Some(policy.clone()),
    )
    .map_err(SwapError::Wf)?;
    let target = Repr::Dense {
        dim,
        dtype: ScalarKind::Bf16,
    };
    let value = Value::new(target.clone(), Payload::Scalars(out), meta).map_err(SwapError::Wf)?;
    let cert = SwapCertificate::Bounded {
        src: src.repr().clone(),
        target,
        policy_used: policy.clone(),
        bound,
    };
    Ok((value, cert))
}
