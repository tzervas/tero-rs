//! The **Dense ↔ VSA bounded swap** (M-231; RFC-0002 §5 — the legal-pair table's "Dense ↔ VSA,
//! Bounded/probabilistic (ε, δ), VSA capacity results"; RFC-0003; T0.2; ADR-010).
//!
//! `dense_to_vsa` encodes a **bipolar** Dense vector `x ∈ {−1,+1}ⁿ` as the MAP-I superposition of
//! signed atoms from a **deterministic, versioned codebook**: `hv = Σᵢ xᵢ·aᵢ` — a genuine
//! `n`-item bundle of bipolar atoms, so the **cited capacity theorem applies** (T0.2:
//! Clarkson/Thomas). `vsa_to_dense` decodes by signed correlation: `x̂ᵢ = sign(⟨hv, aᵢ⟩)`. The
//! emitted certificate is a **δ** (`ProbabilityBound`) and its basis is *derived from how the
//! bound was obtained, never asserted* (RFC-0002 §3):
//!
//! - **`ProvenThm`** when the checked instantiation holds — `vsa_dim ≥ requiredDim(n, δ)` (the
//!   M-001/M-131 pattern, reusing `mycelium_vsa::capacity`); the swap fails to decode with
//!   probability ≤ δ.
//! - **`EmpiricalFit`** otherwise, when the *trial-validated profile* covers the instance
//!   (`n ≤ 16`, `vsa_dim ≥ 32·n`, δ = [`DENSE_VSA_EMP_DELTA`]) — exercised with exactly
//!   [`DENSE_VSA_EMP_TRIALS`] round-trip trials in `tests/dense_vsa.rs`.
//! - An instance neither basis covers is an explicit [`SwapError::InsufficientCapacity`] — a
//!   type error, not a `Declared` gamble (RFC-0002 §5).
//!
//! **Honest scope (v1).** Only bipolar Dense vectors encode (a non-`±1` component is an explicit
//! [`SwapError::NotBipolar`] — the capacity theorem covers bundles of bipolar atoms, and a
//! weighted-superposition bound is not in the corpus); only `swap.dense_vsa.enc.v1` products
//! decode (the δ describes *this* encoding's retrieval, nothing else — provenance-gated like the
//! M-241 unbind regime); a vanished correlation is an explicit [`SwapError::AmbiguousDecode`].

use mycelium_core::{
    operation_hash, Bound, BoundBasis, BoundKind, ContentHash, GuaranteeStrength, Meta, Payload,
    Provenance, Repr, ScalarKind, SparsityClass, Value,
};
use mycelium_vsa::capacity;

use crate::{SwapCertificate, SwapError};

/// The op name of the encode direction — also the provenance marker `vsa_to_dense` requires.
pub(crate) const ENC_OP: &str = "swap.dense_vsa.enc.v1";
/// The op name of the decode direction.
pub(crate) const DEC_OP: &str = "swap.dense_vsa.dec.v1";

/// The VSA model the swap targets (the atoms are bipolar and the encoding is the MAP-I additive
/// superposition).
pub const DENSE_VSA_MODEL: &str = "MAP-I";

/// Empirical profile — maximum Dense components covered by the trials.
pub const DENSE_VSA_EMP_MAX_COMPONENTS: u32 = 16;
/// Empirical profile — minimum `vsa_dim / components` ratio covered by the trials.
pub const DENSE_VSA_EMP_DIM_FACTOR: u32 = 32;
/// Empirical profile — the validated δ.
pub const DENSE_VSA_EMP_DELTA: f64 = 0.05;
/// Empirical profile — the trial count `tests/dense_vsa.rs` runs.
pub const DENSE_VSA_EMP_TRIALS: u64 = 10_000;
/// Empirical profile — the method recorded in the `EmpiricalFit` basis.
pub const DENSE_VSA_EMP_METHOD: &str = "Monte-Carlo enc→dec sign-recovery round trip (bipolar \
     components, n ≤ 16, vsa_dim ≥ 32·n, versioned deterministic codebook)";

const CAPACITY_SWAP_CITATION: &str = "MAP-I superposition retrieval fails w.p. ≤ δ when \
     d ≥ requiredDim(m, δ) = ⌈(2/μ²)·ln(m/δ)⌉ (μ = 0.1) — Clarkson-Ubaru-Yang 2023 (Thm 6); \
     Thomas-Dasgupta-Rosing 2021; side-conditions checked: bipolar components, \
     vsa_dim ≥ requiredDim(components, δ)";

/// The `i`-th codebook atom at `dim` — deterministic and versioned (`enc.v1`), so every party
/// (the swap, the decoder, the M-210 checker's re-derivation) reconstructs the identical
/// codebook from the op name alone. A tiny LCG keyed by the atom index (house style).
fn codebook_atom(i: usize, dim: u32) -> Vec<f64> {
    let mut s = (i as u64 ^ 0xD5EA_C0DE_0000_0001)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (s >> 63) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// The honest δ bound for `(components, vsa_dim, delta)`: `ProvenThm` iff the checked capacity
/// instantiation holds; else `EmpiricalFit` iff the trial-validated profile covers the instance
/// (and the requested δ is not tighter than the validated one); else an explicit refusal.
fn delta_bound(components: u32, vsa_dim: u32, delta: f64) -> Result<Bound, SwapError> {
    let required = capacity::required_dim(u64::from(components), delta, capacity::MARGIN_MU);
    if u64::from(vsa_dim) >= required {
        return Ok(Bound {
            kind: BoundKind::Probability { delta },
            basis: BoundBasis::ProvenThm {
                citation: CAPACITY_SWAP_CITATION.to_owned(),
            },
        });
    }
    let profile_covers = components <= DENSE_VSA_EMP_MAX_COMPONENTS
        && vsa_dim >= components * DENSE_VSA_EMP_DIM_FACTOR
        && delta >= DENSE_VSA_EMP_DELTA;
    if profile_covers {
        return Ok(Bound {
            kind: BoundKind::Probability {
                delta: DENSE_VSA_EMP_DELTA,
            },
            basis: BoundBasis::EmpiricalFit {
                trials: DENSE_VSA_EMP_TRIALS,
                method: DENSE_VSA_EMP_METHOD.to_owned(),
            },
        });
    }
    Err(SwapError::InsufficientCapacity {
        components,
        dim: vsa_dim,
        required,
    })
}

fn swap_meta(
    op: &str,
    src: &Value,
    bound: &Bound,
    policy: &ContentHash,
) -> Result<Meta, SwapError> {
    Meta::new(
        Provenance::Derived {
            op: operation_hash(op),
            inputs: vec![src.content_hash()],
        },
        mycelium_numerics::basis_strength(&bound.basis),
        Some(bound.clone()),
        None,
        None,
        Some(policy.clone()),
    )
    .map_err(SwapError::Wf)
}

/// Encode a bipolar `Dense{n, F32}` value into a `Vsa{MAP-I, vsa_dim}` superposition, emitting a
/// `Bounded` certificate whose δ basis is derived (module docs). Explicit refusals: a
/// non-`Dense{F32}` source, an approximate source (the composition rule with a source bound is
/// not defined — same M-211 scope), a non-bipolar component, and an instance no basis covers.
pub fn dense_to_vsa(
    src: &Value,
    vsa_dim: u32,
    delta: f64,
    policy: &ContentHash,
) -> Result<(Value, SwapCertificate), SwapError> {
    let Repr::Dense {
        dim: components,
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
    if let Some(index) = xs.iter().position(|&x| x != 1.0 && x != -1.0) {
        return Err(SwapError::NotBipolar { index });
    }
    let bound = delta_bound(components, vsa_dim, delta)?;

    let mut hv = vec![0.0f64; vsa_dim as usize];
    for (i, &x) in xs.iter().enumerate() {
        for (h, a) in hv.iter_mut().zip(codebook_atom(i, vsa_dim)) {
            *h += x * a;
        }
    }
    let target = Repr::Vsa {
        model: DENSE_VSA_MODEL.to_owned(),
        dim: vsa_dim,
        sparsity: SparsityClass::Dense,
    };
    let meta = swap_meta(ENC_OP, src, &bound, policy)?;
    let value =
        Value::new(target.clone(), Payload::Hypervector(hv), meta).map_err(SwapError::Wf)?;
    let cert = SwapCertificate::Bounded {
        src: src.repr().clone(),
        target,
        policy_used: policy.clone(),
        bound,
    };
    Ok((value, cert))
}

/// Decode a `swap.dense_vsa.enc.v1` product back to a bipolar `Dense{components, F32}` value by
/// signed correlation against the same versioned codebook. The δ certificate re-derives from the
/// instance's own side-conditions (the failure event *is* this decode mis-recovering a
/// component). Explicit refusals: a non-encoding source (provenance-gated — the bound describes
/// nothing else), a components/dim pair no basis covers, a vanished correlation.
pub fn vsa_to_dense(
    src: &Value,
    components: u32,
    delta: f64,
    policy: &ContentHash,
) -> Result<(Value, SwapCertificate), SwapError> {
    let Repr::Vsa {
        ref model,
        dim: vsa_dim,
        ..
    } = *src.repr()
    else {
        return Err(SwapError::WrongSource {
            expected: "VSA{MAP-I}",
        });
    };
    if model != DENSE_VSA_MODEL {
        return Err(SwapError::WrongSource {
            expected: "VSA{MAP-I}",
        });
    }
    let Payload::Hypervector(hv) = src.payload() else {
        return Err(SwapError::WrongSource {
            expected: "VSA{MAP-I}",
        });
    };
    // The δ describes retrieval from *this* encoding; decoding anything else would tag a bound
    // that covers nothing (VR-5) — provenance-gated like the M-241 unbind regime.
    match src.meta().provenance() {
        Provenance::Derived { op, .. } if op == &operation_hash(ENC_OP) => {}
        _ => return Err(SwapError::NotDenseVsaEncoding),
    }
    let bound = delta_bound(components, vsa_dim, delta)?;

    let mut xs = Vec::with_capacity(components as usize);
    for i in 0..components as usize {
        let dot: f64 = hv
            .iter()
            .zip(codebook_atom(i, vsa_dim))
            .map(|(h, a)| h * a)
            .sum();
        if dot == 0.0 {
            return Err(SwapError::AmbiguousDecode { index: i });
        }
        xs.push(if dot > 0.0 { 1.0 } else { -1.0 });
    }
    let target = Repr::Dense {
        dim: components,
        dtype: ScalarKind::F32,
    };
    let meta = swap_meta(DEC_OP, src, &bound, policy)?;
    let value = Value::new(target.clone(), Payload::Scalars(xs), meta).map_err(SwapError::Wf)?;
    let cert = SwapCertificate::Bounded {
        src: src.repr().clone(),
        target,
        policy_used: policy.clone(),
        bound,
    };
    Ok((value, cert))
}
