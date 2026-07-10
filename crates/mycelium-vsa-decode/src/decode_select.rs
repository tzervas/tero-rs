//! **Decode-methodology selection** (M-350; RFC-0010, Accepted; RFC-0005 §4 site 3; G2/G4/VR-5).
//!
//! A factorization request can be decoded by **more than one methodology**, and they carry **different
//! honest guarantee tags**: brute-force enumeration of all `∏ᵢ kᵢ` codebook combinations is **`Exact`**
//! (it *is* the RFC-0009 §5.3 differential oracle) but tractable only when the operational capacity is
//! small; the iterative resonator (RFC-0009 §3) is **`Empirical`** and only inside its validated
//! `{F, ∏kᵢ, d}` regime. This module reifies that choice as the **third site of the one RFC-0005
//! selection mechanism** (`mycelium-select`; no parallel selector — DRY/SoC):
//!
//! - the policy ([`decode_method_policy`]) is an ordered decision table over **exact** decode facts
//!   (`F`, `∏kᵢ`, `d`, regime membership) choosing among `{ BruteForceExact, Resonator, Refuse }`;
//! - selection goes through `mycelium_select::select_decode_method`, so every choice emits the
//!   mandatory [`Explanation`] (RFC-0005 §2.2 — no selection without EXPLAIN);
//! - the **guarantee tag is read off the chosen arm** (RFC-0010 §4.4): brute-force ⇒ `Exact` (with an
//!   identifiability check), resonator ⇒ `Empirical`, else an explicit `Refuse` — never a silent
//!   best-effort decode (G2).
//!
//! # Honesty floor (RFC-0010 §4.5 — non-negotiable)
//! A first-class override (`forced`) cannot escape the floor: a forced `BruteForceExact` whose `∏kᵢ`
//! exceeds the enumeration budget, a forced `BruteForceExact` on a **non-identifiable** instance, and a
//! forced `Resonator` **outside** the validated regime all still **`Refuse`**. The selector never
//! upgrades a tag (VR-5); the `mycelium-core::recon` `≤Empirical` ceiling is untouched.

use mycelium_core::{GuaranteeStrength, Meta, Provenance, Repr, SparsityClass};
use mycelium_select::{
    select_decode_method, Action, Candidate, CostModel, DecodeFacts, Predicate, Rule, SelectError,
    SelectionInputs, SelectionPolicy,
};

use mycelium_vsa::{
    factorize, CleanupMemory, Match, ResonatorParams, ResonatorTrace, VsaError, VsaModel,
    MAPI_RESONATOR_PROFILE,
};

// Re-export the selection types a caller needs to read a [`DecodeSelection`] (RFC-0010), so the
// decode site is usable without a direct `mycelium-select` dependency.
pub use mycelium_select::{DecodeMethod, Explanation};

/// The default enumeration budget: brute force is chosen when `∏ᵢ kᵢ ≤` this. Set to the widened
/// resonator capacity edge ([`MAPI_RESONATOR_PROFILE`]'s `max_capacity` = 4096) so that *every*
/// in-regime request gets the **strongest honest guarantee available** — the `Exact` brute-force
/// decode — and the resonator carries only the genuinely-intractable corners. Recorded in (and hashed
/// into) the policy identity — a different budget is a different `PolicyRef` (RFC-0010 §4.3).
///
/// **This default is guarantee-maximal, not latency-minimal.** The §8 wall-clock instrument
/// (`tests/decode_select.rs::decode_method_enum_budget_crossover`) measured the *cost-parity*
/// crossover at **`∏k ≈ 100–128`** (d-independent — both methods scale with `d`): brute force is
/// cheaper only for `∏k ≲ 64`, and at the regime edge `∏k=4096` it costs ≈ **19×** the resonator
/// (≈76 ms vs ≈4 ms at d=4096) to buy the `Exact` tag over the `Empirical` one. A latency-sensitive
/// caller passes a smaller budget (≈128 ⇒ cost-optimal); a caller wanting `Exact` past the regime
/// (e.g. an `F=4` enumerable instance) passes a larger one. The knob is the policy; the EXPLAIN cost
/// lines surface the trade per call (RFC-0010 §8).
pub const DEFAULT_ENUM_BUDGET: u128 = MAPI_RESONATOR_PROFILE.max_capacity;

/// The candidate-index convention of [`decode_method_policy`] (brute-force, resonator, refuse).
fn method_index(m: DecodeMethod) -> usize {
    match m {
        DecodeMethod::BruteForceExact => 0,
        DecodeMethod::Resonator => 1,
        DecodeMethod::Refuse => 2,
    }
}

/// Build the **default decode-method policy** (RFC-0010 §4): three candidates
/// `[BruteForceExact, Resonator, Refuse]` and an ordered table —
/// `(1)` `∏kᵢ ≤ enum_budget → BruteForceExact` (prefer the stronger `Exact` decode whenever
/// enumeration is tractable); `(2)` `in-regime → Resonator`; default `→ Refuse`. First match wins
/// (RFC-0005 §2.3). The policy is content-addressed; `enum_budget` is part of its identity.
#[must_use]
pub fn decode_method_policy(enum_budget: u128) -> SelectionPolicy {
    let candidates = vec![
        Candidate::Decode(DecodeMethod::BruteForceExact),
        Candidate::Decode(DecodeMethod::Resonator),
        Candidate::Decode(DecodeMethod::Refuse),
    ];
    let rules = vec![
        Rule {
            when: Predicate::CapacityAtMost(enum_budget),
            action: Action::Choose(0),
        },
        Rule {
            when: Predicate::InResonatorRegime,
            action: Action::Choose(1),
        },
    ];
    SelectionPolicy::new(
        "decode-method.v1",
        candidates,
        rules,
        method_index(DecodeMethod::Refuse), // mandatory default arm (RFC-0005 §2.1)
        CostModel {
            storage_weight: 1.0,
        },
    )
    .expect("the fixed decode-method policy is well-formed by construction")
}

/// Compute the **exact decode facts** for a request and pack them into the RFC-0005 queryable inputs
/// (with a `Vsa` `src` so the EXPLAIN trace also carries the representation). `in_regime` is
/// `MAPI_RESONATOR_PROFILE::check(F, kᵢ, d)` — a fact about the inputs, not a sampled estimate.
fn decode_inputs<M: VsaModel>(model: &M, sizes: &[usize], dim: u32) -> SelectionInputs {
    let mut capacity: u128 = 1;
    for &k in sizes {
        capacity = capacity.saturating_mul(k as u128);
    }
    let in_regime = MAPI_RESONATOR_PROFILE
        .check(sizes.len(), sizes, dim)
        .is_ok();
    let facts = DecodeFacts {
        factors: sizes.len() as u32,
        capacity,
        dim,
        in_regime,
    };
    let src = Repr::Vsa {
        model: model.model_id().to_owned(),
        dim,
        sparsity: SparsityClass::Dense,
    };
    SelectionInputs::from_meta(src, &Meta::exact(Provenance::Root)).with_decode(facts)
}

/// The mandatory **EXPLAIN** for a decode-method choice (RFC-0005 §4), without executing the decode:
/// which arm the policy picks for `{F, ∏kᵢ, d}` and the per-candidate costs. Total and deterministic —
/// re-derivable from `(policy, inputs)` alone (the LSP EXPLAIN surface consumes this; SC-5).
#[must_use]
pub fn explain_decode_method<M: VsaModel>(
    model: &M,
    sizes: &[usize],
    dim: u32,
    enum_budget: u128,
) -> Explanation {
    let policy = decode_method_policy(enum_budget);
    let inputs = decode_inputs(model, sizes, dim);
    mycelium_select::explain(&policy, &inputs)
}

/// A reified decode-method selection result (RFC-0010): the chosen methodology, the mandatory EXPLAIN
/// trace, the recovered per-slot factors, the **guarantee tag the chosen arm earns**, and — for the
/// resonator arm — its inspectable run trace.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeSelection {
    /// The methodology the policy (or a forced override) selected.
    pub method: DecodeMethod,
    /// The mandatory selection EXPLAIN record (inputs, per-candidate costs, matched rule, chosen arm).
    pub explanation: Explanation,
    /// The recovered factor per slot (in slot order).
    pub factors: Vec<Match>,
    /// The guarantee tag of this decode — `Exact` (brute force) or `Empirical` (resonator). Read off
    /// the chosen arm, never asserted independently (RFC-0010 §4.4 / VR-5).
    pub guarantee: GuaranteeStrength,
    /// The resonator run trace, present only when the resonator arm ran (`EXPLAIN` on the loop too).
    pub resonator_trace: Option<ResonatorTrace>,
}

/// **Automatic factor reconstruction** (RFC-0010): select the decode methodology for `s` against
/// `codebooks` and run it, returning the recovered factors with the **guarantee tag of the chosen
/// arm**. Brute-force enumeration (when `∏ᵢ kᵢ ≤ enum_budget`) is `Exact` and identifiability-checked;
/// the resonator (in-regime) is `Empirical`; otherwise an explicit [`VsaError::DecodeRefused`].
///
/// `forced` pins an arm (first-class override, recorded in the EXPLAIN), but **cannot escape the
/// honesty floor** (RFC-0010 §4.5): a forced `BruteForceExact` beyond `enum_budget` or on a
/// non-identifiable instance, and a forced `Resonator` out of regime, all still refuse.
pub fn reconstruct_factors_auto<M: VsaModel>(
    model: &M,
    s: &[f64],
    codebooks: &[CleanupMemory],
    resonator_params: &ResonatorParams,
    enum_budget: u128,
    forced: Option<DecodeMethod>,
) -> Result<DecodeSelection, VsaError> {
    // --- input validation (explicit, never silent — mirrors `resonator::run_loop`) ---
    if codebooks.is_empty() {
        return Err(VsaError::EmptyCodebook);
    }
    let dim = s.len();
    for cb in codebooks {
        if cb.is_empty() {
            return Err(VsaError::EmptyCodebook);
        }
        if cb.dim() as usize != dim {
            return Err(VsaError::DimMismatch {
                expected: dim,
                got: cb.dim() as usize,
            });
        }
    }

    let sizes: Vec<usize> = codebooks.iter().map(CleanupMemory::len).collect();
    let dim_u32 = dim as u32;
    let policy = decode_method_policy(enum_budget);
    let inputs = decode_inputs(model, &sizes, dim_u32);
    let (method, explanation) =
        select_decode_method(&policy, &inputs, forced.map(method_index)).map_err(select_err)?;

    match method {
        DecodeMethod::BruteForceExact => {
            // Honesty floor: a *forced* brute force could exceed the budget; refuse rather than
            // enumerate an intractable grid (RFC-0010 §4.5).
            let capacity = inputs.decode.map_or(0, |d| d.capacity);
            if capacity > enum_budget {
                return Err(VsaError::DecodeRefused {
                    detail: format!(
                        "forced BruteForceExact but ∏k={capacity} exceeds the enumeration budget {enum_budget}"
                    ),
                });
            }
            let (tuple, best, second) = brute_force_argmax(model, s, codebooks)?;
            // Identifiability: the true tuple must be the *unique* global arg-max for an Exact claim.
            if second.is_finite() && best <= second {
                return Err(VsaError::NonIdentifiable {
                    runner_up_similarity: second,
                });
            }
            let runner_up = if second.is_finite() { second } else { -1.0 };
            let factors = exact_factors(codebooks, &tuple, best, runner_up);
            Ok(DecodeSelection {
                method,
                explanation,
                factors,
                guarantee: GuaranteeStrength::Exact,
                resonator_trace: None,
            })
        }
        DecodeMethod::Resonator => {
            // Honesty floor: a *forced* resonator could be out of regime; refuse with the profile's
            // own explicit reason rather than running an unvalidated decode (RFC-0010 §4.5).
            MAPI_RESONATOR_PROFILE.check(sizes.len(), &sizes, dim_u32)?;
            let out = factorize(model, s, codebooks, resonator_params)?;
            Ok(DecodeSelection {
                method,
                explanation,
                factors: out.factors,
                guarantee: GuaranteeStrength::Empirical,
                resonator_trace: Some(out.trace),
            })
        }
        DecodeMethod::Refuse => Err(VsaError::DecodeRefused {
            detail: format!(
                "∏k={} exceeds the enumeration budget {enum_budget} and the request is outside the \
                 resonator regime (F={}, d={dim})",
                inputs.decode.map_or(0, |d| d.capacity),
                sizes.len()
            ),
        }),
    }
}

/// Map a `mycelium-select` adapter error onto the explicit decode refusal (G2 — never a silent pick).
fn select_err(e: SelectError) -> VsaError {
    VsaError::DecodeRefused {
        detail: format!("decode-method selection failed: {e}"),
    }
}

/// The brute-force oracle as a decode (RFC-0009 §5.3): the global arg-max tuple over all `∏ᵢ kᵢ`
/// combinations, plus the best and runner-up similarities (the identifiability margin). Only ever
/// called when `∏ᵢ kᵢ ≤ enum_budget`, so the enumeration is bounded.
fn brute_force_argmax<M: VsaModel>(
    model: &M,
    s: &[f64],
    codebooks: &[CleanupMemory],
) -> Result<(Vec<usize>, f64, f64), VsaError> {
    let f = codebooks.len();
    let atoms: Vec<Vec<&[f64]>> = codebooks
        .iter()
        .map(|cb| cb.atoms().map(|(_, a)| a).collect())
        .collect();
    let mut idx = vec![0usize; f];
    let mut best_tuple = vec![0usize; f];
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    loop {
        // Bind the current combination and score it against `s`.
        let mut acc = atoms[0][idx[0]].to_vec();
        for (slot, row) in atoms.iter().enumerate().skip(1) {
            acc = model.bind(&acc, row[idx[slot]])?;
        }
        let sim = model.similarity(s, &acc);
        if sim > best {
            second = best;
            best = sim;
            best_tuple.clone_from(&idx);
        } else if sim > second {
            second = sim;
        }
        // Increment the mixed-radix counter; return once the most-significant slot overflows.
        let mut carry = 0;
        idx[carry] += 1;
        while idx[carry] == atoms[carry].len() {
            idx[carry] = 0;
            carry += 1;
            if carry == f {
                return Ok((best_tuple, best, second));
            }
            idx[carry] += 1;
        }
    }
}

/// Build the per-slot [`Match`] set for an `Exact` brute-force decode. The decode is exact, so each
/// slot's confidence is the (global) product similarity (`= 1.0` for a clean bipolar instance) and the
/// margin is the global identifiability gap `best − runner_up` — the honest quantity behind the
/// `Exact` claim (a small gap is what the [`VsaError::NonIdentifiable`] refusal guards).
fn exact_factors(
    codebooks: &[CleanupMemory],
    tuple: &[usize],
    best: f64,
    runner_up: f64,
) -> Vec<Match> {
    codebooks
        .iter()
        .zip(tuple)
        .map(|(cb, &index)| {
            let label = cb
                .atoms()
                .nth(index)
                .map_or_else(|| index.to_string(), |(l, _)| l.to_owned());
            Match {
                label,
                index,
                confidence: best,
                margin: best - runner_up,
            }
        })
        .collect()
}
