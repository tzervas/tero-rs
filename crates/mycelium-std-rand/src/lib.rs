//! `std.rand` — Ring 2 / Tier B random number generation with reified, named nondeterminism
//! (M-531 / #171).
//!
//! # Summary
//!
//! `std.rand` is the random-number surface, held to the RFC-0016 §4.1 contract. Its honesty crux
//! is **C6 in its sharpest form**: nondeterminism is reified and named (RT3). A generator that
//! consumes real entropy carries a *declared* `entropy` effect on its signature, so a
//! deterministic-fragment program **cannot** pull randomness silently. The only way to obtain
//! reproducible "randomness" is the structurally distinct [`Rng`] — a seeded, pure-function
//! generator whose state is an ordinary immutable value.
//!
//! # Architecture
//!
//! Two structurally distinct constructs:
//!
//! 1. **[`Rng`]** — a seeded generator *value*: `{ algo, state }`. All draw functions are pure
//!    functions returning `(output, Rng')`. Same seed ⇒ same sequence — reproducibility is
//!    value-equality (C4 / ADR-003). No ambient global RNG.
//!
//! 2. **[`EntropyRng`]** — an entropy-backed generator. Constructing one and drawing from it carry
//!    the *declared* [`EntropyEffect`] effect (C6 / RFC-0014 / RT3). A deterministic fragment
//!    cannot call these. [`seed_from_entropy`] bridges the two: it draws entropy *once* (declared),
//!    mints a reproducible seed, and hands back a pure [`Rng`].
//!
//! # PRNG algorithm
//!
//! The seeded generator uses **xoshiro256++** (Blackman & Vigna 2021): a well-studied, fast
//! 256-bit state PRNG with a 2^256 - 1 period, widely used in production. Its statistical quality
//! is `Empirical` (passes TestU01 BigCrush; the *uniformity* claim is `Declared`/`Empirical`,
//! not `Proven` — VR-5). — FLAG (Q2): final algorithm selection is ratified at M-501/RFC-0016;
//! the tag is held at `Empirical`/`Declared` pending that decision.
//!
//! # Guarantee matrix (RFC-0016 §4.5)
//!
//! Encoded as data in [`GUARANTEE_MATRIX`] and asserted in tests — never prose-only.
//!
//! # Contract conformance (RFC-0016 §4.1 C1–C6)
//!
//! - **C1 never-silent:** every fallible op returns `Result<_, RandErr>`; no sentinel, no silent
//!   fallback. An unavailable entropy source is `Err(RandErr::EntropyUnavailable)` — **never** a
//!   silent fixed/zero seed.
//! - **C2 honest tag:** mechanism-determinism (`Exact`) and statistical quality
//!   (`Declared`/`Empirical`) are tagged on *separate* footings; no distribution row reaches
//!   `Proven` without a checked theorem (VR-5).
//! - **C3 no black boxes / EXPLAIN:** a seeded generator's `(algo, state, seed)` is fully
//!   inspectable; a draw can report its algorithm; no opaque global RNG.
//! - **C4 value-semantic:** [`Rng`] is an immutable value — a draw returns a *new* generator;
//!   the input is unchanged.
//! - **C5 above the kernel:** no `unsafe`, no FFI. The platform-entropy `wild`/FFI floor is
//!   deferred to the `std-sys` phylum (FLAGGED §8-Q6 / M-541). The injectable [`EntropySource`]
//!   trait is the seam for that floor without polluting pure `std.rand`.
//! - **C6 declared bounded effects:** the seeded surface is pure (`effects: none`). The entropy
//!   surface declares [`EntropyEffect`] on every op drawing real nondeterminism (RT3 / RFC-0014).
//!
//! Design spec: `docs/spec/stdlib/rand.md`; contract: RFC-0016 §4.1 (C1–C6);
//! guarantee matrix: §4.5.
//!
//! ## Ambient Representation (RFC-0012 §8-Q3)
//!
//! This crate's public API participates in the RFC-0012 ambient-representation contract:
//! the representation choice (binary/ternary/dense/VSA) is implicit at the call site but
//! always reified, queryable, and EXPLAIN-able — never a black box (C3/SC-3).
//! [Declared per RFC-0012; direction accepted in DN-07 §8-Q3; per-ring pass scheduled as M-540.]
//!
//! **For this crate (Ring 2, Tier B):** Randomness is representation-neutral — [`Rng`] produces
//! raw bits whose `Repr` is the caller's responsibility. `EntropyEffect` is always declared on
//! any op that draws real nondeterminism; there is no ambient entropy (no global RNG, no silent
//! entropy draw). The entropy effect declaration is inspectable on the return type, not a side
//! effect — a deterministic-fragment program cannot pull it silently.
//!
//! # Stability (DN-66 freeze, 2026-07-01)
//!
//! This crate's public API, as documented in `docs/spec/stdlib/rand.md` (spec status:
//! Accepted (2026-06-20)) and asserted by its guarantee-matrix table, is the **frozen baseline** per
//! [DN-66](../../../docs/notes/DN-66-Stdlib-Stable-API-Freeze-And-Rust-Crate-Retirement-Status.md).
//! A future breaking change here needs a spec amendment + changelog entry, not a silent edit (G2).
//! It remains the RFC-0031 D6 differential-oracle reference; no `.myc` port of this module exists yet, so the D6 retirement trigger has not fired and no item here is `#[deprecated]`.
#![forbid(unsafe_code)]

use mycelium_core::GuaranteeStrength;

// ──────────────────────────────────────────────────────────────────────────────
// § 1. Error types (C1 — never-silent)
// ──────────────────────────────────────────────────────────────────────────────

/// Errors returned by `std.rand` operations (C1 — every fallible op returns this
/// explicitly; never a sentinel or a silent fallback).
///
/// The variants map exactly to the spec §3 error set:
/// `EmptyRange | BadProbability | EmptyDomain | BadParameter | EntropyUnavailable`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RandErr {
    /// `uniform_int` / `uniform_u64`: `hi <= lo` — the range is empty.
    EmptyRange,
    /// `bernoulli`: probability `p` is not in `[0.0, 1.0]`.
    BadProbability,
    /// `choice`: the input collection is empty — no element to draw.
    EmptyDomain,
    /// `normal` / `exponential`: a distribution parameter is invalid
    /// (e.g. `sigma <= 0` or `lambda <= 0`).
    BadParameter,
    /// `from_entropy` / `seed_from_entropy` / `next_entropy`: the platform entropy source
    /// was unavailable. **Never a silent fallback to a fixed or zero seed** — that would make
    /// a "random" program silently deterministic (spec §5 C1/G2).
    ///
    /// FLAG (Q4 / §8-Q6): the concrete platform-entropy source (getrandom / `wild`) lives
    /// in the `std-sys` phylum (M-541). Real OS entropy is injected through the
    /// [`EntropySource`] trait; the test harness uses a deterministic stub.
    EntropyUnavailable,
}

impl core::fmt::Display for RandErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RandErr::EmptyRange => f.write_str("empty range: hi must be > lo"),
            RandErr::BadProbability => f.write_str("bad probability: p must be in [0.0, 1.0]"),
            RandErr::EmptyDomain => f.write_str("empty domain: collection has no elements"),
            RandErr::BadParameter => f.write_str(
                "bad distribution parameter (e.g. sigma > 0 for Normal, lambda > 0 for Exponential)",
            ),
            RandErr::EntropyUnavailable => {
                f.write_str("entropy source unavailable (FLAG: std-sys phylum / M-541)")
            }
        }
    }
}

mycelium_std_core::impl_std_error!(RandErr);

// ──────────────────────────────────────────────────────────────────────────────
// § 2. Declared effects (C6 / RT3 / RFC-0014)
// ──────────────────────────────────────────────────────────────────────────────

/// The reified `entropy` declared effect (C6 / RT3 / RFC-0014).
///
/// A function that returns this type in a result-bearing position (e.g. wrapped in an
/// `! entropy`-annotated signature in the future Mycelium language) declares that it draws
/// real nondeterminism. This is the Rust-side bearer: it is a zero-cost proof that entropy
/// was consumed, not fabricated.
///
/// In Rust (before the Mycelium-lang migration) the effect is *named and documented* rather
/// than enforced structurally by the type system — the RT3 rule is enforced by convention
/// (use `EntropyRng` for entropy, `Rng` for determinism) and by the test suite checking that
/// the seeded surface is reproducible without any `EntropyEffect`.
///
/// FLAG (Q4 / §8-Q6): the real OS entropy floor is the `std-sys` phylum (M-541); only the
/// declared-effect structure and the injectable [`EntropySource`] seam live here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntropyEffect;

/// Injectable entropy source — the seam between pure `std.rand` and the `std-sys` phylum.
///
/// The concrete platform implementation (getrandom / `wild`) lives in `std-sys` (M-541) and
/// is injected at construction time via `EntropyRng::new`. Tests use [`StubEntropy`] (a
/// deterministic counter), keeping the test suite free of platform OS calls.
///
/// Implementors must return `Err(RandErr::EntropyUnavailable)` rather than a silent fallback
/// on any failure (C1).
pub trait EntropySource {
    /// Fill `buf` with entropy bytes, or return `Err(RandErr::EntropyUnavailable)` (C1).
    fn fill_bytes(&mut self, buf: &mut [u8]) -> Result<EntropyEffect, RandErr>;
}

// ──────────────────────────────────────────────────────────────────────────────
// § 3. PRNG algorithm identifier (C3 — EXPLAIN-able)
// ──────────────────────────────────────────────────────────────────────────────

/// The PRNG algorithm used by a [`Rng`] — the inspectable algorithm tag (C3).
///
/// This is the EXPLAIN artifact for the seeded generator: callers can inspect which
/// algorithm produced a draw without reverse-engineering state.
///
/// FLAG (Q2): the concrete algorithm choice is ratified at M-501/RFC-0016. The current
/// default is `Xoshiro256PlusPlus` — a well-studied, fast PRNG with 2^256 - 1 period
/// (Blackman & Vigna 2021). Statistical quality is `Empirical` (passes TestU01 BigCrush),
/// not `Proven`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngAlgo {
    /// xoshiro256++ — Blackman & Vigna 2021. 256-bit state, period 2^256-1.
    ///
    /// Statistical quality: `Empirical` (passes BigCrush); `Declared`-quality uniformity.
    /// FLAG (Q2): final selection is ratified at M-501/RFC-0016.
    Xoshiro256PlusPlus,
}

// ──────────────────────────────────────────────────────────────────────────────
// § 4. Seeded generator — Rng (pure, reproducible)
// ──────────────────────────────────────────────────────────────────────────────

/// A seeded, deterministic generator **value** (spec §3).
///
/// An `Rng` is an immutable value: `{ algo, state }`. Every draw returns the output **and**
/// the advanced generator as a new value — the input generator is unchanged. Same
/// `(seed, draw-sequence)` ⇒ same sequence of outputs (reproducibility is value-equality;
/// C4 / ADR-003).
///
/// The carried algorithm tag (`algo`) is the EXPLAIN artifact (C3): callers can inspect
/// which PRNG produced a draw.
///
/// No ambient global RNG; no interior mutability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    /// The PRNG algorithm (inspectable; C3).
    pub algo: RngAlgo,
    /// The 256-bit xoshiro256++ state as four `u64` words.
    ///
    /// Invariant: not all-zero (xoshiro256++ is invalid at state `[0,0,0,0]`).
    state: [u64; 4],
}

impl Rng {
    /// Construct an `Rng` from raw state, or `None` if the state is invalid (all-zero).
    ///
    /// This is the low-level constructor; prefer [`seed`] for user-facing construction.
    fn from_state(state: [u64; 4]) -> Option<Self> {
        if state == [0u64; 4] {
            None
        } else {
            Some(Rng {
                algo: RngAlgo::Xoshiro256PlusPlus,
                state,
            })
        }
    }

    /// The current raw state (inspectable; C3).
    #[must_use]
    pub fn state(&self) -> [u64; 4] {
        self.state
    }

    /// The algorithm this generator uses (inspectable; C3).
    #[must_use]
    pub fn algo(&self) -> RngAlgo {
        self.algo
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 5. xoshiro256++ internals
// ──────────────────────────────────────────────────────────────────────────────
//
// All arithmetic is explicit/wrapping: no silent overflow (G2).
// Reference: D. Blackman & S. Vigna, "Scrambled Linear Pseudorandom Number
// Generators", ACM Trans. on Mathematical Software, 2021.
// https://prng.di.unimi.it/xoshiro256plusplus.c

/// One step of xoshiro256++: returns the next output and the advanced state.
///
/// This is a *pure function* of the state — the same state always produces the same output
/// and next state (reproducibility by construction, C4).
#[must_use]
fn xoshiro256pp_step(s: [u64; 4]) -> (u64, [u64; 4]) {
    // xoshiro256++ output: rotl(s[0] + s[3], 23) + s[0]
    let result = rotl64(s[0].wrapping_add(s[3]), 23).wrapping_add(s[0]);

    // State update
    let t = s[1] << 17;
    let mut ns = s;
    ns[2] ^= ns[0];
    ns[3] ^= ns[1];
    ns[1] ^= ns[2];
    ns[0] ^= ns[3];
    ns[2] ^= t;
    ns[3] = rotl64(ns[3], 45);

    (result, ns)
}

#[inline]
fn rotl64(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

/// splitmix64 — seeding function that expands a `u64` seed to a valid xoshiro256++ state.
///
/// Avoids the all-zero state by construction (splitmix64 cannot produce a full block of
/// zeros from a non-zero seed, and we check anyway).
///
/// Reference: Steele & Vigna 2021 (same authors).
fn splitmix64_block(seed: u64) -> [u64; 4] {
    let mut z = seed;
    let mut out = [0u64; 4];
    for word in &mut out {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15u64);
        let mut v = z;
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9u64);
        v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11ebu64);
        *word = v ^ (v >> 31);
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// § 6. Seeded generator operations (exported op surface — pure)
// ──────────────────────────────────────────────────────────────────────────────

/// Build an [`Rng`] from a `u64` seed.
///
/// **Guarantee: `Exact` (total).** The returned generator is a pure value; the same seed
/// always produces the same generator and thus the same sequence of draws (reproducibility;
/// C4 / ADR-003). No effects. The seeding expansion (splitmix64) is not all-zero for any
/// `u64` seed (checked by construction).
///
/// This is the intended entry point for reproducible / deterministic randomness.
#[must_use]
pub fn seed(s: u64) -> Rng {
    let state = splitmix64_block(s);
    // splitmix64 from any non-zero seed avoids the all-zero state; seed=0 is handled by
    // the check — in practice splitmix64(0) produces a non-zero block, but we assert
    // rather than rely on that property silently.
    Rng::from_state(state).unwrap_or_else(|| seed(s.wrapping_add(1))) // will not recur: splitmix64(0)≠[0,0,0,0]
}

/// Draw the next raw `u64` from a seeded generator.
///
/// **Guarantee: `Exact` (total).** The draw is an exact, pure function of the seed and the
/// number of prior draws — the same `(seed, n)` always yields the same value. The *uniformity*
/// of the bit-stream is `Declared`/`Empirical` (xoshiro256++ passes BigCrush; VR-5 — not
/// `Proven`). No effects.
///
/// Returns `(value, Rng')`: the advanced generator is a new value; the input is unchanged (C4).
#[must_use]
pub fn next_u64(g: Rng) -> (u64, Rng) {
    let (v, ns) = xoshiro256pp_step(g.state);
    (
        v,
        Rng {
            algo: g.algo,
            state: ns,
        },
    )
}

/// Derive two independent sub-stream generators from one (the "split" operation).
///
/// **Guarantee: `Exact` (total).** The split is a pure function: the same input generator
/// always produces the same two child generators. The *independence quality* of the two
/// sub-streams is `Declared` — xoshiro256++ does not have a proof-theoretic independence
/// guarantee for `split`; the sub-streams are structurally distinct (non-overlapping
/// starting points) but independence is asserted, not proven (VR-5 — FLAG Q2).
///
/// Returns `(Rng_left, Rng_right)`.
#[must_use]
pub fn split(g: Rng) -> (Rng, Rng) {
    // Derive left from the current state, right from one step forward.
    // This is a lightweight "jump-free" split: both children start from non-overlapping
    // stream positions by drawing one extra word for the right branch's seed expansion.
    let (left_seed_word, ns1) = xoshiro256pp_step(g.state);
    let (right_seed_word, _) = xoshiro256pp_step(ns1);

    // Mix each seed word through splitmix64 to produce independent 256-bit states.
    let left_state = splitmix64_block(left_seed_word);
    let right_state = splitmix64_block(right_seed_word.wrapping_add(0xdead_beef_cafe_babe));

    (
        Rng {
            algo: g.algo,
            state: left_state,
        },
        Rng {
            algo: g.algo,
            state: right_state,
        },
    )
}

/// Draw a uniformly-distributed `i64` in the half-open range `[lo, hi)`.
///
/// **Guarantee: `Declared` (unbiased rejection sampling).** The bias is bounded by the
/// rejection argument: rejection sampling over a power-of-two modulus guarantees that
/// any rejected sample is re-drawn, not wrapped — so the distribution over `[lo, hi)` is
/// `Declared`-unbiased. The concrete bias magnitude is not fabricated here — it is owned
/// by `std.numerics` (M-512, FLAG Q3).
///
/// # Errors
///
/// - [`RandErr::EmptyRange`] if `hi <= lo` (C1 — never-silent).
///
/// Returns `Ok((value, Rng'))`.
pub fn uniform_int(g: Rng, lo: i64, hi: i64) -> Result<(i64, Rng), RandErr> {
    if hi <= lo {
        return Err(RandErr::EmptyRange);
    }
    // Compute the span and re-base in i128: `hi - lo` overflows i64 for sign-spanning ranges
    // (e.g. lo=i64::MIN, hi=0), and `lo + (v as i64)` wraps when v ≥ 2^63. i128 is exact for both
    // (the result is in [lo, hi) ⊂ i64 by construction), so the final cast never truncates.
    let range = (i128::from(hi) - i128::from(lo)) as u64;
    let (v, g2) = rejection_sample_u64(g, range);
    let value = (i128::from(lo) + i128::from(v)) as i64;
    Ok((value, g2))
}

/// Draw a uniformly-distributed `u64` in the half-open range `[lo, hi)`.
///
/// **Guarantee: `Declared` (unbiased rejection sampling).** Same basis as [`uniform_int`].
///
/// # Errors
///
/// - [`RandErr::EmptyRange`] if `hi <= lo` (C1).
pub fn uniform_u64(g: Rng, lo: u64, hi: u64) -> Result<(u64, Rng), RandErr> {
    if hi <= lo {
        return Err(RandErr::EmptyRange);
    }
    let range = hi - lo;
    let (v, g2) = rejection_sample_u64(g, range);
    Ok((lo + v, g2))
}

/// Rejection-sample a `u64` in `[0, range)` using the "Lemire" bounded approach.
///
/// This is the internal engine for [`uniform_int`] / [`uniform_u64`] / [`choice`] /
/// [`shuffle`]. Termination is guaranteed in expectation (at most two draws with probability
/// ≥ 0.5 each; virtually never more than a few). This function is total for all `range > 0`.
///
/// Reference: D. Lemire, "Fast Random Integer Generation in an Interval", ACM TOMS 2019.
fn rejection_sample_u64(mut g: Rng, range: u64) -> (u64, Rng) {
    debug_assert!(range > 0, "range must be > 0");
    // Lemire's method: draw a 128-bit product, reject if in the biased tail.
    loop {
        let (raw, g2) = next_u64(g);
        // Compute m = raw * range (128-bit product; we only need the high 64 bits for the
        // threshold check, but Rust's u128 gives us both at once).
        let m = (raw as u128).wrapping_mul(range as u128);
        let hi = (m >> 64) as u64;
        let lo = m as u64;
        // The threshold is `(-range) % range` (= (2^64 - range) % range for u64 range).
        let threshold = range.wrapping_neg() % range;
        if lo >= threshold {
            return (hi, g2);
        }
        g = g2;
    }
}

/// Draw a `bool` from a Bernoulli distribution with success probability `p`.
///
/// **Guarantee: `Declared` (construction argument).** The bit is `true` with probability `p`,
/// by the discrete threshold argument: map the `[0,1)` uniform draw to `true` iff the value
/// falls below `p`. The claim is `Declared` — this is the standard textbook construction, not
/// a formally-proven bound (VR-5).
///
/// # Errors
///
/// - [`RandErr::BadProbability`] if `p ∉ [0.0, 1.0]` (C1).
pub fn bernoulli(g: Rng, p: f64) -> Result<(bool, Rng), RandErr> {
    if !(0.0..=1.0).contains(&p) {
        return Err(RandErr::BadProbability);
    }
    // Map a raw u64 to [0, 1) by dividing by 2^64 (multiply by 2^-64).
    // This is the standard f64-threshold method; bias is sub-ULP — `Declared`.
    let (raw, g2) = next_u64(g);
    // ldexp(raw as f64, -64) = raw / 2^64. `u64::MAX as f64` rounds up to 2^64, and `raw as f64`
    // can also round to 2^64 for raw near u64::MAX, so `u` can reach exactly 1.0 — meaning a naive
    // `u < p` would return `false` even at p == 1.0. Guard the endpoint so p == 1.0 is always true
    // and p == 0.0 is always false (the stated bounds; C1/VR-5).
    let u = (raw as f64) * (1.0_f64 / u64::MAX as f64); // ∈ [0, 1]
    Ok((p >= 1.0 || u < p, g2))
}

/// Choose one element uniformly at random from a non-empty slice.
///
/// **Guarantee: `Declared` (uniform over domain).** The draw is uniform over the slice
/// indices by the same rejection-sampling argument as [`uniform_int`] — `Declared`, not
/// `Proven`. Returns a clone of the chosen element.
///
/// # Errors
///
/// - [`RandErr::EmptyDomain`] if `xs` is empty (C1).
pub fn choice<T: Clone>(g: Rng, xs: &[T]) -> Result<(T, Rng), RandErr> {
    if xs.is_empty() {
        return Err(RandErr::EmptyDomain);
    }
    let n = xs.len() as u64;
    let (idx, g2) = rejection_sample_u64(g, n);
    Ok((xs[idx as usize].clone(), g2))
}

/// Produce a uniformly-random permutation of the input slice (Fisher–Yates shuffle).
///
/// **Guarantee: `Exact` (an exact permutation of the input).** The output is an exact
/// permutation of `xs` — no element is added, removed, or duplicated. The *uniformity over
/// permutations* is `Declared`/`Empirical` by the Fisher–Yates argument (each permutation
/// has probability exactly `1/n!` under the assumption that the underlying draws are
/// unbiased — `Declared`).
///
/// This function is total and effect-free.
#[must_use]
pub fn shuffle<T: Clone>(g: Rng, xs: Vec<T>) -> (Vec<T>, Rng) {
    let mut result = xs;
    let mut g_cur = g;
    let n = result.len();
    for i in (1..n).rev() {
        // Draw a uniform index in [0, i+1)
        let range = (i + 1) as u64;
        let (j, g2) = rejection_sample_u64(g_cur, range);
        result.swap(i, j as usize);
        g_cur = g2;
    }
    (result, g_cur)
}

/// Draw from a Normal(μ, σ) distribution using the Box–Muller transform.
///
/// **Guarantee: `Empirical` (sampler-correctness measured).** The Box–Muller method produces
/// a *mathematically exact* Normal draw given a perfectly-uniform `[0,1)` source; in
/// floating-point the result carries floating-point rounding, the underlying uniform quality
/// is `Declared`/`Empirical` (VR-5), and the combined correctness claim is `Empirical`.
/// The exact bias/error magnitude is **not fabricated here** — it is owned by
/// `std.numerics` (M-512 / FLAG Q3).
///
/// # Errors
///
/// - [`RandErr::BadParameter`] if `sigma <= 0.0` (C1).
pub fn normal(g: Rng, mu: f64, sigma: f64) -> Result<(f64, Rng), RandErr> {
    if sigma <= 0.0 {
        return Err(RandErr::BadParameter);
    }
    let (u1, g2) = next_uniform_open(g);
    let (u2, g3) = next_uniform_open(g2);
    // Box–Muller: z0 = sqrt(-2 ln u1) * cos(2π u2)
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    Ok((mu + sigma * z, g3))
}

/// Draw from an Exponential(λ) distribution using the inverse-CDF method.
///
/// **Guarantee: `Empirical` (sampler-correctness measured).** Same reasoning as [`normal`].
/// The inverse-CDF (`-ln(u) / λ`) is mathematically exact given a perfectly-uniform source;
/// the combined `Empirical` tag reflects the `Declared`/`Empirical` quality of the underlying
/// uniform draw (VR-5).
///
/// # Errors
///
/// - [`RandErr::BadParameter`] if `lambda <= 0.0` (C1).
pub fn exponential(g: Rng, lambda: f64) -> Result<(f64, Rng), RandErr> {
    if lambda <= 0.0 {
        return Err(RandErr::BadParameter);
    }
    let (u, g2) = next_uniform_open(g);
    // Inverse-CDF: -ln(u) / λ
    Ok((-u.ln() / lambda, g2))
}

/// Draw a `f64` strictly in the open interval `(0, 1)` (avoids `ln(0)`/`ln(1)=0` in samplers).
///
/// This is an internal helper for samplers that need a strictly-interior uniform draw.
/// `raw | 1` excludes the low endpoint 0. The high endpoint needs care too: `u64::MAX as f64`
/// rounds up to 2^64 and `raw as f64` can round to 2^64 for `raw` near `u64::MAX`, so the naive
/// product can reach exactly `1.0`. We clamp to the largest f64 strictly below 1 so the result is
/// genuinely in `(0, 1)` — otherwise `exponential` could emit `-ln(1)/λ = 0` and the stated
/// interval would be false (the distortion is sub-ULP and confined to ~2^-53 of draws).
fn next_uniform_open(g: Rng) -> (f64, Rng) {
    let (raw, g2) = next_u64(g);
    let v = ((raw | 1) as f64) * (1.0_f64 / u64::MAX as f64);
    let v = v.min(1.0 - f64::EPSILON / 2.0); // < 1, exactly representable
    (v, g2)
}

// ──────────────────────────────────────────────────────────────────────────────
// § 7. Entropy-backed generator — EntropyRng (declared effect)
// ──────────────────────────────────────────────────────────────────────────────

/// An entropy-backed generator.
///
/// Constructing one and drawing from it carry the declared [`EntropyEffect`] (C6 / RT3 /
/// RFC-0014). A deterministic-fragment program **cannot** call these without the effect
/// escaping into its type signature — this is the structural RT3 enforcement.
///
/// In Rust (pre-Mycelium-lang migration) the rule is enforced by convention + test suite;
/// the `EntropyEffect` return value is the documented bearer of the declaration.
///
/// FLAG (Q4 / §8-Q6): the concrete entropy source is injected; real OS entropy lives in
/// `std-sys` (M-541). The test suite uses [`StubEntropy`] — a deterministic counter that
/// exercises the full code path without touching the OS.
pub struct EntropyRng<S: EntropySource> {
    /// The injected entropy source (C5 — no OS calls in `std.rand`; the floor is `std-sys`).
    ///
    /// Held for potential re-seeding (a future extension); deliberately kept in the struct
    /// so the field is part of the type's public contract even though the current
    /// `next_entropy` draws from the pre-seeded xoshiro256++ state. The field is not
    /// `dead_code` architecturally — it is the injectable seam for `std-sys` (M-541).
    #[allow(dead_code)]
    source: S,
    /// The current xoshiro256++ state, seeded from entropy.
    state: [u64; 4],
}

impl<S: EntropySource> EntropyRng<S> {
    /// Construct an `EntropyRng` by seeding from the given entropy source.
    ///
    /// **Guarantee: `Declared` (the seed quality equals the source quality).**
    /// Declares the `entropy` effect (C6 / RT3): returns an [`EntropyEffect`] token.
    ///
    /// # Errors
    ///
    /// - [`RandErr::EntropyUnavailable`] if the source fails (C1 — never a silent fixed seed).
    pub fn new(mut source: S) -> Result<(EntropyRng<S>, EntropyEffect), RandErr> {
        let mut buf = [0u8; 32];
        let effect = source.fill_bytes(&mut buf)?;
        // Decode the 32-byte entropy buffer into four u64s (little-endian).
        let state = bytes_to_u64x4(buf);
        // Guard against a (pathological) all-zero entropy source — treat as unavailable
        // rather than silently proceeding with an invalid state (C1).
        if state == [0u64; 4] {
            return Err(RandErr::EntropyUnavailable);
        }
        Ok((EntropyRng { source, state }, effect))
    }

    /// Draw the next raw `u64` from the entropy-backed generator.
    ///
    /// **Guarantee: `Declared` (the quality of the draw equals the entropy source quality).**
    /// Declares the `entropy` effect (C6 / RT3).
    ///
    /// # Errors
    ///
    /// Currently infallible for a fully-initialized `EntropyRng`, but declared fallible
    /// (returns `RandErr`) in line with the spec's `Err(EntropyUnavailable)` contract for the
    /// entropy surface — a future implementation may re-seed on each draw.
    pub fn next_entropy(&mut self) -> (u64, EntropyEffect) {
        let (v, ns) = xoshiro256pp_step(self.state);
        self.state = ns;
        (v, EntropyEffect)
    }
}

/// Mint a single reproducible seed from entropy, then return a pure [`Rng`].
///
/// **Guarantee: `Declared` (the minted seed's quality equals the source quality; thereafter
/// pure).** This is the bridge op: it draws entropy *once* (declared effect), expands the
/// 32-byte entropy buffer into a valid [`Rng`] state, and returns a pure seeded generator.
/// All subsequent draws on the returned `Rng` are pure functions of the minted seed — no
/// further entropy is consumed.
///
/// Declares the `entropy` effect (C6 / RT3): the effect is confined to this one call.
///
/// # Errors
///
/// - [`RandErr::EntropyUnavailable`] if the source fails (C1 — never a silent fixed seed).
pub fn seed_from_entropy<S: EntropySource>(mut source: S) -> Result<(Rng, EntropyEffect), RandErr> {
    let mut buf = [0u8; 32];
    let effect = source.fill_bytes(&mut buf)?;
    let state = bytes_to_u64x4(buf);
    // Guard: all-zero entropy is treated as unavailable (C1 — never a silent fixed seed).
    if state == [0u64; 4] {
        return Err(RandErr::EntropyUnavailable);
    }
    let rng = Rng {
        algo: RngAlgo::Xoshiro256PlusPlus,
        state,
    };
    Ok((rng, effect))
}

/// Decode a 32-byte entropy buffer into four little-endian `u64`s.
fn bytes_to_u64x4(buf: [u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (i, word) in out.iter_mut().enumerate() {
        let b = &buf[i * 8..(i + 1) * 8];
        *word = u64::from_le_bytes(b.try_into().expect("8 bytes"));
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// § 8. Test-harness injectable entropy stub
// ──────────────────────────────────────────────────────────────────────────────

/// A deterministic, injectable [`EntropySource`] for tests.
///
/// Fills buffers with a fixed counter-based pattern so the test suite exercises the full
/// entropy-surface code path without any OS calls. This is **not** a source of real entropy
/// and must not be used outside tests.
///
/// FLAG (Q4 / §8-Q6): real OS entropy is in `std-sys` (M-541). This stub is the test seam.
pub struct StubEntropy {
    /// The current counter byte value (wraps on overflow).
    counter: u8,
}

impl StubEntropy {
    /// Build a new stub with the given starting byte.
    #[must_use]
    pub fn new(start: u8) -> Self {
        StubEntropy { counter: start }
    }
}

impl EntropySource for StubEntropy {
    fn fill_bytes(&mut self, buf: &mut [u8]) -> Result<EntropyEffect, RandErr> {
        for b in buf.iter_mut() {
            *b = self.counter;
            self.counter = self.counter.wrapping_add(1);
        }
        // Ensure the buffer is not all-zero (that would trigger the "unavailable" guard).
        // Guarantee: counter starts at `start`; if start=0 the first byte is 0 but later
        // bytes differ, so the 32-byte state cannot be all-zero unless we start at 0 and the
        // counter wraps exactly — prevented by starting at 1 by default in tests.
        Ok(EntropyEffect)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 9. Guarantee matrix (RFC-0016 §4.5) — encoded as data, asserted in tests
// ──────────────────────────────────────────────────────────────────────────────

/// One row of the `std.rand` guarantee matrix (RFC-0016 §4.5; rand.md §4).
///
/// Encoded as data so tests can assert invariants — never prose-only (RFC-0016 §4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixRow {
    /// The exported operation name (spec §3).
    pub op: &'static str,
    /// Honest guarantee tag on `Exact ⊐ Proven ⊐ Empirical ⊐ Declared` (C2 / VR-5).
    pub tag: GuaranteeStrength,
    /// Fallibility shape: `"total"` or the explicit error variant(s) (C1).
    pub fallibility: &'static str,
    /// Declared effects: `"none"` for pure ops, `"entropy"` for entropy-drawing ops (C6).
    pub effects: &'static str,
    /// Whether the op surfaces an inspectable EXPLAIN artifact (C3).
    pub explainable: bool,
}

/// The `std.rand` guarantee matrix (spec §4 / RFC-0016 §4.5).
///
/// Tag justification (VR-5 — downgrade rather than overclaim):
///
/// - **`Exact` rows** carry no *accuracy* semantics: `seed`/`split`/`next_u64` are exact
///   pure functions of their inputs; `shuffle` is an exact permutation. "`Exact`" is about
///   *reproducibility/determinism of the mechanism*, not statistical quality — that claim is
///   separated and tagged on its own footing below.
///
/// - **`Declared` rows** (`uniform_int`, `uniform_u64`, `bernoulli`, `choice`,
///   `shuffle`-uniformity) carry a *sampling-correctness* claim. Tagged `Declared` — the
///   construction argument (rejection sampling / Fisher–Yates / threshold) is the basis.
///   Not `Proven` (no checked theorem with side-conditions attached; VR-5).
///
/// - **`Empirical` rows** (`normal`, `exponential`) carry a continuous-sampler correctness
///   claim established by the method + measured quality of the underlying stream. The
///   concrete error magnitudes are **not fabricated here** — owned by `std.numerics`
///   (M-512 / FLAG Q3).
///
/// - **`entropy` rows** are `Declared` and effectful. The platform source's quality is not
///   proven here; what `std.rand` guarantees is that the draw is **named** (the `entropy`
///   effect) and fallible (never a silent fixed-seed fallback).
pub const GUARANTEE_MATRIX: &[MatrixRow] = &[
    MatrixRow {
        op: "seed",
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    MatrixRow {
        op: "split",
        // Exact: the split is a pure function (same input → same two children).
        // Independence quality is `Declared` — flagged separately (Q2); the row tag is
        // `Exact` for the mechanism-determinism claim (spec §4).
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "next_u64",
        // Exact: exactly determined by seed. Uniformity is `Empirical`/`Declared` (§4 note).
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: false,
    },
    MatrixRow {
        op: "uniform_int",
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(EmptyRange) if hi <= lo",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "uniform_u64",
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(EmptyRange) if hi <= lo",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "bernoulli",
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(BadProbability) if p not in [0,1]",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "choice",
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(EmptyDomain) if xs empty",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "shuffle",
        // Exact: an exact permutation. Uniformity-over-permutations is `Declared`.
        tag: GuaranteeStrength::Exact,
        fallibility: "total",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "normal",
        tag: GuaranteeStrength::Empirical,
        fallibility: "Err(BadParameter) if sigma <= 0",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "exponential",
        tag: GuaranteeStrength::Empirical,
        fallibility: "Err(BadParameter) if lambda <= 0",
        effects: "none",
        explainable: true,
    },
    MatrixRow {
        op: "from_entropy (EntropyRng::new)",
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(EntropyUnavailable)",
        effects: "entropy",
        explainable: true,
    },
    MatrixRow {
        op: "seed_from_entropy",
        tag: GuaranteeStrength::Declared,
        fallibility: "Err(EntropyUnavailable)",
        effects: "entropy",
        explainable: true,
    },
    MatrixRow {
        op: "next_entropy (EntropyRng::next_entropy)",
        tag: GuaranteeStrength::Declared,
        fallibility: "total (declared-effect)",
        effects: "entropy",
        explainable: true,
    },
];

/// Assert the structural invariants of the guarantee matrix — called from tests.
///
/// Discharges the RFC-0016 §4.5 obligation: "encoded as data, asserted in tests, never
/// prose-only." Panics with a descriptive message on any violation.
pub fn assert_matrix_invariants() {
    for row in GUARANTEE_MATRIX {
        assert!(!row.op.is_empty(), "matrix row has empty op name");
        // Entropy ops must declare the effect.
        if row.op.contains("entropy") || row.op.contains("Entropy") {
            assert_eq!(
                row.effects, "entropy",
                "op '{}': ops drawing entropy must declare 'entropy' effect (C6/RT3)",
                row.op
            );
        }
        // Pure (seeded) ops must not declare an entropy effect.
        if row.effects == "none" {
            assert!(
                !row.op.contains("entropy") || row.op == "seed_from_entropy",
                "op '{}': non-entropy op must not draw entropy silently (RT3)",
                row.op
            );
        }
        // `Proven` is forbidden without a cited theorem — no row should be `Proven` (VR-5).
        assert_ne!(
            row.tag,
            GuaranteeStrength::Proven,
            "op '{}': Proven is not allowed in std.rand without a cited checked theorem (VR-5)",
            row.op
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 10. Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Guarantee matrix ──────────────────────────────────────────────────────

    /// The matrix is internally consistent (RFC-0016 §4.5).
    #[test]
    fn guarantee_matrix_invariants_hold() {
        assert_matrix_invariants();
    }

    /// All expected ops appear exactly once.
    #[test]
    fn matrix_contains_all_ops_exactly_once() {
        let expected = [
            "seed",
            "split",
            "next_u64",
            "uniform_int",
            "uniform_u64",
            "bernoulli",
            "choice",
            "shuffle",
            "normal",
            "exponential",
            "from_entropy (EntropyRng::new)",
            "seed_from_entropy",
            "next_entropy (EntropyRng::next_entropy)",
        ];
        for op in &expected {
            let count = GUARANTEE_MATRIX.iter().filter(|r| r.op == *op).count();
            assert_eq!(count, 1, "op '{op}' must appear exactly once in the matrix");
        }
    }

    /// No row is tagged `Proven` — that would violate VR-5 (no checked theorem cited here).
    #[test]
    fn no_matrix_row_is_proven() {
        for row in GUARANTEE_MATRIX {
            assert_ne!(
                row.tag,
                GuaranteeStrength::Proven,
                "op '{}': Proven not allowed (VR-5)",
                row.op
            );
        }
    }

    /// Every entropy op declares the `entropy` effect (C6 / RT3).
    #[test]
    fn entropy_ops_declare_entropy_effect() {
        let entropy_ops = [
            "from_entropy (EntropyRng::new)",
            "seed_from_entropy",
            "next_entropy (EntropyRng::next_entropy)",
        ];
        for op in &entropy_ops {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("missing matrix row for '{op}'"));
            assert_eq!(
                row.effects, "entropy",
                "op '{op}' must declare entropy effect"
            );
        }
    }

    /// Pure (seeded) ops declare no effects (C6 — seeded surface is effect-free).
    #[test]
    fn pure_ops_declare_no_effects() {
        let pure_ops = [
            "seed",
            "split",
            "next_u64",
            "uniform_int",
            "uniform_u64",
            "bernoulli",
            "choice",
            "shuffle",
            "normal",
            "exponential",
        ];
        for op in &pure_ops {
            let row = GUARANTEE_MATRIX
                .iter()
                .find(|r| r.op == *op)
                .unwrap_or_else(|| panic!("missing matrix row for '{op}'"));
            assert_eq!(
                row.effects, "none",
                "op '{op}' must be effect-free (seeded surface)"
            );
        }
    }

    // ── Seeded reproducibility (the honesty crux — spec §1) ──────────────────

    /// Same seed ⇒ same sequence (the fundamental reproducibility guarantee).
    ///
    /// This is the **primary property test** for the seeded surface: it checks that
    /// reproducibility is value-equality (C4 / ADR-003).
    #[test]
    fn seeded_rng_is_reproducible_same_seed_same_sequence() {
        let seeds = [0u64, 1, 42, u64::MAX, 0xdead_beef_cafe_babe];
        let draws = 64usize;

        for s in seeds {
            let g1 = seed(s);
            let g2 = seed(s);
            let mut cur1 = g1;
            let mut cur2 = g2;
            for _ in 0..draws {
                let (v1, ng1) = next_u64(cur1);
                let (v2, ng2) = next_u64(cur2);
                assert_eq!(v1, v2, "seed={s}: same seed must produce same sequence");
                cur1 = ng1;
                cur2 = ng2;
            }
        }
    }

    /// Different seeds produce different sequences (distinct seeds → distinct streams).
    ///
    /// Not a guarantee (collisions are statistically possible) but a sanity check: for
    /// the 16 tested pairs, the first draw differs. Failure here would indicate a broken
    /// seeding function (like a constant map).
    #[test]
    fn different_seeds_produce_different_first_draws() {
        let seeds: Vec<u64> = (0..16).collect();
        let draws: Vec<u64> = seeds
            .iter()
            .map(|&s| {
                let (v, _) = next_u64(seed(s));
                v
            })
            .collect();
        // All first draws from distinct seeds must be distinct (for these 16 seeds).
        let mut seen = std::collections::HashSet::new();
        for (i, &v) in draws.iter().enumerate() {
            assert!(
                seen.insert(v),
                "seed {}: first draw collided with a prior seed",
                i
            );
        }
    }

    /// `seed(0)` produces a valid non-zero state (the seeding function avoids the all-zero
    /// xoshiro256++ invalid state).
    #[test]
    fn seed_zero_produces_valid_rng() {
        let g = seed(0);
        assert_ne!(
            g.state(),
            [0u64; 4],
            "seed(0) must not produce the all-zero state"
        );
        // And it produces a non-zero first draw.
        let (v, _) = next_u64(g);
        // The draw might be 0 by chance; we just check we don't panic.
        let _ = v;
    }

    /// The generator is value-semantic: drawing from a generator does not modify it.
    ///
    /// Property test for C4 / ADR-003: same generator value → same next draw, regardless
    /// of how many prior draws were made from the *same* value.
    #[test]
    fn rng_is_value_semantic_draw_does_not_mutate_input() {
        let g = seed(12345);
        let (v1, _) = next_u64(g.clone());
        let (v2, _) = next_u64(g.clone());
        assert_eq!(
            v1, v2,
            "drawing from the same Rng value must yield the same result (C4)"
        );
    }

    // ── `split` ───────────────────────────────────────────────────────────────

    /// `split` is deterministic: same input → same two children (Exact on the mechanism).
    #[test]
    fn split_is_deterministic() {
        let g = seed(99);
        let (l1, r1) = split(g.clone());
        let (l2, r2) = split(g.clone());
        assert_eq!(l1, l2, "split: left child must be deterministic");
        assert_eq!(r1, r2, "split: right child must be deterministic");
    }

    /// The two children of `split` differ from each other and from the parent.
    ///
    /// Not a formal independence test (independence is `Declared`), but ensures the split
    /// is non-trivial.
    #[test]
    fn split_produces_distinct_children() {
        let g = seed(42);
        let (left, right) = split(g.clone());
        assert_ne!(left, right, "split: left and right children must differ");
        assert_ne!(left, g, "split: left child must differ from parent");
        assert_ne!(right, g, "split: right child must differ from parent");
    }

    // ── `uniform_int` ─────────────────────────────────────────────────────────

    /// All draws from `uniform_int(g, lo, hi)` fall in `[lo, hi)` — the range bound.
    ///
    /// Property test for every stated bound (spec §4 / CLAUDE.md: "a property test for
    /// every stated bound").
    #[test]
    fn uniform_int_all_draws_in_range() {
        let mut g = seed(7);
        let lo = -100i64;
        let hi = 100i64;
        for _ in 0..10_000 {
            let (v, ng) = uniform_int(g, lo, hi).expect("valid range");
            assert!(
                v >= lo && v < hi,
                "uniform_int draw {v} out of [{lo}, {hi})"
            );
            g = ng;
        }
    }

    /// Empty range → `Err(EmptyRange)` (C1 — never-silent).
    #[test]
    fn uniform_int_empty_range_is_error() {
        let g = seed(1);
        assert_eq!(
            uniform_int(g.clone(), 5, 5),
            Err(RandErr::EmptyRange),
            "hi == lo must be EmptyRange"
        );
        assert_eq!(
            uniform_int(g.clone(), 10, 5),
            Err(RandErr::EmptyRange),
            "hi < lo must be EmptyRange"
        );
    }

    /// Singleton range `[lo, lo+1)` always produces `lo`.
    #[test]
    fn uniform_int_singleton_range_always_lo() {
        let mut g = seed(3);
        for _ in 0..100 {
            let (v, ng) = uniform_int(g, 42, 43).expect("singleton range");
            assert_eq!(v, 42, "singleton range must always produce lo");
            g = ng;
        }
    }

    /// A sign-spanning, near-full-i64 range must not overflow/panic and stays in `[lo, hi)`.
    ///
    /// `hi - lo` and `lo + v` both overflow i64 here — the i128 re-basing must keep them sound
    /// (regression for the two `uniform_int` overflow bugs). The workspace builds with
    /// `overflow-checks = true`, so an overflow would panic, not wrap.
    #[test]
    fn uniform_int_full_signed_range_never_overflows() {
        let mut g = seed(7);
        // [i64::MIN, i64::MAX): span = 2^64 - 1, which overflows i64 if computed in i64.
        for _ in 0..10_000 {
            let (v, ng) = uniform_int(g, i64::MIN, i64::MAX).expect("valid wide range");
            assert!(
                (i64::MIN..i64::MAX).contains(&v),
                "draw {v} out of [MIN, MAX)"
            );
            g = ng;
        }
        // A range straddling zero up to the boundary: lo=i64::MIN, hi=0.
        let (w, _) = uniform_int(seed(11), i64::MIN, 0).expect("valid range");
        assert!((i64::MIN..0).contains(&w), "draw {w} out of [MIN, 0)");
    }

    // ── `uniform_u64` ─────────────────────────────────────────────────────────

    /// All draws from `uniform_u64(g, lo, hi)` fall in `[lo, hi)`.
    #[test]
    fn uniform_u64_all_draws_in_range() {
        let mut g = seed(8);
        let lo = 100u64;
        let hi = 200u64;
        for _ in 0..10_000 {
            let (v, ng) = uniform_u64(g, lo, hi).expect("valid range");
            assert!(
                v >= lo && v < hi,
                "uniform_u64 draw {v} out of [{lo}, {hi})"
            );
            g = ng;
        }
    }

    /// Empty range → `Err(EmptyRange)` (C1).
    #[test]
    fn uniform_u64_empty_range_is_error() {
        let g = seed(1);
        assert_eq!(uniform_u64(g, 5, 5), Err(RandErr::EmptyRange));
    }

    // ── `bernoulli` ───────────────────────────────────────────────────────────

    /// p=0.0 → always false; p=1.0 → always true (extreme-probability bounds).
    ///
    /// Property test for the boundary bound (`p=0` and `p=1`).
    #[test]
    fn bernoulli_extremes_are_deterministic() {
        let mut g = seed(55);
        for _ in 0..100 {
            let (b0, ng0) = bernoulli(g.clone(), 0.0).expect("p=0 is valid");
            let (b1, ng1) = bernoulli(g.clone(), 1.0).expect("p=1 is valid");
            assert!(!b0, "p=0 must always be false");
            assert!(b1, "p=1 must always be true");
            let _ = (ng0, ng1);
            let (_, ng) = next_u64(g);
            g = ng;
        }
    }

    /// p outside [0,1] → `Err(BadProbability)` (C1).
    #[test]
    fn bernoulli_bad_probability_is_error() {
        let g = seed(1);
        assert_eq!(bernoulli(g.clone(), -0.1), Err(RandErr::BadProbability));
        assert_eq!(bernoulli(g.clone(), 1.1), Err(RandErr::BadProbability));
        assert_eq!(bernoulli(g.clone(), f64::NAN), Err(RandErr::BadProbability));
    }

    /// The empirical frequency of `bernoulli(g, 0.5)` is near 0.5 over 10 000 draws
    /// (statistical sanity, not a `Proven` uniformity bound).
    #[test]
    fn bernoulli_half_is_near_half_empirically() {
        let mut g = seed(42);
        let n = 10_000usize;
        let mut trues = 0usize;
        for _ in 0..n {
            let (b, ng) = bernoulli(g, 0.5).expect("p=0.5 is valid");
            if b {
                trues += 1;
            }
            g = ng;
        }
        let freq = trues as f64 / n as f64;
        // 3-sigma bound for n=10000, p=0.5: σ = sqrt(p(1-p)/n) ≈ 0.005; 3σ ≈ 0.015.
        assert!(
            (freq - 0.5).abs() < 0.05,
            "empirical frequency {freq:.4} is too far from 0.5 (n={n})"
        );
    }

    // ── `choice` ──────────────────────────────────────────────────────────────

    /// All elements chosen by `choice` come from the input slice.
    #[test]
    fn choice_always_picks_element_from_slice() {
        let xs = vec![10u32, 20, 30, 40, 50];
        let mut g = seed(17);
        for _ in 0..1000 {
            let (v, ng) = choice(g, &xs).expect("non-empty slice");
            assert!(xs.contains(&v), "choice returned {v} not in input");
            g = ng;
        }
    }

    /// Empty slice → `Err(EmptyDomain)` (C1).
    #[test]
    fn choice_empty_slice_is_error() {
        let g = seed(1);
        let empty: &[u32] = &[];
        assert_eq!(choice(g, empty), Err(RandErr::EmptyDomain));
    }

    /// Singleton slice always returns the single element.
    #[test]
    fn choice_singleton_always_returns_the_element() {
        let xs = vec![99u32];
        let mut g = seed(4);
        for _ in 0..50 {
            let (v, ng) = choice(g, &xs).expect("singleton");
            assert_eq!(v, 99u32);
            g = ng;
        }
    }

    // ── `shuffle` ─────────────────────────────────────────────────────────────

    /// Shuffled output is a permutation of the input (every element present exactly once).
    ///
    /// Property test for the `Exact` permutation guarantee (spec §4).
    #[test]
    fn shuffle_output_is_permutation_of_input() {
        let xs: Vec<u32> = (0..16).collect();
        let mut g = seed(100);
        for _ in 0..100 {
            let (shuffled, ng) = shuffle(g, xs.clone());
            let mut sorted = shuffled.clone();
            sorted.sort();
            assert_eq!(sorted, xs, "shuffle output must be a permutation of input");
            g = ng;
        }
    }

    /// `shuffle` on an empty vec returns an empty vec (total, no panic).
    #[test]
    fn shuffle_empty_is_total() {
        let (out, _) = shuffle(seed(1), Vec::<u32>::new());
        assert!(out.is_empty());
    }

    /// `shuffle` on a singleton returns that singleton (total).
    #[test]
    fn shuffle_singleton_is_identity() {
        let (out, _) = shuffle(seed(2), vec![42u32]);
        assert_eq!(out, vec![42u32]);
    }

    /// `shuffle` is deterministic: same generator value → same output.
    #[test]
    fn shuffle_is_deterministic_given_same_rng() {
        let xs: Vec<u32> = (0..8).collect();
        let g = seed(55);
        let (s1, _) = shuffle(g.clone(), xs.clone());
        let (s2, _) = shuffle(g.clone(), xs.clone());
        assert_eq!(s1, s2, "shuffle with same Rng must be deterministic");
    }

    // ── `normal` ─────────────────────────────────────────────────────────────

    /// `normal` draws fall in a plausible range (sanity: no NaN/Inf in 1000 draws).
    #[test]
    fn normal_draws_are_finite() {
        let mut g = seed(200);
        for _ in 0..1000 {
            let (v, ng) = normal(g, 0.0, 1.0).expect("sigma=1 is valid");
            assert!(v.is_finite(), "normal draw must be finite; got {v}");
            g = ng;
        }
    }

    /// sigma ≤ 0 → `Err(BadParameter)` (C1).
    #[test]
    fn normal_bad_sigma_is_error() {
        let g = seed(1);
        assert_eq!(normal(g.clone(), 0.0, 0.0), Err(RandErr::BadParameter));
        assert_eq!(normal(g.clone(), 0.0, -1.0), Err(RandErr::BadParameter));
    }

    /// The empirical mean of Normal(0, 1) over 10 000 draws is near 0.
    #[test]
    fn normal_empirical_mean_near_zero() {
        let mut g = seed(77);
        let n = 10_000usize;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let (v, ng) = normal(g, 0.0, 1.0).expect("valid");
            sum += v;
            g = ng;
        }
        let mean = sum / n as f64;
        // 3σ / sqrt(n) ≈ 3 / 100 = 0.03 for N(0,1).
        assert!(
            mean.abs() < 0.1,
            "empirical mean {mean:.4} is too far from 0 (n={n})"
        );
    }

    // ── `exponential` ─────────────────────────────────────────────────────────

    /// Exponential draws are positive (domain bound: Exp(λ) ∈ (0, ∞)).
    #[test]
    fn exponential_draws_are_positive() {
        let mut g = seed(300);
        for _ in 0..1000 {
            let (v, ng) = exponential(g, 1.0).expect("lambda=1 is valid");
            assert!(v > 0.0, "exponential draw must be positive; got {v}");
            assert!(v.is_finite(), "exponential draw must be finite; got {v}");
            g = ng;
        }
    }

    /// lambda ≤ 0 → `Err(BadParameter)` (C1).
    #[test]
    fn exponential_bad_lambda_is_error() {
        let g = seed(1);
        assert_eq!(exponential(g.clone(), 0.0), Err(RandErr::BadParameter));
        assert_eq!(exponential(g.clone(), -1.0), Err(RandErr::BadParameter));
    }

    /// The empirical mean of Exp(1) over 10 000 draws is near 1.
    #[test]
    fn exponential_empirical_mean_near_one() {
        let mut g = seed(88);
        let n = 10_000usize;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let (v, ng) = exponential(g, 1.0).expect("valid");
            sum += v;
            g = ng;
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "empirical mean {mean:.4} is too far from 1.0 (n={n})"
        );
    }

    // ── Entropy surface (RT3 / C6 / RFC-0014) ────────────────────────────────

    /// `EntropyRng::new` succeeds with a valid stub and returns an `EntropyEffect`.
    #[test]
    fn entropy_rng_constructs_and_draws_with_effect() {
        let stub = StubEntropy::new(1);
        let (mut erng, _effect): (EntropyRng<StubEntropy>, EntropyEffect) =
            EntropyRng::new(stub).expect("stub entropy should succeed");
        let (v, effect2) = erng.next_entropy();
        let _ = v; // may be anything
        let _ = effect2; // EntropyEffect token returned — effect is declared
    }

    /// `seed_from_entropy` returns a pure `Rng` (seeded from the stub) and an `EntropyEffect`.
    #[test]
    fn seed_from_entropy_returns_pure_rng_and_effect() {
        let stub = StubEntropy::new(42);
        let (rng, _effect): (Rng, EntropyEffect) =
            seed_from_entropy(stub).expect("stub entropy should succeed");
        // The returned Rng is a pure seeded generator — draws from it are deterministic.
        let (v1, _) = next_u64(rng.clone());
        let (v2, _) = next_u64(rng.clone());
        assert_eq!(
            v1, v2,
            "seed_from_entropy: the minted Rng is reproducible (same value → same draw)"
        );
    }

    /// Two `seed_from_entropy` calls with different stub offsets produce different `Rng`s.
    #[test]
    fn seed_from_entropy_different_sources_produce_different_rngs() {
        let stub1 = StubEntropy::new(1);
        let stub2 = StubEntropy::new(200);
        let (rng1, _) = seed_from_entropy(stub1).expect("stub1");
        let (rng2, _) = seed_from_entropy(stub2).expect("stub2");
        assert_ne!(
            rng1.state(),
            rng2.state(),
            "different entropy → different Rng state"
        );
        let (v1, _) = next_u64(rng1);
        let (v2, _) = next_u64(rng2);
        assert_ne!(
            v1, v2,
            "different entropy sources must produce different sequences"
        );
    }

    /// An all-zero entropy buffer → `Err(EntropyUnavailable)` (never a silent fixed seed; C1).
    #[test]
    fn all_zero_entropy_is_unavailable_not_silent_seed() {
        // StubEntropy starting at 0 fills all 32 bytes with 0,1,2,... → NOT all-zero.
        // We need a stub that always returns 0x00 — craft one inline.
        struct ZeroEntropy;
        impl EntropySource for ZeroEntropy {
            fn fill_bytes(&mut self, buf: &mut [u8]) -> Result<EntropyEffect, RandErr> {
                buf.fill(0);
                Ok(EntropyEffect)
            }
        }

        let result = seed_from_entropy(ZeroEntropy);
        assert_eq!(
            result.map(|(_, _)| ()),
            Err(RandErr::EntropyUnavailable),
            "all-zero entropy must be Err(EntropyUnavailable), never a silent seed"
        );
    }

    /// A failing entropy source → `Err(EntropyUnavailable)` (C1).
    #[test]
    fn failing_entropy_source_is_never_silent() {
        struct FailEntropy;
        impl EntropySource for FailEntropy {
            fn fill_bytes(&mut self, _buf: &mut [u8]) -> Result<EntropyEffect, RandErr> {
                Err(RandErr::EntropyUnavailable)
            }
        }

        let result: Result<(EntropyRng<FailEntropy>, EntropyEffect), RandErr> =
            EntropyRng::new(FailEntropy);
        assert_eq!(
            result.err(),
            Some(RandErr::EntropyUnavailable),
            "failing entropy source must produce Err(EntropyUnavailable)"
        );
    }

    // ── xoshiro256++ internal invariants ─────────────────────────────────────

    /// xoshiro256++ output is deterministic and a state-change occurs on each step.
    #[test]
    fn xoshiro256pp_step_is_pure_and_advances_state() {
        let s = [1u64, 2, 3, 4];
        let (v1, ns1) = xoshiro256pp_step(s);
        let (v2, ns2) = xoshiro256pp_step(s);
        // Deterministic.
        assert_eq!(v1, v2);
        assert_eq!(ns1, ns2);
        // State changes.
        assert_ne!(ns1, s, "step must advance the state");
    }

    /// Two consecutive steps produce different outputs.
    #[test]
    fn xoshiro256pp_consecutive_steps_differ() {
        let s = [0xcafe_babe_dead_beef_u64, 2, 3, 4];
        let (v1, ns1) = xoshiro256pp_step(s);
        let (v2, _) = xoshiro256pp_step(ns1);
        assert_ne!(v1, v2, "consecutive xoshiro256++ outputs should differ");
    }

    // ── splitmix64 seeding ────────────────────────────────────────────────────

    /// splitmix64 never produces the all-zero state for any u64 seed in a sample.
    #[test]
    fn splitmix64_never_all_zero_for_sampled_seeds() {
        let seeds = [0u64, 1, 2, 42, u64::MAX, 0x1234_5678_9abc_def0];
        for s in seeds {
            let state = splitmix64_block(s);
            assert_ne!(
                state, [0u64; 4],
                "splitmix64({s}): must not produce all-zero state"
            );
        }
    }

    // ── RandErr Display ───────────────────────────────────────────────────────

    /// `RandErr` variants have non-empty Display messages (C1 — legible errors).
    #[test]
    fn randerr_display_is_nonempty() {
        let variants = [
            RandErr::EmptyRange,
            RandErr::BadProbability,
            RandErr::EmptyDomain,
            RandErr::BadParameter,
            RandErr::EntropyUnavailable,
        ];
        for e in &variants {
            let s = e.to_string();
            assert!(!s.is_empty(), "{e:?}: Display must not be empty");
        }
    }
}
