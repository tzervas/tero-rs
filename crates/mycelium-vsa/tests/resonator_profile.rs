//! The trial-validated **resonator profile** gate + the staged **capacity sweep** (RFC-0009 §5.2 /
//! §9 Q4 / §11; M-350). The gate (`mapi_resonator_profile_holds_over_declared_trials`) is the single
//! test that *earns* the `Empirical` δ for [`MAPI_RESONATOR_PROFILE`]: it runs **exactly**
//! `profile.trials` Monte-Carlo trials at the profile's worst covered point (max factors, max
//! codebook, min dim), scoring **exact-tuple recovery against ground truth** (not self-reported
//! convergence — §8.1 P5), and asserts the measured failure rate stays at or below `profile.delta`.
//! The `#[ignore]`d `resonator_capacity_sweep` is the manual instrument that *maps the operational
//! edge* across `{F, k, d}` (the data behind the chosen envelope). No `rand` dependency (deterministic
//! LCG). The const δ is the conservative ceiling these runs confirm, never asserted ahead of them.
//!
//! ## proptest migration (M-654 / ADR-021 Gate A3)
//!
//! `mapi_resonator_profile_holds_over_declared_trials` is migrated from a hand-rolled
//! `for trial in 0..p.trials { Lcg::new(salt ^ trial) … }` loop to proptest.  proptest generates a
//! `Vec<u64>` of exactly `p.trials` seeds; the same `recovery_fails` helper is reused.  The
//! statistical assertion (rate ≤ declared δ) is preserved unchanged.  `measure_rate` is retained
//! for the `#[ignore]`d sweep/ablation instruments — they are not migrated as they are manual
//! instruments, not correctness property tests.

use mycelium_vsa::{
    factorize, Cleanup, CleanupMemory, MapI, ResonatorParams, VsaModel, MAPI_RESONATOR_PROFILE,
};
use proptest::prelude::*;

/// The canonical knobs the profile validates and records (kept in sync with the const's `method`).
/// The adopted cleanup is the §10.3 wall-breach [`Cleanup::Hebbian`] (`sign(Σⱼ simⱼ·cⱼ)`).
const PROFILE_CLEANUP: Cleanup = Cleanup::Hebbian;
const PROFILE_BUDGET: u64 = 50;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn bipolar(&mut self, dim: u32) -> Vec<f64> {
        (0..dim)
            .map(|_| {
                if (self.next_u64() >> 63) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }
}

/// One trial: build `f` fresh codebooks of `k` bipolar atoms at `dim`, pick a true tuple, bind it,
/// factorize (the given `cleanup`, iteration budget), and return `true` iff the resonator fails to
/// recover **exactly** the true tuple. A wrong-fixed-point `Ok`, an oscillation/budget error, or a
/// below-gate refusal all count as failure (RFC-0009 §5.3/§6).
fn recovery_fails(
    model: &MapI,
    f: usize,
    k: usize,
    dim: u32,
    cleanup: Cleanup,
    budget: u64,
    lcg: &mut Lcg,
) -> bool {
    let mut mems = Vec::with_capacity(f);
    let mut atoms = Vec::with_capacity(f);
    for i in 0..f {
        let mut mem = CleanupMemory::new(dim);
        let mut slot = Vec::with_capacity(k);
        for j in 0..k {
            let a = lcg.bipolar(dim);
            mem.insert(format!("{i}:{j}"), a.clone()).unwrap();
            slot.push(a);
        }
        mems.push(mem);
        atoms.push(slot);
    }
    let truth: Vec<usize> = (0..f)
        .map(|_| (lcg.next_u64() % k as u64) as usize)
        .collect();
    let mut s = atoms[0][truth[0]].clone();
    for slot in 1..f {
        s = model.bind(&s, &atoms[slot][truth[slot]]).unwrap();
    }
    let mut params = ResonatorParams::mapi_default(budget, lcg.next_u64());
    params.cleanup = cleanup;
    match factorize(model, &s, &mems, &params) {
        Ok(out) => (0..f).any(|i| out.factors[i].index != truth[i]),
        Err(_) => true,
    }
}

/// Measure the exact-recovery failure rate over `trials` at a `{f, k, dim}` point with
/// `cleanup`/`budget`. Used by the `#[ignore]`d sweep/ablation instruments below.
fn measure_rate(
    f: usize,
    k: usize,
    dim: u32,
    cleanup: Cleanup,
    budget: u64,
    trials: u64,
    salt: u64,
) -> (u64, f64) {
    let model = MapI::new(dim);
    let mut failures = 0u64;
    for trial in 0..trials {
        let mut lcg = Lcg::new(salt ^ trial);
        if recovery_fails(&model, f, k, dim, cleanup, budget, &mut lcg) {
            failures += 1;
        }
    }
    (failures, failures as f64 / trials as f64)
}

proptest! {
    // Default to ONE batch (the original single-batch LCG behaviour) so `cargo test`/`just check`
    // stay fast; `PROPTEST_CASES=N` opts into N independent batches (CI seed rotation / extra power).
    #![proptest_config(ProptestConfig { cases: 1, ..ProptestConfig::default() })]
    /// The profile gate: measured failure rate ≤ `MAPI_RESONATOR_PROFILE.delta` over exactly
    /// `profile.trials` Monte-Carlo trials at the worst covered point (max factors, max codebook,
    /// min dim), with canonical cleanup/budget.
    ///
    /// proptest generates `p.trials` independent seeds per case; `PROPTEST_CASES` controls how many
    /// independent batches run (default 1 — the single-batch behaviour; set `PROPTEST_CASES=N` for N
    /// batches). CI rotates seeds across runs automatically.
    ///
    /// Transparency: the measured rate and count are emitted via `eprintln!` — run with
    /// `--nocapture` to observe them.
    #[test]
    fn mapi_resonator_profile_holds_over_declared_trials(
        seeds in proptest::collection::vec(any::<u64>(), MAPI_RESONATOR_PROFILE.trials as usize)
    ) {
        let p = &MAPI_RESONATOR_PROFILE;
        let model = MapI::new(p.min_dim);
        let failures: u64 = seeds
            .iter()
            .filter(|&&seed| {
                let mut lcg = Lcg::new(seed);
                recovery_fails(
                    &model,
                    p.max_factors,
                    p.max_codebook,
                    p.min_dim,
                    PROFILE_CLEANUP,
                    PROFILE_BUDGET,
                    &mut lcg,
                )
            })
            .count() as u64;
        let rate = failures as f64 / p.trials as f64;
        // Transparency: emit the measured evidence so --nocapture shows it.
        eprintln!(
            "resonator profile worst point (F={}, k={}, d={}, cleanup={PROFILE_CLEANUP:?}): \
             {failures}/{} failures, rate={rate} (δ={})",
            p.max_factors, p.max_codebook, p.min_dim, p.trials, p.delta
        );
        prop_assert!(
            rate <= p.delta,
            "measured resonator failure rate {rate} ({failures}/{}) exceeds the profile's δ={} \
             (F={}, k={}, d={}) — the Empirical tag would outrun its evidence (VR-5)",
            p.trials,
            p.delta,
            p.max_factors,
            p.max_codebook,
            p.min_dim
        );
    }
}

/// Maps the MAP-I resonator's operational-capacity edge across `{F, k, d}` (RFC-0009 §9 Q4). Run
/// manually (`--ignored --nocapture`); it is the evidence behind the chosen `MAPI_RESONATOR_PROFILE`
/// envelope and the recorded δ. Staged exactly as the maintainer directed: (1) hold d=4096 and map the
/// edge at F≤3, k≤16; (2) tighten by raising d to 8192 (and β); (3) push hardest at k=32.
#[test]
#[ignore = "capacity-sweep instrument: heavy; run manually with --ignored --nocapture"]
fn resonator_capacity_sweep() {
    // (F, k, d, β, budget, trials, salt)
    let points: &[(usize, usize, u32, f64, u64, u64, u64)] = &[
        // Stage 1 — map the edge at d=4096. F=3,k=8 is the validated widening; k=16 is past the wall.
        (2, 8, 4096, 6.0, 50, 300, 0x5_0001),
        (3, 8, 4096, 6.0, 50, 300, 0x5_0002),
        (3, 16, 4096, 6.0, 50, 300, 0x5_0003),
        // Stage 2 — tighten by raising d (works for the in-regime k=8 corner; the k=16 wall does not
        // tighten even at d=8192, β=10).
        (3, 8, 8192, 6.0, 50, 300, 0x5_0004),
        (3, 16, 8192, 10.0, 80, 300, 0x5_0005),
        // Stage 3 — push hardest at k=32: far past the operational capacity (∏k ≫ d).
        (3, 32, 8192, 10.0, 80, 200, 0x5_0006),
    ];
    eprintln!("F    k    ∏k        d       β     budget trials  fails  rate");
    for &(f, k, dim, beta, budget, trials, salt) in points {
        let prod: u128 = (0..f).map(|_| k as u128).product();
        let (fails, rate) =
            measure_rate(f, k, dim, Cleanup::Softmax { beta }, budget, trials, salt);
        eprintln!(
            "{f:<4} {k:<4} {prod:<9} {dim:<7} {beta:<5} {budget:<6} {trials:<7} {fails:<6} {rate}"
        );
    }
}

/// **§10.3 cleanup ablation — the wall-breach measurement.** Compares the four cleanup variants at the
/// corners where `Softmax` collapses (`∏k → d`): F=3, k∈{16,32}, d∈{4096,8192,16384}. The bipolarizing
/// variants (`SoftmaxSign`, `Hebbian`) keep the explain-away on the `±1` alphabet, so the MAP-I unbind
/// stays exact instead of compounding real-valued crosstalk — the hypothesis this run tests. The
/// evidence here (failure rate per variant per corner) is what justifies any widening of
/// `MAPI_RESONATOR_PROFILE`; a variant that does *not* move the wall is recorded too (honest boundary,
/// VR-5). Run manually: `--ignored --nocapture`.
#[test]
#[ignore = "cleanup-ablation instrument: heavy; run manually with --ignored --nocapture"]
fn resonator_cleanup_ablation() {
    // The wall corners (F=3) plus the in-regime F=3,k=8 control, at the budgets the variants get.
    // (F, k, d, budget, trials, salt)
    let corners: &[(usize, usize, u32, u64, u64, u64)] = &[
        (3, 8, 4096, 50, 300, 0xA_0001),
        (3, 16, 4096, 50, 300, 0xA_0002),
        (3, 16, 8192, 50, 300, 0xA_0003),
        (3, 16, 16384, 50, 300, 0xA_0004),
        (3, 32, 8192, 80, 200, 0xA_0005),
        (3, 32, 16384, 80, 200, 0xA_0006),
    ];
    let variants: &[(&str, Cleanup)] = &[
        ("Softmax{6}", Cleanup::Softmax { beta: 6.0 }),
        ("SoftmaxSign{6}", Cleanup::SoftmaxSign { beta: 6.0 }),
        ("Hebbian", Cleanup::Hebbian),
        ("ArgMax", Cleanup::ArgMax),
    ];
    eprintln!("variant         F    k    ∏k        d       budget trials  fails  rate");
    for &(name, cleanup) in variants {
        for &(f, k, dim, budget, trials, salt) in corners {
            let prod: u128 = (0..f).map(|_| k as u128).product();
            let (fails, rate) = measure_rate(f, k, dim, cleanup, budget, trials, salt);
            eprintln!(
                "{name:<15} {f:<4} {k:<4} {prod:<9} {dim:<7} {budget:<6} {trials:<7} {fails:<6} {rate}"
            );
        }
    }
}
