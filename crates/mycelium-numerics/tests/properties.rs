//! Property tests for the verified-numerics kernels (E2-4; RFC-0001 §4.7 — Soundness, Monotonicity,
//! Determinism). Migrated from hand-rolled LCG (Phase-1 style) to **proptest** (M-654 Gate A3):
//! shrinking is enabled, `PROPTEST_CASES` controls the trial count, and CI rotates the seed via
//! `PROPTEST_SEED`.
//!
//! Case-tiering (DN-20): these property tests are the *empirical basis* for the numerics guarantee
//! tags (VR-5), so they are never dropped — only their case count is tiered. `PROPTEST_CASES`
//! selects the count: the everyday `just check` runs a LOW count (default 8) for fast feedback,
//! `just check-full` runs a HIGH count (256+) for full statistical power on release. With the env
//! var unset the default below (`DEFAULT_CASES`) keeps the bound exercised at modest power.

use mycelium_core::{Bound, BoundBasis, BoundKind, GuaranteeStrength, NormKind};
use mycelium_numerics::{
    accuracy_to_probability, check_error_claim, check_union_claim, compose_error_bound,
    recompute_error, AffineForm, ApRhlJudgment, Certificate, CheckOutcome, ErrorBound, ErrorOp,
    ProbBound,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies — mirror the original LCG ranges exactly.
// ---------------------------------------------------------------------------

/// Uniform in [-1, 1] (matches `Lcg::unit()`).
fn unit() -> BoxedStrategy<f64> {
    (-1.0f64..=1.0f64).boxed()
}

/// Uniform in [0, hi] (matches `Lcg::nonneg(hi)`).
fn nonneg(hi: f64) -> BoxedStrategy<f64> {
    (0.0f64..=hi).boxed()
}

// Default case count when `PROPTEST_CASES` is unset. Kept LOW (DN-20) so the everyday loop is fast;
// the bound is still exercised every commit. `just check` sets `PROPTEST_CASES=8`, `just check-full`
// sets `PROPTEST_CASES=256` for release-grade statistical power.
const DEFAULT_CASES: u32 = 8;

// Build the proptest config, reading `PROPTEST_CASES` EXPLICITLY (DN-20). NOTE: proptest only
// auto-reads `PROPTEST_CASES` for a `ProptestConfig::default()` whose `cases` is left untouched —
// the previous `ProptestConfig { cases: 20_000, .. }` literal HARDCODED 20 000 and silently ignored
// the env var. We therefore parse it ourselves so the case count is genuinely tiered. CI still
// rotates the seed via `PROPTEST_SEED`.
fn cfg() -> ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_CASES);
    ProptestConfig {
        cases,
        ..ProptestConfig::default()
    }
}

// Mutant-witness configs (the `*_refuses_*` guard tests). These exercise a refusal property where a
// modest case count is sufficient; they too honor `PROPTEST_CASES` (DN-20) with a low default so the
// fast tier stays fast while `just check-full` raises the count. Same precedence reasoning as `cfg()`.
fn witness_cfg() -> ProptestConfig {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_CASES);
    ProptestConfig {
        cases,
        ..ProptestConfig::default()
    }
}

// ---------------------------------------------------------------------------
// ErrorBound / AffineForm
// ---------------------------------------------------------------------------

// **Soundness (affine, linear ops are exact).** For every noise assignment, the composed form
// evaluates to *exactly* the corresponding real operation — `add`/`sub`/`neg`/`scale` introduce no
// error (the affine domain is exact on linear maps; ADR-010 §1).
proptest! {
    #![proptest_config(cfg())]
    #[test]
    fn affine_linear_ops_are_exact(
        // x: center in [-5,5] over shared sym 0, plus private sym 1
        x_center in (-5.0f64..=5.0f64),
        x_r0    in nonneg(3.0),
        x_r1    in nonneg(2.0),
        // y: center in [-5,5] over shared sym 0, plus private sym 2
        y_center in (-5.0f64..=5.0f64),
        y_r0    in nonneg(3.0),
        y_r2    in nonneg(2.0),
        // Noise assignments for syms 0, 1, 2
        e0 in unit(),
        e1 in unit(),
        e2 in unit(),
        // Scale factor in [-4,4]
        c in (-4.0f64..=4.0f64),
    ) {
        let x = AffineForm::uncertain(x_center, 0, x_r0)
            .unwrap()
            .add(&AffineForm::uncertain(0.0, 1, x_r1).unwrap());
        let y = AffineForm::uncertain(y_center, 0, y_r0)
            .unwrap()
            .add(&AffineForm::uncertain(0.0, 2, y_r2).unwrap());
        let assign = |s| match s {
            0 => e0,
            1 => e1,
            2 => e2,
            _ => 0.0,
        };

        let approx_add = x.add(&y).eval(assign);
        let exact_add  = x.eval(assign) + y.eval(assign);
        prop_assert!((approx_add - exact_add).abs() < 1e-9);

        let approx_sub = x.sub(&y).eval(assign);
        let exact_sub  = x.eval(assign) - y.eval(assign);
        prop_assert!((approx_sub - exact_sub).abs() < 1e-9);

        let approx_scale = x.scale(c).eval(assign);
        prop_assert!((approx_scale - c * x.eval(assign)).abs() < 1e-9);

        let approx_neg = x.neg().eval(assign);
        prop_assert!((approx_neg + x.eval(assign)).abs() < 1e-9);
    }
}

// **Soundness (affine `mul`).** The true product lies inside `[center ± radius]` of the composed
// form for every noise assignment — the second-order remainder is soundly over-approximated.
proptest! {
    #![proptest_config(cfg())]
    #[test]
    fn affine_mul_is_sound(
        x_center in (-5.0f64..=5.0f64),
        x_r0    in nonneg(3.0),
        x_r1    in nonneg(2.0),
        y_center in (-5.0f64..=5.0f64),
        y_r0    in nonneg(3.0),
        y_r2    in nonneg(2.0),
        e0 in unit(),
        e1 in unit(),
        e2 in unit(),
    ) {
        let x = AffineForm::uncertain(x_center, 0, x_r0)
            .unwrap()
            .add(&AffineForm::uncertain(0.0, 1, x_r1).unwrap());
        let y = AffineForm::uncertain(y_center, 0, y_r0)
            .unwrap()
            .add(&AffineForm::uncertain(0.0, 2, y_r2).unwrap());
        // Fresh symbol 9 not used by x or y.
        let prod = x.mul(&y, 9);
        let assign = |s| match s {
            0 => e0,
            1 => e1,
            2 => e2,
            _ => 0.0,
        };
        let true_product = x.eval(assign) * y.eval(assign);
        prop_assert!(
            (true_product - prod.center()).abs() <= prod.radius() + 1e-9,
            "mul unsound: |{true_product} - {}| > radius {}",
            prod.center(),
            prod.radius()
        );
    }
}

// **Soundness (scalar `ErrorBound`).** The composed `eps` upper-bounds the true deviation of the
// composed *values* for `add`/`sub`/`scale`/`mul` over both signs of the deviation. With outward
// rounding (A2-01) the composed `eps` is a true upper bound, so the assertions hold with **zero**
// slack — the previous `1e-9`/`1e-6` slacks (which masked the ulp-scale unsoundness, A2-07) are
// removed.
proptest! {
    #![proptest_config(cfg())]
    #[test]
    fn error_bound_scalar_is_sound(
        ex in nonneg(4.0),
        ey in nonneg(4.0),
        // True deviations within the per-input bounds (both signs exercised).
        dx_unit in unit(),
        dy_unit in unit(),
        c  in (-3.0f64..=3.0f64),
        x0 in (-6.0f64..=6.0f64),
        y0 in (-6.0f64..=6.0f64),
    ) {
        let bx = ErrorBound::new(ex, NormKind::Linf).unwrap();
        let by = ErrorBound::new(ey, NormKind::Linf).unwrap();
        // Scale to stay within the declared error bounds (matches original `rng.unit() * ex`).
        let dx = dx_unit * ex;
        let dy = dy_unit * ey;

        // add: |dx + dy| <= eps_add (both sides; A2-07 fix — was only the positive side).
        prop_assert!((dx + dy).abs() <= bx.add(&by).unwrap().eps());
        // sub: |dx - dy| <= eps_sub
        prop_assert!((dx - dy).abs() <= bx.sub(&by).unwrap().eps());
        // scale
        prop_assert!((c * dx).abs() <= bx.scale(c).eps());
        // mul about centers x0, y0: |(x0+dx)(y0+dy) - x0 y0| <= eps_mul
        let true_dev = ((x0 + dx) * (y0 + dy) - x0 * y0).abs();
        prop_assert!(true_dev <= bx.mul(&by, x0, y0).unwrap().eps());
    }
}

/// **Outward rounding (A2-01 / C1-01 regression; mutant-witness).** A composition whose real sum is
/// not representable must yield an `eps` *strictly greater* than the round-to-nearest sum — otherwise
/// the `Proven` tag `compose_error_bound` attaches would not be backed. Reverting `ErrorBound::add`
/// to `self.eps() + other.eps()` (plain RN) makes this fail.
#[test]
fn error_bound_add_rounds_outward() {
    // 1.0 and 2^-54: their real sum is unrepresentable and rounds to exactly 1.0 under RN.
    let a = ErrorBound::new(1.0, NormKind::Linf).unwrap();
    let b = ErrorBound::new(2f64.powi(-54), NormKind::Linf).unwrap();
    let composed = a.add(&b).unwrap().eps();
    assert!(
        composed > 1.0,
        "composed eps {composed} did not round outward above the RN sum 1.0"
    );
    assert!(composed >= 1.0 + 2f64.powi(-54));
    // Exact composition is preserved: 0 + 0 stays exactly 0 (Exact ⊕ Exact is not inflated).
    let zero = ErrorBound::exact(NormKind::Linf);
    assert_eq!(zero.add(&zero).unwrap().eps(), 0.0);
}

// **Monotonicity.** Raising any input `eps` can only raise the composed `eps`.
proptest! {
    #![proptest_config(cfg())]
    #[test]
    fn error_bound_is_monotone(
        ex   in nonneg(4.0),
        ey   in nonneg(4.0),
        bump in nonneg(2.0),
    ) {
        let lo = ErrorBound::new(ex, NormKind::L2).unwrap();
        let hi = ErrorBound::new(ex + bump, NormKind::L2).unwrap();
        let y  = ErrorBound::new(ey, NormKind::L2).unwrap();
        prop_assert!(hi.add(&y).unwrap().eps() >= lo.add(&y).unwrap().eps());
        prop_assert!(hi.mul(&y, 2.0, 3.0).unwrap().eps() >= lo.mul(&y, 2.0, 3.0).unwrap().eps());
    }
}

// **Determinism.** Identical inputs → identical composed `eps` (so composed bounds are
// content-addressable).
proptest! {
    #![proptest_config(cfg())]
    #[test]
    fn error_bound_is_deterministic(
        ex in nonneg(4.0),
        ey in nonneg(4.0),
    ) {
        let x = ErrorBound::new(ex, NormKind::Rel).unwrap();
        let y = ErrorBound::new(ey, NormKind::Rel).unwrap();
        prop_assert_eq!(x.add(&y), x.add(&y));
        prop_assert_eq!(x.mul(&y, 1.5, 2.5), x.mul(&y, 1.5, 2.5));
    }
}

/// Mixing norms is refused, never silently coerced (G2).
#[test]
fn error_bound_refuses_norm_mismatch() {
    let x = ErrorBound::new(1.0, NormKind::L1).unwrap();
    let y = ErrorBound::new(1.0, NormKind::L2).unwrap();
    assert!(x.add(&y).is_none());
    assert!(x.mul(&y, 1.0, 1.0).is_none());
}

/// Constructor refusals are explicit `None`, never a silent coercion (A2-08).
#[test]
fn constructors_refuse_out_of_range() {
    assert!(ErrorBound::new(f64::NAN, NormKind::L2).is_none());
    assert!(ErrorBound::new(-1.0, NormKind::L2).is_none());
    assert!(ErrorBound::new(f64::INFINITY, NormKind::L2).is_none());
    assert!(ProbBound::new(1.5).is_none());
    assert!(ProbBound::new(-0.1).is_none());
    assert!(ProbBound::new(f64::NAN).is_none());
    assert!(ApRhlJudgment::new(-0.1, 0.0).is_none());
    assert!(ApRhlJudgment::new(0.0, 1.5).is_none());
    assert!(ApRhlJudgment::new(f64::INFINITY, 0.0).is_none());
}

/// `AffineForm::uncertain` refuses a non-finite center or a non-finite/negative radius — infinite
/// uncertainty is an explicit `None`, **never** a silent collapse to an exact (radius-0) form (A2-03;
/// mutant-witness: reverting to the infallible constructor that drops a non-finite radius makes the
/// `is_none` checks fail).
#[test]
fn uncertain_refuses_non_finite() {
    assert!(AffineForm::uncertain(0.0, 0, f64::INFINITY).is_none());
    assert!(AffineForm::uncertain(0.0, 0, f64::NAN).is_none());
    assert!(AffineForm::uncertain(0.0, 0, -1.0).is_none());
    assert!(AffineForm::uncertain(f64::INFINITY, 0, 1.0).is_none());
    // A finite, non-negative radius is accepted; radius 0 is the exact constant.
    assert_eq!(AffineForm::uncertain(2.0, 0, 0.0).unwrap().radius(), 0.0);
    assert!(AffineForm::uncertain(2.0, 0, 1.5).unwrap().radius() >= 1.5);
}

// ---------------------------------------------------------------------------
// ProbBound
// ---------------------------------------------------------------------------

/// **Soundness (union bound).** The union δ upper-bounds the empirical failure rate of independent
/// events with the given per-event probabilities; and it never exceeds 1.
///
/// NOTE: the empirical simulation from the original (200k LCG trials) is retained as-is because
/// proptest shrinking on a stochastic simulation is not meaningful — the simulation itself is the
/// soundness witness, not a generated input. The 200k sample gives a 3σ confidence interval of
/// ±0.003 around the true rate, well inside the 0.01 slack.
#[test]
fn union_bound_is_sound() {
    // Use a fast deterministic RNG (SplitMix) for the simulation — same seed as the original LCG.
    let mut state: u64 = 6u64.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next_f64 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state as f64 / u64::MAX as f64
    };

    let deltas = [0.01, 0.02, 0.05];
    let bounds: Vec<ProbBound> = deltas.iter().map(|d| ProbBound::new(*d).unwrap()).collect();
    let claimed = ProbBound::union(&bounds);
    assert!(claimed.delta() <= 1.0);

    let mut failures = 0u64;
    let n = 200_000u64;
    for _ in 0..n {
        let any = deltas.iter().any(|d| next_f64() < *d);
        if any {
            failures += 1;
        }
    }
    let empirical = failures as f64 / n as f64;
    // Union bound must over-estimate the true "any fails" probability.
    assert!(
        claimed.delta() + 0.01 >= empirical,
        "union {} < emp {empirical}",
        claimed.delta()
    );
}

/// **Monotonicity + saturation.** Adding a failure mode never lowers δ; δ saturates at 1.
#[test]
fn union_bound_is_monotone_and_saturates() {
    let a = ProbBound::new(0.4).unwrap();
    let b = ProbBound::new(0.4).unwrap();
    let c = ProbBound::new(0.9).unwrap();
    assert!(a.or(&b).delta() >= a.delta());
    assert_eq!(a.or(&b).or(&c).delta(), 1.0); // 0.4+0.4+0.9 -> clamp to 1
}

/// **Determinism.** Same δ inputs → same union; empty union is `certain`.
#[test]
fn union_bound_is_deterministic() {
    let xs: Vec<ProbBound> = [0.1, 0.2]
        .iter()
        .map(|d| ProbBound::new(*d).unwrap())
        .collect();
    assert_eq!(ProbBound::union(&xs), ProbBound::union(&xs));
    assert_eq!(ProbBound::union::<&[ProbBound]>(&[]), ProbBound::certain());
}

/// apRHL `[SEQ]`: ε adds (privacy factors `e^ε` multiply), δ adds and saturates (ADR-010 §2).
#[test]
fn aprhl_seq_composes() {
    let j1 = ApRhlJudgment::new(0.5, 0.01).unwrap();
    let j2 = ApRhlJudgment::new(0.3, 0.02).unwrap();
    let seq = j1.seq(&j2);
    assert!((seq.eps() - 0.8).abs() < 1e-12);
    assert!((seq.delta() - 0.03).abs() < 1e-12);
    // Saturation at δ = 1.
    let big = ApRhlJudgment::new(0.0, 0.7).unwrap();
    assert_eq!(big.seq(&big).delta(), 1.0);
}

// ---------------------------------------------------------------------------
// tier-i checker
// ---------------------------------------------------------------------------

/// The checker accepts a sound (≥ re-derivation) claim and **rejects a too-tight** one — never a
/// silent pass (ADR-010 "Trusted base"; RFC-0002 §2).
#[test]
fn checker_rejects_too_tight_claims() {
    let x = ErrorBound::new(2.0, NormKind::Linf).unwrap();
    let y = ErrorBound::new(3.0, NormKind::Linf).unwrap();
    let inputs = [x, y];
    // Sound re-derivation of add = 5.0.
    let recomputed = recompute_error(&inputs, ErrorOp::Add).unwrap();
    assert!((recomputed.eps() - 5.0).abs() < 1e-12);

    // Exact claim: valid.
    let exact_claim = ErrorBound::new(5.0, NormKind::Linf).unwrap();
    assert_eq!(
        check_error_claim(&inputs, ErrorOp::Add, exact_claim),
        CheckOutcome::Valid
    );
    // Looser claim: valid (sound, allowed).
    let loose = ErrorBound::new(7.0, NormKind::Linf).unwrap();
    assert_eq!(
        check_error_claim(&inputs, ErrorOp::Add, loose),
        CheckOutcome::Valid
    );
    // Too-tight claim: rejected.
    let tight = ErrorBound::new(4.0, NormKind::Linf).unwrap();
    assert!(matches!(
        check_error_claim(&inputs, ErrorOp::Add, tight),
        CheckOutcome::Rejected { .. }
    ));
    // Norm mismatch: malformed.
    let wrong_norm = ErrorBound::new(5.0, NormKind::L1).unwrap();
    assert_eq!(
        check_error_claim(&inputs, ErrorOp::Add, wrong_norm),
        CheckOutcome::Malformed
    );
}

/// The union checker likewise rejects a δ claim below `min(1, Σδ)`.
#[test]
fn union_checker_rejects_too_tight() {
    let inputs = [ProbBound::new(0.1).unwrap(), ProbBound::new(0.2).unwrap()];
    assert_eq!(
        check_union_claim(&inputs, ProbBound::new(0.3).unwrap()),
        CheckOutcome::Valid
    );
    assert!(matches!(
        check_union_claim(&inputs, ProbBound::new(0.2).unwrap()),
        CheckOutcome::Rejected { .. }
    ));
}

/// The tier-i checker is **not vacuous in the small-ε regime** (A2-02; mutant-witness): a claim of
/// `eps = 0` against a tiny but nonzero re-derivation (~5e-13) is rejected — where the previous
/// absolute `1e-12` slack would have silently accepted it (claiming exactness for an approximate
/// result). Restoring the absolute `CHECK_TOL = 1e-12` makes this fail.
#[test]
fn checker_is_not_vacuous_for_tiny_bounds() {
    let x = ErrorBound::new(2.5e-13, NormKind::Linf).unwrap();
    let y = ErrorBound::new(2.5e-13, NormKind::Linf).unwrap();
    let inputs = [x, y];
    let recomputed = recompute_error(&inputs, ErrorOp::Add).unwrap();
    assert!(recomputed.eps() >= 5e-13);
    // Claiming exactness (eps = 0) for an approximate result must be rejected.
    let zero_claim = ErrorBound::new(0.0, NormKind::Linf).unwrap();
    assert!(matches!(
        check_error_claim(&inputs, ErrorOp::Add, zero_claim),
        CheckOutcome::Rejected { .. }
    ));
    // The honest (≥ re-derivation) claim is still accepted.
    assert_eq!(
        check_error_claim(&inputs, ErrorOp::Add, recomputed),
        CheckOutcome::Valid
    );
}

// ---------------------------------------------------------------------------
// cross-kernel + certificate
// ---------------------------------------------------------------------------

/// The single sanctioned cross-kernel rule: within tolerance ⇒ inherits the accuracy confidence;
/// outside ⇒ honest worst case δ = 1 (ADR-010 §4).
#[test]
fn accuracy_to_probability_is_honest() {
    let acc = ErrorBound::new(0.5, NormKind::L2).unwrap();
    // Within tolerance: failure prob = the accuracy bound's own confidence slack.
    assert_eq!(
        accuracy_to_probability(acc, 1.0, 0.03).unwrap(),
        ProbBound::new(0.03).unwrap()
    );
    // Exceeds tolerance: worst case.
    assert_eq!(
        accuracy_to_probability(acc, 0.25, 0.03).unwrap(),
        ProbBound::new(1.0).unwrap()
    );
    // Malformed tolerance.
    assert!(accuracy_to_probability(acc, -1.0, 0.0).is_none());
}

/// The shared certificate round-trips through its serialized form.
#[test]
fn certificate_round_trips() {
    let cert = Certificate::new(0.25, 0.01, GuaranteeStrength::Proven).unwrap();
    let json = serde_json::to_string(&cert).unwrap();
    let back: Certificate = serde_json::from_str(&json).unwrap();
    assert_eq!(cert, back);
    assert!(json.contains("\"strength\":\"Proven\""));
    // Out-of-range δ is refused.
    assert!(Certificate::new(0.0, 1.5, GuaranteeStrength::Declared).is_none());
}

// ---------------------------------------------------------------------------
// compose_error_bound (the M-204 entry)
// ---------------------------------------------------------------------------

fn error_bound(eps: f64, basis: BoundBasis) -> Bound {
    Bound {
        kind: BoundKind::Error {
            eps,
            norm: NormKind::Linf,
        },
        basis,
    }
}

/// Composing two `Proven` error bounds via `add` stays `Proven` (affine composition is itself sound)
/// with the composition citation, and `eps` is the kernel re-derivation.
#[test]
fn compose_keeps_proven_and_sums_eps() {
    let x = error_bound(
        1.0,
        BoundBasis::ProvenThm {
            citation: "thm-x".to_owned(),
        },
    );
    let y = error_bound(
        2.0,
        BoundBasis::ProvenThm {
            citation: "thm-y".to_owned(),
        },
    );
    let composed = compose_error_bound(&[&x, &y], ErrorOp::Add).unwrap();
    assert_eq!(composed.strength, GuaranteeStrength::Proven);
    match composed.bound.kind {
        BoundKind::Error { eps, .. } => assert!((eps - 3.0).abs() < 1e-12),
        _ => panic!("expected Error"),
    }
    assert!(matches!(composed.bound.basis, BoundBasis::ProvenThm { .. }));
}

/// The meet degrades the composed strength to the weakest input (VR-5): `Proven ⊕ Empirical →
/// Empirical`, carrying the fewest trials; `… ⊕ Declared → Declared`.
#[test]
fn compose_meets_strength_down() {
    let proven = error_bound(
        1.0,
        BoundBasis::ProvenThm {
            citation: "thm".to_owned(),
        },
    );
    let empirical = error_bound(
        2.0,
        BoundBasis::EmpiricalFit {
            trials: 10_000,
            method: "frady".to_owned(),
        },
    );
    let declared = error_bound(0.5, BoundBasis::UserDeclared);

    let pe = compose_error_bound(&[&proven, &empirical], ErrorOp::Add).unwrap();
    assert_eq!(pe.strength, GuaranteeStrength::Empirical);
    assert!(matches!(
        pe.bound.basis,
        BoundBasis::EmpiricalFit { trials: 10_000, .. }
    ));

    let pd = compose_error_bound(&[&proven, &declared], ErrorOp::Add).unwrap();
    assert_eq!(pd.strength, GuaranteeStrength::Declared);
    assert_eq!(pd.bound.basis, BoundBasis::UserDeclared);
}

/// A non-`Error` input bound has no error-composition rule → `None` (the caller refuses honestly,
/// never fabricates a bound).
#[test]
fn compose_refuses_non_error_bounds() {
    let capacity = Bound {
        kind: BoundKind::Capacity {
            items: 5,
            dim: 1000,
        },
        basis: BoundBasis::ProvenThm {
            citation: "cap".to_owned(),
        },
    };
    let err = error_bound(
        1.0,
        BoundBasis::ProvenThm {
            citation: "e".to_owned(),
        },
    );
    assert!(compose_error_bound(&[&capacity, &err], ErrorOp::Add).is_none());
    assert!(compose_error_bound(&[], ErrorOp::Add).is_none());
}

/// `compose_error_bound` refuses (returns `None`) when the composition overflows to non-finite,
/// rather than emitting a fabricated `inf` bound (A2-04; mutant-witness: removing the
/// `ErrorBound::new` re-validation in `compose_error_bound` makes this return `Some`).
#[test]
fn compose_refuses_overflow_to_non_finite() {
    let huge = error_bound(
        f64::MAX,
        BoundBasis::ProvenThm {
            citation: "x".to_owned(),
        },
    );
    // f64::MAX + f64::MAX overflows to +inf — must be refused, not emitted as a bound.
    assert!(compose_error_bound(&[&huge, &huge], ErrorOp::Add).is_none());
}

// ---------------------------------------------------------------------------
// Additional mutant-killing witnesses (M-654 Part 2)
// ---------------------------------------------------------------------------

/// **Mutant-witness: `round::add_up` returns `s` unchanged when `add_err > 0`.**
/// Mutant: flip the `> 0` guard to `>= 0` in `add_up`, inflating exact sums.
/// The exact-preservation path is tested here: a representable sum must NOT be inflated.
///
/// Location: `crates/mycelium-numerics/src/round.rs` fn `add_up`.
#[test]
fn add_up_does_not_inflate_exact_results() {
    // 1.0 + 2.0 is exactly representable; outward-rounding must not push it above 3.0.
    let eb1 = ErrorBound::new(1.0, NormKind::Linf).unwrap();
    let eb2 = ErrorBound::new(2.0, NormKind::Linf).unwrap();
    assert_eq!(eb1.add(&eb2).unwrap().eps(), 3.0);
    // 0.0 + 0.0 stays exactly 0.
    let zero = ErrorBound::exact(NormKind::Linf);
    assert_eq!(zero.add(&zero).unwrap().eps(), 0.0);
}

/// **Mutant-witness: `ProbBound::or` uses wrong min-clamp direction.**
/// Mutant: replace `.min(1.0)` with `.max(1.0)` in the `or`/`union` implementation.
///
/// Location: `crates/mycelium-numerics/src/prob.rs` fn `union` / `.min(1.0)`.
#[test]
fn union_clamp_is_min_not_max() {
    let a = ProbBound::new(0.1).unwrap();
    let b = ProbBound::new(0.2).unwrap();
    // Sum = 0.3, well below 1.0; the clamp must not inflate it to 1.0.
    let u = a.or(&b);
    assert!(u.delta() < 1.0, "union clamp inflated 0.3 to {}", u.delta());
    assert!((u.delta() - 0.3).abs() < 1e-12);
}

/// **Mutant-witness: `ErrorBound::scale` uses `1.0` instead of `c.abs()`.**
/// Mutant: replace `c.abs()` with `1.0` in the scale mul (drops the scaling factor).
///
/// Location: `crates/mycelium-numerics/src/error.rs` `ErrorBound::scale`.
#[test]
fn scale_by_zero_yields_zero_eps() {
    let b = ErrorBound::new(5.0, NormKind::Linf).unwrap();
    assert_eq!(b.scale(0.0).eps(), 0.0, "scale(0) must yield eps=0");
    // And scale by 2 doubles the bound.
    assert!((b.scale(2.0).eps() - 10.0).abs() < 1e-12);
}

/// **Mutant-witness: `ApRhlJudgment::seq` swaps eps and delta channels.**
/// Mutant: `seq` adds `self.eps` into the delta field and vice versa.
///
/// Location: `crates/mycelium-numerics/src/prob.rs` `ApRhlJudgment::seq`.
#[test]
fn aprhl_seq_channels_are_distinct() {
    let j1 = ApRhlJudgment::new(1.0, 0.1).unwrap();
    let j2 = ApRhlJudgment::new(0.5, 0.2).unwrap();
    let s = j1.seq(&j2);
    // eps must sum to 1.5 (not 0.3), delta to 0.3 (not 1.5).
    assert!((s.eps() - 1.5).abs() < 1e-12, "seq eps = {}", s.eps());
    assert!((s.delta() - 0.3).abs() < 1e-12, "seq delta = {}", s.delta());
}

/// **Mutant-witness: `check_error_claim` uses `<` instead of `>=` in the validity guard.**
/// Mutant: flip the Valid condition, making exact claims spuriously rejected.
///
/// Location: `crates/mycelium-numerics/src/cert.rs` fn `check_error_claim`.
#[test]
fn check_error_claim_accepts_exact_match() {
    let x = ErrorBound::new(1.0, NormKind::Linf).unwrap();
    let recomputed = recompute_error(&[x], ErrorOp::Neg).unwrap();
    // Exact match must be Valid.
    assert_eq!(
        check_error_claim(&[x], ErrorOp::Neg, recomputed),
        CheckOutcome::Valid
    );
}

/// **Mutant-witness: `recompute_error` `Neg` arm returns wrong value or ignores arity.**
/// Mutant: in the `Neg` arm, return `inputs[0].add(&inputs[0])` instead of `neg`.
///
/// Location: `crates/mycelium-numerics/src/cert.rs` fn `recompute_error`, `ErrorOp::Neg` arm.
#[test]
fn recompute_neg_preserves_eps() {
    let x = ErrorBound::new(3.7, NormKind::L2).unwrap();
    let r = recompute_error(&[x], ErrorOp::Neg).unwrap();
    assert_eq!(r.eps(), 3.7, "neg must preserve eps magnitude");
    // Wrong arity: two inputs for Neg is malformed.
    let y = ErrorBound::new(1.0, NormKind::L2).unwrap();
    assert!(recompute_error(&[x, y], ErrorOp::Neg).is_none());
}

/// **Mutant-witness: `basis_strength` maps wrong variant to wrong strength.**
/// Mutant: return `Empirical` for `ProvenThm` (or `Proven` for `UserDeclared`).
///
/// Location: `crates/mycelium-numerics/src/cert.rs` fn `basis_strength`.
#[test]
fn basis_strength_maps_each_variant() {
    assert_eq!(
        mycelium_numerics::basis_strength(&BoundBasis::ProvenThm {
            citation: "t".to_owned()
        }),
        GuaranteeStrength::Proven
    );
    assert_eq!(
        mycelium_numerics::basis_strength(&BoundBasis::EmpiricalFit {
            trials: 1000,
            method: "m".to_owned()
        }),
        GuaranteeStrength::Empirical
    );
    assert_eq!(
        mycelium_numerics::basis_strength(&BoundBasis::UserDeclared),
        GuaranteeStrength::Declared
    );
}

// **Mutant-witness: `ErrorBound::new` accepts negative eps or NaN.**
// Mutant: remove the `eps >= 0.0` guard, so `ErrorBound::new(-0.1, …)` returns `Some`.
//
// Location: `crates/mycelium-numerics/src/error.rs` fn `ErrorBound::new`.
proptest! {
    #![proptest_config(witness_cfg())]
    #[test]
    fn error_bound_new_refuses_negative_or_nonfinite(
        eps in prop_oneof![
            (-1e6f64..-1e-30f64),               // negative finite
            Just(f64::NEG_INFINITY),
            Just(f64::INFINITY),
            Just(f64::NAN),
        ]
    ) {
        prop_assert!(ErrorBound::new(eps, NormKind::Linf).is_none());
    }
}

// **Mutant-witness: `ProbBound::new` accepts values outside [0,1].**
// Mutant: remove the `(0..=1).contains` guard.
//
// Location: `crates/mycelium-numerics/src/prob.rs` fn `ProbBound::new`.
proptest! {
    #![proptest_config(witness_cfg())]
    #[test]
    fn prob_bound_new_refuses_out_of_range(
        delta in prop_oneof![
            (1.0001f64..=1e6f64),          // > 1
            (-1e6f64..=-0.0001f64),        // < 0
            Just(f64::NAN),
        ]
    ) {
        prop_assert!(ProbBound::new(delta).is_none());
    }
}

/// **Mutant-witness: `compose_error_bound` returns `Some` for empty input.**
/// Mutant: remove the early `if inputs.is_empty() { return None }` guard.
///
/// Location: `crates/mycelium-numerics/src/cert.rs` fn `compose_error_bound`.
#[test]
fn compose_refuses_empty_inputs() {
    assert!(compose_error_bound(&[], ErrorOp::Add).is_none());
    assert!(compose_error_bound(&[], ErrorOp::Sub).is_none());
}

/// **Mutant-witness: `AffineForm::radius` uses plain `+` instead of `round::add_up`.**
/// Mutant: replace `fold(0.0, round::add_up)` with `fold(0.0, |a, b| a + b)` in `radius`.
///
/// Location: `crates/mycelium-numerics/src/error.rs` `AffineForm::radius`.
#[test]
fn affine_radius_rounds_outward() {
    // Build a form with two coefficients whose sum is unrepresentable under RN.
    let a = AffineForm::uncertain(0.0, 0, 1.0).unwrap();
    let b = AffineForm::uncertain(0.0, 1, 2f64.powi(-54)).unwrap();
    let sum = a.add(&b);
    // The radius must be > 1.0 (not rounded down to 1.0 by RN).
    assert!(
        sum.radius() > 1.0,
        "radius was rounded down: {}",
        sum.radius()
    );
}

/// **Mutant-witness: `accuracy_to_probability` uses `<` instead of `<=` for the eps/tau check.**
/// Mutant: `acc.eps() < tau` instead of `acc.eps() <= tau`. Killed by the boundary case eps == tau.
///
/// Location: `crates/mycelium-numerics/src/cert.rs` fn `accuracy_to_probability`.
#[test]
fn accuracy_to_probability_boundary_case() {
    // eps == tau exactly; must be "within tolerance", returning acc_delta (not 1.0).
    let acc = ErrorBound::new(0.5, NormKind::Linf).unwrap();
    let result = accuracy_to_probability(acc, 0.5, 0.05).unwrap();
    assert_eq!(result, ProbBound::new(0.05).unwrap());
}

// ---------------------------------------------------------------------------
// Additional mutant-killing witnesses — Part 2 survivors from cargo-mutants
// ---------------------------------------------------------------------------

/// **Mutant-witness: `recompute_error` deletes the `Scale` arm.**
/// Mutant: delete the `[x] => Some(x.scale(c))` arm of `ErrorOp::Scale`. Without that arm,
/// Scale arity-1 returns `None`. Killed by calling recompute_error with Scale and asserting Some.
///
/// Location: `crates/mycelium-numerics/src/cert.rs:79` `recompute_error`, `Scale([x])` arm.
#[test]
fn recompute_scale_works() {
    let x = ErrorBound::new(3.0, NormKind::Linf).unwrap();
    let r = recompute_error(&[x], ErrorOp::Scale(2.0)).unwrap();
    // 3.0 * 2.0 = 6.0 (outward-rounded, exact here).
    assert!(
        (r.eps() - 6.0).abs() < 1e-12,
        "Scale recompute eps = {}",
        r.eps()
    );
    // Wrong arity: two inputs for Scale is malformed.
    let y = ErrorBound::new(1.0, NormKind::Linf).unwrap();
    assert!(recompute_error(&[x, y], ErrorOp::Scale(1.0)).is_none());
}

/// **Mutant-witness: `recompute_error` deletes the `Mul` arm.**
/// Mutant: delete the `[x, y] => x.mul(y, …)` arm of `ErrorOp::Mul`. Without that arm,
/// Mul arity-2 returns `None`. Killed by calling recompute_error with Mul and asserting Some.
///
/// Location: `crates/mycelium-numerics/src/cert.rs:83` `recompute_error`, `Mul([x,y])` arm.
#[test]
fn recompute_mul_works() {
    let x = ErrorBound::new(1.0, NormKind::Linf).unwrap();
    let y = ErrorBound::new(2.0, NormKind::Linf).unwrap();
    // eps_mul = |x0|*eps_y + |y0|*eps_x + eps_x*eps_y = 3*2 + 4*1 + 1*2 = 12.
    let r = recompute_error(
        &[x, y],
        ErrorOp::Mul {
            x0_mag: 3.0,
            y0_mag: 4.0,
        },
    )
    .unwrap();
    assert!(r.eps() >= 12.0, "Mul recompute eps = {}", r.eps());
    // Wrong arity: single input for Mul is malformed.
    assert!(recompute_error(
        &[x],
        ErrorOp::Mul {
            x0_mag: 1.0,
            y0_mag: 1.0
        }
    )
    .is_none());
}

/// **Mutant-witness: `Certificate::eps()` returns a constant (0.0/1.0/-1.0).**
/// Mutant: replace `self.eps` with `0.0` (or 1.0 or -1.0) in the getter. Killed by asserting
/// that two certificates with distinct eps values actually return distinct eps().
///
/// Location: `crates/mycelium-numerics/src/cert.rs:186` `Certificate::eps`.
#[test]
fn certificate_eps_getter_returns_stored_value() {
    let c1 = Certificate::new(0.37, 0.01, GuaranteeStrength::Proven).unwrap();
    let c2 = Certificate::new(0.73, 0.01, GuaranteeStrength::Proven).unwrap();
    // If eps() always returned 0.0, both would be 0.0 and equal.
    assert!((c1.eps() - 0.37).abs() < 1e-12, "c1.eps() = {}", c1.eps());
    assert!((c2.eps() - 0.73).abs() < 1e-12, "c2.eps() = {}", c2.eps());
    assert!(c1.eps() != c2.eps(), "getter must return the stored value");
}

/// **Mutant-witness: `Certificate::delta()` returns a constant (0.0/1.0/-1.0).**
/// Mutant: replace `self.delta` with `0.0` (or 1.0 or -1.0) in the getter.
///
/// Location: `crates/mycelium-numerics/src/cert.rs:191` `Certificate::delta`.
#[test]
fn certificate_delta_getter_returns_stored_value() {
    let c1 = Certificate::new(0.0, 0.25, GuaranteeStrength::Proven).unwrap();
    let c2 = Certificate::new(0.0, 0.75, GuaranteeStrength::Proven).unwrap();
    assert!(
        (c1.delta() - 0.25).abs() < 1e-12,
        "c1.delta() = {}",
        c1.delta()
    );
    assert!(
        (c2.delta() - 0.75).abs() < 1e-12,
        "c2.delta() = {}",
        c2.delta()
    );
    assert!(
        c1.delta() != c2.delta(),
        "getter must return the stored value"
    );
}

/// **Mutant-witness: `composed_basis` deletes the `ProvenThm{citation}` filter_map arm.**
/// Mutant: the filter_map body drops the `BoundBasis::ProvenThm { citation }` arm, so the
/// `inputs` vec is always empty and the citation never includes theorem names. Killed by
/// checking that the composed citation actually contains the input theorem citations.
///
/// Location: `crates/mycelium-numerics/src/cert.rs:288` `composed_basis`, ProvenThm filter arm.
#[test]
fn composed_basis_includes_input_citations() {
    let x = error_bound(
        1.0,
        BoundBasis::ProvenThm {
            citation: "sentinel-thm-alpha".to_owned(),
        },
    );
    let y = error_bound(
        2.0,
        BoundBasis::ProvenThm {
            citation: "sentinel-thm-beta".to_owned(),
        },
    );
    let composed = compose_error_bound(&[&x, &y], ErrorOp::Add).unwrap();
    // The citation must contain both input theorem names — not just the bare affine citation.
    match &composed.bound.basis {
        BoundBasis::ProvenThm { citation } => {
            assert!(
                citation.contains("sentinel-thm-alpha"),
                "composed citation missing alpha: {citation}"
            );
            assert!(
                citation.contains("sentinel-thm-beta"),
                "composed citation missing beta: {citation}"
            );
        }
        other => panic!("expected ProvenThm, got {other:?}"),
    }
}

/// **Mutant-witness: `error_norm` returns `None` always, or deletes the `Error{norm}` arm.**
/// Mutant (a): replace whole function body with `None`. Mutant (b): delete the `BoundKind::Error`
/// match arm. Killed by calling `error_norm` on an Error bound and asserting `Some(norm)`.
///
/// Location: `crates/mycelium-numerics/src/cert.rs:356–357` `error_norm`.
#[test]
fn error_norm_extracts_norm() {
    use mycelium_numerics::error_norm;
    let b = error_bound(
        1.0,
        BoundBasis::ProvenThm {
            citation: "t".to_owned(),
        },
    );
    // An Error bound must return Some with the correct norm.
    assert_eq!(error_norm(&b), Some(NormKind::Linf));
    // A non-Error bound must return None.
    let cap = Bound {
        kind: BoundKind::Capacity { items: 1, dim: 1 },
        basis: BoundBasis::UserDeclared,
    };
    assert_eq!(error_norm(&cap), None);
}

/// **Mutant-witness: `AffineForm::uncertain` changes `> 0.0` to `>= 0.0` (stores zero term).**
/// Mutant: `if radius > 0.0` → `if radius >= 0.0` means a radius=0.0 form stores a zero-coeff
/// term. The radius() is unchanged (0 * 1.0 = 0.0), but the form has extra structure. We detect
/// this by comparing an uncertain(c, s, 0.0) to a constant(c) via structural eval: if a zero
/// term is stored on sym `s`, eval with that sym as 1.0 differs from eval with sym=0 ONLY if the
/// coefficient is truly stored. With coeff=0 it doesn't matter. A better test: an extra zero-coeff
/// term means eval(assign) would still equal center (×0 = 0). So instead we test that the EQ
/// invariant holds: uncertain(c, s, 0) == AffineForm::constant(c).
///
/// NOTE: `AffineForm` does not implement `Eq` beyond `PartialEq` but it DOES implement
/// `PartialEq`. The equality impl only cares about the terms map; a zero-coeff term that is
/// accidentally stored means the forms differ structurally.
///
/// Location: `crates/mycelium-numerics/src/error.rs:78` `AffineForm::uncertain`, zero-radius guard.
#[test]
fn uncertain_zero_radius_equals_constant() {
    // The exact-constant form has an empty `terms` map. Use an arbitrary non-special constant.
    let center = 7.654_321_0_f64;
    let exact = AffineForm::constant(center);
    // uncertain with radius=0 must be equal to constant — no term stored for sym 0.
    let via_uncertain = AffineForm::uncertain(center, 0, 0.0).unwrap();
    // AffineForm: PartialEq on (center, terms). If the mutant stores a 0-coeff term, ≠.
    assert_eq!(
        exact, via_uncertain,
        "uncertain(c, s, 0) must equal constant(c) — no zero-coeff term stored (center={center})"
    );
}

/// **Mutant-witness: `AffineForm::eval` changes `+` to `-` in the accumulator.**
/// Mutant: `self.center + self.terms.iter().map(…).sum()` → `self.center - sum`.
/// Killed by evaluating a form with a known positive coefficient and checking sign of result.
///
/// Location: `crates/mycelium-numerics/src/error.rs:106` `AffineForm::eval`.
#[test]
fn affine_eval_sign_is_positive_accumulation() {
    // Form: center=10.0, coefficient on sym 0 is +3.0.
    let f = AffineForm::uncertain(10.0, 0, 3.0).unwrap();
    // With sym 0 = +1.0: eval = 10.0 + 3.0*1.0 = 13.0 (NOT 10.0 - 3.0 = 7.0).
    let val = f.eval(|s| if s == 0 { 1.0 } else { 0.0 });
    assert!(
        (val - 13.0).abs() < 1e-12,
        "eval should be 13.0 (center + term), got {val}"
    );
}

/// **Mutant-witness: `AffineForm::add` changes `+` to `*` in the center field.**
/// Mutant: `center: self.center + other.center` → `center: self.center * other.center`.
/// Killed by adding two forms with non-zero centers and checking the center is the sum, not product.
///
/// Location: `crates/mycelium-numerics/src/error.rs:151` `AffineForm::add`, center computation.
#[test]
fn affine_add_center_is_sum_not_product() {
    // 3.0 + 5.0 = 8.0; 3.0 * 5.0 = 15.0 — clearly distinct.
    let a = AffineForm::uncertain(3.0, 0, 0.0).unwrap(); // center 3, no terms
    let b = AffineForm::uncertain(5.0, 1, 0.0).unwrap(); // center 5, no terms
    let s = a.add(&b);
    assert!(
        (s.center() - 8.0).abs() < 1e-12,
        "add center must be sum (8.0), got {}",
        s.center()
    );
}

/// **Mutant-witness: `round::add_err` changes `-` to `+` in the two-sum formula.**
/// Mutant: `(a - (s - bv)) + (b - bv)` → `(a + (s - bv)) + (b - bv)`. This corrupts the
/// round-off computation so `add_up` no longer detects when the IEEE sum rounded down.
/// Killed by checking the outward-rounding case that depends on a positive `add_err`:
/// 1.0 + 2^-54 must produce a composed eps > 1.0.
///
/// NOTE: This is the same case as `error_bound_add_rounds_outward`. The add_err mutant makes
/// the guard condition `add_err(1.0, 2^-54) > 0.0` return wrong sign, so `add_up` doesn't
/// push up. The existing test already exercises this path — the mutant should already be
/// caught. Including a direct structural check here for round::add_err sign correctness.
///
/// Location: `crates/mycelium-numerics/src/round.rs:18` `add_err`, subtraction sign.
#[test]
fn add_err_sign_for_rounded_down_sum() {
    // 1.0 + 1e-17: IEEE rounds DOWN (result is 1.0, the real sum is slightly above).
    // The correct add_err for a down-rounded sum is POSITIVE.
    // We observe add_err indirectly via add_up: if add_err were always wrong-sign,
    // add_up would never call next_up, and the result would stay at 1.0.
    let a = ErrorBound::new(1.0, NormKind::Linf).unwrap();
    let b = ErrorBound::new(1e-17, NormKind::Linf).unwrap();
    let composed = a.add(&b).unwrap().eps();
    // The real sum 1.0 + 1e-17 rounds DOWN to 1.0; add_up must push to > 1.0.
    assert!(
        composed > 1.0,
        "add_err mutant: composed eps was not rounded outward: {composed}"
    );
}

/// **Mutant-witness: `round::mul_err` always returns 0.0 or -1.0.**
/// Mutant (a) 0.0: `mul_up` never calls `next_up` → downward-rounded products not pushed up.
/// Mutant (b) -1.0: `mul_up` never calls `next_up` either (mul_err=-1.0 is never > 0).
///
/// Killed by strict-inequality tests on a downward-rounded product:
/// `a = 1 + 2^-52`, `a * a` rounds DOWN to `1 + 2^-51` in f64. With mul_err returning
/// the true error of mul, mul_up pushes to `(1 + 2^-51).next_up()`. With 0.0 or -1.0, it stays.
/// We observe this via `ErrorBound::scale(a)` which calls `mul_up(a, eps)` internally.
///
/// Location: `crates/mycelium-numerics/src/round.rs:25` `mul_err`.
#[test]
fn mul_up_rounds_outward_for_inexact_product() {
    // a = 1 + 2^-52 (largest f64 with biased exponent 0).
    // a * a (true) = 1 + 2^-51 + 2^-104, which rounds DOWN to 1 + 2^-51 in f64.
    // mul_up(a, a) must return (1 + 2^-51).next_up(), which is strictly > a*a (the f64 product).
    let a = 1.0_f64 + 2f64.powi(-52);
    let f64_product = a * a; // = 1 + 2^-51 (rounded down)
                             // Access mul_up(a, a) via ErrorBound::scale(a) on a bound with eps = a.
    let b = ErrorBound::new(a, NormKind::Linf).unwrap();
    let scaled = b.scale(a);
    // With correct mul_up: scaled.eps() > f64_product (pushed up by one ULP).
    // With 0.0/−1.0 mutant: scaled.eps() == f64_product (no push).
    assert!(
        scaled.eps() > f64_product,
        "mul_up must push downward-rounded product above f64 result: \
         scaled={} but f64_product={}",
        scaled.eps(),
        f64_product
    );
    // Confirm the exact product is NOT inflated (guards against a hypothetical always-next_up).
    let exact_b = ErrorBound::new(2.0, NormKind::Linf).unwrap();
    let exact_scale = exact_b.scale(3.0);
    assert_eq!(
        exact_scale.eps(),
        6.0,
        "exact product must not be inflated by mul_up: {}",
        exact_scale.eps()
    );
}

/// **Mutant-witness: `round::mul_up` flips guard `> 0` to `< 0`.**
/// Mutant: `if mul_err(a, b) > 0.0 { p.next_up() }` → `if mul_err(a, b) < 0.0 { p.next_up() }`.
/// With the flipped guard, mul_up DOES NOT call next_up when the product rounds DOWN (failing
/// to correct). Same observable effect as the mul_err→0.0/−1.0 mutants.
///
/// Killed by the same strict-inequality check in `mul_up_rounds_outward_for_inexact_product`
/// (which shares the core logic), plus an exact-product non-inflation check here.
///
/// Location: `crates/mycelium-numerics/src/round.rs:45` `mul_up`, guard direction.
#[test]
fn mul_up_guard_direction() {
    // Exact product: mul_up(4.0, 2.5) must return exactly 10.0, not 10.0.next_up().
    // (The flipped-guard mutant with mul_err correctly computed: for an exact product,
    //  mul_err = 0.0, so `< 0.0` is false → no next_up → result is exactly 10.0. Same. ✓)
    let b = ErrorBound::new(2.5, NormKind::Linf).unwrap();
    let s = b.scale(4.0);
    assert_eq!(
        s.eps(),
        10.0,
        "exact product must not be inflated: {}",
        s.eps()
    );

    // Downward-rounded product (same case as mul_up_rounds_outward_for_inexact_product):
    // a = 1 + 2^-52, a*a rounds DOWN → mul_up must push strictly above the f64 product.
    // With the flipped guard: mul_err(a,a) > 0 (it rounds down), so `< 0.0` is false → no push.
    let a = 1.0_f64 + 2f64.powi(-52);
    let f64_product = a * a;
    let eb = ErrorBound::new(a, NormKind::Linf).unwrap();
    assert!(
        eb.scale(a).eps() > f64_product,
        "mul_up guard-flip mutant: failed to push downward-rounded product above f64 result \
         (got {}, expected > {})",
        eb.scale(a).eps(),
        f64_product
    );
}
