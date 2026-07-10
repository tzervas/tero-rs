//! M-230 — property tests **exercising the per-op rounding bound** (SC-2): over a deterministic
//! 20k-pair sweep per dtype, every `add`/`sub`/`scale` result element stays within the disclosed
//! relative ε of the exact (`f64`) reference, and the op result discloses exactly that bound.
//!
//! Exponents are drawn from a ±20 window so the `f64` reference arithmetic is *negligibly*
//! perturbed relative to the disclosed ε: products/scale are **exact** (24×24-bit mantissas →
//! ≤48-bit product, well inside f64's 53-bit significand), and sums/differences carry at most a
//! single f64 round-off (~2⁻⁵³) — far inside the per-op ε (≥2⁻²⁴), so the comparison still
//! measures only the op's own rounding, the thing the bound claims to cover.

use mycelium_core::{BoundKind, GuaranteeStrength, NormKind, ScalarKind};
use mycelium_dense::{DenseSpace, BF16_OP_REL_EPS, F32_OP_REL_EPS};

/// Deterministic generator of on-grid values (a tiny LCG — no rand dependency, house style).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A finite nonzero value on the `dtype` grid with exponent in `[−20, 20]`.
    fn on_grid(&mut self, dtype: ScalarKind) -> f64 {
        let r = self.next();
        let mantissa = ((r & 0x7F_FFFF) | 0x80_0000) as f32; // 24-bit, MSB set
        let exp = ((r >> 24) % 41) as i32 - 20;
        let sign = if (r >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        let x = sign * mantissa * (exp as f32).exp2() / (2.0_f32).powi(23);
        let x = f64::from(x);
        match dtype {
            ScalarKind::Bf16 => {
                // Snap to the bf16 grid (round via the same f32-truncation trick).
                #[allow(clippy::cast_possible_truncation)]
                let xf = x as f32;
                let bits = xf.to_bits();
                let lsb = (bits >> 16) & 1;
                f64::from(f32::from_bits(((bits + 0x7FFF + lsb) >> 16) << 16))
            }
            _ => x,
        }
    }
}

const PAIRS: usize = 20_000;

fn sweep(dtype: ScalarKind, eps: f64) {
    let space = DenseSpace::new(1, dtype).unwrap();
    let mut rng = Lcg::new(0xD15EA5E ^ dtype as u64);
    // A5-05: count the ops that actually exercised the bound. Without this the `else { continue }`
    // on a refusal would let the sweep pass *vacuously* if the ops ever regressed to always-refuse.
    let mut successes = 0usize;
    for i in 0..PAIRS {
        let x = rng.on_grid(dtype);
        let y = rng.on_grid(dtype);
        let a = space.value(vec![x]).unwrap();
        let b = space.value(vec![y]).unwrap();

        for (name, result, truth) in [
            ("add", space.add_values(&a, &b), x + y),
            ("sub", space.sub_values(&a, &b), x - y),
            ("scale", space.scale_value(&a, y), x * y),
        ] {
            // Subnormal results are an allowed explicit refusal (the bound does not cover them).
            let Ok(v) = result else { continue };
            let got = match v.payload() {
                mycelium_core::Payload::Scalars(s) => s[0],
                _ => unreachable!(),
            };
            let dev = (got - truth).abs();
            assert!(
                dev <= eps * truth.abs() + f64::MIN_POSITIVE,
                "pair {i} ({name}, {dtype:?}): |{got} − {truth}| = {dev} exceeds ε·|t| = {}",
                eps * truth.abs()
            );
            // The disclosed bound is exactly the per-op ε the sweep just exercised.
            assert_eq!(v.meta().guarantee(), GuaranteeStrength::Proven);
            match v.meta().bound().map(|b| &b.kind) {
                Some(BoundKind::Error { eps: e, norm }) => {
                    assert_eq!(*e, eps);
                    assert_eq!(*norm, NormKind::Rel);
                }
                other => panic!("expected an Error bound, got {other:?}"),
            }
            successes += 1;
        }
    }
    // A5-05: the bound must have been exercised on the overwhelming majority of the 3×PAIRS ops.
    // On-grid inputs in the ±20 window only rarely produce a subnormal refusal, so a regression to
    // always-refuse (or a logic bug that skips the asserts) can no longer pass this test silently.
    let total = 3 * PAIRS;
    assert!(
        successes >= total - total / 100,
        "{dtype:?}: only {successes}/{total} ops exercised the bound — the sweep is near-vacuous"
    );
}

/// F32 ops stay within `u = 2⁻²⁴` (single rounding) over 20k pairs × 3 ops.
#[test]
fn f32_ops_respect_the_disclosed_bound() {
    sweep(ScalarKind::F32, F32_OP_REL_EPS);
}

/// BF16 ops stay within `2⁻⁸ + 2⁻²³` (two-rounding composition) over 20k pairs × 3 ops.
#[test]
fn bf16_ops_respect_the_disclosed_bound() {
    sweep(ScalarKind::Bf16, BF16_OP_REL_EPS);
}
