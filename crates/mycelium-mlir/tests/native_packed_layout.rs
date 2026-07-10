//! M-610 — the **E3 wrong-layout soundness differential carried onto the native packed-ternary
//! kernel** (RFC-0004 §5/§8; NFR-1/NFR-4/NFR-7; DN-01; ADR-009; phase-6.md M-610).
//!
//! M-251's `wrong_layout.rs` proves a mislabeled `Meta.physical` tag is caught at the AOT
//! *relayout* stage (`run_with_layout`). This file carries the **same** soundness property one
//! level deeper — onto the **actual compiled BitNet dot kernel** (the M-360 scalar/SIMD kernels now
//! on the native backend, M-610). The kernel does not silently "know" the layout: it decodes the
//! [`PackScheme`] its [`KernelLayout`] records, that record is queryable + `EXPLAIN`-able, and:
//!
//! - **A correctly-labeled layout passes:** the kernel that decodes scheme `S` over a buffer packed
//!   under `S` reproduces the packing-independent oracle ([`ternary_dot_ref`]), and the M-210 shared
//!   checker (`ObservationalEquiv`) **validates** the pair — the recorded layout is honest.
//! - **A mislabeled layout is caught:** a kernel decoding scheme `W` over a buffer packed under the
//!   *true* scheme `S` (`W ≠ S`) misreads the weights ⇒ a different dot ⇒ the **same** checker
//!   reports an explicit `NotValidated { Diverged }`. The native packed-ternary layout record is
//!   trusted **only because a wrong one is caught here** (NFR-7).
//!
//! Honesty (VR-5): the dot is exact `i64` integer arithmetic (no guarantee upgraded); the verdict is
//! an observational-equivalence check through the single shared M-210 checker. Skips when `clang`
//! is absent. Guard 7: a discrimination test confirms the pass is not vacuous (the verdict flips
//! solely on the layout tag).

mod common;
use common::i64_value;

use mycelium_cert::{check, CheckVerdict, Evidence, NotValidatedReason, RefinementRelation};
use mycelium_core::{GuaranteeStrength, PackScheme, PhysicalLayout, Trit};
use mycelium_mlir::{
    compile_bitnet_dot_for, needed_bytes_for, pack_trits, ternary_dot_ref, AotError,
    BitnetDotKernel,
};

fn weights(n: usize) -> Vec<Trit> {
    let mut s = 0xB1C2_D3E4_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            match (s >> 33) % 3 {
                0 => Trit::Neg,
                1 => Trit::Zero,
                _ => Trit::Pos,
            }
        })
        .collect()
}
fn activations(n: usize) -> Vec<i32> {
    let mut s = 0x0F1E_2D3C_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((s >> 40) % 201) as i64 - 100) as i32
        })
        .collect()
}

/// Pack `w` under its *true* scheme `packed_as`, then present the buffer to a kernel that decodes
/// `read_as`: zero-pad to the bytes `read_as` requires so the read is the modeled **misread** (a
/// wrong layout tag over the *same* buffer, zero-extended), never an explicit short-buffer error —
/// exactly the relayout model in `pack::relayout_trits`. Returns `None` on the house toolchain skip.
fn kernel_dot_misread(
    w: &[Trit],
    x: &[i32],
    packed_as: PackScheme,
    read_as: PackScheme,
) -> Option<i64> {
    let kernel = match compile_bitnet_dot_for(read_as) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return None,
        Err(e) => panic!("kernel compile ({read_as:?}) failed: {e}"),
    };
    let mut buf = pack_trits(w, packed_as);
    let needed = needed_bytes_for(read_as, w.len());
    if buf.len() < needed {
        buf.resize(needed, 0);
    }
    Some(
        kernel
            .call(&buf, x, w.len())
            .expect("kernel runs on padded buffer"),
    )
}

/// Compile a kernel for `scheme`, skipping on a missing toolchain.
fn kernel(scheme: PackScheme) -> Option<BitnetDotKernel> {
    match compile_bitnet_dot_for(scheme) {
        Ok(k) => Some(k),
        Err(AotError::ToolchainMissing(_)) => None,
        Err(e) => panic!("kernel compile ({scheme:?}) failed: {e}"),
    }
}

#[test]
fn a_correctly_labeled_layout_passes_the_kernel_differential() {
    for scheme in [PackScheme::I2S, PackScheme::Tl1, PackScheme::Tl2] {
        let Some(k) = kernel(scheme) else { return };
        // The kernel reports the exact PhysicalLayout it decodes (the honest meta.physical record).
        assert_eq!(k.layout().physical(), PhysicalLayout::TritPacked { scheme });
        for n in [1usize, 4, 7, 12, 64, 257] {
            let w = weights(n);
            let x = activations(n);
            let packed = pack_trits(&w, scheme);
            let got = k.call(&packed, &x, n).expect("kernel runs");
            let oracle = ternary_dot_ref(&w, &x);
            assert_eq!(
                got, oracle,
                "{scheme:?} kernel diverged from oracle at n={n}"
            );
            // The same M-210 checker validates the observational-equivalence pair (NFR-7).
            assert_eq!(
                check(
                    &i64_value(oracle),
                    &i64_value(got),
                    RefinementRelation::ObservationalEquiv,
                    mycelium_numerics::Certificate::exact(),
                    &Evidence::Observational,
                ),
                CheckVerdict::Validated {
                    strength: GuaranteeStrength::Exact
                },
                "{scheme:?} n={n}: a correctly-labeled layout must pass the NFR-7 check"
            );
        }
    }
}

#[test]
fn a_mislabeled_layout_is_caught_by_the_kernel_differential() {
    // The buffer is packed under the TRUE scheme but decoded by a kernel for the WRONG scheme — a
    // genuine native misread. The three bitnet packings are mutually-distinct encodings (pack.rs),
    // so the decode diverges, and the shared checker catches it.
    let pairs = [
        (PackScheme::I2S, PackScheme::Tl1), // same 2-bit byte length, different rotation LUT
        (PackScheme::Tl1, PackScheme::I2S),
        (PackScheme::Tl2, PackScheme::I2S), // bitstream packed, read as 2-bit (padded) — misread
        (PackScheme::I2S, PackScheme::Tl2),
    ];
    let mut ran = false;
    for (true_scheme, wrong_scheme) in pairs {
        assert_ne!(true_scheme, wrong_scheme);
        // n chosen so the all-Zero degeneracy cannot hide the divergence; mixed weights below.
        let n = 64;
        let w = weights(n);
        let x = activations(n);
        let oracle = ternary_dot_ref(&w, &x);
        let Some(misread) = kernel_dot_misread(&w, &x, true_scheme, wrong_scheme) else {
            return; // toolchain skip
        };
        ran = true;
        assert_ne!(
            misread, oracle,
            "packing as {true_scheme:?}, decoding as {wrong_scheme:?} must misread the weights"
        );
        let verdict = check(
            &i64_value(oracle),
            &i64_value(misread),
            RefinementRelation::ObservationalEquiv,
            mycelium_numerics::Certificate::exact(),
            &Evidence::Observational,
        );
        assert!(
            matches!(
                verdict,
                CheckVerdict::NotValidated {
                    reason: NotValidatedReason::Diverged { .. },
                    ..
                }
            ),
            "packing {true_scheme:?} read as {wrong_scheme:?} must fail the NFR-7 check, got {verdict:?}"
        );
    }
    assert!(
        ran,
        "at least one mislabel pair must have run (non-vacuous)"
    );
}

#[test]
fn the_kernel_verdict_flips_solely_on_the_layout_tag() {
    // Guard 7: holding the data fixed, flipping which scheme the kernel decodes flips the verdict
    // from Validated (correct tag) to NotValidated (wrong tag) — so the pass is meaningful and is
    // about the layout, nothing else. The buffer is packed under I2_S throughout.
    let n = 64;
    let w = weights(n);
    let x = activations(n);
    let oracle = ternary_dot_ref(&w, &x);

    let validates = |read_as: PackScheme| -> Option<bool> {
        let got = kernel_dot_misread(&w, &x, PackScheme::I2S, read_as)?;
        Some(matches!(
            check(
                &i64_value(oracle),
                &i64_value(got),
                RefinementRelation::ObservationalEquiv,
                mycelium_numerics::Certificate::exact(),
                &Evidence::Observational,
            ),
            CheckVerdict::Validated { .. }
        ))
    };

    // The correct tag (I2_S over an I2_S buffer) validates; the wrong tags are caught.
    let Some(correct) = validates(PackScheme::I2S) else {
        return; // toolchain skip
    };
    assert!(correct, "correct tag (I2_S over an I2_S buffer) validates");
    assert_eq!(
        validates(PackScheme::Tl1),
        Some(false),
        "wrong tag (TL1) is caught"
    );
    assert_eq!(
        validates(PackScheme::Tl2),
        Some(false),
        "wrong tag (TL2) is caught"
    );
}
