//! In-crate white-box tests for [`crate::bitnet`] (M-360; CLAUDE.md test-layout rule). Extracted from
//! the logic file (M-797 lazy retrofit, as-touched by M-728). `use crate::bitnet::*` gives white-box
//! access to the kernel emitters/layout record.

use crate::bitnet::*;
use crate::llvm::AotError;
use crate::pack::pack_trits;
use mycelium_core::{PackScheme, PhysicalLayout, Trit};

/// Deterministic ternary/activation test data (small LCGs) — fixed, not a statistical sample.
fn weights(n: usize) -> Vec<Trit> {
    let mut s = 0x1234_5678_u64;
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
    let mut s = 0x9E37_79B9_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // small signed activations so the i64 accumulator never overflows in tests
            (((s >> 40) % 201) as i64 - 100) as i32
        })
        .collect()
}

#[test]
fn ir_is_inspectable_and_deterministic() {
    let ir = emit_bitnet_dot_ir();
    // Inspectable: the unpack-compute loop is visible (no opaque pass — FR-C3 / RFC-0004 §6).
    assert!(ir.contains("define i64 @myc_bitnet_dot(ptr %w, ptr %x, i64 %n)"));
    assert!(ir.contains("load i8")); // loads packed weight bytes from the runtime pointer
    assert!(ir.contains("and i32 %shifted, 3")); // extracts the 2-bit I2_S code
    assert!(ir.contains("sub i32 %code, 1")); // forms the signed ternary weight
    assert!(ir.contains("mul i64") && ir.contains("add i64")); // multiply-accumulate
    assert_eq!(emit_bitnet_dot_ir(), emit_bitnet_dot_ir());
}

#[test]
fn ref_matches_hand_computed() {
    // [-1, 0, +1] · [7, 9, 4] = -7 + 0 + 4 = -3. Pins the oracle itself.
    let w = vec![Trit::Neg, Trit::Zero, Trit::Pos];
    let x = vec![7, 9, 4];
    assert_eq!(ternary_dot_ref(&w, &x), -3);
}

#[test]
fn jit_dot_matches_reference() {
    // Mutant-witness: a wrong shift/mask (e.g. extracting the wrong lane) or a `code` instead of
    // `code-1` weight would diverge from the oracle on this mixed data.
    for n in [1usize, 4, 5, 7, 64, 256, 1000] {
        let w = weights(n);
        let x = activations(n);
        match jit_ternary_dot(&w, &x) {
            Ok(got) => assert_eq!(got, ternary_dot_ref(&w, &x), "dot mismatch at n={n}"),
            Err(AotError::ToolchainMissing(_)) => return, // environment skip
            Err(e) => panic!("unexpected BitNet JIT error at n={n}: {e}"),
        }
    }
}

#[test]
fn compile_once_call_many_is_consistent() {
    // The compile-once/run-many shape (used by the E1 harness): the same kernel instance over
    // different buffers must each match the oracle.
    let kernel = match compile_bitnet_dot() {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("compile failed: {e}"),
    };
    for n in [16usize, 100, 333] {
        let w = weights(n);
        let x = activations(n);
        let packed = pack_trits(&w, KERNEL_SCHEME);
        assert_eq!(
            kernel.call(&packed, &x, n).unwrap(),
            ternary_dot_ref(&w, &x),
            "compiled kernel diverged at n={n}"
        );
    }
}

#[test]
fn tl1_and_tl2_ir_unpack_correctly() {
    // The scheme-specific unpack is visible in each emitted kernel (no opaque pass).
    let tl1 = emit_bitnet_dot_ir_for(PackScheme::Tl1).unwrap();
    assert!(tl1.contains("urem i32 %c1, 3")); // TL1 inverts rot=2: d01 = (code+1) mod 3
    assert!(tl1.contains("(Tl1; M-360)"));
    let tl2 = emit_bitnet_dot_ir_for(PackScheme::Tl2).unwrap();
    assert!(tl2.contains("udiv i64 %i, 3")); // TL2 (true 1.67 b/w): 3 trits per 5-bit group
    assert!(tl2.contains("and i16 %wsh, 31")); // extract the 5-bit LUT-index code
    assert!(tl2.contains("select i1 %inrange")); // the clamped 2-byte window read (no OOB)
    assert!(tl2.contains("urem i64 %q, 3")); // digit = (code / 3^pos) mod 3
                                             // Deterministic per scheme.
    assert_eq!(tl2, emit_bitnet_dot_ir_for(PackScheme::Tl2).unwrap());
}

#[test]
fn jit_dot_matches_reference_all_schemes() {
    // Mutant-witness: each scheme decodes its packing differently (rot / base-3 order); a kernel
    // that used the wrong unpack would diverge from the oracle on this mixed data. The oracle is
    // packing-independent (operates on unpacked trits), so all three must hit the *same* sum.
    for scheme in [PackScheme::I2S, PackScheme::Tl1, PackScheme::Tl2] {
        for n in [1usize, 4, 5, 7, 10, 64, 257, 1000] {
            let w = weights(n);
            let x = activations(n);
            match jit_ternary_dot_for(&w, &x, scheme) {
                Ok(got) => {
                    assert_eq!(got, ternary_dot_ref(&w, &x), "{scheme:?} mismatch at n={n}");
                }
                Err(AotError::ToolchainMissing(_)) => return, // environment skip
                Err(e) => panic!("unexpected {scheme:?} JIT error at n={n}: {e}"),
            }
        }
    }
}

#[test]
fn non_bitnet_schemes_are_explicit_refusals() {
    // Only the three bitnet packings have a kernel; any other scheme is an explicit
    // UnsupportedScheme, never a silent misdecode (the emitter refuses before any compile).
    for scheme in [
        PackScheme::Unpacked,
        PackScheme::TwoBitPerTrit,
        PackScheme::FiveTritPerByte,
    ] {
        assert!(matches!(
            emit_bitnet_dot_ir_for(scheme),
            Err(AotError::UnsupportedScheme(_))
        ));
        assert!(matches!(
            compile_bitnet_dot_for(scheme),
            Err(AotError::UnsupportedScheme(_))
        ));
    }
}

#[test]
fn tl2_uses_the_true_167_bitstream_bound() {
    // TL2 is the true bitnet.cpp 1.67-b/w layout: 3 trits → 5 bits, bit-packed. 10 trits → 4
    // groups → 20 bits → 3 bytes (not the old 2-byte 5/byte placeholder). The kernel decodes the
    // bitstream and a too-short buffer is still an explicit refusal.
    let kernel = match compile_bitnet_dot_for(PackScheme::Tl2) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("compile failed: {e}"),
    };
    assert_eq!(kernel.scheme(), PackScheme::Tl2);
    let n = 10;
    let w = weights(n);
    let x = activations(n);
    let packed = pack_trits(&w, PackScheme::Tl2);
    assert_eq!(
        packed.len(),
        3,
        "10 trits → ⌈5·⌈10/3⌉/8⌉ = 3 bytes at 1.67 b/w"
    );
    assert_eq!(
        kernel.call(&packed, &x, n).unwrap(),
        ternary_dot_ref(&w, &x)
    );
    // Two bytes cannot hold 10 TL2 trits (need 3) → explicit refusal, never an OOB read.
    assert!(matches!(
        kernel.call(&[0u8, 0u8], &x, n),
        Err(AotError::Run(_))
    ));
}

#[test]
fn kernel_layout_records_the_inspectable_physical_layout() {
    // M-610: the layout record is the reified Meta.physical the kernel decodes — queryable,
    // EXPLAIN-able, and exactly the PhysicalLayout that travels on a result's Meta (so the E3
    // wrong-layout differential can compare it to the true packing).
    for scheme in [PackScheme::I2S, PackScheme::Tl1, PackScheme::Tl2] {
        let layout = KernelLayout::new(scheme);
        assert_eq!(layout.scheme(), scheme);
        assert_eq!(layout.physical(), PhysicalLayout::TritPacked { scheme });
        // The EXPLAIN names the scheme and the measured density — no black box (NFR-1/NFR-4).
        let ex = layout.explain();
        assert!(
            ex.contains(&format!("{scheme:?}")),
            "EXPLAIN names scheme: {ex}"
        );
        assert!(ex.contains("bits/element"), "EXPLAIN reports density: {ex}");
    }
    // The measured bits-per-element matches the codec the kernel actually decodes (pack.rs):
    // 2-bit schemes are exactly 2.0; TL2 is the true 1.67-b/w bitstream.
    assert!((KernelLayout::new(PackScheme::I2S).bits_per_element() - 2.0).abs() < 1e-9);
    assert!((KernelLayout::new(PackScheme::Tl1).bits_per_element() - 2.0).abs() < 1e-9);
    let tl2 = KernelLayout::new(PackScheme::Tl2).bits_per_element();
    assert!((1.66..=1.68).contains(&tl2), "TL2 ≈ 1.67 b/w, got {tl2}");
}

#[test]
fn compiled_kernel_reports_its_layout() {
    // The compiled kernel carries the same inspectable record as its scheme — the native
    // packed-ternary path records meta.physical, it is not hidden lowering (M-610).
    let kernel = match compile_bitnet_dot_for(PackScheme::Tl2) {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("compile failed: {e}"),
    };
    assert_eq!(kernel.layout().scheme(), PackScheme::Tl2);
    assert_eq!(
        kernel.layout().physical(),
        PhysicalLayout::TritPacked {
            scheme: PackScheme::Tl2
        }
    );
}

#[test]
fn short_buffers_are_explicit_errors() {
    // Mutant-witness: dropping the bounds checks would let the kernel read out of bounds; a short
    // buffer must be an explicit refusal, never an OOB load.
    let kernel = match compile_bitnet_dot() {
        Ok(k) => k,
        Err(AotError::ToolchainMissing(_)) => return,
        Err(e) => panic!("compile failed: {e}"),
    };
    let packed = pack_trits(&weights(8), KERNEL_SCHEME); // 2 bytes
    assert!(matches!(
        kernel.call(&packed, &[1, 2, 3], 8),
        Err(AotError::Run(_))
    )); // too few acts
    assert!(matches!(
        kernel.call(&[0u8], &activations(8), 8),
        Err(AotError::Run(_))
    )); // too few bytes
}
