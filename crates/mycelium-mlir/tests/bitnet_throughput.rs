//! M-360 / M-610 — **measured packed-ternary throughput** for the native BitNet dot kernels
//! (FR-C3/NFR-4/G3; RFC-0004 §5; phase-3.md E1 / phase-6.md M-610).
//!
//! The scalar and hand-vectorized (SIMD) kernels for all three bitnet packings (I2_S/TL1/TL2) are
//! correctness-verified by the differentials (`simd_differential.rs`, `native_packed_layout.rs`).
//! This harness reports their **measured compute throughput** over runtime-pointer buffers — the
//! genuine packed-ternary multiply-accumulate the optimiser cannot fold (the weight/activation
//! buffers are function arguments). It is the honest "throughput is measured" deliverable.
//!
//! **Honesty (VR-5/G3).** There is **no pre-written target**: the test prints elements/ns for each
//! kernel and the *measured* SIMD↔scalar ratio, and asserts only that the numbers are well-formed
//! (finite, positive) and that the compute is correct (it re-checks the dot against the oracle).
//! The guarantee on the *number* is `Empirical` (a wall-clock measurement on this machine), never a
//! claimed bound. The dot itself is exact `i64` arithmetic. The comparison baseline (the scalar
//! kernel) is stated explicitly.
//!
//! `#[ignore]` by default — it is a timing harness, not part of the fast unit gate. Run with:
//! `cargo test -p mycelium-mlir --release -- --ignored bitnet_throughput --nocapture`.

use mycelium_core::{PackScheme, Trit};
use mycelium_mlir::{
    compile_bitnet_dot_for, compile_bitnet_dot_simd, compile_bitnet_dot_simd_tl1,
    compile_bitnet_dot_simd_tl2, pack_trits, ternary_dot_ref, AotError, BitnetDotKernel,
    KernelLayout,
};

/// A SIMD-kernel compiler entry point (one per bitnet packing) — aliased to keep the harness table
/// readable (and to satisfy clippy's complex-type lint).
type SimdCompiler = fn() -> Result<BitnetDotKernel, AotError>;

fn weights(n: usize) -> Vec<Trit> {
    let mut s = 0xDEAD_BEEF_u64;
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
    let mut s = 0xCAFE_F00D_u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (((s >> 40) % 201) as i64 - 100) as i32
        })
        .collect()
}

/// Median per-call nanoseconds of `n`-element dot over `iters` calls (median is robust to scheduler
/// noise). The kernel is compiled once (compile-once/call-many — the deployment shape).
fn bench_ns(kernel: &BitnetDotKernel, packed: &[u8], x: &[i32], n: usize, iters: usize) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    // Bind once (resolve the symbol a single time, M-682) so the timed loop measures the kernel, not
    // per-iteration `dlsym`.
    let bound = kernel.bind().expect("bind kernel");
    // warm-up (page-in, branch predictor) — not measured.
    for _ in 0..8 {
        std::hint::black_box(bound.call(packed, x, n).expect("kernel runs"));
    }
    for _ in 0..iters {
        let t = std::time::Instant::now();
        std::hint::black_box(bound.call(packed, x, n).expect("kernel runs"));
        #[allow(clippy::cast_precision_loss)]
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[test]
#[ignore = "perf measurement: run with --release --ignored --nocapture"]
fn bitnet_throughput_is_measured() {
    if cfg!(debug_assertions) {
        eprintln!(
            "bitnet_throughput skip: debug build — re-run with --release for a meaningful timing \
             (`cargo test -p mycelium-mlir --release -- --ignored bitnet_throughput --nocapture`)."
        );
        return;
    }

    const N: usize = 8192; // a realistic per-row length; large enough to amortise call overhead
    const ITERS: usize = 2000;
    let x = activations(N);

    println!(
        "\nM-360/M-610 packed-ternary dot throughput — N={N} elements, {ITERS} iters (median)."
    );
    println!("Honest: measured on this machine, no pre-written target (VR-5). Dot is exact i64.");
    println!("Layout records (the inspectable meta.physical the kernel decodes):");

    // Each scheme: scalar kernel + its SIMD counterpart, both correctness-rechecked, both timed.
    let simd_compilers: [(PackScheme, SimdCompiler); 3] = [
        (PackScheme::I2S, compile_bitnet_dot_simd),
        (PackScheme::Tl1, compile_bitnet_dot_simd_tl1),
        (PackScheme::Tl2, compile_bitnet_dot_simd_tl2),
    ];

    for (scheme, simd_compile) in simd_compilers {
        let w = weights(N);
        let packed = pack_trits(&w, scheme);
        let oracle = ternary_dot_ref(&w, &x);
        let layout = KernelLayout::new(scheme);
        println!("  {}", layout.explain());

        let scalar = match compile_bitnet_dot_for(scheme) {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => {
                eprintln!("  {scheme:?}: clang absent — skipping (house idiom).");
                return;
            }
            Err(e) => panic!("{scheme:?} scalar compile failed: {e}"),
        };
        let simd = match simd_compile() {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("{scheme:?} SIMD compile failed: {e}"),
        };

        // Correctness gate: both kernels must reproduce the oracle before we trust their timing.
        assert_eq!(
            scalar.call(&packed, &x, N).unwrap(),
            oracle,
            "{scheme:?} scalar"
        );
        assert_eq!(
            simd.call(&packed, &x, N).unwrap(),
            oracle,
            "{scheme:?} SIMD"
        );

        let scalar_ns = bench_ns(&scalar, &packed, &x, N, ITERS);
        let simd_ns = bench_ns(&simd, &packed, &x, N, ITERS);
        #[allow(clippy::cast_precision_loss)]
        let elems = N as f64;
        let scalar_eps = elems / scalar_ns; // elements per ns
        let simd_eps = elems / simd_ns;
        let ratio = scalar_ns / simd_ns; // SIMD speedup over the scalar baseline (measured)

        println!("    scalar : {scalar_ns:>9.1} ns/call  ({scalar_eps:>6.3} elem/ns)   [baseline]");
        println!(
            "    SIMD   : {simd_ns:>9.1} ns/call  ({simd_eps:>6.3} elem/ns)   measured x{ratio:.2} vs scalar"
        );

        // Well-formedness of the measurement (no target asserted — VR-5): finite, positive.
        assert!(
            scalar_ns.is_finite() && scalar_ns > 0.0,
            "scalar timing well-formed"
        );
        assert!(
            simd_ns.is_finite() && simd_ns > 0.0,
            "SIMD timing well-formed"
        );
        assert!(ratio.is_finite() && ratio > 0.0, "ratio well-formed");
    }
    println!("(The ratio is whatever this machine measured; correctness is the oracle, not the clock.)\n");
}
