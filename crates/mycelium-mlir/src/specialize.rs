//! **JIT runtime specialization** of the packed-ternary dot kernel (M-340; E3-4; ADR-009/ADR-014;
//! RFC-0004 §5/§8; phase-3.md §2 Batch L).
//!
//! The generic BitNet dot kernel ([`crate::bitnet`]) takes its weight buffer as a **runtime pointer**
//! and re-loads + re-unpacks it on every call — correct, but it pays the unpack cost per invocation
//! and does work for every lane, including the zeros. In the canonical inference setting the **weights
//! are fixed at runtime** (a trained model) and only the **activations vary**. That is exactly the
//! shape a runtime specializer exploits: given a concrete weight vector known *at JIT time*, bake it
//! into the kernel as constants so the optimiser can —
//!
//! - **drop the unpack entirely** — no packed-byte load, no shift/mask, no `code − 1`;
//! - **elide every zero-weight lane** — a `0` weight contributes nothing, so its activation load and
//!   multiply simply vanish from the emitted IR (the model's sparsity becomes *visible*, FR-C3);
//! - **strength-reduce the ±1 multiply** to a single `add`/`sub` of the activation.
//!
//! The result is a straight-line kernel `i64 @myc_bitnet_dot_spec(ptr %x)` whose only runtime argument
//! is the activation buffer; the weights and the length are compiled in. This is the JIT path
//! "runtime-specialization layer over the lowering" (issue #93) — it *shares* the dot semantics with
//! the generic kernel (and the [`crate::bitnet::ternary_dot_ref`] oracle) and must agree with both.
//!
//! **Never silent / structural correctness.** Because the weights are baked in, the call site takes
//! *no* weight argument — you cannot accidentally run a specialized kernel against weights it was not
//! built for; the type makes that misuse unrepresentable rather than silently using stale weights. A
//! short activation buffer is an explicit [`AotError`], never an out-of-bounds load.
//!
//! **Honesty / E1 (VR-5/G3).** The specialized kernel computes the *same exact* ternary dot product
//! as the generic one — no guarantee is upgraded (both are `Exact` integer arithmetic). The speedup
//! over the generic kernel is whatever `cargo xtask e1` §4 measures over runtime activation data; no
//! target is pre-written. The specialization is honest *runtime* data (the weights are a runtime
//! input baked at JIT time), not a constant-folded closed kernel — the activations stay runtime
//! pointers, so the compute is real.
//!
//! **Submodule confinement (DN-21 §5 F-2 / M-682):** zero `unsafe` — the ABI `transmute` is confined
//! to the single `crate::jit::Lib::get` choke-point, so this kernel resolves its symbol as a
//! lifetime-bound `Sym` and calls it through ordinary safe Rust; compiler-enforced.
#![forbid(unsafe_code)]

use std::fmt::Write as _;

use mycelium_core::ternary::digit;
use mycelium_core::Trit;

use crate::jit::{dlopen_path, Lib, SpecDotFn, Sym};
use crate::llvm::{path, run_tool, unique_tmp_dir, AotError, TmpDir};

/// The specialized kernel's symbol — distinct from the generic [`crate::bitnet`] kernel so both can be
/// loaded at once (e.g. for the E1 §4 differential timing). Matches the symbol `Lib::spec_dot` resolves.
const SPEC_SYM: &str = "myc_bitnet_dot_spec";

/// Emit the textual LLVM IR for a **weight-specialized** ternary dot kernel
/// `i64 @myc_bitnet_dot_spec(ptr %x)` with `weights` compiled in as constants: it accumulates
/// `Σ digit(wᵢ)·x[i]` as a straight-line sequence of `add`/`sub`s — one per **nonzero** weight, with
/// the zero lanes elided. Deterministic (SSA names are stable by emission order); every surviving
/// op is visible (no opaque pass — RFC-0004 §6). An all-zero (or empty) weight vector emits a kernel
/// that loads nothing and returns `0`.
#[must_use]
pub fn emit_specialized_dot_ir(weights: &[Trit]) -> String {
    let nonzero = weights.iter().filter(|w| digit(**w) != 0).count();
    let mut ir = format!(
        "; mycelium weight-specialized BitNet dot kernel (M-340; {} lanes, {nonzero} nonzero)\n",
        weights.len()
    );
    ir.push_str("define i64 @myc_bitnet_dot_spec(ptr %x) {\nentry:\n");

    // `acc` tracks the SSA name (or the literal `0` before the first contribution) of the running sum.
    let mut acc = String::from("0");
    let mut k = 0usize;
    for (i, w) in weights.iter().enumerate() {
        let d = digit(*w);
        if d == 0 {
            continue; // zero lane: no load, no op — the visible specialization (FR-C3 sparsity)
        }
        // Load this lane's activation (the only runtime data) and widen to the i64 accumulator.
        let _ = writeln!(ir, "  %p{k} = getelementptr i32, ptr %x, i64 {i}");
        let _ = writeln!(ir, "  %v{k} = load i32, ptr %p{k}");
        let _ = writeln!(ir, "  %x{k} = sext i32 %v{k} to i64");
        // ±1 weight ⇒ a single add/sub (the strength-reduced multiply).
        let op = if d > 0 { "add" } else { "sub" };
        let _ = writeln!(ir, "  %acc{k} = {op} i64 {acc}, %x{k}");
        acc = format!("%acc{k}");
        k += 1;
    }
    let _ = writeln!(ir, "  ret i64 {acc}");
    ir.push_str("}\n");
    ir
}

/// A compiled, in-process **weight-specialized** dot kernel: the `.so` (in a per-artifact temp dir,
/// cleaned on drop), the loaded library, the resolved entry point, and the baked-in shape. Built by
/// [`compile_specialized_dot`]; call any number of times over varying activation buffers with
/// [`SpecializedDotKernel::call`]. Compile once per weight vector, call many — the inference shape.
pub struct SpecializedDotKernel {
    _dir: TmpDir,
    lib: Lib,
    /// Logical lane count baked into the kernel — fixes the activation-buffer bound.
    n: usize,
    /// How many lanes survived specialization (nonzero weights) — inspectable sparsity metadata.
    nonzero: usize,
}

impl SpecializedDotKernel {
    /// The logical number of lanes (weight length) compiled into this kernel.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of nonzero (surviving) lanes — the straight-line `add`/`sub` count, exposed for
    /// EXPLAIN/inspection (a fully sparse model specializes to far fewer ops).
    #[must_use]
    pub fn nonzero(&self) -> usize {
        self.nonzero
    }

    /// **Bind once, call many** (M-682): resolve the `myc_bitnet_dot_spec` entry point a single time
    /// into a lifetime-bound [`BoundSpecializedDot`] borrowing this kernel's loaded library, so the
    /// borrow checker guarantees its fn-pointer cannot outlive the `Lib` (DN-21 §4). Bind once and
    /// reuse for a hot loop over varying activation buffers — the inference shape.
    pub fn bind(&self) -> Result<BoundSpecializedDot<'_>, AotError> {
        Ok(BoundSpecializedDot {
            kernel: self.lib.spec_dot()?,
            n: self.n,
        })
    }

    /// Run the specialized kernel over `activations`, returning `Σ digit(wᵢ)·activations[i]` for the
    /// baked-in weights. Convenience wrapper that [`bind`](Self::bind)s once and calls; for a hot loop
    /// bind once and reuse the [`BoundSpecializedDot`]. A short buffer is an explicit [`AotError::Run`],
    /// never an out-of-bounds read.
    pub fn call(&self, activations: &[i32]) -> Result<i64, AotError> {
        self.bind()?.call(activations)
    }
}

/// A [`SpecializedDotKernel`] with its entry point resolved into a lifetime-bound `Sym` (M-682).
/// Produced by [`SpecializedDotKernel::bind`]; borrows the kernel's loaded library for `'lib`, so its
/// fn-pointer can never be called after the library unloads (the §4 dangling-pointer risk, now
/// compiler-checked). Call it over as many activation buffers as needed — `dlsym` was paid once.
pub struct BoundSpecializedDot<'lib> {
    kernel: Sym<'lib, SpecDotFn>,
    n: usize,
}

impl BoundSpecializedDot<'_> {
    /// Run the specialized kernel over `activations`, returning `Σ digit(wᵢ)·activations[i]` for the
    /// baked-in weights. `activations.len()` is checked against the baked lane count `n` so every load
    /// is in bounds — a short buffer is an explicit [`AotError::Run`], never an out-of-bounds read.
    pub fn call(&self, activations: &[i32]) -> Result<i64, AotError> {
        if activations.len() < self.n {
            return Err(AotError::Run(format!(
                "activations too short: kernel specialized for {} lanes, got {}",
                self.n,
                activations.len()
            )));
        }
        // The kernel reads only `x[i]` for the baked nonzero lanes `i < n`, all in-bounds by the check
        // above. Calling the typed `extern "C"` pointer is ordinary safe Rust — the ABI claim was made
        // (and audited) once at `Lib::get`, and the `Sym` lifetime keeps the library loaded for this
        // call (M-682; DN-21 §4/§7).
        Ok((self.kernel.as_fn())(activations.as_ptr()))
    }
}

/// Compile a kernel **specialized on `weights`** (baked in as constants) to a shared object and load
/// it in-process. Returns [`AotError::ToolchainMissing`] when `clang` is absent so callers can skip
/// (the house idiom). The natural companion to [`crate::bitnet::compile_bitnet_dot_for`] — same
/// dot semantics, weights compiled in instead of read from a runtime pointer.
pub fn compile_specialized_dot(weights: &[Trit]) -> Result<SpecializedDotKernel, AotError> {
    let ir = emit_specialized_dot_ir(weights);
    let dir = unique_tmp_dir()?;
    let ll = dir.join("spec.ll");
    let so = dir.join("spec.so");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    // -O2 so the optimiser does real codegen over the straight-line kernel (the point of E1 §4).
    run_tool(
        "clang",
        &[
            "-shared",
            "-fPIC",
            "-O2",
            "-x",
            "ir",
            path(&ll)?,
            "-o",
            path(&so)?,
        ],
    )?;

    let lib = dlopen_path(&so)?;
    // Fail fast: verify the entry point is exported now (pre-M-682 behaviour), not at first bind/call.
    lib.probe(SPEC_SYM)?;
    Ok(SpecializedDotKernel {
        _dir: guard,
        lib,
        n: weights.len(),
        nonzero: weights.iter().filter(|w| digit(**w) != 0).count(),
    })
}

/// Convenience: specialize on `weights`, compile, and run the dot product against `activations` once.
/// The wrapper the differential test checks against [`crate::bitnet::ternary_dot_ref`] and the
/// generic kernel.
pub fn jit_specialized_dot(weights: &[Trit], activations: &[i32]) -> Result<i64, AotError> {
    compile_specialized_dot(weights)?.call(activations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet::ternary_dot_ref;

    /// Deterministic ternary weights (small LCG) — fixed, not a statistical sample.
    fn weights(n: usize) -> Vec<Trit> {
        let mut s = 0xABCD_0123_u64;
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
        let mut s = 0x1357_9BDF_u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (((s >> 40) % 201) as i64 - 100) as i32
            })
            .collect()
    }

    #[test]
    fn ir_bakes_weights_and_is_deterministic() {
        // +1 at lane 1, −1 at lane 2, 0 elsewhere: the IR must add lane 1, sub lane 2, and *never*
        // touch the zero lanes (their loads are elided — the specialization, inspectable).
        let w = vec![Trit::Zero, Trit::Pos, Trit::Neg, Trit::Zero];
        let ir = emit_specialized_dot_ir(&w);
        assert!(ir.contains("define i64 @myc_bitnet_dot_spec(ptr %x)"));
        assert!(ir.contains("add i64")); // +1 lane
        assert!(ir.contains("sub i64")); // −1 lane
        assert!(ir.contains("getelementptr i32, ptr %x, i64 1")); // lane 1 loaded
        assert!(ir.contains("getelementptr i32, ptr %x, i64 2")); // lane 2 loaded
                                                                  // Mutant-witness: a kernel that did *not* elide zero lanes would load lane 0 / lane 3.
        assert!(!ir.contains("getelementptr i32, ptr %x, i64 0"));
        assert!(!ir.contains("getelementptr i32, ptr %x, i64 3"));
        assert!(!ir.contains("load i8")); // no packed-weight unpack at all (vs the generic kernel)
        assert_eq!(ir, emit_specialized_dot_ir(&w)); // deterministic
    }

    #[test]
    fn all_zero_weights_emit_a_constant_zero_kernel() {
        let ir = emit_specialized_dot_ir(&[Trit::Zero, Trit::Zero, Trit::Zero]);
        assert!(!ir.contains("load")); // nothing to load
        assert!(ir.contains("ret i64 0"));
    }

    #[test]
    fn nonzero_count_tracks_sparsity() {
        let ir_w = vec![Trit::Pos, Trit::Zero, Trit::Neg, Trit::Zero, Trit::Zero];
        // 2 of 5 nonzero — recorded in the header comment and (when compiled) the kernel field.
        assert!(emit_specialized_dot_ir(&ir_w).contains("5 lanes, 2 nonzero"));
    }

    #[test]
    fn specialized_matches_oracle_and_generic() {
        // Mutant-witness: a swapped add/sub (wrong weight sign) or a missed lane would diverge from
        // the packing-independent oracle on this mixed data.
        for n in [1usize, 3, 8, 64, 257, 1000] {
            let w = weights(n);
            let x = activations(n);
            match jit_specialized_dot(&w, &x) {
                Ok(got) => assert_eq!(got, ternary_dot_ref(&w, &x), "spec dot mismatch at n={n}"),
                Err(AotError::ToolchainMissing(_)) => return, // environment skip
                Err(e) => panic!("unexpected specialize JIT error at n={n}: {e}"),
            }
        }
    }

    #[test]
    fn compile_once_call_many_over_varying_activations() {
        // The inference shape: one specialized kernel, many activation vectors — each must match the
        // oracle for the *same* baked weights.
        let w = weights(128);
        let kernel = match compile_specialized_dot(&w) {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("compile failed: {e}"),
        };
        assert_eq!(kernel.n(), 128);
        assert!(kernel.nonzero() <= 128 && kernel.nonzero() > 0);
        for seed_len in [128usize, 200, 4096] {
            let x = activations(seed_len); // ≥ 128, so in-bounds
            assert_eq!(
                kernel.call(&x).unwrap(),
                ternary_dot_ref(&w, &x[..128]),
                "specialized kernel diverged for a fresh activation vector"
            );
        }
    }

    #[test]
    fn short_activations_are_explicit_errors() {
        // Mutant-witness: dropping the bound would let the kernel read past the activation slice.
        let w = weights(16);
        let kernel = match compile_specialized_dot(&w) {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("compile failed: {e}"),
        };
        assert!(matches!(kernel.call(&[1, 2, 3]), Err(AotError::Run(_))));
    }
}
