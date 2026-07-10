//! **Dynamic-VSA JIT execution** (M-855; E25-1; **RFC-0039 §5.3**; ADR-009 the hybrid-execution
//! decision this landing realizes for VSA — the lift is recorded append-only at RFC-0039 §6, cross-
//! referenced against ADR-009, no separate ADR-009 amendment; OQ-1 resolved 2026-06-30).
//!
//! ## What this is
//! The M-340 in-process JIT (`crate::jit`: `emit IR at runtime → clang -shared → dlopen → dlsym →
//! call`) gains a **dynamic-VSA** kernel over the same real-`Vec<f64>` **MAP-I / BSC / HRR / FHRR**
//! fragment [`crate::vsa_codegen`] lowers for the AOT path — bind/unbind/bundle/permute/similarity —
//! for workloads where the model/dimension/op are **runtime values, not Rust-compile-time constants**:
//! a data-dependent hypervector dimension, or a model chosen by a runtime branch. Every `(model, op,
//! dim)` combination a caller can construct a [`VsaProgram`] for is JIT-compiled fresh at call time
//! (specialized at JIT time, matching RFC-0039 §5.3's "data-dependent dimension… specialized at JIT
//! time, not AOT time"); there is no static enumeration to fall behind.
//!
//! ## Reuse, not a second algebra (DRY; digit-for-digit with the reference)
//! This module does **not** re-derive the VSA algebra. It reuses, verbatim, from [`crate::vsa_codegen`]
//! (M-854): [`VsaProgram`]/[`VsaCgOp`]/[`VsaModelId`]/[`VsaAotError`]/[`VsaResult`]/[`VsaExplain`] (the
//! shared program/result/error/EXPLAIN shapes), `VsaProgram::validate` (the same alphabet / dim /
//! empirical-profile / capacity side-condition checks — so this module inherits the **exact same
//! never-silent refusal surface** as the AOT path, never a second, divergent one), the pure per-op
//! trig/wrap/similarity helpers (`emit_cosine`/`emit_hamming_sim`/`emit_phase_sim`/`emit_wrap_phase`/
//! `hrr_involution`), the SSA counter, `f64_const`, `intrinsic_decls`, `mk_explain`/
//! `emit_explain_comment` (so the EXPLAIN record is textually identical between AOT and JIT), and
//! `VsaArtifact::for_shape` + its read-back methods (`reconstruct_value`/`result_meta`/`result_bound`)
//! — so the **result `Value` a JIT run stamps is built by the exact same code as the AOT path's**,
//! never a second, potentially-drifting Meta-construction. What genuinely differs is the **sink**: AOT
//! prints each component via `printf` to a subprocess's stdout and reads it back by parsing text; the
//! JIT **stores** each component's `f64` bit pattern into a caller-owned `[u64]` buffer passed by
//! pointer and reads it back directly, in-process (no subprocess, no text parsing) — the M-340 pattern
//! `jit.rs` already applies to the bit/trit subset, extended here to VSA.
//!
//! ## Never-silent selection (G2; RFC-0039 §5.3)
//! There is no heuristic AOT/JIT chooser. [`vsa_jit_compile_and_run`] is a distinctly-named entry point
//! — reachable only by a caller writing it, exactly as `crate::jit::jit_run` is the only path to the
//! bit/trit JIT and [`crate::mode::ExecMode::Jit`] is reachable only by naming that variant. (This
//! module does **not** add a `VsaProgram` arm to [`crate::mode::ExecMode`]: that enum's `run` dispatch
//! is typed over `Node`/`PrimRegistry`/`SwapEngine` — the bit/trit calculus's shape — and `VsaProgram`
//! is a structurally different input, so folding it in would force an unrelated signature onto that
//! dispatcher. The *principle* `ExecMode` embodies — a mode is reachable only by being named, never
//! inferred — is upheld here at the API level instead: [`vsa_compile_and_run`](crate::vsa_compile_and_run)
//! (AOT) and [`vsa_jit_compile_and_run`] (JIT) are two distinctly-named functions; nothing routes
//! between them automatically. FLAGged for the integrating parent: RFC-0039 §6 could additionally
//! cross-reference this API-level discipline if a future revision wants the literal `ExecMode` enum
//! widened — out of scope for this landing, which keeps `crates/mycelium-mlir/**` self-contained.)
//! Every out-of-fragment program (SBC/MAP-B, a sparse/complex carrier, an off-alphabet operand, an
//! empirical op outside its trial-validated profile, insufficient MAP-I bundle capacity) is the same
//! explicit [`VsaAotError`] refusal `vsa_codegen` already defines — reused, never re-invented.
//!
//! ## What is *not* covered (honest scope; VR-5)
//! RFC-0039 §5.3's decision text also names **cleanup / resonator loops** (RFC-0009 factorization) as
//! a dynamic-VSA JIT target. `mycelium-vsa` already has a reference implementation
//! (`resonator.rs`/`cleanup.rs`), but JIT-compiling its iterative decode (softmax cleanup, convergence
//! bookkeeping, `IterationRecord`/`StopReason` trace) is a substantially larger, distinct effort this
//! landing does **not** attempt — it is not claimed `Empirical` here, and remains explicit future work
//! (tracked for the integrating parent to file as a follow-up issue). This module covers exactly the
//! op surface named in this task's scope: bind/unbind/bundle/permute/similarity over MAP-I/BSC/HRR/
//! FHRR, dynamically specialized.
//!
//! ## Guarantee tag (VR-5 — never upgraded past the basis)
//! [`VSA_JIT_GUARANTEE`] is **`Empirical`**: the basis is the JIT≡interp/`mycelium-vsa`-reference
//! differential (`tests/vsa_jit_differential.rs`) together with the `cargo-mutants` witness on this
//! module — no proof object is linked into this codegen. This is the **codegen-correctness** claim,
//! distinct from the **value**-level tag the read-back `Value` carries (unchanged — the reference's own
//! RFC-0003 §4.1 tag, reused verbatim via `VsaArtifact::result_meta`; `Proven` only for the MAP-I
//! `bundle`'s checked capacity side-condition, never the in-progress multi-hop M-832 research).
//!
//! **Intentional unsafe (ADR-014; confined per DN-21/M-682).** This module adds **zero** new `unsafe` —
//! it reuses `crate::jit::dlopen_path`/`Lib::vsa_kernel`/`Sym::as_fn`, whose one ABI `transmute` lives
//! entirely inside `jit.rs`'s audited `Lib::get`. `#![forbid(unsafe_code)]` is not set crate-wide (the
//! crate's one `unsafe` module is `jit.rs`), but this file itself contains no `unsafe` block.

use std::fmt::Write as _;
use std::path::PathBuf;

use crate::jit::dlopen_path;
use crate::llvm::{path, run_tool, unique_tmp_dir, TmpDir};
use crate::vsa_codegen::{
    aot_to_vsa, emit_cosine, emit_explain_comment, emit_hamming_sim, emit_phase_sim,
    emit_wrap_phase, f64_const, hrr_involution, intrinsic_decls, mk_explain, Ssa, VsaAotError,
    VsaArtifact, VsaCgOp, VsaExplain, VsaModelId, VsaProgram, VsaResult,
};

use mycelium_core::GuaranteeStrength;

/// The codegen-correctness guarantee for the dynamic-VSA JIT path (M-855; RFC-0039 §5.3) —
/// **`Empirical`**, never upgraded past its checked basis (VR-5). See the module doc's "Guarantee tag"
/// section for the full basis statement.
pub const VSA_JIT_GUARANTEE: GuaranteeStrength = GuaranteeStrength::Empirical;

// ─── one computed VSA result component (mirrors the print-vs-store sink split) ──────────────────

/// One JIT-computed VSA result component: a live SSA register (`%rN`, bitcast to `i64` before the
/// store) or a value already folded to a host constant at emit time (permute; BSC's majority bit) —
/// stored directly as its bit pattern, no `bitcast` instruction needed. Mirrors exactly which values
/// `vsa_codegen`'s print sink treats as "print the constant directly"
/// (`emit_print_const_f64_bits`) vs "bitcast-then-print" (`emit_print_f64_bits`), so the JIT emits the
/// same *shape* of IR as the AOT path over the same op — only the final sink differs (store vs print).
#[derive(Debug, Clone)]
pub(crate) enum Component {
    /// A live SSA register (or, degenerately, a literal operand string) — `bitcast`-ed before storing.
    Reg(String),
    /// A host-folded constant — its bit pattern is stored directly, no `bitcast`.
    Const(f64),
}

// ─── pure per-op component computation (no sink; digit-for-digit with mycelium-vsa) ──────────────

/// Compute `bind`/`unbind`'s `dim` result components (no sink). Mirrors `vsa_codegen::emit_bind`'s
/// arithmetic exactly (same instructions, same order) — only the print call is omitted here; the
/// caller stores instead.
pub(crate) fn compute_bind(
    prog: &VsaProgram,
    inverse: bool,
    ssa: &mut Ssa,
    body: &mut String,
) -> Vec<Component> {
    let a = &prog.items[0];
    let b = &prog.items[1];
    let mut out = Vec::with_capacity(prog.dim as usize);
    match prog.model {
        // MAP-I: elementwise product (self-inverse — unbind == bind).
        VsaModelId::MapI => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let p = ssa.fresh();
                let _ = writeln!(
                    body,
                    "  {p} = fmul double {}, {}",
                    f64_const(ai),
                    f64_const(bi)
                );
                out.push(Component::Reg(p));
            }
        }
        // BSC: elementwise XOR on {0,1} == |a - b| (self-inverse).
        VsaModelId::Bsc => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let d = ssa.fresh();
                let _ = writeln!(
                    body,
                    "  {d} = fsub double {}, {}",
                    f64_const(ai),
                    f64_const(bi)
                );
                let r = ssa.fresh();
                let _ = writeln!(body, "  {r} = call double @llvm.fabs.f64(double {d})");
                out.push(Component::Reg(r));
            }
        }
        // HRR: circular convolution; unbind convolves with the involution of b.
        VsaModelId::Hrr => {
            let bv: Vec<f64> = if inverse {
                hrr_involution(b)
            } else {
                b.clone()
            };
            out.extend(
                compute_cconv(a, &bv, ssa, body)
                    .into_iter()
                    .map(Component::Reg),
            );
        }
        // FHRR: phase add (bind) / phase sub (unbind), each wrapped to (-pi, pi].
        VsaModelId::Fhrr => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let raw = ssa.fresh();
                let fop = if inverse { "fsub" } else { "fadd" };
                let _ = writeln!(
                    body,
                    "  {raw} = {fop} double {}, {}",
                    f64_const(ai),
                    f64_const(bi)
                );
                let wrapped = emit_wrap_phase(&raw, ssa, body);
                out.push(Component::Reg(wrapped));
            }
        }
    }
    out
}

/// Compute HRR circular convolution `out[k] = sum_i a[i]*b[(k+d-i) mod d]` in `f64`, accumulating
/// left-to-right exactly as `Hrr::cconv` / `vsa_codegen::emit_cconv`. Returns the `dim` SSA registers
/// (no sink).
pub(crate) fn compute_cconv(a: &[f64], b: &[f64], ssa: &mut Ssa, body: &mut String) -> Vec<String> {
    let d = a.len();
    let mut out = Vec::with_capacity(d);
    for k in 0..d {
        let mut acc = "0.0".to_owned();
        for (i, &ai) in a.iter().enumerate() {
            let bi = b[(k + d - i) % d];
            let p = ssa.fresh();
            let _ = writeln!(
                body,
                "  {p} = fmul double {}, {}",
                f64_const(ai),
                f64_const(bi)
            );
            let next = ssa.fresh();
            let _ = writeln!(body, "  {next} = fadd double {acc}, {p}");
            acc = next;
        }
        out.push(acc);
    }
    out
}

/// Compute `bundle`'s `dim` result components for **MAP-I / HRR / BSC** (no sink; FHRR's degenerate-
/// component handling needs the dedicated [`compute_fhrr_bundle`] instead, so it is refused here — the
/// caller dispatches on `prog.model` before calling this). Mirrors `vsa_codegen::emit_bundle`'s
/// non-FHRR arithmetic exactly.
pub(crate) fn compute_bundle_non_fhrr(
    prog: &VsaProgram,
    ssa: &mut Ssa,
    body: &mut String,
) -> Vec<Component> {
    let items = &prog.items;
    let dim = prog.dim as usize;
    let mut out = Vec::with_capacity(dim);
    match prog.model {
        // MAP-I / HRR: elementwise sum, accumulating left-to-right (matches the reference's `+=`).
        VsaModelId::MapI | VsaModelId::Hrr => {
            for idx in 0..dim {
                let mut acc = f64_const(items[0][idx]);
                for item in &items[1..] {
                    let next = ssa.fresh();
                    let _ = writeln!(
                        body,
                        "  {next} = fadd double {acc}, {}",
                        f64_const(item[idx])
                    );
                    acc = next;
                }
                out.push(Component::Reg(acc));
            }
        }
        // BSC: majority - count ones; > half -> 1, < half -> 0, tie -> first operand's bit. Folded
        // host-side (operands are constants and on the {0,1} alphabet), exactly mirroring
        // `Bsc::bundle` / `vsa_codegen::emit_bundle`'s BSC arm.
        VsaModelId::Bsc => {
            let half = items.len() as f64 / 2.0;
            for idx in 0..dim {
                let n: f64 = items.iter().map(|v| v[idx]).sum();
                let bit = if n > half {
                    1.0
                } else if n < half {
                    0.0
                } else {
                    items[0][idx]
                };
                out.push(Component::Const(bit));
            }
        }
        VsaModelId::Fhrr => unreachable!("caller dispatches FHRR bundle to compute_fhrr_bundle"),
    }
    out
}

/// Compute the FHRR `bundle`'s `dim` result components **and** a single OR-accumulated `i1` "any
/// component degenerate" flag register — never branching mid-loop (unlike the AOT print path's
/// per-component early-exit-on-sentinel), so the caller can emit exactly **one** branch at the end
/// (mirrors `jit.rs`'s overflow-status convention: refuse via a status code, not a mid-stream sentinel
/// token). This is **observably equivalent** to the AOT path: `Fhrr::bundle` fails the whole bundle
/// **iff any** component's phasor sum vanishes, and so does this — reading the AOT sentinel-scan
/// read-back (`VsaArtifact::parse_stdout` scans the *entire* line for the sentinel, regardless of
/// position), the two protocols agree on every case, they just signal it differently (a token in the
/// output stream vs a nonzero return code). Mirrors `Fhrr::bundle`'s algebra digit-for-digit: per
/// component, `re = sum cos(theta)`, `im = sum sin(theta)`, degenerate iff `sqrt(re^2+im^2) < 1e-9`,
/// else `wrap(atan2(im, re))` — the wrapped value is still computed even when degenerate (atan2 is
/// total; the value is simply discarded by the caller when the flag is set).
pub(crate) fn compute_fhrr_bundle(
    prog: &VsaProgram,
    ssa: &mut Ssa,
    body: &mut String,
) -> (Vec<String>, String) {
    let items = &prog.items;
    let dim = prog.dim as usize;
    let mut values = Vec::with_capacity(dim);
    let mut any_deg = String::new();
    for idx in 0..dim {
        let mut re = "0.0".to_owned();
        let mut im = "0.0".to_owned();
        for item in items {
            let theta = f64_const(item[idx]);
            let c = ssa.fresh();
            let _ = writeln!(body, "  {c} = call double @cos(double {theta})");
            let s = ssa.fresh();
            let _ = writeln!(body, "  {s} = call double @sin(double {theta})");
            let re_next = ssa.fresh();
            let _ = writeln!(body, "  {re_next} = fadd double {re}, {c}");
            let im_next = ssa.fresh();
            let _ = writeln!(body, "  {im_next} = fadd double {im}, {s}");
            re = re_next;
            im = im_next;
        }
        let re2 = ssa.fresh();
        let _ = writeln!(body, "  {re2} = fmul double {re}, {re}");
        let im2 = ssa.fresh();
        let _ = writeln!(body, "  {im2} = fmul double {im}, {im}");
        let sumsq = ssa.fresh();
        let _ = writeln!(body, "  {sumsq} = fadd double {re2}, {im2}");
        let mag = ssa.fresh();
        let _ = writeln!(body, "  {mag} = call double @llvm.sqrt.f64(double {sumsq})");
        let deg = ssa.fresh();
        let _ = writeln!(body, "  {deg} = fcmp olt double {mag}, {}", f64_const(1e-9));
        let theta = ssa.fresh();
        let _ = writeln!(
            body,
            "  {theta} = call double @atan2(double {im}, double {re})"
        );
        let wrapped = emit_wrap_phase(&theta, ssa, body);
        values.push(wrapped);
        any_deg = if idx == 0 {
            deg
        } else {
            let next = ssa.fresh();
            let _ = writeln!(body, "  {next} = or i1 {any_deg}, {deg}");
            next
        };
    }
    (values, any_deg)
}

/// Compute `permute`'s `dim` result components — a pure host-side cyclic-left-rotation fold (no SSA
/// needed; every component is the input's own exact `f64`, a coordinate bijection — matches
/// `vsa_codegen::emit_permute` exactly).
pub(crate) fn compute_permute(prog: &VsaProgram) -> Result<Vec<f64>, VsaAotError> {
    let a = &prog.items[0];
    let shift = prog
        .shift
        .ok_or_else(|| VsaAotError::Malformed("permute needs a shift".to_owned()))?;
    let d = a.len() as i64;
    Ok((0..a.len())
        .map(|i| {
            let src = (i as i64 + shift).rem_euclid(d) as usize;
            a[src]
        })
        .collect())
}

// ─── the store sink (mirrors jit.rs's `store i8 ... ; ret i32 0/1` overflow-status convention) ───

/// Emit `store i64 <bits>, ptr %out[i]` for every component, `bitcast`-ing live registers first (a
/// [`Component::Const`] embeds its bit pattern directly, matching `emit_print_const_f64_bits`'s
/// bitcast-free optimization on the AOT side).
pub(crate) fn emit_store_components(comps: &[Component], ssa: &mut Ssa, body: &mut String) {
    for (i, c) in comps.iter().enumerate() {
        let bits = match c {
            Component::Reg(r) => {
                let b = ssa.fresh();
                let _ = writeln!(body, "  {b} = bitcast double {r} to i64");
                b
            }
            Component::Const(x) => x.to_bits().to_string(),
        };
        let p = ssa.fresh();
        let _ = writeln!(body, "  {p} = getelementptr i64, ptr %out, i64 {i}");
        let _ = writeln!(body, "  store i64 {bits}, ptr {p}");
    }
}

// ─── top-level JIT kernel emission ────────────────────────────────────────────────────────────────

/// Emit the dynamic-VSA JIT kernel `myc_vsa_kernel(ptr %out) -> i32` for `prog` — one op over one
/// model, with `prog`'s `dim`/`model`/`op` read as ordinary runtime Rust values (never Rust-compile-
/// time constants), so a caller building `prog` from data (a streamed dimension, a runtime model
/// choice) gets a kernel specialized to exactly that shape. Returns the IR, the (mode-agnostic, reused
/// verbatim from `vsa_codegen`) [`VsaExplain`] record, and the `out`-buffer width (in `u64` elements).
///
/// The read-back status protocol: `0` = ok, `out[0..width)` holds the `dim` (or `1`, for a
/// measurement) IEEE-754 bit patterns; `1` = refused at runtime — **only** the FHRR degenerate-bundle
/// case reaches this (mirrors `VsaAotError::DegenerateBundleComponent`; `out` is left unwritten).
pub(crate) fn emit_vsa_jit_ir(
    prog: &VsaProgram,
) -> Result<(String, VsaExplain, usize), VsaAotError> {
    prog.validate()?;
    let explain = mk_explain(prog);

    let mut out = String::from(
        "; mycelium in-process dynamic-VSA JIT kernel (real-Vec<f64> MAP-I/BSC/HRR/FHRR; M-855; \
         RFC-0039 §5.3)\n",
    );
    emit_explain_comment(&explain, &mut out);
    let decls = intrinsic_decls(prog.model, prog.op);
    out.push_str(&decls);
    out.push_str("define i32 @myc_vsa_kernel(ptr %out) {\nentry:\n");

    let mut ssa = Ssa::new();
    let mut body = String::new();
    let width;

    match prog.op {
        VsaCgOp::Bind | VsaCgOp::Unbind => {
            let inverse = prog.op == VsaCgOp::Unbind;
            let comps = compute_bind(prog, inverse, &mut ssa, &mut body);
            width = comps.len();
            emit_store_components(&comps, &mut ssa, &mut body);
            body.push_str("  ret i32 0\n");
        }
        VsaCgOp::Bundle if prog.model == VsaModelId::Fhrr => {
            let (values, any_deg) = compute_fhrr_bundle(prog, &mut ssa, &mut body);
            width = values.len();
            let deg_lbl = ssa.fresh_label();
            let ok_lbl = ssa.fresh_label();
            let _ = writeln!(body, "  br i1 {any_deg}, label %{deg_lbl}, label %{ok_lbl}");
            let _ = writeln!(body, "{deg_lbl}:");
            body.push_str("  ret i32 1\n");
            let _ = writeln!(body, "{ok_lbl}:");
            let comps: Vec<Component> = values.into_iter().map(Component::Reg).collect();
            emit_store_components(&comps, &mut ssa, &mut body);
            body.push_str("  ret i32 0\n");
        }
        VsaCgOp::Bundle => {
            let comps = compute_bundle_non_fhrr(prog, &mut ssa, &mut body);
            width = comps.len();
            emit_store_components(&comps, &mut ssa, &mut body);
            body.push_str("  ret i32 0\n");
        }
        VsaCgOp::Permute => {
            let vals = compute_permute(prog)?;
            width = vals.len();
            let comps: Vec<Component> = vals.into_iter().map(Component::Const).collect();
            emit_store_components(&comps, &mut ssa, &mut body);
            body.push_str("  ret i32 0\n");
        }
        VsaCgOp::Similarity => {
            let a = &prog.items[0];
            let b = &prog.items[1];
            let sim = match prog.model {
                VsaModelId::MapI | VsaModelId::Hrr => emit_cosine(a, b, &mut ssa, &mut body),
                VsaModelId::Bsc => emit_hamming_sim(a, b, &mut ssa, &mut body),
                VsaModelId::Fhrr => emit_phase_sim(a, b, &mut ssa, &mut body),
            };
            width = 1;
            emit_store_components(&[Component::Reg(sim)], &mut ssa, &mut body);
            body.push_str("  ret i32 0\n");
        }
    }
    out.push_str(&body);
    out.push_str("}\n");
    Ok((out, explain, width))
}

// ─── compile / call (dlopen; the M-340 in-process pattern applied to the VSA kernel) ─────────────

/// A JIT-compiled dynamic-VSA kernel: the `.so` on disk (per-artifact temp dir, cleaned on drop) plus
/// the read-back shape. Produced by [`vsa_jit_compile`]; call any number of times in-process with
/// [`VsaJitArtifact::call`].
pub struct VsaJitArtifact {
    _dir: TmpDir,
    so: PathBuf,
    op: VsaCgOp,
    model: VsaModelId,
    dim: u32,
    bundle_delta: Option<f64>,
    item_count: u64,
    width: usize,
}

impl VsaJitArtifact {
    /// Call the kernel in-process (`dlopen` -> `dlsym` -> call) and read the result back, applying the
    /// never-silent status protocol (G2): a nonzero status is the FHRR degenerate-bundle refusal,
    /// **never** a silently-returned garbage buffer. On success, reconstructs the result exactly as the
    /// AOT path would (`VsaArtifact::for_shape` + its reused read-back methods) — so a JIT `Value`
    /// carries the identical `Meta`/guarantee/bound the AOT artifact would for the same program.
    pub fn call(&self) -> Result<VsaResult, VsaAotError> {
        let lib = dlopen_path(&self.so).map_err(aot_to_vsa)?; // dlclose on drop
        let kernel = lib.vsa_kernel().map_err(aot_to_vsa)?;
        let mut buf = vec![0u64; self.width];
        let status = (kernel.as_fn())(buf.as_mut_ptr());
        let shape = VsaArtifact::for_shape(
            self.op,
            self.model,
            self.dim,
            self.bundle_delta,
            self.item_count,
        );
        decode_jit_result(self.op, &shape, status, &buf)
    }
}

/// Decode the JIT kernel's raw `(status, out-buffer)` pair into a [`VsaResult`] — the never-silent
/// read-back protocol (`status != 0` => [`VsaAotError::DegenerateBundleComponent`], mirroring the AOT
/// sentinel), value-op reconstruction delegated to `shape.reconstruct_value` (identical to the AOT
/// path). `pub(crate)` so this is **witnessable without `clang`/`dlopen`** (the M-854 toolchain-
/// independent mutant-witness lesson): a test can call it directly with a synthetic status/buffer.
pub(crate) fn decode_jit_result(
    op: VsaCgOp,
    shape: &VsaArtifact,
    status: i32,
    buf: &[u64],
) -> Result<VsaResult, VsaAotError> {
    if status != 0 {
        return Err(VsaAotError::DegenerateBundleComponent);
    }
    if op.is_value_op() {
        shape.reconstruct_value(buf)
    } else if buf.len() != 1 {
        Err(VsaAotError::Parse(format!(
            "measurement expected 1 element, got {}",
            buf.len()
        )))
    } else {
        Ok(VsaResult::Measurement(f64::from_bits(buf[0])))
    }
}

/// Compile a dynamic-VSA program to an in-process JIT kernel (emit IR -> `clang -shared` -> ready to
/// `dlopen`) without calling it. Returns [`VsaAotError::ToolchainMissing`] when `clang` is absent
/// (callers skip, the house idiom); any out-of-fragment program is the same explicit refusal
/// [`emit_vsa_jit_ir`] returns (reused from `vsa_codegen::VsaProgram::validate`, never a second
/// refusal surface).
pub fn vsa_jit_compile(prog: &VsaProgram) -> Result<VsaJitArtifact, VsaAotError> {
    let (ir, _explain, width) = emit_vsa_jit_ir(prog)?;
    let dir = unique_tmp_dir().map_err(aot_to_vsa)?;
    let ll = dir.join("vsa_jit.ll");
    let so = dir.join("vsa_jit.so");
    let guard = TmpDir(dir);
    std::fs::write(&ll, ir.as_bytes()).map_err(|e| VsaAotError::Run(format!("write IR: {e}")))?;
    run_tool(
        "clang",
        &[
            "-shared",
            "-fPIC",
            "-x",
            "ir",
            path(&ll).map_err(aot_to_vsa)?,
            "-o",
            path(&so).map_err(aot_to_vsa)?,
            "-lm",
        ],
    )
    .map_err(aot_to_vsa)?;
    Ok(VsaJitArtifact {
        _dir: guard,
        so,
        op: prog.op,
        model: prog.model,
        dim: prog.dim,
        bundle_delta: prog.bundle_delta,
        item_count: prog.items.len() as u64,
        width,
    })
}

/// Compile + call a dynamic-VSA program in-process: the JIT execution path the M-855 differential
/// checks against the interpreter/`mycelium-vsa` reference — the **only** entry point to the dynamic-
/// VSA JIT (never-silent selection, G2: a caller reaches it exclusively by writing this name, exactly
/// as `crate::jit::jit_run` is the only path to the bit/trit JIT).
pub fn vsa_jit_compile_and_run(prog: &VsaProgram) -> Result<VsaResult, VsaAotError> {
    vsa_jit_compile(prog)?.call()
}
