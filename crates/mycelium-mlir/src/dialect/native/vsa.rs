//! MLIR-dialect-native sibling of [`crate::vsa_codegen`] (M-856b; RFC-0039 §5.2; epic E25-1).
//!
//! Lowers the **same** [`VsaProgram`] [`crate::vsa_codegen`] compiles to direct LLVM IR text, to a
//! genuine `arith`/`func`/`math` MLIR module (plus the `llvm` dialect for the `printf` read-back
//! plumbing — see [`super::dense`]'s module doc for the rationale and the validated
//! `arith.remf`-vs-`llvm.frem` semantics pitfall, which this module also avoids for the FHRR phase
//! wrap). Covers the four 1.0.0-native-mandatory models — **MAP-I, BSC, HRR, FHRR** — over
//! `bind`/`unbind`/`bundle`/`permute`/`similarity`, mirroring [`crate::vsa_codegen`]'s algorithm
//! digit-for-digit (same operation order, same `f64` accumulation, same never-silent
//! `DEGENERATE`/alphabet/capacity refusals). [`VsaProgram::validate`] and the [`VsaArtifact`]
//! read-back/reconstruct logic are the **same** code the direct-LLVM path uses (`pub(crate)` for
//! exactly this reuse) — the two compiled paths can never silently disagree on what they accept or
//! how they stamp a result `Value` (DRY; VR-5).
//!
//! **Host-folded ops stay host-folded.** `permute` and the BSC majority `bundle` bit are computed
//! host-side in [`crate::vsa_codegen`] too (the operands are compile-time constants) — this module
//! keeps that fold and emits the already-known result as a bit-pattern constant, exactly as the
//! direct-LLVM path does; no runtime IR needlessly recomputes a host-decidable value.
//!
//! **libMLIR-gated (ADR-019)** — [`VsaAotError::ToolchainMissing`] on an absent toolchain (skip,
//! never a faked pass).
//!
//! **Guarantee tag:** `Empirical` — differential-checked; never `Proven` absent a checked
//! equivalence proof (VR-5). **Zero `unsafe`** (inherits `#![forbid(unsafe_code)]`).

use std::fmt::Write as _;
use std::process::Command;

use crate::llvm::{path, unique_tmp_dir, TmpDir};
use crate::vsa_codegen::{
    hrr_involution, VsaAotError, VsaArtifact, VsaCgOp, VsaModelId, VsaProgram, VsaResult,
    VSA_DEGENERATE_SENTINEL,
};

use super::DialectError;

fn dialect_err_to_vsa(e: DialectError) -> VsaAotError {
    match e {
        DialectError::ToolchainMissing(t) => VsaAotError::ToolchainMissing(t),
        DialectError::Compile(s) => VsaAotError::Compile(s),
        DialectError::Run(s) => VsaAotError::Run(s),
        DialectError::Parse(s) => VsaAotError::Parse(s),
        DialectError::Wf(s) => VsaAotError::Wf(s),
        other => VsaAotError::Run(other.to_string()),
    }
}

// ─── SSA / block-label counters ───────────────────────────────────────────────────────────────

struct Ssa(usize);
impl Ssa {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("%w{n}")
    }
}

struct Bbc(usize);
impl Bbc {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("vbb{n}")
    }
}

/// Render an `f64` as an exact MLIR `f64` hex bit-pattern constant (verified bit-exact — see
/// [`super::dense`]'s module doc).
fn mlir_f64_const(x: f64) -> String {
    format!("0x{:016X}", x.to_bits())
}

// ─── printf-based read-back (mirrors `super::dense`'s I/O helpers exactly) ───────────────────────

fn emit_print_f64_bits(d: &str, ssa: &mut Ssa, out: &mut String) {
    let bits = ssa.fresh();
    let _ = writeln!(out, "    {bits} = arith.bitcast {d} : f64 to i64");
    let fmt = ssa.fresh();
    let _ = writeln!(out, "    {fmt} = llvm.mlir.addressof @fmt_u64 : !llvm.ptr");
    let p = ssa.fresh();
    let _ = writeln!(
        out,
        "    {p} = llvm.call @printf({fmt}, {bits}) vararg(!llvm.func<i32 (ptr, ...)>) : \
         (!llvm.ptr, i64) -> i32"
    );
}

/// Print a *constant* `f64`'s bit pattern directly (for host-folded ops — permute, the BSC majority
/// bit) — no runtime arithmetic needed, mirrors `vsa_codegen::emit_print_const_f64_bits`.
fn emit_print_const_f64_bits(x: f64, ssa: &mut Ssa, out: &mut String) {
    let c = ssa.fresh();
    let _ = writeln!(out, "    {c} = arith.constant {} : f64", mlir_f64_const(x));
    emit_print_f64_bits(&c, ssa, out);
}

fn emit_print_sentinel(global: &str, ssa: &mut Ssa, out: &mut String) {
    let fmt = ssa.fresh();
    let _ = writeln!(out, "    {fmt} = llvm.mlir.addressof @{global} : !llvm.ptr");
    let p = ssa.fresh();
    let _ = writeln!(
        out,
        "    {p} = llvm.call @printf({fmt}) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32"
    );
}

fn emit_newline(ssa: &mut Ssa, out: &mut String) {
    let fmt = ssa.fresh();
    let _ = writeln!(out, "    {fmt} = llvm.mlir.addressof @fmt_nl : !llvm.ptr");
    let p = ssa.fresh();
    let _ = writeln!(
        out,
        "    {p} = llvm.call @printf({fmt}) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32"
    );
}

// ─── bind / unbind (mirrors `vsa_codegen::emit_bind` per model) ──────────────────────────────────

fn emit_bind(
    prog: &VsaProgram,
    inverse: bool,
    ssa: &mut Ssa,
    out: &mut String,
) -> Result<(), VsaAotError> {
    let a = &prog.items[0];
    let b = &prog.items[1];
    match prog.model {
        // MAP-I: elementwise product (self-inverse — unbind == bind).
        VsaModelId::MapI => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let av = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {av} = arith.constant {} : f64",
                    mlir_f64_const(ai)
                );
                let bv = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {bv} = arith.constant {} : f64",
                    mlir_f64_const(bi)
                );
                let p = ssa.fresh();
                let _ = writeln!(out, "    {p} = arith.mulf {av}, {bv} : f64");
                emit_print_f64_bits(&p, ssa, out);
            }
        }
        // BSC: elementwise XOR on {0,1} == |a − b| (self-inverse).
        VsaModelId::Bsc => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let av = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {av} = arith.constant {} : f64",
                    mlir_f64_const(ai)
                );
                let bv = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {bv} = arith.constant {} : f64",
                    mlir_f64_const(bi)
                );
                let d = ssa.fresh();
                let _ = writeln!(out, "    {d} = arith.subf {av}, {bv} : f64");
                let r = ssa.fresh();
                let _ = writeln!(out, "    {r} = math.absf {d} : f64");
                emit_print_f64_bits(&r, ssa, out);
            }
        }
        // HRR: circular convolution; unbind convolves with the involution of b.
        VsaModelId::Hrr => {
            let bv: Vec<f64> = if inverse {
                hrr_involution(b)
            } else {
                b.clone()
            };
            emit_cconv(a, &bv, ssa, out);
        }
        // FHRR: phase add (bind) / phase sub (unbind), each wrapped to (−π, π].
        VsaModelId::Fhrr => {
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let av = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {av} = arith.constant {} : f64",
                    mlir_f64_const(ai)
                );
                let bv = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {bv} = arith.constant {} : f64",
                    mlir_f64_const(bi)
                );
                let raw = ssa.fresh();
                let fop = if inverse { "arith.subf" } else { "arith.addf" };
                let _ = writeln!(out, "    {raw} = {fop} {av}, {bv} : f64");
                let wrapped = emit_wrap_phase(&raw, ssa, out);
                emit_print_f64_bits(&wrapped, ssa, out);
            }
        }
    }
    emit_newline(ssa, out);
    Ok(())
}

/// Emit HRR circular convolution `out[k] = Σᵢ a[i]·b[(k+d−i) mod d]` in `f64`, accumulating
/// left-to-right — mirrors `vsa_codegen::emit_cconv` exactly (the reference's naive `O(d²)` form,
/// unrolled at emit time since the operands are compile-time constants).
fn emit_cconv(a: &[f64], b: &[f64], ssa: &mut Ssa, out: &mut String) {
    let d = a.len();
    for k in 0..d {
        let mut acc = ssa.fresh();
        let _ = writeln!(out, "    {acc} = arith.constant 0.0 : f64");
        for (i, &ai) in a.iter().enumerate() {
            let bi = b[(k + d - i) % d];
            let av = ssa.fresh();
            let _ = writeln!(
                out,
                "    {av} = arith.constant {} : f64",
                mlir_f64_const(ai)
            );
            let bv = ssa.fresh();
            let _ = writeln!(
                out,
                "    {bv} = arith.constant {} : f64",
                mlir_f64_const(bi)
            );
            let p = ssa.fresh();
            let _ = writeln!(out, "    {p} = arith.mulf {av}, {bv} : f64");
            let next = ssa.fresh();
            let _ = writeln!(out, "    {next} = arith.addf {acc}, {p} : f64");
            acc = next;
        }
        emit_print_f64_bits(&acc, ssa, out);
    }
}

// ─── bundle (mirrors `vsa_codegen::emit_bundle`/`emit_fhrr_bundle`) ──────────────────────────────

fn emit_bundle(
    prog: &VsaProgram,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    out: &mut String,
) -> Result<(), VsaAotError> {
    let items = &prog.items;
    let dim = prog.dim as usize;
    match prog.model {
        // MAP-I / HRR: elementwise sum, accumulating left-to-right.
        VsaModelId::MapI | VsaModelId::Hrr => {
            for idx in 0..dim {
                let mut acc = ssa.fresh();
                let _ = writeln!(
                    out,
                    "    {acc} = arith.constant {} : f64",
                    mlir_f64_const(items[0][idx])
                );
                for item in &items[1..] {
                    let iv = ssa.fresh();
                    let _ = writeln!(
                        out,
                        "    {iv} = arith.constant {} : f64",
                        mlir_f64_const(item[idx])
                    );
                    let next = ssa.fresh();
                    let _ = writeln!(out, "    {next} = arith.addf {acc}, {iv} : f64");
                    acc = next;
                }
                emit_print_f64_bits(&acc, ssa, out);
            }
        }
        // BSC: majority vote, folded host-side exactly as `vsa_codegen::emit_bundle` does (the
        // operands are constants on the {0,1} alphabet) — emitted as a bit-pattern constant.
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
                emit_print_const_f64_bits(bit, ssa, out);
            }
        }
        // FHRR: per-component complex-sum-renormalized phasor.
        VsaModelId::Fhrr => {
            emit_fhrr_bundle(items, dim, ssa, bbc, out);
        }
    }
    emit_newline(ssa, out);
    Ok(())
}

/// Emit the FHRR bundle: per component `re = Σ cos θ`, `im = Σ sin θ`; a never-silent `DEGENERATE`
/// sentinel branch when `√(re²+im²) < 1e-9`, else `wrap(atan2(im, re))`. Mirrors
/// `vsa_codegen::emit_fhrr_bundle` exactly, using `math.cos`/`math.sin`/`math.atan2`/`math.sqrt`
/// (all real MLIR `math`-dialect ops, verified against `mlir-opt-18` — see the module doc).
fn emit_fhrr_bundle(
    items: &[Vec<f64>],
    dim: usize,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    out: &mut String,
) {
    for idx in 0..dim {
        let mut re = ssa.fresh();
        let _ = writeln!(out, "    {re} = arith.constant 0.0 : f64");
        let mut im = ssa.fresh();
        let _ = writeln!(out, "    {im} = arith.constant 0.0 : f64");
        for item in items {
            let theta = ssa.fresh();
            let _ = writeln!(
                out,
                "    {theta} = arith.constant {} : f64",
                mlir_f64_const(item[idx])
            );
            let c = ssa.fresh();
            let _ = writeln!(out, "    {c} = math.cos {theta} : f64");
            let s = ssa.fresh();
            let _ = writeln!(out, "    {s} = math.sin {theta} : f64");
            let re_next = ssa.fresh();
            let _ = writeln!(out, "    {re_next} = arith.addf {re}, {c} : f64");
            let im_next = ssa.fresh();
            let _ = writeln!(out, "    {im_next} = arith.addf {im}, {s} : f64");
            re = re_next;
            im = im_next;
        }
        let re2 = ssa.fresh();
        let _ = writeln!(out, "    {re2} = arith.mulf {re}, {re} : f64");
        let im2 = ssa.fresh();
        let _ = writeln!(out, "    {im2} = arith.mulf {im}, {im} : f64");
        let sumsq = ssa.fresh();
        let _ = writeln!(out, "    {sumsq} = arith.addf {re2}, {im2} : f64");
        let mag = ssa.fresh();
        let _ = writeln!(out, "    {mag} = math.sqrt {sumsq} : f64");
        let thresh = ssa.fresh();
        let _ = writeln!(
            out,
            "    {thresh} = arith.constant {} : f64",
            mlir_f64_const(1e-9)
        );
        let deg = ssa.fresh();
        let _ = writeln!(out, "    {deg} = arith.cmpf olt, {mag}, {thresh} : f64");
        let deg_lbl = bbc.fresh();
        let ok_lbl = bbc.fresh();
        let _ = writeln!(out, "    cf.cond_br {deg}, ^{deg_lbl}, ^{ok_lbl}");
        let _ = writeln!(out, "  ^{deg_lbl}:");
        emit_print_sentinel("s_deg", ssa, out);
        let z = ssa.fresh();
        let _ = writeln!(out, "    {z} = arith.constant 0 : i32");
        let _ = writeln!(out, "    func.return {z} : i32");
        let _ = writeln!(out, "  ^{ok_lbl}:");
        let theta = ssa.fresh();
        let _ = writeln!(out, "    {theta} = math.atan2 {im}, {re} : f64");
        let wrapped = emit_wrap_phase(&theta, ssa, out);
        emit_print_f64_bits(&wrapped, ssa, out);
    }
}

// ─── permute (host-folded — mirrors `vsa_codegen::emit_permute`) ────────────────────────────────

fn emit_permute(prog: &VsaProgram, ssa: &mut Ssa, out: &mut String) -> Result<(), VsaAotError> {
    let a = &prog.items[0];
    let shift = prog
        .shift
        .ok_or_else(|| VsaAotError::Malformed("permute needs a shift".to_owned()))?;
    let d = a.len() as i64;
    for i in 0..a.len() {
        let src = (i as i64 + shift).rem_euclid(d) as usize;
        emit_print_const_f64_bits(a[src], ssa, out);
    }
    emit_newline(ssa, out);
    Ok(())
}

// ─── similarity (mirrors `vsa_codegen::emit_similarity`/`emit_cosine`/`emit_hamming_sim`/`emit_phase_sim`) ─

fn emit_dot_acc(xs: &[f64], ys: &[f64], ssa: &mut Ssa, out: &mut String) -> String {
    let mut acc = ssa.fresh();
    let _ = writeln!(out, "    {acc} = arith.constant 0.0 : f64");
    for (x, y) in xs.iter().zip(ys.iter()) {
        let xv = ssa.fresh();
        let _ = writeln!(
            out,
            "    {xv} = arith.constant {} : f64",
            mlir_f64_const(*x)
        );
        let yv = ssa.fresh();
        let _ = writeln!(
            out,
            "    {yv} = arith.constant {} : f64",
            mlir_f64_const(*y)
        );
        let p = ssa.fresh();
        let _ = writeln!(out, "    {p} = arith.mulf {xv}, {yv} : f64");
        let next = ssa.fresh();
        let _ = writeln!(out, "    {next} = arith.addf {acc}, {p} : f64");
        acc = next;
    }
    acc
}

fn emit_cosine(a: &[f64], b: &[f64], ssa: &mut Ssa, out: &mut String) -> String {
    let dot = emit_dot_acc(a, b, ssa, out);
    let na2 = emit_dot_acc(a, a, ssa, out);
    let nb2 = emit_dot_acc(b, b, ssa, out);
    let na = ssa.fresh();
    let _ = writeln!(out, "    {na} = math.sqrt {na2} : f64");
    let nb = ssa.fresh();
    let _ = writeln!(out, "    {nb} = math.sqrt {nb2} : f64");
    let denom = ssa.fresh();
    let _ = writeln!(out, "    {denom} = arith.mulf {na}, {nb} : f64");
    let zero = ssa.fresh();
    let _ = writeln!(out, "    {zero} = arith.constant 0.0 : f64");
    let na_z = ssa.fresh();
    let _ = writeln!(out, "    {na_z} = arith.cmpf oeq, {na}, {zero} : f64");
    let nb_z = ssa.fresh();
    let _ = writeln!(out, "    {nb_z} = arith.cmpf oeq, {nb}, {zero} : f64");
    let any_z = ssa.fresh();
    let _ = writeln!(out, "    {any_z} = arith.ori {na_z}, {nb_z} : i1");
    let q = ssa.fresh();
    let _ = writeln!(out, "    {q} = arith.divf {dot}, {denom} : f64");
    let sim = ssa.fresh();
    let _ = writeln!(out, "    {sim} = arith.select {any_z}, {zero}, {q} : f64");
    sim
}

fn emit_hamming_sim(a: &[f64], b: &[f64], ssa: &mut Ssa, out: &mut String) -> String {
    let mut hamm = ssa.fresh();
    let _ = writeln!(out, "    {hamm} = arith.constant 0.0 : f64");
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let av = ssa.fresh();
        let _ = writeln!(
            out,
            "    {av} = arith.constant {} : f64",
            mlir_f64_const(ai)
        );
        let bv = ssa.fresh();
        let _ = writeln!(
            out,
            "    {bv} = arith.constant {} : f64",
            mlir_f64_const(bi)
        );
        let eq = ssa.fresh();
        let _ = writeln!(out, "    {eq} = arith.cmpf oeq, {av}, {bv} : f64");
        let zero = ssa.fresh();
        let _ = writeln!(out, "    {zero} = arith.constant 0.0 : f64");
        let one = ssa.fresh();
        let _ = writeln!(out, "    {one} = arith.constant 1.0 : f64");
        let inc = ssa.fresh();
        let _ = writeln!(out, "    {inc} = arith.select {eq}, {zero}, {one} : f64");
        let next = ssa.fresh();
        let _ = writeln!(out, "    {next} = arith.addf {hamm}, {inc} : f64");
        hamm = next;
    }
    let len = a.len() as f64;
    let lenv = ssa.fresh();
    let _ = writeln!(
        out,
        "    {lenv} = arith.constant {} : f64",
        mlir_f64_const(len)
    );
    let ratio = ssa.fresh();
    let _ = writeln!(out, "    {ratio} = arith.divf {hamm}, {lenv} : f64");
    let two = ssa.fresh();
    let _ = writeln!(
        out,
        "    {two} = arith.constant {} : f64",
        mlir_f64_const(2.0)
    );
    let scaled = ssa.fresh();
    let _ = writeln!(out, "    {scaled} = arith.mulf {two}, {ratio} : f64");
    let one_c = ssa.fresh();
    let _ = writeln!(
        out,
        "    {one_c} = arith.constant {} : f64",
        mlir_f64_const(1.0)
    );
    let sim = ssa.fresh();
    let _ = writeln!(out, "    {sim} = arith.subf {one_c}, {scaled} : f64");
    sim
}

fn emit_phase_sim(a: &[f64], b: &[f64], ssa: &mut Ssa, out: &mut String) -> String {
    let mut acc = ssa.fresh();
    let _ = writeln!(out, "    {acc} = arith.constant 0.0 : f64");
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let av = ssa.fresh();
        let _ = writeln!(
            out,
            "    {av} = arith.constant {} : f64",
            mlir_f64_const(ai)
        );
        let bv = ssa.fresh();
        let _ = writeln!(
            out,
            "    {bv} = arith.constant {} : f64",
            mlir_f64_const(bi)
        );
        let diff = ssa.fresh();
        let _ = writeln!(out, "    {diff} = arith.subf {av}, {bv} : f64");
        let c = ssa.fresh();
        let _ = writeln!(out, "    {c} = math.cos {diff} : f64");
        let next = ssa.fresh();
        let _ = writeln!(out, "    {next} = arith.addf {acc}, {c} : f64");
        acc = next;
    }
    let len = a.len() as f64;
    let lenv = ssa.fresh();
    let _ = writeln!(
        out,
        "    {lenv} = arith.constant {} : f64",
        mlir_f64_const(len)
    );
    let sim = ssa.fresh();
    let _ = writeln!(out, "    {sim} = arith.divf {acc}, {lenv} : f64");
    sim
}

fn emit_similarity(prog: &VsaProgram, ssa: &mut Ssa, out: &mut String) -> Result<(), VsaAotError> {
    let a = &prog.items[0];
    let b = &prog.items[1];
    let sim = match prog.model {
        VsaModelId::MapI | VsaModelId::Hrr => emit_cosine(a, b, ssa, out),
        VsaModelId::Bsc => emit_hamming_sim(a, b, ssa, out),
        VsaModelId::Fhrr => emit_phase_sim(a, b, ssa, out),
    };
    emit_print_f64_bits(&sim, ssa, out);
    emit_newline(ssa, out);
    Ok(())
}

/// Emit `wrap_phase(theta) = let u = theta llvm.frem TAU; if u < 0 { u + TAU } else u; if u > π { u
/// − TAU } else u`, mirroring `vsa_codegen::emit_wrap_phase` digit-for-digit. Uses **`llvm.frem`**
/// (the `llvm` dialect's own remainder op, `fmod` semantics), **not** `arith.remf` — verified
/// empirically that `arith.remf` is MLIR's *IEEE remainder* (round-to-nearest quotient), which
/// disagrees with `fmod`/Rust's `%`/the direct-LLVM path's `frem` on this exact case
/// (`7.5 rem 2.0`: `arith.remf` → `-0.5`, `llvm.frem`/Rust `%` → `1.5`). See the module doc.
fn emit_wrap_phase(theta: &str, ssa: &mut Ssa, out: &mut String) -> String {
    let tau = ssa.fresh();
    let _ = writeln!(
        out,
        "    {tau} = arith.constant {} : f64",
        mlir_f64_const(std::f64::consts::TAU)
    );
    let pi = ssa.fresh();
    let _ = writeln!(
        out,
        "    {pi} = arith.constant {} : f64",
        mlir_f64_const(std::f64::consts::PI)
    );
    let r0 = ssa.fresh();
    let _ = writeln!(out, "    {r0} = llvm.frem {theta}, {tau} : f64");
    let zero = ssa.fresh();
    let _ = writeln!(out, "    {zero} = arith.constant 0.0 : f64");
    let neg = ssa.fresh();
    let _ = writeln!(out, "    {neg} = arith.cmpf olt, {r0}, {zero} : f64");
    let plus = ssa.fresh();
    let _ = writeln!(out, "    {plus} = arith.addf {r0}, {tau} : f64");
    let u = ssa.fresh();
    let _ = writeln!(out, "    {u} = arith.select {neg}, {plus}, {r0} : f64");
    let gt = ssa.fresh();
    let _ = writeln!(out, "    {gt} = arith.cmpf ogt, {u}, {pi} : f64");
    let shifted = ssa.fresh();
    let _ = writeln!(out, "    {shifted} = arith.subf {u}, {tau} : f64");
    let r = ssa.fresh();
    let _ = writeln!(out, "    {r} = arith.select {gt}, {shifted}, {u} : f64");
    r
}

// ─── module assembly + the pipeline (mirrors `super::dense`'s / `super::{compile, compile_and_run}`) ─

/// Emit the full MLIR module for `prog`.
pub fn emit_vsa_mlir(prog: &VsaProgram) -> Result<String, VsaAotError> {
    prog.validate()?;

    let mut ssa = Ssa(0);
    let mut bbc = Bbc(0);
    let mut body = String::new();
    match prog.op {
        VsaCgOp::Bind => emit_bind(prog, false, &mut ssa, &mut body)?,
        VsaCgOp::Unbind => emit_bind(prog, true, &mut ssa, &mut body)?,
        VsaCgOp::Bundle => emit_bundle(prog, &mut ssa, &mut bbc, &mut body)?,
        VsaCgOp::Permute => emit_permute(prog, &mut ssa, &mut body)?,
        VsaCgOp::Similarity => emit_similarity(prog, &mut ssa, &mut body)?,
    }

    let needs_deg = matches!(prog.model, VsaModelId::Fhrr) && matches!(prog.op, VsaCgOp::Bundle);

    let mut module = String::from(
        "// mycelium MLIR-dialect VSA codegen (M-856b; RFC-0039 §5.2; mirrors vsa_codegen.rs)\n",
    );
    module.push_str("module {\n");
    module.push_str(
        "  llvm.mlir.global internal constant @fmt_u64(\"%llu \\00\") : !llvm.array<6 x i8>\n",
    );
    module.push_str(
        "  llvm.mlir.global internal constant @fmt_nl(\"\\0A\\00\") : !llvm.array<2 x i8>\n",
    );
    if needs_deg {
        module.push_str(
            "  llvm.mlir.global internal constant @s_deg(\"DEGENERATE\\00\") : !llvm.array<11 x i8>\n",
        );
    }
    module.push_str("  llvm.func @printf(!llvm.ptr, ...) -> i32\n");
    module.push_str("  func.func @main() -> i32 {\n");
    module.push_str(&body);
    let r = ssa.fresh();
    let _ = writeln!(module, "    {r} = arith.constant 0 : i32");
    let _ = writeln!(module, "    func.return {r} : i32");
    module.push_str("  }\n}\n");
    Ok(module)
}

/// Lower `prog` through the real MLIR pipeline to LLVM IR text (`mlir-opt` → `mlir-translate`),
/// without compiling/running it.
pub fn lower_to_llvm_ir(prog: &VsaProgram) -> Result<String, VsaAotError> {
    let mlir = emit_vsa_mlir(prog)?;
    let tools = super::resolve_tools().map_err(dialect_err_to_vsa)?;

    let dir = unique_tmp_dir().map_err(|_| VsaAotError::Run("mkdir tmp".to_owned()))?;
    let mlir_path = dir.join("vsa.mlir");
    let guard = TmpDir(dir);
    std::fs::write(&mlir_path, mlir.as_bytes())
        .map_err(|e| VsaAotError::Run(format!("write MLIR: {e}")))?;

    let lowered_mlir = super::run_capture(
        &tools.mlir_opt,
        &[
            "--convert-math-to-llvm",
            "--convert-math-to-libm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            path(&mlir_path).map_err(|e| VsaAotError::Run(e.to_string()))?,
        ],
        "mlir-opt",
    )
    .map_err(dialect_err_to_vsa)?;

    let lowered_path = guard.0.join("vsa.lowered.mlir");
    std::fs::write(&lowered_path, lowered_mlir.as_bytes())
        .map_err(|e| VsaAotError::Run(format!("write lowered MLIR: {e}")))?;
    let llvm_ir = super::run_capture(
        &tools.mlir_translate,
        &[
            "--mlir-to-llvmir",
            path(&lowered_path).map_err(|e| VsaAotError::Run(e.to_string()))?,
        ],
        "mlir-translate",
    )
    .map_err(dialect_err_to_vsa)?;
    drop(guard);
    Ok(llvm_ir)
}

/// Compile `prog` through the MLIR pipeline to a native executable, reusing [`VsaArtifact`]'s
/// read-back (DRY — see the module doc). `-lm` links the libm `cos`/`sin`/`atan2` FHRR pulls in.
pub fn dialect_compile(prog: &VsaProgram) -> Result<VsaArtifact, VsaAotError> {
    let llvm_ir = lower_to_llvm_ir(prog)?;

    let dir = unique_tmp_dir().map_err(|_| VsaAotError::Run("mkdir tmp".to_owned()))?;
    let ll = dir.join("vsa.ll");
    let bin = dir.join("vsa");
    let guard = TmpDir(dir);
    std::fs::write(&ll, llvm_ir.as_bytes())
        .map_err(|e| VsaAotError::Run(format!("write LLVM IR: {e}")))?;

    let tools = super::resolve_tools().map_err(dialect_err_to_vsa)?;
    let out = Command::new(&tools.clang)
        .args([
            path(&ll).map_err(|e| VsaAotError::Run(e.to_string()))?,
            "-o",
            path(&bin).map_err(|e| VsaAotError::Run(e.to_string()))?,
            "-Wno-override-module",
            "-lm",
        ])
        .output()
        .map_err(|_| VsaAotError::ToolchainMissing(tools.clang.clone()))?;
    if !out.status.success() {
        return Err(VsaAotError::Compile(format!(
            "clang: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }

    Ok(VsaArtifact::from_binary(
        guard,
        bin,
        prog.op,
        prog.model,
        prog.dim,
        prog.bundle_delta,
        prog.items.len() as u64,
    ))
}

/// Compile + run `prog` through the MLIR-dialect pipeline — the dialect leg of the M-856b three-way
/// VSA differential (interp ≡ direct-LLVM ≡ dialect).
pub fn dialect_compile_and_run(prog: &VsaProgram) -> Result<VsaResult, VsaAotError> {
    dialect_compile(prog)?.run()
}

#[allow(dead_code)] // referenced by doc comments/tests; kept for the sentinel-token contract record
const _SENTINEL: &str = VSA_DEGENERATE_SENTINEL;
