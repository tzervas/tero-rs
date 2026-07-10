//! MLIR-dialect-native sibling of [`crate::dense_codegen`] (M-856b; RFC-0039 §5.1; epic E25-1).
//!
//! [`crate::dense_codegen`] lowers the un-quantized F32/BF16 `mycelium-dense` element-wise fragment
//! straight to **direct LLVM IR text**. This module lowers the **same** [`DenseProgram`] to a genuine
//! `arith`/`func`/`cf` MLIR module (mixed with the `llvm` dialect only for the `printf` read-back
//! plumbing — MLIR's own core dialect for calling a C-ABI variadic function; the *numeric*
//! computation stays in `arith`/`math`, exactly mirroring the M-601/M-725/M-857 precedent in
//! [`super`]), then drives it through the same verified libMLIR pipeline those fragments use:
//!
//! ```text
//! mlir-opt-<v> --convert-math-to-llvm --convert-math-to-libm --convert-cf-to-llvm --convert-func-to-llvm
//!   --convert-arith-to-llvm --reconcile-unrealized-casts
//!   | mlir-translate-<v> --mlir-to-llvmir
//! ```
//!
//! **Algorithm identity (DRY).** Every op mirrors [`crate::dense_codegen`]'s algorithm digit-for-digit
//! (same rounding, same side-condition checks, same read-back sentinels) — this module re-expresses
//! the *IR syntax* in `arith`/`math`/`llvm` dialect ops, it does not re-derive the *algorithm*.
//! [`DenseProgram::validate`] and the [`DenseArtifact`] read-back/reconstruct logic are the **same**
//! code as the direct-LLVM path (`pub(crate)` for exactly this reuse) — the two compiled paths can
//! never silently diverge on what they accept or how they stamp a result `Value` (VR-5).
//!
//! **A validated MLIR quirk this module works around.** `arith.remf` is MLIR's *IEEE-754 remainder*
//! (round-to-nearest quotient), **not** `fmod`/Rust's `%` — verified empirically against `mlir-opt-18`
//! (`7.5 arith.remf 2.0 = -0.5`, not `1.5`). [`crate::vsa_codegen`]'s FHRR phase wrap needs the `fmod`
//! semantics `llvm.frem` provides bit-for-bit (matching the direct-LLVM path's own `frem`, and Rust's
//! `f64::%`), so this module (and [`super::vsa`]) use `llvm.frem` — the `llvm` dialect's own binary
//! op — never `arith.remf`, for any modulo/wrap.
//!
//! **libMLIR-gated (ADR-019).** Every entry point returns [`DenseAotError::ToolchainMissing`] when
//! `mlir-opt-<v>`/`mlir-translate-<v>`/`clang-<v>` are absent — skip, never a faked pass (the M-725
//! `ran_mlir` non-vacuity discipline).
//!
//! **Guarantee tag:** `Empirical` — differential-checked against the interpreter and the direct-LLVM
//! path; never `Proven` absent a checked equivalence proof (VR-5).
//!
//! **Zero `unsafe`** — inherits the crate's `#![forbid(unsafe_code)]` (this module adds none).

use std::fmt::Write as _;
use std::process::Command;

use mycelium_core::ScalarKind;
use mycelium_dense::DENSE_MIN_NORMAL;

use crate::dense_codegen::{
    DenseAotError, DenseArtifact, DenseCgOp, DenseProgram, DenseResult, DENSE_OVERFLOW_SENTINEL,
    DENSE_SUBNORMAL_SENTINEL,
};
use crate::llvm::{path, unique_tmp_dir, TmpDir};

use super::DialectError;

/// Map a [`DialectError`] (from the shared [`super::resolve_tools`]) to a [`DenseAotError`],
/// preserving the never-silent classification (toolchain-missing stays a skip).
fn dialect_err_to_dense(e: DialectError) -> DenseAotError {
    match e {
        DialectError::ToolchainMissing(t) => DenseAotError::ToolchainMissing(t),
        DialectError::Compile(s) => DenseAotError::Compile(s),
        DialectError::Run(s) => DenseAotError::Run(s),
        DialectError::Parse(s) => DenseAotError::Parse(s),
        DialectError::Wf(s) => DenseAotError::Wf(s),
        other => DenseAotError::Run(other.to_string()),
    }
}

// ─── SSA / block-label counters (local; mirror `dense_codegen::Ssa`'s shape, MLIR `%v{n}` naming) ─

struct Ssa(usize);
impl Ssa {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("%v{n}")
    }
}

struct Bbc(usize);
impl Bbc {
    fn fresh(&mut self) -> String {
        let n = self.0;
        self.0 += 1;
        format!("dbb{n}")
    }
}

/// Render an `f64` as an exact MLIR `f64` hex bit-pattern constant — verified empirically
/// (`mlir-opt-18`) to parse as the *bit pattern*, not a decimal-to-float conversion, exactly like
/// LLVM's own hex float literals (no decimal round-trip; bit-exact).
fn mlir_f64_const(x: f64) -> String {
    format!("0x{:016X}", x.to_bits())
}

/// Render an `f64` (already on-grid / exact) as an MLIR `f32` hex bit-pattern constant. Unlike LLVM's
/// `.ll` textual convention (which widens an `f32` hex literal to the *double* bit pattern), MLIR's
/// `arith.constant … : f32` uses the **natural 32-bit** hex width — verified empirically.
fn mlir_f32_const(x: f64) -> String {
    #[allow(clippy::cast_possible_truncation)] // on-grid checked upstream: exact narrowing
    let xf = x as f32;
    format!("0x{:08X}", xf.to_bits())
}

// ─── printf-based read-back (the `llvm` dialect I/O fragment; RFC-0004 §6 — dumpable, not opaque) ─

/// Print one `f64` SSA value's IEEE-754 bit pattern as a decimal `u64` via `llvm.call @printf` — the
/// same read-back protocol [`crate::dense_codegen`] uses (`"%llu "` per element, one trailing
/// newline), so [`DenseArtifact::run`] parses either path's stdout identically (DRY).
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

/// Print a sentinel string (the never-silent OVERFLOW/SUBNORMAL refusal marker).
fn emit_print_sentinel(global: &str, ssa: &mut Ssa, out: &mut String) {
    let fmt = ssa.fresh();
    let _ = writeln!(out, "    {fmt} = llvm.mlir.addressof @{global} : !llvm.ptr");
    let p = ssa.fresh();
    let _ = writeln!(
        out,
        "    {p} = llvm.call @printf({fmt}) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32"
    );
}

/// Print the trailing newline that terminates the result line.
fn emit_newline(ssa: &mut Ssa, out: &mut String) {
    let fmt = ssa.fresh();
    let _ = writeln!(out, "    {fmt} = llvm.mlir.addressof @fmt_nl : !llvm.ptr");
    let p = ssa.fresh();
    let _ = writeln!(
        out,
        "    {p} = llvm.call @printf({fmt}) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32"
    );
}

// ─── grid rounding (mirrors `dense_codegen::emit_round_to_grid` digit-for-digit) ─────────────────

/// Round an `f32` SSA value to the dtype grid in IR: F32 is identity; BF16 rounds to nearest-even on
/// the bit pattern (`(bits + 0x7FFF + lsb) >> 16 << 16`), the same trick
/// [`crate::dense_codegen::emit_round_to_grid`] emits as raw LLVM ops, re-expressed in `arith`.
fn emit_round_to_grid(dtype: ScalarKind, val: &str, ssa: &mut Ssa, out: &mut String) -> String {
    match dtype {
        ScalarKind::F32 => val.to_owned(),
        ScalarKind::Bf16 => {
            let bits = ssa.fresh();
            let _ = writeln!(out, "    {bits} = arith.bitcast {val} : f32 to i32");
            let c16 = ssa.fresh();
            let _ = writeln!(out, "    {c16} = arith.constant 16 : i32");
            let sh = ssa.fresh();
            let _ = writeln!(out, "    {sh} = arith.shrui {bits}, {c16} : i32");
            let c1 = ssa.fresh();
            let _ = writeln!(out, "    {c1} = arith.constant 1 : i32");
            let lsb = ssa.fresh();
            let _ = writeln!(out, "    {lsb} = arith.andi {sh}, {c1} : i32");
            let c7fff = ssa.fresh();
            let _ = writeln!(out, "    {c7fff} = arith.constant 32767 : i32");
            let add1 = ssa.fresh();
            let _ = writeln!(out, "    {add1} = arith.addi {bits}, {c7fff} : i32");
            let add2 = ssa.fresh();
            let _ = writeln!(out, "    {add2} = arith.addi {add1}, {lsb} : i32");
            let shr = ssa.fresh();
            let _ = writeln!(out, "    {shr} = arith.shrui {add2}, {c16} : i32");
            let shl = ssa.fresh();
            let _ = writeln!(out, "    {shl} = arith.shli {shr}, {c16} : i32");
            let r = ssa.fresh();
            let _ = writeln!(out, "    {r} = arith.bitcast {shl} : i32 to f32");
            r
        }
        ScalarKind::F16 | ScalarKind::F64 => val.to_owned(),
    }
}

/// Emit the never-silent subnormal/overflow check + print, mirroring
/// [`crate::dense_codegen::emit_check_and_print`]'s four-block shape (`ovf` / `chk_sub` / `sub` /
/// `ok`) as real MLIR basic blocks (`cf.cond_br`). Leaves the `^ok` block **open** (no terminator) so
/// the caller can continue emitting the next element directly into it (chaining every element's
/// check into one `@main` CFG, exactly as the direct-LLVM path chains straight-line blocks).
fn emit_check_and_print(val: &str, ssa: &mut Ssa, bbc: &mut Bbc, out: &mut String) {
    let absf = ssa.fresh();
    let _ = writeln!(out, "    {absf} = math.absf {val} : f32");
    let pinf = ssa.fresh();
    let _ = writeln!(out, "    {pinf} = arith.constant 0x7F800000 : f32");
    let is_inf = ssa.fresh();
    let _ = writeln!(out, "    {is_inf} = arith.cmpf oeq, {absf}, {pinf} : f32");
    let is_nan = ssa.fresh();
    let _ = writeln!(out, "    {is_nan} = arith.cmpf uno, {val}, {val} : f32");
    let nonfinite = ssa.fresh();
    let _ = writeln!(out, "    {nonfinite} = arith.ori {is_inf}, {is_nan} : i1");
    let zero32 = ssa.fresh();
    let _ = writeln!(out, "    {zero32} = arith.constant 0.0 : f32");
    let is_zero = ssa.fresh();
    let _ = writeln!(out, "    {is_zero} = arith.cmpf oeq, {val}, {zero32} : f32");
    let min_normal = ssa.fresh();
    let _ = writeln!(
        out,
        "    {min_normal} = arith.constant {} : f32",
        mlir_f32_const(DENSE_MIN_NORMAL)
    );
    let lt_min = ssa.fresh();
    let _ = writeln!(
        out,
        "    {lt_min} = arith.cmpf olt, {absf}, {min_normal} : f32"
    );
    let true1 = ssa.fresh();
    let _ = writeln!(out, "    {true1} = arith.constant true");
    let nz = ssa.fresh();
    let _ = writeln!(out, "    {nz} = arith.xori {is_zero}, {true1} : i1");
    let subnormal = ssa.fresh();
    let _ = writeln!(out, "    {subnormal} = arith.andi {nz}, {lt_min} : i1");

    let ovf_lbl = bbc.fresh();
    let chk_sub_lbl = bbc.fresh();
    let sub_lbl = bbc.fresh();
    let ok_lbl = bbc.fresh();
    let _ = writeln!(
        out,
        "    cf.cond_br {nonfinite}, ^{ovf_lbl}, ^{chk_sub_lbl}"
    );
    let _ = writeln!(out, "  ^{ovf_lbl}:");
    emit_print_sentinel("s_ovf", ssa, out);
    let z1 = ssa.fresh();
    let _ = writeln!(out, "    {z1} = arith.constant 0 : i32");
    let _ = writeln!(out, "    func.return {z1} : i32");
    let _ = writeln!(out, "  ^{chk_sub_lbl}:");
    let _ = writeln!(out, "    cf.cond_br {subnormal}, ^{sub_lbl}, ^{ok_lbl}");
    let _ = writeln!(out, "  ^{sub_lbl}:");
    emit_print_sentinel("s_sub", ssa, out);
    let z2 = ssa.fresh();
    let _ = writeln!(out, "    {z2} = arith.constant 0 : i32");
    let _ = writeln!(out, "    func.return {z2} : i32");
    let _ = writeln!(out, "  ^{ok_lbl}:");
    // In range: extend to f64 and print the bit pattern.
    let d = ssa.fresh();
    let _ = writeln!(out, "    {d} = arith.extf {val} : f32 to f64");
    emit_print_f64_bits(&d, ssa, out);
}

// ─── op emission (mirrors `dense_codegen`'s `emit_elementwise`/`emit_neg`/`emit_scale`/…) ────────

fn emit_elementwise(
    prog: &DenseProgram,
    fop: &str,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    out: &mut String,
) -> Result<(), DenseAotError> {
    let b = prog
        .b
        .as_ref()
        .ok_or_else(|| DenseAotError::Malformed("binary op needs operand b".to_owned()))?;
    for (&ai, &bi) in prog.a.iter().zip(b.iter()) {
        let x = mlir_f32_const(ai);
        let y = mlir_f32_const(bi);
        let xv = ssa.fresh();
        let _ = writeln!(out, "    {xv} = arith.constant {x} : f32");
        let yv = ssa.fresh();
        let _ = writeln!(out, "    {yv} = arith.constant {y} : f32");
        let r = ssa.fresh();
        let _ = writeln!(out, "    {r} = {fop} {xv}, {yv} : f32");
        let rounded = emit_round_to_grid(prog.dtype, &r, ssa, out);
        emit_check_and_print(&rounded, ssa, bbc, out);
    }
    emit_newline(ssa, out);
    Ok(())
}

fn emit_neg(prog: &DenseProgram, ssa: &mut Ssa, out: &mut String) {
    for &ai in &prog.a {
        let x = mlir_f32_const(ai);
        let xv = ssa.fresh();
        let _ = writeln!(out, "    {xv} = arith.constant {x} : f32");
        let r = ssa.fresh();
        let _ = writeln!(out, "    {r} = arith.negf {xv} : f32");
        // neg is exact on a symmetric grid — no rounding / side-condition trap needed.
        let d = ssa.fresh();
        let _ = writeln!(out, "    {d} = arith.extf {r} : f32 to f64");
        emit_print_f64_bits(&d, ssa, out);
    }
    emit_newline(ssa, out);
}

fn emit_scale(
    prog: &DenseProgram,
    ssa: &mut Ssa,
    bbc: &mut Bbc,
    out: &mut String,
) -> Result<(), DenseAotError> {
    let c = prog
        .scale
        .ok_or_else(|| DenseAotError::Malformed("scale needs a factor".to_owned()))?;
    let cc = mlir_f32_const(c);
    let ccv = ssa.fresh();
    let _ = writeln!(out, "    {ccv} = arith.constant {cc} : f32");
    for &ai in &prog.a {
        let x = mlir_f32_const(ai);
        let xv = ssa.fresh();
        let _ = writeln!(out, "    {xv} = arith.constant {x} : f32");
        let r = ssa.fresh();
        let _ = writeln!(out, "    {r} = arith.mulf {ccv}, {xv} : f32");
        let rounded = emit_round_to_grid(prog.dtype, &r, ssa, out);
        emit_check_and_print(&rounded, ssa, bbc, out);
    }
    emit_newline(ssa, out);
    Ok(())
}

/// Accumulate `Σ xᵢ·yᵢ` in `f64`, left-to-right — mirrors `dense_codegen::emit_dot_acc`.
fn emit_dot_acc(xs: &[f64], ys: &[f64], ssa: &mut Ssa, out: &mut String) -> String {
    let mut acc = ssa.fresh();
    let _ = writeln!(out, "    {acc} = arith.constant 0.0 : f64");
    for (x, y) in xs.iter().zip(ys.iter()) {
        let xc = ssa.fresh();
        let _ = writeln!(
            out,
            "    {xc} = arith.constant {} : f64",
            mlir_f64_const(*x)
        );
        let yc = ssa.fresh();
        let _ = writeln!(
            out,
            "    {yc} = arith.constant {} : f64",
            mlir_f64_const(*y)
        );
        let p = ssa.fresh();
        let _ = writeln!(out, "    {p} = arith.mulf {xc}, {yc} : f64");
        let next = ssa.fresh();
        let _ = writeln!(out, "    {next} = arith.addf {acc}, {p} : f64");
        acc = next;
    }
    acc
}

fn emit_dot(prog: &DenseProgram, ssa: &mut Ssa, out: &mut String) -> Result<(), DenseAotError> {
    let b = prog
        .b
        .as_ref()
        .ok_or_else(|| DenseAotError::Malformed("dot needs operand b".to_owned()))?;
    let acc = emit_dot_acc(&prog.a, b, ssa, out);
    emit_print_f64_bits(&acc, ssa, out);
    emit_newline(ssa, out);
    Ok(())
}

fn emit_similarity(
    prog: &DenseProgram,
    ssa: &mut Ssa,
    out: &mut String,
) -> Result<(), DenseAotError> {
    let b = prog
        .b
        .as_ref()
        .ok_or_else(|| DenseAotError::Malformed("similarity needs operand b".to_owned()))?;
    let dot = emit_dot_acc(&prog.a, b, ssa, out);
    let na2 = emit_dot_acc(&prog.a, &prog.a, ssa, out);
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
    emit_print_f64_bits(&sim, ssa, out);
    emit_newline(ssa, out);
    Ok(())
}

// ─── module assembly + the pipeline (mirrors `super::{emit_mlir, compile, compile_and_run}`) ─────

/// Emit the full MLIR module for `prog`: `arith`/`func`/`math` ops for the computation, `llvm` ops
/// for the `printf` read-back plumbing. Returns an explicit [`DenseAotError`] for a program the
/// reference itself refuses (dtype/dim/grid — same [`DenseProgram::validate`] the direct-LLVM path
/// runs).
pub fn emit_dense_mlir(prog: &DenseProgram) -> Result<String, DenseAotError> {
    prog.validate()?;
    let needs_check = matches!(prog.op, DenseCgOp::Add | DenseCgOp::Sub | DenseCgOp::Scale);
    let needs_sqrt = matches!(prog.op, DenseCgOp::Similarity);

    let mut ssa = Ssa(0);
    let mut bbc = Bbc(0);
    let mut body = String::new();
    match prog.op {
        DenseCgOp::Add => emit_elementwise(prog, "arith.addf", &mut ssa, &mut bbc, &mut body)?,
        DenseCgOp::Sub => emit_elementwise(prog, "arith.subf", &mut ssa, &mut bbc, &mut body)?,
        DenseCgOp::Neg => emit_neg(prog, &mut ssa, &mut body),
        DenseCgOp::Scale => emit_scale(prog, &mut ssa, &mut bbc, &mut body)?,
        DenseCgOp::Dot => emit_dot(prog, &mut ssa, &mut body)?,
        DenseCgOp::Similarity => emit_similarity(prog, &mut ssa, &mut body)?,
    }

    let mut module = String::from(
        "// mycelium MLIR-dialect Dense codegen (M-856b; RFC-0039 §5.1; mirrors dense_codegen.rs)\n",
    );
    module.push_str("module {\n");
    module.push_str(
        "  llvm.mlir.global internal constant @fmt_u64(\"%llu \\00\") : !llvm.array<6 x i8>\n",
    );
    module.push_str(
        "  llvm.mlir.global internal constant @fmt_nl(\"\\0A\\00\") : !llvm.array<2 x i8>\n",
    );
    if needs_check {
        module.push_str(
            "  llvm.mlir.global internal constant @s_sub(\"SUBNORMAL\\00\") : !llvm.array<10 x i8>\n",
        );
        module.push_str(
            "  llvm.mlir.global internal constant @s_ovf(\"OVERFLOW\\00\") : !llvm.array<9 x i8>\n",
        );
    }
    let _ = needs_sqrt; // math.sqrt needs no declaration (a `math` dialect op, lowered by --convert-math-to-llvm).
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
pub fn lower_to_llvm_ir(prog: &DenseProgram) -> Result<String, DenseAotError> {
    let mlir = emit_dense_mlir(prog)?;
    let tools = super::resolve_tools().map_err(dialect_err_to_dense)?;

    let dir = unique_tmp_dir().map_err(|_| DenseAotError::Run("mkdir tmp".to_owned()))?;
    let mlir_path = dir.join("dense.mlir");
    let guard = TmpDir(dir);
    std::fs::write(&mlir_path, mlir.as_bytes())
        .map_err(|e| DenseAotError::Run(format!("write MLIR: {e}")))?;

    let lowered_mlir = super::run_capture(
        &tools.mlir_opt,
        &[
            "--convert-math-to-llvm",
            "--convert-math-to-libm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            path(&mlir_path).map_err(|e| DenseAotError::Run(e.to_string()))?,
        ],
        "mlir-opt",
    )
    .map_err(dialect_err_to_dense)?;

    let lowered_path = guard.0.join("dense.lowered.mlir");
    std::fs::write(&lowered_path, lowered_mlir.as_bytes())
        .map_err(|e| DenseAotError::Run(format!("write lowered MLIR: {e}")))?;
    let llvm_ir = super::run_capture(
        &tools.mlir_translate,
        &[
            "--mlir-to-llvmir",
            path(&lowered_path).map_err(|e| DenseAotError::Run(e.to_string()))?,
        ],
        "mlir-translate",
    )
    .map_err(dialect_err_to_dense)?;
    // `guard` must outlive the reads above; drop it explicitly now.
    drop(guard);
    Ok(llvm_ir)
}

/// Compile `prog` through the MLIR pipeline to a native executable, reusing [`DenseArtifact`]'s
/// read-back (DRY — the direct-LLVM and MLIR-dialect Dense paths can never silently disagree on how a
/// result is stamped). Returns [`DenseAotError::ToolchainMissing`] when the MLIR toolchain is absent.
pub fn dialect_compile(prog: &DenseProgram) -> Result<DenseArtifact, DenseAotError> {
    let llvm_ir = lower_to_llvm_ir(prog)?;

    let dir = unique_tmp_dir().map_err(|_| DenseAotError::Run("mkdir tmp".to_owned()))?;
    let ll = dir.join("dense.ll");
    let bin = dir.join("dense");
    let guard = TmpDir(dir);
    std::fs::write(&ll, llvm_ir.as_bytes())
        .map_err(|e| DenseAotError::Run(format!("write LLVM IR: {e}")))?;

    let tools = super::resolve_tools().map_err(dialect_err_to_dense)?;
    let out = Command::new(&tools.clang)
        .args([
            path(&ll).map_err(|e| DenseAotError::Run(e.to_string()))?,
            "-o",
            path(&bin).map_err(|e| DenseAotError::Run(e.to_string()))?,
            "-Wno-override-module",
        ])
        .output()
        .map_err(|_| DenseAotError::ToolchainMissing(tools.clang.clone()))?;
    if !out.status.success() {
        return Err(DenseAotError::Compile(format!(
            "clang: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }

    Ok(DenseArtifact::from_binary(
        guard, bin, prog.op, prog.dim, prog.dtype,
    ))
}

/// Compile + run `prog` through the MLIR-dialect pipeline — the dialect leg of the M-856b three-way
/// Dense differential (interp ≡ direct-LLVM ≡ dialect).
pub fn dialect_compile_and_run(prog: &DenseProgram) -> Result<DenseResult, DenseAotError> {
    dialect_compile(prog)?.run()
}

#[allow(dead_code)] // referenced by doc comments/tests; kept for the sentinel-token contract record
const _SENTINELS: (&str, &str) = (DENSE_OVERFLOW_SENTINEL, DENSE_SUBNORMAL_SENTINEL);
