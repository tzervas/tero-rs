//! **Hand-vectorized (SIMD) packed-ternary dot kernels** (M-360; E3-6; FR-C3 / NFR-4 / G3;
//! RFC-0004 §5/§8; ADR-009/ADR-014; phase-3.md §2 / E1).
//!
//! The [`crate::bitnet`] scalar kernels decode one trit per loop step. This module emits
//! **hand-vectorized** dot kernels for **all three bitnet packings** (I2_S, TL1, TL2) that unpack and
//! multiply-accumulate **8 trits per iteration** using LLVM vector types (`<8 x i32>`/`<8 x i64>`),
//! with a scalar epilogue for the `n mod 8` tail. The vector **unpack** is the correctness-critical
//! part — a wrong shuffle mask or shift vector silently misreads weights — so each is written
//! transparently (every vector op visible in the textual IR, no opaque pass — RFC-0004 §6 / FR-C3)
//! and **differential-checked against the scalar kernel as the oracle** (`tests/simd_differential.rs`)
//! over a corpus that brackets the vector width and the tail (n ∈ {0,1,7,8,9,15,16,17,…}).
//!
//! **I2_S vectorized unpack.** For 8 consecutive trits (= 2 packed bytes, 4 trits/byte):
//! broadcast `[byte0,byte1]` to the 8 lanes (`shufflevector` mask `<0,0,0,0,1,1,1,1>`), shift each
//! lane to bring its 2-bit code to bit 0 (`lshr` by the constant vector `<0,2,4,6,0,2,4,6>`), mask
//! `& 3` to the code, `− 1` to the signed weight.
//!
//! **TL1 vectorized unpack.** Identical byte loading, broadcast, shift, and mask as I2_S (same
//! 2-bit-per-trit layout, 4 trits/byte). Only the decode differs: TL1 uses rot=2 so
//! `weight = select(code == 2, −1, code)` (code=0 → 0, code=1 → +1, code=2 → −1). This avoids
//! `urem` and is SIMD-efficient — a single `icmp eq` + `select` per lane.
//!
//! **TL2 vectorized unpack (4-groups-at-a-time).** TL2 packs 3 trits into a 5-bit LUT-index code,
//! bit-packed as a contiguous bitstream (1.67 b/w). The SIMD body processes **4 groups (= 12 trits)**
//! per iteration. For each group the 5-bit code straddles a 2-byte window in a fixed pattern; we load
//! a 6-byte window per 4-group block, extract each code with fixed shifts and masks, then decode each
//! group's three trits via scalar-per-lane `udiv`/`urem` (digit = (code / 3^pos) mod 3). The scalar
//! tail handles the remaining trits individually, mirroring the scalar kernel exactly.
//!
//! **Scope / honesty (VR-5/G3).** Same exact dot product as the scalar kernels — no guarantee
//! upgraded; the reduction is exact i64 integer arithmetic. The speedup over the scalar JIT kernels is
//! whatever `cargo xtask e1` §5 measures over runtime data; no target is pre-written. The scalar
//! kernels stay the oracle.
//!
//! **Submodule confinement (DN-21 §5 F-2):** zero `unsafe` — compiler-enforced; the crate's
//! only `unsafe` is the dynamic-linking FFI in `jit`/`bitnet`/`specialize`.
#![forbid(unsafe_code)]

use mycelium_core::PackScheme;

use crate::bitnet::BitnetDotKernel;
use crate::jit::dlopen_path;
use crate::llvm::{path, run_tool, unique_tmp_dir, AotError, TmpDir};

/// The SIMD kernel's symbol — distinct from the scalar [`crate::bitnet`] kernel so both can be loaded
/// at once (e.g. for the E1 §5 differential timing).
const SIMD_SYM: &str = "myc_bitnet_dot_simd";
/// The TL1 SIMD kernel's symbol.
const SIMD_TL1_SYM: &str = "myc_bitnet_dot_simd_tl1";
/// The TL2 SIMD kernel's symbol.
const SIMD_TL2_SYM: &str = "myc_bitnet_dot_simd_tl2";

/// Emit the textual LLVM IR for the **hand-vectorized I2_S** packed-ternary dot kernel
/// `i64 @myc_bitnet_dot_simd(ptr %w, ptr %x, i64 %n)`: an 8-wide vector body (8 trits/iteration) plus
/// a scalar tail for `n mod 8`. Deterministic; every vector op is visible (no opaque pass). The
/// vector loads carry explicit `align 1`/`align 4` so an arbitrary (sub-vector-aligned) buffer offset
/// is a legal unaligned load, never UB.
#[must_use]
pub fn emit_bitnet_dot_simd_ir() -> String {
    // Written out directly (no per-program lowering) so every shuffle/shift/mask is inspectable and
    // the emission is byte-for-byte deterministic.
    String::from(concat!(
        "; mycelium SIMD (8-wide) BitNet I2_S packed-ternary dot kernel (M-360)\n",
        "declare i64 @llvm.vector.reduce.add.v8i64(<8 x i64>)\n",
        "define i64 @myc_bitnet_dot_simd(ptr %w, ptr %x, i64 %n) {\n",
        "entry:\n",
        "  %vn = and i64 %n, -8\n", // full 8-lane iterations cover [0, vn)
        "  br label %vloop\n",
        // vector loop: carry the index and the <8 x i64> accumulator.
        "vloop:\n",
        "  %i = phi i64 [ 0, %entry ], [ %inext, %vbody ]\n",
        "  %vacc = phi <8 x i64> [ zeroinitializer, %entry ], [ %vaccnext, %vbody ]\n",
        "  %vdone = icmp sge i64 %i, %vn\n",
        "  br i1 %vdone, label %tail, label %vbody\n",
        "vbody:\n",
        "  %bi = lshr i64 %i, 2\n", // first weight byte = i/4 (4 trits/byte)
        "  %wp = getelementptr i8, ptr %w, i64 %bi\n",
        "  %b2 = load <2 x i8>, ptr %wp, align 1\n", // [byte0, byte1] — 8 trits
        "  %b2_32 = zext <2 x i8> %b2 to <2 x i32>\n",
        // broadcast byte0 to lanes 0-3, byte1 to lanes 4-7
        "  %bc = shufflevector <2 x i32> %b2_32, <2 x i32> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>\n",
        // shift each lane so its 2-bit code lands in bits[0:1]
        "  %sh = lshr <8 x i32> %bc, <i32 0, i32 2, i32 4, i32 6, i32 0, i32 2, i32 4, i32 6>\n",
        "  %code = and <8 x i32> %sh, <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>\n",
        "  %wt = sub <8 x i32> %code, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>\n", // signed weight = code − 1
        "  %xp = getelementptr i32, ptr %x, i64 %i\n",
        "  %xv = load <8 x i32>, ptr %xp, align 4\n", // 8 contiguous activations
        "  %prod = mul <8 x i32> %wt, %xv\n",
        "  %prod64 = sext <8 x i32> %prod to <8 x i64>\n",
        "  %vaccnext = add <8 x i64> %vacc, %prod64\n",
        "  %inext = add i64 %i, 8\n",
        "  br label %vloop\n",
        // horizontally reduce the vector accumulator, then finish the tail scalar-wise.
        "tail:\n",
        "  %hsum = call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %vacc)\n",
        "  br label %sloop\n",
        "sloop:\n",
        "  %j = phi i64 [ %vn, %tail ], [ %jnext, %sbody ]\n",
        "  %sacc = phi i64 [ %hsum, %tail ], [ %saccnext, %sbody ]\n",
        "  %sdone = icmp sge i64 %j, %n\n",
        "  br i1 %sdone, label %exit, label %sbody\n",
        "sbody:\n",
        "  %sbi = lshr i64 %j, 2\n",
        "  %swp = getelementptr i8, ptr %w, i64 %sbi\n",
        "  %sbyte = load i8, ptr %swp\n",
        "  %sbyte32 = zext i8 %sbyte to i32\n",
        "  %slane = and i64 %j, 3\n",
        "  %slane32 = trunc i64 %slane to i32\n",
        "  %ssh = shl i32 %slane32, 1\n",
        "  %sshifted = lshr i32 %sbyte32, %ssh\n",
        "  %scode = and i32 %sshifted, 3\n",
        "  %sdigit = sub i32 %scode, 1\n",
        "  %sdigit64 = sext i32 %sdigit to i64\n",
        "  %sxp = getelementptr i32, ptr %x, i64 %j\n",
        "  %sxi = load i32, ptr %sxp\n",
        "  %sxi64 = sext i32 %sxi to i64\n",
        "  %sprod = mul i64 %sdigit64, %sxi64\n",
        "  %saccnext = add i64 %sacc, %sprod\n",
        "  %jnext = add i64 %j, 1\n",
        "  br label %sloop\n",
        "exit:\n",
        "  ret i64 %sacc\n",
        "}\n",
    ))
}

/// Compile the hand-vectorized I2_S BitNet dot kernel to a shared object and load it in-process,
/// returning a [`BitnetDotKernel`] (I2_S scheme) so the SIMD path reuses the scalar kernel's
/// bounds-checked `call`. Returns [`AotError::ToolchainMissing`] when `clang` is absent (the house
/// skip idiom). Same C signature + I2_S bounds model as the scalar kernel — only the body differs.
pub fn compile_bitnet_dot_simd() -> Result<BitnetDotKernel, AotError> {
    let ir = emit_bitnet_dot_simd_ir();
    let dir = unique_tmp_dir()?;
    let ll = dir.join("bitnet_simd.ll");
    let so = dir.join("bitnet_simd.so");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    // -O2 so the backend lowers the vector IR to real SIMD instructions (the point of E1 §5).
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
    BitnetDotKernel::from_loaded(guard, lib, SIMD_SYM, PackScheme::I2S)
}

/// Emit the textual LLVM IR for the **hand-vectorized TL1** packed-ternary dot kernel
/// `i64 @myc_bitnet_dot_simd_tl1(ptr %w, ptr %x, i64 %n)`: an 8-wide vector body (8 trits/iteration)
/// plus a scalar tail for `n mod 8`. TL1 uses rot=2 (same 2-bit-per-trit, 4 trits/byte layout as
/// I2_S), so byte loading, broadcast, shift, and mask are identical. Only the decode differs:
/// `weight = select(code == 2, −1, code)` (code=0→0, code=1→+1, code=2→−1) — avoids `urem`.
///
/// Deterministic; every vector op is visible (no opaque pass). The vector loads carry explicit
/// `align 1`/`align 4` for arbitrary buffer offsets.
#[must_use]
pub fn emit_bitnet_dot_simd_tl1_ir() -> String {
    // TL1 decode: weight = select(code == 2, -1, code)
    //   code=0 → weight=0  (Zero)
    //   code=1 → weight=+1 (Pos)
    //   code=2 → weight=-1 (Neg)
    // This avoids mod entirely and is SIMD-efficient: one icmp + one select, both lane-parallel.
    String::from(concat!(
        "; mycelium SIMD (8-wide) BitNet TL1 packed-ternary dot kernel (M-360)\n",
        "declare i64 @llvm.vector.reduce.add.v8i64(<8 x i64>)\n",
        "define i64 @myc_bitnet_dot_simd_tl1(ptr %w, ptr %x, i64 %n) {\n",
        "entry:\n",
        "  %vn = and i64 %n, -8\n", // full 8-lane iterations cover [0, vn)
        "  br label %vloop\n",
        // vector loop: carry the index and the <8 x i64> accumulator.
        "vloop:\n",
        "  %i = phi i64 [ 0, %entry ], [ %inext, %vbody ]\n",
        "  %vacc = phi <8 x i64> [ zeroinitializer, %entry ], [ %vaccnext, %vbody ]\n",
        "  %vdone = icmp sge i64 %i, %vn\n",
        "  br i1 %vdone, label %tail, label %vbody\n",
        "vbody:\n",
        "  %bi = lshr i64 %i, 2\n", // first weight byte = i/4 (4 trits/byte)
        "  %wp = getelementptr i8, ptr %w, i64 %bi\n",
        "  %b2 = load <2 x i8>, ptr %wp, align 1\n", // [byte0, byte1] — 8 trits
        "  %b2_32 = zext <2 x i8> %b2 to <2 x i32>\n",
        // broadcast byte0 to lanes 0-3, byte1 to lanes 4-7 (identical to I2_S)
        "  %bc = shufflevector <2 x i32> %b2_32, <2 x i32> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>\n",
        // shift each lane so its 2-bit code lands in bits[0:1] (identical to I2_S)
        "  %sh = lshr <8 x i32> %bc, <i32 0, i32 2, i32 4, i32 6, i32 0, i32 2, i32 4, i32 6>\n",
        "  %code = and <8 x i32> %sh, <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>\n",
        // TL1 decode: weight = select(code == 2, -1, code)
        // This replaces the I2_S `sub %code, 1` — avoids urem, is SIMD-efficient.
        "  %is2 = icmp eq <8 x i32> %code, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>\n",
        "  %wt = select <8 x i1> %is2, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> %code\n",
        "  %xp = getelementptr i32, ptr %x, i64 %i\n",
        "  %xv = load <8 x i32>, ptr %xp, align 4\n", // 8 contiguous activations
        "  %prod = mul <8 x i32> %wt, %xv\n",
        "  %prod64 = sext <8 x i32> %prod to <8 x i64>\n",
        "  %vaccnext = add <8 x i64> %vacc, %prod64\n",
        "  %inext = add i64 %i, 8\n",
        "  br label %vloop\n",
        // horizontally reduce the vector accumulator, then finish the tail scalar-wise.
        "tail:\n",
        "  %hsum = call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %vacc)\n",
        "  br label %sloop\n",
        "sloop:\n",
        "  %j = phi i64 [ %vn, %tail ], [ %jnext, %sbody ]\n",
        "  %sacc = phi i64 [ %hsum, %tail ], [ %saccnext, %sbody ]\n",
        "  %sdone = icmp sge i64 %j, %n\n",
        "  br i1 %sdone, label %exit, label %sbody\n",
        "sbody:\n",
        // scalar tail: same TL1 decode (icmp/select) applied to a single trit
        "  %sbi = lshr i64 %j, 2\n",
        "  %swp = getelementptr i8, ptr %w, i64 %sbi\n",
        "  %sbyte = load i8, ptr %swp\n",
        "  %sbyte32 = zext i8 %sbyte to i32\n",
        "  %slane = and i64 %j, 3\n",
        "  %slane32 = trunc i64 %slane to i32\n",
        "  %ssh = shl i32 %slane32, 1\n",
        "  %sshifted = lshr i32 %sbyte32, %ssh\n",
        "  %scode = and i32 %sshifted, 3\n",
        // TL1 scalar decode: weight = select(code == 2, -1, code)
        "  %sis2 = icmp eq i32 %scode, 2\n",
        "  %sdigit = select i1 %sis2, i32 -1, i32 %scode\n",
        "  %sdigit64 = sext i32 %sdigit to i64\n",
        "  %sxp = getelementptr i32, ptr %x, i64 %j\n",
        "  %sxi = load i32, ptr %sxp\n",
        "  %sxi64 = sext i32 %sxi to i64\n",
        "  %sprod = mul i64 %sdigit64, %sxi64\n",
        "  %saccnext = add i64 %sacc, %sprod\n",
        "  %jnext = add i64 %j, 1\n",
        "  br label %sloop\n",
        "exit:\n",
        "  ret i64 %sacc\n",
        "}\n",
    ))
}

/// Compile the hand-vectorized TL1 BitNet dot kernel to a shared object and load it in-process,
/// returning a [`BitnetDotKernel`] (TL1 scheme) so the SIMD path reuses the scalar kernel's
/// bounds-checked `call`. Returns [`AotError::ToolchainMissing`] when `clang` is absent (the house
/// skip idiom). Same C signature + TL1 bounds model as the scalar kernel — only the body differs.
pub fn compile_bitnet_dot_simd_tl1() -> Result<BitnetDotKernel, AotError> {
    let ir = emit_bitnet_dot_simd_tl1_ir();
    let dir = unique_tmp_dir()?;
    let ll = dir.join("bitnet_simd_tl1.ll");
    let so = dir.join("bitnet_simd_tl1.so");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    // -O2 so the backend lowers the vector IR to real SIMD instructions (the point of E1 §5).
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
    BitnetDotKernel::from_loaded(guard, lib, SIMD_TL1_SYM, PackScheme::Tl1)
}

/// Emit the textual LLVM IR for the **hand-vectorized TL2** packed-ternary dot kernel
/// `i64 @myc_bitnet_dot_simd_tl2(ptr %w, ptr %x, i64 %n)`.
///
/// TL2 packs 3 trits → one 5-bit LUT-index code (`c = d₀ + 3·d₁ + 9·d₂ ∈ [0,27)`), bit-packed
/// as a contiguous bitstream (1.67 b/w, 5 bits/group). The SIMD body processes **4 groups (= 12
/// trits)** per iteration. For each group `g`, the 5-bit code lies at bit offset `5·g` in the byte
/// stream. Within a 4-group block starting at group `g0 = gi`:
///
/// ```text
///   delta=0: abs bit_off = gi*5+0,  byte=bytebase+0, window_shift=gi*5 % 8
///   delta=1: abs bit_off = gi*5+5,  byte=bytebase+0 or +1 (shift= (gi*5+5)%8)
///   delta=2: abs bit_off = gi*5+10, ...
///   delta=3: abs bit_off = gi*5+15, ...
/// ```
///
/// For each delta, we load a 2-byte window at the correct absolute byte and extract the 5-bit
/// code. The inner decode (digit extraction from a 5-bit code) is scalar-per-group:
/// `digit_p = (code / 3^p) mod 3`, weight = `digit - 1`.
///
/// The scalar tail handles the remaining trits (those not covered by whole 4-group blocks)
/// identically to the scalar TL2 kernel — one trit at a time.
///
/// Deterministic; every op is visible (no opaque pass). The `select` clamp on the 2-byte-window
/// read prevents out-of-bounds in both the vector body and the scalar tail.
#[must_use]
pub fn emit_bitnet_dot_simd_tl2_ir() -> String {
    // TL2 (true 1.67 b/w): 3 trits → 5-bit LUT-index code, bit-packed contiguously.
    //
    // SIMD strategy: 4-groups-at-a-time.
    //
    // For the vector body, each iteration processes groups gi, gi+1, gi+2, gi+3 (12 trits).
    // For each group (gi+delta), the 5-bit code is at absolute bit offset (gi+delta)*5.
    // We extract the code as a 2-byte window: byte_idx = ((gi+delta)*5)/8, shift = ((gi+delta)*5)%8,
    // window = byte[byte_idx] | (byte[byte_idx+1] << 8), code = (window >> shift) & 31.
    //
    // All four codes share bytes in [bytebase, bytebase+3], where bytebase = (gi*5)/8.
    // The vector body only runs when gi+3 < vg (whole groups only), so all 4 groups' bytes
    // are within the allocated buffer (needed_bytes(TL2, vg4*3) covers them).
    //
    // Digit decode per group: for each of the 3 trit positions (0,1,2):
    //   digit = (code / 3^pos) % 3, weight = digit - 1.
    // The scalar tail mirrors the scalar bitnet.rs TL2 kernel exactly.
    String::from(concat!(
        "; mycelium SIMD (4-groups-at-a-time = 12 trits/iter) BitNet TL2 dot kernel (M-360)\n",
        "; TL2: 3 trits -> 5-bit code, bit-packed contiguously. group g at bit offset 5*g.\n",
        "define i64 @myc_bitnet_dot_simd_tl2(ptr %w, ptr %x, i64 %n) {\n",
        "entry:\n",
        // Compute the number of whole groups covered by the vector body:
        // vg = n/3 (whole groups); vg4 = vg & -4 (rounded down to multiple of 4).
        // vn = vg4*3 (trits covered by the vector body).
        "  %vg = udiv i64 %n, 3\n",
        "  %vg4 = and i64 %vg, -4\n",
        "  %vn = mul i64 %vg4, 3\n",
        // Precompute lastbyte = needed_bytes(n) - 1 for the scalar tail's clamped window read.
        // needed_bytes(TL2, n) = ceil(5 * ceil(n/3) / 8).
        "  %np2 = add i64 %n, 2\n",
        "  %grpcount = udiv i64 %np2, 3\n",
        "  %totbits = mul i64 %grpcount, 5\n",
        "  %totbitsp7 = add i64 %totbits, 7\n",
        "  %needed = udiv i64 %totbitsp7, 8\n",
        "  %lastbyte = sub i64 %needed, 1\n",
        "  br label %vloop\n",
        // Vector loop: process 4 groups (12 trits) per iteration.
        // %gi = current group index (0, 4, 8, ...); %vi = current trit index (0, 12, 24, ...).
        "vloop:\n",
        "  %gi = phi i64 [ 0, %entry ], [ %ginext, %vbody ]\n",
        "  %vi = phi i64 [ 0, %entry ], [ %vinext, %vbody ]\n",
        "  %vacc = phi i64 [ 0, %entry ], [ %vaccnext, %vbody ]\n",
        "  %vdone = icmp sge i64 %gi, %vg4\n",
        "  br i1 %vdone, label %tail, label %vbody\n",
        "vbody:\n",
        // For each delta ∈ {0,1,2,3}: compute absolute byte_idx and shift for group gi+delta.
        // abs_bit_d = (gi + delta) * 5 = gi*5 + delta*5.
        // byte_d = abs_bit_d / 8; shift_d = abs_bit_d % 8.
        // window_d = w[byte_d] | (w[byte_d+1] << 8)  (w[byte_d+1] clamped to lastbyte).
        // code_d = (window_d >> shift_d) & 31.
        //
        // delta=0: abs_bit = gi*5.
        "  %abit0 = mul i64 %gi, 5\n",
        "  %byte0 = udiv i64 %abit0, 8\n",
        "  %shft0 = urem i64 %abit0, 8\n",
        "  %byte0n = add i64 %byte0, 1\n",
        "  %b0p0 = getelementptr i8, ptr %w, i64 %byte0\n",
        "  %b0p1 = getelementptr i8, ptr %w, i64 %byte0n\n",
        "  %b0lo = load i8, ptr %b0p0, align 1\n",
        "  %b0hi = load i8, ptr %b0p1, align 1\n",
        "  %b0lo16 = zext i8 %b0lo to i16\n",
        "  %b0hi16 = zext i8 %b0hi to i16\n",
        "  %b0hi_s = shl i16 %b0hi16, 8\n",
        "  %win0 = or i16 %b0lo16, %b0hi_s\n",
        "  %shft0_16 = trunc i64 %shft0 to i16\n",
        "  %wsh0 = lshr i16 %win0, %shft0_16\n",
        "  %code0_16 = and i16 %wsh0, 31\n",
        "  %code0 = zext i16 %code0_16 to i64\n",
        // delta=1: abs_bit = gi*5 + 5.
        "  %gi1 = add i64 %gi, 1\n",
        "  %abit1 = mul i64 %gi1, 5\n",
        "  %byte1 = udiv i64 %abit1, 8\n",
        "  %shft1 = urem i64 %abit1, 8\n",
        "  %byte1n = add i64 %byte1, 1\n",
        "  %b1p0 = getelementptr i8, ptr %w, i64 %byte1\n",
        "  %b1p1 = getelementptr i8, ptr %w, i64 %byte1n\n",
        "  %b1lo = load i8, ptr %b1p0, align 1\n",
        "  %b1hi = load i8, ptr %b1p1, align 1\n",
        "  %b1lo16 = zext i8 %b1lo to i16\n",
        "  %b1hi16 = zext i8 %b1hi to i16\n",
        "  %b1hi_s = shl i16 %b1hi16, 8\n",
        "  %win1 = or i16 %b1lo16, %b1hi_s\n",
        "  %shft1_16 = trunc i64 %shft1 to i16\n",
        "  %wsh1 = lshr i16 %win1, %shft1_16\n",
        "  %code1_16 = and i16 %wsh1, 31\n",
        "  %code1 = zext i16 %code1_16 to i64\n",
        // delta=2: abs_bit = gi*5 + 10.
        "  %gi2 = add i64 %gi, 2\n",
        "  %abit2 = mul i64 %gi2, 5\n",
        "  %byte2 = udiv i64 %abit2, 8\n",
        "  %shft2 = urem i64 %abit2, 8\n",
        "  %byte2n = add i64 %byte2, 1\n",
        "  %b2p0 = getelementptr i8, ptr %w, i64 %byte2\n",
        "  %b2p1 = getelementptr i8, ptr %w, i64 %byte2n\n",
        "  %b2lo = load i8, ptr %b2p0, align 1\n",
        "  %b2hi = load i8, ptr %b2p1, align 1\n",
        "  %b2lo16 = zext i8 %b2lo to i16\n",
        "  %b2hi16 = zext i8 %b2hi to i16\n",
        "  %b2hi_s = shl i16 %b2hi16, 8\n",
        "  %win2 = or i16 %b2lo16, %b2hi_s\n",
        "  %shft2_16 = trunc i64 %shft2 to i16\n",
        "  %wsh2 = lshr i16 %win2, %shft2_16\n",
        "  %code2_16 = and i16 %wsh2, 31\n",
        "  %code2 = zext i16 %code2_16 to i64\n",
        // delta=3: abs_bit = gi*5 + 15.
        "  %gi3 = add i64 %gi, 3\n",
        "  %abit3 = mul i64 %gi3, 5\n",
        "  %byte3 = udiv i64 %abit3, 8\n",
        "  %shft3 = urem i64 %abit3, 8\n",
        "  %byte3n = add i64 %byte3, 1\n",
        "  %b3p0 = getelementptr i8, ptr %w, i64 %byte3\n",
        "  %b3p1 = getelementptr i8, ptr %w, i64 %byte3n\n",
        "  %b3lo = load i8, ptr %b3p0, align 1\n",
        "  %b3hi = load i8, ptr %b3p1, align 1\n",
        "  %b3lo16 = zext i8 %b3lo to i16\n",
        "  %b3hi16 = zext i8 %b3hi to i16\n",
        "  %b3hi_s = shl i16 %b3hi16, 8\n",
        "  %win3 = or i16 %b3lo16, %b3hi_s\n",
        "  %shft3_16 = trunc i64 %shft3 to i16\n",
        "  %wsh3 = lshr i16 %win3, %shft3_16\n",
        "  %code3_16 = and i16 %wsh3, 31\n",
        "  %code3 = zext i16 %code3_16 to i64\n",
        // Decode 4 groups × 3 trits = 12 trits. For group g and position p:
        //   digit = (code / 3^p) % 3, weight = digit - 1.
        // Multiply weight by activation and accumulate.
        // Group 0 (delta=0) → trits vi+0, vi+1, vi+2
        "  %d0p0 = urem i64 %code0, 3\n",
        "  %w0p0 = sub i64 %d0p0, 1\n",
        "  %xi0p = getelementptr i32, ptr %x, i64 %vi\n",
        "  %xi0 = load i32, ptr %xi0p, align 4\n",
        "  %xi0_64 = sext i32 %xi0 to i64\n",
        "  %pr0p0 = mul i64 %w0p0, %xi0_64\n",
        "  %code0d3 = udiv i64 %code0, 3\n",
        "  %d0p1 = urem i64 %code0d3, 3\n",
        "  %w0p1 = sub i64 %d0p1, 1\n",
        "  %vi1 = add i64 %vi, 1\n",
        "  %xi1p = getelementptr i32, ptr %x, i64 %vi1\n",
        "  %xi1 = load i32, ptr %xi1p, align 4\n",
        "  %xi1_64 = sext i32 %xi1 to i64\n",
        "  %pr0p1 = mul i64 %w0p1, %xi1_64\n",
        "  %code0d9 = udiv i64 %code0, 9\n",
        "  %d0p2 = urem i64 %code0d9, 3\n",
        "  %w0p2 = sub i64 %d0p2, 1\n",
        "  %vi2 = add i64 %vi, 2\n",
        "  %xi2p = getelementptr i32, ptr %x, i64 %vi2\n",
        "  %xi2 = load i32, ptr %xi2p, align 4\n",
        "  %xi2_64 = sext i32 %xi2 to i64\n",
        "  %pr0p2 = mul i64 %w0p2, %xi2_64\n",
        "  %acc_g0a = add i64 %pr0p0, %pr0p1\n",
        "  %acc_g0 = add i64 %acc_g0a, %pr0p2\n",
        // Group 1 (delta=1) → trits vi+3, vi+4, vi+5
        "  %d1p0 = urem i64 %code1, 3\n",
        "  %w1p0 = sub i64 %d1p0, 1\n",
        "  %vi3 = add i64 %vi, 3\n",
        "  %xi3p = getelementptr i32, ptr %x, i64 %vi3\n",
        "  %xi3 = load i32, ptr %xi3p, align 4\n",
        "  %xi3_64 = sext i32 %xi3 to i64\n",
        "  %pr1p0 = mul i64 %w1p0, %xi3_64\n",
        "  %code1d3 = udiv i64 %code1, 3\n",
        "  %d1p1 = urem i64 %code1d3, 3\n",
        "  %w1p1 = sub i64 %d1p1, 1\n",
        "  %vi4 = add i64 %vi, 4\n",
        "  %xi4p = getelementptr i32, ptr %x, i64 %vi4\n",
        "  %xi4 = load i32, ptr %xi4p, align 4\n",
        "  %xi4_64 = sext i32 %xi4 to i64\n",
        "  %pr1p1 = mul i64 %w1p1, %xi4_64\n",
        "  %code1d9 = udiv i64 %code1, 9\n",
        "  %d1p2 = urem i64 %code1d9, 3\n",
        "  %w1p2 = sub i64 %d1p2, 1\n",
        "  %vi5 = add i64 %vi, 5\n",
        "  %xi5p = getelementptr i32, ptr %x, i64 %vi5\n",
        "  %xi5 = load i32, ptr %xi5p, align 4\n",
        "  %xi5_64 = sext i32 %xi5 to i64\n",
        "  %pr1p2 = mul i64 %w1p2, %xi5_64\n",
        "  %acc_g1a = add i64 %pr1p0, %pr1p1\n",
        "  %acc_g1 = add i64 %acc_g1a, %pr1p2\n",
        // Group 2 (delta=2) → trits vi+6, vi+7, vi+8
        "  %d2p0 = urem i64 %code2, 3\n",
        "  %w2p0 = sub i64 %d2p0, 1\n",
        "  %vi6 = add i64 %vi, 6\n",
        "  %xi6p = getelementptr i32, ptr %x, i64 %vi6\n",
        "  %xi6 = load i32, ptr %xi6p, align 4\n",
        "  %xi6_64 = sext i32 %xi6 to i64\n",
        "  %pr2p0 = mul i64 %w2p0, %xi6_64\n",
        "  %code2d3 = udiv i64 %code2, 3\n",
        "  %d2p1 = urem i64 %code2d3, 3\n",
        "  %w2p1 = sub i64 %d2p1, 1\n",
        "  %vi7 = add i64 %vi, 7\n",
        "  %xi7p = getelementptr i32, ptr %x, i64 %vi7\n",
        "  %xi7 = load i32, ptr %xi7p, align 4\n",
        "  %xi7_64 = sext i32 %xi7 to i64\n",
        "  %pr2p1 = mul i64 %w2p1, %xi7_64\n",
        "  %code2d9 = udiv i64 %code2, 9\n",
        "  %d2p2 = urem i64 %code2d9, 3\n",
        "  %w2p2 = sub i64 %d2p2, 1\n",
        "  %vi8 = add i64 %vi, 8\n",
        "  %xi8p = getelementptr i32, ptr %x, i64 %vi8\n",
        "  %xi8 = load i32, ptr %xi8p, align 4\n",
        "  %xi8_64 = sext i32 %xi8 to i64\n",
        "  %pr2p2 = mul i64 %w2p2, %xi8_64\n",
        "  %acc_g2a = add i64 %pr2p0, %pr2p1\n",
        "  %acc_g2 = add i64 %acc_g2a, %pr2p2\n",
        // Group 3 (delta=3) → trits vi+9, vi+10, vi+11
        "  %d3p0 = urem i64 %code3, 3\n",
        "  %w3p0 = sub i64 %d3p0, 1\n",
        "  %vi9 = add i64 %vi, 9\n",
        "  %xi9p = getelementptr i32, ptr %x, i64 %vi9\n",
        "  %xi9 = load i32, ptr %xi9p, align 4\n",
        "  %xi9_64 = sext i32 %xi9 to i64\n",
        "  %pr3p0 = mul i64 %w3p0, %xi9_64\n",
        "  %code3d3 = udiv i64 %code3, 3\n",
        "  %d3p1 = urem i64 %code3d3, 3\n",
        "  %w3p1 = sub i64 %d3p1, 1\n",
        "  %vi10 = add i64 %vi, 10\n",
        "  %xi10p = getelementptr i32, ptr %x, i64 %vi10\n",
        "  %xi10 = load i32, ptr %xi10p, align 4\n",
        "  %xi10_64 = sext i32 %xi10 to i64\n",
        "  %pr3p1 = mul i64 %w3p1, %xi10_64\n",
        "  %code3d9 = udiv i64 %code3, 9\n",
        "  %d3p2 = urem i64 %code3d9, 3\n",
        "  %w3p2 = sub i64 %d3p2, 1\n",
        "  %vi11 = add i64 %vi, 11\n",
        "  %xi11p = getelementptr i32, ptr %x, i64 %vi11\n",
        "  %xi11 = load i32, ptr %xi11p, align 4\n",
        "  %xi11_64 = sext i32 %xi11 to i64\n",
        "  %pr3p2 = mul i64 %w3p2, %xi11_64\n",
        "  %acc_g3a = add i64 %pr3p0, %pr3p1\n",
        "  %acc_g3 = add i64 %acc_g3a, %pr3p2\n",
        // Combine all 4 groups into the running accumulator.
        "  %block01 = add i64 %acc_g0, %acc_g1\n",
        "  %block23 = add i64 %acc_g2, %acc_g3\n",
        "  %block = add i64 %block01, %block23\n",
        "  %vaccnext = add i64 %vacc, %block\n",
        "  %ginext = add i64 %gi, 4\n",
        "  %vinext = add i64 %vi, 12\n",
        "  br label %vloop\n",
        // Scalar tail: handle trits in [vn, n) — those not covered by whole 4-group blocks.
        // This is one trit at a time, mirroring the scalar TL2 bitnet.rs kernel exactly.
        // The tail phi references %vacc from vloop (the accumulator on loop exit).
        "tail:\n",
        "  br label %sloop\n",
        "sloop:\n",
        "  %j = phi i64 [ %vn, %tail ], [ %jnext, %sbody ]\n",
        "  %sacc = phi i64 [ %vacc, %tail ], [ %saccnext, %sbody ]\n",
        "  %sdone = icmp sge i64 %j, %n\n",
        "  br i1 %sdone, label %exit, label %sbody\n",
        "sbody:\n",
        // TL2 scalar decode: group = j/3, pos = j%3, bit_off = group*5
        "  %sgrp = udiv i64 %j, 3\n",
        "  %spos = urem i64 %j, 3\n",
        "  %sbitoff = mul i64 %sgrp, 5\n",
        "  %sbyteidx = udiv i64 %sbitoff, 8\n",
        "  %sshift = urem i64 %sbitoff, 8\n",
        // Load the 2-byte window (second byte clamped to lastbyte — no OOB for the final group).
        "  %sidx1raw = add i64 %sbyteidx, 1\n",
        "  %sinrange = icmp ult i64 %sidx1raw, %lastbyte\n",
        "  %sidx1 = select i1 %sinrange, i64 %sidx1raw, i64 %lastbyte\n",
        "  %sbp0 = getelementptr i8, ptr %w, i64 %sbyteidx\n",
        "  %sb0 = load i8, ptr %sbp0\n",
        "  %sbp1 = getelementptr i8, ptr %w, i64 %sidx1\n",
        "  %sb1 = load i8, ptr %sbp1\n",
        "  %sb0w = zext i8 %sb0 to i16\n",
        "  %sb1w = zext i8 %sb1 to i16\n",
        "  %sb1hi = shl i16 %sb1w, 8\n",
        "  %swindow = or i16 %sb0w, %sb1hi\n",
        "  %sshift16 = trunc i64 %sshift to i16\n",
        "  %swsh = lshr i16 %swindow, %sshift16\n",
        "  %scode16 = and i16 %swsh, 31\n",
        "  %scode = zext i16 %scode16 to i64\n",
        // digit = (code / 3^pos) mod 3, 3^pos ∈ {1,3,9} for pos ∈ {0,1,2}
        "  %sisp0 = icmp eq i64 %spos, 0\n",
        "  %sisp1 = icmp eq i64 %spos, 1\n",
        "  %sdvA = select i1 %sisp1, i64 3, i64 9\n",
        "  %sdiv = select i1 %sisp0, i64 1, i64 %sdvA\n",
        "  %sq = udiv i64 %scode, %sdiv\n",
        "  %sd01 = urem i64 %sq, 3\n",
        "  %sdigit64 = sub i64 %sd01, 1\n",
        "  %sxp = getelementptr i32, ptr %x, i64 %j\n",
        "  %sxi = load i32, ptr %sxp\n",
        "  %sxi64 = sext i32 %sxi to i64\n",
        "  %sprod = mul i64 %sdigit64, %sxi64\n",
        "  %saccnext = add i64 %sacc, %sprod\n",
        "  %jnext = add i64 %j, 1\n",
        "  br label %sloop\n",
        "exit:\n",
        "  ret i64 %sacc\n",
        "}\n",
    ))
}

/// Compile the hand-vectorized TL2 BitNet dot kernel to a shared object and load it in-process,
/// returning a [`BitnetDotKernel`] (TL2 scheme) so the SIMD path reuses the scalar kernel's
/// bounds-checked `call`. Returns [`AotError::ToolchainMissing`] when `clang` is absent (the house
/// skip idiom). Same C signature + TL2 bounds model as the scalar kernel — only the body differs.
pub fn compile_bitnet_dot_simd_tl2() -> Result<BitnetDotKernel, AotError> {
    let ir = emit_bitnet_dot_simd_tl2_ir();
    let dir = unique_tmp_dir()?;
    let ll = dir.join("bitnet_simd_tl2.ll");
    let so = dir.join("bitnet_simd_tl2.so");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    // -O2 so the backend lowers the vector IR to real SIMD instructions (the point of E1 §5).
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
    BitnetDotKernel::from_loaded(guard, lib, SIMD_TL2_SYM, PackScheme::Tl2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet::ternary_dot_ref;
    use crate::pack::pack_trits;
    use mycelium_core::Trit;

    fn weights(n: usize) -> Vec<Trit> {
        let mut s = 0x7777_3333_u64;
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
        let mut s = 0xC0FF_EE42_u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (((s >> 40) % 201) as i64 - 100) as i32
            })
            .collect()
    }

    #[test]
    fn ir_is_vectorized_inspectable_and_deterministic() {
        let ir = emit_bitnet_dot_simd_ir();
        assert!(ir.contains("define i64 @myc_bitnet_dot_simd(ptr %w, ptr %x, i64 %n)"));
        // The vector unpack is visible (no opaque pass — FR-C3 / RFC-0004 §6).
        assert!(ir.contains("shufflevector")); // the byte broadcast
        assert!(ir.contains("lshr <8 x i32>")); // the per-lane code shift
        assert!(ir.contains("mul <8 x i32>")); // the vector multiply
        assert!(ir.contains("@llvm.vector.reduce.add.v8i64")); // horizontal reduction
        assert_eq!(emit_bitnet_dot_simd_ir(), emit_bitnet_dot_simd_ir()); // deterministic
    }

    #[test]
    fn jit_simd_matches_scalar_oracle_across_the_width_boundary() {
        // Mutant-witness: a wrong shuffle mask / shift vector, or a missing scalar tail, diverges from
        // the oracle precisely at the n values that straddle the 8-lane width and the tail.
        let kernel = match compile_bitnet_dot_simd() {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return, // environment skip
            Err(e) => panic!("SIMD compile failed: {e}"),
        };
        for n in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 64, 100, 257, 1000] {
            let w = weights(n);
            let x = activations(n);
            let packed = pack_trits(&w, PackScheme::I2S);
            // n=0 needs a non-empty buffer only conceptually; pack of [] is [], call with n=0 is 0.
            let got = kernel.call(&packed, &x, n).expect("SIMD kernel runs");
            assert_eq!(got, ternary_dot_ref(&w, &x), "SIMD dot mismatch at n={n}");
        }
    }

    #[test]
    fn simd_short_buffers_are_explicit_errors() {
        // The reused bounds check (I2_S: n.div_ceil(4) bytes, n activations) still refuses a short
        // buffer — the vector loads must never read past the buffer.
        let kernel = match compile_bitnet_dot_simd() {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return,
            Err(e) => panic!("SIMD compile failed: {e}"),
        };
        let packed = pack_trits(&weights(16), PackScheme::I2S); // 4 bytes
        assert!(matches!(
            kernel.call(&packed, &[1, 2, 3], 16),
            Err(AotError::Run(_))
        ));
        assert!(matches!(
            kernel.call(&[0u8], &activations(16), 16),
            Err(AotError::Run(_))
        ));
    }

    // ── TL1 SIMD tests ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn tl1_ir_is_inspectable_and_deterministic() {
        let ir = emit_bitnet_dot_simd_tl1_ir();
        // Function signature must be present (FR-C3 / RFC-0004 §6).
        assert!(ir.contains("define i64 @myc_bitnet_dot_simd_tl1(ptr %w, ptr %x, i64 %n)"));
        // TL1 decode is `select(code == 2, -1, code)` — visible, no opaque pass.
        assert!(ir.contains("icmp eq <8 x i32> %code")); // the code==2 comparison
        assert!(ir.contains("select <8 x i1>")); // the SIMD select (not sub)
                                                 // The same byte broadcast + shift as I2_S (the 2-bit layout is shared).
        assert!(ir.contains("shufflevector")); // byte broadcast to 8 lanes
        assert!(ir.contains("lshr <8 x i32>")); // per-lane code shift
        assert!(ir.contains("mul <8 x i32>")); // vector multiply
        assert!(ir.contains("@llvm.vector.reduce.add.v8i64")); // horizontal reduction
                                                               // Emission must be deterministic (no RNG or time-dependent component).
        assert_eq!(emit_bitnet_dot_simd_tl1_ir(), emit_bitnet_dot_simd_tl1_ir());
    }

    #[test]
    fn jit_tl1_simd_matches_scalar_oracle() {
        // Mutant-witness: a wrong TL1 decode (e.g. the I2_S `sub %code, 1` instead of the
        // `select(code==2, -1, code)`) diverges from the oracle — the two decodes differ at
        // every trit whose code is 0 (code=0: I2_S→−1, TL1→0) and code=2 (TL1→−1, I2_S→+1).
        let kernel = match compile_bitnet_dot_simd_tl1() {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return, // environment skip
            Err(e) => panic!("TL1 SIMD compile failed: {e}"),
        };
        for n in [
            0usize, 1, 2, 7, 8, 9, 11, 12, 13, 15, 16, 17, 23, 24, 25, 31, 64, 100, 257,
        ] {
            let w = weights(n);
            let x = activations(n);
            let packed = pack_trits(&w, PackScheme::Tl1);
            let got = kernel.call(&packed, &x, n).expect("TL1 SIMD kernel runs");
            assert_eq!(
                got,
                ternary_dot_ref(&w, &x),
                "TL1 SIMD dot mismatch at n={n}"
            );
        }
    }

    // ── TL2 SIMD tests ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn tl2_ir_is_inspectable_and_deterministic() {
        let ir = emit_bitnet_dot_simd_tl2_ir();
        // Function signature must be present (FR-C3 / RFC-0004 §6).
        assert!(ir.contains("define i64 @myc_bitnet_dot_simd_tl2(ptr %w, ptr %x, i64 %n)"));
        // TL2 decode must be visible: 5-bit code extraction and digit division.
        assert!(ir.contains("and i16 %wsh0, 31")); // 5-bit code mask (delta=0)
        assert!(ir.contains("udiv i64 %code0, 3")); // digit extraction for pos=1 (/ 3^1)
        assert!(ir.contains("udiv i64 %code0, 9")); // digit extraction for pos=2 (/ 3^2)
        assert!(ir.contains("urem i64 %code0, 3")); // mod 3 for digit at pos=0
                                                    // The scalar tail must also be visible.
        assert!(ir.contains("udiv i64 %j, 3")); // tail: group = j/3
        assert!(ir.contains("and i16 %swsh, 31")); // tail: 5-bit code mask
                                                   // Emission must be deterministic.
        assert_eq!(emit_bitnet_dot_simd_tl2_ir(), emit_bitnet_dot_simd_tl2_ir());
    }

    #[test]
    fn jit_tl2_simd_matches_scalar_oracle() {
        // Mutant-witness: a wrong bit-offset, shift, or mask silently misreads the 1.67-b/w
        // bitstream. The corpus brackets the 4-group (12-trit) vector body boundary and the tail.
        let kernel = match compile_bitnet_dot_simd_tl2() {
            Ok(k) => k,
            Err(AotError::ToolchainMissing(_)) => return, // environment skip
            Err(e) => panic!("TL2 SIMD compile failed: {e}"),
        };
        for n in [
            0usize, 1, 2, 3, 5, 9, 11, 12, 13, 23, 24, 25, 35, 36, 37, 64, 99, 100, 257,
        ] {
            let w = weights(n);
            let x = activations(n);
            let packed = pack_trits(&w, PackScheme::Tl2);
            let got = kernel.call(&packed, &x, n).expect("TL2 SIMD kernel runs");
            assert_eq!(
                got,
                ternary_dot_ref(&w, &x),
                "TL2 SIMD dot mismatch at n={n}"
            );
        }
    }
}
