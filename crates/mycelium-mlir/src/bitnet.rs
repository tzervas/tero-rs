//! **BitNet-class packed-ternary acceleration** — a runtime-data compute kernel (M-360; E3-6;
//! FR-C3 / NFR-4 / G3; RFC-0004 §5/§8; ADR-009/ADR-014; phase-3.md §2 / E1).
//!
//! The canonical BitNet primitive is the **ternary multiply-accumulate**: a dot product `y = Σ wᵢ·xᵢ`
//! where the weights `wᵢ` are balanced ternary `{−1,0,+1}` and the activations `xᵢ` are integers.
//! Because the weights are ternary the per-element multiply collapses to add / subtract / skip — the
//! "pack-store-load-**unpack-compute**" loop RFC-0004 §5 reuses from bitnet.cpp. This module emits
//! that loop as **textual LLVM IR** (inspectable, no opaque pass — FR-C3 "metadata, not hidden
//! lowering"; RFC-0004 §6), JIT-compiles it (`clang -shared`), and calls it **in-process** (the
//! M-340 dynamic loader) over buffers passed as **runtime pointers**.
//!
//! **Why this closes the open E1 question.** The earlier native/JIT kernels (M-301/M-303/M-340) bake
//! their inputs in as constants, so `clang` constant-folds the whole computation — the measured time
//! is call/spawn overhead, not compute (honestly captioned, never claimed as throughput; VR-5). Here
//! the weight and activation buffers are *function arguments*: the optimiser cannot fold them, so the
//! per-call time is **genuine packed-ternary compute** over `n` elements. That is the runtime-input
//! kernel E1 (`cargo xtask e1`) needs to finally report a compute-throughput number.
//!
//! **Scope / honesty.** All three bitnet packings — **I2_S** (the RFC-0004 §5 default: 2-bit,
//! 4 trits/byte), **TL1** (2-bit, rotated LUT), and **TL2** (true 1.67 b/w: 3 trits → a 5-bit
//! LUT-index, bit-packed) — each as a
//! **scalar** (non-SIMD) loop with the unpack inlined per [`PackScheme`]. Every scheme's kernel is
//! differential-checked against [`ternary_dot_ref`] (the obvious Rust oracle, decoding the *same*
//! packing through `pack::pack_trits`) so the in-IR unpack is verified, not asserted. What is
//! **not** claimed: parity with bitnet.cpp's hand-tuned **SIMD** kernels — that is the next M-360
//! increment; the E1 verdict reports the measured number and states the comparison baseline
//! explicitly (no pre-written perf claim, VR-5/G3).
//!
//! **Submodule confinement (DN-21 §5 F-2 / M-682):** zero `unsafe` — the ABI `transmute` is now
//! confined to the single `crate::jit::Lib::get` choke-point, so this kernel resolves its symbol as
//! a lifetime-bound `Sym` and calls it through ordinary safe Rust; compiler-enforced.
#![forbid(unsafe_code)]

use std::fmt::Write as _;

use mycelium_core::ternary::digit;
use mycelium_core::{PackScheme, PhysicalLayout, Trit};

use crate::jit::{dlopen_path, BitnetDotFn, Lib, Sym};
use crate::llvm::{path, run_tool, unique_tmp_dir, AotError, TmpDir};
use crate::pack::{needed_bytes, pack_trits};

/// The **inspectable physical-layout record** a packed-ternary kernel decodes (M-610; NFR-1/NFR-4;
/// DN-01; RFC-0004 §5). This is the kernel's reified `Meta.physical` claim: which [`PackScheme`] it
/// reads, expressed as the [`PhysicalLayout`] record that travels on a result's `Meta`, plus the
/// *actual* byte/bit density the kernel's loads assume (derived from [`crate::pack::needed_bytes`] —
/// the single source of truth for the buffer the kernel reads, never a separately-asserted number).
///
/// Packing is a **schedule concern recorded on `Meta.physical`, never hidden lowering** (DN-01): the
/// kernel does not silently "know" a layout — it carries this record, it is queryable, and an
/// `EXPLAIN` ([`KernelLayout::explain`]) renders it. A *wrong* record fed to the kernel misreads the
/// buffer; that mislabel is caught by the kernel-level wrong-layout differential (M-251 E3 carried
/// onto the native kernel; `tests/native_packed_layout.rs`), so the record is trusted **only because
/// a wrong one is caught** (NFR-7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelLayout {
    /// The packing scheme the kernel's unpack body decodes.
    scheme: PackScheme,
}

impl KernelLayout {
    /// The layout for `scheme`. Only the three bitnet packings (I2_S/TL1/TL2) have a native kernel,
    /// so this is the layout a compiled [`BitnetDotKernel`] reports; constructing it for another
    /// scheme is still meaningful as a *record* but no kernel decodes it.
    #[must_use]
    pub fn new(scheme: PackScheme) -> Self {
        Self { scheme }
    }

    /// The packing scheme.
    #[must_use]
    pub fn scheme(self) -> PackScheme {
        self.scheme
    }

    /// The reified `Meta.physical` record — the [`PhysicalLayout`] that travels on a result's `Meta`
    /// (`PhysicalLayout::TritPacked { scheme }`). This is the value the E3 wrong-layout differential
    /// (M-251) compares against the *true* packing; a mismatch is the soundness hazard it catches.
    #[must_use]
    pub fn physical(self) -> PhysicalLayout {
        PhysicalLayout::TritPacked {
            scheme: self.scheme,
        }
    }

    /// The **actual** bits-per-element the kernel's loads assume, measured from the byte buffer the
    /// kernel reads over a large run ([`crate::pack::needed_bytes`] — the codec the kernel decodes,
    /// not a cost-model estimate). For the byte-aligned 2-bit schemes this is exactly `2.0`; for the
    /// TL2 5-bit-field bitstream it converges to `5/3 ≈ 1.667`. Honest/`Empirical`: it is the density
    /// of the concrete layout this kernel decodes, computed over a 60 000-element reference window.
    #[must_use]
    pub fn bits_per_element(self) -> f64 {
        const N: usize = 60_000;
        #[allow(clippy::cast_precision_loss)]
        {
            (needed_bytes(self.scheme, N) as f64) * 8.0 / (N as f64)
        }
    }

    /// A human-readable `EXPLAIN` of the physical layout — what the kernel actually reads, so the
    /// packing is auditable and never a black box (NFR-1/NFR-4; DN-01). Names the scheme, the
    /// grouping (trits per byte / per bit-field), and the measured bits-per-element density.
    #[must_use]
    pub fn explain(self) -> String {
        let grouping = match self.scheme {
            PackScheme::Unpacked => "1 trit/byte".to_owned(),
            PackScheme::I2S | PackScheme::Tl1 | PackScheme::TwoBitPerTrit => {
                "4 trits/byte (2-bit code per trit)".to_owned()
            }
            PackScheme::FiveTritPerByte => "5 trits/byte (base-3, 1.6 b/w)".to_owned(),
            PackScheme::Tl2 => {
                "3 trits -> a 5-bit LUT-index, bit-packed contiguously (1.67 b/w bitstream)"
                    .to_owned()
            }
        };
        format!(
            "PhysicalLayout::TritPacked {{ scheme: {:?} }} — {grouping}; \
             measured {:.3} bits/element (from pack::needed_bytes, the codec the kernel decodes)",
            self.scheme,
            self.bits_per_element(),
        )
    }
}

/// The packing this kernel decodes inline by default. **I2_S** is the RFC-0004 §5 default (2-bit,
/// 4 trits/byte, `rot = 0` so a code `c ∈ {0,1,2}` is the base-3 digit and the signed weight is
/// `c − 1`).
pub const KERNEL_SCHEME: PackScheme = PackScheme::I2S;

/// The reference (oracle) ternary dot product `Σ digit(wᵢ)·xᵢ` over `i64`, the exact semantics the
/// JIT kernel must reproduce. `digit` is the balanced-ternary signed value (`mycelium_core::ternary`).
/// Operates on the **unpacked** trits + activations; the kernel decodes the I2_S packing of the same
/// weights at runtime, so a match proves the in-IR unpack is correct.
#[must_use]
pub fn ternary_dot_ref(weights: &[Trit], activations: &[i32]) -> i64 {
    weights
        .iter()
        .zip(activations)
        .map(|(&w, &x)| digit(w) * i64::from(x))
        .sum()
}

/// Emit the textual LLVM IR for the **I2_S** packed-ternary dot kernel — the default scheme. Equal
/// to [`emit_bitnet_dot_ir_for`]`(PackScheme::I2S)`; retained as the stable entry point the E1
/// harness and the original tests call.
#[must_use]
pub fn emit_bitnet_dot_ir() -> String {
    // I2_S is statically in the supported set, so this never errors.
    emit_bitnet_dot_ir_for(PackScheme::I2S).expect("I2_S has a BitNet kernel")
}

/// Emit the textual LLVM IR for the packed-ternary dot kernel
/// `i64 @myc_bitnet_dot(ptr %w, ptr %x, i64 %n)` decoding `scheme` inline: it loops `i ∈ [0, n)`,
/// loads the packed weight byte, **unpacks** the `i`-th trit under `scheme` to a signed weight
/// `∈ {−1,0,+1}`, loads the activation `x[i]`, and accumulates `weight·x` into an `i64`.
/// Deterministic; one transparent op per step (no opaque pass — RFC-0004 §6). The shared loop
/// scaffold is identical across schemes; only the unpack body differs (the three bitnet packings
/// I2_S/TL1/TL2 — every other [`PackScheme`] returns [`AotError::UnsupportedScheme`]).
pub fn emit_bitnet_dot_ir_for(scheme: PackScheme) -> Result<String, AotError> {
    let unpack = match scheme {
        // I2_S (rot=0): the 2-bit code *is* the base-3 digit, signed weight = code − 1.
        PackScheme::I2S => concat!(
            "  %bi = lshr i64 %i, 2\n",                    // byte index = i / 4
            "  %wp = getelementptr i8, ptr %w, i64 %bi\n", // &w[bi]
            "  %byte = load i8, ptr %wp\n",
            "  %byte32 = zext i8 %byte to i32\n",
            "  %lane = and i64 %i, 3\n", // lane = i % 4
            "  %lane32 = trunc i64 %lane to i32\n",
            "  %sh = shl i32 %lane32, 1\n", // shift = lane * 2
            "  %shifted = lshr i32 %byte32, %sh\n",
            "  %code = and i32 %shifted, 3\n", // 2-bit code ∈ {0,1,2}
            "  %digit = sub i32 %code, 1\n",   // signed weight ∈ {-1,0,1} (I2_S rot=0)
            "  %digit64 = sext i32 %digit to i64\n",
        )
        .to_string(),
        // TL1 (rot=2): code = (d01 + 2) mod 3, so invert it — d01 = (code + 1) mod 3 — then
        // signed weight = d01 − 1. (code+1 ∈ {1,2,3}; urem 3 ∈ {1,2,0}; −1 ∈ {0,1,−1}.)
        PackScheme::Tl1 => concat!(
            "  %bi = lshr i64 %i, 2\n",
            "  %wp = getelementptr i8, ptr %w, i64 %bi\n",
            "  %byte = load i8, ptr %wp\n",
            "  %byte32 = zext i8 %byte to i32\n",
            "  %lane = and i64 %i, 3\n",
            "  %lane32 = trunc i64 %lane to i32\n",
            "  %sh = shl i32 %lane32, 1\n",
            "  %shifted = lshr i32 %byte32, %sh\n",
            "  %code = and i32 %shifted, 3\n", // 2-bit code ∈ {0,1,2}
            "  %c1 = add i32 %code, 1\n",      // invert rot=2: d01 = (code+1) mod 3
            "  %d01 = urem i32 %c1, 3\n",
            "  %digit = sub i32 %d01, 1\n", // signed weight ∈ {-1,0,1}
            "  %digit64 = sext i32 %digit to i64\n",
        )
        .to_string(),
        // TL2 (true bitnet.cpp 1.67 b/w): 3 trits → a 5-bit LUT-index code, bit-packed. Trit i is at
        // group g = i/3, position p = i%3, bit offset 5·g; the code = (5-bit field), digit =
        // (code / 3ᵖ) mod 3, signed weight = digit − 1. The 5-bit field can straddle two bytes, so we
        // read a 2-byte window; the second byte index is **clamped to the last valid byte** (needed −
        // 1, computed from n) so the read never goes out of bounds even for the final group, whose
        // field fits in one byte (the spilled high bits are masked off by `& 31`).
        PackScheme::Tl2 => concat!(
            // needed = ⌈5·⌈n/3⌉ / 8⌉; lastbyte = needed − 1 (loop-invariant; LICM hoists it).
            "  %np2 = add i64 %n, 2\n",
            "  %grpcount = udiv i64 %np2, 3\n",
            "  %totbits = mul i64 %grpcount, 5\n",
            "  %totbitsp7 = add i64 %totbits, 7\n",
            "  %needed = udiv i64 %totbitsp7, 8\n",
            "  %lastbyte = sub i64 %needed, 1\n",
            // this trit's group / position / bit offset
            "  %grp = udiv i64 %i, 3\n",
            "  %pos = urem i64 %i, 3\n",
            "  %bitoff = mul i64 %grp, 5\n",
            "  %byteidx = udiv i64 %bitoff, 8\n",
            "  %shift = urem i64 %bitoff, 8\n",
            // second byte index, clamped to lastbyte (branch-free)
            "  %idx1raw = add i64 %byteidx, 1\n",
            "  %inrange = icmp ult i64 %idx1raw, %lastbyte\n",
            "  %idx1 = select i1 %inrange, i64 %idx1raw, i64 %lastbyte\n",
            // load the 2-byte window and extract the 5-bit code
            "  %bp0 = getelementptr i8, ptr %w, i64 %byteidx\n",
            "  %b0 = load i8, ptr %bp0\n",
            "  %bp1 = getelementptr i8, ptr %w, i64 %idx1\n",
            "  %b1 = load i8, ptr %bp1\n",
            "  %b0w = zext i8 %b0 to i16\n",
            "  %b1w = zext i8 %b1 to i16\n",
            "  %b1hi = shl i16 %b1w, 8\n",
            "  %window = or i16 %b0w, %b1hi\n",
            "  %shift16 = trunc i64 %shift to i16\n",
            "  %wsh = lshr i16 %window, %shift16\n",
            "  %code16 = and i16 %wsh, 31\n",
            "  %code = zext i16 %code16 to i64\n",
            // digit = (code / 3^pos) mod 3, 3^pos ∈ {1,3,9} for pos ∈ {0,1,2}
            "  %isp0 = icmp eq i64 %pos, 0\n",
            "  %isp1 = icmp eq i64 %pos, 1\n",
            "  %dvA = select i1 %isp1, i64 3, i64 9\n",
            "  %div = select i1 %isp0, i64 1, i64 %dvA\n",
            "  %q = udiv i64 %code, %div\n",
            "  %d01 = urem i64 %q, 3\n",      // base-3 digit ∈ {0,1,2}
            "  %digit64 = sub i64 %d01, 1\n", // signed weight ∈ {-1,0,1}
        )
        .to_string(),
        other => return Err(AotError::UnsupportedScheme(format!("{other:?}"))),
    };

    // A fixed kernel (no per-program lowering), written out directly so every load/shift/mul is
    // visible. SSA names are stable, so the emission is byte-for-byte deterministic.
    let mut ir = format!("; mycelium BitNet packed-ternary dot kernel ({scheme:?}; M-360)\n");
    ir.push_str("define i64 @myc_bitnet_dot(ptr %w, ptr %x, i64 %n) {\n");
    ir.push_str("entry:\n  br label %loop\n");
    // loop header: carry the index and the running accumulator as phis.
    ir.push_str("loop:\n");
    ir.push_str("  %i = phi i64 [ 0, %entry ], [ %inext, %body ]\n");
    ir.push_str("  %acc = phi i64 [ 0, %entry ], [ %accnext, %body ]\n");
    ir.push_str("  %done = icmp sge i64 %i, %n\n");
    ir.push_str("  br i1 %done, label %exit, label %body\n");
    // body: unpack one trit (scheme-specific, producing %digit64) and multiply-accumulate.
    ir.push_str("body:\n");
    ir.push_str(&unpack);
    let _ = write!(
        ir,
        concat!(
            "  %xp = getelementptr i32, ptr %x, i64 %i\n", // &x[i]
            "  %xi = load i32, ptr %xp\n",
            "  %xi64 = sext i32 %xi to i64\n",
            "  %prod = mul i64 %digit64, %xi64\n",
            "  %accnext = add i64 %acc, %prod\n",
            "  %inext = add i64 %i, 1\n",
            "  br label %loop\n",
        )
    );
    // exit: %acc is the loop phi, which dominates here — return it.
    ir.push_str("exit:\n  ret i64 %acc\n}\n");
    Ok(ir)
}

/// A compiled, in-process BitNet dot kernel: the `.so` (in a per-artifact temp dir, cleaned on drop),
/// the dynamic-library handle (kept loaded for the kernel's lifetime), and the entry-point symbol name.
/// **Compile once, call many** — the natural shape for the E1 throughput measurement. To call many
/// cheaply, [`bind`](Self::bind) once (resolves the symbol a single time into a lifetime-bound
/// `Sym`) and call the resulting [`BoundBitnetDot`] in the hot loop; the convenience
/// [`call`](Self::call) binds per invocation for one-shot use.
pub struct BitnetDotKernel {
    _dir: TmpDir,
    lib: Lib,
    /// The kernel's entry-point symbol (`myc_bitnet_dot` for the generic kernel, a `myc_bitnet_dot_simd*`
    /// for the [`crate::simd`] variants) — resolved at [`bind`](Self::bind) time, never stored as a
    /// raw pointer (M-682: no `*mut c_void` field survives the co-location risk; DN-21 §4).
    sym: &'static str,
    /// The packing the kernel decodes — fixes the weight-buffer bounds (`n.div_ceil(trits/byte)`)
    /// so the check and the emitted GEP stride agree.
    scheme: PackScheme,
}

impl BitnetDotKernel {
    /// Wrap an already-compiled + loaded `i64 <sym>(ptr %w, ptr %x, i64 %n)` artifact. `pub(crate)`
    /// so a sibling codegen module (the M-360 SIMD kernel) reuses this struct's bounds-checked
    /// [`call`](Self::call) instead of re-rolling the FFI — the SIMD kernel has the identical C
    /// signature and `scheme` bounds model, only a different (hand-vectorized) body + `sym`.
    pub(crate) fn from_loaded(
        dir: TmpDir,
        lib: Lib,
        sym: &'static str,
        scheme: PackScheme,
    ) -> Result<Self, AotError> {
        // Fail fast: verify the entry point is exported now (preserving the pre-M-682 behaviour where
        // the symbol was resolved eagerly at load), rather than deferring a missing-symbol error to
        // the first bind/call.
        lib.probe(sym)?;
        Ok(Self {
            _dir: dir,
            lib,
            sym,
            scheme,
        })
    }

    /// The packing this kernel decodes inline.
    #[must_use]
    pub fn scheme(&self) -> PackScheme {
        self.scheme
    }

    /// The kernel's **inspectable physical-layout record** (M-610): the reified `Meta.physical`
    /// claim — which [`PhysicalLayout`] it decodes and the measured bits-per-element — `EXPLAIN`-able
    /// via [`KernelLayout::explain`]. The packing is metadata on the lowered artifact, not hidden
    /// lowering (NFR-1/NFR-4; DN-01). A wrong record is caught by the kernel-level wrong-layout
    /// differential (M-251 E3), so the record is trusted only because a mislabel is caught (NFR-7).
    #[must_use]
    pub fn layout(&self) -> KernelLayout {
        KernelLayout::new(self.scheme)
    }

    /// **Bind once, call many** (M-682): resolve the entry point a single time into a lifetime-bound
    /// [`BoundBitnetDot`] borrowing this kernel's loaded library. The returned handle carries the
    /// `&self` borrow, so the borrow checker guarantees it (and its fn-pointer) cannot outlive the
    /// `Lib` — no co-location convention (DN-21 §4). Resolving here (not per call) keeps the E1
    /// throughput loop free of per-iteration `dlsym` overhead.
    pub fn bind(&self) -> Result<BoundBitnetDot<'_>, AotError> {
        Ok(BoundBitnetDot {
            kernel: self.lib.bitnet_dot(self.sym)?,
            scheme: self.scheme,
        })
    }

    /// Run the kernel over `packed_weights` and `activations`, summing the first `n` ternary products.
    /// Convenience wrapper that [`bind`](Self::bind)s once and calls — for a hot loop, bind once and
    /// reuse the [`BoundBitnetDot`]. Lengths are checked (see [`BoundBitnetDot::call`]); a short buffer
    /// is an explicit [`AotError`], never an out-of-bounds read.
    pub fn call(
        &self,
        packed_weights: &[u8],
        activations: &[i32],
        n: usize,
    ) -> Result<i64, AotError> {
        self.bind()?.call(packed_weights, activations, n)
    }
}

/// A [`BitnetDotKernel`] with its entry point resolved into a lifetime-bound `Sym` (M-682). Produced
/// by [`BitnetDotKernel::bind`]; borrows the kernel's loaded library for `'lib`, so its fn-pointer can
/// never be called after the library unloads (the §4 dangling-pointer risk, now compiler-checked).
/// Call it as many times as needed over varying buffers — the `dlsym` cost was paid once at bind.
pub struct BoundBitnetDot<'lib> {
    kernel: Sym<'lib, BitnetDotFn>,
    scheme: PackScheme,
}

impl BoundBitnetDot<'_> {
    /// Run the kernel over `packed_weights` (packed under the bound kernel's `scheme`) and
    /// `activations`, summing the first `n` ternary products. The lengths are checked against `n`
    /// (≥ `pack::needed_bytes(scheme, n)` weight bytes — `n.div_ceil(4)` for I2_S/TL1, the 5-bit
    /// bitstream length for TL2 — and ≥ `n` activations) so the native loads are always in bounds —
    /// a short buffer is an explicit [`AotError`], never an out-of-bounds read.
    pub fn call(
        &self,
        packed_weights: &[u8],
        activations: &[i32],
        n: usize,
    ) -> Result<i64, AotError> {
        let need_bytes = crate::pack::needed_bytes(self.scheme, n);
        if packed_weights.len() < need_bytes {
            return Err(AotError::Run(format!(
                "packed weights too short: need {need_bytes} bytes for {n} trits, got {}",
                packed_weights.len()
            )));
        }
        if activations.len() < n {
            return Err(AotError::Run(format!(
                "activations too short: need {n}, got {}",
                activations.len()
            )));
        }
        let n_i64 = i64::try_from(n).map_err(|_| AotError::Run(format!("n too large: {n}")))?;
        // The bounds checks above guarantee the kernel reads only `w[0..needed_bytes(scheme, n)]` and
        // `x[0..n]`, both in-bounds for the slices (the TL2 kernel clamps its 2-byte window read to the
        // last valid byte). Calling the typed `extern "C"` pointer is ordinary safe Rust — the ABI
        // claim was made (and audited) once at `Lib::get`, and the `Sym` lifetime keeps the library
        // loaded for this call (M-682; DN-21 §4/§7).
        Ok((self.kernel.as_fn())(
            packed_weights.as_ptr(),
            activations.as_ptr(),
            n_i64,
        ))
    }
}

/// Compile the **I2_S** BitNet dot kernel to a shared object and load it in-process. Equal to
/// [`compile_bitnet_dot_for`]`(PackScheme::I2S)`; the stable entry point the E1 harness calls.
pub fn compile_bitnet_dot() -> Result<BitnetDotKernel, AotError> {
    compile_bitnet_dot_for(PackScheme::I2S)
}

/// Compile the BitNet dot kernel for `scheme` to a shared object and load it in-process. Returns
/// [`AotError::ToolchainMissing`] when `clang` is absent so callers can skip (the house idiom), and
/// [`AotError::UnsupportedScheme`] for a packing with no kernel (anything but I2_S/TL1/TL2).
pub fn compile_bitnet_dot_for(scheme: PackScheme) -> Result<BitnetDotKernel, AotError> {
    let ir = emit_bitnet_dot_ir_for(scheme)?;
    let dir = unique_tmp_dir()?;
    let ll = dir.join("bitnet.ll");
    let so = dir.join("bitnet.so");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    // -O2 so the optimiser does real codegen over the runtime-pointer loop (the point of E1).
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
    BitnetDotKernel::from_loaded(guard, lib, "myc_bitnet_dot", scheme)
}

/// Convenience: pack `weights` under [`KERNEL_SCHEME`] (I2_S), compile the kernel, and run the dot
/// product against `activations` once. The wrapper the differential test checks against
/// [`ternary_dot_ref`].
pub fn jit_ternary_dot(weights: &[Trit], activations: &[i32]) -> Result<i64, AotError> {
    jit_ternary_dot_for(weights, activations, KERNEL_SCHEME)
}

/// As [`jit_ternary_dot`], but for an explicit `scheme` — packs `weights` under `scheme` and runs
/// the matching kernel, so the in-IR unpack is checked against the same packing.
pub fn jit_ternary_dot_for(
    weights: &[Trit],
    activations: &[i32],
    scheme: PackScheme,
) -> Result<i64, AotError> {
    let packed = pack_trits(weights, scheme);
    compile_bitnet_dot_for(scheme)?.call(&packed, activations, weights.len().min(activations.len()))
}
