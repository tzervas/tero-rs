//! In-process **JIT** execution (M-340; E3-4; ADR-009; ADR-014; phase-3.md §2 Batch L).
//!
//! Compiles the bit/trit subset to a shared object (`clang -shared`) and calls it **in-process** via
//! `dlopen`/`dlsym` — removing the process-spawn overhead of the AOT path (M-303). It reuses the
//! *same* lowering (`crate::llvm::lower_program`) and the *same* element encoding/decoding
//! (`emit_char_code`/`decode_result`) as the AOT path, so the JIT is a genuine fourth execution path
//! that must agree with the reference interpreter on the observable (`repr + payload + guarantee`,
//! NFR-7/RR-12) — checked in `tests/jit_differential.rs`.
//!
//! **Intentional unsafe (ADR-014; confined per DN-21/M-682).** This module is the workspace's *only*
//! `unsafe`: the dynamic-linker FFI (`dlopen`/`dlsym`/`dlclose`, declared with a bare `extern "C"` —
//! no `libc` dependency) and the ABI `transmute` behind `Lib`'s symbol resolution. The fn-pointer
//! `transmute` lives in one private `unsafe fn get` (so the ABI claim is never made by safe code) and
//! is exposed only through audited **safe, fixed-type accessors** (`jit_kernel`/`bitnet_dot`/`spec_dot`)
//! that each assert the correct signature against the IR this crate emits. The kernels therefore call
//! their resolved pointers through ordinary safe Rust and are themselves `#![forbid(unsafe_code)]`.
//! Each `unsafe` carries a `// SAFETY:` justification and
//! `#[cfg_attr(not(debug_assertions), allow(unsafe_code))]` (warns in dev/test as the caution
//! incentive, silent in release).
//!
//! **Honesty / E1.** The kernel is *closed* (constants baked in), so `clang` constant-folds it — the
//! in-process per-call time measures call overhead, not kernel compute. A calibrated
//! compute-throughput verdict still needs kernels over *runtime data* (M-360, real packed-ternary
//! kernels). This module establishes the JIT *path* + NFR-7 equivalence, **not** the E1 throughput
//! number (VR-5 — not pre-written).

use std::ffi::{c_void, CString};
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;

use mycelium_core::{Node, Value};

use crate::llvm::{
    decode_result, emit_char_code, lower_program, path, run_tool, unique_tmp_dir, AotError,
    LaneKind, TmpDir,
};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

const RTLD_NOW: c_int = 2;

/// Emit the JIT kernel as `i32 @myc_kernel(ptr %out)`: it writes each result element's ASCII char
/// into `out[i]` (one op per element — same transparent rendering as the AOT path) and **returns the
/// overflow status** (0 = ok, 1 = balanced-ternary overflow). The non-`void` return is the in-process
/// half of the read-back protocol: on overflow the kernel returns 1 *without* writing `out`, mirroring
/// the AOT sentinel line and the interpreter's `EvalError::Overflow`. Deterministic.
///
/// `pub(crate)` for white-box access from the in-crate test module (`src/tests/jit.rs`); not part of
/// the public surface (the public entry points are [`compile_so`]/[`jit_run`]).
pub(crate) fn emit_kernel_fn(node: &Node) -> Result<(String, LaneKind, usize), AotError> {
    let lowered = lower_program(node)?;
    if !lowered.funcs.is_empty() {
        // Closures (Increment-2) emit top-level functions + a bump arena that this single-kernel JIT
        // emitter does not assemble — refuse explicitly rather than emit a kernel that calls
        // undefined symbols (G2). The AOT `compile_and_run` path supports closures.
        return Err(AotError::UnsupportedNode(
            "closures (Increment-2) are not supported on the JIT path; use the AOT \
             compile_and_run path"
                .to_owned(),
        ));
    }
    let kind = lowered.result.kind;
    let width = lowered.result.vals.len();
    let vals = lowered.result.vals;
    let overflow = lowered.overflow;
    let mut ssa = lowered.ssa;

    let mut ir = String::from("; mycelium direct-LLVM JIT kernel (M-340)\n");
    ir.push_str("define i32 @myc_kernel(ptr %out) {\nentry:\n");
    ir.push_str(&lowered.body);

    let emit_stores_and_ok = |ir: &mut String, ssa: &mut crate::llvm::Ssa| {
        for (i, v) in vals.iter().enumerate() {
            let c = emit_char_code(kind, v, ssa, ir);
            let t = ssa.fresh();
            let _ = writeln!(ir, "  {t} = trunc i32 {c} to i8");
            let p = ssa.fresh();
            let _ = writeln!(ir, "  {p} = getelementptr i8, ptr %out, i64 {i}");
            let _ = writeln!(ir, "  store i8 {t}, ptr {p}");
        }
        ir.push_str("  ret i32 0\n");
    };

    match overflow {
        None => emit_stores_and_ok(&mut ir, &mut ssa),
        // Branch on the runtime overflow flag: return 1 (no stores) on overflow, else write + 0.
        Some(ovf) => {
            let _ = writeln!(ir, "  br i1 {ovf}, label %ovf, label %ok");
            ir.push_str("ovf:\n  ret i32 1\nok:\n");
            emit_stores_and_ok(&mut ir, &mut ssa);
        }
    }
    ir.push_str("}\n");
    Ok((ir, kind, width))
}

/// A JIT-compiled kernel: the `.so` on disk (in a per-artifact temp dir, cleaned on drop) + the
/// result shape. Produced by [`compile_so`]; call any number of times in-process with
/// [`JitArtifact::call`].
pub struct JitArtifact {
    _dir: TmpDir,
    so: PathBuf,
    kind: LaneKind,
    width: usize,
}

impl JitArtifact {
    /// Call the kernel in-process (`dlopen` → `dlsym` → call) and read the result back as an `Exact`
    /// `Value`. Returns an explicit [`AotError`] on any FFI failure — never a silent/garbage result.
    pub fn call(&self) -> Result<Value, AotError> {
        let lib = dlopen_path(&self.so)?; // dlclose on drop

        // Resolve through the safe typed accessor (M-682): `kernel` is a lifetime-bound `Sym` that
        // borrows `lib`, so the borrow checker guarantees the fn-pointer is not called after `lib`
        // (the `.so`) is unloaded. `buf` is exactly `self.width` bytes and the kernel writes one byte
        // per result element only on the ok path, so the write is in-bounds.
        let kernel = lib.jit_kernel()?;
        let mut buf = vec![0u8; self.width];
        let status = (kernel.as_fn())(buf.as_mut_ptr());
        // Read-back protocol: a non-zero status means the in-process kernel overflowed the m-trit
        // range — an explicit error, never a silently-wrapped (and unwritten) buffer.
        if status != 0 {
            return Err(AotError::Overflow(format!(
                "fixed-width result out of {}-trit range",
                self.width
            )));
        }
        decode_result(self.kind, self.width, buf.iter().map(|&b| b as char))
    }
}

/// The `extern "C"` signature of every packed-ternary dot kernel — the generic [`crate::bitnet`]
/// kernel and the [`crate::simd`] hand-vectorized variants all share it: `i64 f(ptr %w, ptr %x, i64 %n)`.
pub(crate) type BitnetDotFn = extern "C" fn(*const u8, *const i32, i64) -> i64;
/// The `extern "C"` signature of the weight-specialized dot kernel: `i64 f(ptr %x)` (the weights are
/// compiled in, so only the activation buffer is a runtime argument).
pub(crate) type SpecDotFn = extern "C" fn(*const i32) -> i64;
/// The `extern "C"` signature of the JIT element-wise kernel: `i32 f(ptr %out)` (the overflow status).
pub(crate) type JitKernelFn = extern "C" fn(*mut u8) -> i32;
/// The `extern "C"` signature of the **dynamic-VSA** JIT kernel (M-855; `vsa_jit.rs`): `i32
/// f(ptr %out)` writing `width` `i64` IEEE-754 bit patterns (one per hypervector component, or one for
/// a bare-`f64` measurement) and returning a status (`0` = ok, `1` = refused — mirrors the AOT
/// `DEGENERATE` sentinel/`DegenerateBundleComponent`, never a silently-wrapped value). Distinct from
/// [`JitKernelFn`] because the element width differs (`u64` bit patterns vs `u8` char codes).
pub(crate) type VsaJitKernelFn = extern "C" fn(*mut u64) -> i32;

/// A loaded shared library that `dlclose`s itself on drop. `pub(crate)` so other in-crate JIT kernels
/// (e.g. the M-360 BitNet dot kernel) reuse the same dynamic-loader rather than re-rolling the FFI.
pub(crate) struct Lib(*mut c_void);
impl Lib {
    /// Resolve `symbol` in this library to a raw function/data address (an explicit error if absent).
    fn sym(&self, symbol: &str) -> Result<*mut c_void, AotError> {
        let name = CString::new(symbol)
            .map_err(|e| AotError::Run(format!("symbol name has interior NUL: {e}")))?;
        lookup_sym(self.0, &name)
    }

    /// Verify `symbol` is exported (it resolves via `dlsym`) without retaining a typed pointer. Called
    /// at compile/load time so a missing entry point **fails fast** — preserving the pre-M-682
    /// behaviour where `compile_*` resolved the symbol eagerly, rather than deferring the error to the
    /// first `bind`/`call`.
    pub(crate) fn probe(&self, symbol: &str) -> Result<(), AotError> {
        self.sym(symbol).map(|_| ())
    }

    /// Resolve `symbol` to a **typed, lifetime-bound** function pointer [`Sym<'_, T>`](Sym) — the
    /// crate's single ABI-`transmute` choke-point (M-682; DN-21 §4/§6/§7).
    ///
    /// # Safety
    /// `T` MUST be the exact `extern "C"` fn-pointer type of `symbol`'s compiled signature. That ABI
    /// claim is definitionally outside the type system (the irreducible unsafe of DN-21 §7); a mismatch
    /// is UB. This is **private + `unsafe`** precisely so the claim is never made by safe code — it is
    /// made once, in an audited safe wrapper ([`Self::jit_kernel`] / [`Self::bitnet_dot`] /
    /// [`Self::spec_dot`]) whose hard-coded `T` matches the IR this crate emits. Callers then get only
    /// the safe, fixed-type accessors.
    #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
    unsafe fn get<T: Copy>(&self, symbol: &str) -> Result<Sym<'_, T>, AotError> {
        let raw = self.sym(symbol)?; // non-null or an explicit Err
                                     // `sym` errors on a null result, and a fn pointer is pointer-sized on every supported target;
                                     // assert both preconditions of the `transmute_copy` in dev/test (DN-21 §6 M-679 ethos).
        debug_assert!(
            !raw.is_null(),
            "Lib::sym must return a non-null address or Err"
        );
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<*mut c_void>(),
            "Sym<T> requires T to be a pointer-sized fn pointer"
        );
        // SAFETY: `raw` is the non-null address `dlsym` returned for `symbol` in this still-loaded
        // library (`sym` returns `Err` on null); the caller (an audited wrapper below) guarantees `T`
        // is the symbol's `extern "C"` ABI. A function pointer has the same size as `*mut c_void`, so
        // `transmute_copy` reads exactly the pointer (the size is debug-asserted above). The returned
        // `Sym` borrows `&self`, binding the pointer's lifetime to this `Lib`, so it can never be
        // called after the library is dropped (DN-21 §4).
        #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
        let fptr = unsafe { std::mem::transmute_copy::<*mut c_void, T>(&raw) };
        Ok(Sym {
            fptr,
            _lib: PhantomData,
        })
    }

    /// Resolve the JIT element-wise kernel `myc_kernel` to a lifetime-bound, safe-to-call [`Sym`].
    pub(crate) fn jit_kernel(&self) -> Result<Sym<'_, JitKernelFn>, AotError> {
        // SAFETY: `emit_kernel_fn` emits exactly `define i32 @myc_kernel(ptr %out)`, so `JitKernelFn`
        // is this symbol's ABI — the one ABI claim, made here against the emitter that is its source.
        #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
        unsafe {
            self.get::<JitKernelFn>("myc_kernel")
        }
    }

    /// Resolve a packed-ternary dot kernel (`myc_bitnet_dot` or a `myc_bitnet_dot_simd*`) to a
    /// lifetime-bound, safe-to-call [`Sym`]. `symbol` MUST be one of those emitter-produced kernels
    /// (every one shares the [`BitnetDotFn`] signature — `bitnet::emit_bitnet_dot_ir_for` and the
    /// `simd::emit_*` variants); the in-crate call sites pass exactly those names.
    pub(crate) fn bitnet_dot(&self, symbol: &str) -> Result<Sym<'_, BitnetDotFn>, AotError> {
        // SAFETY: every packed-ternary dot kernel this crate emits defines
        // `i64 <symbol>(ptr %w, ptr %x, i64 %n)`, so `BitnetDotFn` is its ABI for any such `symbol`.
        #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
        unsafe {
            self.get::<BitnetDotFn>(symbol)
        }
    }

    /// Resolve the weight-specialized dot kernel `myc_bitnet_dot_spec` to a lifetime-bound,
    /// safe-to-call [`Sym`].
    pub(crate) fn spec_dot(&self) -> Result<Sym<'_, SpecDotFn>, AotError> {
        // SAFETY: `specialize::emit_specialized_dot_ir` emits exactly
        // `define i64 @myc_bitnet_dot_spec(ptr %x)`, so `SpecDotFn` is this symbol's ABI.
        #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
        unsafe {
            self.get::<SpecDotFn>("myc_bitnet_dot_spec")
        }
    }

    /// Resolve the dynamic-VSA JIT kernel `myc_vsa_kernel` to a lifetime-bound, safe-to-call [`Sym`]
    /// (M-855; RFC-0039 §5.3).
    pub(crate) fn vsa_kernel(&self) -> Result<Sym<'_, VsaJitKernelFn>, AotError> {
        // SAFETY: `vsa_jit::emit_vsa_jit_ir` emits exactly `define i32 @myc_vsa_kernel(ptr %out)`
        // writing `i64` bit-pattern elements, so `VsaJitKernelFn` is this symbol's ABI.
        #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
        unsafe {
            self.get::<VsaJitKernelFn>("myc_vsa_kernel")
        }
    }
}

/// A symbol resolved out of a [`Lib`], carrying that library's lifetime (M-682; DN-21 §4). `T` is the
/// symbol's `extern "C"` function-pointer type. The `PhantomData<&'lib Lib>` makes the borrow checker
/// enforce — structurally, not by field-ordering convention — that the symbol (a JIT'd fn-pointer)
/// does not outlive the `Lib` that owns it, closing the §4 co-location dangling-pointer risk. The
/// ABI `transmute` is audited once, in `Lib`'s typed accessors; calling the pointer via [`Sym::as_fn`]
/// is then ordinary safe Rust (the type already encodes the contract).
pub(crate) struct Sym<'lib, T: Copy> {
    fptr: T,
    _lib: PhantomData<&'lib Lib>,
}

impl<T: Copy> Sym<'_, T> {
    /// The typed function pointer. Sound to call for as long as `self` lives — and because `self`
    /// borrows the owning [`Lib`], that is exactly "while the library is loaded" (the lifetime the
    /// type carries). No `unsafe` here: a correctly-typed `extern "C" fn` pointer is safe to call.
    pub(crate) fn as_fn(&self) -> T {
        self.fptr
    }
}
impl Drop for Lib {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a handle returned by `dlopen` and not closed elsewhere; closing it
        // once on drop is the matching `dlclose`. **Dangling-pointer obligation (DN-21 §4):** any
        // address `dlsym` derived from this handle (a JIT'd fn-pointer) is unloaded by this
        // `dlclose`, so no derived pointer may be called after the owning `Lib` is dropped. Today
        // every call site keeps the `Lib` live for the whole call (the fn-ptr is resolved and
        // invoked while a `&Lib` / co-located `_lib` field is in scope), so no derived pointer
        // outlives the handle — M-682 lifts that co-location convention into a compiler-checked
        // lifetime.
        #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
        unsafe {
            dlclose(self.0);
        }
    }
}

/// `dlopen` a shared object by path, returning a [`Lib`] that closes it on drop. The reusable loader
/// shared by [`JitArtifact`] and the M-360 BitNet kernel.
pub(crate) fn dlopen_path(so: &std::path::Path) -> Result<Lib, AotError> {
    let cpath = CString::new(path(so)?)
        .map_err(|e| AotError::Run(format!("so path has interior NUL: {e}")))?;
    Ok(Lib(open_lib(&cpath)?))
}

fn open_lib(so: &CString) -> Result<*mut c_void, AotError> {
    // SAFETY: `so` is a valid NUL-terminated path (a `CString`) to the `.so` just written;
    // `RTLD_NOW` resolves symbols eagerly so a bad library fails here rather than at call time.
    // **Platform assumption (DN-21 §3):** `RTLD_NOW = 2` is hard-coded for the glibc/musl Linux ABI
    // (no `libc` constant is pulled in — an intentional ADR-014 zero-dependency choice); the JIT is
    // Linux-only today. **Global-constructor safety:** the JIT IR emits no `@llvm.global_ctors`
    // (`Empirical` — verified by reading `emit_kernel_fn`/`emit_*_ir`), so `dlopen` runs no foreign
    // constructor code on load.
    #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
    let handle = unsafe { dlopen(so.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        Err(AotError::Run(
            "dlopen failed for the JIT shared object".to_owned(),
        ))
    } else {
        Ok(handle)
    }
}

fn lookup_sym(handle: *mut c_void, sym: &CString) -> Result<*mut c_void, AotError> {
    // The handle reached here only via `open_lib`, which returns `Err` on a null `dlopen` result,
    // so `handle` is non-null by construction; assert it in dev/test (DN-21 §6 M-679).
    debug_assert!(
        !handle.is_null(),
        "lookup_sym requires a live (non-null) dlopen handle"
    );
    // SAFETY: `handle` is a live `dlopen` handle — it originates from `open_lib`, which returns the
    // handle only after checking it is non-null, and the only callers (`Lib::sym`) hold it in a live
    // `Lib`, so the library is still loaded at this point; `sym` is a valid NUL-terminated C string.
    #[cfg_attr(not(debug_assertions), allow(unsafe_code))]
    let ptr = unsafe { dlsym(handle, sym.as_ptr()) };
    if ptr.is_null() {
        Err(AotError::Run(
            "dlsym could not find `myc_kernel`".to_owned(),
        ))
    } else {
        Ok(ptr)
    }
}

/// Compile the bit/trit-subset program to a shared object without calling it. Returns
/// [`AotError::ToolchainMissing`] when `clang` is absent so callers can skip; out-of-subset
/// constructs are the same explicit refusals as [`crate::emit_llvm_ir`].
pub fn compile_so(node: &Node) -> Result<JitArtifact, AotError> {
    let (ir, kind, width) = emit_kernel_fn(node)?;
    let dir = unique_tmp_dir()?;
    let ll = dir.join("jit.ll");
    let so = dir.join("jit.so");
    let guard = TmpDir(dir);

    std::fs::write(&ll, ir.as_bytes()).map_err(|e| AotError::Run(format!("write IR: {e}")))?;
    run_tool(
        "clang",
        &["-shared", "-fPIC", "-x", "ir", path(&ll)?, "-o", path(&so)?],
    )?;

    Ok(JitArtifact {
        _dir: guard,
        so,
        kind,
        width,
    })
}

/// Compile the program to a shared object and call it once, in-process. The convenience wrapper over
/// [`compile_so`] + [`JitArtifact::call`]; the JIT execution path checked against the interpreter
/// (NFR-7).
pub fn jit_run(node: &Node) -> Result<Value, AotError> {
    compile_so(node)?.call()
}
