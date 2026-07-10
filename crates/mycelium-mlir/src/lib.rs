//! `mycelium-mlir` — the AOT path: a textual **ternary-dialect skeleton** (M-150), a **real**
//! `arith`/`func`→LLVM dialect lowering behind the `mlir-dialect` feature (M-601), and a
//! **direct-LLVM-IR backend** that genuinely compiles the full v0 calculus to native code (M-301/
//! M-373/M-378/M-379; RFC-0004 §2/§6/§11; ADR-007/009/019; T1.5; phase-3.md §1).
//!
//! **Scope / honesty.** The ratified AOT path is `MLIR → LLVM` (RFC-0004 §2). On Linux libMLIR is
//! now provisionable (`scripts/setup-mlir.sh`; ADR-019), so the real dialect lowering (`dialect::native`,
//! feature `mlir-dialect`) lands for the bit/trit element-wise fragment and is differential-checked
//! three ways (M-602). The richer data/closure/recursion fragment is carried by the direct-LLVM
//! backend ([`llvm`]); anything the standard MLIR dialects cannot faithfully express is an explicit
//! never-silent refusal routed there (VR-5/G2). What lands here:
//!
//! - [`dialect::emit`] — a **textual** ternary-dialect rendering of the lowered IR
//!   (`mycelium-core::lower` A-normal form): one dialect op per binding, every value/attr visible.
//!   This is the *per-stage-dumpable, no-opaque-pass* anchor (RFC-0004 §6) — text, not native code,
//!   and the dumpable skeleton of the MLIR path. Always available (no toolchain needed).
//! - `dialect::native` *(feature `mlir-dialect`, OFF by default; M-601)* — the **real** lowering:
//!   for the bit/trit element-wise fragment it emits a genuine `arith`/`func` MLIR module and drives
//!   it through `mlir-opt`/`mlir-translate` to **real LLVM IR**, then `clang` → native → read-back
//!   (the same read-back as [`llvm`], so the two compiled paths compare like-for-like). Probes the
//!   toolchain and skips gracefully when absent (ADR-019). Guarantee: `Empirical` (M-602 differential).
//! - [`aot::run`] — the **env-machine** runnable model: an independent big-step evaluator over the
//!   lowered ANF (sequential binding evaluation) vs the reference interpreter's small-step
//!   substitution (M-110). A genuine *two-path* check for the interp↔AOT differential (M-151/NFR-7).
//! - [`llvm::compile_and_run`] — the **compiled native artifact** (M-301; RFC-0004 §2's *direct-LLVM
//!   fallback*): for the bit subset it emits textual LLVM IR ([`llvm::emit_llvm_ir`], one SSA op per
//!   output bit), drives `llc` + `clang` to a real executable, runs it, and reads the result back.
//!   This is a *third, compiled* execution path; everything outside the bit subset is an explicit
//!   [`llvm::AotError`] refusal (never silent), with `llc`/`clang` absence reported as a skippable
//!   `ToolchainMissing`. The interp↔native differential (M-302) checks it against the interpreter.
//! - [`budget::DepthBudget`] — the **dynamic depth budget** for the env-machine's control stack
//!   (DN-05 §2.4 / DN05-Q5): with the M-347 trampoline the control stack is on the heap, so the depth
//!   ceiling is a *policy over memory headroom*, derived from detected `MemAvailable`/`RLIMIT_AS`
//!   (zero-`unsafe`, pure-`std` `/proc`) with a conservative static fallback and an `EXPLAIN`-able
//!   basis — never a magic constant, never an abort ([`aot::default_depth_budget`]).
//! - [`inject::Image`] — the **in-process hot-inject** prototype (M-341; ADR-016/017): a hash-keyed
//!   `ContentHash → entry` dispatch table over the M-340 JIT. A call resolves to a compiled entry if
//!   present, else interprets (RFC-0004 §9 continuum); injection loads a content-addressed unit and
//!   registers a *new* `hash → entry`, never mutating a live entry (immutability dissolves the
//!   atomicity hazard); the recompile set is the changed dependency-closure by hash reachability
//!   ([`inject::recompile_closure`]). The injected-compiled path is M-210-checked against the
//!   interpreter (NFR-7).
//! - [`mode::run`] — the **explicit, never-silently-selected execution-mode** dispatcher (M-727;
//!   RFC-0029 §7.3): one named [`mode::ExecMode`] (`Interpreter`/`Aot`/`Jit`) per call, no default
//!   and no fallback arm, so the JIT is reachable *only* by naming it (G2). interp ≡ AOT ≡ JIT over
//!   the subset is the M-729 three-way differential (`tests/threeway_codegen_differential.rs`).
//! - [`accel::accelerated_ternary_dot`] — the **BitNet packed-ternary acceleration behind an explicit
//!   capability flag** (M-728; FR-C3; RFC-0029 §7.4): engaged iff the `bitnet-accel` feature is ON
//!   *and* the runtime capability ([`accel::BitnetCapability`]) is present; otherwise an **explicit,
//!   recorded** graceful degradation to [`bitnet::ternary_dot_ref`] (the [`accel::AccelOutcome`]
//!   carries which path ran + an `EXPLAIN`-able reason — never a silent slow path, G2).

pub mod accel;
pub mod aot;
pub mod bitnet;
pub mod budget;
pub mod channel;
// M-865: harness-level parallel dispatch (Scheduler::run_indexed) for the direct-LLVM AOT + JIT
// paths, extending M-862's top-level pure-argument batch to the two compiled execution paths this
// crate owns (`Op`-headed batches only — see the module docs for the honest `Construct` scope carve-out).
pub mod concurrent;
pub mod deploy;
// M-853 (RFC-0039 §5.1): native direct-LLVM codegen of `Repr::Dense` element-wise ops (un-quantized
// F32/BF16 fragment) — differential-checked against `mycelium-dense`, mutant-witnessed, honest tags.
pub mod dense_codegen;
pub mod dialect;
pub mod inject;
pub mod inject_cert;
pub mod inject_gate;
pub mod jit;
pub mod llvm;
pub mod mode;
pub mod pack;
pub mod passes;
pub mod rc_plan;
pub mod runtime;
pub mod simd;
pub mod specialize;
pub mod swap_codegen;
pub mod trampoline;
pub mod vr4;
// M-854 (RFC-0039 §5.2): native direct-LLVM codegen of `Repr::Vsa` hypervector ops (real-Vec<f64>
// MAP-I/BSC/HRR/FHRR fragment) — differential-checked against `mycelium-vsa`, mutant-witnessed, honest
// per-op tags (SBC/MAP-B refused never-silently).
pub mod vsa_codegen;
// M-855 (RFC-0039 §5.3): the dynamic-VSA JIT — the M-340 in-process `dlopen` JIT extended to the
// M-854 real-Vec<f64> MAP-I/BSC/HRR/FHRR fragment, for data-dependent-dimension / runtime-model-
// selection workloads. Reuses vsa_codegen's program/error/EXPLAIN/read-back shapes verbatim; the ADR-
// 009 dynamic-VSA JIT deferral lift is recorded append-only at RFC-0039 §6 (OQ-1, no separate ADR-009
// amendment).
pub mod vsa_jit;

pub use accel::{
    accelerated_ternary_dot, AccelOutcome, BitnetCapability, DegradeReason, Path as AccelPath,
    ACCEL_FEATURE_ENABLED,
};
pub use aot::{
    default_depth_budget, run, run_core, run_core_with_effects, run_core_with_fuel, run_with_layout,
};
pub use bitnet::{
    compile_bitnet_dot, compile_bitnet_dot_for, emit_bitnet_dot_ir, emit_bitnet_dot_ir_for,
    jit_ternary_dot, jit_ternary_dot_for, ternary_dot_ref, BitnetDotKernel, KernelLayout,
};
pub use budget::{
    AutoDepthBudget, DepthBasis, DepthBudget, DepthResolution, MemSource, StaticDepthBudget,
    StaticReason, STATIC_FALLBACK_DEPTH,
};
pub use channel::{Network, Receiver, Sender, TryRecv, TrySend};
// M-865: the harness-level AOT/JIT concurrent-batch dispatch + its EXPLAIN-able plan type.
pub use concurrent::{
    compile_and_run_concurrent, compile_and_run_concurrent_with_swap_mode, jit_run_concurrent,
    plan_concurrent, ConcurrentPlan,
};
pub use deploy::{DeployError, NativeArtifact};
pub use dialect::emit;
#[cfg(feature = "mlir-dialect")]
pub use dialect::native::{
    compile as mlir_compile, compile_and_run as mlir_compile_and_run, emit_mlir, lower_to_llvm_ir,
    Compiled as MlirCompiled, DialectError, MlirTools, ResultKind,
};
pub use inject::{recompile_closure, Image, InjectError, Resolution};
pub use inject_cert::{signed_message, InjectCert};
pub use inject_gate::{
    declared_digest64, Admission, EnforcementGrain, InjectMode, InjectPolicy, PolicyDeviation,
    PolicyError, PolicyManifest, SignatureScheme, SignerId, TestScheme, TrustRoot, VerifyRefusal,
};
pub use jit::{compile_so, jit_run, JitArtifact};
pub use llvm::{compile, compile_and_run, emit_llvm_ir, AotError, CompiledArtifact};
pub use mode::{run as run_mode, ExecMode, ModeError};
pub use pack::{needed_bytes as needed_bytes_for, pack_trits, relayout_trits, unpack_trits};
pub use rc_plan::{emit_reclamation_plan, run_with_reclamation, RcPlanError, RcRun};
pub use runtime::{
    run_colony, run_reclaim, Colony, ColonyError, Deadlock, Poll, ReclaimError, ReclaimRun, Scope,
    SweepOrder, Task, TaskCtx,
};
pub use simd::{
    compile_bitnet_dot_simd, compile_bitnet_dot_simd_tl1, compile_bitnet_dot_simd_tl2,
    emit_bitnet_dot_simd_ir, emit_bitnet_dot_simd_tl1_ir, emit_bitnet_dot_simd_tl2_ir,
};
pub use specialize::{
    compile_specialized_dot, emit_specialized_dot_ir, jit_specialized_dot, SpecializedDotKernel,
};
// M-852: native `Swap`-node codegen for the certified binary↔ternary class + the reified,
// EXPLAIN-able cert mode (DEFAULT compile-time re-check · OPT-IN reuse-interp).
pub use llvm::{
    compile_and_run_with_swap_mode, compile_with_swap_mode, emit_llvm_ir_with_swap_mode,
};
pub use swap_codegen::{legal_pair as swap_legal_pair, SwapCertMode, SwapExplain};
// M-853 (RFC-0039 §5.1): native Dense element-wise codegen (un-quantized F32/BF16) + its inspectable
// EXPLAIN record. `dot`/`similarity` are bare-`f64` measurements; the quantized variants stay an
// explicit never-silent refusal (E20-1 gate).
pub use dense_codegen::{
    dense_compile, dense_compile_and_run, emit_dense_llvm_ir, DenseAotError, DenseArtifact,
    DenseCgOp, DenseExplain, DenseProgram, DenseResult, DENSE_CODEGEN_GUARANTEE,
};
pub use vr4::{cross_backend_gate, Backend, BackendStage, CrossBackendGate, StageStatus};
// M-854 (RFC-0039 §5.2): native VSA hypervector codegen (real-Vec<f64> MAP-I/BSC/HRR/FHRR) + its
// inspectable EXPLAIN record. `similarity` is a bare-`f64` measurement; SBC/MAP-B and the ADR-031
// element-space/complex carriers stay explicit never-silent refusals (OQ-3 / E20-1 gate).
pub use vsa_codegen::{
    emit_vsa_llvm_ir, resolve_model as resolve_vsa_model, vsa_compile, vsa_compile_and_run,
    VsaAotError, VsaArtifact, VsaCgOp, VsaExplain, VsaModelId, VsaProgram, VsaResult,
    FHRR_BUNDLE_PROFILE, HRR_BUNDLE_PROFILE, VSA_CODEGEN_GUARANTEE,
};
// M-855 (RFC-0039 §5.3): the dynamic-VSA JIT — in-process `dlopen` execution over the same real-
// Vec<f64> MAP-I/BSC/HRR/FHRR fragment as `vsa_codegen`, for data-dependent-dimension / runtime-model-
// selection workloads. `VsaJitArtifact`/errors/results are shared with the AOT path (`vsa_codegen`).
pub use vsa_jit::{vsa_jit_compile, vsa_jit_compile_and_run, VsaJitArtifact, VSA_JIT_GUARANTEE};

#[cfg(test)]
mod tests;
