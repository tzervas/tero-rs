//! In-crate test modules for `mycelium-mlir` (CLAUDE.md test-layout rule).
//! White-box access via `use crate::…::*`; logic files carry no `#[cfg(test)]` inline code.

mod accel;
mod aot;
mod bitnet;
mod dense_codegen;
mod dialect;
mod inject_gate_tests;
mod inject_policy_tests;
mod inject_tests;
mod jit;
mod llvm;
mod mode;
mod trampoline;
// `dialect::native` only compiles under `mlir-dialect`, so its white-box tests are gated to match.
#[cfg(feature = "mlir-dialect")]
mod native;
// `dialect::native::{dense,vsa}` (M-856b) — same feature gate as `native` above.
#[cfg(feature = "mlir-dialect")]
mod native_dense;
#[cfg(feature = "mlir-dialect")]
mod native_vsa;
mod passes;
mod rc_plan_tests;
mod swap_codegen;
mod vsa_codegen;
mod vsa_jit;
