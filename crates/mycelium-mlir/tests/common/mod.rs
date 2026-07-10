//! Shared test helpers for the mycelium-mlir differential test suite.
//!
//! Only byte-for-byte identical helpers are placed here (M-376 P2). Helpers with
//! slight variants across files (e.g. `tern` with `i64`/`m` vs `Vec<Trit>`, `policy`
//! vs `policy_ref`) are kept local to their respective test files.
//!
//! Each test file is compiled as an independent binary; not every binary uses every item
//! here, so `dead_code` warnings are suppressed at the module level (the standard pattern
//! for shared Rust test helpers).
#![allow(dead_code)]

use mycelium_core::{GuaranteeStrength, Meta, Payload, Provenance, Repr, Trit, Value};

/// An 8-bit binary `Value` from a bit array. Identical across differential.rs,
/// native_differential.rs, jit_differential.rs, and wrong_layout.rs.
pub fn byte(bits: [bool; 8]) -> Value {
    Value::new(
        Repr::Binary { width: 8 },
        Payload::Bits(bits.to_vec()),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A ternary `Value` from a `Vec<Trit>` (m = trits.len()). Identical in
/// native_differential.rs and jit_differential.rs.
pub fn tern(trits: Vec<Trit>) -> Value {
    let m = trits.len() as u32;
    Value::new(
        Repr::Ternary { trits: m },
        Payload::Trits(trits),
        Meta::exact(Provenance::Root),
    )
    .unwrap()
}

/// A 64-bit binary `Value` encoding an `i64` (LSB-first). Used by simd_differential.rs
/// and specialize_differential.rs to route dot-product scalars through the M-210 checker.
pub fn i64_value(x: i64) -> Value {
    let bits: Vec<bool> = (0..64).map(|b| (x >> b) & 1 == 1).collect();
    Value::new(
        Repr::Binary { width: 64 },
        Payload::Bits(bits),
        Meta::exact(Provenance::Root),
    )
    .expect("64-bit value")
}

/// Fixed bit-pattern A: `[true, false, true, true, false, false, true, false]`.
/// Shared across differential.rs, native_differential.rs, jit_differential.rs, wrong_layout.rs.
pub const A: [bool; 8] = [true, false, true, true, false, false, true, false];

/// Fixed bit-pattern B: `[false, false, true, false, true, false, true, true]`.
/// Shared across native_differential.rs and jit_differential.rs.
pub const B: [bool; 8] = [false, false, true, false, true, false, true, true];

/// All-ones bit-pattern. Shared across differential.rs and native_differential.rs.
pub const ONES: [bool; 8] = [true; 8];

/// The observable triple extracted from a `Value`: `(repr, payload, guarantee)`.
/// Shared (identical) in differential.rs and native_differential.rs.
pub type Observable<'a> = (&'a Repr, &'a Payload, GuaranteeStrength);

/// Extract the observable triple from a `Value`.
/// Shared (identical) in differential.rs and native_differential.rs.
pub fn observable(v: &Value) -> Observable<'_> {
    (v.repr(), v.payload(), v.meta().guarantee())
}
