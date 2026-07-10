//! `mycelium-numerics` — the **verified-numerics foundation** (E2-4; ADR-010; RFC-0001 §4.7).
//!
//! Two bound kernels meeting at one shared certificate, exactly as ADR-010 decides:
//!
//! - the [`error`] kernel composes ε-magnitude bounds through **affine arithmetic**
//!   ([`ErrorBound`], [`AffineForm`]); and
//! - the [`prob`] kernel composes δ failure-probability bounds through the **union bound** and the
//!   apRHL `[SEQ]` rule ([`ProbBound`], [`ApRhlJudgment`]).
//!
//! They are *different monoids* (ADR-010/T0.1c — a settled negative result) that meet at the shared
//! [`Certificate`] `{ε, δ, strength}`, where `strength` composes by `meet`. The **tier-i Rust
//! checker** ([`check_error_claim`]/[`check_union_claim`]) re-derives a composition and rejects any
//! claim tighter than the re-derivation — never a silent pass (ADR-010 "Trusted base"; RFC-0002 §2).
//! The one sanctioned cross-kernel inference is accuracy→probability ([`accuracy_to_probability`]).
//!
//! This crate is a certificate *consumer*, deliberately **outside** `mycelium-core` (KC-3 / SoC):
//! the trusted kernel stays small while honest approximation composes here.
//!
//! All three normative composition properties (RFC-0001 §4.7) are property-tested in
//! `tests/properties.rs`: **Soundness**, **Monotonicity**, **Determinism**.
//!
//! **Trusted-base discipline (ADR-014 / DN-21 §5 F-1):** zero `unsafe` — compiler-enforced.
#![forbid(unsafe_code)]

pub mod cert;
pub mod error;
pub mod prob;
mod round;

pub use cert::{
    accuracy_to_probability, basis_strength, check_error_claim, check_union_claim,
    compose_error_bound, error_norm, recompute_error, Certificate, CheckOutcome, ComposedBound,
    ErrorOp,
};
pub use error::{AffineForm, ErrorBound, NoiseSym};
pub use prob::{ApRhlJudgment, ProbBound};
