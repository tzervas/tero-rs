# mycelium-std-math

> Ring-2 numeric function surface — abs, min/max, pow, sqrt, exp, log, trig, and rounding.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.math` provides the ordinary numeric-function surface over honest numerics (M-525). Exact
integer and rational ops tag `Exact`; every approximate result carries a declared `ErrorBound`
from the `mycelium-numerics` ε kernel (ADR-010) and is tagged `Declared` — never `Proven` —
because the transcendental compute floor (Rust `f64` intrinsics / platform libm) is not yet
audited (`wild`/FFI floor, FLAG M-541). Every domain restriction (`sqrt` of negative, `log` of
zero, division by zero, `tan` pole, `asin` out of range) is an explicit `Err(MathErr)` — never
a NaN, ±Inf, sentinel, or silent clamp (C1/G2).

## Key items

- `MathErr` — explicit domain-error enum: `DivByZero`, `NegativeDomain`, `NonPositiveDomain`, `BadBase`, `PoleDomain`, `OutOfDomain`, `Overflow`.
- `Approx<f64>` — approximate result carrier with attached `Bound` and `GuaranteeStrength`.
- `ApproxExplain` — EXPLAIN artifact for an `Approx` result (C3/SC-3).
- `RoundMode` — reified rounding mode for rounding ops.
- `GUARANTEE_MATRIX` — per-op guarantee table asserted in tests (RFC-0016 §4.5).

## Guarantee posture

Exact integer/rational ops: `Exact`. All approximate ops (transcendentals, rounding): `Declared`
(honest floor for unaudited libm/`wild` compute; will upgrade to `Proven`/`Empirical` when M-541
`std-sys` audit is delivered — VR-5).

## Design references

- RFC-0016 (core + standard library contract, C1–C6); ADR-010 (ε/δ bound kernels); ADR-014 (`wild`/FFI floor).
- Spec: `docs/spec/stdlib/math.md` (M-525, #166).

## Role in the workspace

Numeric function layer for all Ring-2 stdlib consumers and user-facing math operations. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-math).
