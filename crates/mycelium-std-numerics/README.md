# mycelium-std-numerics

> ε/δ bound carrier and meet-composition surface for the Mycelium standard library.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.numerics` is the Ring-2/Tier-B ε/δ error-bound surface (M-512). It wraps approximate
numeric results in `Approx<T>`, a carrier that holds a value, its `ErrorBound`, and a
`GuaranteeStrength` tag. Constructors enforce the honesty rule: `declared` accepts any bound,
`empirical` requires a trial corpus, and `proven` requires a sealed `ProvenThm` witness token —
you cannot claim `Proven` without a checked theorem. The `EXPLAIN` surface (`Explanation`) is
always inspectable (C3/SC-3). `check_error` and `check_union` let callers assert that a bound
meets a threshold, returning an explicit `Err(CheckErr)` rather than silently passing.

## Key items

- `Approx<T>` — error-bound carrier (`value`, `ErrorBound`, `GuaranteeStrength`).
- `ProvenThm` — sealed witness token; required to construct a `Proven`-tagged `Approx`.
- `Explanation` — EXPLAIN artifact for an `Approx` result (C3/SC-3).
- `NumErr` — numeric domain error (e.g. negative probability, invalid bound).
- `CheckErr` — error returned when a bound assertion fails.
- `error_bound` — construct an `Approx` with a `Declared` error bound.
- `prob_bound` — construct an `Approx` with a declared probability bound.
- `union_delta` — meet-compose two `Approx` bounds (worst-case union).
- `accuracy_to_probability` — convert an accuracy target to a probability bound.
- `check_error` / `check_union` — assert a bound meets a threshold; `Err(CheckErr)` on failure.
- `DECLARED_FLOAT_EPS` — `2.0 * f64::EPSILON`; the honest declared floor for f64 approximations.
- `ErrorBound`, `ErrorOp`, `KernelProbBound` — re-exported from `mycelium-numerics`.

## Design references

- RFC-0016 (core + standard library contract, C1–C6); ADR-010 (ε/δ bound kernels); ADR-011.
- Spec: `docs/spec/stdlib/numerics.md` (M-512).

## Role in the workspace

Error-bound primitives consumed by `std.math`, `std.dense`, and any crate that carries `Approx` results. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-numerics).
