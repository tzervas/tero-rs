# mycelium-std-testing

> `std.testing` — property, golden, and differential test harness: a skipped or undetermined check is always reported, never a silent pass.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.testing` is the repo's verification discipline as a library (M-534, #174): property testing
(`for_all` with seed-reproducible shrinking), snapshot testing (`golden`), and oracle testing
(`differential`). The honesty crux is C1/G2 applied to the test report itself — a skipped or
undetermined check produces an explicit `Verdict::Skipped`, which aggregates distinctly from
`Verdict::Pass`, so "green" can never silently include "did not actually check". The harness never
upgrades a passing `for_all` run from `Empirical` to `Proven` (VR-5).

## Key items

- `for_all` — property test: seeded, shrink-to-minimal, reproducible; backs `Empirical` claims, never `Proven`.
- `golden` — snapshot test: missing baseline → `Verdict::Skipped{NeedsRecord}`, never auto-accept.
- `differential` — oracle test: unavailable backend → `Verdict::Skipped{BackendUnavailable}`, never silent pass.
- `summarize` / `is_green` — honest aggregator: `Skipped`/`Undetermined` stay distinct from `Pass`.
- `Verdict` / `FailRecord` / `SkipReason` — first-class, inspectable verdict types (C3/G11).
- `Rng` / `Gen<T>` / `Budget` — seeded generator surface (no undeclared entropy — C6/RT3).
- `guarantee_matrix::MATRIX` — all ops encoded as `Exact` tags, asserted in tests.

## Guarantee posture

All harness ops are `Exact` as mechanisms (a verdict is a deterministic function of the run); the
harness never inflates the subject crate's tag — a passing `for_all` backs `Empirical`, not `Proven`
(VR-5). Source is ground truth.

## Design references

- RFC-0016 §4.1 (C1–C6); RFC-0003/T0.2 (VSA capacity); ADR-003; M-151/M-210 (differential oracle pattern).
- Tasks: M-534 (#174).
- Spec: `docs/spec/stdlib/testing.md`.

## Role in the workspace

Ring 2 / Tier B verification harness; adds no trusted code and checks the trusted base. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-testing).
