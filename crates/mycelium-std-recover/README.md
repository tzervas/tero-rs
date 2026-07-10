# mycelium-std-recover

> `std.recover` — the declarative recovery bridge: every error is recovered or re-propagated, never dropped.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.recover` is the ergonomic library surface of `std.recover`, implementing the Rust-first half of
M-520. Recovery elaborates to an L0 `Match` over the error sum — no new kernel node (RFC-0014 §4.3 /
KC-3). The core invariant is I1: `handle_classified` always yields `Resolution::Recovered` or
`Resolution::Propagated` — there is no `Dropped` variant, so every error is explicitly accounted for.
Tags are inherited on `Retry` success, at most `Declared` on `Fallback`, and `Exact` on an `Ok`
pass-through, and are never laundered upward (I2/VR-5).

## Key items

- `RecoveryAction` — the v0 closed action set: `Fallback` / `Retry` / `Escalate` / `CleanupThenPropagate`.
- `RecoveryPolicy<T>` — reified, content-addressed policy; `PolicyRef = ContentHash` (RFC-0005/ADR-006).
- `Outcome<T,E>` / `Resolution<T,E>` — input and output sums; no `Dropped` variant (I1).
- `handle_classified` / `recover_classified` — never-silent drivers: `Recovered | Propagated`.
- `check_effects` / `Budgets` / `EffectBudgetExhausted` — undeclared-effect detection and budget enforcement (I3/I4).
- `ClassRegistry` / `ClassName` — minimal error-class registry (Rust-first stand-in for `std.diag`).
- `guarantee_matrix::MATRIX` — RFC-0016 §4.5 per-op tags encoded as data, asserted in tests.

## Guarantee posture

Per-op guarantee tags are encoded in the guarantee matrix and asserted in tests (never prose-only).
`handle_classified` carries `Exact` on a clean `Ok` path; `Declared` at most for a `Fallback`
substitution. Tags are never upgraded past their basis (VR-5). Source is ground truth.

## Design references

- RFC-0016 §4.1 (C1–C6); RFC-0014 (declared effects); ADR-006 (content-addressed policies).
- Tasks: M-520 (#156).
- Spec: `docs/spec/stdlib/recover.md`.

## Role in the workspace

Ring 1 / Tier A declarative error-recovery bridge over the `mycelium-diag` record. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-recover).
