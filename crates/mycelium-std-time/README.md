# mycelium-std-time

> `std.time` — typed clocks, durations, and instants: cross-source subtraction is a compile-time error; wall-clock reads are `Declared` + effectful, never dressed as pure values.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.time` is the value-semantic time surface for Mycelium (M-529): an immutable signed `Duration`
(i128 nanoseconds), three typed instant kinds (`MonoInstant`, `WallInstant`, `LogicalInstant`), pure
duration/instant arithmetic, and an injectable `ClockSource` that declares each read's effect on its
return type. The honesty crux is structural: cross-source subtraction (`MonoInstant − WallInstant`)
does not exist as a function — it is a compile-time type error. Wall-clock reads carry
`DeclaredTimeEntropy<T>`, so a deterministic fragment cannot read a wall clock without naming the
entropy effect. Overflow is `Err(Overflow)`, never a wrap or clamp. Arithmetic tags `Exact`; clock
reads tag `Declared` — no overclaiming (VR-5, C2).

## Key items

- `Duration` — signed i128-nanosecond span; all arithmetic is checked (`Err(Overflow)` on range exhaustion).
- `MonoInstant` / `WallInstant` / `LogicalInstant` — three typed, cross-source-incompatible instant kinds.
- `DeclaredTime<T>` / `DeclaredTimeEntropy<T>` — marker return types reifying `time` / `{ time, entropy }` effects (C6/RFC-0014).
- `ClockSource` — injectable clock trait; production uses `SystemClock`, tests use `ManualClock`.
- `mono_now` / `wall_now` / `logical_now` — free-function wrappers over `ClockSource` (all `Declared`).
- `duration_add` / `duration_sub` / `duration_scale` / `mono_diff` / `wall_diff` / `logical_diff` — exact arithmetic (all `Exact`).
- `GUARANTEE_MATRIX` — 11-row matrix (arithmetic=`Exact`, clock reads=`Declared`), asserted in tests.

## Guarantee posture

Pure arithmetic tags `Exact`; every clock read tags `Declared` and is effectful — a wall read is
never upgraded to a pure value (VR-5). Source is ground truth.

## Design references

- RFC-0016 §4.1 (C1–C6); RFC-0014 (effect declaration); RFC-0008 §4.7 (logical clock); ADR-014 (OS clock floor); ADR-003.
- Tasks: M-529.
- Spec: `docs/spec/stdlib/time.md`.

## Role in the workspace

Ring 2 / Tier B clock and duration surface; injectable `ClockSource` enables deterministic testing without touching OS state. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-time).
