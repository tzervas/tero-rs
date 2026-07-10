# mycelium-std-runtime

> `std.runtime` — the fungal concurrency surface: Colony/Scope structured concurrency, Task/Network,
> OS-thread scheduler, deadlock-free dataflow, supervision, and the E12 three-layer memory model.

**Tier:** runtime  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.runtime` implements the v0 R1 API surface decided in ADR-020. It provides structured concurrency
(`Colony`/`Scope`), cooperative tasks (`Task`/`TaskCtx`/`Poll`), sweep ordering and deadlock
detection (`SweepOrder`/`Deadlock`), and a bounded-backpressure channel surface (`Network`/`Sender`/
`Receiver`). Beyond the cooperative surface it now executes on real OS threads: a fixed-pool
`Scheduler` (M-709) with fair FIFO dispatch; `run_dataflow`/`run_dataflow_scheduled` (M-711) that
surface no-progress as an explicit `Deadlock`, never a silent hang (G2); and `supervision` (M-713)
providing structured-cancellation (`CancelTree`), explicit per-child outcome collection, and an
EXPLAIN-able bounded-cascade restart policy.

The crate also hosts the E12 three-layer memory model runtime (DN-32 / RFC-0027): reclamation EXPLAIN/
audit records (`reclamation`), a non-atomic intra-hypha RC cell (`rc`), region-based batched
scope-exit reclamation (`region`), live-executor scope/region wiring (`scope_region`), and the
`ChannelClose` third reclamation trigger in `network`. The three triggers are RcZero, ScopeExit,
and ChannelClose; all emit explicit `ReclamationRecord`s, never silent frees.

## Key items

- `colony::Colony` / `colony::Scope` — structured-concurrency root and scope.
- `task::Task` / `task::TaskCtx` / `task::Poll` — cooperative task surface.
- `task::SweepOrder` / `task::Deadlock` — sweep ordering and explicit deadlock type.
- `network::Network` / `network::Sender` / `network::Receiver` — bounded channel surface.
- `scheduler::Scheduler` — fixed OS-thread pool with fair FIFO dispatch (M-709).
- `dataflow::run_dataflow` / `run_dataflow_scheduled` — deadlock-free communicating-task execution (M-711).
- `supervision::CancelTree` / `run_supervised` / `supervise_with_restart` — cancellation + supervision (M-713).
- `reclamation` (MEM-1) — reclamation EXPLAIN/audit records and never-silent sink contract.
- `rc` (MEM-2) — non-atomic intra-hypha RC cell + rc-probe decision.
- `region::Region` / `ScopeNodeId` / `RegionEpoch` (MEM-3) — region-based batched scope-exit reclamation.
- `scope_region::RegionScope` — live-executor wiring tying a `Region` lifecycle to a structured scope.
- `guarantee_matrix::MATRIX` — per-op guarantee tags encoded as data, asserted in tests.

## Guarantee posture

Per-op guarantee tags are encoded in `guarantee_matrix::MATRIX` and asserted in tests (never
prose-only). The RT2 sequentialization differential is property-tested (`Empirical`). Source is
ground truth.

## Design references

- ADR-020 (Colony/Scope placement); RFC-0008 (runtime contract RT1–RT7); DN-32 (three-layer memory
  model); RFC-0027 (reclamation EXPLAIN/audit); DN-33 (MEM-4 static uniqueness).
- Tasks: M-521, E12-1 (M-709/M-711/M-713).
- Spec: `docs/spec/stdlib/runtime.md`.

## Role in the workspace

The fungal concurrency and memory-model runtime; depends on `mycelium-rt-abi` (M-883/M-884) for the
relocated reclamation (MEM-1) and supervision (M-713, reusing the M-356 composition kernel)
surfaces, both re-exported here unchanged for existing consumers. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-runtime).
