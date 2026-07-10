# mycelium-bench

> Honest benchmarking and evaluation harness (E-BENCH): measures the existing execution backends over a shared v0-calculus corpus and emits a deterministic WIN/LOSS/REGRESSION report.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-bench` is the measurement counterpart to the whole project — it tells us where Mycelium
wins and where it loses, across execution backends, and surfaces regressions rather than hiding
them. Over a shared corpus of v0-calculus programs, it times the interpreter (the trusted
differential baseline), the AOT env-machine, the JIT, the direct-LLVM backend, and (behind the
`mlir-dialect` feature) the MLIR-dialect path. For each (backend, case) pair it captures wall time
and result, then classifies the result against the interpreter into a `Verdict`: speed WIN/LOSS,
correctness LOSS (differential divergence), capability LOSS (unlowerable node with reason), runtime
error, or environmental skip. It also ingests the LLM-harness report so language-leverage data
(KC-2/SC-5b: quality, latency, token cost) sits alongside execution data. Every measured number is
`Empirical`; a debug build is refused for performance numbers; a differential divergence is always
recorded as a correctness LOSS (VR-5 — no pre-written performance target).

**Multicore scaling + regression gates (M-859).** Beyond the single-core WIN/LOSS table, the harness
can also measure how a *batch of independent programs* scales across `1..=N` OS worker threads (via
the real `mycelium-std-runtime` `Scheduler`, M-709), and gate a fresh run's timings against a
committed baseline JSON captured on a specific host. Both are opt-in (`--scaling [N]` /
`--baseline <FILE>` on the `bench` binary) and purely measurement — neither changes any backend's
semantics or execution path. Every scaling/regression figure is `Empirical` with an explicit trial
count; the Amdahl serial-fraction estimate is an explicitly-labeled coarse two-point fit, never a
target.

## Key items

- `run_corpus` / `RunRecord` / `CaseRecord` — measure all corpus cases across all backends.
- `Backend` / `Engines` / `Outcome` — the execution backend abstraction (`is_process_spawn_bound`
  flags the compiled paths that exec a fresh native process per call).
- `corpus` / `Case` / `Fragment` — the shared v0-calculus program corpus.
- `classify` / `Verdict` / `Speed` — classify a measured outcome against the interpreter baseline.
- `run_scaling` / `ScalingRun` / `ScalingPoint` / `ScalingOutcome` — multicore scaling curves over
  the OS-thread `Scheduler` (M-859).
- `verdict::regression_classify` / `verdict::RegressionBaseline` / `verdict::RegressionOutcome` —
  this-run-vs-committed-baseline regression gating, host-tag-scoped (M-859).
- `Report` / `Honesty` / `LlmSection` / `report::RegressionSection` — the deterministic markdown + JSON
  report with the WIN/LOSS table, the scaling section, and the regression-gate section.
- `parse_any_llm_json` / `LlmReport` — LLM-harness report ingestion.
- `bench` binary — the harness entry point (`--out <DIR>`, `--stdout`, `--scaling [N]`,
  `--baseline <FILE>`).
- `baselines/BASELINE-<host-tag>.json` — committed regression baselines (see
  [`baselines/`](baselines/)), one per host tag they were captured on.

## Design references

- E-BENCH, M-859
- NFR-7, ADR-007
- M-212, M-250, M-303, M-340, M-360
- RFC-0008 RT1·RT2 (the `Scheduler` the scaling suite drives)
- KC-2, SC-5b
- VR-5, G2

## Role in the workspace

Depends on `mycelium-core`, `mycelium-interp`, `mycelium-mlir`, `mycelium-l1`, `mycelium-cert`, and
`mycelium-std-runtime` (the OS-thread `Scheduler` the scaling suite drives); measures the backends
without modifying them. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-bench).
