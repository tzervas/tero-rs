# mycelium-interp

> Reference interpreter — the trusted executable small-step semantics for the Core IR (RFC-0004; ADR-009; M-110).

**Tier:** compiler  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-interp` is the *meaning* of a Mycelium program: a call-by-value, small-step substitution evaluator over closed Core IR `Node`s. The AOT path (M-150/M-151) is differential-tested against it — not the other way round. Errors are always explicit (`EvalError`) and the interpreter is never silent (SC-3/G2). Zero `unsafe` — compiler-enforced.

The evaluator covers the full v0 calculus: let-bindings, primitive ops, swaps, algebraic data (`Construct`/`Match`), first-class functions (`Lam`/`App`), `Fix` (structural recursion with a fuel clock), and mutual recursion (`FixGroup`). Approximate composition is refused where no ε-propagation rule is defined (ADR-010/M-204).

## Key items

- `Interpreter` — the reference interpreter: a `PrimRegistry` + `SwapEngine`, iterating `step` to a normal form.
- `Interpreter::step` — one small-step reduction (the `⟶` relation from RFC-0004 §2).
- `Interpreter::eval` / `eval_core` — multi-step evaluation to a `Value` or `CoreValue` (repr + data fragment).
- `EvalError` — exhaustive explicit refusal type covering free variables, type errors, overflow, fuel exhaustion, depth limits, effect budgets, and swap failures.
- `PrimRegistry` — dispatch table for named primitive operations (extensible via `SwapEngine`).
- `Supervisor` / `CancelToken` — structured concurrency primitives for the runtime layer.
- `Budgets` / `EffectBudget` — named effect-budget ledger (RFC-0014 §4.5/§4.8).

## Guarantee posture

Metadata is threaded honestly: an `Op`/`Swap` result's guarantee is the `meet` of its inputs and the operation's own intrinsic strength (RFC-0001 §4.7). Provenance is `Derived{op, inputs}` over content hashes. A free variable, unknown primitive, or unsupported swap is always an explicit `EvalError`, never a silent default.

## Design references

- RFC-0004, RFC-0007, RFC-0011, RFC-0014, ADR-009, ADR-010, ADR-014, M-110, M-120, M-204, NFR-7

## Role in the workspace

Depends on `mycelium-core` and `mycelium-numerics`. Used by `mycelium-l1`, `mycelium-mlir`, `mycelium-cert`, and the differential test harness. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-interp).
