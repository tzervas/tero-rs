# mycelium-l1

> L1 surface prototype (RFC-0006/RFC-0007; NON-NORMATIVE until those RFCs are ratified): lexer, parser, typechecker, totality checker, evaluator, and elaborator to Core IR.

**Tier:** compiler  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-l1` is the Mycelium surface-language frontend. A hand-written lexer and recursive-descent parser validate the ratified DN-02 vocabulary against the `docs/spec/grammar/` conformance corpus (`accept/` parses, `reject/` is explicitly rejected). The v0 monomorphic typechecker and structural totality checker (`checkty`, `totality`) gate the `matured` annotation; the fuel-guarded big-step evaluator (`eval`) runs checked programs over the same trusted prim/swap engines as the L0 paths. The elaborator (`elab`) lowers the evaluation-complete fragment to closed Core IR terms — refusing everything else with an explicit `Residual`, never a partial artifact.

Pattern matching covers data types, `Binary`/`Ternary` literal arms, and nested patterns (M-320); exhaustiveness and redundancy are decided by the Maranget usefulness algorithm. The three-way differential (L1-eval ↔ elaborate→L0-interp ↔ AOT) lives in `tests/differential.rs` (NFR-7). Zero `unsafe` — compiler-enforced.

## Key items

- `parse` / `parse_phylum` — lexer + recursive-descent parser; every malformed input is an explicit `ParseError`, never a silent accept.
- `check_nodule` / `check_phylum` — v0 monomorphic typechecker + static guarantee grading (RFC-0018 stage-1a).
- `totality::Totality` — structural totality checker; gates `matured`.
- `eval::Evaluator` — fuel-guarded big-step L1 evaluator.
- `elab::elaborate` — lowers the evaluation-complete fragment to closed L0 `Node`s.
- `mono::monomorphize` — monomorphization for generic definitions.
- `usefulness` — Maranget exhaustiveness/redundancy checker for nested patterns.

## Design references

- RFC-0006, RFC-0007, RFC-0008, RFC-0011, RFC-0017, RFC-0018, ADR-014, DN-02, DN-03, DN-06, M-320, M-662, M-664, M-665, M-666, NFR-7

## Role in the workspace

Depends on `mycelium-core`, `mycelium-interp`, `mycelium-cert`, and `mycelium-stack`. The primary entry point for source-to-Core-IR compilation; `mycelium-mlir` is a dev-dependency for the three-way differential. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-l1).
