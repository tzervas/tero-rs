# mycelium-lsp

> Minimal toolchain surface (FR-S5): the invariant linter (M-141), canonical formatter (M-142), and the LSP feedback facade (M-140/M-221) that exposes semantic-feedback artifact kinds — diagnostics, swap certificates, bound/guarantee annotations, and selection EXPLAIN traces — over one surface (SC-5 channel).

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-lsp` is the toolchain surface kept deliberately outside the small auditable kernel
(KC-3). It exposes the invariant linter, canonical formatter, RFC-0013 structured diagnostics with
the M-362 auto-baseline and routing, and the M-221 selection EXPLAIN channel ("why was this
representation chosen?" — RFC-0005 §4) over one facade. The crate also provides the LSP wire
protocol (JSON-RPC 2.0 over stdio), document sync (`didOpen`/`didChange` → parse → check →
elaborate), completions, hover, go-to-definition, semantic tokens, and an RFC-0014 recovery handler.

## Key items

- `analyze` / `Feedback` / `FeedbackSummary` — the semantic-feedback artifact: diagnostics, swap sites, guarantee annotations, lowering dumps.
- `lint` / `lint_structured_header` — the M-141 invariant linter and header linter.
- `format` — the canonical formatter surface.
- `present` / `ClassRegistry` / `DiagnosticPolicy` — RFC-0013 diagnostic presentation and M-362 baseline routing.
- `serve` / `serve_stdio` — LSP server entry points (JSON-RPC 2.0 over stdio).
- `recover` / `handle` — RFC-0014 recovery handler with effect budgets.
- `parse_llm_canonical` — LLM canonical form parser with depth limit.

## Design references

- M-140, M-141, M-142, M-221, M-310, M-345, M-362
- FR-S5
- RFC-0005 §4, RFC-0013, RFC-0014
- SC-5
- KC-3

## Role in the workspace

Depends on `mycelium-core`, `mycelium-interp`, `mycelium-cert`, `mycelium-select`, `mycelium-l1`, and `mycelium-proj`; provides the toolchain surface consumed by `mycelium-check`, `mycelium-lint`, and the LSP server. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-lsp).
