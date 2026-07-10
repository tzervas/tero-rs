# mycelium-lint

> `myc-lint` — lint and auto-fix (M-366): surfaces the M-141 invariant lints and header lints as actionable, reified, opt-in fixes with a `suggest`/`apply`/`scaffold` boundary.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-lint` surfaces the M-141 invariant lints and the M-358/M-359 header lints as actionable
findings, each with a reified fix offer and an explicit `FixTier` boundary: `suggest` (printed,
never auto-applied), `apply` (behaviour-preserving, applied only on `--fix`), and `scaffold` (an
incomplete skeleton the author completes — never auto-applied, because a control-flow change is
always the author's declared, bounded, opt-in choice). In v0, `--fix` applies nothing: every lint
fix maps to `suggest` or `scaffold`, so `myc-lint` v0 cannot silently rewrite code. The §4.1
documentation quality-bar lint (M-363 §6, 8 checks) is active: the check-name set lives in
`mycelium-doc` as the single source of truth and is re-exported here as `DOC_QUALITY_CHECKS`.

## Key items

- `lint_file` — lints a single `.myc` source and returns a `Vec<LintFinding>`.
- `LintFinding` — one finding with its optional reified `Fix`.
- `FixTier` — the opt-in boundary: `Suggest` / `Apply` / `Scaffold`.
- `Fix` — a reified fix offer with tier, description, and optional scaffold text.
- `DOC_QUALITY_CHECKS` — the M-363 §4.1 quality-bar check names (single source of truth in `mycelium-doc`).
- `myc-lint` binary — CLI entrypoint; supports `--fix` flag and project-wide walks.

## Design references

- M-366, M-141
- RFC-0014 I1/I5
- G2, KC-3

## Role in the workspace

Depends on `mycelium-l1`, `mycelium-lsp`, `mycelium-doc`, and `mycelium-cli-common`; provides the lint-and-fix tooling above the kernel. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-lint).
