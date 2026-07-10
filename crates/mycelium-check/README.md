# mycelium-check

> Project-aware correctness/type-check driver (`myc-check`): resolves a `mycelium-proj.toml` project, checks the whole phylum/program, and aggregates every refusal as a structured RFC-0013 diagnostic routed via the M-362 auto-baseline.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-check` is the driver layer above the trusted M-210 checker (`mycelium_l1::check_nodule`).
It discovers `.myc` sources from a project manifest, checks the whole phylum, and aggregates every
parse and check refusal as a structured `Finding` with an RFC-0013 baseline level and route. The
driver exits non-zero on any error so CI can gate on it. It changes nothing about what the checker
decides — the trusted checker is the base (KC-3); this adds discovery, aggregation, and honest
routing. A `ParseError` or `CheckError` is always an explicit finding with a site (G2 — never a
silent pass); the driver does not fabricate a finer diagnostic class than the checker structurally
provides (VR-5).

## Key items

- `check_project` — resolves the manifest, walks sources, invokes the checker, aggregates findings.
- `Finding` / `FindingKind` — one aggregated diagnostic (Parse or Check) with baseline level and route.
- `Report` — the full aggregated result: all findings plus the file count checked.
- `myc-check` binary — CI-usable entrypoint; exits non-zero on any error.

## Design references

- M-365, M-362
- RFC-0013
- KC-3, G2, VR-5

## Role in the workspace

Depends on `mycelium-l1`, `mycelium-lsp`, `mycelium-proj`, and `mycelium-cli-common`; provides the project-scoped check driver for the toolchain. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-check).
