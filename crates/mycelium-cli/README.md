# mycelium-cli

> `myc` — the one-command toolchain driver (M-733): `myc init|build|check|test|run` over a Mycelium phylum, with DN-22 structured, actionable diagnostics.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-cli` provides the single front door over the Mycelium toolchain. `myc init` scaffolds a
phylum, `myc build` packages it as a content-addressed spore (M-368), `myc check` type-checks it
via the L1 front-end, `myc test` runs available verification, and `myc run` executes a project's
`.myc` sources through the reference interpreter — a single source via the M-908 v0 path, two or
more via the M-909 multi-nodule path (manifest-driven project loading + nodule linking, with an
explicit, named `Report` for every unresolved/duplicate/cyclic nodule reference — never a silent
narrowing or a stub). Every user-visible failure is a structured `Report` with a stable code,
human-readable message, optional source location, and actionable help; no raw Rust panic ever
reaches the user (G2). The driver orchestrates real library APIs directly — no fragile subprocess
plumbing.

## Key items

- `myc` binary — the single toolchain entry point.
- `Report` — structured, actionable diagnostic (code, message, location, help, exit code); renders as `error[<code>]: <message>`.
- `init` / `build` / `check` / `test` subcommands — each does real end-to-end work.
- `run` subcommand — executes single-nodule (M-908) and multi-nodule (M-909) projects end-to-end through the reference interpreter; nodule-link failures (unresolved/duplicate/cyclic `use` refs) are explicit, named `Report`s.

## Design references

- M-733, M-368, M-359, M-908, M-909
- E16-1
- DN-22
- RFC-0013
- G2, VR-5

## Role in the workspace

Depends on `mycelium-proj`, `mycelium-spore`, `mycelium-l1`, and `mycelium-cli-common`; provides the unified toolchain CLI above the kernel (KC-3). See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-cli).
