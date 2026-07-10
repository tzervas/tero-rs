# mycelium-cli-common

> Small, dependency-free helper shared by the toolchain CLIs (M-643): folds out the duplicated stdin-or-file reader, the `.myc` source walker, and the hand-rolled argument loop.

**Tier:** tooling  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-cli-common` extracts three near-identical patterns that had drifted across `mycfmt`,
`myc-check`, `myc-lint`, and `myc-sec` into one place: `read_source` (stdin-or-file input),
`walk_myc` (recursive, sorted `.myc` source discovery), and `Args` (a transparent cursor over
`env::args()`). The contract is behaviour-preservation — each helper reproduces the bins' observable
behaviour byte-for-byte. The crate is `std`-only with no external dependencies (KC-3); a missing or
unreadable input is always a reported, structured outcome, never a hidden empty read (G2).

## Key items

- `read_source` — reads stdin (`-` sentinel) or a file path; prints a diagnostic to stderr and returns `Err(ReadError)` on any failure (never-silent, G2).
- `walk_myc` — recursively collects `.myc` files under a directory, sorted, skipping dotfiles and `target/`; an unreadable directory is an explicit `Err`.
- `Args` — a transparent cursor over `std::env::args().skip(1)`; names the value-flag idiom without changing parsing behaviour.
- `STDIN_SENTINEL` — the `-` constant shared across all toolchain CLIs.

## Design references

- M-643
- KC-3, G2

## Role in the workspace

Provides the shared CLI substrate (no external deps, `std`-only) consumed by `mycelium-fmt`, `mycelium-check`, `mycelium-lint`, and `mycelium-sec`. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-cli-common).
