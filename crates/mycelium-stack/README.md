# mycelium-stack

> Host-stack management for the L1 frontend's recursive passes (checker/elaborator), kept outside the trusted kernel so `mycelium-l1` stays `unsafe`-free and auditable (ADR-014; KC-3).

**Tier:** compiler  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-stack` provides `with_deep_stack`: a function that runs a recursive compiler pass (checker or elaborator) on a dedicated worker thread with a large, lazily-committed 256 MiB stack. The address space is reserved up front and physical pages are committed only as recursion actually deepens — so a shallow program pays nearly nothing. Zero `unsafe` — pure `std::thread`.

The design is intentionally transitional: the kernel's explicit depth budgets (`MAX_EXPR_DEPTH`, `MAX_CHECK_DEPTH`, the evaluator's fuel clock) are the portable primitive that carries to the future Mycelium-native frontend. This crate is the Rust-host adapter that ensures those budgets — not a host-stack overflow — are what stops a pathological input. It is expected to disappear when the frontend self-hosts.

## Key items

- `with_deep_stack` — runs a closure on a 256 MiB worker thread stack; panics propagate unchanged. `unsafe`-free; pure `std::thread`.
- `DEEP_STACK_BYTES` — the 256 MiB stack constant (comfortably exceeds the checker's 4096-level budget at any measured frame size).

## Design references

- ADR-014, KC-3, RFC-0007

## Role in the workspace

No upstream Mycelium dependencies. Used by `mycelium-l1` (checker and elaborator). The grow-on-demand hybrid (optional `stacker` feature) is documented in `Cargo.toml` but off by default; it would contain any upstream `unsafe` in a single audited leaf, never in Mycelium-authored source. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-stack).
