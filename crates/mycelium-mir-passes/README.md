# mycelium-mir-passes

> MEM-4 (DN-33): the RC-annotated IR and reference-counting lowering passes (static uniqueness analysis / Perceus-style RC emission and elision). Optimisation-only and OUTSIDE the trusted Core IR (KC-3).

**Tier:** compiler  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-mir-passes` implements the MEM-4 leg of the DN-32 three-layer memory model. The design was ratified in DN-33 (status: Accepted). This crate is optimisation-only and deliberately outside the trusted Core IR: it consumes `mycelium_core::Node` read-only and produces a separate RC-annotated IR (`rc_ir::RcNode`). A bug here is a missed optimisation — never unsafety — because the runtime `RcCell` probe (`mycelium-std-runtime::rc`) remains the sound fallback (DN-33 §2). Zero `unsafe` — compiler-enforced.

What is built (MEM-4 B0 + Increment 1 + Increment 2): naive fully-owned RC emission (`emit::emit_owned`), borrow elision for non-escaping let-bindings (`emit::emit_elided`), `rc == 1` reuse annotation for sole-owned single moves (`emit::emit_reuse`), a structural balance invariant checker (`balance`), and a reference RC-evaluator differential (`eval::differential`) that checks owned vs elided emissions reclaim the same multiset with no use-after-free while strictly reducing `Dup` count. Recursion (`Fix`/`FixGroup`) is refused explicitly and is a later increment.

## Key items

- `rc_ir::RcNode` — the RC-annotated IR mirror of the Core IR first-order fragment plus `Dup`/`Drop`/`Borrow`/`DropAfter`/`MoveUnique` wrappers.
- `rc_ir::Mode` — per-binding ownership mode: `Owned` or `Borrowed`.
- `emit::emit_owned` — naive fully-owned RC emission (`Node → RcNode`): `k` uses → `k-1` `Dup`s, unused binding → `Drop`.
- `emit::emit_elided` — borrow-elision pass: non-escaping `let` bindings become `Borrow`/`DropAfter`, zero `Dup`s.
- `emit::emit_reuse` — superset of `emit_elided`: a sole-owned single move (`emit::is_sole_owned_move`) is emitted as `RcNode::MoveUnique`, recording a statically-guaranteed `rc == 1` reuse site.
- `balance::check_balance` — structural balance invariant: `1 + dups == uses + drops` per owned binding.
- `eval::differential` — the reference RC-evaluator differential (the Q3 soundness check from DN-33 §8.1); `eval::RcError::UnsoundUnique` machine-verifies every `MoveUnique` annotation.

## Guarantee posture

The balance invariant is `Exact` (by construction, independently checked). No performance claim is made for the `Dup`/`Drop` reduction — any count figure stays `Declared` until measured on a corpus (DN-33 §8.1 Q5). The differential's agreement property is `Empirical` over the tested corpus.

## Design references

- DN-33, DN-32, M-654, E12, KC-3

## Role in the workspace

Depends on `mycelium-core` only (read-only consumer). The MEM-4 follow-ons (Increment 2/3, recursion, interprocedural) are tracked as M-797. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-mir-passes).
