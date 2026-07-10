# mycelium-select

> Selection-policy language: total, non-learned, content-addressed decision tables with an explicit cost function and mandatory EXPLAIN (RFC-0005; ADR-006; M-220/M-221/M-222).

**Tier:** kernel  ·  **Status:** Rust-first implementation  ·  **License:** MIT

## Overview

`mycelium-select` provides the representation selection machinery. A `SelectionPolicy` is an ordered decision table — `(predicate over queryable Meta) → candidate` — with a finite candidate set, an explicit `CostModel`, and a mandatory default arm. Predicates form a non-Turing-complete structural language so every policy is total and terminating by construction. Every selection emits an `Explanation` carrying the inputs considered, costs, matched rule, chosen option, and override state — no selection happens without one (M-221). Same inputs always produce the same choice (determinism).

Policies are content-addressed via `SelectionPolicy::policy_ref` (BLAKE3 over the canonical serialization) so "which policy chose this?" is always answerable from `Meta.policy_used`. One mechanism covers two sites: swap-target selection (`select_swap_target`, RFC-0002) and packing-schedule selection (`select_packing`, RFC-0004 §5). Zero `unsafe` — compiler-enforced.

## Key items

- `SelectionPolicy` — the decision table; `policy_ref()` gives the content-addressed `PolicyRef`.
- `select` / `explain` — the core selection entry points; always emit an `Explanation`.
- `Predicate` — the non-Turing-complete predicate language over `SelectionInputs`.
- `SelectionInputs` — the queryable inputs: `Repr`, `GuaranteeStrength`, `Bound`, `SparsityObs`, and optional `DecodeFacts`.
- `Explanation` — the mandatory per-selection trace.
- `CostModel` — the cost function used by `Action::Cheapest`.
- `select_swap_target` / `select_packing` — thin adapters for the two selection sites.

## Design references

- RFC-0005, RFC-0010, ADR-006, M-220, M-221, M-222

## Role in the workspace

Depends on `mycelium-core` only (KC-3). Used by `mycelium-vsa` (decode-method site) and the swap/packing selection sites. Dev-dependencies include `mycelium-cert` and `mycelium-interp` for the M-222 swap-site wiring test. See the [workspace overview](../../README.md). Further reading: the [doc index](../../docs/Doc-Index.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-select).
