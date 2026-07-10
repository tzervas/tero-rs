# mycelium-std-select

> `std.select` — selection DSL with mandatory EXPLAIN capability: every selection returns a choice and an inspectable explanation.

**Tier:** stdlib  ·  **Status:** implemented (Rust-first), pending ratification  ·  **License:** MIT

## Overview

`std.select` is the ergonomic library surface over `mycelium-select`: a total, non-learned,
content-addressed selection-policy DSL with a mandatory EXPLAIN record on every selection. Every
call to `select` or `select_with_override` returns `(candidate, Explanation)` — there is no code
path that yields a choice without its explanation (C3/SC-3). The `Explanation` carries the matched
rule, per-candidate costs in declared storage bits, the chosen candidate, and the content address
of the deciding policy, making "why this choice?" always answerable. All ops are `Exact` (the
policy is a total predicate over exact metadata — nothing probabilistic or learned).

## Key items

- `SelectionPolicy` — a validated, content-addressed, immutable decision table.
- `build` — validate and construct a policy; explicit `PolicyError` if non-total or malformed.
- `select` — evaluate the table and return `(Candidate, Explanation)` (`Exact`, fallible).
- `explain` — derive the `Explanation` for `(policy, inputs)` without performing selection (total).
- `select_with_override` — forced candidate by index; override state recorded in `Explanation`.
- `policy_ref` — content address (`PolicyRef`) of a validated policy (ADR-003).
- `GUARANTEE_MATRIX` — 5-row matrix encoded as data, asserted in tests (RFC-0016 §4.5).
- Site adapters: `select_swap_target`, `select_packing`, `select_decode_method`, `select_layout`.

## Design references

- RFC-0016 §4.1 (C1–C6); RFC-0005 (selection policy / EXPLAIN); ADR-003 (content addressing); M-220 (cost units in bits).
- Tasks: M-519.
- Spec: `docs/spec/stdlib/select.md`.

## Role in the workspace

Ring 1 / Tier A EXPLAIN capability surface; wraps `mycelium-select` (M-221) without adding trusted code. See the [workspace overview](../../README.md). Further reading: the [stdlib spec index](../../docs/spec/stdlib/README.md) and this crate's entry in the [agent code index](../../docs/api-index/INDEX.md#mycelium-std-select).
