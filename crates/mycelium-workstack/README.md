# mycelium-workstack

The shared **recursion-budget + guarded-stack** leaf for RFC-0041 (Wave-1). This is the *canonical
home* of the never-silent recursion budget every Mycelium execution machine charges against — the L1
evaluator, the L0 reference interpreter, and the AOT env-machine (RFC-0041 §4.1).

It deliberately extracts **only** the shared *budget + guarded-stack helper* — **not** a universal
`WorkStack<Frame>`. Each machine keeps its own bespoke frame/loop shape (a substitution machine, a CEK
env machine, a frame machine — §4.6); only the *counters, limits, and the never-silent over-budget
surface* live here.

## What is (and is not) here

| Item | Role |
| --- | --- |
| `RecursionBudget` | Per-invocation depth (§4.0 metric, default **4096**) + memory + work-step ceilings; tunable, deterministic defaults. |
| `BudgetError` | The canonical never-silent surface. `DepthExceeded { limit: u32 }` is *the* variant the interp/AOT `EvalError::DepthLimit` reconcile to (W4/W3½, §5.1). `OutOfBudget { kind, limit, requested }` for bytes/work-steps. |
| `DepthGuard` (via `try_enter`) · `charge_bytes` · `charge_steps` | Consumer-side charging. The charge happens at each machine's frame-push/env-insert site — never in this leaf. |
| `ProcessArena` / `ArenaReservation` | Process-wide memory ceiling (§4.2): a shared atomic byte counter so the *sum* over concurrent passes (LSP re-analyses, parallel eval workers, spore batch) cannot exceed a per-process ceiling. |
| `ensure_sufficient_stack` | Thin host-stack guard helper. **W1:** delegates to `mycelium_stack::with_deep_stack` (256 MiB worker). **W2:** a body swap to fine-grained `stacker::ensure_sufficient_stack` (runtime-gated grow); signature stays stable. |
| `assert_mem_ceiling_honors_floor` | The §4.2 determinism invariant `mem_ceiling >= depth_floor * max_frame_bytes`, as a checked function. **W1 provides it; W2 wires it at startup** (the per-machine `max_frame_bytes` census is W2). |

## Architecture (DN-68: acyclic, downward-only)

`mycelium-workstack` is a **leaf**: `#![forbid(unsafe_code)]`, depending on `std` and the
`mycelium-stack` host-stack adapter **only** — never on `mycelium-interp` / `mycelium-core` /
`mycelium-l1` (those are *upward*). This is the §4.1 deps-cycle fix: the leaf exposes only
counters/limits and the *charge happens consumer-side*, so no dependency cycle forms.

## House rules

- **Never-silent (G2):** every over-budget path returns a `BudgetError` — never a panic, `abort`, or
  silent truncation.
- **Honesty (VR-5):** the budgets are `Declared` (asserted config) and their sufficiency is `Empirical`
  (validated by trials/fixtures), **never** `Proven` — there is no machine-checked theorem here, only
  checked runtime guards.
- **Common-mode risk (§4.1):** because the three machines share this core, the differential can no
  longer cross-validate it — so `src/tests/isolation.rs` exercises the budget/guard against a *synthetic
  frame type* with mutant-witness cases (the crate is in `cargo-mutants` scope; a remove-guard mutant
  must not survive).

## API-shape note (`&self`, not `&mut self`)

`try_enter` / `charge_bytes` / `charge_steps` take **`&self`** with interior mutability, not `&mut self`.
A `&mut self`-borrowing RAII `DepthGuard` cannot satisfy the required "nested enters compose": the outer
guard's exclusive borrow would lock the budget for its whole scope, so an inner `try_enter` could not
reborrow, and charging alongside a live guard would not type-check. `&self` + `Cell` is the design that
makes nesting and concurrent charging work. `RecursionBudget` stays `Send` (movable into an
`ensure_sufficient_stack` worker) but is `!Sync` — cross-pass sharing goes through `ProcessArena`.
