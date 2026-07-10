# Mycelium honest benchmark report

> Tool `mycelium-bench` — profile `release` — `mlir-dialect` feature: OFF — host: x86_64-linux, 4 hw threads (provenance only)

Guarantee lattice: `Exact ⊐ Proven ⊐ Empirical ⊐ Declared`.

**Honesty:** Every measured number is Empirical (a trial mean with its trial count + spread); a capability loss / skip / runtime error is Declared. No verdict is Proven or Exact, and no performance target is pre-written (VR-5). A differential divergence from the trusted interpreter is a recorded correctness LOSS; an unlowerable node is a recorded capability LOSS (G2 — never omitted).

Speed band: a backend within ±10% of the interpreter is *neutral*; faster is a **WIN**, slower a **LOSS (speed)**. Trusted baseline: the **interpreter** (in-process; NFR-7/ADR-007).

Tally across the run: **2 win(s)**, 0 neutral, **26 speed-loss(es)**, **0 correctness-loss(es)**, **14 capability-loss(es)**, 0 runtime-error(s), 14 skip(s).

**Microbench caveats (honest):** numbers are warmup + min-mean over batches via `std::time::Instant` (no `criterion`). The compiled native paths (`direct-llvm`, `mlir-dialect`) are **process-spawn-bound**: each invocation execs a fresh native artifact, so for a trivial kernel the per-invocation figure is spawn-dominated, **not** kernel compute (the honest M-602/E1 finding — surfaced, not buried). `jit` runs in-process (`dlopen`) so it is not spawn-bound. A debug build is refused for perf numbers.

## WIN / LOSS / regression table

Each non-baseline backend vs the interpreter, per case. `ratio` is `interp / backend` (>1 ⇒ backend faster). Tag is per-row.

| case | fragment | backend | verdict | ratio | tag | reason / detail |
|---|---|---|---|---|---|---|
| `bit-literal` | bit-subset | `aot-env` | LOSS (speed) | 0.01x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-literal` | bit-subset | `jit` | LOSS (speed) | 0.00x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-literal` | bit-subset | `direct-llvm` | LOSS (speed) | 0.00x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-literal` | bit-subset | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `bit-not` | bit-subset | `aot-env` | LOSS (speed) | 0.06x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-not` | bit-subset | `jit` | LOSS (speed) | 0.04x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-not` | bit-subset | `direct-llvm` | LOSS (speed) | 0.00x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-not` | bit-subset | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `bit-xor-not` | bit-subset | `aot-env` | LOSS (speed) | 0.14x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-xor-not` | bit-subset | `jit` | LOSS (speed) | 0.10x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-xor-not` | bit-subset | `direct-llvm` | LOSS (speed) | 0.00x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-xor-not` | bit-subset | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `bit-let-chain` | bit-subset | `aot-env` | LOSS (speed) | 0.23x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-let-chain` | bit-subset | `jit` | LOSS (speed) | 0.19x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-let-chain` | bit-subset | `direct-llvm` | LOSS (speed) | 0.01x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-let-chain` | bit-subset | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `trit-neg` | bit-subset | `aot-env` | LOSS (speed) | 0.06x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-neg` | bit-subset | `jit` | LOSS (speed) | 0.04x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-neg` | bit-subset | `direct-llvm` | LOSS (speed) | 0.00x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `trit-neg` | bit-subset | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `trit-add` | bit-subset | `aot-env` | LOSS (speed) | 0.09x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-add` | bit-subset | `jit` | LOSS (speed) | 0.06x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-add` | bit-subset | `direct-llvm` | LOSS (speed) | 0.00x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `trit-add` | bit-subset | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `swap-roundtrip` | swap | `aot-env` | LOSS (speed) | 0.15x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `swap-roundtrip` | swap | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: swap to Ternary { trits: 6 } (the subset is straight-line bit/trit ops; M-301) |
| `swap-roundtrip` | swap | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: swap to Ternary { trits: 6 } (the subset is straight-line bit/trit ops; M-301) |
| `swap-roundtrip` | swap | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `data-match-repr` | data | `aot-env` | LOSS (speed) | 0.08x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-match-repr` | data | `jit` | LOSS (speed) | 0.05x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-match-repr` | data | `direct-llvm` | LOSS (speed) | 0.00x | Empirical | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `data-match-repr` | data | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `data-construct` | data | `aot-env` | LOSS (speed) | 0.02x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-construct` | data | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-construct` | data | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-construct` | data | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `data-nested-match` | data | `aot-env` | LOSS (speed) | 0.12x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-nested-match` | data | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-nested-match` | data | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-nested-match` | data | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `rec-self` | recursion | `aot-env` | LOSS (speed) | 0.62x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `rec-self` | recursion | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-self` | recursion | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-self` | recursion | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `rec-build` | recursion | `aot-env` | WIN | 1.17x | Empirical |  |
| `rec-build` | recursion | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-build` | recursion | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-build` | recursion | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `rec-mutual` | recursion | `aot-env` | WIN | 1.13x | Empirical |  |
| `rec-mutual` | recursion | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: FixGroup: mutual recursion is not supported in Increment-3 (only single Fix with a λparam.Match body is supported; RFC-0004 §11.6; G2) |
| `rec-mutual` | recursion | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: FixGroup: mutual recursion is not supported in Increment-3 (only single Fix with a λparam.Match body is supported; RFC-0004 §11.6; G2) |
| `rec-mutual` | recursion | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |
| `rec-fold` | recursion | `aot-env` | LOSS (speed) | 0.25x | Empirical | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `rec-fold` | recursion | `jit` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-fold` | recursion | `direct-llvm` | LOSS (capability) | — | Declared | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-fold` | recursion | `mlir-dialect` | skipped | — | Declared | the `mlir-dialect` feature is off (build with --features mlir-dialect) |

## Per-case timings (ns/call, Empirical)

Interpreter baseline + each backend that produced a timed value. The best ns/call is shown; the worst/best spread (a noise flag) is in the JSON projection (`ns_per_call_worst`), omitted from this compact table. `—` = not timed (skip / capability loss / error).

| case | interp ns | aot-env ns | jit ns | direct-llvm ns | mlir-dialect ns |
|---|---|---|---|---|---|
| `bit-literal` | 101.8 | 21.6k | 39.4k | 1.29M | — |
| `bit-not` | 1.5k | 24.6k | 39.3k | 1.25M | — |
| `bit-xor-not` | 3.9k | 27.7k | 40.4k | 1.27M | — |
| `bit-let-chain` | 7.6k | 32.7k | 39.5k | 1.26M | — |
| `trit-neg` | 1.5k | 25.0k | 39.7k | 1.29M | — |
| `trit-add` | 2.2k | 25.3k | 39.3k | 1.26M | — |
| `swap-roundtrip` | 4.4k | 28.5k | — | — | — |
| `data-match-repr` | 1.9k | 25.3k | 40.1k | 1.26M | — |
| `data-construct` | 403.0 | 23.2k | — | — | — |
| `data-nested-match` | 4.5k | 39.3k | — | — | — |
| `rec-self` | 34.4k | 55.8k | — | — | — |
| `rec-build` | 57.0k | 48.7k | — | — | — |
| `rec-mutual` | 72.8k | 64.4k | — | — | — |
| `rec-fold` | 45.1k | 179.8k | — | — | — |

One-time compile cost (emit IR → toolchain → native, NOT in the per-run figures above):

- `bit-literal` / `jit`: 78.91M (one-time)
- `bit-literal` / `direct-llvm`: 122.97M (one-time)
- `bit-not` / `jit`: 82.88M (one-time)
- `bit-not` / `direct-llvm`: 118.85M (one-time)
- `bit-xor-not` / `jit`: 89.16M (one-time)
- `bit-xor-not` / `direct-llvm`: 120.45M (one-time)
- `bit-let-chain` / `jit`: 79.54M (one-time)
- `bit-let-chain` / `direct-llvm`: 122.67M (one-time)
- `trit-neg` / `jit`: 82.78M (one-time)
- `trit-neg` / `direct-llvm`: 124.43M (one-time)
- `trit-add` / `jit`: 82.47M (one-time)
- `trit-add` / `direct-llvm`: 128.39M (one-time)
- `data-match-repr` / `jit`: 87.30M (one-time)
- `data-match-repr` / `direct-llvm`: 120.32M (one-time)

## Where we're losing (explicit)

### Capability losses (a backend cannot lower the program — the reason, never omitted, G2)

| case | backend | reason |
|---|---|---|
| `swap-roundtrip` | `jit` | unsupported node for the AOT subset: swap to Ternary { trits: 6 } (the subset is straight-line bit/trit ops; M-301) |
| `swap-roundtrip` | `direct-llvm` | unsupported node for the AOT subset: swap to Ternary { trits: 6 } (the subset is straight-line bit/trit ops; M-301) |
| `data-construct` | `jit` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-construct` | `direct-llvm` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-nested-match` | `jit` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `data-nested-match` | `direct-llvm` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-self` | `jit` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-self` | `direct-llvm` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-build` | `jit` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-build` | `direct-llvm` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-mutual` | `jit` | unsupported node for the AOT subset: FixGroup: mutual recursion is not supported in Increment-3 (only single Fix with a λparam.Match body is supported; RFC-0004 §11.6; G2) |
| `rec-mutual` | `direct-llvm` | unsupported node for the AOT subset: FixGroup: mutual recursion is not supported in Increment-3 (only single Fix with a λparam.Match body is supported; RFC-0004 §11.6; G2) |
| `rec-fold` | `jit` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |
| `rec-fold` | `direct-llvm` | unsupported node for the AOT subset: Construct field: expected a repr lane but found a data value |

### Speed losses (slower than the in-process interpreter — measured, with the derivable reason)

| case | backend | ratio (interp/backend) | reason |
|---|---|---|---|
| `bit-literal` | `aot-env` | 0.01x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-literal` | `jit` | 0.00x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-literal` | `direct-llvm` | 0.00x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-not` | `aot-env` | 0.06x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-not` | `jit` | 0.04x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-not` | `direct-llvm` | 0.00x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-xor-not` | `aot-env` | 0.14x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-xor-not` | `jit` | 0.10x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-xor-not` | `direct-llvm` | 0.00x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `bit-let-chain` | `aot-env` | 0.23x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-let-chain` | `jit` | 0.19x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `bit-let-chain` | `direct-llvm` | 0.01x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `trit-neg` | `aot-env` | 0.06x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-neg` | `jit` | 0.04x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-neg` | `direct-llvm` | 0.00x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `trit-add` | `aot-env` | 0.09x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-add` | `jit` | 0.06x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `trit-add` | `direct-llvm` | 0.00x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `swap-roundtrip` | `aot-env` | 0.15x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-match-repr` | `aot-env` | 0.08x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-match-repr` | `jit` | 0.05x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-match-repr` | `direct-llvm` | 0.00x | process-spawn-bound: the per-invocation time is dominated by spawning a fresh native process, not kernel compute (M-602/E1) — expected for a trivial kernel vs the in-process interpreter |
| `data-construct` | `aot-env` | 0.02x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `data-nested-match` | `aot-env` | 0.12x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `rec-self` | `aot-env` | 0.62x | slower than the in-process interpreter on this case (measured; no target — VR-5) |
| `rec-fold` | `aot-env` | 0.25x | slower than the in-process interpreter on this case (measured; no target — VR-5) |

## LLM-harness leverage (KC-2 / SC-5b)

Source: `/home/user/mycelium/tools/llm-harness/reports/20260617T182214Z-report.json` — **SYNTHETIC sample** (a fixture run — NOT real model quality; never treated as evidence, per the harness's own VR-5/V-03 rule).

> mycelium-llm-validation v0.1.0 — run 20260617T182214Z — mode=mock — SYNTHETIC (fixture; not real model quality) (4 validations: 1 pass / 3 mock-pass / 0 skip / 0 fail)

| validation | status | tag | latency (s) | prompt tok | gen tok | message |
|---|---|---|---|---|---|---|
| `V-01-determinism` | mock-PASS | Declared | — | — | — | [MOCK] Determinism check simulated with fixture — not a real model run. Fixture outputs matched: True |
| `V-02-json-projection` | mock-PASS | Declared | — | — | — | [MOCK] JSON-projection check against fixture — not a real model run. Fixture parsed and validated OK |
| `V-03-tag-honesty` | PASS | — | — | — | — | Tag-honesty gate PASSED. Correctly rejected 2 forbidden tag(s), correctly allowed 2 compliant tag(s).  |
| `V-04-latency-tokens` | mock-PASS | Declared | 0.0000 | 12 | 3 | [MOCK] Latency/token report — synthetic numbers. wall_seconds=0.0 is a sentinel meaning 'not measured (mock mode)'. Prompt tokens and generated tokens are fixture values. |
