# `mycelium-transpile` — a Rust → Mycelium transpiler PoC

> **Status:** a spike (M-873, kickoff `trx`; vet loop M-1000, kickoff `trx2`). Not the gated full
> phase of DN-34 §5 — DN-34 stays **Draft**, this enacts nothing. Everything it emits is **`Declared`**
> (a heuristic `syn` → surface-text mapping, unvalidated by a Mycelium parser/typechecker); the *vet*
> verdict it now measures is **`Empirical`** (see below). Grounding: `docs/notes/DN-34-Rust-to-Mycelium-Transpiler-Strategy.md`.

Reads **one** Rust crate's source (`syn::parse_file`) and produces two artifacts per file:

1. a best-effort `.myc` rendering of every top-level item it can express, and
2. a structured, **never-silent** gap report (`.gap.json`) naming every construct it could not (or,
   absent a confirmed grammar mapping, would not) express — never a silent drop (G2).

Every emission path traces to a specific production in `docs/spec/grammar/mycelium.ebnf`; any
fallback/uncovered arm returns `Err(GapReason)` rather than emit a placeholder (the DN-34 §4/§8
flag-don't-guess principle — never a fabricated body).

## CLI

```text
mycelium-transpile [--vet] <input.rs | input-dir> <out-dir>
```

- `<input.rs>` — one file → writes `<out-dir>/<stem>.myc` + `<out-dir>/<stem>.gap.json`.
- `<input-dir>` (a crate's `src/`) — recurses every `*.rs` (skipping test infrastructure),
  transpiles each independently, writes the per-file pair for every file **plus** `summary.json`
  (per-file + aggregate counts) and `union.gap.json` (every gap, plus aggregate category counts).

## The two coverage metrics — `expressible_fraction` vs `checked_fraction`

- **`expressible_fraction`** (emission-only, **`Declared`**): the fraction of non-test top-level items
  for which *some* `.myc` text was emitted. It never runs the toolchain, so it systematically
  **over-counts** — an emitted fragment that does not parse or type-check still counts.
- **`checked_fraction`** (**`Empirical`**, M-1000): the fraction that is **myc-check-clean** — measured
  by running the **real** `myc check` oracle (`crates/mycelium-check`) over each emitted `.myc`. This
  is the number that matters for the self-hosting port: it says how much of a draft the toolchain
  actually accepts.

Both use the **same denominator** — non-test top-level items — so they are directly comparable and
`checked_fraction ≤ expressible_fraction` always holds. `myc check` is a *per-file* oracle, so the
checked numerator is **file-gated**: a file's emitted items are credited only when the file's *entire*
emitted `.myc` is clean; a file that fails contributes 0 (we never guess which item broke it —
VR-5/G2). `checked_fraction` is therefore an honestly-conservative lower bound. See
`src/vet.rs`'s module docs for the full statement.

## The vet loop (`--vet`)

`--vet` runs the vet loop after transpiling: it invokes `myc check` per emitted `.myc`, folds each
outcome (exit class + first diagnostic) into `<out-dir>/vet.json`, and prints `checked_fraction`
alongside `expressible_fraction`. A draft is then `myc-check-clean` or `gap/vet-flagged` — never
silently broken. An oracle that cannot be *run at all* (binary absent) is recorded as
`ToolUnavailable` — **never** counted as clean.

The oracle is the pre-built binary named by the `MYC_CHECK_CMD` env var when set (the sanctioned,
build-lock-safe form — no nested `cargo`), else a `cargo run -p mycelium-check` fallback.

`scripts/checks/transpile-vet.sh` wraps this the safe way: it builds `myc-check` **once**, exports
`MYC_CHECK_CMD=<built binary>`, and runs `--vet` over a representative target set (a semcore probe +
unported stdlib crates + the `std-cmp` pilot). It is **advisory** (measures the transpiler; prints,
never gates), mirroring `scripts/checks/myc-dogfood.sh`. Run it with no args for the default set, or
pass explicit `<crate-src | .rs>` targets.

## Layout

`src/transpile.rs` (driver) · `src/emit.rs` (`.myc` emitter) · `src/map.rs` (Rust type → `type_ref`
table) · `src/gap.rs` (the never-silent gap taxonomy) · `src/reserved.rs` (the M-1001 reserved-word
collision guard) · `src/batch.rs` (directory mode) · `src/vet.rs` (the M-1000 vet loop) ·
`src/bin/mycelium-transpile.rs` (CLI). Tests live in `src/tests/` per the house test-layout rule (no
inline tests in logic files).
