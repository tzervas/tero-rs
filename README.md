# tero-rs

Standalone **Tero** engine: Layer-1 deterministic corpus index + optional Layer-2 VSA retrieval, served over MCP (`tero-mcp`) and HTTP (`tero-http`).

Named for Atsushi Tero’s slime-mold network experiments — route queries to the citations that answer them.

## Layout

```text
tero-rs/
  Cargo.toml          # single package `tero` (v0.2.0)
  src/                # library + bins under src/bin/
  src/md/             # markdown corpus ingest (in-tree; not mycelium)
  src/vsa2/           # Layer-2 VSA algebra (in-tree MAP-I + cleanup)
  scripts/check.sh
  docs/
```

**No mycelium language crates.** Former vendored `mycelium-*` workspace members were extraction residue and have been removed. Required markdown parsing and MAP-I algebra live inside this package.

## Build & check

```bash
./scripts/check.sh          # fmt, clippy -D warnings, release build, test
./scripts/check.sh --fix   # apply cargo fmt first
```

Bins: `tero-index`, `tero-http`, `tero-mcp`, `tero-eval`.

### Optional memory tools (`memory` feature)

Build with `cargo build --features memory` to link [memory-gate-rs](https://github.com/tzervas/memory-gate-rs) and expose MCP tools `memory_store`, `memory_retrieve`, and `memory_consolidate`. Layer-1 citations and dense memory are **separate scopes** — MG hits use the `memory_hits` envelope, not L1 `citations`.

| Env | Purpose |
|-----|---------|
| `TERO_MEMORY_ENABLED` | `1` or `true` to open the store at startup (default off) |
| `TERO_MEMORY_DB` | SQLite path for `SqliteVecStore` (required when enabled) |
| `TERO_MEMORY_MODEL` | Embedding catalog id (default `bge-small-en-v1.5`) |

Token scopes: `memory-read` (retrieve), `memory-write` (store + consolidate). Orthogonal to `read` / `refresh`.

Contract: workspace bulletin `join/tero-memory-feature` / `join-surfaces-tero-mg-mcp.md`.

## Releases / history hygiene

Tags `v0.1.0`–`v0.1.3` and any GHCR images from the mycelium-workspace era should be **retired** after this standalone cut lands. A **history rewrite** (filter-repo / orphan re-root) is planned under branch protection — see `docs/HISTORY_SANITIZE.md`. Do not force-push `main` without maintainer approval and a coordinated release wipe.

## Companions

- `tero-mcp` (Python presenter / packaging, if separate)
- Consumers: cabal-devmelopner, agents

License: MIT.
