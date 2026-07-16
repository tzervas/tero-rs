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

## Releases / history hygiene

Tags `v0.1.0`–`v0.1.3` and any GHCR images from the mycelium-workspace era should be **retired** after this standalone cut lands. A **history rewrite** (filter-repo / orphan re-root) is planned under branch protection — see `docs/HISTORY_SANITIZE.md`. Do not force-push `main` without maintainer approval and a coordinated release wipe.

## Companions

- `tero-mcp` (Python presenter / packaging, if separate)
- Consumers: cabal-devmelopner, agents

License: MIT.
