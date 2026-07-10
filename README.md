# tero-rs

**Status:** Extracted toolchain core (2026-07-10)

The Rust kernel and binary fronts for Tero (Layer-1 index + query engine, DN-87; M-1016/M-1017).

This is the supporting extraction of `mycelium-tero` + related crates (the transparent memory substrate) from the larger Mycelium workspace. It powers:

- `tero-mcp` (Rust `tero-mcp` bin + Python `tero-mcp-lite` presenter)
- tero-index / tero-http / tero-eval bins
- Dynamic tool surface for MCP consumers (cabal-devmelopner, agents)

See [crates/mycelium-tero](crates/mycelium-tero) for the core implementation and fronts.

**Companion projects (toolchain tranche):** tero-mcp, cabal-devmelopner.

## Quick start (hygiene + tero)
```bash
cd tero-rs
./scripts/check.sh --fix   # fmt, clippy, targeted test
# After docs: /root/git/scripts/update-tero.sh tero-rs
# Query: /root/git/scripts/tero.sh tero-rs text_search "..."
```

## Tero-first

Use tero MCP or `/root/git/scripts/tero.sh tero-rs ...` (after index) for all design/roadmap/historic context. See AGENTS.md.

## 1.0 Readiness

Part of `tooling-1.0-readiness-2026-07-10` wave (priority 1: Toolchain Core).

See docs/ROADMAP.md and CHANGELOG.md .

**Current versions:** workspace crates at 0.0.0 baseline; key fronts (mycelium-tero) targeted for 0.1.0+ path documented.

License: MIT (per workspace).

---

*This README bootstrapped for tero index + cabal assessment in 1.0 wave. Append-only updates.*
