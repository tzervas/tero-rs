# tero-rs

**Status:** Extracted toolchain core (2026-07-10)

The Rust kernel and binary fronts for Tero (Layer-1 index + query engine, DN-87; M-1016/M-1017).

This is the supporting extraction of the `tero` crate + related crates (the transparent memory substrate) from the larger Mycelium workspace. It powers:

- `tero-mcp` (Rust `tero-mcp` bin + Python `tero-mcp-lite` presenter)
- tero-index / tero-http / tero-eval bins
- Dynamic tool surface for MCP consumers (cabal-devmelopner, agents)

See [crates/tero](crates/tero) for the core implementation and fronts.

**Companion projects (toolchain tranche):** tero-mcp, cabal-devmelopner.

## About the name

**Tero** is named for **Atsushi Tero**, whose slime-mold (*Physarum polycephalum*) experiments
showed that a simple organism, with no central controller, grows near-optimal transport networks —
famously reconstructing a network resembling the Tokyo rail map by connecting food sources with
efficient, fault-tolerant routes. The name fits: Tero's job is to *route to the relevant
information* — its indexing and search optimize the path from a query to the citations that answer
it, a conceptually similar route-finding-over-a-network optimization.

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

**Current versions:** the `tero` crate (formerly `mycelium-tero`; the shipped fronts
`tero-index`/`tero-http`/`tero-mcp`/`tero-eval`) at **0.1.2**; the 16 vendored `mycelium-*`
dependency crates remain at 0.0.0. The 0.1.2 bump reflects the public crate rename `mycelium-tero`
→ `tero` plus the clean-extraction strip (40 non-dependency `mycelium-*` crates removed; workspace
57 → 17 crates); binary behavior is unchanged from 0.1.1. The `v0.1.2` git tag is on `origin`; the
GitHub Release and GHCR image for 0.1.2 are pending explicit maintainer authorization. 1.0.0 is not
yet justified — see docs/ROADMAP.md (hardened fronts, cabal positive assess, and the
vendored-vs-published dependency decision).

License: MIT (per workspace).

---

*This README bootstrapped for tero index + cabal assessment in 1.0 wave. Append-only updates.*
