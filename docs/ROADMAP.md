# tero-rs — Product Roadmap

**Status:** Living (2026-07-10, tooling-1.0-readiness wave)  
**North star:** Stable, cited, hardened Rust kernel + fronts for Tero Layer-1 (DN-87) powering all downstream (tero-mcp, cabal, agents). Extracted supporting crate.

Companion: [../AGENTS.md](../AGENTS.md), tero-mcp/docs/ROADMAP.md

## 1.0 Target Criteria (from tooling-1.0-readiness-2026-07-10 wave)
- Stable: check.sh green, cargo test/clippy clean on tero fronts, no critical unwraps in prod paths.
- Performant: relevant benches (mycelium-bench) or query profiles documented.
- Hardened: security-scan, deny.toml if applicable, C0 refusals.
- Docs & Release: ROADMAP/CHANGELOG/AGENTS current + tero cites; semver bump or explicit 1.0 path doc.
- Verification: cabal --use-tero assessment (with local index) + re-run checks.

## Current State (Tero-cited + surveys)
- On feature/1.0-readiness (evolved chore/semver-baseline-v0.1.0).
- Hygiene: scripts/check.sh present + passing (fmt/clippy/test targeted to mycelium-tero).
- Tero index: bootstrapped (requires root md files for items >0).
- Semver: workspace + mycelium-tero at 0.0.0 (baseline chore); bumping key crates toward 0.1.0 with rationale (see CHANGELOG).
- Gaps vs wave (from cabal attempt + tero): empty index initially blocked cabal assess; missing root docs/AGENTS/README (now added); no top-level version in Cargo for "tero-rs" project view; limited dedicated tests in front (relies on integration in tero-mcp + consumers).

## Waves / Steps for Toolchain P1
- Hygiene + branch guard (done this tranche).
- Bootstrap docs + tero index (this tranche; enables cites).
- Semver bump + CHANGELOG entry (0.1.0 path justified as extracted PoC-to-stable; full 1.0 after consumer 1.0s + VSA gate open?).
- Add deny.toml / security if missing; expand check.sh coverage.
- Re-verify with cabal assessment + tero.sh queries post changes.
- Propagate: land to dev, main --no-ff; update-tero; ghcr or release artifacts if applicable (see plan.md).

## Relation to siblings
- tero-mcp: Python lite + packaging over the Rust tero-mcp bin. See its ROADMAP for MCP surface stability.
- cabal-devmelopner: primary consumer + self-driver for this wave.

**Next after this tranche:** full green hygiene + cabal "is 1.0 ready?" positive with cites; PR land.

---

*Initial for 1.0 wave. Append-only. Tero cites: workspacecabalteroreadiness, plan.md sections on P1, wave doc "Concrete First Steps (tero-rs)" .*
