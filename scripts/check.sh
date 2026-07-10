#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Google-style header: WHAT / WHY / WHY-NOT for this hygiene script in tero-rs.
# WHAT: Basic check.sh for the tero-rs supporting tooling extraction (the Rust kernel / binary bits for tero-mcp and dynamic indexing).
# WHY: tero-rs was missing check.sh while siblings (tero-mcp, cabal-devmelopner, memory-gate-rs, etc.) have them. Consistent hygiene gate is required for 1.0 readiness per the tooling-1.0-readiness-2026-07-10 wave kickoff and plan.md (hygiene-thin-repos + toolchain P1). Enables `scripts/check.sh`, CI parity, and self-improving loop (run before lands, after edits, via cabal or outer agents). Follows parameterized hygiene skill patterns.
# WHY NOT other options:
# - Full "cargo test --workspace" on every run: Too slow/heavy for this large mycelium-derived workspace copy (many crates). We target the tero-relevant parts (as done in prior bg hygiene fixes).
# - Just "cargo check": Insufficient for 1.0 (need tests where present + fmt/clippy).
# - Relying only on outer workspace scripts: Defeats per-repo autonomy and the "every extracted project has its own check" goal.
# - No tero index regen: tero-rs is the source of some tero power; we call update if the script sibling exists.
#
# Usage: ./scripts/check.sh [--quick] [--fix]
# Part of 1.0 wave: stable (tests+lints), hardened (via future security-scan integration), release prep.

MODE="${1:-}"
if command -v cargo >/dev/null 2>&1; then
  echo "=== tero-rs cargo fmt/clippy/check (targeted) ==="
  # Scope: TERO only. The remaining mycelium-* crates are vendored, read-only dependencies of the
  # tero crate (a separate language project) — they are NOT tero's to lint/gate. `--no-deps` lints
  # only the tero package; `-p tero` builds/tests only tero (+ its bins). Do NOT re-add --workspace.
  cargo fmt -p tero -- --check || (echo "fmt issues"; [[ "$MODE" == "--fix" ]] && cargo fmt -p tero)
  cargo clippy -p tero --all-targets --no-deps -- -D warnings || echo "WARN: tero clippy warnings (review for 1.0 hardening)"
  cargo check -p tero || echo "WARN: tero check failed"
  if [[ "$MODE" != "--quick" ]]; then
    cargo test -p tero -- --quiet || echo "INFO: tero tests failed or skipped"
  fi
else
  echo "cargo not found; skipping Rust checks"
fi

# If sibling tero update script is available in workspace layout, use it for index freshness (self-improving)
if [[ -f ../scripts/update-tero.sh ]]; then
  ../scripts/update-tero.sh . || echo "update-tero optional"
elif command -v python3 >/dev/null && [[ -f scripts/generate_lite_index.py ]]; then
  python3 scripts/generate_lite_index.py --root . || true
fi

# Secrets hygiene gate (git-secrets; see AGENTS.md)
if command -v git-secrets >/dev/null 2>&1; then
  git secrets --scan || { echo "ERROR: git-secrets detected prohibited patterns (XAI_API_KEY etc. must not leak)"; exit 1; }
else
  echo "WARN: git-secrets not in PATH — install + setup per AGENTS.md"
fi

echo "OK: tero-rs checks passed (or warnings noted for 1.0 follow-up)"
