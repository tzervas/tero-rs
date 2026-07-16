#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"

if [[ "$MODE" == "--fix" ]]; then
  cargo fmt
else
  cargo fmt --check
fi
cargo clippy --all-targets -- -D warnings
cargo build --release
cargo test --quiet

echo "OK: tero-rs checks passed"
