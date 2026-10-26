#!/usr/bin/env bash
# verify.sh — pre-merge checklist runner (docs/GIT-WORKFLOW.md "Before any merge to main").
# Usage: scripts/verify.sh [--quick]
#   --quick  skip clippy (fast inner loop)
set -uo pipefail
cd "$(dirname "$0")/.."

QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
FAIL=0
step() { printf '\n== %s ==\n' "$1"; }

step "cargo build --workspace"
cargo build --workspace || FAIL=1

step "cargo test --workspace"
cargo test --workspace || FAIL=1

if [ "$QUICK" = 0 ]; then
  if cargo clippy --version >/dev/null 2>&1; then
    step "cargo clippy --workspace"
    cargo clippy --workspace --all-targets -- -D warnings || FAIL=1
  else
    step "clippy (skipped — not installed)"
  fi
  if rustfmt --version >/dev/null 2>&1; then
    step "cargo fmt --check"
    cargo fmt --check || FAIL=1
  else
    step "rustfmt (skipped — not installed)"
  fi
fi

step "reference clones untouched"
if git status --porcelain "Reference Projects" 2>/dev/null | grep -q .; then
  echo "FAIL: Reference Projects/ has modifications"; FAIL=1
else
  echo "ok"
fi

step "no weights tracked by git"
if git ls-files | grep -qiE '\.(gguf|safetensors)$'; then
  echo "FAIL: weight files tracked"; FAIL=1
else
  echo "ok"
fi

echo
if [ "$FAIL" = 0 ]; then echo "verify: PASS"; else echo "verify: FAIL"; fi
exit $FAIL
