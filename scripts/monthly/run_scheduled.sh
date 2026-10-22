#!/usr/bin/env bash
set -euo pipefail

MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${BASE_BRANCH:-under_review}"
REMOTE="${REMOTE:-origin}"
PREFIX="${BRANCH_PREFIX:-codex/monthly-data-update-}"
MONTH="${MONTH:-$(date +%Y-%m)}"

if [[ "${FORCE:-0}" != 1 ]]; then
  if [[ "$(date +%u)" != 1 ]] || (( 10#$(date +%d) > 7 )); then
    echo "Not the first Monday of the month; nothing to do."
    exit 0
  fi
fi

WORKTREE="$MAIN/output/monthly/$MONTH/repo"
BRANCH="$PREFIX$MONTH"

git -C "$MAIN" fetch "$REMOTE" "$BASE"
if [[ ! -d "$WORKTREE" ]]; then
  git -C "$MAIN" worktree add -B "$BRANCH" "$WORKTREE" "$REMOTE/$BASE"
fi
ln -sfn "$MAIN/.env" "$WORKTREE/.env"

cd "$WORKTREE"
ARGS=(run --month "$MONTH")
[[ "${NO_PR:-0}" == 1 ]] && ARGS+=(--no_pr)
exec pixi run --manifest-path "$MAIN/pixi.toml" python scripts/monthly/run_monthly_workflow.py "${ARGS[@]}"
