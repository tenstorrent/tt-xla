#!/usr/bin/env bash
#
# Runs the api-check: a pytest collection-only pass over the model test
# files. Collection imports every test module, which in turn imports
# every model loader — surfacing transformers API breakage (renamed
# classes, removed exports, restructured modules) without instantiating
# anything or running any inference.
#
# Iteration 1:   sub-loops up to --max-subloops attempts. Each failed
#                attempt invokes Claude on the failure context, then
#                re-runs collection. Designed for the first pass over a
#                freshly-bumped transformers pin where many imports may
#                be broken at once.
#
# Iteration 2+: collapses to a fast sanity check — one run; if it fails,
#                one Claude pass + one re-run, then accept the result.
#                The outer orchestrator handles further iterations.
#
# Exit code:    0 iff the final collection ended green.
#
# Usage:
#   run-api-check.sh --iteration <N> [--max-subloops <K>]

set -euo pipefail

ITERATION=""
MAX_SUBLOOPS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iteration)     ITERATION="$2"; shift 2 ;;
    --max-subloops)  MAX_SUBLOOPS="$2"; shift 2 ;;
    *)               echo "::error::unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ITERATION" ]]; then
  echo "::error::--iteration is required" >&2
  exit 2
fi

# Iter 2+ collapses to a single fix attempt; the outer loop handles more.
if [[ "$ITERATION" -gt 1 ]]; then
  MAX_SUBLOOPS=2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
CLAUDE_FIX="${REPO_ROOT}/.github/scripts/uplift/transformers/run-claude-fix.sh"
FAILURE_CONTEXT="${RUNNER_TEMP:-/tmp}/api_check_failures.txt"
# Targets that pytest --collect-only walks. Add more files here if/when
# additional collection roots need API-level vetting.
COLLECT_TARGETS=("tests/runner/test_models.py")

# The repo's `venv/activate` script wires up the real virtualenv that
# carries pytest + transformers + the tt-xla wheel. Source it if we
# haven't been activated already, so this script works whether the
# caller activated the venv or not.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/venv/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/venv/activate"
fi

run_check() {
  rm -f "$FAILURE_CONTEXT"
  local rc=0
  # pytest exits 0 on clean collection, non-zero on any collection error.
  # Capture full stdout+stderr to the failure context, then mirror it to
  # the CI log. The `|| rc=$?` keeps set -e from aborting on pytest's
  # non-zero exit so we can return the code intentionally.
  (cd "$REPO_ROOT" && \
    pytest --collect-only -q --no-header "${COLLECT_TARGETS[@]}") \
    >"$FAILURE_CONTEXT" 2>&1 || rc=$?
  cat "$FAILURE_CONTEXT"
  return $rc
}

attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "=== api-check attempt ${attempt}/${MAX_SUBLOOPS} (iteration ${ITERATION}) ==="

  if run_check; then
    echo "::notice::api-check passed on attempt ${attempt}"
    exit 0
  fi

  # run_check already streamed the pytest output above; no need to re-print.

  if [[ "$attempt" -ge "$MAX_SUBLOOPS" ]]; then
    echo "::error::api-check did not converge within ${MAX_SUBLOOPS} attempts" >&2
    exit 1
  fi

  echo "Invoking Claude fix pass on api-check failures…"
  bash "$CLAUDE_FIX" --failures "$FAILURE_CONTEXT" --scope api-check
done
