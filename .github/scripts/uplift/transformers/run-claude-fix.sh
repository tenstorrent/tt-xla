#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

FAILURES=""
SCOPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --failures) FAILURES="$2"; shift 2 ;;
    --scope)    SCOPE="$2";    shift 2 ;;
    *)          echo "::error::unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$FAILURES" || -z "$SCOPE" ]]; then
  echo "::error::--failures and --scope are required" >&2
  exit 2
fi

if [[ ! -f "$FAILURES" ]]; then
  echo "::error::failure context file not found: $FAILURES" >&2
  exit 2
fi

case "$SCOPE" in
  api-check|model-test-uplifts|model-perf-uplift) ;;
  *) echo "::error::--scope must be one of: api-check, model-test-uplifts, model-perf-uplift (got: $SCOPE)" >&2
     exit 2 ;;
esac

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "::error::claude CLI not found on PATH (CLAUDE_BIN=$CLAUDE_BIN)" >&2
  exit 2
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "::error::ANTHROPIC_API_KEY is required" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
TIMEOUT="${CLAUDE_FIX_TIMEOUT:-30m}"

# Clear any stale summary from a previous iteration so Claude either
# writes a fresh one or the orchestrator's commit step falls back to
# its generic message. Without this, an iteration where Claude doesn't
# update the summary would reuse the PREVIOUS iteration's message.
rm -f "${REPO_ROOT}/.github/transformers-uplift/fix-summary.md"

BASELINE_FILE="baseline_failures.txt"
PROMPT=$(cat <<PROMPT_EOF
Use the \`transformers-uplift-fix\` skill in
\`.claude/skills/transformers-uplift-fix/SKILL.md\` to fix the failures
below. Apply the skill's instructions, rules and scope-specific guidance.

Scope:            ${SCOPE}
Current version:  ${CURRENT_VERSION:-unknown}
Target version:   ${TARGET_VERSION:-unknown}
Failure context:  ${FAILURES}
Baseline failures: ${BASELINE_FILE}   (failures observed on the last main nightly — may be empty if the baseline download was skipped or matched nothing)
Read the failure file first.
PROMPT_EOF
)

echo "=== Invoking Claude fix pass (scope=${SCOPE}, timeout=${TIMEOUT}) ==="
echo "Failures file: ${FAILURES} ($(wc -l <"$FAILURES") lines)"

# Run inside the repo root so Claude's file-tool resolution is correct.
cd "$REPO_ROOT"

rc=0

env -u GH_TOKEN -u GITHUB_TOKEN \
  timeout "$TIMEOUT" \
  "$CLAUDE_BIN" -p "$PROMPT" \
  --verbose \
  --allowed-tools "Read,Edit,Write,Bash,Grep,Glob" \
  || rc=$?

if [[ $rc -eq 124 ]]; then
  echo "::warning::Claude fix pass hit ${TIMEOUT} timeout (rc=124)"
elif [[ $rc -ne 0 ]]; then
  echo "::warning::Claude fix pass exited with rc=${rc}"
fi

exit $rc
