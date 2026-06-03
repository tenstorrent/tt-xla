#!/usr/bin/env bash
#
# Invokes Claude on the supplied failure context to fix transformers
# compatibility issues in the working tree of tt-xla + tt-forge-models.
#
# Inputs come from the calling stage:
#   --failures <path>            file containing the captured pytest /
#                                hardware-test output for the current
#                                iteration. Fed verbatim to Claude.
#   --scope api-check|base-coverage
#                                tells Claude whether the failures are
#                                import-time (api-check) or runtime
#                                (base-coverage), so it can frame its
#                                approach accordingly.
#
# This script does NOT stage, commit, or push anything — the orchestrator
# job owns the git steps. Claude is expected to leave the working tree
# dirty with whatever edits it deemed appropriate.
#
# Exit code:  whatever Claude exits with. Non-zero is informational only
#             (the orchestrator re-runs the check after this script).
#
# Required env:
#   ANTHROPIC_API_KEY      Claude Code authentication
#   CLAUDE_FIX_TIMEOUT     optional, GNU `timeout` spec (default 30m)
#   CLAUDE_BIN             optional, claude binary path (default: `claude`)

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

PROMPT=$(cat <<PROMPT_EOF
Use the \`transformers-uplift-fix\` skill in
\`.claude/skills/transformers-uplift-fix/SKILL.md\` to fix the failures
below. Apply the skill's instructions, rules and scope-specific guidance.

Scope:           ${SCOPE}
Current version: ${CURRENT_VERSION:-unknown}
Target version:  ${TARGET_VERSION:-unknown}
Failure context: ${FAILURES}
Read that failure file first; it contains information about the failed tests.
PROMPT_EOF
)

echo "=== Invoking Claude fix pass (scope=${SCOPE}, timeout=${TIMEOUT}) ==="
echo "Failures file: ${FAILURES} ($(wc -l <"$FAILURES") lines)"

# Run inside the repo root so Claude's file-tool resolution is correct.
cd "$REPO_ROOT"

rc=0
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
