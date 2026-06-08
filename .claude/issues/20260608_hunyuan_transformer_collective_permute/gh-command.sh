#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
#
# BEFORE RUNNING: the compiler error already links tenstorrent/tt-mlir#3370 (OPEN).
# Strongly consider posting this repro as a COMMENT on #3370 instead of filing a duplicate:
#
#   gh issue comment 3370 --repo tenstorrent/tt-mlir --body-file "$(dirname "${BASH_SOURCE[0]}")/draft.md"
#
# Only file a fresh issue if you have confirmed the verifier error
# ("requires the same type for all operands and results") is a DISTINCT bug.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue create \
  --repo tenstorrent/tt-mlir \
  --title "$(cat "$DIR/title.txt")" \
  --label bug \
  --body-file "$DIR/draft.md"

# Optional — add manually:
#   --assignee @me
#   --milestone "<name>"
#   set Type: Bug in the tt-mlir GitHub UI after creation
#   link parent tracker / tt-xla bringup escalation in GitHub UI
