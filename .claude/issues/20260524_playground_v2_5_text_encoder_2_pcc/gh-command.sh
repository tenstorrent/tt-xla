#!/usr/bin/env bash
# Draft only — review draft.md before running.
# Do NOT run automatically; edit labels/assignee/milestone before creating.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue create \
  --repo tenstorrent/tt-xla \
  --title "$(cat "$DIR/title.txt")" \
  --label bug \
  --body-file "$DIR/draft.md"

# Optional — add manually if needed:
#   --assignee @me
#   --milestone "<milestone-name>"
#   then link parent tracker issue from the GitHub UI
#   and set "Type: Bug" via the GitHub issue UI (no stable CLI field id)
