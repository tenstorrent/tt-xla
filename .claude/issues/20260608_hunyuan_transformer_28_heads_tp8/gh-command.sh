#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
#
# BEFORE RUNNING: the (1,8) mesh is an invalid TP config (28 heads not divisible by 8),
# so the compile failure is expected. The real ask is diagnostic quality / front-end
# validation — consider swapping --label bug for --label enhancement.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue create \
  --repo tenstorrent/tt-xla \
  --title "$(cat "$DIR/title.txt")" \
  --label bug \
  --body-file "$DIR/draft.md"

# Optional — add manually:
#   --assignee @me
#   --milestone "<name>"
#   set Type: Bug in the tt-xla GitHub UI after creation
#   link parent tracker / HunyuanImage bringup escalation in GitHub UI
