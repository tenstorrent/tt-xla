#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
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
#   --label model-bringup --label tensor-parallel   # if these labels exist in the repo
#   set Type: Bug in the tt-xla GitHub UI after creation (gh CLI cannot set it)
#   link parent tracker #4462 in GitHub UI after creation
