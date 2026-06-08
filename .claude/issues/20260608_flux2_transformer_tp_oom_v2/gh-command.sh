#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue create \
  --repo tenstorrent/tt-xla \
  --title "$(cat "$DIR/title.txt")" \
  --label bug \
  --label multichip \
  --label bringup \
  --body-file "$DIR/draft.md"

# Optional — add manually after creation:
#   --assignee @me
#   --milestone "<name>"
#   Set "Type: Bug" in the GitHub UI (gh CLI cannot set the Type field).
#   Link parent tracker #4705 (Model Bringup: FLUX.2) in the GitHub UI.
