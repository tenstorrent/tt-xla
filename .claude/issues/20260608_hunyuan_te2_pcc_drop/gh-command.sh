#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
#
# HEADS UP: this is very likely a DUPLICATE of tenstorrent/tt-xla#4784
# (same component, PCC 0.9827, empty-body bare tracker). Prefer updating #4784
# in place with draft.md instead of filing a new issue:
#
#   gh issue edit 4784 --repo tenstorrent/tt-xla --body-file "$DIR/draft.md"
#
# Only run the `gh issue create` below if you have decided a fresh issue is warranted.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue create \
  --repo tenstorrent/tt-xla \
  --title "$(cat "$DIR/title.txt")" \
  --label bug \
  --body-file "$DIR/draft.md"

# Optional — add manually:
#   --assignee @me
#   --milestone "Models: Vision/Other"
#   link parent tracker #4773 in GitHub UI after creation
#   set Type: Bug in the tt-xla UI (gh CLI cannot set it)
