#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
#
# BEFORE RUNNING: this draft documents TWO root causes (catastrophic SPMD attention-sharding
# bug -> PCC 0.277, AND a residual device fp32 op-level numeric gap that caps replicated/MLP-only
# at ~0.95-0.96). Decide whether to file as one issue or split into two (SPMD/compiler vs
# op-numerics) depending on who owns each fix.

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
