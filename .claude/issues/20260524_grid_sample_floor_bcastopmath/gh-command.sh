#!/usr/bin/env bash
# Ready-to-run command. DO NOT execute automatically — issue-create skill never
# files issues on behalf of the developer.
#
# Optional flags to add manually if/when filing:
#   --assignee @me
#   --milestone "<name>"
#   # link parent tracker by editing the body or via UI after creation

gh issue create \
  --repo tenstorrent/tt-mlir \
  --title "$(cat .claude/issues/20260524_grid_sample_floor_bcastopmath/title.txt)" \
  --label bug \
  --body-file .claude/issues/20260524_grid_sample_floor_bcastopmath/draft.md
