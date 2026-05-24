#!/usr/bin/env bash
# Copy-paste this command to create the issue.
# Do NOT run automatically — developer reviews body, then runs.

gh issue create \
  --repo tenstorrent/tt-xla \
  --title "$(cat .claude/issues/20260524_oft_grid_sample_oom/title.txt)" \
  --label bug \
  --body-file .claude/issues/20260524_oft_grid_sample_oom/draft.md

# Optional flags to add manually (not auto-suggested):
#   --assignee @me
#   --milestone "<milestone-name>"
# After creation, in the GitHub UI:
#   - Set Type to "Bug" (tt-xla uses a Type field; gh CLI cannot set it)
#   - Link to parent tracker / bringup issue if one exists
#   - Cross-link to #3419
