#!/usr/bin/env bash
# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.
#
# DUPLICATE of tenstorrent/tt-xla#4780. This is NOT a `gh issue create`.
# The draft is a SUPPLEMENTAL COMMENT for the existing issue. Review draft.md,
# then post it as a comment if you agree it adds value:
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue comment 4780 \
  --repo tenstorrent/tt-xla \
  --body-file "$DIR/draft.md"

# If — after review — you decide this warrants a SEPARATE tracking issue instead
# (e.g. a tt-xla "transformer needs fused-SDPA before TP fits" sub-task), use:
#
#   gh issue create --repo tenstorrent/tt-xla \
#     --title "$(cat "$DIR/title.txt")" --label bug --body-file "$DIR/draft.md"
#
# ...and edit title.txt/draft.md to drop the DUPLICATE framing first.
