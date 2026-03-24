#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Detect Transformers library uplift.
# Args: $1 = file containing changed file paths (one per line)
#       $2 = file with PR diff
# Exit code 0 if uplift detected, 1 otherwise.

set -euo pipefail

changed_files="$1"
diff_file="$2"

if ! grep -q "^venv/requirements-dev.txt$" "$changed_files"; then
    exit 1
fi

req_diff=$(sed -n '/^diff --git a\/venv\/requirements-dev\.txt/,/^diff --git /p' "$diff_file")
if echo "$req_diff" | grep -qE '^[+-]transformers[=<>!~]'; then
    echo "Detected transformers version change in venv/requirements-dev.txt"
    exit 0
fi

exit 1
