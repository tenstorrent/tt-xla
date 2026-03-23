#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Detect TT-MLIR dependency uplift.
# Args: $1 = file containing changed file paths (one per line)
#       $2 = file with PR diff
# Exit code 0 if uplift detected, 1 otherwise.

set -euo pipefail

changed_files="$1"
diff_file="$2"

# Check if third_party/CMakeLists.txt is in the changed files
if ! grep -q "^third_party/CMakeLists.txt$" "$changed_files"; then
    exit 1
fi

cmake_diff=$(sed -n '/^diff --git a\/third_party\/CMakeLists\.txt/,/^diff --git /p' "$diff_file")
if echo "$cmake_diff" | grep -qE '^[+-][[:space:]]*set\(TT_MLIR_VERSION'; then
    echo "Detected TT_MLIR_VERSION change in third_party/CMakeLists.txt"
    exit 0
fi

exit 1
