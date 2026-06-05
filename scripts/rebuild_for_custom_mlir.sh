#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Rebuild tt-xla against a custom tt-mlir SHA. Run from tt-xla root.
#
# Usage:
#   scripts/rebuild_for_custom_mlir.sh <ttmlir-sha> [<ttmlir-local-path>]
#
# Args:
#   ttmlir-sha         Commit to pin TT_MLIR_VERSION to.
#   ttmlir-local-path  Optional. Required when <ttmlir-sha> only exists in a
#                      local tt-mlir checkout (e.g. an unpushed rebased commit).
#                      Points GIT_REPOSITORY at this path so CMake's
#                      ExternalProject_Add can find the SHA.
#
# Note: a local-only SHA without a local path will make CMake fail at
# `git checkout <sha>` (github doesn't have it). It may succeed by accident
# if the vendored clone already has the SHA from a previous run — don't
# rely on it.

set -euo pipefail

SHA="${1:?tt-mlir SHA required}"
LOCAL_PATH="${2:-}"
[[ -n "$LOCAL_PATH" ]] && LOCAL_PATH="$(cd "$LOCAL_PATH" && pwd)"

CMAKE=third_party/CMakeLists.txt
git checkout -- "$CMAKE"
sed -i "s/set(TT_MLIR_VERSION \"[^\"]*\")/set(TT_MLIR_VERSION \"$SHA\")/" "$CMAKE"

if [[ -n "$LOCAL_PATH" ]]; then
    sed -i "s|GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git|GIT_REPOSITORY $LOCAL_PATH|" "$CMAKE"
    # CMake's ExternalProject update step won't reliably refresh origin when
    # GIT_REPOSITORY changes, so do it manually: update the clone's remote,
    # fetch the new SHA, check it out. Doing this in place (vs. `rm -rf $SRC`)
    # also avoids a full re-clone of tt-mlir and its nested tt-metal source +
    # build cache, which is multi-GB.
    SRC=third_party/tt-mlir/src/tt-mlir
    if [[ -d "$SRC/.git" ]]; then
        git -C "$SRC" remote set-url origin "$LOCAL_PATH"
        git -C "$SRC" fetch origin "$SHA" 2>/dev/null || true
        git -C "$SRC" -c advice.detachedHead=false checkout -f "$SHA" 2>/dev/null || rm -rf "$SRC"
    fi
fi

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
