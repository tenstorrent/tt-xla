#/bin/sh

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

build_type="$1"
ttmlir_build_type="$2"
shift 2

if [ -z "$build_type" ]; then
    build_type="Release"
fi
if [ -z "$ttmlir_build_type" ]; then
    ttmlir_build_type="Release"
fi

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DTTMLIR_BUILD_TYPE=$ttmlir_build_type -DCMAKE_C_COMPILER="clang-17" -DCMAKE_CXX_COMPILER="clang++-17" -DCMAKE_CXX_COMPILER_LAUNCHER="ccache" "$@"
