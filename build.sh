#/bin/sh

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "tt-xla build script - configure and build the project"
    echo ""
    echo "Usage: build.sh [build_type] [ttmlir_build_type] [additional arguments]"
    echo "  build_type: The build type for tt-xla (default: Release)"
    echo "  ttmlir_build_type: The build type for tt-mlir (default: Release)"
    echo "  Additional arguments are forwarded to CMake"
    echo ""
    echo "Running without arguments but with the project already configured will just build it."
    exit 0
fi

# Project is already configured and no build type is set, so just build it.
if [ "$#" -eq 0 ] && [ -d "build" ]; then
    cmake --build build
    exit $?
fi

build_type="${1:-Release}"
ttmlir_build_type="${2:-Release}"
shift 2

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DTTMLIR_BUILD_TYPE=$ttmlir_build_type -DCMAKE_C_COMPILER="clang-17" -DCMAKE_CXX_COMPILER="clang++-17" -DCMAKE_CXX_COMPILER_LAUNCHER="ccache" "$@"
cmake --build build
