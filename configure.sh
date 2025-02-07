#/bin/sh

build_type="$1"
shift

if [ -z "$build_type" ]; then
    build_type="Release"
fi

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_C_COMPILER="clang-17" -DCMAKE_CXX_COMPILER="clang++-17" -DCMAKE_CXX_COMPILER_LAUNCHER="ccache" "$@"
