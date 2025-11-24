#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_perf.sh [OPTIONS]
#
# Automated git bisect script for performance testing in tt-xla
#
# OPTIONS:
#   -c, --command COMMAND       Benchmark command to run (required if no defaults)
#   -t, --threshold THRESHOLD   Performance threshold value (required if no defaults)
#   -p, --pattern PATTERN       Grep pattern to extract metric (default: "Sample per second:\s*\K[0-9.]+")
#   -h, --help                  Show this help message
#
# EXAMPLES:
#   # Use with default resnet benchmark (threshold 680)
#   git bisect start
#   git bisect bad HEAD
#   git bisect good 051ebb20
#   git bisect run ./scripts/bisect_perf.sh
#
#   # Custom benchmark and threshold (resnet with different threshold)
#   git bisect run ./scripts/bisect_perf.sh \
#     -c "python ../tt-forge/benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -df bfloat16 -lp 128" \
#     -t 680
#
#   # Different model with custom metric extraction
#   git bisect run ./scripts/bisect_perf.sh \
#     -c "python ../tt-forge/benchmark/benchmark.py -p tt-xla -m vgg16 -bs 8 -df bfloat16 -lp 128" \
#     -t 500 \
#     -p "Sample per second:\s*\K[0-9.]+"
#
# EXIT CODES:
#   0   - Good commit (performance >= threshold)
#   1   - Bad commit (performance < threshold)
#   125 - Untestable commit (build failed, benchmark failed, etc.)

# Default values for resnet benchmark
DEFAULT_COMMAND="python ../tt-forge/benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -df bfloat16 -lp 128"
DEFAULT_THRESHOLD=680
DEFAULT_PATTERN="Sample per second:\s*\K[0-9.]+"

# Parse command line arguments
BENCHMARK_COMMAND=""
PERF_THRESHOLD=""
METRIC_PATTERN=""

show_help() {
    sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--command)
            BENCHMARK_COMMAND="$2"
            shift 2
            ;;
        -t|--threshold)
            PERF_THRESHOLD="$2"
            shift 2
            ;;
        -p|--pattern)
            METRIC_PATTERN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Use defaults if not specified
BENCHMARK_COMMAND="${BENCHMARK_COMMAND:-$DEFAULT_COMMAND}"
PERF_THRESHOLD="${PERF_THRESHOLD:-$DEFAULT_THRESHOLD}"
METRIC_PATTERN="${METRIC_PATTERN:-$DEFAULT_PATTERN}"

echo "======================================"
echo "Testing commit: $(git rev-parse --short HEAD)"
echo "Benchmark: $BENCHMARK_COMMAND"
echo "Threshold: $PERF_THRESHOLD"
echo "======================================"

# Activate the TT-XLA environment
echo "Activating TT-XLA environment..."
# Find tt-xla root (we're already in it during git bisect)
TTXLA_ROOT=$(git rev-parse --show-toplevel)
cd "$TTXLA_ROOT"
source venv/activate

# Apply the CMakeLists.txt fix for incremental builds
echo "Applying CMakeLists.txt fix..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/cmake_fix.patch" ]; then
    git apply "$SCRIPT_DIR/cmake_fix.patch" 2>/dev/null || echo "Patch already applied or not needed"
fi

# Update submodules to match this commit's version
echo "Updating submodules..."
git submodule update --init --recursive

# Build with CMake
echo "Building project..."
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
if ! cmake --build build; then
    echo "Build failed, marking as untestable"
    git checkout third_party/CMakeLists.txt 2>/dev/null || true
    exit 125  # Tell git bisect to skip this commit
fi

# Run the benchmark
echo "Running benchmark..."
BENCHMARK_OUTPUT=$(eval "$BENCHMARK_COMMAND" 2>&1)
BENCHMARK_EXIT_CODE=$?
echo "$BENCHMARK_OUTPUT"

if [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    echo "Benchmark failed, marking as untestable"
    git checkout third_party/CMakeLists.txt 2>/dev/null || true
    exit 125
fi

# Extract performance metric from the output
PERF_VALUE=$(echo "$BENCHMARK_OUTPUT" | grep -oP "$METRIC_PATTERN")

if [ -z "$PERF_VALUE" ]; then
    echo "Could not extract performance metric, marking as untestable"
    git checkout third_party/CMakeLists.txt 2>/dev/null || true
    exit 125  # Tell git bisect to skip this commit
fi

echo ""
echo "Performance: $PERF_VALUE (threshold: $PERF_THRESHOLD)"

# Restore the file before exiting
git checkout third_party/CMakeLists.txt 2>/dev/null || true

# Compare performance (using awk since bc may not be available)
if awk "BEGIN {exit !($PERF_VALUE >= $PERF_THRESHOLD)}"; then
    echo "✓ GOOD: Performance is above threshold"
    exit 0
else
    echo "✗ BAD: Performance is below threshold"
    exit 1
fi
