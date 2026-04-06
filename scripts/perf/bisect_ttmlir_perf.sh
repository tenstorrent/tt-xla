#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_ttmlir_perf.sh [OPTIONS]
#
# Automated git bisect script for tt-mlir performance testing
# This script is called by git bisect run from within the tt-mlir submodule
# It builds tt-xla with the current tt-mlir commit and tests performance
#
# OPTIONS:
#   -c, --command COMMAND       Benchmark command to run (required if no defaults)
#   -t, --threshold THRESHOLD   Performance threshold value (required if no defaults)
#   -p, --pattern PATTERN       Grep pattern to extract metric (default: "Sample per second:\s*\K[0-9.]+")
#   -r, --revert COMMIT         Revert this commit if present in history (useful for isolating additional regressions)
#   -h, --help                  Show this help message
#
# EXAMPLES:
#   # Use with default resnet benchmark (threshold 680)
#   cd third_party/tt-mlir/src/tt-mlir
#   git bisect start
#   git bisect bad 0e3246fb8
#   git bisect good ca7ac8c3c
#   git bisect run ../../../../scripts/bisect_ttmlir_perf.sh
#
#   # Custom benchmark and threshold
#   git bisect run ../../../../scripts/bisect_ttmlir_perf.sh \
#     -c "python ../tt-forge/benchmark/benchmark.py -p tt-xla -m vgg16 -bs 8 -df bfloat16 -lp 128" \
#     -t 500
#
#   # Custom metric extraction pattern (using absolute path or set in PATH)
#   git bisect run $TTXLA_ROOT/scripts/bisect_ttmlir_perf.sh \
#     -c "python bench.py" \
#     -t 100 \
#     -p "Throughput:\s*\K[0-9.]+"
#
#   # Revert a known bad commit to isolate additional regressions
#   git bisect run ../../../../scripts/bisect_ttmlir_perf.sh \
#     -r 70efb12f5f75c7b1642542e7c3ce330c472ea038
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
REVERT_COMMIT=""

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
        -r|--revert)
            REVERT_COMMIT="$2"
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

# Determine tt-xla root (script is in scripts/ subdirectory of tt-xla)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTXLA_ROOT="$(dirname "$SCRIPT_DIR")"

TTMLIR_COMMIT=$(git rev-parse --short HEAD)
TTMLIR_COMMIT_FULL=$(git rev-parse HEAD)

# Create .bisect-run directory in tt-xla root for all bisect artifacts
BISECT_RUN_DIR="$TTXLA_ROOT/.bisect-run"
mkdir -p "$BISECT_RUN_DIR"

LOG_FILE="$BISECT_RUN_DIR/bisect_ttmlir_${TTMLIR_COMMIT}.log"

echo "======================================"
echo "Testing tt-mlir commit: $TTMLIR_COMMIT"
echo "Benchmark: $BENCHMARK_COMMAND"
echo "Threshold: $PERF_THRESHOLD"
echo "Log file: $LOG_FILE"
echo "======================================"

# Create log file
touch "$LOG_FILE"
echo "Testing tt-mlir commit: $TTMLIR_COMMIT" > "$LOG_FILE"
echo "Full commit: $TTMLIR_COMMIT_FULL" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

# Check if we need to revert a specific commit
if [ -n "$REVERT_COMMIT" ]; then
    echo "Checking if revert commit $REVERT_COMMIT is in history..." | tee -a "$LOG_FILE"
    if git merge-base --is-ancestor "$REVERT_COMMIT" HEAD 2>/dev/null; then
        echo "Reverting commit $REVERT_COMMIT..." | tee -a "$LOG_FILE"
        if git revert --no-commit "$REVERT_COMMIT" >> "$LOG_FILE" 2>&1; then
            echo "Successfully reverted $REVERT_COMMIT" | tee -a "$LOG_FILE"
        else
            echo "Failed to revert $REVERT_COMMIT, marking as untestable" | tee -a "$LOG_FILE"
            git revert --abort 2>/dev/null || true
            exit 125
        fi
    else
        echo "Commit $REVERT_COMMIT not in history, skipping revert" | tee -a "$LOG_FILE"
    fi
fi

# Go to tt-xla root
cd "$TTXLA_ROOT"

# Cleanup function (defined here for use throughout the script)
cleanup() {
    cd "$TTXLA_ROOT" 2>/dev/null || true
    git checkout third_party/CMakeLists.txt 2>/dev/null || true
    if [ -n "$REVERT_COMMIT" ]; then
        # Reset tt-mlir submodule to undo the revert
        cd "$TTXLA_ROOT/third_party/tt-mlir/src/tt-mlir" 2>/dev/null || true
        git reset --hard HEAD 2>/dev/null || true
        cd "$TTXLA_ROOT" 2>/dev/null || true
    fi
}

# Activate the TT-XLA environment
echo "Activating TT-XLA environment..." | tee -a "$LOG_FILE"
source venv/activate

# Modify CMakeLists.txt to use current tt-mlir commit
echo "Setting tt-mlir version to $TTMLIR_COMMIT_FULL..." | tee -a "$LOG_FILE"
sed -i "s/set(TT_MLIR_VERSION \"[^\"]*\")/set(TT_MLIR_VERSION \"$TTMLIR_COMMIT_FULL\")/" third_party/CMakeLists.txt

# Build with CMake (incremental build)
echo "Building project..." | tee -a "$LOG_FILE"
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release >> "$LOG_FILE" 2>&1
if ! cmake --build build >> "$LOG_FILE" 2>&1; then
    echo "Build failed, marking as untestable" | tee -a "$LOG_FILE"
    cleanup
    exit 125  # Tell git bisect to skip this commit
fi
echo "Build completed successfully" | tee -a "$LOG_FILE"

# Run the benchmark
echo "Running benchmark..." | tee -a "$LOG_FILE"
BENCHMARK_OUTPUT=$(eval "$BENCHMARK_COMMAND" 2>&1 | tee -a "$LOG_FILE")
BENCHMARK_EXIT_CODE=${PIPESTATUS[0]}

if [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    echo "Benchmark failed, marking as untestable" | tee -a "$LOG_FILE"
    cleanup
    exit 125
fi

# Extract performance metric from the output
PERF_VALUE=$(echo "$BENCHMARK_OUTPUT" | grep -oP "$METRIC_PATTERN")

if [ -z "$PERF_VALUE" ]; then
    echo "Could not extract performance metric, marking as untestable" | tee -a "$LOG_FILE"
    cleanup
    exit 125  # Tell git bisect to skip this commit
fi

echo "" | tee -a "$LOG_FILE"
echo "Performance: $PERF_VALUE (threshold: $PERF_THRESHOLD)" | tee -a "$LOG_FILE"

# Compare performance (using awk since bc may not be available)
if awk "BEGIN {exit !($PERF_VALUE >= $PERF_THRESHOLD)}"; then
    echo "✓ GOOD: Performance is above threshold" | tee -a "$LOG_FILE"
    cleanup
    exit 0
else
    echo "✗ BAD: Performance is below threshold" | tee -a "$LOG_FILE"
    cleanup
    exit 1
fi
