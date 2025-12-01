#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_perf_auto.sh [OPTIONS]
#
# Automated performance bisect tool that handles both tt-xla and tt-mlir bisecting
# This script will automatically bisect tt-xla commits, and if the regression is
# caused by a tt-mlir uplift, it will automatically bisect into tt-mlir to find
# the exact commit that caused the regression.
#
# OPTIONS:
#   -g, --good COMMIT           Good commit (required)
#   -b, --bad COMMIT            Bad commit (default: HEAD)
#   -c, --command COMMAND       Benchmark command to run
#   -t, --threshold THRESHOLD   Performance threshold value
#   -p, --pattern PATTERN       Grep pattern to extract metric
#   -r, --revert COMMIT         Revert this tt-mlir commit if present in history (useful for isolating additional regressions)
#   -l, --log-dir DIR           Log directory (default: .bisect-run/bisect_auto_TIMESTAMP)
#   -h, --help                  Show this help message
#
# EXAMPLES:
#   # Basic usage with default resnet benchmark
#   ./scripts/bisect_perf_auto.sh -g 051ebb20 -b HEAD
#
#   # With custom benchmark and threshold
#   ./scripts/bisect_perf_auto.sh -g 051ebb20 -b HEAD \
#     -c "python ../tt-forge/benchmark/benchmark.py -p tt-xla -m vgg16 -bs 8 -df bfloat16 -lp 128" \
#     -t 500
#
#   # Revert a known bad tt-mlir commit to isolate additional regressions
#   ./scripts/bisect_perf_auto.sh -g 051ebb20 -b HEAD -r 70efb12f5f75c7b1642542e7c3ce330c472ea038
#
# EXIT CODES:
#   0   - Successfully identified bad commit
#   1   - Error during bisect process

# Default values
GOOD_COMMIT=""
BAD_COMMIT="HEAD"
BENCHMARK_COMMAND="python ../tt-forge/benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -df bfloat16 -lp 128"
PERF_THRESHOLD=680
METRIC_PATTERN="Sample per second:\s*\K[0-9.]+"
REVERT_COMMIT=""
# Use .bisect-run directory in tt-xla root
BISECT_RUN_DIR="$(pwd)/.bisect-run"
mkdir -p "$BISECT_RUN_DIR"
LOG_DIR="$BISECT_RUN_DIR/bisect_auto_$(date +%Y%m%d_%H%M%S)"

show_help() {
    sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--good)
            GOOD_COMMIT="$2"
            shift 2
            ;;
        -b|--bad)
            BAD_COMMIT="$2"
            shift 2
            ;;
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
        -l|--log-dir)
            LOG_DIR="$2"
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

# Validate required arguments
if [ -z "$GOOD_COMMIT" ]; then
    echo "Error: Good commit is required (-g, --good)"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/bisect_auto.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

log_separator() {
    echo "========================================" | tee -a "$MAIN_LOG"
}

log_separator
log "Starting Automated Performance Bisect"
log_separator
log "Good commit: $GOOD_COMMIT"
log "Bad commit: $BAD_COMMIT"
log "Benchmark: $BENCHMARK_COMMAND"
log "Threshold: $PERF_THRESHOLD"
log "Log directory: $LOG_DIR"
log_separator

# Find tt-xla root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTXLA_ROOT="$(dirname "$SCRIPT_DIR")"

# Copy bisect scripts to .bisect-run directory to preserve them during git operations
log "Copying bisect scripts to $BISECT_RUN_DIR/scripts for safety..."
mkdir -p "$BISECT_RUN_DIR/scripts"
cp "$SCRIPT_DIR"/bisect*.sh "$BISECT_RUN_DIR/scripts/"
log "Scripts copied to: $BISECT_RUN_DIR/scripts/"

# Update script paths to use copied versions
BISECT_PERF_SCRIPT="$BISECT_RUN_DIR/scripts/bisect_perf.sh"
BISECT_TTMLIR_SCRIPT="$BISECT_RUN_DIR/scripts/bisect_ttmlir_perf.sh"
BISECT_TTMETAL_SCRIPT="$BISECT_RUN_DIR/scripts/bisect_ttmetal_perf.sh"
cd "$TTXLA_ROOT"

log "Working directory: $(pwd)"
log ""

# Phase 1: Bisect tt-xla
log_separator
log "PHASE 1: Bisecting tt-xla commits"
log_separator

log "Starting git bisect in tt-xla..."
git bisect start
git bisect bad "$BAD_COMMIT"
git bisect good "$GOOD_COMMIT"

log "Running bisect with performance tests..."
BISECT_SCRIPT="$BISECT_PERF_SCRIPT"
BISECT_ARGS="-c \"$BENCHMARK_COMMAND\" -t \"$PERF_THRESHOLD\" -p \"$METRIC_PATTERN\""
if [ -n "$REVERT_COMMIT" ]; then
    BISECT_ARGS="$BISECT_ARGS -r \"$REVERT_COMMIT\""
    log "Will revert tt-mlir commit $REVERT_COMMIT if present during bisect"
fi
eval "git bisect run \"$BISECT_SCRIPT\" $BISECT_ARGS" 2>&1 | tee -a "$MAIN_LOG"

# Capture the first bad commit
FIRST_BAD_COMMIT=$(git bisect view --pretty=format:'%H' | head -1)
FIRST_BAD_COMMIT_SHORT=$(git rev-parse --short "$FIRST_BAD_COMMIT")

log ""
log_separator
log "tt-xla bisect completed!"
log "First bad commit: $FIRST_BAD_COMMIT_SHORT"
log_separator

# Get commit message to check if it's a tt-mlir uplift
COMMIT_MSG=$(git log --format=%s -n 1 "$FIRST_BAD_COMMIT")
log "Commit message: $COMMIT_MSG"

# Reset bisect before checking files
git bisect reset

# Check if this is a tt-mlir uplift commit
IS_TTMLIR_UPLIFT=0
if echo "$COMMIT_MSG" | grep -iq "uplift.*tt-mlir\|tt-mlir.*uplift"; then
    log "✓ Detected tt-mlir uplift commit"
    IS_TTMLIR_UPLIFT=1
elif git diff "${FIRST_BAD_COMMIT}^" "$FIRST_BAD_COMMIT" -- third_party/CMakeLists.txt | grep -q "TT_MLIR_VERSION"; then
    log "✓ Detected TT_MLIR_VERSION change in third_party/CMakeLists.txt"
    IS_TTMLIR_UPLIFT=1
elif git diff "${FIRST_BAD_COMMIT}^" "$FIRST_BAD_COMMIT" -- third_party/tt-mlir | grep -q "Subproject commit"; then
    log "✓ Detected tt-mlir submodule update"
    IS_TTMLIR_UPLIFT=1
fi

if [ $IS_TTMLIR_UPLIFT -eq 0 ]; then
    log ""
    log "════════════════════════════════════════"
    log "FINAL RESULT"
    log "════════════════════════════════════════"
    log "First bad commit: $FIRST_BAD_COMMIT_SHORT"
    log "This is NOT a tt-mlir uplift commit."
    log "The regression is in tt-xla code."
    log ""
    log "View details: git show $FIRST_BAD_COMMIT_SHORT"
    log "════════════════════════════════════════"
    exit 0
fi

# Phase 2: Bisect tt-mlir
log ""
log_separator
log "PHASE 2: Bisecting tt-mlir commits"
log_separator

# Extract tt-mlir versions from good and bad commits
log "Extracting tt-mlir versions..."

git checkout "$GOOD_COMMIT" --quiet
GOOD_TTMLIR=$(grep 'set(TT_MLIR_VERSION' third_party/CMakeLists.txt | grep -oP '"\K[^"]+' | head -1 | tr -d '\n\r')
log "Good commit tt-mlir version: $GOOD_TTMLIR"

git checkout "$FIRST_BAD_COMMIT" --quiet
BAD_TTMLIR=$(grep 'set(TT_MLIR_VERSION' third_party/CMakeLists.txt | grep -oP '"\K[^"]+' | head -1 | tr -d '\n\r')
log "Bad commit tt-mlir version: $BAD_TTMLIR"

if [ "$GOOD_TTMLIR" = "$BAD_TTMLIR" ]; then
    log "ERROR: tt-mlir versions are the same in good and bad commits!"
    log "Cannot bisect tt-mlir."
    git checkout "$BAD_COMMIT" --quiet
    exit 1
fi

# Go to tt-mlir submodule
TTMLIR_DIR="$TTXLA_ROOT/third_party/tt-mlir/src/tt-mlir"
if [ ! -d "$TTMLIR_DIR" ]; then
    log "ERROR: tt-mlir directory not found: $TTMLIR_DIR"
    git checkout "$BAD_COMMIT" --quiet
    exit 1
fi

cd "$TTMLIR_DIR"
log "Changed to tt-mlir directory: $(pwd)"

# Ensure we have the commits
git fetch origin 2>&1 | tee -a "$MAIN_LOG"

# Start bisect in tt-mlir
log ""
log "Starting git bisect in tt-mlir..."
git bisect start
git bisect bad "$BAD_TTMLIR"
git bisect good "$GOOD_TTMLIR"

log "Running tt-mlir bisect with performance tests..."
TTMLIR_BISECT_SCRIPT="$BISECT_TTMLIR_SCRIPT"
TTMLIR_BISECT_ARGS="-c \"$BENCHMARK_COMMAND\" -t \"$PERF_THRESHOLD\" -p \"$METRIC_PATTERN\""
if [ -n "$REVERT_COMMIT" ]; then
    TTMLIR_BISECT_ARGS="$TTMLIR_BISECT_ARGS -r \"$REVERT_COMMIT\""
fi

# Run the bisect and capture output
eval "git bisect run \"$TTMLIR_BISECT_SCRIPT\" $TTMLIR_BISECT_ARGS" 2>&1 | tee -a "$MAIN_LOG"

# Capture the first bad tt-mlir commit
FIRST_BAD_TTMLIR=$(git bisect view --pretty=format:'%H' | head -1)
FIRST_BAD_TTMLIR_SHORT=$(git rev-parse --short "$FIRST_BAD_TTMLIR")

log ""
log_separator
log "tt-mlir bisect completed!"
log "First bad tt-mlir commit: $FIRST_BAD_TTMLIR_SHORT"
log_separator

# Get commit details
TTMLIR_COMMIT_MSG=$(git log --format=%s -n 1 "$FIRST_BAD_TTMLIR")
TTMLIR_COMMIT_AUTHOR=$(git log --format='%an' -n 1 "$FIRST_BAD_TTMLIR")
TTMLIR_COMMIT_DATE=$(git log --format='%ad' -n 1 "$FIRST_BAD_TTMLIR")

git bisect reset

# Check if the bad tt-mlir commit is a tt-metal uplift
log ""
log "Checking if tt-mlir commit is a tt-metal uplift..."

# Navigate to tt-mlir third_party directory to check TT_METAL_VERSION
TTMLIR_THIRD_PARTY_DIR="$TTMLIR_DIR/third_party"
cd "$TTMLIR_THIRD_PARTY_DIR"

# Extract TT_METAL_VERSION from bad commit and its parent
git checkout "$FIRST_BAD_TTMLIR" --quiet 2>/dev/null || true
BAD_TTMETAL_VERSION=$(grep 'set(TT_METAL_VERSION' CMakeLists.txt | grep -oP '"\K[^"]+' | head -1 | tr -d '\n\r')
log "Bad commit TT_METAL_VERSION: '$BAD_TTMETAL_VERSION'"

git checkout "${FIRST_BAD_TTMLIR}^" --quiet 2>/dev/null || true
PARENT_TTMETAL_VERSION=$(grep 'set(TT_METAL_VERSION' CMakeLists.txt | grep -oP '"\K[^"]+' | head -1 | tr -d '\n\r')
log "Parent commit TT_METAL_VERSION: '$PARENT_TTMETAL_VERSION'"

# Return to bad commit
git checkout "$FIRST_BAD_TTMLIR" --quiet 2>/dev/null || true

# Compare versions
IS_TTMETAL_UPLIFT=0
if [ -n "$BAD_TTMETAL_VERSION" ] && [ -n "$PARENT_TTMETAL_VERSION" ]; then
    BAD_LEN=${#BAD_TTMETAL_VERSION}
    PARENT_LEN=${#PARENT_TTMETAL_VERSION}
    MIN_LEN=$((BAD_LEN < PARENT_LEN ? BAD_LEN : PARENT_LEN))

    BAD_PREFIX="${BAD_TTMETAL_VERSION:0:$MIN_LEN}"
    PARENT_PREFIX="${PARENT_TTMETAL_VERSION:0:$MIN_LEN}"

    log "Comparing prefixes (length $MIN_LEN): '$PARENT_PREFIX' vs '$BAD_PREFIX'"

    if [ "$BAD_PREFIX" != "$PARENT_PREFIX" ]; then
        log "✓ Detected tt-metal version change: $PARENT_TTMETAL_VERSION -> $BAD_TTMETAL_VERSION"
        IS_TTMETAL_UPLIFT=1
    else
        log "Versions are the same (no tt-metal uplift detected)"
    fi
else
    log "Could not extract TT_METAL_VERSION from one or both commits"
fi

# Return to tt-xla root
cd "$TTXLA_ROOT"
git checkout "$BAD_COMMIT" --quiet

if [ $IS_TTMETAL_UPLIFT -eq 0 ]; then
    # Final summary - tt-mlir regression
    log ""
    log "════════════════════════════════════════"
    log "FINAL RESULT"
    log "════════════════════════════════════════"
    log ""
    log "Root Cause Found in tt-mlir:"
    log "  Commit:  $FIRST_BAD_TTMLIR_SHORT"
    log "  Author:  $TTMLIR_COMMIT_AUTHOR"
    log "  Date:    $TTMLIR_COMMIT_DATE"
    log "  Message: $TTMLIR_COMMIT_MSG"
    log ""
    log "This commit was introduced to tt-xla in:"
    log "  Commit:  $FIRST_BAD_COMMIT_SHORT"
    log "  Message: $COMMIT_MSG"
    log ""
    log "View tt-mlir commit: cd $TTMLIR_DIR && git show $FIRST_BAD_TTMLIR_SHORT"
    log "View tt-xla commit:  git show $FIRST_BAD_COMMIT_SHORT"
    log ""
    log "All logs saved to: $LOG_DIR"
    log "Main log: $MAIN_LOG"
    log "tt-mlir test logs: $BISECT_RUN_DIR/bisect_ttmlir_*.log"
    log "════════════════════════════════════════"
    exit 0
fi

# Phase 3: Bisect tt-metal
log ""
log_separator
log "PHASE 3: Bisecting tt-metal commits"
log_separator

# Good and bad tt-metal versions already extracted
GOOD_TTMETAL="$PARENT_TTMETAL_VERSION"
BAD_TTMETAL="$BAD_TTMETAL_VERSION"

log "Good tt-metal version: $GOOD_TTMETAL"
log "Bad tt-metal version: $BAD_TTMETAL"

if [ "$GOOD_TTMETAL" = "$BAD_TTMETAL" ]; then
    log "ERROR: tt-metal versions are the same in good and bad commits!"
    log "Cannot bisect tt-metal."
    exit 1
fi

# Go to tt-metal submodule
TTMETAL_DIR="$TTMLIR_DIR/third_party/tt-metal/src/tt-metal"
if [ ! -d "$TTMETAL_DIR" ]; then
    log "ERROR: tt-metal directory not found: $TTMETAL_DIR"
    exit 1
fi

cd "$TTMETAL_DIR"
log "Changed to tt-metal directory: $(pwd)"

# Ensure we have the commits
git fetch origin 2>&1 | tee -a "$MAIN_LOG"

# Start bisect in tt-metal
log ""
log "Starting git bisect in tt-metal..."
git bisect start
git bisect bad "$BAD_TTMETAL"
git bisect good "$GOOD_TTMETAL"

log "Running tt-metal bisect with performance tests..."
TTMETAL_BISECT_SCRIPT="$BISECT_TTMETAL_SCRIPT"
TTMETAL_BISECT_ARGS="-c \"$BENCHMARK_COMMAND\" -t \"$PERF_THRESHOLD\" -p \"$METRIC_PATTERN\""

# Use the tt-mlir uplift commit as reference for fixing build failures
TTMETAL_BISECT_ARGS="$TTMETAL_BISECT_ARGS -f \"$FIRST_BAD_TTMLIR\""
log "Using tt-mlir commit $FIRST_BAD_TTMLIR_SHORT as build fix reference"

if [ -n "$REVERT_COMMIT" ]; then
    TTMETAL_BISECT_ARGS="$TTMETAL_BISECT_ARGS -r \"$REVERT_COMMIT\""
fi

# Run the bisect and capture output
eval "git bisect run \"$TTMETAL_BISECT_SCRIPT\" $TTMETAL_BISECT_ARGS" 2>&1 | tee -a "$MAIN_LOG"

# Capture the first bad tt-metal commit
FIRST_BAD_TTMETAL=$(git bisect view --pretty=format:'%H' | head -1)
FIRST_BAD_TTMETAL_SHORT=$(git rev-parse --short "$FIRST_BAD_TTMETAL")

log ""
log_separator
log "tt-metal bisect completed!"
log "First bad tt-metal commit: $FIRST_BAD_TTMETAL_SHORT"
log_separator

# Get commit details
TTMETAL_COMMIT_MSG=$(git log --format=%s -n 1 "$FIRST_BAD_TTMETAL")
TTMETAL_COMMIT_AUTHOR=$(git log --format='%an' -n 1 "$FIRST_BAD_TTMETAL")
TTMETAL_COMMIT_DATE=$(git log --format='%ad' -n 1 "$FIRST_BAD_TTMETAL")

git bisect reset

# Return to tt-xla root
cd "$TTXLA_ROOT"
git checkout "$BAD_COMMIT" --quiet

# Final summary - tt-metal regression
log ""
log "════════════════════════════════════════"
log "FINAL RESULT"
log "════════════════════════════════════════"
log ""
log "Root Cause Found in tt-metal:"
log "  Commit:  $FIRST_BAD_TTMETAL_SHORT"
log "  Author:  $TTMETAL_COMMIT_AUTHOR"
log "  Date:    $TTMETAL_COMMIT_DATE"
log "  Message: $TTMETAL_COMMIT_MSG"
log ""
log "This commit was introduced to tt-mlir in:"
log "  Commit:  $FIRST_BAD_TTMLIR_SHORT"
log "  Message: $TTMLIR_COMMIT_MSG"
log ""
log "Which was introduced to tt-xla in:"
log "  Commit:  $FIRST_BAD_COMMIT_SHORT"
log "  Message: $COMMIT_MSG"
log ""
log "View tt-metal commit: cd $TTMETAL_DIR && git show $FIRST_BAD_TTMETAL_SHORT"
log "View tt-mlir commit: cd $TTMLIR_DIR && git show $FIRST_BAD_TTMLIR_SHORT"
log "View tt-xla commit:  git show $FIRST_BAD_COMMIT_SHORT"
log ""
log "All logs saved to: $LOG_DIR"
log "Main log: $MAIN_LOG"
log "tt-mlir test logs: $BISECT_RUN_DIR/bisect_ttmlir_*.log"
log "tt-metal test logs: $BISECT_RUN_DIR/bisect_ttmetal_*.log"
log "════════════════════════════════════════"

exit 0
