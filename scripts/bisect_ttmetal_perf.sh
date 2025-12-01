#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_ttmetal_perf.sh [OPTIONS]
#
# Automated git bisect script for tt-metal performance testing
# This script is called by git bisect run from within the tt-metal submodule
# It builds tt-xla with the current tt-metal commit and tests performance
#
# OPTIONS:
#   -c, --command COMMAND          Benchmark command to run (required if no defaults)
#   -t, --threshold THRESHOLD      Performance threshold value (required if no defaults)
#   -p, --pattern PATTERN          Grep pattern to extract metric (default: "Sample per second:\s*\K[0-9.]+")
#   -r, --revert COMMIT            Revert this tt-mlir commit if present in history (useful for isolating additional regressions)
#   -f, --fix-build-ref COMMIT     Reference tt-mlir commit to use for fixing build failures (e.g., fb45bf24c861)
#   -h, --help                     Show this help message
#
# EXAMPLES:
#   # Use with default resnet benchmark (threshold 680)
#   cd third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
#   git bisect start
#   git bisect bad HEAD
#   git bisect good abc123
#   git bisect run ../../../../../../scripts/bisect_ttmetal_perf.sh
#
#   # Custom benchmark and threshold
#   git bisect run ../../../../../../scripts/bisect_ttmetal_perf.sh \
#     -c "python ../tt-forge/benchmark/benchmark.py -p tt-xla -m vgg16 -bs 8 -df bfloat16 -lp 128" \
#     -t 500
#
#   # Revert a known bad commit to isolate additional regressions
#   git bisect run ../../../../../../scripts/bisect_ttmetal_perf.sh \
#     -r abc123def456
#
#   # Use reference tt-mlir commit to auto-fix build failures
#   git bisect run ../../../../../../scripts/bisect_ttmetal_perf.sh \
#     --fix-build-ref fb45bf24c861c168ece23dc2ae237c696a88cff3
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
FIX_BUILD_REF=""

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
        -f|--fix-build-ref)
            FIX_BUILD_REF="$2"
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

TTMETAL_COMMIT=$(git rev-parse --short HEAD)
TTMETAL_COMMIT_FULL=$(git rev-parse HEAD)

# Create .bisect-run directory in tt-xla root for all bisect artifacts
BISECT_RUN_DIR="$TTXLA_ROOT/.bisect-run"
mkdir -p "$BISECT_RUN_DIR"

LOG_FILE="$BISECT_RUN_DIR/bisect_ttmetal_${TTMETAL_COMMIT}.log"

echo "======================================"
echo "Testing tt-metal commit: $TTMETAL_COMMIT"
echo "Benchmark: $BENCHMARK_COMMAND"
echo "Threshold: $PERF_THRESHOLD"
echo "Log file: $LOG_FILE"
echo "======================================"

# Create log file
touch "$LOG_FILE"
echo "Testing tt-metal commit: $TTMETAL_COMMIT" > "$LOG_FILE"
echo "Full commit: $TTMETAL_COMMIT_FULL" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

# Update submodules for this commit
echo "Updating tt-metal submodules..." | tee -a "$LOG_FILE"
if ! git submodule update --init --recursive >> "$LOG_FILE" 2>&1; then
    echo "Warning: Submodule update failed, continuing anyway" | tee -a "$LOG_FILE"
fi

# Navigate to directories first (needed for revert logic)
# tt-metal is at: tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
# Up 8 levels: ../../../../../../../../
TTMETAL_DIR=$(pwd)
TTXLA_ROOT="$(cd "$TTMETAL_DIR/../../../../../../../.." && pwd)"
# tt-mlir third_party dir is 3 levels up from tt-metal
TTMLIR_THIRD_PARTY_DIR="$(cd "$TTMETAL_DIR/../../.." && pwd)"
# tt-mlir source root is parent of third_party
TTMLIR_SRC_DIR="$(dirname "$TTMLIR_THIRD_PARTY_DIR")"

# Check if we need to revert a specific tt-mlir commit
if [ -n "$REVERT_COMMIT" ]; then
    echo "Checking if tt-mlir revert commit $REVERT_COMMIT is in history..." | tee -a "$LOG_FILE"
    cd "$TTMLIR_SRC_DIR"
    if git merge-base --is-ancestor "$REVERT_COMMIT" HEAD 2>/dev/null; then
        echo "Reverting tt-mlir commit $REVERT_COMMIT..." | tee -a "$LOG_FILE"
        if git revert --no-commit "$REVERT_COMMIT" >> "$LOG_FILE" 2>&1; then
            echo "Successfully reverted $REVERT_COMMIT in tt-mlir" | tee -a "$LOG_FILE"
        else
            echo "Failed to revert $REVERT_COMMIT in tt-mlir, marking as untestable" | tee -a "$LOG_FILE"
            git revert --abort 2>/dev/null || true
            cd "$TTXLA_ROOT"
            exit 125
        fi
    else
        echo "Commit $REVERT_COMMIT not in tt-mlir history, skipping revert" | tee -a "$LOG_FILE"
    fi
fi

cd "$TTXLA_ROOT"

# Cleanup function (defined here for use throughout the script)
cleanup() {
    cd "$TTXLA_ROOT" 2>/dev/null || true
    # Restore tt-mlir third_party/CMakeLists.txt
    cd "$TTMLIR_THIRD_PARTY_DIR" 2>/dev/null || true
    git checkout CMakeLists.txt 2>/dev/null || true
    # Restore tt-xla third_party/CMakeLists.txt (from cmake_fix.patch)
    cd "$TTXLA_ROOT" 2>/dev/null || true
    git checkout third_party/CMakeLists.txt 2>/dev/null || true
    # Restore tt-mlir to original commit if we checked out parent
    if [ -n "$FIX_BUILD_REF" ]; then
        cd "$TTMLIR_SRC_DIR" 2>/dev/null || true
        git checkout "$FIX_BUILD_REF" --quiet 2>/dev/null || true
        git reset --hard HEAD 2>/dev/null || true
        git clean -fd 2>/dev/null || true  # Remove untracked files (e.g., venv/ created by Claude)
        cd "$TTXLA_ROOT" 2>/dev/null || true
    elif [ -n "$REVERT_COMMIT" ]; then
        # Reset tt-mlir to undo the revert
        cd "$TTMLIR_SRC_DIR" 2>/dev/null || true
        git reset --hard HEAD 2>/dev/null || true
        git clean -fd 2>/dev/null || true  # Remove untracked files
        cd "$TTXLA_ROOT" 2>/dev/null || true
    fi
}

# Activate the TT-XLA environment
echo "Activating TT-XLA environment..." | tee -a "$LOG_FILE"
source venv/activate

# Apply the CMakeLists.txt fix for incremental builds
echo "Applying CMakeLists.txt fix..." | tee -a "$LOG_FILE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/cmake_fix.patch" ]; then
    git apply "$SCRIPT_DIR/cmake_fix.patch" 2>/dev/null || echo "Patch already applied or not needed"
fi

# Modify tt-mlir third_party/CMakeLists.txt to use current tt-metal commit
echo "Setting tt-metal version to $TTMETAL_COMMIT_FULL..." | tee -a "$LOG_FILE"
cd "$TTMLIR_THIRD_PARTY_DIR"
sed -i "s/set(TT_METAL_VERSION \"[^\"]*\")/set(TT_METAL_VERSION \"$TTMETAL_COMMIT_FULL\")/" CMakeLists.txt

cd "$TTXLA_ROOT"

# Build with CMake (incremental build)
echo "Building project..." | tee -a "$LOG_FILE"
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release >> "$LOG_FILE" 2>&1
if ! cmake --build build >> "$LOG_FILE" 2>&1; then
    echo "Build failed" | tee -a "$LOG_FILE"

    # If fix-build-ref is provided, try to fix build using Claude agent
    if [ -n "$FIX_BUILD_REF" ]; then
        echo "Attempting to fix build using reference commit $FIX_BUILD_REF with Claude agent..." | tee -a "$LOG_FILE"

        # Extract recent build errors (last 200 lines)
        BUILD_ERRORS=$(tail -200 "$LOG_FILE")

        cd "$TTMLIR_SRC_DIR"

        # Reset ALL local changes (revert, Claude modifications, CMakeLists.txt, etc.)
        echo "Resetting all local changes to prepare for parent checkout..." | tee -a "$LOG_FILE"
        git reset --hard HEAD >> "$LOG_FILE" 2>&1
        git clean -fd >> "$LOG_FILE" 2>&1

        # Checkout parent of FIX_BUILD_REF so we start BEFORE the compatibility fixes
        echo "Checking out parent of $FIX_BUILD_REF (before compatibility fixes)..." | tee -a "$LOG_FILE"
        if ! git checkout "${FIX_BUILD_REF}^" --quiet >> "$LOG_FILE" 2>&1; then
            echo "Failed to checkout parent of $FIX_BUILD_REF, marking as untestable" | tee -a "$LOG_FILE"
            cleanup
            exit 125
        fi

        # Re-apply TT_METAL_VERSION modification to parent commit
        echo "Re-applying TT_METAL_VERSION modification on parent commit..." | tee -a "$LOG_FILE"
        cd "$TTMLIR_THIRD_PARTY_DIR"
        sed -i "s/set(TT_METAL_VERSION \"[^\"]*\")/set(TT_METAL_VERSION \"$TTMETAL_COMMIT_FULL\")/" CMakeLists.txt

        # Now re-apply the revert if needed (to the parent commit)
        if [ -n "$REVERT_COMMIT" ]; then
            cd "$TTMLIR_SRC_DIR"
            if git merge-base --is-ancestor "$REVERT_COMMIT" HEAD 2>/dev/null; then
                echo "Re-applying revert of $REVERT_COMMIT on parent commit..." | tee -a "$LOG_FILE"
                if ! git revert --no-commit "$REVERT_COMMIT" >> "$LOG_FILE" 2>&1; then
                    echo "Failed to re-apply revert on parent, marking as untestable" | tee -a "$LOG_FILE"
                    git revert --abort 2>/dev/null || true
                    cleanup
                    exit 125
                fi
            fi
        fi

        cd "$TTMLIR_SRC_DIR"
        CURRENT_TTMLIR_COMMIT=$(git rev-parse HEAD)

        # Update tt-xla's TT_MLIR_VERSION to match the parent commit we checked out
        # This prevents ExternalProject from trying to checkout FIX_BUILD_REF and hitting git conflicts
        echo "Updating tt-xla's TT_MLIR_VERSION to parent commit..." | tee -a "$LOG_FILE"
        cd "$TTXLA_ROOT"
        sed -i "s/set(TT_MLIR_VERSION \"[^\"]*\")/set(TT_MLIR_VERSION \"$CURRENT_TTMLIR_COMMIT\")/" third_party/CMakeLists.txt

        # Try building with the parent commit first - it might already work!
        echo "Trying to build with parent commit (before compatibility fixes)..." | tee -a "$LOG_FILE"
        cd "$TTXLA_ROOT"
        if cmake --build build >> "$LOG_FILE" 2>&1; then
            echo "✓ Build succeeded with parent commit! No Claude fixes needed." | tee -a "$LOG_FILE"
            # Continue with benchmark (don't exit, let the script continue below)
        else
            echo "Build still fails with parent commit, will invoke Claude..." | tee -a "$LOG_FILE"

            # Extract recent build errors (last 200 lines)
            BUILD_ERRORS=$(tail -200 "$LOG_FILE")

            cd "$TTMLIR_SRC_DIR"
            CURRENT_TTMLIR_COMMIT=$(git rev-parse HEAD)

            # Create a prompt for Claude to fix the build (with errors embedded inline)
            FIX_PROMPT="IMPORTANT: This is an automated script. Work autonomously without asking any questions. Make your best judgment and proceed with fixes.

CONTEXT:
- tt-mlir has been checked out to commit $CURRENT_TTMLIR_COMMIT (parent of $FIX_BUILD_REF)
- This is BEFORE the compatibility fixes were applied
- We're testing with tt-metal commit $TTMETAL_COMMIT (an intermediate version)
- The build is failing due to API incompatibilities between tt-mlir and tt-metal

REFERENCE COMMIT:
- Commit $FIX_BUILD_REF contains compatibility fixes for tt-metal
- You can examine the diff to see what API changes were made
- However, that commit was written for the FINAL tt-metal version, not this intermediate one

YOUR TASK:
1. Read the build errors below to understand what's failing

   COMMON ERROR TYPES:
   a) API signature mismatches - function calls with wrong number/type of arguments
      → Fix by examining tt-metal API at commit $TTMETAL_COMMIT and adjusting calls

   b) Header redefinition errors - types/enums defined in BOTH build/include/ AND source directories
      → This happens when headers are both generated and in source tree
      → Solutions:
         - Modify tt-mlir's common.h to exclude one include path
         - Add include guards or conditional compilation
         - Check if CMakeLists.txt needs updates to exclude duplicate headers
         - Look at whether build/include versions are auto-generated and should be preferred

   c) Missing symbols - functions/types that don't exist in this tt-metal version
      → May need to use alternative APIs or conditionally compile

2. Examine the diff between $CURRENT_TTMLIR_COMMIT and $FIX_BUILD_REF to see what fixes were made
   - Note: Those fixes were for the FINAL tt-metal version, not this intermediate one

3. Adapt those fixes for THIS specific tt-metal version ($TTMETAL_COMMIT)
   - The intermediate tt-metal API may be different from the final version
   - You may need to add OR remove parameters/arguments depending on the API
   - For header conflicts, check both build/include/ and source directories
   - Examine the actual tt-metal code at commit $TTMETAL_COMMIT to understand the correct API

4. Make the necessary changes to fix the build
   - Focus on tt-mlir files (in $TTMLIR_SRC_DIR)
   - For header issues, prioritize modifying include statements in tt-mlir's common.h or similar files
   - Avoid modifying tt-metal source directly - it's in a git submodule

5. Test the build using these commands IN THIS EXACT ORDER:
   cd $TTXLA_ROOT
   source venv/activate
   cmake --build build

6. If build fails, iterate until it succeeds or you determine it's not fixable
   - Check if new errors appear after your fixes
   - For persistent header conflicts, consider if the build/ generated headers should take precedence

BUILD ERRORS:
---
$BUILD_ERRORS
---

ENVIRONMENT:
- Your current working directory: $TTXLA_ROOT
- tt-xla root: $TTXLA_ROOT
- tt-mlir source: $TTMLIR_SRC_DIR
- To build correctly: First cd $TTXLA_ROOT, then source venv/activate, then cmake --build build

DO NOT:
- Commit anything
- Ask questions
- Wait for confirmation

VERIFY your fixes by building before finishing."

            # Save prompt to file in .bisect-run directory
            PROMPT_FILE="$BISECT_RUN_DIR/fix_build_prompt_${TTMETAL_COMMIT}.txt"
            echo "$FIX_PROMPT" > "$PROMPT_FILE"

            # Invoke Claude CLI to fix the build
            echo "Invoking Claude to fix build..." | tee -a "$LOG_FILE"
            echo "Claude prompt: $PROMPT_FILE" | tee -a "$LOG_FILE"
            echo "=====================================" | tee -a "$LOG_FILE"

            # Save Claude's output separately for debugging
            CLAUDE_OUTPUT_LOG="$BISECT_RUN_DIR/claude_output_${TTMETAL_COMMIT}.log"

            cd "$TTXLA_ROOT"
            # Use timeout to prevent hanging (10 minutes max), pass prompt with -p flag
            # Allow only necessary tools: Read, Edit, Bash, Glob, Grep
            # Bypass permissions for fully automated operation
            # Use Opus model for better reasoning on complex API compatibility issues
            if timeout 600 claude -p "$FIX_PROMPT" --model opus --allowed-tools "Read Edit Bash Glob Grep" --permission-mode bypassPermissions --verbose > "$CLAUDE_OUTPUT_LOG" 2>&1; then
                # Append Claude's output to main log
                cat "$CLAUDE_OUTPUT_LOG" >> "$LOG_FILE"
                echo "" >> "$LOG_FILE"
                echo "=====================================" | tee -a "$LOG_FILE"
                echo "Claude completed build fix attempt" | tee -a "$LOG_FILE"

                # Check what files Claude changed in tt-mlir
                cd "$TTMLIR_SRC_DIR"
                CHANGED_FILES=$(git status --short 2>/dev/null || echo "Could not get git status")
                echo "Files modified by Claude:" | tee -a "$LOG_FILE"
                echo "$CHANGED_FILES" | tee -a "$LOG_FILE"

                if [ -z "$CHANGED_FILES" ] || [ "$CHANGED_FILES" = "Could not get git status" ]; then
                    echo "⚠ WARNING: No files were modified by Claude!" | tee -a "$LOG_FILE"
                fi
                echo "=====================================" | tee -a "$LOG_FILE"

                # Try rebuilding
                cd "$TTXLA_ROOT"
                echo "Rebuilding after Claude's fixes..." | tee -a "$LOG_FILE"
                REBUILD_LOG="$TTXLA_ROOT/rebuild_${TTMETAL_COMMIT}.log"
                if cmake --build build > "$REBUILD_LOG" 2>&1; then
                    echo "✓ Build succeeded after Claude's fixes!" | tee -a "$LOG_FILE"
                    cat "$REBUILD_LOG" >> "$LOG_FILE"
                    # Continue with benchmark (don't exit here)
                else
                    echo "✗ Build still failed after Claude's fixes" | tee -a "$LOG_FILE"
                    echo "New build errors:" | tee -a "$LOG_FILE"
                    tail -50 "$REBUILD_LOG" | tee -a "$LOG_FILE"
                    echo "" | tee -a "$LOG_FILE"
                    echo "Debug information:" | tee -a "$LOG_FILE"
                    echo "  - Initial errors: $BUILD_ERROR_LOG" | tee -a "$LOG_FILE"
                    echo "  - Claude output: $CLAUDE_OUTPUT_LOG" | tee -a "$LOG_FILE"
                    echo "  - Claude prompt: $PROMPT_FILE" | tee -a "$LOG_FILE"
                    echo "  - Rebuild errors: $REBUILD_LOG" | tee -a "$LOG_FILE"
                    echo "  - Changed files: $CHANGED_FILES" | tee -a "$LOG_FILE"
                    echo "Marking commit as untestable" | tee -a "$LOG_FILE"
                    cleanup
                    exit 125
                fi
            else
                CLAUDE_EXIT_CODE=$?
                cat "$CLAUDE_OUTPUT_LOG" >> "$LOG_FILE"
                echo "" | tee -a "$LOG_FILE"
                echo "✗ Claude invocation failed (exit code $CLAUDE_EXIT_CODE)" | tee -a "$LOG_FILE"
                echo "Claude output: $CLAUDE_OUTPUT_LOG" | tee -a "$LOG_FILE"
                echo "Marking commit as untestable" | tee -a "$LOG_FILE"
                cleanup
                exit 125
            fi
        fi  # End of "else: build still fails with parent" block
        echo "=====================================" | tee -a "$LOG_FILE"
    else
        # No fix-build-ref provided, just mark as untestable
        echo "Build failed, marking as untestable" | tee -a "$LOG_FILE"
        cleanup
        exit 125  # Tell git bisect to skip this commit
    fi
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
