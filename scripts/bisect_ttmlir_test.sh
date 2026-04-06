#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_ttmlir_test.sh [OPTIONS]
#
# Automated git bisect script for tt-mlir functional test regression
# This script is called by git bisect run from within the tt-mlir submodule.
# It builds tt-xla with the current tt-mlir commit and runs the test.
#
# OPTIONS:
#   -t, --test TEST_ID          Pytest node ID to run (required)
#   -f, --fix-build-ref COMMIT  Reference tt-mlir commit to use for fixing build failures
#                               (the bad/uplift tt-mlir version that is known to build correctly)
#   -h, --help                  Show this help message
#
# EXAMPLES:
#   # Manual usage from within tt-mlir submodule
#   cd third_party/tt-mlir/src/tt-mlir
#   git bisect start
#   git bisect bad 0e3246fb8
#   git bisect good ca7ac8c3c
#   git bisect run ../../../../scripts/bisect_ttmlir_test.sh \
#     -t "tests/jax/single_chip/test_ops.py::test_add"
#
#   # With build fix reference (when intermediate commits may have API breaks)
#   git bisect run ../../../../scripts/bisect_ttmlir_test.sh \
#     -t "tests/jax/single_chip/test_ops.py::test_add" \
#     -f 0e3246fb8
#
# EXIT CODES:
#   0   - Good commit (test passes)
#   1   - Bad commit (test fails)
#   125 - Untestable commit (build failed, environment issue, etc.)

TEST_ID=""
FIX_BUILD_REF=""

show_help() {
    sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            TEST_ID="$2"
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

if [ -z "$TEST_ID" ]; then
    echo "Error: Test ID is required (-t, --test)"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Determine tt-xla root (script is in scripts/ subdirectory of tt-xla)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTXLA_ROOT="$(dirname "$SCRIPT_DIR")"

# tt-mlir source dir is the git bisect working directory
TTMLIR_SRC_DIR=$(pwd)

TTMLIR_COMMIT=$(git rev-parse --short HEAD)
TTMLIR_COMMIT_FULL=$(git rev-parse HEAD)

# Create .bisect-run directory in tt-xla root for all bisect artifacts
BISECT_RUN_DIR="$TTXLA_ROOT/.bisect-run"
mkdir -p "$BISECT_RUN_DIR"

LOG_FILE="$BISECT_RUN_DIR/bisect_ttmlir_test_${TTMLIR_COMMIT}.log"

echo "======================================"
echo "Testing tt-mlir commit: $TTMLIR_COMMIT"
echo "Test: $TEST_ID"
echo "Log file: $LOG_FILE"
echo "======================================"

touch "$LOG_FILE"
echo "Testing tt-mlir commit: $TTMLIR_COMMIT" > "$LOG_FILE"
echo "Full commit: $TTMLIR_COMMIT_FULL" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"

# Cleanup function
cleanup() {
    cd "$TTXLA_ROOT" 2>/dev/null || true
    git checkout third_party/CMakeLists.txt 2>/dev/null || true
    if [ -n "$FIX_BUILD_REF" ]; then
        cd "$TTMLIR_SRC_DIR" 2>/dev/null || true
        git checkout "$FIX_BUILD_REF" --quiet 2>/dev/null || true
        git reset --hard HEAD 2>/dev/null || true
        git clean -fd 2>/dev/null || true
        cd "$TTXLA_ROOT" 2>/dev/null || true
    fi
}

cd "$TTXLA_ROOT"

# Activate the TT-XLA environment
echo "Activating TT-XLA environment..." | tee -a "$LOG_FILE"
source venv/activate

# Modify third_party/CMakeLists.txt to use current tt-mlir commit
echo "Setting tt-mlir version to $TTMLIR_COMMIT_FULL..." | tee -a "$LOG_FILE"
sed -i "s/set(TT_MLIR_VERSION \"[^\"]*\")/set(TT_MLIR_VERSION \"$TTMLIR_COMMIT_FULL\")/" third_party/CMakeLists.txt

# Build with CMake (incremental build)
echo "Building project..." | tee -a "$LOG_FILE"
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release >> "$LOG_FILE" 2>&1
if ! cmake --build build >> "$LOG_FILE" 2>&1; then
    echo "Build failed" | tee -a "$LOG_FILE"

    if [ -n "$FIX_BUILD_REF" ]; then
        echo "Attempting to fix build using reference commit $FIX_BUILD_REF with Claude agent..." | tee -a "$LOG_FILE"

        cd "$TTMLIR_SRC_DIR"

        # Reset ALL local changes before checkout
        echo "Resetting all local changes..." | tee -a "$LOG_FILE"
        git reset --hard HEAD >> "$LOG_FILE" 2>&1
        git clean -fd >> "$LOG_FILE" 2>&1

        # Checkout parent of FIX_BUILD_REF to start BEFORE its compatibility fixes
        echo "Checking out parent of $FIX_BUILD_REF (before compatibility fixes)..." | tee -a "$LOG_FILE"
        if ! git checkout "${FIX_BUILD_REF}^" --quiet >> "$LOG_FILE" 2>&1; then
            echo "Failed to checkout parent of $FIX_BUILD_REF, marking as untestable" | tee -a "$LOG_FILE"
            cleanup
            exit 125
        fi

        CURRENT_TTMLIR_COMMIT=$(git rev-parse HEAD)

        # Update tt-xla's TT_MLIR_VERSION to match the parent commit
        cd "$TTXLA_ROOT"
        echo "Updating tt-xla's TT_MLIR_VERSION to parent commit..." | tee -a "$LOG_FILE"
        sed -i "s/set(TT_MLIR_VERSION \"[^\"]*\")/set(TT_MLIR_VERSION \"$CURRENT_TTMLIR_COMMIT\")/" third_party/CMakeLists.txt

        # Try building with the parent commit first — it might already work
        echo "Trying to build with parent commit (before compatibility fixes)..." | tee -a "$LOG_FILE"
        if cmake --build build >> "$LOG_FILE" 2>&1; then
            echo "✓ Build succeeded with parent commit! No Claude fixes needed." | tee -a "$LOG_FILE"
            # Fall through to test run
        else
            echo "Build still fails with parent commit, will invoke Claude..." | tee -a "$LOG_FILE"

            BUILD_ERRORS=$(tail -200 "$LOG_FILE")

            cd "$TTMLIR_SRC_DIR"
            CURRENT_TTMLIR_COMMIT=$(git rev-parse HEAD)

            FIX_PROMPT="IMPORTANT: This is an automated script. Work autonomously without asking any questions. Make your best judgment and proceed with fixes.

CONTEXT:
- tt-mlir has been checked out to commit $CURRENT_TTMLIR_COMMIT (parent of $FIX_BUILD_REF)
- This is BEFORE the compatibility fixes in $FIX_BUILD_REF were applied
- We are bisecting tt-mlir commits to find a functional test regression
- The build is failing due to API incompatibilities between this intermediate tt-mlir and tt-xla

REFERENCE COMMIT:
- Commit $FIX_BUILD_REF is the bad/uplift tt-mlir version that is known to build with tt-xla
- Examine the diff between $CURRENT_TTMLIR_COMMIT and $FIX_BUILD_REF to see what API changes exist
- Those changes may not apply directly — adapt them for this intermediate version

YOUR TASK:
1. Read the build errors below to understand what is failing

   COMMON ERROR TYPES:
   a) API signature mismatches — function calls with wrong number/type of arguments
      → Examine tt-mlir source at $CURRENT_TTMLIR_COMMIT and adjust tt-xla calls to match
   b) Missing symbols — functions/types that do not exist in this version
      → May need to use alternative APIs or conditionally compile
   c) Header or type redefinition errors
      → Add include guards, conditional compilation, or fix include order

2. Look at the diff: git diff $CURRENT_TTMLIR_COMMIT $FIX_BUILD_REF
3. Adapt the necessary fixes to this specific intermediate version
4. Modify files in tt-xla ($TTXLA_ROOT) or tt-mlir ($TTMLIR_SRC_DIR) as needed
5. Test the build IN THIS EXACT ORDER:
   cd $TTXLA_ROOT
   source venv/activate
   cmake --build build
6. Iterate until build succeeds or you determine it is not fixable

BUILD ERRORS:
---
$BUILD_ERRORS
---

ENVIRONMENT:
- tt-xla root: $TTXLA_ROOT
- tt-mlir source: $TTMLIR_SRC_DIR

DO NOT commit anything, ask questions, or wait for confirmation.
VERIFY your fixes by building before finishing."

            PROMPT_FILE="$BISECT_RUN_DIR/fix_build_prompt_ttmlir_test_${TTMLIR_COMMIT}.txt"
            echo "$FIX_PROMPT" > "$PROMPT_FILE"

            CLAUDE_OUTPUT_LOG="$BISECT_RUN_DIR/claude_output_ttmlir_test_${TTMLIR_COMMIT}.log"

            cd "$TTXLA_ROOT"
            echo "Invoking Claude to fix build..." | tee -a "$LOG_FILE"
            echo "Claude prompt: $PROMPT_FILE" | tee -a "$LOG_FILE"
            echo "=====================================" | tee -a "$LOG_FILE"

            if timeout 600 claude -p "$FIX_PROMPT" --model opus --allowed-tools "Read Edit Bash Glob Grep" --permission-mode bypassPermissions --verbose > "$CLAUDE_OUTPUT_LOG" 2>&1; then
                cat "$CLAUDE_OUTPUT_LOG" >> "$LOG_FILE"
                echo "" >> "$LOG_FILE"
                echo "=====================================" | tee -a "$LOG_FILE"
                echo "Claude completed build fix attempt" | tee -a "$LOG_FILE"

                cd "$TTMLIR_SRC_DIR"
                CHANGED_FILES=$(git status --short 2>/dev/null || echo "Could not get git status")
                echo "Files modified by Claude:" | tee -a "$LOG_FILE"
                echo "$CHANGED_FILES" | tee -a "$LOG_FILE"
                if [ -z "$CHANGED_FILES" ] || [ "$CHANGED_FILES" = "Could not get git status" ]; then
                    echo "⚠ WARNING: No files were modified by Claude!" | tee -a "$LOG_FILE"
                fi
                echo "=====================================" | tee -a "$LOG_FILE"

                cd "$TTXLA_ROOT"
                echo "Rebuilding after Claude's fixes..." | tee -a "$LOG_FILE"
                REBUILD_LOG="$BISECT_RUN_DIR/rebuild_ttmlir_test_${TTMLIR_COMMIT}.log"
                if cmake --build build > "$REBUILD_LOG" 2>&1; then
                    echo "✓ Build succeeded after Claude's fixes!" | tee -a "$LOG_FILE"
                    cat "$REBUILD_LOG" >> "$LOG_FILE"
                    # Fall through to test run
                else
                    echo "✗ Build still failed after Claude's fixes" | tee -a "$LOG_FILE"
                    tail -50 "$REBUILD_LOG" | tee -a "$LOG_FILE"
                    echo "  Claude output: $CLAUDE_OUTPUT_LOG" | tee -a "$LOG_FILE"
                    echo "  Claude prompt: $PROMPT_FILE" | tee -a "$LOG_FILE"
                    echo "  Rebuild errors: $REBUILD_LOG" | tee -a "$LOG_FILE"
                    echo "Marking commit as untestable" | tee -a "$LOG_FILE"
                    cleanup
                    exit 125
                fi
            else
                CLAUDE_EXIT_CODE=$?
                cat "$CLAUDE_OUTPUT_LOG" >> "$LOG_FILE"
                echo "" | tee -a "$LOG_FILE"
                echo "✗ Claude invocation failed (exit code $CLAUDE_EXIT_CODE)" | tee -a "$LOG_FILE"
                echo "  Claude output: $CLAUDE_OUTPUT_LOG" | tee -a "$LOG_FILE"
                echo "Marking commit as untestable" | tee -a "$LOG_FILE"
                cleanup
                exit 125
            fi
        fi
    else
        echo "Build failed, marking as untestable" | tee -a "$LOG_FILE"
        cleanup
        exit 125
    fi
fi
echo "Build completed successfully" | tee -a "$LOG_FILE"

# Restart chip before running test
echo "Restarting chip..." | tee -a "$LOG_FILE"
tt-smi -r 2>&1 | tee -a "$LOG_FILE" || {
    echo "tt-smi failed — reinstalling" | tee -a "$LOG_FILE"
    deactivate 2>/dev/null || true
    pip uninstall tt-smi -y 2>/dev/null || true
    source "$TTXLA_ROOT/venv/activate"
    pip install tt-smi 2>&1 | tee -a "$LOG_FILE"
    tt-smi -r 2>&1 | tee -a "$LOG_FILE"
}
sleep 3

# Run the test
echo "Running test: $TEST_ID" | tee -a "$LOG_FILE"
python -m pytest -x "$TEST_ID" 2>&1 | tee -a "$LOG_FILE"
TEST_EXIT=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
if [ $TEST_EXIT -eq 0 ]; then
    echo "✓ GOOD: Test passed" | tee -a "$LOG_FILE"
    cleanup
    exit 0
else
    echo "✗ BAD: Test failed (exit code $TEST_EXIT)" | tee -a "$LOG_FILE"
    cleanup
    exit 1
fi
