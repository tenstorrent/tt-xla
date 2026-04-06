#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_test_auto.sh [OPTIONS]
#
# Automated functional test bisect tool that handles both tt-xla and tt-mlir bisecting.
# Phase 1 bisects tt-xla commits using pre-built CI wheels (no local build needed).
# If the blamed commit is a tt-mlir uplift, Phase 2 bisects into tt-mlir using a
# local build, invoking Claude automatically to fix any intermediate build failures.
#
# OPTIONS:
#   -t, --test TEST_ID          Pytest node ID to run (required)
#   -g, --good-sha SHA          Last known-good tt-xla commit (required)
#   -b, --bad-sha SHA           First known-bad tt-xla commit (default: HEAD)
#   -l, --log-dir DIR           Log directory (default: .bisect-run/bisect_test_auto_TIMESTAMP)
#   -h, --help                  Show this help message
#
# EXAMPLES:
#   # Basic usage
#   ./scripts/bisect_test_auto.sh \
#     -t "tests/jax/single_chip/test_ops.py::test_add" \
#     -g 051ebb20 \
#     -b HEAD
#
#   # Full pytest node ID from a CI failure
#   ./scripts/bisect_test_auto.sh \
#     -t "tests/runner/test_models.py::test_all_models_torch[resnet/pytorch-single_device-inference]" \
#     -g 051ebb20 \
#     -b a3f91bc2
#
# EXIT CODES:
#   0   - Successfully identified bad commit
#   1   - Error during bisect process

TEST_ID=""
GOOD_SHA=""
BAD_SHA="HEAD"
BISECT_RUN_DIR="$(pwd)/.bisect-run"
mkdir -p "$BISECT_RUN_DIR"
LOG_DIR="$BISECT_RUN_DIR/bisect_test_auto_$(date +%Y%m%d_%H%M%S)"

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
        -g|--good-sha)
            GOOD_SHA="$2"
            shift 2
            ;;
        -b|--bad-sha)
            BAD_SHA="$2"
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

if [ -z "$TEST_ID" ]; then
    echo "Error: Test ID is required (-t, --test)"
    echo "Use -h or --help for usage information"
    exit 1
fi
if [ -z "$GOOD_SHA" ]; then
    echo "Error: Good commit is required (-g, --good-sha)"
    echo "Use -h or --help for usage information"
    exit 1
fi

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/bisect_test_auto.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

log_separator() {
    echo "========================================" | tee -a "$MAIN_LOG"
}

log_separator
log "Starting Automated Test Bisect"
log_separator
log "Test:       $TEST_ID"
log "Good SHA:   $GOOD_SHA"
log "Bad SHA:    $BAD_SHA"
log "Log dir:    $LOG_DIR"
log_separator

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTXLA_ROOT="$(dirname "$SCRIPT_DIR")"

# Copy bisect scripts to .bisect-run to preserve them during git operations
log "Copying bisect scripts to $BISECT_RUN_DIR/scripts for safety..."
mkdir -p "$BISECT_RUN_DIR/scripts"
cp "$SCRIPT_DIR"/bisect*.sh "$BISECT_RUN_DIR/scripts/"
log "Scripts copied to: $BISECT_RUN_DIR/scripts/"

BISECT_TEST_SCRIPT="$BISECT_RUN_DIR/scripts/bisect_test.sh"
BISECT_TTMLIR_TEST_SCRIPT="$BISECT_RUN_DIR/scripts/bisect_ttmlir_test.sh"

cd "$TTXLA_ROOT"
log "Working directory: $(pwd)"
log ""

# ---------------------------------------------------------------------------
# Phase 1: Bisect tt-xla using CI wheels
# ---------------------------------------------------------------------------
log_separator
log "PHASE 1: Bisecting tt-xla commits (using CI wheels)"
log_separator

# Pre-verification: ensure good commit passes and bad commit fails before bisecting
log "Pre-verification: testing good commit ($GOOD_SHA)..."
git checkout "$GOOD_SHA" --quiet
"$BISECT_TEST_SCRIPT" -t "$TEST_ID"
PREVERIFY_GOOD_EXIT=$?
if [ $PREVERIFY_GOOD_EXIT -ne 0 ]; then
    log "ERROR: Test does not pass on good commit ($GOOD_SHA) — aborting bisect."
    log "       Exit code: $PREVERIFY_GOOD_EXIT"
    log "       Fix the test environment and retry."
    git checkout "$BAD_SHA" --quiet
    exit 1
fi
log "Good commit verified: test passes on $GOOD_SHA"

log ""
log "Pre-verification: testing bad commit ($BAD_SHA)..."
git checkout "$BAD_SHA" --quiet
"$BISECT_TEST_SCRIPT" -t "$TEST_ID"
PREVERIFY_BAD_EXIT=$?
if [ $PREVERIFY_BAD_EXIT -eq 0 ]; then
    log "ERROR: Test passes on bad commit ($BAD_SHA) — it is not actually bad. Aborting."
    exit 1
elif [ $PREVERIFY_BAD_EXIT -eq 125 ]; then
    log "WARNING: Bad commit is untestable (no CI wheel / artifact expired). Proceeding with bisect anyway."
else
    log "Bad commit verified: test fails with exit code $PREVERIFY_BAD_EXIT on $BAD_SHA"
fi
log ""

log "Starting git bisect in tt-xla..."
git bisect start
git bisect bad "$BAD_SHA"
git bisect good "$GOOD_SHA"

log "Running bisect with test: $TEST_ID"
git bisect run "$BISECT_TEST_SCRIPT" -t "$TEST_ID" 2>&1 | tee -a "$MAIN_LOG"

# Capture the first bad commit
FIRST_BAD_COMMIT=$(git bisect view --pretty=format:'%H' | head -1)
FIRST_BAD_COMMIT_SHORT=$(git rev-parse --short "$FIRST_BAD_COMMIT")

log ""
log_separator
log "tt-xla bisect completed!"
log "First bad commit: $FIRST_BAD_COMMIT_SHORT"
log_separator

COMMIT_MSG=$(git log --format=%s -n 1 "$FIRST_BAD_COMMIT")
log "Commit message: $COMMIT_MSG"

git bisect reset

# Check if this is a tt-mlir uplift commit
IS_TTMLIR_UPLIFT=0
if echo "$COMMIT_MSG" | grep -iq "uplift.*tt-mlir\|tt-mlir.*uplift"; then
    log "✓ Detected tt-mlir uplift commit (message match)"
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
    log "This is NOT a tt-mlir uplift. The regression is in tt-xla code."
    log ""
    log "View details: git show $FIRST_BAD_COMMIT_SHORT"
    log "All logs saved to: $LOG_DIR"
    log "════════════════════════════════════════"
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: Bisect tt-mlir
# ---------------------------------------------------------------------------
log ""
log_separator
log "PHASE 2: Bisecting tt-mlir commits (local build)"
log_separator

log "Extracting tt-mlir versions from good and bad tt-xla commits..."

git checkout "$GOOD_SHA" --quiet
GOOD_TTMLIR=$(grep 'set(TT_MLIR_VERSION' third_party/CMakeLists.txt | grep -oP '"\K[^"]+' | head -1 | tr -d '\n\r')
log "Good commit tt-mlir version: $GOOD_TTMLIR"

git checkout "$FIRST_BAD_COMMIT" --quiet
BAD_TTMLIR=$(grep 'set(TT_MLIR_VERSION' third_party/CMakeLists.txt | grep -oP '"\K[^"]+' | head -1 | tr -d '\n\r')
log "Bad commit tt-mlir version:  $BAD_TTMLIR"

if [ "$GOOD_TTMLIR" = "$BAD_TTMLIR" ]; then
    log "ERROR: tt-mlir versions are the same in good and bad commits — cannot bisect."
    git checkout "$BAD_SHA" --quiet
    exit 1
fi

TTMLIR_DIR="$TTXLA_ROOT/third_party/tt-mlir/src/tt-mlir"
if [ ! -d "$TTMLIR_DIR" ]; then
    log "ERROR: tt-mlir directory not found: $TTMLIR_DIR"
    git checkout "$BAD_SHA" --quiet
    exit 1
fi

cd "$TTMLIR_DIR"
log "Changed to tt-mlir directory: $(pwd)"

git fetch origin 2>&1 | tee -a "$MAIN_LOG"

log ""
log "Starting git bisect in tt-mlir..."
git bisect start
git bisect bad "$BAD_TTMLIR"
git bisect good "$GOOD_TTMLIR"

# Pass BAD_TTMLIR as fix-build-ref: it is the tt-mlir version known to build with tt-xla
# at the bad/uplift commit, so Claude can use its diff as a reference for intermediate breaks
log "Running tt-mlir bisect with test: $TEST_ID"
log "Using $BAD_TTMLIR as build fix reference"
git bisect run "$BISECT_TTMLIR_TEST_SCRIPT" -t "$TEST_ID" -f "$BAD_TTMLIR" 2>&1 | tee -a "$MAIN_LOG"

FIRST_BAD_TTMLIR=$(git bisect view --pretty=format:'%H' | head -1)
FIRST_BAD_TTMLIR_SHORT=$(git rev-parse --short "$FIRST_BAD_TTMLIR")

log ""
log_separator
log "tt-mlir bisect completed!"
log "First bad tt-mlir commit: $FIRST_BAD_TTMLIR_SHORT"
log_separator

TTMLIR_COMMIT_MSG=$(git log --format=%s -n 1 "$FIRST_BAD_TTMLIR")
TTMLIR_COMMIT_AUTHOR=$(git log --format='%an' -n 1 "$FIRST_BAD_TTMLIR")
TTMLIR_COMMIT_DATE=$(git log --format='%ad' -n 1 "$FIRST_BAD_TTMLIR")

git bisect reset

# Return to tt-xla root at original bad commit
cd "$TTXLA_ROOT"
git checkout "$BAD_SHA" --quiet

log ""
log "════════════════════════════════════════"
log "FINAL RESULT"
log "════════════════════════════════════════"
log ""
log "Root cause found in tt-mlir:"
log "  Commit:  $FIRST_BAD_TTMLIR_SHORT"
log "  Author:  $TTMLIR_COMMIT_AUTHOR"
log "  Date:    $TTMLIR_COMMIT_DATE"
log "  Message: $TTMLIR_COMMIT_MSG"
log ""
log "Introduced to tt-xla in:"
log "  Commit:  $FIRST_BAD_COMMIT_SHORT"
log "  Message: $COMMIT_MSG"
log ""
log "View tt-mlir commit: cd $TTMLIR_DIR && git show $FIRST_BAD_TTMLIR_SHORT"
log "View tt-xla commit:  git show $FIRST_BAD_COMMIT_SHORT"
log ""
log "All logs saved to: $LOG_DIR"
log "Main log:          $MAIN_LOG"
log "tt-mlir test logs: $BISECT_RUN_DIR/bisect_ttmlir_test_*.log"
log "Claude outputs:    $BISECT_RUN_DIR/claude_output_ttmlir_test_*.log"
log "════════════════════════════════════════"

exit 0
