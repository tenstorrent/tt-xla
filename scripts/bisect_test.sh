#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Usage: bisect_test.sh [OPTIONS]
#
# Automated git bisect script for functional test regression in tt-xla
#
# Run as orchestrator (verifies good/bad commits, then bisects):
#   bisect_test.sh -t TEST_ID -g GOOD_SHA -b BAD_SHA
#
# Run as bisect runner (called internally by git bisect run):
#   bisect_test.sh -t TEST_ID
#
# OPTIONS:
#   -t, --test TEST_ID          Pytest node ID to run (required)
#   -g, --good-sha SHA          Last known-good commit SHA (required for orchestration mode)
#   -b, --bad-sha SHA           First known-bad commit SHA (required for orchestration mode)
#   -h, --help                  Show this help message
#
# EXAMPLES:
#   ./scripts/general/bisect_test.sh \
#     -t "tests/runner/test_models.py::test_all_models_torch[resnet/pytorch-single_device-inference]" \
#     -g 051ebb20 \
#     -b HEAD
#
# EXIT CODES:
#   0   - Good commit (test passes)
#   1   - Bad commit (test fails)
#   125 - Untestable commit (no CI wheel, artifact expired, env issue, etc.)

TEST_ID=""
GOOD_SHA=""
BAD_SHA=""

show_help() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
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

# Orchestration mode requires both --good-sha and --bad-sha
if [ -n "$GOOD_SHA" ] && [ -z "$BAD_SHA" ]; then
    echo "Error: --bad-sha is required when --good-sha is provided"
    exit 1
fi
if [ -n "$BAD_SHA" ] && [ -z "$GOOD_SHA" ]; then
    echo "Error: --good-sha is required when --bad-sha is provided"
    exit 1
fi

TTXLA_ROOT=$(git rev-parse --show-toplevel)
BISECT_RUN_DIR="$TTXLA_ROOT/.bisect-run"
GITHUB_REPO="tenstorrent/tt-xla"
mkdir -p "$BISECT_RUN_DIR"

# ---------------------------------------------------------------------------
# restart_chip: reset the TT device before every test run
# ---------------------------------------------------------------------------
restart_chip() {
    local log_file="$1"
    echo "=== Restarting chip ===" | tee -a "$log_file"
    tt-smi -r 2>&1 | tee -a "$log_file" || {
        echo "tt-smi failed — reinstalling" | tee -a "$log_file"
        deactivate 2>/dev/null || true
        pip uninstall tt-smi -y 2>/dev/null || true
        source "$TTXLA_ROOT/venv/activate"
        pip install tt-smi 2>&1 | tee -a "$log_file"
        tt-smi -r 2>&1 | tee -a "$log_file"
    }
    sleep 3
}

# ---------------------------------------------------------------------------
# install_wheel: download and install the CI wheel for a given full commit SHA
# Returns 125 if no wheel is available (skippable commit)
# ---------------------------------------------------------------------------
install_wheel() {
    local full_commit="$1"
    local short_commit
    short_commit=$(git rev-parse --short "$full_commit")
    local log_file="$2"

    echo "=== Finding CI wheel for $short_commit ===" | tee -a "$log_file"
    local run_id
    run_id=$(gh api "repos/$GITHUB_REPO/actions/runs?head_sha=$full_commit&event=push&per_page=1000" \
        --jq '.workflow_runs[] | select(.name == "On push") | .id' 2>/dev/null | head -1)

    if [ -z "$run_id" ]; then
        echo "No 'On push' CI run found for $short_commit — untestable" | tee -a "$log_file"
        return 125
    fi

    echo "CI run ID: $run_id" | tee -a "$log_file"
    local wheel_dir="/tmp/bisect_wheels/$short_commit"
    rm -rf "$wheel_dir"
    mkdir -p "$wheel_dir"

    # Discover the exact artifact name from the run (the SHA suffix length used by CI
    # at build time may differ from what git rev-parse --short returns locally today).
    local artifact_name
    artifact_name=$(gh api "repos/$GITHUB_REPO/actions/runs/$run_id/artifacts?per_page=100" \
        --jq '.artifacts[] | select(.name | test("^xla-whl-release-")) | .name' 2>/dev/null | head -1)

    if [ -z "$artifact_name" ]; then
        echo "No xla-whl-release artifact found in run $run_id — untestable" | tee -a "$log_file"
        return 125
    fi

    echo "Artifact name: $artifact_name" | tee -a "$log_file"
    gh run download "$run_id" \
        --repo "$GITHUB_REPO" \
        --dir "$wheel_dir" \
        --name "$artifact_name" 2>&1 | tee -a "$log_file"

    local wheel
    wheel=$(find "$wheel_dir" -name "*.whl" | head -1)
    if [ -z "$wheel" ]; then
        echo "Wheel not found — artifact may have expired, marking as untestable" | tee -a "$log_file"
        return 125
    fi

    echo "Installing wheel: $wheel" | tee -a "$log_file"
    pip install "$wheel" --quiet 2>&1 | tee -a "$log_file"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Wheel install failed, marking as untestable" | tee -a "$log_file"
        return 125
    fi
    echo "Wheel installed successfully" | tee -a "$log_file"
    return 0
}

# ---------------------------------------------------------------------------
# run_test_at_current_commit: restart chip, install wheel, run pytest at HEAD
# Returns the pytest exit code, or 125 if the commit is untestable
# ---------------------------------------------------------------------------
run_test_at_current_commit() {
    local full_commit short_commit log_file
    full_commit=$(git rev-parse HEAD)
    short_commit=$(git rev-parse --short HEAD)
    log_file="$BISECT_RUN_DIR/bisect_test_${short_commit}.log"

    echo "======================================"
    echo "Testing commit: $short_commit"
    echo "Test: $TEST_ID"
    echo "======================================"

    echo "Testing commit: $short_commit" > "$log_file"
    echo "Test: $TEST_ID" >> "$log_file"
    echo "Date: $(date)" >> "$log_file"
    echo "======================================" >> "$log_file"

    restart_chip "$log_file"

    echo "=== Activating venv ===" | tee -a "$log_file"
    source "$TTXLA_ROOT/venv/activate"

    if ! python -c "import pytest" 2>/dev/null; then
        echo "ERROR: pytest not found in venv." | tee -a "$log_file"
        echo "       Run: pip install -r $TTXLA_ROOT/venv/requirements-dev.txt" | tee -a "$log_file"
        return 125
    fi

    echo "=== Updating submodules ===" | tee -a "$log_file"
    git submodule update --init --recursive >> "$log_file" 2>&1

    install_wheel "$full_commit" "$log_file"
    local install_exit=$?
    if [ $install_exit -ne 0 ]; then
        return 125
    fi

    echo "=== Running test ===" | tee -a "$log_file"
    python -m pytest -x "$TEST_ID" 2>&1 | tee -a "$log_file"
    local test_exit=${PIPESTATUS[0]}

    echo "" | tee -a "$log_file"
    if [ $test_exit -eq 0 ]; then
        echo "GOOD: Test passed" | tee -a "$log_file"
    else
        echo "BAD: Test failed (exit code $test_exit)" | tee -a "$log_file"
    fi

    return $test_exit
}

# ---------------------------------------------------------------------------
# BISECT RUNNER MODE — called by git bisect run (no --good-sha/--bad-sha)
# ---------------------------------------------------------------------------
if [ -z "$GOOD_SHA" ] && [ -z "$BAD_SHA" ]; then
    cd "$TTXLA_ROOT"
    run_test_at_current_commit
    test_exit=$?

    if [ $test_exit -eq 0 ]; then
        exit 0
    elif [ $test_exit -eq 1 ]; then
        exit 1
    else
        exit 125
    fi
fi

# ---------------------------------------------------------------------------
# ORCHESTRATION MODE — verify good, verify bad, then run git bisect
# ---------------------------------------------------------------------------
cd "$TTXLA_ROOT"

GOOD_FULL=$(git rev-parse "$GOOD_SHA")
BAD_FULL=$(git rev-parse "$BAD_SHA")
GOOD_SHORT=${GOOD_FULL:0:8}
BAD_SHORT=${BAD_FULL:0:8}

COMMIT_COUNT=$(git rev-list --count "${GOOD_FULL}..${BAD_FULL}")

echo "======================================"
echo "Test:             $TEST_ID"
echo "Good SHA:         $GOOD_FULL"
echo "Bad SHA:          $BAD_FULL"
echo "Commits in range: $COMMIT_COUNT"
echo "======================================"

# --- Phase 1: Verify good commit ---
echo ""
echo "=== Phase 1: Verifying good commit ($GOOD_SHORT) ==="
git checkout "$GOOD_FULL"
run_test_at_current_commit
GOOD_EXIT=$?

if [ $GOOD_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Test is not passing on good commit ($GOOD_SHORT) — aborting bisect."
    echo "       Exit code: $GOOD_EXIT"
    echo "       Log: $BISECT_RUN_DIR/bisect_test_${GOOD_SHORT}.log"
    git checkout -
    exit 1
fi
echo "Good commit verified: test passes on $GOOD_SHORT"

# --- Phase 2: Capture reference exit code from bad commit ---
echo ""
echo "=== Phase 2: Running on bad commit ($BAD_SHORT) ==="
git checkout "$BAD_FULL"
run_test_at_current_commit
BAD_EXIT=$?
echo "Bad commit exit code: $BAD_EXIT (reference for bisect)"

# --- Phase 3: Run git bisect ---
echo ""
echo "=== Phase 3: Starting git bisect ==="
git bisect reset >/dev/null 2>&1 || true
git bisect start
git bisect bad "$BAD_FULL"
git bisect good "$GOOD_FULL"
git bisect run "$TTXLA_ROOT/scripts/bisect_test.sh" -t "$TEST_ID"
