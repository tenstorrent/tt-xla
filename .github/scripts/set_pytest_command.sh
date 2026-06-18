#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the base pytest command + verbosity for a test job and export them to
# GITHUB_ENV (BASE_PYTEST_CMD, VERBOSITY). Used by call-test.yml for both model and
# perf jobs.
#
# Inputs (environment), set by the calling workflow step:
#   FORKED            - matrix.build.forked
#   FORGE_MODELS      - matrix.build.forge-models
#   RUNS_ON_ORIGINAL  - matrix.build.runs-on-original (may be empty)
#   RUNS_ON           - matrix.build.runs-on
#   TEST_MARK         - matrix.build.test-mark (may be empty)
#   CONTAINS          - matrix.build.contains (may be empty)
#   RERUN_ATTEMPT     - steps.check-rerun-attempt.outputs.rerun_attempt
#   PARALLEL_GROUPS   - matrix.build.parallel-groups (may be empty)
#   GROUP_ID          - matrix.build.group-id (may be empty)
#   DIR               - matrix.build.dir
#   EXTRA_ARGS        - matrix.build.args (may be empty)
#   DUMP_IRS          - inputs.dump_irs

# NOTE: Torch tests must be run in separate processes to avoid current issues with
# test isolation. See https://github.com/tenstorrent/tt-xla/issues/795
PYTEST_FORKED=""
if [[ "$FORKED" == "true" ]]; then
  PYTEST_FORKED="--forked"
fi

# Set verbosity: default to -sv, switch to -vv when forking to get progress percentage
if [[ -n "$PYTEST_FORKED" ]]; then
  VERBOSITY="-vv"
else
  VERBOSITY="-sv"
fi

# Pass arch to forge models tests to resolve arch_overrides in test_config files.
ARCH=""
if [[ "$FORGE_MODELS" == "true" ]]; then
  ARCH="--arch ${RUNS_ON_ORIGINAL:-$RUNS_ON}"
fi

MARKS_ARG=""
if [[ -n "$TEST_MARK" ]]; then
  MARKS_ARG="-m '$TEST_MARK'"
fi

CONTAINS_ARG=""
if [[ -n "$CONTAINS" ]]; then
  CONTAINS_ARG="-k '$CONTAINS'"
fi

if [ "$RERUN_ATTEMPT" == "true" ]; then
  echo "Only running previously failed tests"
  PYTEST_SPLITS=""
else
  echo "Running all tests"
  PYTEST_SPLITS="--splits ${PARALLEL_GROUPS:-1} \
    --group ${GROUP_ID:-1} \
    --splitting-algorithm least_duration"
fi

# Add -m/-k only when provided to avoid empty-argument parsing issues.
BASE_PYTEST_CMD="$PYTEST_FORKED --log-memory \
  --durations=0 \
  ${DIR} \
  $ARCH \
  $MARKS_ARG ${EXTRA_ARGS} \
  $CONTAINS_ARG \
  $PYTEST_SPLITS"

if [[ "$DUMP_IRS" == "true" ]]; then
  BASE_PYTEST_CMD="$BASE_PYTEST_CMD --dump-irs"
fi

echo "BASE_PYTEST_CMD=$BASE_PYTEST_CMD" >> "$GITHUB_ENV"
echo "VERBOSITY=$VERBOSITY" >> "$GITHUB_ENV"
