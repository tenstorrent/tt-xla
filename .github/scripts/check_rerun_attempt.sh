#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Detect a workflow re-run and, if so, compute which previously-failed tests to rerun.
# Writes `.pytest_tests_to_run` (consumed at pytest collection time by
# tests/conftest.py) and sets the step output `rerun_attempt=true` when there are
# specific tests to rerun. Used by call-test.yml for both model and perf jobs.
#
# Inputs (environment), set by the calling workflow step:
#   RUN_ATTEMPT  - github.run_attempt
#   RUN_ID       - github.run_id
#   REPO         - github.repository
#   JOB_INDEX    - strategy.job-index
#   MATRIX_HASH  - test matrix hash (test-reports artifact name component)
# Requires the venv (for find_all_failed_tests.py) and the gh CLI, run from work-dir.

source venv/activate
rm -f .pytest_tests_to_run
if [ "$RUN_ATTEMPT" -gt 1 ]; then
  echo "Rerun attempt detected"
  attempt=$((RUN_ATTEMPT - 1))
  rm -rf old-reports
  mkdir old-reports
  set +e
  result=1
  while [ $attempt -ge 1 ]; do
    echo "Downloading test reports from attempt $attempt"
    gh run download "$RUN_ID" \
      --repo "$REPO" \
      --pattern "test-reports-${MATRIX_HASH}-$attempt.${JOB_INDEX}-*" \
      -D old-reports
    result=$?
    if [ $result -eq 0 ]; then
      break
    fi
    attempt=$((attempt - 1))
  done
  if [ $result -ne 0 ]; then
    echo "No previous test reports found"
  else
    echo "Found test reports from attempt $attempt"
    python .github/scripts/find_all_failed_tests.py old-reports
    if [ -f ".pytest_tests_to_run" ]; then
      echo "There are tests to rerun"
      echo "rerun_attempt=true" >> "$GITHUB_OUTPUT"
    else
      echo "All tests will rerun"
    fi
  fi
else
  echo "First attempt"
fi
