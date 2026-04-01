#!/usr/bin/env bash
# Run model tests against n150 and extract failure context on error.
#
# Usage:
#   run-model-tests.sh [--junitxml PATH]
#
# Optional --junitxml flag is forwarded to pytest for CI reporting.
# When tests fail, the extraction script writes failure_context.txt
# and the script exits non-zero.

set -uo pipefail

EXTRA_ARGS=("$@")

rm -f pytest.log failure_context.txt report_*.xml

source venv/activate
python -m pytest -vv --forked --log-memory \
  --durations=0 \
  ./tests/runner/test_models.py::test_all_models_torch \
  --arch n150 \
  -m "n150 and expected_passing and push" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee pytest.log

TEST_EXIT=${PIPESTATUS[0]}

if [ "$TEST_EXIT" -ne 0 ]; then
  echo "Tests failed (exit $TEST_EXIT). Extracting failure context..."
  python .github/scripts/extract_transformers_failure_context.py . . failure_context.txt
fi

exit "$TEST_EXIT"
