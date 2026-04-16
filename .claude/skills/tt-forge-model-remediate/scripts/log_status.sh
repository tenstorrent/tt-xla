#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_name> <reason>" >&2
    exit 1
fi

RESULTS_YAML=$PWD/results.yaml

TEST_NAME="$1"
STATUS="$2"
DATE=$(date +%Y%m%d_%H%M%S)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Initialize file with empty list if it doesn't exist
if [ ! -s "${RESULTS_YAML}" ]; then
  echo "results:" > "${RESULTS_YAML}"
fi
cat >> "${RESULTS_YAML}" <<EOF
- test_name: "${TEST_NAME}"
  status: "${STATUS}"
  date: "${DATE}"
  branch: "${BRANCH}"
EOF
