#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_name> <reason>" >&2
    exit 1
fi

LOGS_DIR=$PWD/logs
RESULTS_YAML=$TT_XLA_ROOT/results.yaml
LOCK_FILE=/tmp/.results.yaml.lock
mkdir -p $LOGS_DIR

TEST_NAME="$1"
STATUS="$2"
SAFE_NAME=$(echo "${TEST_NAME}" | tr '/:[]\.' '_')
DATE=$(date +%Y%m%d_%H%M%S)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

(
  flock -x 200
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
) 200>"${LOCK_FILE}"
