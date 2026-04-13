#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_name> <reason>" >&2
    exit 1
fi

LOGS_DIR=$PWD/logs
RESULTS_YAML=$LOGS_DIR/results.yaml
LOCK_FILE=$LOGS_DIR/.results.yaml.lock
mkdir -p $LOGS_DIR

TEST_NAME="$0"
STATUS="$1"
SAFE_NAME=$(echo "${TEST_NAME}" | tr '/:[]\.' '_')
DATE=$(date +%Y%m%d_%H%M%S)
echo "$1" > $LOGS_DIR/${SAFE_NAME}_${DATE}.log

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
EOF
) 200>"${LOCK_FILE}"
