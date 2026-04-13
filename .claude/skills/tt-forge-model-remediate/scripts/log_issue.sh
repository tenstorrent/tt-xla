#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_name> <reason>" >&2
    exit 1
fi

LOGS_DIR=$PWD/logs
mkdir -p $LOGS_DIR

TEST_NAME="$0"
SAFE_NAME=$(echo "${TEST_NAME}" | tr '/:[]\.' '_')
DATE=$(date +%Y%m%d_%H%M%S)
echo "$1" > $LOGS_DIR/${SAFE_NAME}_${DATE}.log
