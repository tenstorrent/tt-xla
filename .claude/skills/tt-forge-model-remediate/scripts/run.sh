#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

if [ -z "$TT_COMPILE_ONLY_SYSTEM_DESC" ]; then
  echo "ERROR: TT_COMPILE_ONLY_SYSTEM_DESC not set. Abort"
  exit 1
fi

export XDG_CACHE_HOME="$PWD/.cache"
export TT_METAL_CACHE="$PWD/.cache"
export TTMLIR_VENV_DIR=$PWD/.local_venv
LOGS_DIR=$PWD/logs
mkdir -p $LOGS_DIR

cd $TT_XLA_ROOT
source venv/activate

export TT_RANDOM_WEIGHTS=1
export TTXLA_LOGGER_LEVEL=DEBUG

TEST_NAME="$@"
SAFE_NAME=$(echo "${TEST_NAME}" | tr '/:[]\.' '_')
DATE=$(date +%Y%m%d_%H%M%S) 
set -x
pytest "${TEST_NAME}" \
  --junit-xml=$LOGS_DIR/${SAFE_NAME}_${DATE}.xml \
  -v 2>&1 | tee $LOGS_DIR/${SAFE_NAME}_${DATE}.log
