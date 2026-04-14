#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

if [ ! -f $TT_XLA_ROOT/.env ]; then
  echo "Env to set HF_TOKEN or TT_COMPILE_ONLY_SYSTEM_DESC doesn't exist"
  exit 1
fi

export XDG_CACHE_HOME="$PWD/.cache"
export TT_METAL_CACHE="$PWD/.cache"
export TTMLIR_VENV_DIR=$PWD/.local_venv
LOGS_DIR=$PWD/logs
mkdir -p $LOGS_DIR

source $TT_XLA_ROOT/.env

cd $TT_XLA_ROOT
source venv/activate

if [[ -n "$TT_COMPILE_ONLY_SYSTEM_DESC" ]]; then
  export TT_RANDOM_WEIGHTS=1
fi
export TTXLA_LOGGER_LEVEL=DEBUG

TEST_NAME="$@"
SAFE_NAME=$(echo "${TEST_NAME}" | tr '/:[]\.' '_')
DATE=$(date +%Y%m%d_%H%M%S)
set -x
pytest "${TEST_NAME}" -v
