#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

export XDG_CACHE_HOME="$PWD/.cache"
export TT_METAL_CACHE="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export TTMLIR_VENV_DIR=$PWD/.local_venv

rm -rf $XDG_CACHE_HOME
rm -rf $TT_METAL_CACHE
rm -rf $HF_HOME
rm -rf $TTMLIR_VENV_DIR

cd $TT_XLA_ROOT
source venv/activate

set -x

pip install -e python_package --no-deps --no-build-isolation
