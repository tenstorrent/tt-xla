#!/bin/bash

if [ -z "$TT_XLA_ROOT" ]; then
  echo "ERROR: TT_XLA_ROOT not set. Abort"
  exit 1
fi

export TTMLIR_TOOLCHAIN_DIR=$PWD/.local_venv

rm -r $TTMLIR_TOOLCHAIN_DIR
