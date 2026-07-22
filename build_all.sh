#!/bin/bash
set -eo pipefail
cd /home/mvasiljevic/tt-xla
echo "===== BUILD START $(date) ====="
echo "----- activating venv (installs python deps) -----"
source venv/activate
echo "TTXLA_ENV_ACTIVATED=$TTXLA_ENV_ACTIVATED  TTMLIR_TOOLCHAIN_DIR=$TTMLIR_TOOLCHAIN_DIR"
echo "----- cmake configure -----"
cmake -G Ninja -B build 2>&1
echo "----- cmake build -----"
cmake --build build 2>&1
echo "===== BUILD DONE $(date) rc=$? ====="
