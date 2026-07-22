#!/bin/bash
set -o pipefail
cd /home/mvasiljevic/tt-xla
source venv/activate >/dev/null 2>&1
echo "===== REBUILD START $(date) ====="
echo "--- rebuild tt-mlir (incremental) ---"
cmake --build third_party/tt-mlir/src/tt-mlir/build 2>&1 | tail -30
echo "rc_ttmlir=$?"
echo "--- rebuild tt-xla plugin (incremental) ---"
touch third_party/tt-mlir/src/tt-mlir-stamp/tt-mlir-update 2>/dev/null; cmake --build build 2>&1 | tail -30
echo "rc_ttxla=$?"
echo "===== REBUILD DONE $(date) ====="
