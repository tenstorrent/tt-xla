#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Installs python requirements for third party tt-mlir project.

# Exit immediately if a command exits with a non-zero status
set -e

source ${TTPJRT_SOURCE_DIR}/venv/bin/activate

TT_MLIR_ENV_DIR=${TTPJRT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/env

# Install llvm requirements
LLVM_VERSION=$(grep -oP 'set\(LLVM_PROJECT_VERSION "\K[^"]+' ${TT_MLIR_ENV_DIR}/CMakeLists.txt)
REQUIREMENTS_CACHE_DIR=${TTPJRT_SOURCE_DIR}/venv/bin/requirements_cache
LLVM_REQUIREMENTS_PATH=${REQUIREMENTS_CACHE_DIR}/llvm-requirements-${LLVM_VERSION}.txt
if [ ! -e $LLVM_REQUIREMENTS_PATH ]; then
  mkdir -p $REQUIREMENTS_CACHE_DIR
  wget -O $LLVM_REQUIREMENTS_PATH "https://github.com/llvm/llvm-project/raw/$LLVM_VERSION/mlir/python/requirements.txt" --quiet
fi
pip install -r $LLVM_REQUIREMENTS_PATH

# Install tt-mlir requirements
pip install -r ${TT_MLIR_ENV_DIR}/build-requirements.txt
