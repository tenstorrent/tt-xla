#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Installs python requirements for third party tt-mlir project.

# Exit immediately if a command exits with a non-zero status
set -e

# Only activate the virtual environment if not already activated
if [ -z "${TTXLA_ENV_ACTIVATED}" ] || [ "${TTXLA_ENV_ACTIVATED}" != "1" ]; then
  source ${TTPJRT_SOURCE_DIR}/venv/bin/activate
fi

TT_MLIR_ENV_DIR=${TTPJRT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/env

# Install llvm requirements
LLVM_VERSION=$(grep -oP 'set\(LLVM_PROJECT_VERSION "\K[^"]+' ${TT_MLIR_ENV_DIR}/CMakeLists.txt)
REQUIREMENTS_CACHE_DIR=${TTPJRT_SOURCE_DIR}/venv/bin/requirements_cache
LLVM_REQUIREMENTS_PATH=${REQUIREMENTS_CACHE_DIR}/llvm-requirements-${LLVM_VERSION}.txt
if [ ! -e $LLVM_REQUIREMENTS_PATH ]; then
  mkdir -p $REQUIREMENTS_CACHE_DIR
  wget -O $LLVM_REQUIREMENTS_PATH "https://github.com/llvm/llvm-project/raw/$LLVM_VERSION/mlir/python/requirements.txt" --quiet
fi
command -v uv &> /dev/null && PIP="uv pip" || PIP="pip" # use uv pip if uv is available
$PIP install -r $LLVM_REQUIREMENTS_PATH

# Install tt-mlir requirements
$PIP install -r ${TT_MLIR_ENV_DIR}/build-requirements.txt

# Sync nanobind with tt-metal's pinned version to avoid header/ODR mismatches.
# tt-metal pins its nanobind version via CPM; we read that version from the
# CPM source cache (populated once tt-metal has been downloaded) and force-
# reinstall the matching PyPI package so that the compiler sees consistent headers.
TT_METAL_CPM_CACHE="${TTPJRT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/.cpmcache"
NANOBIND_HEADER=$(find "${TT_METAL_CPM_CACHE}/nanobind" -maxdepth 3 -name "nanobind.h" -path "*/include/nanobind/nanobind.h" 2>/dev/null | head -1)
if [ -n "${NANOBIND_HEADER}" ]; then
    NB_MAJOR=$(grep "#define NB_VERSION_MAJOR" "${NANOBIND_HEADER}" | awk '{print $3}')
    NB_MINOR=$(grep "#define NB_VERSION_MINOR" "${NANOBIND_HEADER}" | awk '{print $3}')
    NB_PATCH=$(grep "#define NB_VERSION_PATCH" "${NANOBIND_HEADER}" | awk '{print $3}')
    NB_VERSION="${NB_MAJOR}.${NB_MINOR}.${NB_PATCH}"
    echo "Syncing nanobind to tt-metal's pinned version: ${NB_VERSION}"
    $PIP install --force-reinstall "nanobind==${NB_VERSION}"
else
    echo "tt-metal CPM cache not yet populated; skipping nanobind version sync"
fi
