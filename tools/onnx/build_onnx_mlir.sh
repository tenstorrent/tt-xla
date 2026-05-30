#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build onnx-mlir + onnx-mlir-opt using onnx-mlir's bundled LLVM/MLIR pin.
#
# NOTE: Building against TTMLIR_TOOLCHAIN_DIR MLIR fails — tt-mlir's MLIR is
# newer than onnx-mlir@4400cbc expects (Krnl tblgen + stablehlo API mismatch).
# We build the LLVM revision from onnx-mlir's utils/clone-mlir.sh instead.
# The emitted StableHLO is still consumed by tt-xla via WS1 direct SHLO ingestion.
#
# Usage (from tt-xla repo root):
#   source venv/activate
#   tools/onnx/build_onnx_mlir.sh
#
# Optional env overrides:
#   ONNX_MLIR_COMMIT=<git-sha>
#   ONNX_MLIR_BUILD_JOBS=16
#   ONNX_MLIR_FORCE_LLVM_REBUILD=1
#   ONNX_MLIR_USE_TTMLIR_MLIR=1   # experimental; usually fails (see note above)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/build"
SRC_DIR="${BUILD_ROOT}/onnx-mlir-src"
LLVM_SRC_DIR="${SRC_DIR}/llvm-project"
LLVM_BUILD_DIR="${LLVM_SRC_DIR}/build"
CMAKE_BUILD_DIR="${BUILD_ROOT}/onnx-mlir-build"
INSTALL_DIR="${BUILD_ROOT}/install"
COMMIT_FILE="${SCRIPT_DIR}/onnx_mlir_commit.txt"

if [[ -f "${REPO_ROOT}/venv/activate" && -z "${TTXLA_ENV_ACTIVATED:-}" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/venv/activate"
fi

DEFAULT_COMMIT="$(grep -v '^#' "${COMMIT_FILE}" | grep -v '^[[:space:]]*$' | head -1 | tr -d '[:space:]')"
: "${ONNX_MLIR_COMMIT:=${DEFAULT_COMMIT}}"

if [[ -z "${ONNX_MLIR_COMMIT}" ]]; then
  echo "ERROR: Could not read pinned commit from ${COMMIT_FILE}" >&2
  exit 1
fi

JOBS="${ONNX_MLIR_BUILD_JOBS:-$(nproc)}"

if command -v clang++-20 >/dev/null 2>&1; then
  CXX_COMPILER="clang++-20"
  CC_COMPILER="clang-20"
elif command -v clang++ >/dev/null 2>&1; then
  CXX_COMPILER="clang++"
  CC_COMPILER="clang"
else
  CXX_COMPILER="c++"
  CC_COMPILER="cc"
fi

echo "==> tt-xla repo:        ${REPO_ROOT}"
echo "==> onnx-mlir commit:   ${ONNX_MLIR_COMMIT}"
echo "==> C++ compiler:       ${CXX_COMPILER}"
echo "==> jobs:               ${JOBS}"
echo "==> install dir:        ${INSTALL_DIR}"

mkdir -p "${BUILD_ROOT}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  echo "==> Cloning onnx-mlir..."
  git clone --filter=blob:none https://github.com/onnx/onnx-mlir.git "${SRC_DIR}"
fi

echo "==> Checking out onnx-mlir ${ONNX_MLIR_COMMIT}..."
git -C "${SRC_DIR}" fetch origin
git -C "${SRC_DIR}" checkout -q "${ONNX_MLIR_COMMIT}"

echo "==> Initializing onnx-mlir submodules..."
git -C "${SRC_DIR}" submodule update --init --depth 1 \
  third_party/onnx \
  third_party/pybind11 \
  third_party/rapidcheck \
  third_party/stablehlo \
  third_party/benchmark

resolve_mlir_dir() {
  if [[ "${ONNX_MLIR_USE_TTMLIR_MLIR:-0}" == "1" ]]; then
    : "${TTMLIR_TOOLCHAIN_DIR:=/opt/ttmlir-toolchain}"
    MLIR_DIR="${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir"
    LLVM_DIR="${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm"
    if [[ ! -d "${MLIR_DIR}" ]]; then
      echo "ERROR: TTMLIR MLIR not found at ${MLIR_DIR}" >&2
      exit 1
    fi
    echo "==> Using TTMLIR toolchain MLIR (experimental): ${MLIR_DIR}"
    return
  fi

  if [[ ! -d "${LLVM_SRC_DIR}/.git" ]]; then
    echo "==> Cloning bundled llvm-project (pin from utils/clone-mlir.sh)..."
    (cd "${SRC_DIR}" && bash utils/clone-mlir.sh)
  fi

  if [[ "${ONNX_MLIR_FORCE_LLVM_REBUILD:-0}" == "1" ]]; then
    rm -rf "${LLVM_BUILD_DIR}"
  fi

  if [[ ! -x "${LLVM_BUILD_DIR}/bin/mlir-tblgen" ]]; then
    echo "==> Building bundled LLVM/MLIR (first time; can take 30–90+ min)..."
    mkdir -p "${LLVM_BUILD_DIR}"
    cmake -G Ninja -S "${LLVM_SRC_DIR}/llvm" -B "${LLVM_BUILD_DIR}" \
      -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
      -DCMAKE_C_COMPILER="${CC_COMPILER}" \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_RTTI=ON \
      -DLLVM_ENABLE_LIBEDIT=OFF
    cmake --build "${LLVM_BUILD_DIR}" -j "${JOBS}"
  else
    echo "==> Reusing existing bundled LLVM build at ${LLVM_BUILD_DIR}"
  fi

  MLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
  LLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"
  echo "==> Using bundled MLIR_DIR: ${MLIR_DIR}"
}

resolve_mlir_dir

rm -rf "${CMAKE_BUILD_DIR}"
mkdir -p "${CMAKE_BUILD_DIR}"
cd "${CMAKE_BUILD_DIR}"

echo "==> Configuring onnx-mlir..."
cmake -G Ninja \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCMAKE_C_COMPILER="${CC_COMPILER}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DONNX_MLIR_ENABLE_JAVA=OFF \
  -DONNX_MLIR_BUILD_TESTS=OFF \
  "${SRC_DIR}"

echo "==> Building onnx-mlir-opt and onnx-mlir (${JOBS} jobs)..."
cmake --build . --target onnx-mlir-opt onnx-mlir -j "${JOBS}"

# Install only the frontend tools we need. Full `cmake --install` fails when
# ONNX_MLIR_BUILD_TESTS=OFF because rapidcheck/benchmark libs are not built.
ONNX_MLIR_BIN_DIR="${CMAKE_BUILD_DIR}/Release/bin"
if [[ ! -x "${ONNX_MLIR_BIN_DIR}/onnx-mlir-opt" ]]; then
  ONNX_MLIR_BIN_DIR="${CMAKE_BUILD_DIR}/bin"
fi
if [[ ! -x "${ONNX_MLIR_BIN_DIR}/onnx-mlir-opt" ]]; then
  echo "ERROR: onnx-mlir-opt not found under ${CMAKE_BUILD_DIR}" >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}/bin"
install -m 755 "${ONNX_MLIR_BIN_DIR}/onnx-mlir-opt" "${INSTALL_DIR}/bin/"
install -m 755 "${ONNX_MLIR_BIN_DIR}/onnx-mlir" "${INSTALL_DIR}/bin/"
echo "==> Installed tools to ${INSTALL_DIR}/bin"

ENV_SCRIPT="${SCRIPT_DIR}/env.sh"
cat > "${ENV_SCRIPT}" <<EOF
# Source this file after building onnx-mlir (tools/onnx/build_onnx_mlir.sh).
export TT_ONNX_MLIR_ROOT="${INSTALL_DIR}"
export TT_ONNX_MLIR_OPT="${INSTALL_DIR}/bin/onnx-mlir-opt"
export TT_ONNX_MLIR="${INSTALL_DIR}/bin/onnx-mlir"
export PATH="${INSTALL_DIR}/bin:\${PATH}"
EOF

echo ""
echo "Build complete."
echo "  source ${ENV_SCRIPT}"
echo "  tools/onnx/smoke_test.sh"
