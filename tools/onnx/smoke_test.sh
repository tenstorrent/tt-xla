#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Smoke test: Add ONNX -> ONNX dialect MLIR -> StableHLO MLIR.
#
# Prerequisites:
#   tools/onnx/build_onnx_mlir.sh
#   pip install onnx numpy   (for fixture generation)
#
# Usage:
#   source venv/activate
#   source tools/onnx/env.sh    # created by build script
#   tools/onnx/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FIXTURES_DIR="${SCRIPT_DIR}/fixtures"
ARTIFACTS_DIR="${SCRIPT_DIR}/build/smoke"
ADD_ONNX="${FIXTURES_DIR}/add.onnx"
ONNX_MLIR="${TT_ONNX_MLIR:-${SCRIPT_DIR}/build/install/bin/onnx-mlir}"
ONNX_MLIR_OPT="${TT_ONNX_MLIR_OPT:-${SCRIPT_DIR}/build/install/bin/onnx-mlir-opt}"

if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.sh"
  ONNX_MLIR="${TT_ONNX_MLIR:-${ONNX_MLIR}}"
  ONNX_MLIR_OPT="${TT_ONNX_MLIR_OPT:-${ONNX_MLIR_OPT}}"
fi

if [[ ! -x "${ONNX_MLIR}" || ! -x "${ONNX_MLIR_OPT}" ]]; then
  echo "ERROR: onnx-mlir tools not found." >&2
  echo "Run: tools/onnx/build_onnx_mlir.sh" >&2
  exit 1
fi

mkdir -p "${FIXTURES_DIR}" "${ARTIFACTS_DIR}"

if [[ ! -f "${ADD_ONNX}" ]]; then
  echo "==> Generating ${ADD_ONNX}..."
  python3 "${SCRIPT_DIR}/gen_add_onnx.py" -o "${ADD_ONNX}"
fi

ONNX_IR_BASE="${ARTIFACTS_DIR}/add"
ONNX_IR="${ONNX_IR_BASE}.onnx.mlir"
SHLO_IR="${ARTIFACTS_DIR}/add.stablehlo.mlir"

echo "==> Import ONNX to ONNX dialect MLIR..."
# -o must be a basename without extension; onnx-mlir appends ".onnx.mlir".
"${ONNX_MLIR}" --EmitONNXIR "${ADD_ONNX}" -o "${ONNX_IR_BASE}"

if [[ ! -f "${ONNX_IR}" ]]; then
  echo "ERROR: Expected ONNX dialect IR at ${ONNX_IR}" >&2
  echo "Files in ${ARTIFACTS_DIR}:" >&2
  ls -la "${ARTIFACTS_DIR}" >&2 || true
  exit 1
fi

echo "==> Lower ONNX dialect to StableHLO..."
"${ONNX_MLIR_OPT}" "${ONNX_IR}" \
  --convert-onnx-to-stablehlo \
  -o "${SHLO_IR}"

if ! grep -q "stablehlo.add" "${SHLO_IR}"; then
  echo "ERROR: Expected stablehlo.add in ${SHLO_IR}" >&2
  exit 1
fi

echo ""
echo "Smoke test PASSED."
echo "  ONNX IR:      ${ONNX_IR}"
echo "  StableHLO IR: ${SHLO_IR}"
echo ""
head -30 "${SHLO_IR}"
