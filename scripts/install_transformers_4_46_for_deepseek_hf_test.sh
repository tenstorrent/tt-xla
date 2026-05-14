#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Temporarily install transformers 4.46.x so Hub DeepSeek-OCR remote code (trust_remote_code)
# can import (e.g. LlamaFlashAttention2 path). Revert after testing — see REVERT line at end.
#
# Usage (from tt-xla repo root):
#   bash scripts/install_transformers_4_46_for_deepseek_hf_test.sh
#   pytest -q tests/torch/models/deepseek_ocr/test_hf_forward_sanity.py

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: expected venv at ${ROOT}/venv/bin/python" >&2
  exit 1
fi

echo "Current transformers:" && "$PY" -c "import transformers; print(transformers.__version__)" || true
echo "Installing transformers==4.46.3 (matches DeepSeek-OCR config.json transformers_version) ..."
"$PY" -m pip install "transformers==4.46.3"
echo "Now:" && "$PY" -c "import transformers; print(transformers.__version__)"
echo
echo "Run HF sanity:"
echo "  cd \"$ROOT\" && PYTHONPATH=. \"$ROOT/venv/bin/pytest\" -q tests/torch/models/deepseek_ocr/test_hf_forward_sanity.py"
echo
echo "REVERT to tt-xla dev pin (venv/requirements-dev.txt):"
echo "  \"$PY\" -m pip install \"transformers==5.2.0\""
