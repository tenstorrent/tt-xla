# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WS4: 2-layer MLP ONNX e2e (onnx-mlir only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))
if str(REPO_ROOT / "tests" / "onnx") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "onnx"))

from onnx_e2e_utils import (  # noqa: E402
    ensure_onnx_fixture,
    require_tt_device,
    run_onnx_e2e,
)


@pytest.mark.push
def test_mlp_onnx_e2e():
    """MLP ONNX → onnx-mlir → StableHLO → PJRT → TT device vs ORT CPU."""
    require_tt_device()

    onnx_path = ensure_onnx_fixture(
        REPO_ROOT / "tools" / "onnx" / "gen_mlp_onnx.py",
        REPO_ROOT / "tools" / "onnx" / "fixtures" / "mlp2.onnx",
    )
    work_dir = REPO_ROOT / "tools" / "onnx" / "build" / "tt_onnx" / "test_mlp_e2e"
    feed = {"X": np.array([[0.5, -1.0, 2.0, 0.25]], dtype=np.float32)}

    run_onnx_e2e(onnx_path, feed, work_dir=work_dir)
