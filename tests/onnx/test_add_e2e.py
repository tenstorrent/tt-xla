# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ONNX tests (Workstream 3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from tt_onnx import ONNXSession  # noqa: E402
from tt_onnx.runtime import max_abs_diff, run_onnxruntime_cpu  # noqa: E402


def _ensure_add_onnx() -> Path:
    fixture = REPO_ROOT / "tools" / "onnx" / "fixtures" / "add.onnx"
    if not fixture.is_file():
        gen = REPO_ROOT / "tools" / "onnx" / "gen_add_onnx.py"
        subprocess.run(
            [sys.executable, str(gen), "-o", str(fixture)],
            check=True,
        )
    return fixture


@pytest.mark.push
def test_add_onnx_e2e():
    """Add ONNX → onnx-mlir → StableHLO → PJRT → TT device vs ORT CPU."""
    import jax

    if not jax.devices("tt"):
        pytest.skip("No TT device available")

    onnx_path = _ensure_add_onnx()
    work_dir = REPO_ROOT / "tools" / "onnx" / "build" / "tt_onnx" / "test_add_e2e"

    a = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    b = np.array([[3.5, 2.5, 1.5, 0.5]], dtype=np.float32)
    feed = {"A": a, "B": b}

    session = ONNXSession(
        onnx_path,
        work_dir=work_dir,
        compile_options={"optimization_level": "0"},
    )

    assert session.bridge_artifacts.stablehlo_mlir.is_file()
    stablehlo = session.bridge_artifacts.stablehlo_text
    assert "stablehlo.add" in stablehlo

    tt_outputs = session.run(feed)
    ref_outputs = run_onnxruntime_cpu(str(onnx_path), feed)

    for name in session.output_names:
        assert name in ref_outputs
        assert max_abs_diff(tt_outputs[name], ref_outputs[name]) < 1e-3
