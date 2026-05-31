# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for ONNX e2e tests (Workstream 4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

from tt_onnx import ONNXSession  # noqa: E402
from tt_onnx.runtime import max_abs_diff, run_onnxruntime_cpu  # noqa: E402


def require_tt_device():
    import jax

    if not jax.devices("tt"):
        pytest.skip("No TT device available")


def ensure_onnx_fixture(gen_script: Path, fixture: Path, *extra_args: str) -> Path:
    if not fixture.is_file():
        cmd = [sys.executable, str(gen_script), "-o", str(fixture), *extra_args]
        subprocess.run(cmd, check=True)
    return fixture


def run_onnx_e2e(
    onnx_path: str | Path,
    feed: Mapping[str, np.ndarray],
    *,
    work_dir: str | Path,
    tol: float = 1e-3,
    compile_options: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Compile and run on TT; assert numerical match vs ORT CPU."""
    session = ONNXSession(
        onnx_path,
        work_dir=work_dir,
        compile_options=compile_options or {"optimization_level": "0"},
    )
    tt_outputs = session.run(feed)
    ref_outputs = run_onnxruntime_cpu(str(onnx_path), feed)
    for name in session.output_names:
        assert name in ref_outputs
        diff = max_abs_diff(tt_outputs[name], ref_outputs[name])
        assert diff < tol, f"{name}: max_abs_diff={diff} >= {tol}"
    return tt_outputs
