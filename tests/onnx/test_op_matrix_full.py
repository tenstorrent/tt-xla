# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WS4 M1.8: full basic op matrix (onnx-mlir only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python_package") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python_package"))

if str(REPO_ROOT / "tests" / "onnx") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "onnx"))

from onnx_e2e_utils import require_tt_device  # noqa: E402
from tt_onnx.op_matrix import FULL_OPS, OpMatrixStatus, op_tolerance, run_op_case  # noqa: E402


def _ensure_full_fixtures(fixture_dir: Path) -> None:
    missing = [op for op in FULL_OPS if not (fixture_dir / f"{op}.onnx").is_file()]
    if not missing:
        return
    gen = REPO_ROOT / "tools" / "onnx" / "gen_op_onnx.py"
    subprocess.run([sys.executable, str(gen), "--all", "--full"], check=True)


def _default_feed(op: str) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_op_onnx", REPO_ROOT / "tools" / "onnx" / "gen_op_onnx.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.default_feed(op)


@pytest.mark.push
@pytest.mark.parametrize("op", FULL_OPS)
def test_op_matrix_full(op: str):
    """M1.8 single-op ONNX lower → compile → execute vs ORT CPU."""
    require_tt_device()

    fixture_dir = REPO_ROOT / "tools" / "onnx" / "fixtures"
    _ensure_full_fixtures(fixture_dir)
    onnx_path = fixture_dir / f"{op}.onnx"
    work_dir = REPO_ROOT / "tools" / "onnx" / "build" / "tt_onnx" / "test_op_matrix_full" / op

    result = run_op_case(
        op,
        onnx_path,
        _default_feed(op),
        work_dir=work_dir,
        tolerance=op_tolerance(op),
        compile_options={"optimization_level": "0"},
    )
    assert result.status == OpMatrixStatus.PASS, (
        f"{op}: {result.status.value} — {result.error or ''}"
    )
