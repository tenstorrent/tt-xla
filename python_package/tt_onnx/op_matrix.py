# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WS4 op matrix runner (onnx-mlir bridge only)."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnx

from . import ONNXSession
from .feed_utils import prepare_feed
from .runtime import max_abs_diff, run_onnxruntime_cpu

SEED_OPS = ("add", "mul", "matmul", "relu", "reshape")
EXTENDED_OPS = (
    "sub",
    "div",
    "sigmoid",
    "reduce_mean",
    "reduce_sum",
    "transpose",
    "concat",
    "slice",
    "conv",
    "layer_norm",
    "softmax",
    "gather",
)
FULL_OPS = SEED_OPS + EXTENDED_OPS
DEFAULT_TOLERANCE = 1e-3
# Per-op tolerances for TT vs ORT CPU on fp32 (tighter default elsewhere).
OP_TOLERANCES: dict[str, float] = {
    "matmul": 2e-3,
    "conv": 2e-3,
    "layer_norm": 2e-3,
    "softmax": 2e-3,
}


def op_tolerance(op: str, default: float = DEFAULT_TOLERANCE) -> float:
    return OP_TOLERANCES.get(op, default)


def seed_op_tolerance(op: str, default: float = DEFAULT_TOLERANCE) -> float:
    """Backward-compatible alias for :func:`op_tolerance`."""
    return op_tolerance(op, default)


class OpMatrixStatus(str, Enum):
    PASS = "pass"
    COMPILE_FAIL = "compile_fail"
    EXECUTE_FAIL = "execute_fail"
    NUMERICAL_FAIL = "numerical_fail"


@dataclass
class OpMatrixResult:
    op: str
    onnx_path: str
    status: OpMatrixStatus
    max_abs_diff: float | None = None
    compile_time_s: float | None = None
    output_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class OpMatrixReport:
    results: list[OpMatrixResult] = field(default_factory=list)
    tolerance: float = 1e-3
    matrix: str = "seed"
    ops: tuple[str, ...] = SEED_OPS

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.status == OpMatrixStatus.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status != OpMatrixStatus.PASS)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.pass_count / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix,
            "ops": list(self.ops),
            "tolerance": self.tolerance,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "pass_rate": self.pass_rate,
            "results": [r.to_dict() for r in self.results],
        }

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def run_op_case(
    op: str,
    onnx_path: str | Path,
    feed: Mapping[str, np.ndarray | list],
    *,
    work_dir: str | Path,
    tolerance: float = 1e-3,
    compile_options: dict[str, str] | None = None,
) -> OpMatrixResult:
    onnx_path = Path(onnx_path)
    model = onnx.load(onnx_path)
    feed_np = prepare_feed(model, feed)

    try:
        session = ONNXSession(
            onnx_path,
            work_dir=work_dir,
            compile_options=compile_options or {"optimization_level": "0"},
        )
    except Exception as exc:
        return OpMatrixResult(
            op=op,
            onnx_path=str(onnx_path),
            status=OpMatrixStatus.COMPILE_FAIL,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )

    try:
        tt_outputs = session.run(feed_np)
        ref_outputs = run_onnxruntime_cpu(str(onnx_path), feed_np)
    except Exception as exc:
        return OpMatrixResult(
            op=op,
            onnx_path=str(onnx_path),
            status=OpMatrixStatus.EXECUTE_FAIL,
            compile_time_s=session.compile_artifacts.compile_time_s,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )

    worst_name = None
    worst_diff = 0.0
    for name in session.output_names:
        diff = max_abs_diff(tt_outputs[name], ref_outputs[name])
        if diff > worst_diff:
            worst_diff = diff
            worst_name = name

    if worst_diff >= tolerance:
        return OpMatrixResult(
            op=op,
            onnx_path=str(onnx_path),
            status=OpMatrixStatus.NUMERICAL_FAIL,
            max_abs_diff=worst_diff,
            compile_time_s=session.compile_artifacts.compile_time_s,
            output_name=worst_name,
            error=f"{worst_name}: max_abs_diff={worst_diff} >= {tolerance}",
        )

    return OpMatrixResult(
        op=op,
        onnx_path=str(onnx_path),
        status=OpMatrixStatus.PASS,
        max_abs_diff=worst_diff,
        compile_time_s=session.compile_artifacts.compile_time_s,
        output_name=worst_name,
    )


def run_op_matrix(
    cases: Sequence[tuple[str, Path, Mapping[str, np.ndarray | list]]],
    *,
    work_root: str | Path,
    tolerance: float = 1e-3,
    compile_options: dict[str, str] | None = None,
    matrix: str = "seed",
    ops: Sequence[str] | None = None,
) -> OpMatrixReport:
    work_root = Path(work_root)
    op_list = tuple(ops) if ops is not None else tuple(op for op, _, _ in cases)
    report = OpMatrixReport(tolerance=tolerance, matrix=matrix, ops=op_list)
    for op, onnx_path, feed in cases:
        result = run_op_case(
            op,
            onnx_path,
            feed,
            work_dir=work_root / op,
            tolerance=op_tolerance(op, tolerance),
            compile_options=compile_options,
        )
        report.results.append(result)
    return report
