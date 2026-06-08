# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2A — IR export for composite GroupNorm fix candidate (opt0).

Expect ``tenstorrent.group_norm`` / ``ttnn.group_norm`` in exported MLIR.

Run on TT device:
  pytest -svv tests/torch/model/zimage_decoder_debug/test_export_ir_composite.py \\
    2>&1 | tee zimage_logs/export_ir_composite.log
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import Framework
from infra.workloads.torch_workload import TorchWorkload

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import OpTester

from .ir_analysis import COMPOSITE_MARKERS, format_summary, summarize_export_dir
from .shared import (
    build_composite_norm2_only,
    build_confirm_prefix_through_norm2_composite,
    d3_input_key,
    norm2_isolated_input_key,
)

IR_EXPORT_ROOT = Path(__file__).resolve().parents[4] / "zimage_decoder_ir"


def _build_module_and_inputs(case: str, context: dict):
    dec = context["vae"].decoder
    stages = context["stages"]
    if case == "composite_norm2_alone":
        return build_composite_norm2_only(dec).eval(), [
            stages[norm2_isolated_input_key()]
        ]
    if case == "composite_prefix_norm2":
        return build_confirm_prefix_through_norm2_composite(dec).eval(), [
            stages[d3_input_key()]
        ]
    raise ValueError(f"Unknown case: {case}")


def _assert_composite_group_norm_present(export_dir: Path) -> None:
    summary = summarize_export_dir(export_dir)
    for stage in ("shlo", "ttir", "ttnn"):
        info = summary["stages"].get(stage, {})
        composite = {k: v for k, v in info.get("composite", {}).items() if v}
        if composite:
            return
    marker_hits = []
    for stage in ("shlo", "ttir", "ttnn"):
        fname = summary["stages"].get(stage, {}).get("file")
        if not fname:
            continue
        text = (export_dir / "irs" / fname).read_text(errors="replace")
        for marker in COMPOSITE_MARKERS:
            if marker in text:
                marker_hits.append(f"{stage}:{marker}")
    assert marker_hits, (
        f"No composite GroupNorm markers under {export_dir}/irs "
        f"(checked {COMPOSITE_MARKERS})"
    )


@pytest.mark.model_test
@pytest.mark.parametrize(
    "case_name",
    ["composite_norm2_alone", "composite_prefix_norm2"],
)
def test_export_ir_composite_groupnorm(vae_decoder_context, case_name: str):
    """Export IR for composite norm2 graphs; assert composite GN markers present."""
    xr.set_device_type("TT")
    export_dir = IR_EXPORT_ROOT / f"{case_name}_opt0"
    export_dir.mkdir(parents=True, exist_ok=True)

    module, inputs = _build_module_and_inputs(case_name, vae_decoder_context)
    compiler_config = CompilerConfig(
        optimization_level=0,
        export_path=str(export_dir),
        export_model_name=f"zimage_{case_name}",
    )
    tester = OpTester(framework=Framework.TORCH, compiler_config=compiler_config)
    workload = TorchWorkload(model=module, args=inputs)
    tester.test(workload)

    irs_dir = export_dir / "irs"
    mlir_files = list(irs_dir.glob("*.mlir")) if irs_dir.is_dir() else []
    assert mlir_files, f"No MLIR exported under {irs_dir}"

    print("\n" + format_summary(summarize_export_dir(export_dir)))
    _assert_composite_group_norm_present(export_dir)
