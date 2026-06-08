# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1.1 — Export MLIR for Z-Image decoder GroupNorm bisect (PASS vs FAIL, opt0 vs opt1).

Three graphs × two optimization levels (6 runs) → IR under ``zimage_decoder_ir/<case>_opt<N>/irs/``.

``full_decoder`` is omitted — it OOMs with the same subtract signature as ``prefix_norm2``.

| case          | Graph                          | Execute expectation |
|---------------|--------------------------------|---------------------|
| norm2_alone   | GroupNorm(32,128) isolated     | PASS                |
| prefix_conv1  | prefix + resnet0 through conv1 | PASS                |
| prefix_norm2  | prefix + resnet0 through norm2 | OOM (minimal repro) |

Run on TT device:
  pytest -svv tests/torch/model/zimage_decoder_debug/test_export_ir_bisect.py 2>&1 | tee zimage_logs/export_ir_bisect.log

Inspect after run:
  rg -n 'tenstorrent\\.group_norm|ttnn\\.group_norm' zimage_decoder_ir/prefix_norm2_opt1/irs/
  rg -n 'subtract|binary_ng' zimage_decoder_ir/prefix_norm2_opt0/irs/ttnn*.mlir
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import Framework
from infra.workloads.torch_workload import TorchWorkload

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import OpTester

from .ir_analysis import format_summary, summarize_export_dir
from .shared import (
    build_confirm_prefix_through_conv1,
    build_confirm_prefix_through_norm2,
    build_norm2_only,
    d3_input_key,
    norm2_isolated_input_key,
)

# tt-xla repo root (…/tests/torch/model/zimage_decoder_debug/ → parents[4])
IR_EXPORT_ROOT = Path(__file__).resolve().parents[4] / "zimage_decoder_ir"

OOM_CASES = frozenset({"prefix_norm2"})


def _build_module_and_inputs(case: str, context: dict):
    dec = context["vae"].decoder
    stages = context["stages"]
    if case == "norm2_alone":
        return build_norm2_only(dec).eval(), [stages[norm2_isolated_input_key()]]
    if case == "prefix_conv1":
        return build_confirm_prefix_through_conv1(dec).eval(), [stages[d3_input_key()]]
    if case == "prefix_norm2":
        return build_confirm_prefix_through_norm2(dec).eval(), [stages[d3_input_key()]]
    raise ValueError(f"Unknown case: {case}")


@pytest.mark.model_test
@pytest.mark.parametrize("opt_level", [0, 1])
@pytest.mark.parametrize(
    "case_name",
    ["norm2_alone", "prefix_conv1", "prefix_norm2"],
)
def test_export_ir_bisect(vae_decoder_context, case_name: str, opt_level: int):
    """Compile with IR export; execute when possible; print GN marker summary."""
    xr.set_device_type("TT")
    export_dir = IR_EXPORT_ROOT / f"{case_name}_opt{opt_level}"
    export_dir.mkdir(parents=True, exist_ok=True)

    module, inputs = _build_module_and_inputs(case_name, vae_decoder_context)
    compiler_config = CompilerConfig(
        optimization_level=opt_level,
        export_path=str(export_dir),
        export_model_name=f"zimage_{case_name}",
    )
    tester = OpTester(framework=Framework.TORCH, compiler_config=compiler_config)
    workload = TorchWorkload(model=module, args=inputs)

    if case_name in OOM_CASES:
        with pytest.raises(RuntimeError):
            tester.test(workload)
    else:
        tester.test(workload)

    irs_dir = export_dir / "irs"
    mlir_files = list(irs_dir.glob("*.mlir")) if irs_dir.is_dir() else []
    assert mlir_files, f"No MLIR exported under {irs_dir}"

    print("\n" + format_summary(summarize_export_dir(export_dir)))
