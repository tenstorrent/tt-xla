# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end chisel repro for the softmax-golden `dimension` bug.

Belongs in tt-mlir at: test/python/chisel/  (sibling of test_builder_chisel_integration.py,
whose pattern this follows: build a tiny graph, run it under chisel.session on device, assert
the per-op numerics records).

The bug: chisel's `softmax_golden` reads `kwargs.get("dim", 1)` but `chisel_ttnn_softmax` passes
the axis as `dimension=`, so the golden always softmaxes over dim 1. Any softmax with
`dimension != 1` therefore gets compared against a wrong-axis golden and records NUMERICS_FAIL,
even though the device kernel is correct. No mask or special input is needed to trigger it.

Expected, with the unfixed golden:
  * dimension=1  -> PASSES (golden's hardcoded default coincides with the real axis)
  * dimension=3  -> FAILS  (golden uses axis 1, device uses axis 3 -> PCC ~= 0)
After the one-line fix (mapping.py `softmax_golden`: read "dimension"), both PASS.

Run (in tt-mlir, on a machine with a device):
    pytest -svv test/python/chisel/test_chisel_softmax_dimension_repro.py
"""
from typing import List, Optional

import chisel
import pytest
import torch
from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "dimension",
    [
        pytest.param(1, id="dim1_ok_even_before_fix"),
        pytest.param(3, id="dim3_fails_before_fix"),
    ],
)
def test_chisel_softmax_dimension(request, device, tmp_path, dimension):
    shape = (2, 4, 8, 16)  # 4-D, like attention scores; dim 3 is the reduce axis

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def softmax_fn(
            x: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.softmax(x, dimension=dimension)

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=False),
    ) as report:
        compile_and_execute_ttir(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    softmax_recs = [
        r
        for r in records
        if r.op == "ttnn.softmax"
        and r.check == "numerics"
        and r.payload.mode == chisel.NumericsMode.ISOLATED
    ]
    assert (
        softmax_recs
    ), f"no isolated softmax numerics record (saw ops: {sorted({r.op for r in records})})"

    for r in softmax_recs:
        assert r.status == chisel.RecordStatus.OK, (
            f"softmax dimension={dimension}: chisel reports {r.status} "
            f"(pcc={r.payload.pcc}). Before the mapping.py fix, dimension=3 fails here "
            f"because the golden softmaxes over axis 1."
        )
        assert (
            r.payload.pcc >= PCC_THRESHOLD
        ), f"softmax dimension={dimension}: PCC {r.payload.pcc} < {PCC_THRESHOLD}"
