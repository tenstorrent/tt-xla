# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Z-Image RoPE mlir repro (issue #4756) — no HF weights or transformer load.

Requires tt-mlir branch ``akannan/zimage_shlo_bug`` (complex gather legalization).

  source venv/activate
  python -m pytest -svv tests/torch/models/z_image/test_rope_repro_standalone.py
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_op_test

from .standalone_rope_repro import ALL_STANDALONE_CASES, build_standalone_case


def _run_case(case_name: str) -> None:
    xr.set_device_type("TT")
    model, inputs = build_standalone_case(case_name)
    run_op_test(model.eval(), inputs, framework=Framework.TORCH)


@pytest.mark.model_test
@pytest.mark.parametrize("case_name", ALL_STANDALONE_CASES)
def test_rope_repro_standalone_runs_single_chip(case_name: str):
    _run_case(case_name)
