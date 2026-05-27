# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op RoPE embedder sanities (issue #4756) — no HF weights.

Requires tt-mlir branch ``akannan/zimage_shlo_bug`` (complex gather legalization).
Without it, these tests fail at TT compile with Error code 13.

  source venv/activate
  python -m pytest -svv tests/torch/models/z_image/test_rope_embedder_op_sanity.py
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_op_test

from .rope_embedder_op_sanity import ALL_OP_SANITY_CASES, build_op_sanity_case


def _run_case(case_name: str) -> None:
    xr.set_device_type("TT")
    model, inputs = build_op_sanity_case(case_name)
    run_op_test(model.eval(), inputs, framework=Framework.TORCH)


@pytest.mark.model_test
@pytest.mark.parametrize("case_name", ALL_OP_SANITY_CASES)
def test_rope_embedder_op_sanity_runs_on_tt(case_name: str):
    """Precompute, gather, and cat steps should compile and match CPU with tt-mlir fix."""
    _run_case(case_name)
