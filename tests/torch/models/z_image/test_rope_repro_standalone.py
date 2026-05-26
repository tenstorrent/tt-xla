# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Z-Image RoPE mlir repro (issue #4756) — no HF weights or transformer load.

  source venv/activate
  python -m pytest -svv tests/torch/models/z_image/test_rope_repro_standalone.py

Uses synthetic ``pos_ids`` and inlined ``RopeEmbedder`` logic from
``standalone_rope_repro.py``. TT compile is expected to fail with Error code 13
for the same op family as ``test_transformer_slice.py`` / full transformer.
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_op_test

from .standalone_rope_repro import (
    COMPILE_REPRO_CASES,
    RUN_ON_TT_CASES,
    _COMPILE_ERROR,
    build_standalone_case,
)


def _run_case(case_name: str) -> None:
    xr.set_device_type("TT")
    model, inputs = build_standalone_case(case_name)
    run_op_test(model.eval(), inputs, framework=Framework.TORCH)


@pytest.mark.model_test
@pytest.mark.parametrize("case_name", sorted(COMPILE_REPRO_CASES))
def test_rope_repro_standalone_compile_fails_single_chip(case_name: str):
    """TT SHLO→TTIR must fail — same complex gather/cat legalization as full model."""
    with pytest.raises((ValueError, RuntimeError), match=_COMPILE_ERROR):
        _run_case(case_name)


@pytest.mark.model_test
@pytest.mark.parametrize("case_name", sorted(RUN_ON_TT_CASES))
def test_rope_repro_standalone_runs_single_chip(case_name: str):
    _run_case(case_name)
