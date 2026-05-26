# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op RoPE embedder sanities (issue #4756) — no HF weights.

  source venv/activate
  python -m pytest -svv tests/torch/models/z_image/test_rope_embedder_op_sanity.py

``rope_embedder_op_sanity.py`` splits ``precompute_freqs_cis`` into steps 1–4,
then whole precompute (5), gather (6–7), and cat (8).
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_op_test

from .rope_embedder_op_sanity import (
    COMPILE_REPRO_CASES,
    RUN_ON_TT_CASES,
    _COMPILE_ERROR,
    build_op_sanity_case,
)


def _run_case(case_name: str) -> None:
    xr.set_device_type("TT")
    model, inputs = build_op_sanity_case(case_name)
    run_op_test(model.eval(), inputs, framework=Framework.TORCH)


@pytest.mark.model_test
@pytest.mark.parametrize("case_name", sorted(RUN_ON_TT_CASES))
def test_rope_embedder_op_sanity_runs_on_tt(case_name: str):
    """Steps 1–5 (precompute / polar chain) should compile and match CPU."""
    _run_case(case_name)


@pytest.mark.model_test
def test_gather_complex_polar_table_only_compile_fails():
    """``[512, 24]`` complex64 polar buffer + gather only (confirms #4756 = complex gather)."""
    with pytest.raises((ValueError, RuntimeError), match=_COMPILE_ERROR):
        _run_case("gather_complex_polar_table_only")


@pytest.mark.model_test
@pytest.mark.parametrize(
    "case_name",
    sorted(COMPILE_REPRO_CASES - {"gather_complex_polar_table_only"}),
)
def test_rope_embedder_op_sanity_compile_fails(case_name: str):
    """Steps 6–8 (complex gather / cat) should fail TT compile with Error 13."""
    with pytest.raises((ValueError, RuntimeError), match=_COMPILE_ERROR):
        _run_case(case_name)
