# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal reproduction tests for SFPI compiler ICE on Blackhole triggered by
sin/cos operations.

The SFPI compiler (GCC 15.1.0, tenstorrent/sfpi:7.31.0[315]) fails with an
internal register allocation error when compiling the trigonometry kernel:

    ckernel_sfpu_trigonometry.h:168:31: internal compiler error:
        in curr_insn_transform, at lra-constraints.cc:4355
    Failed to generate binaries for eltwise_sfpu

This blocks the Wan 2.1 T2V 1.3B transformer (and any model using sin/cos).

See also: tests/torch/models/wan/run_sin_cos_repro.py (standalone runner)
"""

import math

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category

# ---------------------------------------------------------------------------
# Test modules
# ---------------------------------------------------------------------------


class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Cos(torch.nn.Module):
    def forward(self, x):
        return torch.cos(x)


class SinCosConcat(torch.nn.Module):
    def forward(self, x):
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# ---------------------------------------------------------------------------
# Tests — all xfail due to SFPI compiler ICE on Blackhole
# ---------------------------------------------------------------------------

SFPI_ICE_REASON = (
    "SFPI compiler ICE: ckernel_sfpu_trigonometry.h:168 internal compiler error "
    "in curr_insn_transform (lra-constraints.cc:4355) on Blackhole. "
    "See https://github.com/tenstorrent/sfpi for upstream tracking."
)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(reason=SFPI_ICE_REASON, raises=RuntimeError)
def test_sin_blackhole_repro():
    """Bare torch.sin — simplest trigger for SFPI ICE."""
    run_op_test_with_random_inputs(Sin(), [(32, 32)], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(reason=SFPI_ICE_REASON, raises=RuntimeError)
def test_cos_blackhole_repro():
    """Bare torch.cos — simplest trigger for SFPI ICE."""
    run_op_test_with_random_inputs(Cos(), [(32, 32)], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(reason=SFPI_ICE_REASON, raises=RuntimeError)
def test_sin_cos_concat_blackhole_repro():
    """sin+cos concat — matches get_timestep_embedding pattern."""
    run_op_test_with_random_inputs(
        SinCosConcat(), [(1, 160)], framework=Framework.TORCH
    )
