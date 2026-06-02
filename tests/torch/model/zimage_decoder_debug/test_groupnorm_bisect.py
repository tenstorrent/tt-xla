# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Playground #4710-style GroupNorm bisect for Z-Image VAE decoder OOM.

d1 — Stop the chained graph just before ``up_blocks[3].resnets[0].norm1``:
     run conv_in + mid + up_blocks[0..2], return the activation that would
     feed norm1 (same tensor as ``up_block_2`` output). Expected: PASS.

d2 — Run only ``up_blocks[3].resnets[0].norm1`` with CPU golden input from
     ``up_block_2``. Expected: PASS on a clean device if OOM is cumulative-only;
     OOM here would mean norm1 alone is broken (unlikely given ``a_up_block_3``).

d3 — Single graph: prefix + ``norm1`` only. Observed: PASS (not enough to repro).

d3b — Prefix + resnet[0] through ``conv1`` / ``norm2`` / ``full`` (parametrized).

d4 — Prefix + full resnet[0] (channel downsample 256→128).

d5 — Prefix + resnet[0] + resnet[1].

d6 — Prefix + full up_blocks[3] (3 resnets); expect OOM like ``prefix_through_up_4``.

Run:
  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d1_skip_before_up3_resnet0_norm1 2>&1 | tee zimage_decoder_logs/d1_skip_before_up3_norm1.log

  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d2_up3_resnet0_norm1_only 2>&1 | tee zimage_decoder_logs/d2_up3_resnet0_norm1_only.log

  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d3_prefix_up3_then_norm1_only 2>&1 | tee zimage_decoder_logs/d3_prefix_up3_then_norm1_only.log

  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d3b_prefix_up3_resnet0_stop 2>&1 | tee zimage_decoder_logs/d3b_resnet0_stop.log

  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d4_prefix_up3_resnet0_full 2>&1 | tee zimage_decoder_logs/d4_prefix_up3_resnet0_full.log

  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d5_prefix_up3_resnets01 2>&1 | tee zimage_decoder_logs/d5_prefix_up3_resnets01.log

  pytest -svv tests/torch/model/zimage_decoder_debug/test_groupnorm_bisect.py::test_d6_prefix_up3_block_full 2>&1 | tee zimage_decoder_logs/d6_prefix_up3_block_full.log
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from .shared import (
    RESNET0_CUMULATIVE_STOPS,
    build_d1_skip_before_up3_resnet0_norm1,
    build_d2_up3_resnet0_norm1_only,
    build_d3_prefix_up3_then_norm1,
    build_d3b_prefix_up3_resnet0_stop,
    build_d4_prefix_up3_resnet0_full,
    build_d5_prefix_up3_resnets01,
    build_d6_prefix_up3_block_full,
    d1_input_key,
    d2_input_key,
    d3_input_key,
)


def pytest_generate_tests(metafunc):
    if "resnet0_stop" in metafunc.fixturenames:
        metafunc.parametrize("resnet0_stop", RESNET0_CUMULATIVE_STOPS)


@pytest.mark.model_test
def test_d1_skip_before_up3_resnet0_norm1(vae_decoder_context):
    """d1: chained prefix stops before up_blocks[3].resnets[0].norm1 (expect PASS)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d1_skip_before_up3_resnet0_norm1(dec).eval()
    inputs = [stages[d1_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_d2_up3_resnet0_norm1_only(vae_decoder_context):
    """d2: isolate GroupNorm(32, 256) at full resolution (expect PASS if cumulative-only)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d2_up3_resnet0_norm1_only(dec).eval()
    inputs = [stages[d2_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_d3_prefix_up3_then_norm1_only(vae_decoder_context):
    """d3: prefix + norm1 only (observed PASS — extend with d3b–d6)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d3_prefix_up3_then_norm1(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_d3b_prefix_up3_resnet0_stop(vae_decoder_context, resnet0_stop: str):
    """Prefix + resnet[0] through conv1 / norm2 / full (find first cumulative OOM)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d3b_prefix_up3_resnet0_stop(dec, resnet0_stop).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_d4_prefix_up3_resnet0_full(vae_decoder_context):
    """Prefix + full first resnet of up_block_3."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d4_prefix_up3_resnet0_full(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_d5_prefix_up3_resnets01(vae_decoder_context):
    """Prefix + resnet[0] + resnet[1] of up_block_3."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d5_prefix_up3_resnets01(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_d6_prefix_up3_block_full(vae_decoder_context):
    """Prefix + all of up_block_3; expect OOM like prefix_through_up_4."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_d6_prefix_up3_block_full(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)
