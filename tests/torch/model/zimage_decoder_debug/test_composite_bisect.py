# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2B — bisect the **new** composite-GroupNorm OOM on Z-Image VAE decoder @ 1280×720.

Same stage as Phase 1 (old OOM), different lowering
----------------------------------------------------
Both OOMs hit ``decoder.up_blocks[3].resnets[0].norm2`` — ``GroupNorm(32, 128)`` at
``(1, 128, 1280, 720)``.  Only the compiler/runtime path differs:

| Phase | Lowering | Failing op | Allocation | Log |
|-------|----------|------------|------------|-----|
| 1 (old) | Decomposed GN (opt0) | ``ttnn::subtract`` | ~3.77 GiB | ``zimage_logs/vae_decoder.log`` |
| 2A fix | Composite GN on norm2 only | subtract avoided | — | ``composite_prefix_norm2.log`` PASS |
| 2B (new) | **All** decoder GNs composite | ``ttnn::transpose`` → ``ttnn::mean`` | ~1.89 GiB | ``composite_bisect_d3b_norm2.log`` |

New OOM workspace: ``1_887_436_800`` B = 32 groups × 16 tile-padded ch × 921_600 spatial × 4 B.

Bisect conclusion (logs under ``zimage_logs/composite_bisect_*.log``)
---------------------------------------------------------------------
| Test | Result | Notes |
|------|--------|-------|
| ``prefix_through_up[3]`` | PASS | Blocks 0–2 + start of block 3 OK |
| ``d3`` (prefix + norm1 only) | PASS | ``resnet0.norm1`` GN(32, 256) OK |
| ``d3b[conv1]`` | PASS | conv1 OK; norm2 not yet executed |
| ``d3b[norm2]`` | **FAIL** | **Minimal repro** — first hit at ``resnet0.norm2`` composite mean |
| ``d4`` / ``isolated_up_block[3]`` / ``prefix_through_up[4]`` | FAIL | Same 1.89 GiB mean signature |
| ``d2_isolated_norm1`` | PASS | Composite norm1 alone OK |
| ``d2_isolated_norm2`` | PASS | Composite norm2 alone OK (golden after conv1) |

Norm2 composite mean OOMs in the **resnet0 chain** (norm1 + conv1 live), not in isolated
norm2-only graphs.  Isolated ``up_block[3]`` still fails → block-local, not prefix-only.

Next fix target: composite ``group_norm`` mean/transpose lowering in tt-mlir/tt-metal for
``up_blocks[3].resnets[0].norm2`` @ 1280×720.  After fix, run validation ladder in
``test_composite_groupnorm.py`` (``test_composite_vae_decoder_all_groupnorm_passes``).

Run prefix ladder (find first FAIL):
  for n in 0 1 2 3 4; do
    pytest tests/torch/model/zimage_decoder_debug/test_composite_bisect.py::test_composite_prefix_through_up[$n] -svv \\
      2>&1 | tee zimage_logs/composite_bisect_prefix_up_${n}.log
  done

Run targeted gates (resnet0 slice — after d3 PASS + d4 FAIL):
  pytest tests/torch/model/zimage_decoder_debug/test_composite_bisect.py::test_composite_d3b_prefix_up3_resnet0_stop[conv1] -svv \\
    2>&1 | tee zimage_logs/composite_bisect_d3b_conv1.log
  pytest tests/torch/model/zimage_decoder_debug/test_composite_bisect.py::test_composite_d3b_prefix_up3_resnet0_stop[norm2] -svv \\
    2>&1 | tee zimage_logs/composite_bisect_d3b_norm2.log

Run isolated single-op gates (no prefix):
  pytest tests/torch/model/zimage_decoder_debug/test_composite_bisect.py::test_composite_d2_isolated_norm1 -svv \\
    2>&1 | tee zimage_logs/composite_bisect_d2_isolated_norm1.log
  pytest tests/torch/model/zimage_decoder_debug/test_composite_bisect.py::test_composite_d2_isolated_norm2 -svv \\
    2>&1 | tee zimage_logs/composite_bisect_d2_isolated_norm2.log
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from .shared import (
    RESNET0_CUMULATIVE_STOPS,
    build_composite_norm1_only,
    build_composite_norm2_only,
    build_d3_prefix_up3_then_norm1,
    build_d3b_prefix_up3_resnet0_stop,
    build_d4_prefix_up3_resnet0_full,
    build_d5_prefix_up3_resnets01,
    build_d6_prefix_up3_block_full,
    build_decoder_head_composite,
    build_decoder_prefix_composite,
    build_isolated_up_block_composite,
    d2_input_key,
    d3_input_key,
    norm2_isolated_input_key,
    patch_decoder_all_groupnorms_composite,
)


def pytest_generate_tests(metafunc):
    if "resnet0_stop" in metafunc.fixturenames:
        metafunc.parametrize("resnet0_stop", RESNET0_CUMULATIVE_STOPS)


@pytest.mark.model_test
@pytest.mark.parametrize("num_up", [0, 1, 2, 3, 4])
def test_composite_prefix_through_up(vae_decoder_context, num_up: int):
    """Cumulative prefix + composite GN; first FAIL pinpoints the stage."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_decoder_prefix_composite(dec, num_up_blocks=num_up).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_prefix_full_decoder(vae_decoder_context):
    """All up_blocks + decoder head with composite GN."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_decoder_prefix_composite(
        dec, num_up_blocks=len(dec.up_blocks), include_head=True
    ).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
@pytest.mark.parametrize("block_index", [0, 1, 2, 3])
def test_composite_isolated_up_block(vae_decoder_context, block_index: int):
    """Isolated up_block[i] with composite GN (no cumulative prefix memory)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    input_key = "mid_block" if block_index == 0 else f"up_block_{block_index - 1}"
    module = build_isolated_up_block_composite(dec, block_index).eval()
    inputs = [stages[input_key]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_decoder_head_only(vae_decoder_context):
    """Decoder head only — input is golden ``up_block_3`` output (128 ch @ 1280×720)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_decoder_head_composite(dec).eval()
    inputs = [stages["up_block_3"]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d2_isolated_norm1(vae_decoder_context):
    """Isolated resnet0.norm1 composite GN (256 ch) — no prefix."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_composite_norm1_only(dec).eval()
    inputs = [stages[d2_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d2_isolated_norm2(vae_decoder_context):
    """Isolated resnet0.norm2 composite GN (128 ch) — input after conv1."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_composite_norm2_only(dec).eval()
    inputs = [stages[norm2_isolated_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d3_prefix_up3_norm1_only(vae_decoder_context):
    """Prefix + resnet0.norm1 only — gate for 256 ch composite GN @ 1280×720."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    patch_decoder_all_groupnorms_composite(dec)
    module = build_d3_prefix_up3_then_norm1(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d3b_prefix_up3_resnet0_stop(vae_decoder_context, resnet0_stop: str):
    """Prefix + resnet0 through conv1 / norm2 / full (all composite GN)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    patch_decoder_all_groupnorms_composite(dec)
    module = build_d3b_prefix_up3_resnet0_stop(dec, resnet0_stop).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d4_resnet0_full(vae_decoder_context):
    """Prefix + full resnet0 with **all** composite GN (norm1 + norm2)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    patch_decoder_all_groupnorms_composite(dec)
    module = build_d4_prefix_up3_resnet0_full(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d5_prefix_up3_resnets01(vae_decoder_context):
    """Prefix + resnet[0] + resnet[1] of up_block_3."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    patch_decoder_all_groupnorms_composite(dec)
    module = build_d5_prefix_up3_resnets01(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d6_prefix_up3_block_full(vae_decoder_context):
    """Prefix + entire up_block_3 (3 resnets)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    patch_decoder_all_groupnorms_composite(dec)
    module = build_d6_prefix_up3_block_full(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)
