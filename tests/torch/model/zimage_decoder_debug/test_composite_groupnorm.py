# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2A — composite GroupNorm fix candidate for Z-Image decoder OOM.

Replaces ``up_blocks[3].resnets[0].norm2`` with explicit ``composite_group_norm``
(``tenstorrent.group_norm`` StableHLO composite) instead of decomposed subtract.

1. ``test_composite_norm2_alone_passes`` — isolated norm2 @ 1280×720.
2. ``test_composite_prefix_through_norm2_passes`` — minimal cumulative repro with
   composite norm2; should PASS if composite lowering avoids the 3.77 GiB subtract OOM.

3. ``test_composite_vae_decoder_full_passes`` — full decoder; only norm2 composite.
4. ``test_composite_vae_decoder_all_groupnorm_passes`` — full decoder; all GroupNorms composite.

Phase 2B — new OOM when all GroupNorms are composite (same stage as Phase 1)
-----------------------------------------------------------------------------
``test_composite_vae_decoder_all_groupnorm_passes`` FAILs with ``ttnn::transpose`` →
``ttnn::mean`` allocating ~1.89 GiB at the **same layer** as the old subtract OOM:
``up_blocks[3].resnets[0].norm2`` (GroupNorm 32, 128 @ 1280×720).  Phase 2A (norm2-only
composite) avoids the 3.77 GiB subtract for that layer but full decoder still needs all
GNs composite; see ``test_composite_bisect.py`` and ``zimage_logs/composite_bisect_*.log``.

Run full decoder (all GroupNorms composite) on TT device:
  pytest tests/torch/model/zimage_decoder_debug/test_composite_groupnorm.py::test_composite_vae_decoder_all_groupnorm_passes -svv \\
    2>&1 | tee zimage_logs/vae_decoder_all_composite_gn.log
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from .shared import (
    build_composite_norm2_only,
    build_confirm_prefix_through_norm2_composite,
    build_d4_prefix_up3_resnet0_full,
    build_vae_decoder_all_composite_groupnorm,
    build_vae_decoder_composite_norm2,
    d3_input_key,
    norm2_isolated_input_key,
    patch_decoder_norm2_composite,
)


@pytest.mark.model_test
def test_composite_norm2_alone_passes(vae_decoder_context):
    """Isolated norm2 via composite_group_norm must pass at pipeline resolution."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_composite_norm2_only(dec).eval()
    inputs = [stages[norm2_isolated_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_prefix_through_norm2_passes(vae_decoder_context):
    """Cumulative prefix + composite norm2 — Phase 2A fix gate (expect PASS)."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    module = build_confirm_prefix_through_norm2_composite(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_d4_resnet0_full_passes(vae_decoder_context):
    """Prefix + full resnet0 via diffusers ResnetBlock2D with patched composite norm2."""
    xr.set_device_type("TT")
    dec = vae_decoder_context["vae"].decoder
    stages = vae_decoder_context["stages"]

    patch_decoder_norm2_composite(dec)
    module = build_d4_prefix_up3_resnet0_full(dec).eval()
    inputs = [stages[d3_input_key()]]

    run_graph_test(module, inputs, framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_vae_decoder_full_passes(vae_decoder_context):
    """Full Z-Image VAE decoder with composite norm2 on the OOM offender."""
    xr.set_device_type("TT")
    vae = vae_decoder_context["vae"]
    latents = vae_decoder_context["latents"]

    model = build_vae_decoder_composite_norm2(vae)
    run_graph_test(model, [latents], framework=Framework.TORCH)


@pytest.mark.model_test
def test_composite_vae_decoder_all_groupnorm_passes(vae_decoder_context):
    """Full Z-Image VAE decoder with every GroupNorm swapped to composite."""
    xr.set_device_type("TT")
    vae = vae_decoder_context["vae"]
    latents = vae_decoder_context["latents"]

    model = build_vae_decoder_all_composite_groupnorm(vae)
    run_graph_test(model, [latents], framework=Framework.TORCH)
