# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Z-Image VAE decoder — per-stage TT vs CPU sanity tests.

Run the full suite to find the first stage that OOMs or miscompares:

  pytest tests/torch/model/zimage_decoder_debug/ -v

Full-graph failure reference: zimage_logs/decoder.log (TT_FATAL on subtract, ~3.77 GiB).
"""

import pytest
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.z_image.pytorch.src.model_utils import VAEDecoderWrapper


def _get_stage_spec(context, stage_name: str):
    for spec in context["specs"]:
        if spec.name == stage_name:
            return spec
    raise KeyError(f"Unknown stage: {stage_name}")


@pytest.mark.model_test
def test_vae_decoder_full(vae_decoder_context):
    """End-to-end decoder (same as tests/torch/models/z_image/test_vae_decoder.py)."""
    xr.set_device_type("TT")
    vae = vae_decoder_context["vae"]
    latents = vae_decoder_context["latents"]
    model = VAEDecoderWrapper(vae).eval()
    run_graph_test(model, [latents], framework=Framework.TORCH)


@pytest.mark.model_test
def test_decoder_stage(vae_decoder_context, stage_name: str):
    """Run one decoder stage on TT with CPU golden input activations."""
    xr.set_device_type("TT")
    spec = _get_stage_spec(vae_decoder_context, stage_name)
    stages = vae_decoder_context["stages"]

    module = spec.build_module().eval()
    inputs = [stages[spec.input_key]]

    run_graph_test(module, inputs, framework=Framework.TORCH)
