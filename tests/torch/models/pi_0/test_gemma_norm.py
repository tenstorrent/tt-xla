# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for the GemmaRMSNorm that runs after all 18 decoder layers
in the PI0 paligemma language model forward.

The decoder layers sanity passes with PCC >= 0.99, so if this norm
sanity also passes, the PCC drop in the full paligemma forward is from
layerwise accumulation rather than a single broken op.

Inputs are loaded from a prior CPU debug run (set PI0_DEBUG_SAVE_DIR
when running the full model to generate gemma_norm_inputs.pt).
Weights come from the PI0 model loader.
"""

import os

import pytest
import torch
from infra import Framework, run_op_test

from third_party.tt_forge_models.pi_0.pytorch import ModelLoader, ModelVariant


class GemmaNormWrapper(torch.nn.Module):
    """Wraps the final GemmaRMSNorm (no conditioning / prefix path)."""

    def __init__(self, pi0_model):
        super().__init__()
        gemma_model = (
            pi0_model.model
            .paligemma_with_expert
            .paligemma
            .language_model
        )
        self.norm = gemma_model.norm

    def forward(self, hidden_states):
        normed, _ = self.norm(hidden_states, cond=None)
        return normed


class GemmaNormWithCondWrapper(torch.nn.Module):
    """Wraps GemmaRMSNorm with adarms_cond (expert/denoise path)."""

    def __init__(self, pi0_model, adarms_cond):
        super().__init__()
        gemma_model = (
            pi0_model.model
            .paligemma_with_expert
            .paligemma
            .language_model
        )
        self.norm = gemma_model.norm
        self.register_buffer("adarms_cond", adarms_cond)

    def forward(self, hidden_states):
        normed, gate = self.norm(hidden_states, cond=self.adarms_cond)
        return normed


def _load_saved_data():
    path ="pi_0_libero_base_saved_inputs/gemma_norm_inputs.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.single_device
@pytest.mark.parametrize("variant", [ModelVariant.LIBERO_BASE])
def test_gemma_norm(variant):
    """Run the final GemmaRMSNorm with saved post-decoder-layer hidden states.

    If this passes, the PCC drop in the full paligemma forward is purely
    from layerwise accumulation across the 18 decoder layers, not from
    the norm itself.
    """
    saved_data = _load_saved_data()

    loader = ModelLoader(variant)
    model = loader.load_model()

    adarms_cond = saved_data.get("adarms_cond", None)

    if adarms_cond is not None:
        wrapper = GemmaNormWithCondWrapper(model, adarms_cond)
    else:
        wrapper = GemmaNormWrapper(model)

    wrapper.eval()

    inputs = [saved_data["hidden_states"]]

    run_op_test(
        wrapper, inputs,
        framework=Framework.TORCH,
    )
