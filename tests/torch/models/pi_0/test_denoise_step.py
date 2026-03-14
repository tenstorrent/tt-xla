# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for PI0 denoise_step submodule (all 10 denoising steps).

Loads the PI0 model via its third-party loader and replays saved inputs
from a CPU debug run through denoise_step.  Each parametrized step uses
its own saved x_t / timestep while sharing the same past_key_values KV
cache (computed once from the prefix forward).

The denoise_step internally calls embed_suffix → make_att_2d_masks →
paligemma_with_expert.forward (suffix-only path via gemma_expert) →
action_out_proj.  It exercises cumsum on bool masks and the expert
transformer layers.

Run the full model with PI0_DEBUG_SAVE_DIR set to generate the .pt files
before running this test.
"""

import os

import pytest
import torch
from infra import Framework, run_op_test
from transformers import DynamicCache

from third_party.tt_forge_models.pi_0.pytorch import ModelLoader, ModelVariant

_DEBUG_DIR = os.environ.get("PI0_DEBUG_SAVE_DIR", "debug_folder/pi0_debug")
_NUM_STEPS = 10


class DenoiseStepWrapper(torch.nn.Module):
    """Wraps PI0Pytorch.denoise_step as a standalone module for op testing.

    past_key_values (DynamicCache) tensors are stored as registered buffers
    so they travel with the module when moved to a different device via .to().
    """

    def __init__(self, pi0_model, past_key_values):
        super().__init__()
        self.pi0 = pi0_model.model
        self.pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        self.pi0.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        self._n_kv_layers = len(past_key_values.key_cache)
        for i, (k, v) in enumerate(
            zip(past_key_values.key_cache, past_key_values.value_cache)
        ):
            self.register_buffer(f"_kv_key_{i}", k)
            self.register_buffer(f"_kv_val_{i}", v)

        self._seen_tokens = past_key_values._seen_tokens

    def _rebuild_kv_cache(self):
        cache = DynamicCache()
        cache._seen_tokens = self._seen_tokens
        for i in range(self._n_kv_layers):
            cache.key_cache.append(getattr(self, f"_kv_key_{i}"))
            cache.value_cache.append(getattr(self, f"_kv_val_{i}"))
        return cache

    def forward(self, state, prefix_pad_masks, x_t, timestep):
        past_key_values = self._rebuild_kv_cache()
        return self.pi0.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )


def _load_inputs(step: int):
    path = os.path.join(_DEBUG_DIR, f"block3_denoise_step_{step}_inputs.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    inputs = [
        data["state"],
        data["prefix_pad_masks"],
        data["x_t"],
        data["timestep"],
    ]
    return inputs, data["past_key_values"]


@pytest.mark.single_device
@pytest.mark.parametrize("step", list(range(_NUM_STEPS)))
@pytest.mark.parametrize("variant", [ModelVariant.LIBERO_BASE])
def test_pi0_denoise_step(variant, step):
    loader = ModelLoader(variant)
    model = loader.load_model()

    inputs, past_key_values = _load_inputs(step)

    wrapper = DenoiseStepWrapper(model, past_key_values)
    wrapper.eval()

    run_op_test(wrapper, inputs, framework=Framework.TORCH)
