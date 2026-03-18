# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity tests for PI0 paligemma_with_expert.forward (prefix-only path).
 tests that progressively narrow down PCC-drop root cause:
1. test_pi0_paligemma_forward
   Forward only with saved (real) inputs from a prior CPU debug run.
"""


import pytest
import torch
from infra import Framework, run_op_test

from third_party.tt_forge_models.pi_0.pytorch import ModelLoader, ModelVariant



# ---------------------------------------------------------------------------
# Wrapper 1: forward only (pre-computed masks + position_ids as inputs)
# ---------------------------------------------------------------------------
class PaligemmaForwardWrapper(torch.nn.Module):
    """Wraps PaliGemmaWithExpertModel prefix-only forward as a standalone module."""

    def __init__(self, pi0_model):
        super().__init__()
        self.paligemma_with_expert = pi0_model.model.paligemma_with_expert
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    def forward(self, prefix_embs, attention_mask, position_ids):
        (prefix_output, _suffix_output), _past_kv = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return (prefix_output, _suffix_output), _past_kv


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def _load_inputs():
    path = "pi_0_libero_base_saved_inputs/sample_actions_paligemma_forward_inputs.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    return [
        data["prefix_embs"],
        data["prefix_att_2d_masks_4d"],
        data["prefix_position_ids"],
    ]




# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.single_device
@pytest.mark.parametrize("variant", [ModelVariant.LIBERO_BASE])
def test_pi0_paligemma_forward(variant):
    loader = ModelLoader(variant)
    model = loader.load_model()
    wrapper = PaligemmaForwardWrapper(model)
    wrapper.eval()

    inputs = _load_inputs()
    run_op_test(wrapper, inputs, framework=Framework.TORCH)