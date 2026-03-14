# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for PI0 embed_prefix submodule.

Loads the PI0 model via its third-party loader and replays saved inputs
from a CPU debug run through embed_prefix (SigLIP image embedding +
language token embedding).  Compares CPU vs TT-device output.

Run the full model with PI0_DEBUG_SAVE_DIR set to generate the .pt files
before running this test.
"""

import os

import pytest
import torch
from infra import Framework, run_op_test

from third_party.tt_forge_models.pi_0.pytorch import ModelLoader, ModelVariant

_DEBUG_DIR = os.environ.get("PI0_DEBUG_SAVE_DIR", "debug_folder/pi0_debug")


class EmbedPrefixWrapper(torch.nn.Module):
    """Wraps PI0Pytorch.embed_prefix as a standalone module for op testing."""

    def __init__(self, pi0_model):
        super().__init__()
        self.pi0 = pi0_model.model

    def forward(self, *flat_inputs):
        n_images = (len(flat_inputs) - 2) // 2
        images = list(flat_inputs[:n_images])
        img_masks = list(flat_inputs[n_images : 2 * n_images])
        lang_tokens = flat_inputs[2 * n_images]
        lang_masks = flat_inputs[2 * n_images + 1]

        embs, pad_masks, att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        return embs


def _load_inputs():
    path = os.path.join(_DEBUG_DIR, "block1_embed_prefix_inputs.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)

    flat = []
    for img in data["images"]:
        flat.append(img)
    for mask in data["img_masks"]:
        flat.append(mask)
    flat.append(data["lang_tokens"])
    flat.append(data["lang_masks"])
    return flat


@pytest.mark.single_device
@pytest.mark.parametrize("variant", [ModelVariant.LIBERO_BASE])
def test_pi0_embed_prefix(variant):
    loader = ModelLoader(variant)
    model = loader.load_model()
    wrapper = EmbedPrefixWrapper(model)
    wrapper.eval()

    inputs = _load_inputs()
    run_op_test(wrapper, inputs, framework=Framework.TORCH)
