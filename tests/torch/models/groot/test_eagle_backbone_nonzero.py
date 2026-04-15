# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GR00T non-zero input sanity tests.

Tests that use non-zero (dummy) input_ids to bypass the 0-length SIGFPE
and verify that individual blocks work on TT device when given valid tokens.
"""

import os
import torch
import numpy as np
import pytest
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig, PccConfig
from utils import Category

from tests.runner.requirements import RequirementsManager

LOADER_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "third_party", "tt_forge_models", "issac_groot", "pytorch", "loader.py",
))


def _load_model_and_inputs():
    """Load the full GR00T model and real preprocessed inputs."""
    from third_party.tt_forge_models.issac_groot.pytorch import ModelLoader, ModelVariant

    loader = ModelLoader(ModelVariant("Gr00t_N1.5_3B"))
    model = loader.load_model()
    model.eval()

    inputs = loader.load_inputs()

    cpu_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            cpu_inputs[k] = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            cpu_inputs[k] = v.cpu()
        else:
            cpu_inputs[k] = v

    return model, cpu_inputs


class EmbedTokensWrapper(torch.nn.Module):
    def __init__(self, embed_layer):
        super().__init__()
        self.embed = embed_layer

    def forward(self, input_ids):
        return self.embed(input_ids)


class EagleBackboneWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, state, state_mask, eagle_input_ids, eagle_attention_mask,
                eagle_pixel_values, eagle_image_sizes, embodiment_id):
        inputs = {
            "state": state,
            "state_mask": state_mask,
            "eagle_input_ids": eagle_input_ids,
            "eagle_attention_mask": eagle_attention_mask,
            "eagle_pixel_values": eagle_pixel_values,
            "eagle_image_sizes": eagle_image_sizes,
            "embodiment_id": embodiment_id,
        }
        out = self.backbone(inputs)
        return out["backbone_features"]


class ActionHeadPosEmbeddingWrapper(torch.nn.Module):
    def __init__(self, position_embedding):
        super().__init__()
        self.position_embedding = position_embedding

    def forward(self, pos_ids):
        return self.position_embedding(pos_ids)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_embed_tokens_nonzero():
    """
    Embedding(151680, 2048) with input_ids [1, 1] — single dummy token.
    Bypasses 0-length SIGFPE to test the embedding on TT device.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, _ = _load_model_and_inputs()
        gr00t = model.model

        embed_layer = gr00t.backbone.eagle_model.language_model.get_input_embeddings()
        input_ids = torch.tensor([[0]], dtype=torch.long)

        print(f"\n[nonzero] embed_tokens: Embedding({embed_layer.num_embeddings}, {embed_layer.embedding_dim})")
        print(f"  input_ids: shape={input_ids.shape}")

        wrapper = EmbedTokensWrapper(embed_layer)
        wrapper.eval()

        run_op_test(
            wrapper, [input_ids],
            comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_backbone_nonzero():
    """
    Full EagleBackbone with input_ids [1, 1] — single dummy token.
    Bypasses 0-length SIGFPE to test the full backbone graph compilation.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        backbone_inputs["eagle_input_ids"] = torch.tensor([[0]], dtype=torch.long)
        backbone_inputs["eagle_attention_mask"] = torch.tensor([[1]], dtype=torch.long)

        print(f"\n[nonzero] backbone inputs:")
        for k, v in backbone_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        wrapper = EagleBackboneWrapper(gr00t.backbone)
        wrapper.eval()

        args = [
            backbone_inputs["state"],
            backbone_inputs["state_mask"],
            backbone_inputs["eagle_input_ids"],
            backbone_inputs["eagle_attention_mask"],
            backbone_inputs["eagle_pixel_values"],
            backbone_inputs["eagle_image_sizes"],
            backbone_inputs["embodiment_id"],
        ]

        run_op_test(
            wrapper, args,
            comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.95)),
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_action_head_position_embedding():
    """
    FlowmatchingActionHead's position_embedding: Embedding(1024, 1536).
    Input: pos_ids = arange(16) [action_horizon].
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, _ = _load_model_and_inputs()
        gr00t = model.model

        action_head = gr00t.action_head
        pos_embed = action_head.position_embedding

        print(f"\n[nonzero] position_embedding: Embedding({pos_embed.num_embeddings}, {pos_embed.embedding_dim})")

        action_horizon = gr00t.action_horizon
        pos_ids = torch.arange(action_horizon, dtype=torch.long)

        wrapper = ActionHeadPosEmbeddingWrapper(pos_embed)
        wrapper.eval()

        run_op_test(
            wrapper, [pos_ids],
            comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
            framework=Framework.TORCH,
        )
