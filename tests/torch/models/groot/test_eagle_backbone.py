# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GR00T zero-length input sanity tests.

Tests each block iteratively with the REAL model inputs (input_ids [1,0] = 0 tokens).
The model legitimately sends 0 text tokens — all content comes from the image.

Run in order:
  1.  test_1_embed_tokens        — Embedding(151680,2048) with [1,0]
  2.  test_2_extract_feature     — SiglipVisionModel with [1,3,224,224]
  3.  test_3_language_model      — Qwen3 with 0-length inputs_embeds
  4.  test_4_eagle_backbone      — Full EagleBackbone
  5.  test_5_forward_eagle       — EagleBackbone.forward_eagle()
  5b. test_5b_action_head        — FlowmatchingActionHead alone (backbone on CPU)
  6.  test_6_groot_forward       — GR00T_N1_5.forward() (backbone + action head)
"""

import os
import torch
import numpy as np
import pytest
from infra import Framework, run_op_test
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


def _extract_eagle_input(backbone_inputs):
    """Extract eagle-prefixed inputs, remove image_sizes."""
    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v
        for k, v in backbone_inputs.items()
        if k.startswith(eagle_prefix)
    }
    eagle_input.pop("image_sizes", None)
    return eagle_input


# ── Wrappers ──────────────────────────────────────────────────────────────────


class EmbedTokensWrapper(torch.nn.Module):
    def __init__(self, embed_layer):
        super().__init__()
        self.embed = embed_layer

    def forward(self, input_ids):
        return self.embed(input_ids)


class VisionModelWrapper(torch.nn.Module):
    def __init__(self, eagle_model):
        super().__init__()
        self.eagle_model = eagle_model

    def forward(self, pixel_values):
        return self.eagle_model.extract_feature(pixel_values)


class LanguageModelWrapper(torch.nn.Module):
    def __init__(self, language_model, select_layer):
        super().__init__()
        self.language_model = language_model
        self.select_layer = select_layer

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[self.select_layer]


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


class ForwardEagleWrapper(torch.nn.Module):
    """Wraps EagleBackbone.forward_eagle() which is the Eagle2_5_VL call."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        from transformers.image_processing_utils import BatchFeature
        vl_input = BatchFeature(data={
            "eagle_input_ids": input_ids,
            "eagle_attention_mask": attention_mask,
            "eagle_pixel_values": pixel_values,
            "eagle_image_sizes": image_sizes,
        })
        features, _ = self.backbone.forward_eagle(vl_input)
        return features


class GR00TForwardWrapper(torch.nn.Module):
    """Wraps GR00T_N1_5 forward: backbone + action_head on already-prepared inputs."""

    def __init__(self, gr00t_model):
        super().__init__()
        self.backbone = gr00t_model.backbone
        self.action_head = gr00t_model.action_head

    def forward(self, state, state_mask, eagle_input_ids, eagle_attention_mask,
                eagle_pixel_values, eagle_image_sizes, embodiment_id):
        from transformers.image_processing_utils import BatchFeature

        compute_dtype = next(self.action_head.parameters()).dtype
        state = state.to(dtype=compute_dtype)

        backbone_inputs = BatchFeature(data={
            "state": state,
            "state_mask": state_mask,
            "eagle_input_ids": eagle_input_ids,
            "eagle_attention_mask": eagle_attention_mask,
            "eagle_pixel_values": eagle_pixel_values,
            "eagle_image_sizes": eagle_image_sizes,
            "embodiment_id": embodiment_id,
        })
        backbone_outputs = self.backbone(backbone_inputs)
        action_inputs = BatchFeature(data={
            "state": state,
            "state_mask": state_mask,
            "embodiment_id": embodiment_id,
        })
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        return action_head_outputs["action_pred"]


class ActionHeadWrapper(torch.nn.Module):
    """Wraps FlowmatchingActionHead: takes precomputed backbone_features."""

    def __init__(self, action_head):
        super().__init__()
        self.action_head = action_head

    def forward(self, backbone_features, state, embodiment_id):
        from transformers.image_processing_utils import BatchFeature
        backbone_output = BatchFeature(data={
            "backbone_features": backbone_features,
        })
        action_input = BatchFeature(data={
            "state": state,
            "embodiment_id": embodiment_id,
        })
        out = self.action_head(backbone_output, action_input)
        return out["action_pred"]


# ── Tests (run iteratively, all use real 0-length inputs) ─────────────────────


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_1_embed_tokens():
    """
    1. Embedding(151680, 2048) with input_ids [1, 0] (zero tokens).
    Expected: SIGFPE — ttnn::embedding can't handle 0-length input.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        embed_layer = gr00t.backbone.eagle_model.language_model.get_input_embeddings()
        eagle_input = _extract_eagle_input(backbone_inputs)
        input_ids = eagle_input["input_ids"]

        print(f"\n[test_1] embed_tokens: Embedding({embed_layer.num_embeddings}, {embed_layer.embedding_dim})")
        print(f"  input_ids: shape={input_ids.shape}, numel={input_ids.numel()}")

        wrapper = EmbedTokensWrapper(embed_layer)
        wrapper.eval()

        run_op_test(
            wrapper, [input_ids],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_2_extract_feature():
    """
    2. SiglipVisionModel with pixel_values [1, 3, 224, 224].
    No 0-length tensors here — should pass.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        eagle_model = gr00t.backbone.eagle_model
        eagle_input = _extract_eagle_input(backbone_inputs)
        pixel_values = eagle_input["pixel_values"]

        print(f"\n[test_2] extract_feature: pixel_values shape={pixel_values.shape}")

        wrapper = VisionModelWrapper(eagle_model)
        wrapper.eval()

        run_op_test(
            wrapper, [pixel_values],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_3_language_model():
    """
    3. Qwen3 language model with 0-length inputs_embeds [1, 0, 2048].
    Skips embed_tokens (run on CPU), feeds result to language_model.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        eagle_model = gr00t.backbone.eagle_model
        eagle_input = _extract_eagle_input(backbone_inputs)

        with torch.no_grad():
            input_embeds = eagle_model.language_model.get_input_embeddings()(eagle_input["input_ids"])
            attention_mask = eagle_input["attention_mask"]

        print(f"\n[test_3] language_model:")
        print(f"  inputs_embeds: shape={input_embeds.shape}")
        print(f"  attention_mask: shape={attention_mask.shape}")

        wrapper = LanguageModelWrapper(eagle_model.language_model, gr00t.backbone.select_layer)
        wrapper.eval()

        run_op_test(
            wrapper, [input_embeds, attention_mask],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_4_eagle_backbone():
    """
    4. Full EagleBackbone with real inputs (input_ids [1,0]).
    Combines embed_tokens + vision + language model.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        print(f"\n[test_4] eagle_backbone inputs:")
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
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_5_forward_eagle():
    """
    5. EagleBackbone.forward_eagle() — the Eagle2_5_VL call.
    Inputs: input_ids [1,0], attention_mask [1,0], pixel_values [1,3,224,224].
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)
        eagle_input = _extract_eagle_input(backbone_inputs)

        print(f"\n[test_5] forward_eagle inputs:")
        for k, v in eagle_input.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        wrapper = ForwardEagleWrapper(gr00t.backbone)
        wrapper.eval()

        image_sizes = backbone_inputs["eagle_image_sizes"]

        run_op_test(
            wrapper,
            [eagle_input["input_ids"], eagle_input["attention_mask"],
             eagle_input["pixel_values"], image_sizes],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_5b_action_head():
    """
    5b. FlowmatchingActionHead alone with precomputed backbone_features.
    Backbone is run on CPU to get backbone_features [1, 0, 2048] (zero-length!).
    Then action head is tested on TT device in isolation.
    This isolates whether the L1 overflow comes from the action head's embedding
    (position_embedding Embedding(1024,1536)) when compiled as part of the
    action head graph, vs the backbone graph.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        compute_dtype = next(gr00t.action_head.parameters()).dtype
        state = backbone_inputs["state"].to(dtype=compute_dtype)
        embodiment_id = backbone_inputs["embodiment_id"]

        with torch.no_grad():
            backbone_outputs = gr00t.backbone(backbone_inputs)

        backbone_features = backbone_outputs["backbone_features"]

        print(f"\n[test_5b] action_head inputs:")
        print(f"  backbone_features: shape={backbone_features.shape}, dtype={backbone_features.dtype}")
        print(f"  state: shape={state.shape}, dtype={state.dtype}")
        print(f"  embodiment_id: shape={embodiment_id.shape}, dtype={embodiment_id.dtype}")

        torch.manual_seed(42)
        wrapper = ActionHeadWrapper(gr00t.action_head)
        wrapper.eval()

        run_op_test(
            wrapper,
            [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_6_groot_forward():
    """
    6. GR00T_N1_5.forward() — backbone + action head.
    Full model forward with real 0-length inputs.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        model, cpu_inputs = _load_model_and_inputs()
        gr00t = model.model
        backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

        print(f"\n[test_6] groot_forward inputs:")
        for k, v in backbone_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        torch.manual_seed(42)
        wrapper = GR00TForwardWrapper(gr00t)
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
            framework=Framework.TORCH,
        )
