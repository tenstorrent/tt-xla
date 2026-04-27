# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Hang investigation tests for DeepSeek OCR after masked_scatter decomposition fix.

After replacing the flatten-based masked_scatter decomposition with the
mul+add decomposition in the whole model (forge models branch
akannan/replace_decomp_deepseek), the cumsum OOM is resolved but the model
now hangs during execution. These tests isolate individual decoder layers
to identify which layer is the culprit.

Architecture:
  - Layer 0: DeepseekV2DecoderLayer with DeepseekV2MLP (dense)
  - Layer 1: DeepseekV2DecoderLayer with DeepseekV2MoE (64 routed experts, top-6)
  - Layers 2-11: repeat of MoE layers

Test plan:
  1. Layer 0 only (dense MLP) — should pass if hang is in MoE
  2. Layer 1 only (first MoE layer) — expected to hang
  3. Layers 0+1 combined — confirm hang with MoE present

Uses the mul+add decomposition (inlined in vision embed wrapper) which is
known to pass for the vision embedding stage.
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)


def _masked_scatter_decomp_muladd(inputs_embeds_row, mask_1d, source):
    """Row-level cumsum + mul+add flat indexing (known to pass)."""
    S, D = inputs_embeds_row.shape
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    flat_source = source.reshape(-1)
    col_idx = torch.arange(D, device=source.device, dtype=source_idx.dtype)
    flat_idx = source_idx.unsqueeze(-1) * D + col_idx.unsqueeze(0)
    gathered_rows = flat_source[flat_idx.reshape(-1)].reshape(S, D)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


class DeepseekOCRVisionEmbedPipeline(nn.Module):
    """Vision embed pipeline with mul+add decomposition (known to pass)."""

    def __init__(self, ocr_model):
        super().__init__()
        self.embed_tokens = ocr_model.model.embed_tokens
        self.sam_model = ocr_model.model.sam_model
        self.vision_model = ocr_model.model.vision_model
        self.projector = ocr_model.model.projector
        self.image_newline = ocr_model.model.image_newline
        self.view_seperator = ocr_model.model.view_seperator

    def forward(
        self,
        input_ids,
        patches,
        image_ori,
        images_seq_mask,
        images_spatial_crop,
    ):
        attention_mask = None
        position_ids = None
        past_key_values = None

        inputs_embeds = self.embed_tokens(input_ids)

        sam_model = getattr(self, "sam_model", None)
        vision_model = getattr(self, "vision_model", None)

        images = [(patches, image_ori)]

        if (
            sam_model is not None
            and (input_ids.shape[1] != 1 or self.training)
            and torch.sum(images[0][1], dim=(0, 1, 2, 3)).item() != 0
        ):

            idx = 0
            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]
                image_ori = image[1]

                with torch.no_grad():

                    if torch.sum(patches).item() != 0:
                        local_features_1 = sam_model(patches)
                        local_features_2 = vision_model(patches, local_features_1)
                        local_features = torch.cat(
                            (
                                local_features_2[:, 1:],
                                local_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        local_features = self.projector(local_features)
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (
                                global_features_2[:, 1:],
                                global_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        global_features = self.projector(global_features)
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw**0.5)

                        _2, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2**0.5)

                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                        global_features = global_features.view(h, w, n_dim)

                        global_features = torch.cat(
                            [
                                global_features,
                                self.image_newline[None, None, :].expand(h, 1, n_dim),
                            ],
                            dim=1,
                        )

                        global_features = global_features.view(-1, n_dim)

                        local_features = (
                            local_features.view(
                                height_crop_num, width_crop_num, h2, w2, n_dim2
                            )
                            .permute(0, 2, 1, 3, 4)
                            .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                        )
                        local_features = torch.cat(
                            [
                                local_features,
                                self.image_newline[None, None, :].expand(
                                    height_crop_num * h2, 1, n_dim2
                                ),
                            ],
                            dim=1,
                        )
                        local_features = local_features.view(-1, n_dim2)

                        global_local_features = torch.cat(
                            [
                                local_features,
                                global_features,
                                self.view_seperator[None, :],
                            ],
                            dim=0,
                        )

                    else:
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (
                                global_features_2[:, 1:],
                                global_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        global_features = self.projector(global_features)
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw**0.5)

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat(
                            [
                                global_features,
                                self.image_newline[None, None, :].expand(h, 1, n_dim),
                            ],
                            dim=1,
                        )
                        global_features = global_features.view(-1, n_dim)
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :]], dim=0
                        )

                    images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    inputs_embeds[idx] = _masked_scatter_decomp_muladd(
                        inputs_embeds[idx],
                        images_seq_mask[idx],
                        images_in_this_batch,
                    )

                idx += 1

        return inputs_embeds, attention_mask, position_ids, past_key_values


class DeepseekOCRLayerSlicePipeline(nn.Module):
    """Vision embed (mul+add) + DeepseekV2 setup + decoder layers[start:end].

    Used to isolate which decoder layer(s) cause the hang.
    """

    def __init__(self, ocr_model, start, end):
        super().__init__()
        self.vision_embed = DeepseekOCRVisionEmbedPipeline(ocr_model)
        self._use_flash_attention_2 = ocr_model.model._use_flash_attention_2
        self.layers = ocr_model.model.layers[start:end]

    def forward(
        self,
        input_ids,
        patches,
        image_ori,
        images_seq_mask,
        images_spatial_crop,
    ):
        inputs_embeds, _, _, _ = self.vision_embed(
            input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )

        batch_size, seq_length = inputs_embeds.shape[:2]

        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=inputs_embeds.device,
        ).unsqueeze(0)

        if self._use_flash_attention_2:
            attention_mask = None
        else:
            attention_mask = _prepare_4d_mask(
                None, (batch_size, seq_length), inputs_embeds, 0,
            )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        return (hidden_states,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_and_inputs():
    import inspect
    import third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader as deepseek_ocr_loader
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch import (
        ModelLoader,
    )
    from tests.runner.requirements import RequirementsManager

    loader_path = inspect.getsourcefile(deepseek_ocr_loader)
    with RequirementsManager.for_loader(loader_path):
        loader = ModelLoader()
        full_model = loader.load_model(dtype_override=torch.bfloat16)
        raw_inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    return full_model, raw_inputs


def _extract_inputs(raw_inputs):
    return [
        raw_inputs["input_ids"],
        raw_inputs["images"][0][0],
        raw_inputs["images"][0][1],
        raw_inputs["images_seq_mask"],
        raw_inputs["images_spatial_crop"],
    ]


def _make_layer_slice_fixture(full_model_and_raw_inputs, start, end):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = DeepseekOCRLayerSlicePipeline(full_model, start, end)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_model_and_raw_inputs():
    return _load_model_and_inputs()


@pytest.fixture(scope="module")
def layer0_only_model_and_inputs(full_model_and_raw_inputs):
    """Layer 0 only (dense MLP)."""
    return _make_layer_slice_fixture(full_model_and_raw_inputs, 0, 1)


@pytest.fixture(scope="module")
def layer1_only_model_and_inputs(full_model_and_raw_inputs):
    """Layer 1 only (first MoE layer — 64 routed experts, top-6)."""
    return _make_layer_slice_fixture(full_model_and_raw_inputs, 1, 2)


@pytest.fixture(scope="module")
def layer0_and_1_model_and_inputs(full_model_and_raw_inputs):
    """Layers 0-1 (dense + first MoE)."""
    return _make_layer_slice_fixture(full_model_and_raw_inputs, 0, 2)


# ---------------------------------------------------------------------------
# 1. Layer 0 only (dense MLP) — expected to PASS
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_layer0_only(layer0_only_model_and_inputs):
    """
    Vision embed + DeepseekV2 setup + layer 0 only (DeepseekV2MLP, dense).
    Layer 0 has no MoE — expected to pass if the hang is in MoE layers.
    """
    pipeline, inputs = layer0_only_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# 2. Layer 1 only (first MoE) — expected to HANG
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_layer1_only(layer1_only_model_and_inputs):
    """
    Vision embed + DeepseekV2 setup + layer 1 only (DeepseekV2MoE, 64 experts, top-6).
    Layer 1 is the first MoE layer — suspected hang culprit.
    """
    pipeline, inputs = layer1_only_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# 3. Layers 0+1 combined (dense + MoE) — expected to HANG
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_layer0_and_1(layer0_and_1_model_and_inputs):
    """
    Vision embed + DeepseekV2 setup + layers 0-1 (dense + first MoE).
    Confirms hang with MoE present alongside dense layer.
    """
    pipeline, inputs = layer0_and_1_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
