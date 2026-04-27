# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity tests using the ORIGINAL flatten-based masked_scatter decomposition.

This is the decomposition used on the main branch. It flattens mask and data
to 1-D and runs cumsum on [S*D] = [913*1280] = [1,168,640] int64 elements.
This passes in isolation but OOMs at ttnn::cumsum when the full model is run
on TT device (issue #3412).

These tests exist to **reproduce** the OOM and confirm that the original
decomposition is the bottleneck, validating the need for the v2 (mul+add)
decomposition introduced in deepseek_ocr_2.

Tests:
  1. Vision embed only          — may OOM at cumsum
  2. Vision embed + pre-decoder — may OOM at cumsum
  3. Vision embed + layer 0     — expected to OOM at cumsum
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)


def _masked_scatter_decomp_original(inputs_embeds_row, mask_1d, source):
    """Original flatten-based masked_scatter decomposition from main branch.

    Flattens mask and data to 1-D, runs cumsum on [S*D] int64 elements.
    This OOMs on TT device when S*D is large (e.g. 913*1280 = 1,168,640).

    See: https://github.com/tenstorrent/tt-xla/issues/3412
    """
    mask = mask_1d.unsqueeze(-1)
    mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds_row)
    mask_flat = mask_broad.reshape(-1)
    data_flat = data.reshape(-1)
    source_flat = source.reshape(-1)
    mask_i = mask_flat.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
    gathered = source_flat[source_idx]
    result_flat = torch.where(mask_flat, gathered, data_flat)
    return result_flat.view_as(inputs_embeds_row)


class DeepseekOCRVisionEmbedPipeline(nn.Module):
    """Vision embed pipeline using original flatten-based decomposition.

    Uses the same masked_scatter decomposition as the main branch, which
    OOMs at cumsum on TT device due to the large flattened tensor size.
    """

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
                    inputs_embeds[idx] = _masked_scatter_decomp_original(
                        inputs_embeds[idx],
                        images_seq_mask[idx],
                        images_in_this_batch,
                    )

                idx += 1

        return inputs_embeds, attention_mask, position_ids, past_key_values


class DeepseekOCRPreDecoderPipeline(nn.Module):
    """OCR forward (original flatten-based decomp) + V2 forward setup."""

    def __init__(self, ocr_model):
        super().__init__()
        self.vision_embed = DeepseekOCRVisionEmbedPipeline(ocr_model)
        self._use_flash_attention_2 = ocr_model.model._use_flash_attention_2

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
        return hidden_states, attention_mask, position_ids


class DeepseekOCRLayerSlicePipeline(nn.Module):
    """OCR forward (original flatten-based decomp) + V2 setup + decoder layers[start:end]."""

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
    """Load pretrained DeepseekOCR model and sample inputs."""
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
    """Extract the flat tensor list used by all pipeline wrappers."""
    return [
        raw_inputs["input_ids"],
        raw_inputs["images"][0][0],   # patches
        raw_inputs["images"][0][1],   # image_ori
        raw_inputs["images_seq_mask"],
        raw_inputs["images_spatial_crop"],
    ]


# ---------------------------------------------------------------------------
# Fixtures (model loaded once per module, shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_model_and_raw_inputs():
    """Load pretrained model + raw inputs once for every test in this module."""
    return _load_model_and_inputs()


@pytest.fixture(scope="module")
def model_and_inputs(full_model_and_raw_inputs):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = DeepseekOCRVisionEmbedPipeline(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


@pytest.fixture(scope="module")
def pre_decoder_model_and_inputs(full_model_and_raw_inputs):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = DeepseekOCRPreDecoderPipeline(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


@pytest.fixture(scope="module")
def layer0_only_model_and_inputs(full_model_and_raw_inputs):
    """Layer 0 only (dense MLP)."""
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = DeepseekOCRLayerSlicePipeline(full_model, 0, 1)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


# ---------------------------------------------------------------------------
# 1. Vision embed only (original flatten-based decomp)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_vision_embed_pipeline(model_and_inputs):
    """
    Vision-embedding pipeline using original flatten-based decomposition.
    cumsum runs on [S*D] = [1,168,640] int64 elements — expected to OOM
    on TT device.
    """
    pipeline, inputs = model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# 2. Vision embed + pre-decoder (pos IDs, attn mask)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_pre_decoder(pre_decoder_model_and_inputs):
    """
    Vision-embedding + V2 forward setup using original flatten-based decomp.
    Expected to OOM at cumsum on TT device.
    """
    pipeline, inputs = pre_decoder_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# 3. Vision embed + pre-decoder + layer 0 only (dense)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_layer0_only(layer0_only_model_and_inputs):
    """
    Vision-embedding + V2 setup + layer 0 (dense MLP) using original
    flatten-based decomp. Expected to OOM at cumsum on TT device.
    """
    pipeline, inputs = layer0_only_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
