# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test: isolated masked scatter decomposition with exact inputs
from the DeepseekOCR forward pass.

In the model forward, after vision features are computed, the masked
scatter decomp replaces selected positions in inputs_embeds:

  images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
  mask = images_seq_mask[idx].unsqueeze(-1)              # [913, 1] bool
  mask_broad, data = broadcast_tensors(mask, embeds[idx]) # [913, 1280]
  mask_flat = mask_broad.reshape(-1)                      # [1168640] bool
  data_flat = data.reshape(-1)                            # [1168640] bf16
  source_flat = images_in_this_batch.reshape(-1)          # variable bf16
  mask_i = mask_flat.long()                               # [1168640] int64
  source_idx = torch.cumsum(mask_i, 0) - 1                # [1168640] int64
  source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
  gathered = source_flat[source_idx]                      # [1168640] bf16
  result_flat = torch.where(mask_flat, gathered, data_flat) # [1168640] bf16
  result = result_flat.view_as(inputs_embeds[idx])        # [913, 1280] bf16

This test constructs the exact tensors from the real model and runs
the full decomposed masked scatter through run_op_test.

"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


class MaskedScatterDecomp(nn.Module):
    """Full decomposed masked scatter exactly as used in DeepseekOCR."""

    def forward(self, images_seq_mask_row, inputs_embeds_row, images_in_this_batch):
        mask = images_seq_mask_row.unsqueeze(-1)  # [913, 1] bool
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds_row)  # [913, 1280]
        mask_flat = mask_broad.reshape(-1)  # [1168640] bool
        data_flat = data.reshape(-1)  # [1168640] bf16
        source_flat = images_in_this_batch.reshape(-1)  # variable bf16
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        gathered = source_flat[source_idx]
        result_flat = torch.where(mask_flat, gathered, data_flat)
        return result_flat


def _load_inputs():
    """Load model inputs and construct exact masked scatter operands."""
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch import (
        ModelLoader,
    )

    loader = ModelLoader()
    full_model = loader.load_model(dtype_override=torch.bfloat16)
    raw_inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    input_ids = raw_inputs["input_ids"]  # [1, 913]
    images_seq_mask = raw_inputs["images_seq_mask"]  # [1, 913] bool
    images_spatial_crop = raw_inputs["images_spatial_crop"]  # [1, 2]
    patches = raw_inputs["images"][0][0]  # [6, 3, 640, 640]
    image_ori = raw_inputs["images"][0][1]  # [1, 3, 1024, 1024]

    ocr_model = full_model
    embed_tokens = ocr_model.model.embed_tokens
    sam_model = ocr_model.model.sam_model
    vision_model = ocr_model.model.vision_model
    projector = ocr_model.model.projector
    image_newline = ocr_model.model.image_newline
    view_seperator = ocr_model.model.view_seperator

    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids)  # [1, 913, 1280]

        local_features_1 = sam_model(patches)
        local_features_2 = vision_model(patches, local_features_1)
        local_features = torch.cat(
            (
                local_features_2[:, 1:],
                local_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        local_features = projector(local_features)

        global_features_1 = sam_model(image_ori)
        global_features_2 = vision_model(image_ori, global_features_1)
        global_features = torch.cat(
            (
                global_features_2[:, 1:],
                global_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        global_features = projector(global_features)

        _, hw, n_dim = global_features.shape
        h = w = int(hw**0.5)

        _2, hw2, n_dim2 = local_features.shape
        h2 = w2 = int(hw2**0.5)

        crop_shape = images_spatial_crop[0]
        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

        global_features = global_features.view(h, w, n_dim)
        global_features = torch.cat(
            [
                global_features,
                image_newline[None, None, :].expand(h, 1, n_dim),
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
                image_newline[None, None, :].expand(
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
                view_seperator[None, :],
            ],
            dim=0,
        )

        images_in_this_batch = global_local_features

    images_seq_mask_row = images_seq_mask[0]  # [913] bool
    inputs_embeds_row = inputs_embeds[0]  # [913, 1280] bf16

    return images_seq_mask_row, inputs_embeds_row, images_in_this_batch


@pytest.fixture(scope="module")
def model_and_inputs():
    images_seq_mask_row, inputs_embeds_row, images_in_this_batch = _load_inputs()
    pipeline = MaskedScatterDecomp()
    pipeline.eval()
    inputs = [images_seq_mask_row, inputs_embeds_row, images_in_this_batch]
    return pipeline, inputs


@pytest.mark.single_device
def test_deepseek_ocr_masked_scatter_alone(model_and_inputs):
    """
    Isolated decomposed masked scatter with exact inputs from the
    DeepseekOCR vision pipeline.

    Inputs:
      images_seq_mask_row  [913] bool
      inputs_embeds_row    [913, 1280] bf16
      images_in_this_batch [variable, 1280] bf16 (concatenated vision features)
    Output:
      result_flat [1168640] bf16

    If this passes → masked scatter (including cumsum) works fine
    alone, OOM is from combined graph memory pressure.
    If this OOMs → the masked scatter decomp itself is too large
    for the TT device even in isolation.
    """
    pipeline, inputs = model_and_inputs

    run_op_test(
        pipeline,
        inputs,
        framework=Framework.TORCH,
    )
