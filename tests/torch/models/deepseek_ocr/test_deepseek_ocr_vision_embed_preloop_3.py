# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for DeepseekOCR: full DeepseekOCRModel.forward vision pipeline
PLUS DeepseekV2Model.forward steps up to just before the decoder layer loop.

This extends test_deepseek_ocr_vision_embed_posids.py by continuing through
attention_mask preparation and hidden_states initialization — stopping
right before ``for decoder_layer in self.layers:``.

Steps replicated:
  DeepseekOCRModel.forward:
    1. embed input_ids
    2. SAM + CLIP vision pipeline
    3. masked scatter into inputs_embeds
  DeepseekV2Model.forward (up to decoder loop):
    4. resolve output_attentions, output_hidden_states, use_cache, return_dict
    5. get batch_size, seq_length
    6. compute past_key_values_length
    7. create position_ids via torch.arange
    8. _prepare_4d_causal_attention_mask
    9. hidden_states = inputs_embeds


"""

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from infra import Framework, run_op_test


class DeepseekOCRVisionEmbedPreloopPipeline(nn.Module):
    """DeepseekOCRModel.forward + DeepseekV2Model.forward up to decoder loop.

    Exact replica of the whole-model forward path, stopping just before
    ``for decoder_layer in self.layers:``.

    patches, image_ori, images_spatial_crop are passed as forward args
    so run_op_test moves them to the correct device.
    """

    def __init__(self, ocr_model):
        super().__init__()
        self.embed_tokens = ocr_model.model.embed_tokens
        self.sam_model = ocr_model.model.sam_model
        self.vision_model = ocr_model.model.vision_model
        self.projector = ocr_model.model.projector
        self.image_newline = ocr_model.model.image_newline
        self.view_seperator = ocr_model.model.view_seperator
        self._config = ocr_model.model.config
        self._use_flash_attention_2 = (
            ocr_model.model.config._attn_implementation == "flash_attention_2"
        )

    def forward(
        self,
        input_ids,
        patches,
        image_ori,
        images_seq_mask,
        images_spatial_crop,
    ):
        # ===================================================================
        # DeepseekOCRModel.forward — vision pipeline
        # ===================================================================
        attention_mask = None
        position_ids = None
        past_key_values = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None

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
                    mask = images_seq_mask[idx].unsqueeze(-1)
                    mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds[idx])
                    mask_flat = mask_broad.reshape(-1)
                    data_flat = data.reshape(-1)
                    source_flat = images_in_this_batch.reshape(-1)
                    mask_i = mask_flat.long()
                    source_idx = torch.cumsum(mask_i, 0) - 1
                    source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
                    gathered = source_flat[source_idx]
                    result_flat = torch.where(mask_flat, gathered, data_flat)
                    inputs_embeds[idx] = result_flat.view_as(inputs_embeds[idx])

                idx += 1

        # ===================================================================
        # DeepseekV2Model.forward — up to just before decoder layer loop
        # ===================================================================
        input_ids = None

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self._config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self._config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self._config.use_cache

        return_dict = (
            return_dict
            if return_dict is not None
            else self._config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values
                )
            past_key_values_length = past_key_values.get_usable_length(
                seq_length
            )

        if position_ids is None:
            device = (
                input_ids.device
                if input_ids is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        return hidden_states, attention_mask, position_ids


def _load_model_and_inputs():
    """Load pretrained DeepseekOCR model and sample inputs."""
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch import (
        ModelLoader,
    )

    loader = ModelLoader()
    full_model = loader.load_model(dtype_override=torch.bfloat16)
    raw_inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    return full_model, raw_inputs


@pytest.fixture(scope="module")
def model_and_inputs():
    full_model, raw_inputs = _load_model_and_inputs()

    input_ids = raw_inputs["input_ids"]
    images_seq_mask = raw_inputs["images_seq_mask"]
    images_spatial_crop = raw_inputs["images_spatial_crop"]
    patches = raw_inputs["images"][0][0]
    image_ori = raw_inputs["images"][0][1]

    pipeline = DeepseekOCRVisionEmbedPreloopPipeline(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)

    inputs = [input_ids, patches, image_ori, images_seq_mask, images_spatial_crop]
    return pipeline, inputs


# ---------------------------------------------------------------------------
# Sanity: vision pipeline + V2 forward through pre-loop init (CPU vs TT)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_vision_embed_preloop(model_and_inputs):
    """
    Full vision-embedding pipeline + DeepseekV2Model.forward through
    attention_mask prep and hidden_states init, stopping just before
    the decoder layer loop.

    Expected shapes from CPU log:
      input:  input_ids [1,913], patches [6,3,640,640],
              image_ori [1,3,1024,1024], images_seq_mask [1,913],
              images_spatial_crop [1,2]
      output: (hidden_states [1,913,1280],
               attention_mask [1,1,913,913],
               position_ids [1,913])
    """
    pipeline, inputs = model_and_inputs

    run_op_test(
        pipeline,
        inputs,
        framework=Framework.TORCH,
    )
