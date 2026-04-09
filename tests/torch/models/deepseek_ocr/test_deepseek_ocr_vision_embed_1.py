# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for the DeepseekOCR vision-embedding pipeline
(everything inside DeepseekOCRModel.forward BEFORE super().forward()).

Loads the real pretrained model and inputs, wraps the vision + embedding
+ masked-scatter pipeline as an nn.Module, and uses run_op_test to
compare CPU vs TT device.

The forward logic is kept identical to DeepseekOCRModel.forward including
all if-branches, loops, and intermediate lists. Instead of calling
super().forward() (DeepseekV2Model), the module returns the same args
that would be passed to it.


"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


class DeepseekOCRVisionEmbedPipeline(nn.Module):
    """Wraps the DeepseekOCRModel forward logic BEFORE super().forward().

    Exact replica of DeepseekOCRModel.forward with all if-branches, the
    for loop, images_in_this_batch list, idx counter — everything identical
    except the super().forward() call is replaced by returning its args.

    patches, image_ori, images_spatial_crop are passed as forward args
    (not stored on module) so run_op_test moves them to the correct device.
    images list-of-tuples is reconstructed inside forward.
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

        return inputs_embeds, attention_mask, position_ids, past_key_values


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

    pipeline = DeepseekOCRVisionEmbedPipeline(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)

    inputs = [input_ids, patches, image_ori, images_seq_mask, images_spatial_crop]
    return pipeline, inputs


# ---------------------------------------------------------------------------
# Sanity: full vision-embedding pipeline (CPU vs TT)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_vision_embed_pipeline(model_and_inputs):
    """
    Full vision-embedding pipeline (embed + SAM + CLIP + project + scatter)
    on TT device vs CPU.

    This replicates everything inside DeepseekOCRModel.forward before
    super().forward() is called, using real pretrained weights and
    the same sample image input as the whole-model run.

    Expected shapes from CPU log:
      input:  input_ids [1,913], patches [6,3,640,640],
              image_ori [1,3,1024,1024], images_seq_mask [1,913],
              images_spatial_crop [1,2]
      output: (inputs_embeds [1,913,1280], attention_mask, position_ids,
               past_key_values)
    """
    pipeline, inputs = model_and_inputs

    run_op_test(
        pipeline,
        inputs,
        framework=Framework.TORCH,
    )
