# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0



import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)

@staticmethod
# @torch.compiler.disable
def _masked_scatter_decomp(inputs_embeds_row, mask_1d, source):
    """Decomposed masked_scatter_ via row-level cumsum + 2D gather.

    Decorated with @torch.compiler.disable so this block runs eagerly
    on CPU and is excluded from XLA tracing, avoiding TT op accuracy
    issues in cumsum/gather/where while keeping the rest of the model
    compiled for TT device.

    See: https://github.com/tenstorrent/tt-xla/issues/3316
            https://github.com/tenstorrent/tt-xla/issues/3412
    """
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)
    gathered_rows = torch.gather(source, 0, source_idx_2d)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)

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
                    inputs_embeds[idx] = _masked_scatter_decomp(
                        inputs_embeds[idx],
                        images_seq_mask[idx],
                        images_in_this_batch,
                    )
                    # images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    # mask = images_seq_mask[idx].unsqueeze(-1)
                    # mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds[idx])
                    # mask_flat = mask_broad.reshape(-1)
                    # data_flat = data.reshape(-1)
                    # source_flat = images_in_this_batch.reshape(-1)
                    # mask_i = mask_flat.long()
                    # source_idx = torch.cumsum(mask_i, 0) - 1
                    # source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
                    # gathered = source_flat[source_idx]
                    # result_flat = torch.where(mask_flat, gathered, data_flat)
                    # inputs_embeds[idx] = result_flat.view_as(inputs_embeds[idx])

                idx += 1

        return inputs_embeds, attention_mask, position_ids, past_key_values


class DeepseekOCRPreDecoderPipeline(nn.Module):
    """OCR forward (vision embed with new masked scatter decomp) + V2 forward
    setup before decoder layers start.

    Covers: token embedding, SAM + CLIP vision encoding, masked scatter,
    position IDs, and 4-D causal attention mask preparation.
    Returns the tensors that would be fed into the decoder layer loop.
    """

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


class DeepseekOCRSingleDecoderLayerPipeline(nn.Module):
    """OCR forward (vision embed with new masked scatter decomp) + V2 setup
    + only the FIRST decoder layer.

    If this completes but the full 30-layer pass hangs, the issue is with
    stacking multiple decoder layers, not with a single layer.
    """

    def __init__(self, ocr_model):
        super().__init__()
        self.vision_embed = DeepseekOCRVisionEmbedPipeline(ocr_model)
        self._use_flash_attention_2 = ocr_model.model._use_flash_attention_2
        self.first_layer = ocr_model.model.layers[0]

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
        layer_outputs = self.first_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
        return (hidden_states,)


class DeepseekOCRFullSinglePassPipeline(nn.Module):
    """Full model single forward pass: OCR forward (vision embed with new
    masked scatter decomp) + all V2 decoder layers + RMS norm.

    Single iteration only — no autoregressive generation loop.
    Tests whether one pass through the full model hangs.
    """

    def __init__(self, ocr_model):
        super().__init__()
        self.vision_embed = DeepseekOCRVisionEmbedPipeline(ocr_model)
        self._use_flash_attention_2 = ocr_model.model._use_flash_attention_2
        self.layers = ocr_model.model.layers
        self.norm = ocr_model.model.norm

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

        hidden_states = self.norm(hidden_states)
        return (hidden_states,)


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
def single_decoder_layer_model_and_inputs(full_model_and_raw_inputs):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = DeepseekOCRSingleDecoderLayerPipeline(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


@pytest.fixture(scope="module")
def full_single_pass_model_and_inputs(full_model_and_raw_inputs):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = DeepseekOCRFullSinglePassPipeline(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


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
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Sanity: OCR forward + V2 setup before decoder layers (CPU vs TT)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_pre_decoder(pre_decoder_model_and_inputs):
    """
    Vision-embedding pipeline + V2 forward setup (position IDs, 4-D causal
    attention mask) — everything up to but NOT including the decoder layer
    loop.  If this hangs, the attention-mask / position-id computation on
    TT device is the culprit.
    """
    pipeline, inputs = pre_decoder_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Sanity: OCR forward + V2 setup + 1 decoder layer (CPU vs TT)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_single_decoder_layer(single_decoder_layer_model_and_inputs):
    """
    Vision-embedding pipeline + V2 setup + first decoder layer only.
    If this completes but the full 30-layer pass hangs, the issue is with
    stacking multiple decoder layers (e.g. growing memory / graph size),
    not with a single decoder layer in isolation.
    """
    pipeline, inputs = single_decoder_layer_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Sanity: full model single forward pass (CPU vs TT)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_deepseek_ocr_full_single_pass(full_single_pass_model_and_inputs):
    """
    Full model: OCR forward + all 30 decoder layers + RMS norm.
    Single forward pass only (no autoregressive generation loop).
    If this completes but the full generation hangs, the issue is in the
    multi-iteration autoregressive loop, not in a single pass through the
    decoder stack.
    """
    pipeline, inputs = full_single_pass_model_and_inputs
    run_op_test(pipeline, inputs, framework=Framework.TORCH)