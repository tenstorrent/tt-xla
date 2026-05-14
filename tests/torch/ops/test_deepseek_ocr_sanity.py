# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import inspect

from infra import Framework, run_op_test
from tests.runner.requirements import RequirementsManager
import third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader as deepseek_ocr_loader
from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader
from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.modeling_deepseekv2 import (
    _prepare_4d_causal_attention_mask,
)
from utils import Category


class _DeepseekOCRBeforeDecoder(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = causal_lm_model.model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
        return_dict=None,
    ):
        del (
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        sam_model = getattr(self.model, "sam_model", None)
        vision_model = getattr(self.model, "vision_model", None)

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
                        local_features = self.model.projector(local_features)
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1)
                        global_features = torch.cat(
                            (
                                global_features_2[:, 1:],
                                global_features_1.flatten(2).permute(0, 2, 1),
                            ),
                            dim=-1,
                        )
                        global_features = self.model.projector(global_features)
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw**0.5)

                        _2, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2**0.5)

                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                        global_features = global_features.view(h, w, n_dim)

                        global_features = torch.cat(
                            [
                                global_features,
                                self.model.image_newline[None, None, :].expand(
                                    h, 1, n_dim
                                ),
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
                                self.model.image_newline[None, None, :].expand(
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
                                self.model.view_seperator[None, :],
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
                        global_features = self.model.projector(global_features)
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw**0.5)

                        global_features = global_features.view(h, w, n_dim)
                        global_features = torch.cat(
                            [
                                global_features,
                                self.model.image_newline[None, None, :].expand(
                                    h, 1, n_dim
                                ),
                            ],
                            dim=1,
                        )
                        global_features = global_features.view(-1, n_dim)
                        global_local_features = torch.cat(
                            [global_features, self.model.view_seperator[None, :]], dim=0
                        )

                    images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    inputs_embeds[idx].masked_scatter(
                        images_seq_mask[idx].unsqueeze(-1),
                        images_in_this_batch,
                    )

                idx += 1

        return inputs_embeds


class _DeepseekOCRForCausalLMSanity(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = _DeepseekOCRBeforeDecoder(causal_lm_model)

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        return self.model(
            input_ids=input_ids,
            images=[(patches, image_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )


class _DeepseekOCRThroughDecoderLayers(torch.nn.Module):
    def __init__(self, causal_lm_model, num_layers):
        super().__init__()
        self.model = causal_lm_model.model
        self.ocr_model = _DeepseekOCRBeforeDecoder(causal_lm_model)
        self.num_layers = num_layers

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = self.ocr_model(
            input_ids=input_ids,
            images=[(patches, image_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )

        output_attentions = self.model.config.output_attentions
        use_cache = False

        batch_size, seq_length = inputs_embeds.shape[:2]
        position_ids = torch.arange(
            0,
            seq_length,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        position_ids = position_ids.unsqueeze(0)

        attention_mask = _prepare_4d_causal_attention_mask(
            None,
            (batch_size, seq_length),
            inputs_embeds,
            0,
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.model.layers[: self.num_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class _DeepseekOCRThroughDecoderLayer0(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = _DeepseekOCRThroughDecoderLayers(causal_lm_model, num_layers=1)

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        return self.model(
            input_ids=input_ids,
            patches=patches,
            image_ori=image_ori,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )


class _DeepseekOCRThroughDecoderLayers01(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = _DeepseekOCRThroughDecoderLayers(causal_lm_model, num_layers=2)

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        return self.model(
            input_ids=input_ids,
            patches=patches,
            image_ori=image_ori,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )


class _DeepseekOCRThroughAllDecoderLayers(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = _DeepseekOCRThroughDecoderLayers(
            causal_lm_model, num_layers=len(causal_lm_model.model.layers)
        )

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        return self.model(
            input_ids=input_ids,
            patches=patches,
            image_ori=image_ori,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )


class _DeepseekOCRThroughAllDecoderLayersAndNorm(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.decoder = _DeepseekOCRThroughDecoderLayers(
            causal_lm_model, num_layers=len(causal_lm_model.model.layers)
        )

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        hidden_states = self.decoder(
            input_ids=input_ids,
            patches=patches,
            image_ori=image_ori,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )
        return self.decoder.model.norm(hidden_states)


class _DeepseekOCRThroughAllDecoderLayersNormAndLmHead(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.decoder_norm = _DeepseekOCRThroughAllDecoderLayersAndNorm(causal_lm_model)
        self.lm_head = causal_lm_model.lm_head

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        hidden_states = self.decoder_norm(
            input_ids=input_ids,
            patches=patches,
            image_ori=image_ori,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )
        return self.lm_head(hidden_states).float()


class _DeepseekOCRForCausalLMLogitsSanity(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = causal_lm_model

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        outputs = self.model.model(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            images=[(patches, image_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=False,
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        return logits.float()


class _DeepseekOCRForCausalLMFullForwardSanity(torch.nn.Module):
    def __init__(self, causal_lm_model):
        super().__init__()
        self.model = causal_lm_model

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        return self.model(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            use_cache=False,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            images=[(patches, image_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=False,
        )[0]


def _ignore_comparison(device_output, golden_output, args, kwargs):
    del device_output, golden_output, args, kwargs


def _load_deepseek_ocr_model_and_inputs():
    loader_path = inspect.getsourcefile(deepseek_ocr_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        loader = ModelLoader()
        model = loader.load_model(dtype_override=torch.bfloat16)
        inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    patches, image_ori = inputs["images"][0]
    return model, inputs, patches, image_ori


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_embedding_path_before_decoder():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRForCausalLMSanity(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_decoder_layer0_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRThroughDecoderLayer0(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_decoder_layers01_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRThroughDecoderLayers01(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_all_decoder_layers_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRThroughAllDecoderLayers(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_all_decoder_layers_norm_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRThroughAllDecoderLayersAndNorm(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_all_decoder_layers_norm_lm_head_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRThroughAllDecoderLayersNormAndLmHead(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_for_causal_lm_logits_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRForCausalLMLogitsSanity(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_full_forward_perf_sanity():
    model, inputs, patches, image_ori = _load_deepseek_ocr_model_and_inputs()

    run_op_test(
        _DeepseekOCRForCausalLMFullForwardSanity(model),
        [
            inputs["input_ids"],
            patches,
            image_ori,
            inputs["images_seq_mask"],
            inputs["images_spatial_crop"],
        ],
        framework=Framework.TORCH,
        custom_comparator=_ignore_comparison,
    )
