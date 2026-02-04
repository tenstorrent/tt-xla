# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from third_party.tt_forge_models.stereo.pytorch import ModelLoader,ModelVariant
from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs




class decoderrr_layer(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.decoderr_layer_i = model

    def forward(self, hidden_states,attention_mask,encoder_hidden_states,encoder_attention_mask,layer_head_mask,cross_attn_layer_head_mask,past_key_values,output_attentions,use_cache,cache_position):
        outputs = self.decoderr_layer_i(hidden_states=hidden_states,attention_mask=attention_mask,encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=encoder_attention_mask,layer_head_mask=layer_head_mask,cross_attn_layer_head_mask=cross_attn_layer_head_mask,past_key_values=past_key_values,output_attentions=output_attentions,use_cache=use_cache,cache_position=cache_position)
        return outputs,past_key_values

@pytest.mark.parametrize(
    "variant,layer", # layer where PCC Drop is observed for Past Key Values during whole model run for specific variant
    [
        (ModelVariant.MEDIUM, 3),
        (ModelVariant.LARGE, 2),
    ],
)
def test_decoder_layer(variant,layer):
    loader = ModelLoader(variant)
    model = loader.load_model()
    decoder_layer_module = model.decoder.model.decoder.layers[layer]
    data = torch.load(f"{variant}_variant_decoder_layer_{layer}_input_data.pt", map_location="cpu",weights_only=False)
    run_op_test_with_saved_inputs(
        decoderrr_layer(decoder_layer_module),
        [
            data["hidden_states"],
            data["attention_mask"],
            data["encoder_hidden_states"],
            data["encoder_attention_mask"],
            data["layer_head_mask"],
            data["cross_attn_layer_head_mask"],
            data["past_key_values"],
            data["output_attentions"],
            data["use_cache"],
            data["cache_position"]
        ],
        framework=Framework.TORCH,
    )

