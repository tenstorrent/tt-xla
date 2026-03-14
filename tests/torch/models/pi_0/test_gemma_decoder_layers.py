# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity tests for Gemma decoder layers in the PI0 paligemma language model.

"""



import pytest
import torch
from torch import nn
from infra import ComparisonConfig, Framework, run_op_test
from infra.evaluators import AllcloseConfig
from transformers import DynamicCache

from third_party.tt_forge_models.pi_0.pytorch import ModelLoader, ModelVariant

_COMPARISON = ComparisonConfig(
    allclose=AllcloseConfig(enabled=True, rtol=1e-2, atol=1e-2),
)


class GemmaDecoderLayersWrapper(torch.nn.Module):
    """Runs the first N Gemma decoder layers with saved inputs.

    All fixed tensors (causal_mask, position_ids, cache_position,
    position_embeddings) are stored as registered buffers so they move
    to the target device automatically.  A fresh DynamicCache is created
    on each forward call.
    """

    def __init__(self, pi0_model, num_layers, saved_data):
        super().__init__()
        gemma_model = (
            pi0_model.model              # PI0Pytorch
            .paligemma_with_expert       # PaliGemmaWithExpertModel
            .paligemma                   # PaliGemmaForConditionalGeneration
            .language_model              # GemmaModel (via property → self.model.language_model)
        )
        gemma_model.config._attn_implementation = "eager"

        self.decoder_layers = nn.ModuleList(
            list(gemma_model.layers[:num_layers])
        )

        self.register_buffer("causal_mask", saved_data["causal_mask"])
        self.register_buffer("position_ids", saved_data["position_ids"])
        self.register_buffer("cache_position", saved_data["cache_position"])
        self.register_buffer("pos_emb_cos", saved_data["position_embeddings"][0])
        self.register_buffer("pos_emb_sin", saved_data["position_embeddings"][1])

        self._use_cache = saved_data["use_cache"]
        self._output_attentions = saved_data["output_attentions"]

        if "adarms_cond" in saved_data:
            self.register_buffer("adarms_cond", saved_data["adarms_cond"])
        else:
            self.adarms_cond = None

    def forward(self, hidden_states):
        position_embeddings = (self.pos_emb_cos, self.pos_emb_sin)
        past_key_values = DynamicCache()

        for layer in self.decoder_layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=self.causal_mask,
                position_ids=self.position_ids,
                past_key_value=past_key_values,
                output_attentions=self._output_attentions,
                use_cache=self._use_cache,
                cache_position=self.cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=self.adarms_cond,
            )
            hidden_states = layer_outputs[0]

        return hidden_states,past_key_values


def _load_saved_data():
    path = "pi_0_libero_base_saved_inputs/gemma_decoder_layer_inputs.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.single_device
@pytest.mark.parametrize("num_layers", [18])
@pytest.mark.parametrize("variant", [ModelVariant.BASE])
def test_gemma_decoder_layers(variant, num_layers):
    """Run first N Gemma decoder layers and compare CPU vs TT output.

    Parametrized over 18 layers 
    """
    saved_data = _load_saved_data()

    actual_layers = saved_data["num_hidden_layers"]
    if num_layers > actual_layers:
        pytest.skip(f"Model only has {actual_layers} layers, requested {num_layers}")

    loader = ModelLoader(variant)
    model = loader.load_model()

    wrapper = GemmaDecoderLayersWrapper(model, num_layers, saved_data)
    wrapper.eval()

    inputs = [saved_data["hidden_states"]]

    run_op_test(
        wrapper, inputs,
        comparison_config=_COMPARISON,
        framework=Framework.TORCH,
    )
