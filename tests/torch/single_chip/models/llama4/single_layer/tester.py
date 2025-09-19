# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence, Tuple
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Llama4ForConditionalGeneration, AutoConfig


class Llama4SingleLayerTester(TorchModelTester):
    """Tester for single layer of the Llama4 model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        model_name = "meta-llama/Llama-4-Scout-17B-16E"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        config.vision_config.num_hidden_layers = 1
        config.text_config.num_hidden_layers = 4

        # Disable RoPE for all layers
        config.text_config.no_rope_layers = [
            False
        ] * config.text_config.num_hidden_layers

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            # device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Temporary monkeypatch to avoid complex tensors, need to revisit this
        # def mock_rotary_forward(x, position_ids):
        #     # Return identity - no rotation applied
        #     batch_size, seq_len = position_ids.shape
        #     head_dim = x.shape[-1] // self.model.language_model.model.config.num_attention_heads

        #     # Create a dummy tensor that looks like complex but is just real
        #     # This will be ignored since we disabled RoPE usage in layers
        #     dummy_rotation = torch.ones(
        #         batch_size, seq_len, head_dim,
        #         device=x.device,
        #         dtype=x.dtype
        #     )
        #     return dummy_rotation

        # # Replace the rotary embedding forward method
        # self.model.language_model.model.rotary_emb.forward = mock_rotary_forward

        return self.model


    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(torch.bfloat16)
        # Disable caching to avoid errors when lowering model
        inputs["use_cache"] = False
        return inputs