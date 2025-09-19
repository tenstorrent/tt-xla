# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence, Tuple
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextModel


class Llama4TextTester(TorchModelTester):
    """Tester for Llama4Vision model."""

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

        # Get the full config but only use text part
        full_config = AutoConfig.from_pretrained(model_name)
        text_config = full_config.text_config

        # Make model as small as possible for testing
        text_config.num_hidden_layers = 1  # Just test one text layer
        text_config.hidden_size = 512  # Reduce size
        text_config.num_attention_heads = 8
        text_config.num_key_value_heads = 8
        text_config.intermediate_size = 1024
        text_config.vocab_size = 1000  # Reduce vocab size

        # CRITICAL: Fix padding_idx to be within vocab_size range
        text_config.pad_token_id = 0  # Set to valid range or None

        # Disable RoPE for all layers to avoid complex tensors
        #text_config.no_rope_layers = [False] * text_config.num_hidden_layers

        # Create text-only model
        self.model = Llama4TextModel(text_config)
        self.model.eval()

        return self.model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # Create dummy image inputs
        # Standard vision input: pixel values with shape [batch, channels, height, width]
        text_config = self.model.config

        batch_size = 1
        seq_len = 8  # Short sequence
        vocab_size = text_config.vocab_size

        # Create dummy token IDs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,  # Disable caching to avoid errors when lowering model
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }

        return inputs
