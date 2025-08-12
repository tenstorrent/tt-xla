# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import jax.numpy as jnp
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, FlaxPreTrainedModel


class GPT2Tester(JaxModelTester):
    """Tester for GPT2 for autoregressive text generation."""

    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        super().__init__(comparison_config, run_mode)

    #@override
    def _get_model(self):
        if(self._model_path == "gpt2"):
            #Patch EasyDeL to support TT backend
            from easydel.layers.attention_operator._attention_impl import EasyDeLBackends
            if not hasattr(EasyDeLBackends, 'TT'):
                EasyDeLBackends.TT = EasyDeLBackends.CPU
            #print(f"Available devices before model creation: {jax.local_devices()}")
            #print(f"CPU devices: {jax.devices('cpu')}")
            from easydel import AutoEasyDeLModelForCausalLM
            with jax.default_device(jax.devices("cpu")[0]):
                model = AutoEasyDeLModelForCausalLM.from_pretrained(
                    self._model_path,
                    device=jax.devices("cpu")[0],
                    # Add configuration for static shapes
                    config_kwargs={
                        'max_position_embeddings': 512,  # Fixed sequence length
                        'use_cache': False,  # Disable caching which might use dynamic shapes
                    }
                )
            return model
            
        return FlaxGPT2LMHeadModel.from_pretrained(self._model_path)

    #@override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        prompt = "Today is a beautiful day, and I want to"
        tokens = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=64, truncation=True)
        input_ids = jnp.array(tokens.input_ids)
        return input_ids
