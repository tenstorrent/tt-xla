# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import jax
import jax.numpy as jnp

from typing import Optional, Union
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig
from model.model_falcon3 import FlaxFalcon3ForCausalLM


def init_flax_model(
    config: AutoConfig,
    batch_size: int,
    max_len: int,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> tuple[FlaxFalcon3ForCausalLM, dict]:
    """
    Initialize the Flax model with its parameters.
    """
    flax_model = FlaxFalcon3ForCausalLM(config)
    flax_params = flax_model.convert_from_hf_weights(
        config=config,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        max_len=max_len,
    )
    return flax_model, flax_params


def prepare_flax_input(
    flax_model: FlaxFalcon3ForCausalLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    max_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Prepare input for the Flax model.
    """
    inputs = flax_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
    )
    return input_ids, inputs["attention_mask"], inputs["position_ids"]


def run_flax_model(
    flax_params: dict,
    flax_model: FlaxFalcon3ForCausalLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
    max_len: int,
) -> jax.Array:
    """
    Run the Flax model with the given proper parameters, input IDs, attention mask and positon IDs.
    """
    print("✍️ Generating Flax Model output...")
    _, seq_len = input_ids.shape
    jit_generate = jax.jit(
        flax_model.generate,
        static_argnames=("max_new_tokens", "return_dict"),
    )

    token_ids = jit_generate(
        params=flax_params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_len - seq_len,
        return_dict=True,
    )
    return token_ids
