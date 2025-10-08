# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import freeze, unfreeze
from transformers import AutoConfig
from utils.data_utils import compare_results
from utils.flax_utils import *
from utils.torch_utils import *


def run_test(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    config = AutoConfig.from_pretrained(
        model_name,
        # Reduces number of hidden layers for faster testing, but gives gibberish values
        # just for comparison (uncomment line below to allow that)
        # num_hidden_layers=4,
        torch_dtype=torch.float32,
    )
    tokenizer, input_ids, attention_mask = prepare_torch_input(model_name, prompt)

    torch_model = init_torch_model(model_name, config)
    torch_output = run_torch_model(torch_model, input_ids, attention_mask)
    max_len = torch_output.shape[1]

    flax_model, flax_params = init_flax_model(
        config=config,
        batch_size=input_ids.shape[0] * 2,
        max_len=max_len,
        checkpoint_path=None,  # Path to the checkpoint if needed
    )
    input_ids = jnp.array(input_ids.numpy(), dtype=jnp.int32)
    attention_mask = jnp.array(attention_mask.numpy(), dtype=jnp.int32)
    # Duplicate to match batch_size of 2
    # For testing purposes
    input_ids = jnp.repeat(input_ids, 2, axis=0)
    attention_mask = jnp.repeat(attention_mask, 2, axis=0)

    input_ids, attention_mask, position_ids = prepare_flax_input(
        flax_model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_len=max_len,
    )
    print("🔄 Preparing inputs for Flax model...")
    print(f"Attention mask shape: {attention_mask.shape}")
    flax_output = run_flax_model(
        flax_params=flax_params,
        flax_model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )

    print("📦 Flax model output:", flax_output[0], sep="\n")
    print("📦 Torch model output:", torch_output[0], sep="\n")
    print("🈵 Decoding outputs...")
    torch_result = tokenizer.batch_decode(torch_output, skip_special_tokens=False)
    # torch_result = strip_output(torch_result, prompt)
    flax_result = tokenizer.batch_decode(flax_output, skip_special_tokens=False)
    # flax_result = strip_output(flax_result, prompt)

    print("🔍 Comparing outputs...")
    print(compare_results(torch_result[0], flax_result[0]))


if __name__ == "__main__":
    run_test(
        model_name="tiiuae/Falcon3-7B-Instruct",
        prompt="""
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """,
    )
