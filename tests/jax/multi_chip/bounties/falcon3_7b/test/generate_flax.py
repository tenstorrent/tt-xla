# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils.flax_utils import *

MODEL_NAME = "tiiuae/Falcon3-7B-Instruct"
EXAMPLE_PROMPT = """
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """


def main(model_name: str, prompt: str):
    # Example usage
    config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=28,  # can be reduced for faster runtime
        torch_dtype=torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size, seq_len = inputs.input_ids.shape
    max_len = seq_len + 20

    flax_model, flax_params = init_flax_model(config, batch_size, max_len)

    input_ids, attention_mask, position_ids = prepare_flax_input(
        flax_model, inputs.input_ids, inputs.attention_mask, max_len
    )

    generated_ids = run_flax_model(
        flax_params, flax_model, input_ids, attention_mask, position_ids, max_len
    )

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Generated sequence: ", output[0].strip())


if __name__ == "__main__":
    main(model_name=MODEL_NAME, prompt=EXAMPLE_PROMPT)
