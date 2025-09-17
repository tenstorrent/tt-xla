# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from model.jax_config import *
from utils.flax_utils import *
from utils.data_utils import *


def prepare_input(flax_model, input_ids, attention_mask, max_len):
    """
    Prepare input for the Flax model.
    """
    input_ids = torch_to_jnp(input_ids)
    attention_mask = torch_to_jnp(attention_mask)
    input_ids = jnp.repeat(input_ids, 2, axis=0)  # Duplicate for batch size of 2
    attention_mask = jnp.repeat(attention_mask, 2, axis=0)
    inputs = flax_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
    )
    return input_ids, inputs["attention_mask"], inputs["position_ids"]


def shard_input(input_ids, attention_mask, position_ids, device_mesh):

    input_ids = with_named_sharding_constraint(input_ids, device_mesh, P("dp", None))
    attention_mask = with_named_sharding_constraint(
        attention_mask, device_mesh, P("dp", None)
    )
    position_ids = with_named_sharding_constraint(
        position_ids, device_mesh, P("dp", None)
    )

    return input_ids, attention_mask, position_ids


def run_model(
    flax_params, flax_model, input_ids, attention_mask, position_ids, max_len
):
    """
    Run the Flax model with the given input IDs and attention mask.
    """
    print("✍️ Generating Flax Model output...")
    _, seq_len = input_ids.shape
    jit_generate = jax.jit(
        flax_model.generate,
        static_argnames=("max_new_tokens", "return_dict"),
    )

    generated_ids = jit_generate(
        params=flax_params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_len - seq_len,
        return_dict=True,
    )
    return generated_ids


def run_test(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    device_mesh = create_device_mesh(dp_size=2, tp_size=4)
    config = AutoConfig.from_pretrained(
        model_name,
        # Reduces number of hidden layers for faster testing, but gives gibberish values
        # just for comparison (uncomment line below to allow that)
        # num_hidden_layers=4,
        torch_dtype=torch.float32,
    )
    tokenizer, input_ids, attention_mask = tokenize(model_name, prompt)

    batch_size, seq_len = input_ids.shape
    max_len = seq_len + 20
    model, params = init_flax_model(
        config=config,
        batch_size=batch_size * 2,
        max_len=max_len,
    )
    partitioning_rules = model.get_partitioning_rules()
    sharded_params = model.shard_parameters(params, device_mesh, partitioning_rules)

    print("🔄 Preparing inputs...")
    input_ids, attention_mask, position_ids = prepare_input(
        flax_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_len=max_len,
    )

    outputs = run_model(
        flax_params=params,
        flax_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )
    print("🔂 Preparing inputs...")

    input_ids, attention_mask, position_ids = shard_input(
        input_ids, attention_mask, position_ids, device_mesh
    )
    sharded_outputs = run_model(
        flax_params=sharded_params,
        flax_model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )

    print("1️⃣  Flax model output:", outputs[0], sep="\n")
    print("🔢  Torch model output:", sharded_outputs[0], sep="\n")
    print("🈵  Decoding outputs...")
    result_single = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    result_multi = tokenizer.batch_decode(sharded_outputs, skip_special_tokens=False)

    print("🔍 Comparing outputs...")
    print(compare_results(result_single[0], result_multi[0]))


if __name__ == "__main__":
    run_test(
        model_name="tiiuae/Falcon3-7B-Instruct",
        prompt="""
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """,
    )
