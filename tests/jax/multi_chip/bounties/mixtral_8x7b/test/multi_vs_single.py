# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
from multichip.multichipmixtral import FlaxMixtralForCausalLM as ShardedModel
from singlechip.flaxmixtral import FlaxMixtralForCausalLM as NotShardedModel
from transformers import AutoConfig


def run_single_chip(input_ids, attention_mask, max_len, config):
    print("Creating Regular model...")
    model = NotShardedModel(config)
    print("Regular model created")
    _, seq_len = input_ids.shape
    input_ids, attention_mask, position_ids = model.prepare_inputs_for_generation(
        input_ids, max_len, attention_mask
    )
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_len - seq_len,
        position_ids=position_ids,
    )
    return out


def run_multi_chip(input_ids, attention_mask, max_len, config):
    print("Creating sharded model...")
    model = ShardedModel(config)
    print("Sharded Model created")
    _, seq_len = input_ids.shape
    (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
    ) = model.prepare_inputs_for_generation(input_ids, max_len, attention_mask)
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        max_new_tokens=max_len - seq_len,
    )


def make_inputs(config):
    batch_size = 4
    seq_len = 20
    num_new_tokens = 10
    max_len = seq_len + num_new_tokens
    input_ids = np.random.randint(config.vocab_size, size=(batch_size, seq_len))
    attention_mask = np.ones_like(input_ids)

    input_ids = jnp.array(input_ids)
    attention_mask = jnp.array(attention_mask)

    return input_ids, attention_mask, max_len


if __name__ == "__main__":
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 4
    config._attn_implementation = "eager"
    config.intermediate_size = 1024

    input_ids, attention_mask, max_len = make_inputs(config)
    single_chip = run_single_chip(input_ids, attention_mask, max_len, config)
    multi_chip = run_multi_chip(input_ids, attention_mask, max_len, config)

    print(single_chip)
    print(multi_chip)
    if np.array_equal(np.array(single_chip), np.array(multi_chip)):
        print("Test successful!")
    else:
        print("Something doesnt work :(")
