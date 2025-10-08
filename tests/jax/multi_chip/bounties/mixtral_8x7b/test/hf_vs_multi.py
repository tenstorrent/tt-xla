# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from dotenv import load_dotenv
from flax import nnx
from huggingface_hub import login
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

load_dotenv()

from multichip.multichipmixtral import FlaxMixtralForCausalLM

# need a token
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def prepare_output(result):
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>") :].lstrip()

    if result.startswith(prompt):
        result = result[len(prompt) :].lstrip()

    return result


def prepare_pytorch_inputs(model_id: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    return input_ids, attention_mask, tokenizer


def prepare_jax_inputs(jax_model, pt_input_ids, pt_attention_mask, max_len):
    input_ids = jnp.array(pt_input_ids)
    attention_mask = jnp.array(pt_attention_mask)
    (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
    ) = jax_model.prepare_inputs_for_generation(input_ids, max_len, attention_mask)
    return input_ids, attention_mask, position_ids, past_key_values


def load_pytorch_model_from_hf(model_id, config):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, device_map="auto", torch_dtype=torch.float32
    )

    return model


def run_pytorch_model(pt_model, pt_input_ids, pt_attention_mask):
    return pt_model.generate(pt_input_ids, attention_mask=pt_attention_mask)


def load_jax_model(config, pt_model):
    jax_model = FlaxMixtralForCausalLM(config)
    state = nnx.state(jax_model)
    new_state = FlaxMixtralForCausalLM.load_hf_params(state, pt_model, config)
    nnx.update(jax_model, new_state)
    return jax_model


def run_jax_model(
    jax_model,
    jax_input_ids,
    jax_attention_mask,
    jax_position_ids,
    max_len,
    past_key_values,
):
    _, seq_len = jax_input_ids.shape
    out = jax_model.generate(
        input_ids=jax_input_ids,
        attention_mask=jax_attention_mask,
        max_new_tokens=max_len - seq_len,
        past_key_values=past_key_values,
        position_ids=jax_position_ids,
    )
    return jax.device_get(out)


def compare_outputs(pt_outputs, jax_outputs):
    return np.array(pt_outputs) == np.array(jax_outputs)


def run_comparison_test(model_id: str, prompt: str):
    # Setting up the config for both models
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2
    config._attn_implementation = "eager"

    pt_input_ids, pt_attention_mask, tokenizer = prepare_pytorch_inputs(
        model_id, prompt
    )
    pt_model = load_pytorch_model_from_hf(model_id, config)
    pt_outputs = run_pytorch_model(pt_model, pt_input_ids, pt_attention_mask)

    max_len = pt_outputs[0].shape[0]

    jax_model = load_jax_model(config, pt_model)

    (
        jax_input_ids,
        jax_attention_mask,
        jax_position_ids,
        past_key_values,
    ) = prepare_jax_inputs(jax_model, pt_input_ids, pt_attention_mask, max_len)

    jax_outputs = run_jax_model(
        jax_model,
        jax_input_ids,
        jax_attention_mask,
        jax_position_ids,
        max_len,
        past_key_values,
    )

    result_pt = tokenizer.decode(pt_outputs[0], skip_special_tokens=False)
    result_jax = tokenizer.decode(
        torch.tensor(jax_outputs[0]), skip_special_tokens=False
    )
    result_pt = prepare_output(result_pt)
    result_jax = prepare_output(result_jax)

    print(compare_outputs(result_pt, result_jax))


if __name__ == "__main__":
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    prompt = "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
    "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n"
    "F: #### 18 \n\n"
    "Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\n"
    "A: The cost of the house and repairs came out to 80,000 + 50,000 = $<<80000+50000=130000>>130,000\n"
    "He increased the value of the house by 80,000 * 1.5 = <<80000*1.5=120000>>120,000\n"
    "So the new value of the house is 120,000 + 80,000 = $<<120000+80000=200000>>200,000\n"
    "So he made a profit of 200,000 - 130,000 = $<<200000-130000=70000>>70,000\n"
    "F: #### 70000\n\n"
    "Q: A bumper car rink has 12 red cars. They have 2 fewer green cars than they have red cars. They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars.  If the rink has 75 cars in total how many yellow cars do they have?\n"
    "A:"
    run_comparison_test(model_id, prompt)
