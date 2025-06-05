# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
import numpy as np
import jax.numpy as jnp
import os
from dotenv import load_dotenv
from flax import nnx
from gemma import gm

load_dotenv()

# need a token
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def prepare_output(result):
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>") :].lstrip()

    if result.startswith(prompt):
        result = result[len(prompt) :].lstrip()

    return result


def prepare_pytorch_inputs(model_id: str, messages: list[dict]):
    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    return input_ids, attention_mask, processor


def prepare_jax_inputs(jax_model, pt_input_ids, pt_attention_mask, max_len):
    input_ids = jnp.array(pt_input_ids)
    attention_mask = jnp.array(pt_attention_mask)
    input_ids, attention_mask, position_ids = jax_model.prepare_inputs_for_generation(
        input_ids, max_len, attention_mask
    )
    return input_ids, attention_mask, position_ids


def load_pytorch_model_from_hf(model_id, config):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, config=config, device_map="auto", torch_dtype=torch.float32
    )

    return model


def load_jax_model(config, pt_model):
    model = gm.nn.Gemma3_27B()
    params = gm.ckpts.load_params(
        gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
    )

    return model


def run_pytorch_model(pt_model, pt_input_ids, pt_attention_mask):
    return pt_model.generate(pt_input_ids, attention_mask=pt_attention_mask)


def run_jax_model(
    jax_model, jax_input_ids, jax_attention_mask, jax_position_ids, max_len
):
    _, seq_len = jax_input_ids.shape
    return jax_model.generate(
        input_ids=jax_input_ids,
        attention_mask=jax_attention_mask,
        max_new_tokens=max_len - seq_len,
        position_ids=jax_position_ids,
    )


def compare_outputs(pt_outputs, jax_outputs):
    return np.array(pt_outputs) == np.array(jax_outputs)


def run_comparison_test(model_id: str, messages: list[dict]):
    # Setting up the config for both models
    config = AutoConfig.from_pretrained(model_id)
    config.text_config.num_hidden_layers = 2
    config.vision_config.num_hidden_layers = 2
    config._attn_implementation = "eager"

    pt_input_ids, pt_attention_mask, tokenizer = prepare_pytorch_inputs(
        model_id, messages
    )
    pt_model = load_pytorch_model_from_hf(model_id, config)
    pt_outputs = run_pytorch_model(pt_model, pt_input_ids, pt_attention_mask)

    max_len = pt_outputs[0].shape[0]

    jax_model = load_jax_model(config, pt_model)
    jax_input_ids, jax_attention_mask, jax_position_ids = prepare_jax_inputs(
        jax_model, pt_input_ids, pt_attention_mask, max_len
    )
    jax_outputs = run_jax_model(
        jax_model, jax_input_ids, jax_attention_mask, jax_position_ids, max_len
    )

    result_pt = tokenizer.decode(pt_outputs[0], skip_special_tokens=False)
    result_jax = tokenizer.decode(
        torch.tensor(jax_outputs[0]), skip_special_tokens=False
    )

    result_pt = prepare_output(result_pt)
    result_jax = prepare_output(result_jax)

    print(compare_outputs(result_pt, result_jax))


if __name__ == "__main__":
    model_id = "google/gemma-3-27b-it"
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    run_comparison_test(model_id, messages)
