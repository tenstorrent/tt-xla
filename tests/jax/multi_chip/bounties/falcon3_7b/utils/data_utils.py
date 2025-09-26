# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import jax.numpy as jnp
from transformers import AutoTokenizer


def torch_to_jnp(tensor):
    """
    Convert a PyTorch tensor to a JAX array.
    """
    if isinstance(tensor, torch.Tensor):
        return jnp.array(tensor.cpu().numpy())
    else:
        return jnp.array(tensor)


def tokenize(model_name, prompt):
    """
    Prepare input for the PyTorch model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = torch_to_jnp(inputs.input_ids)
    attention_mask = torch_to_jnp(inputs.attention_mask)
    return tokenizer, input_ids, attention_mask


def strip_output(result, prompt: str = "") -> str:
    """
    Strip the output of the model to remove any special tokens or leading text.
    """
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>") :].lstrip()
    if result.startswith(prompt):
        result = result[len(prompt) :].lstrip()
    return result


def compare_results(torch_result: str, flax_result: str) -> str:
    print("HF Result:\n", torch_result)
    print("Flax Result:\n", flax_result)
    if torch_result == flax_result:
        return "✅ Outputs match!"
    else:
        return "❌ Outputs do not match!"
