import argparse
import os
from typing import List

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import (
    Cache,
    LinearAttentionLayer,
    StaticCache,
    StaticLayer,
    StaticSlidingWindowLayer,
)

MODEL_ID = "google/gemma-4-31B-it"

DEFAULT_PROMPTS = [
    "I like taking walks in the",
    "My name is",
    "My favorite color is",
    "Cheese is an excellent",
    "While ham sandwiches are great, I prefer",
    "My bicycle is broken, and",
    "The best way to start my day is with",
    "I love to",
]


def _build_static_cache(config, max_cache_len):
    """Build StaticCache manually, working around transformers [:-0] bug for num_kv_shared_layers=0."""
    config = config.get_text_config(decoder=True)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        if getattr(config, "sliding_window", None) is not None:
            layer_types = ["sliding_attention"] * config.num_hidden_layers
        elif getattr(config, "attention_chunk_size", None) is not None:
            layer_types = ["chunked_attention"] * config.num_hidden_layers
        else:
            layer_types = ["full_attention"] * config.num_hidden_layers
    nksl = getattr(config, "num_kv_shared_layers", 0)
    if nksl and nksl > 0:
        layer_types = layer_types[:-nksl]
    layers = []
    for lt in layer_types:
        if lt == "sliding_attention":
            layers.append(StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.sliding_window))
        elif lt == "chunked_attention":
            layers.append(StaticSlidingWindowLayer(max_cache_len=max_cache_len, sliding_window=config.attention_chunk_size))
        elif lt in ("mamba", "conv", "linear_attention", "moe"):
            layers.append(LinearAttentionLayer())
        else:
            layers.append(StaticLayer(max_cache_len=max_cache_len))
    cache = Cache.__new__(StaticCache)
    Cache.__init__(cache, layers=layers)
    return cache


def setup_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def create_device_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def setup_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    )
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _get_per_layer_cache_config(text_config):
    """Gemma 4 has different head_dim and num_kv_heads for sliding vs full attention layers."""
    num_heads = []
    head_dims = []
    for lt in text_config.layer_types:
        if lt == "full_attention":
            num_heads.append(text_config.num_global_key_value_heads)
            head_dims.append(text_config.global_head_dim)
        else:
            num_heads.append(text_config.num_key_value_heads)
            head_dims.append(text_config.head_dim)
    return num_heads, head_dims


def construct_inputs(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
) -> dict:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    input_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "use_cache": False,
    }

    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Attention mask shape: {inputs.attention_mask.shape}")

    return input_args


def transfer_to_device(model, input_args, device):
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    model = model.to(device)
    return model, input_args


def mark_sharding(model, input_args, mesh):
    # Gemma4ForConditionalGeneration: model.model.language_model.layers[...]
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for layer in text_model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        # Full attention layers use k_eq_v (no v_proj)
        if layer.self_attn.v_proj is not None:
            xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))


def run_prefill(compiled_model, input_args, tokenizer, prompts):
    num_users = input_args["input_ids"].shape[0]

    print("RUNNING PREFILL (single forward pass, no cache)")
    with torch.no_grad():
        output = compiled_model(**input_args)

    output_logits = output.logits.to("cpu")
    next_token_id = output_logits[:, -1].argmax(dim=-1)

    print(f"Logits shape: {output.logits.shape}")
    print()
    for i in range(num_users):
        next_text = tokenizer.decode(next_token_id[i])
        print(f"Result {i}: {prompts[i]}{next_text}")
    print()


def main():
    max_cache_len = 128
    batch_size = 8

    num_devices = xr.global_runtime_device_count()
    is_spmd = num_devices > 1
    if is_spmd:
        setup_spmd()

    device = torch_xla.device()
    mesh = create_device_mesh()

    model, tokenizer = setup_model_and_tokenizer()

    prompts = DEFAULT_PROMPTS[:batch_size]
    input_args = construct_inputs(
        prompts, tokenizer, model.config, batch_size, max_cache_len
    )

    model, input_args = transfer_to_device(model, input_args, device)

    if is_spmd:
        mark_sharding(model, input_args, mesh)

    compiled_model = torch.compile(model, backend="tt")

    run_prefill(compiled_model, input_args, tokenizer, prompts)


if __name__ == "__main__":
    xr.set_device_type("TT")
    main()
