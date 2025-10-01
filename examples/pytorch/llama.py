# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
import numpy as np
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs


def setup_spmd():
    print("Setting up XLA environment...")
    num_devices = xr.global_runtime_device_count()

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


# --------------------------------
# Llama Generation Example
# --------------------------------
def llama():

    # Must be called at start of program.
    setup_spmd()

    # Set up config variables
    tokens_to_generate = 16
    output_tokens = []
    model_hidden_layers = 28
    batch_size = 1
    max_cache_len = 32

    # Connect the device.
    device = xm.xla_device()
    mesh = create_device_mesh()

    # Instantiate model.
    model_name: str = "meta-llama/Llama-3.2-3B"
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    # Default, controllable hidden layer count.
    model.config.num_hidden_layers = model_hidden_layers

    # Instantiate tokenizer.
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Put it in inference mode.
    model = model.eval()

    # Generate inputs.
    inputs = tokenizer.encode_plus(
        "I like taking walks in the",
        return_tensors="pt",
        truncation=True,
    )

    # Instantiate static cache on host (device instantiation leads to trace of unfusable creation ops.)
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position = torch.arange(0, inputs.input_ids.shape[1])
    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    # Move model and inputs to device.
    static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
    static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)

    model = model.to(device)

    # mark shardings on model and inputs.
    xs.mark_sharding(input_args["input_ids"], mesh, (None, None))
    xs.mark_sharding(input_args["cache_position"], mesh, (None,))

    for i, (key, value) in enumerate(
        zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        )
    ):
        xs.mark_sharding(key, mesh, (None, "model", None, None))
        xs.mark_sharding(value, mesh, (None, "model", None, None))

    # shard model internals
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    model.compile(backend="tt")

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        for step in range(tokens_to_generate):
            output: CausalLMOutputWithPast = model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")
            output_text = tokenizer.decode(output_logits[:, -1].argmax(dim=-1))

            output_tokens.append(output_text)
            print(output_text, end="")

            # Update inputs for next iteration
            next_token = output_logits[:, -1].argmax(dim=-1).unsqueeze(-1)
            input_args["input_ids"] = next_token.to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            # reapply shardings for static cache (i/o inplace mutated tensors since they lose sharding annotations)
            for i, (key, value) in enumerate(
                zip(
                    input_args["past_key_values"].key_cache,
                    input_args["past_key_values"].value_cache,
                )
            ):
                xs.mark_sharding(key, mesh, (None, "model", None, None))
                xs.mark_sharding(value, mesh, (None, "model", None, None))


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()
