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

    # Basic XLA configuration
    os.environ[
        "ENABLE_AUTO_PARALLEL"
    ] = "TRUE"  # Enables the auto parallel pass in tt-mlir
    os.environ[
        "CONVERT_SHLO_TO_SHARDY"
    ] = "1"  # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ[
        "MESH_SHAPE"
    ] = f"1,{num_devices}"  # Sets the mesh shape used by the auto parallel pass

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

    setup_spmd()  # must be called @ start of program, crucially before creating device mesh / setting up device.

    # Connect the device.
    device = xm.xla_device()
    mesh = create_device_mesh()

    # Instantiate model.
    model_name: str = "meta-llama/Llama-3.2-3B"
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    model.config.num_hidden_layers = 1
    # Instantiate tokenizer.
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Put it in inference mode
    model = model.eval()

    # Generate inputs.
    inputs = tokenizer.encode_plus(
        "I like taking walks in the",
        return_tensors="pt",
        truncation=True,
    )
    with torch.no_grad():
        # Instantiate static cache on host then transfer it to device to avoid CE creation ops
        batch_size = 1
        max_cache_len = 16
        static_cache: StaticCache = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            # device='xla',  # 'xla' device will create the cache on host then we move it to device
            dtype=torch.bfloat16,
        )

        # move static cache to device after host-side initialization. This gets captured in the compile I think,
        # which breaks stuff up since there's technically a graph break now in the trace, as it looks like these are
        # new different inputs...

        static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
        static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]

        # Experiment - force materialization sync before compilation. FAIL
        # torch_xla.sync()
        # torch_xla._XLAC._xla_sync_multi([*static_cache.key_cache, *static_cache.value_cache], device, wait=False)

        # Experiment - remark static addresses - Fail.
        # for k, v in zip(static_cache.key_cache, static_cache.value_cache):
        #     torch._dynamo.mark_static_address(k)
        #     torch._dynamo.mark_static_address(v)

    # mark shard specs
    cache_position = torch.arange(0, inputs.input_ids.shape[1])
    input_args = {
        "input_ids": inputs.input_ids.to(device),
        "past_key_values": static_cache,
        "cache_position": cache_position.to(device),
        "use_cache": True,
    }

    xs.mark_sharding(input_args["input_ids"], mesh, (None, None))
    xs.mark_sharding(input_args["cache_position"], mesh, (None,))

    # apply shardings
    for i, (key, value) in enumerate(
        zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        )
    ):
        xs.mark_sharding(key, mesh, (None, "model", None, None))
        xs.mark_sharding(value, mesh, (None, "model", None, None))

    # Move inputs and model to device.
    model = model.to(device)

    # shard model internals
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    # Run model (with no gradient calculation since we only need inference).
    tokens_to_generate = 2

    output_tokens = []

    model.compile(backend="tt")

    with torch.no_grad():
        # Custom generation loop impl
        for step in range(tokens_to_generate):
            output: CausalLMOutputWithPast = model(**input_args)
            print(output)
            output_logits: torch.Tensor = output.logits.to("cpu")
            output_text = tokenizer.decode(output_logits[:, -1].argmax(dim=-1))

            output_tokens.append(output_text)
            print("Generated token:", output_text)

            # Update inputs for next iteration
            next_token = output_logits[:, -1].argmax(dim=-1).unsqueeze(-1)
            input_args["input_ids"] = next_token.to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

    print("output tokens:", output_tokens)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()
