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
from typing import List


def setup_spmd():
    """
    Initializes SPMD mode in torch_xla.
    """

    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def setup_model_and_tokenizer(
    model_name: str,
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    model = model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    input_prompt: str,
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
) -> dict:
    """
    Construct inputs including static cache.

    Args:
        input_prompt: Input text prompt
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length

    Returns:
        Dictionary containing input_ids, past_key_values, cache_position, and use_cache
    """
    inputs = tokenizer.encode_plus(
        input_prompt,
        return_tensors="pt",
        truncation=True,
    )

    static_cache: StaticCache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    return input_args


def transfer_to_device(
    model: torch.nn.Module, input_args: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """
    Transfer model and inputs to device.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple of (model, input_args) on device
    """
    input_args["past_key_values"].key_cache = [
        k.to(device) for k in input_args["past_key_values"].key_cache
    ]
    input_args["past_key_values"].value_cache = [
        v.to(device) for v in input_args["past_key_values"].value_cache
    ]
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)

    model = model.to(device)

    return model, input_args


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module, input_args: dict, mesh: Mesh
):
    """
    Mark sharding on inputs and model internals.
    If mark_sharding is not called on a tensor, it is fully replicated across all devices.
        i.e. on cache_positions, input_ids

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        mesh: Device mesh for SPMD operations
    """

    for i, (key, value) in enumerate(
        zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        )
    ):
        xs.mark_sharding(key, mesh, (None, "model", None, None))
        xs.mark_sharding(value, mesh, (None, "model", None, None))

    # Shard model internals
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))


def run_decode_step(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh = None,
    is_spmd: bool = True,
) -> dict:
    """
    Run a single decode step.

    Args:
        compiled_model: Compiled model instance
        input_args: Input arguments dictionary
        tokenizer: Tokenizer instance
        device: Device
        mesh: Device mesh for SPMD operations (optional)
        is_spmd: Whether SPMD mode is enabled

    Returns:
        Updated input_args for next iteration
    """
    output: CausalLMOutputWithPast = compiled_model(**input_args)
    output_logits: torch.Tensor = output.logits.to("cpu")
    output_text = tokenizer.decode(output_logits[:, -1].argmax(dim=-1))

    print(output_text, end="")

    # Update inputs for next iteration
    next_token = output_logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    input_args["input_ids"] = next_token.to(device)

    host_cache_pos = input_args["cache_position"].to("cpu")
    host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
    input_args["cache_position"] = host_cache_pos.to(device)

    # Reapply shardings for static cache if SPMD is enabled
    if is_spmd:
        for i, (key, value) in enumerate(
            zip(
                input_args["past_key_values"].key_cache,
                input_args["past_key_values"].value_cache,
            )
        ):
            xs.mark_sharding(key, mesh, (None, "model", None, None))
            xs.mark_sharding(value, mesh, (None, "model", None, None))

    return input_args


# --------------------------------
# Llama Generation Example
# --------------------------------
def llama():
    # Set up config variables.
    tokens_to_generate: int = 16
    batch_size: int = 1
    max_cache_len: int = 128
    input_prompt: str = "I like taking walks in the"
    model_name: str = "meta-llama/Llama-3.2-3B"

    # Determine if SPMD mode should be enabled, if more than 1 devie is available.
    num_devices = xr.global_runtime_device_count()
    is_spmd: bool = num_devices > 1
    if is_spmd:
        setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Construct inputs including static cache
    input_args = construct_inputs(
        input_prompt, tokenizer, model.config, batch_size, max_cache_len
    )

    # Transfer model and inputs to device
    model, input_args = transfer_to_device(model, input_args, device)

    # Mark sharding on inputs and model internals if SPMD is enabled
    if is_spmd:
        mark_sharding_on_inputs_and_model(model, input_args, mesh)

    # Compile model
    compiled_model = torch.compile(model, backend="tt")

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        for step in range(tokens_to_generate):
            input_args = run_decode_step(
                compiled_model, input_args, tokenizer, device, mesh, is_spmd
            )


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()
