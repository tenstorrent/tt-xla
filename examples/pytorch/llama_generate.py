# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comment - this does not work. There are many materialization issues and accidental on-device operations that happen using this API.
It's a similar problem where the entire graph trace gets captured and executed, and that ends up including stuff we don't want to include
like the tensor initialization and other small stuff that will fail use_spmd sharding check or run into stuff like this during construction:

Ideally we can just compile the forward pass and call generate on it. That is what this code should be doing, but it seems like more stuff is being captured in the graph.

Does _xla_sync_multi help capture less stuff -> No.


loc("slice.4"): error: Could not apply propagated tensor shardings to tensor dimensions
error: Could not update shapes based on their tensor sharding attributes
2025-09-12 16:25:30.889 (   7.742s) [        F8614000]      module_builder.cc:481    ERR| Failed to run stablehlo pipeline
2025-09-12 16:25:30.890 (   7.742s) [        F8614000]      error_instance.cc:49       1| ErrorInstance::PJRT_Error_Message
2025-09-12 16:25:30.890 (   7.742s) [        F8614000]      error_instance.cc:58       1| ErrorInstance::PJRT_Error_GetCode
2025-09-12 16:25:30.890 (   7.742s) [        F8614000]      error_instance.cc:43       1| ErrorInstance::PJRT_Error_Destroy
[james] bypass sanity checks in _prepare_sepcial_tokens
Traceback (most recent call last):
  File "/localdev/jameszianxu/tt-xla/examples/pytorch/llama_generate.py", line 168, in <module>
    llama_generate()
  File "/localdev/jameszianxu/tt-xla/examples/pytorch/llama_generate.py", line 143, in llama_generate
    output_ids = model.generate(
                 ^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 2600, in generate
    result = self._sample(
             ^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 3537, in _sample
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 1814, in _get_initial_cache_position
    cache_position = cache_position[past_length:]

AND

2025-09-12 16:28:23.098 (   7.605s) [        66FFD640]loaded_executable_insta:99     ERR| Requested number of devices to run the executable on (2) doesn't match the compiler estimated number of devices (1)
2025-09-12 16:28:23.098 (   7.605s) [        66FFD640]      error_instance.cc:49       1| ErrorInstance::PJRT_Error_Message
2025-09-12 16:28:23.098 (   7.605s) [        66FFD640]      error_instance.cc:58       1| ErrorInstance::PJRT_Error_GetCode
2025-09-12 16:28:23.098 (   7.605s) [        66FFD640]      error_instance.cc:43       1| ErrorInstance::PJRT_Error_Destroy
Traceback (most recent call last):
  File "/localdev/jameszianxu/tt-xla/examples/pytorch/llama_generate.py", line 202, in <module>
    xr.set_device_type("TT")
    ^^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/examples/pytorch/llama_generate.py", line 177, in llama_generate
    # Use model.generate() with static cache
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 2378, in generate
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 2159, in _prepare_special_tokens
    and isin_mps_friendly(elements=eos_token_tensor, test_elements=pad_token_tensor).any()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/localdev/jameszianxu/tt-xla/venv/lib/python3.11/site-packages/transformers/pytorch_utils.py", line 341, in isin_mps_friendly
    return torch.isin(elements, test_elements)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from transformers.cache_utils import StaticCache
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
# Llama Generation Example using model.generate()
# --------------------------------
def llama_generate():

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
    prompt = "I like taking walks in the"
    inputs = tokenizer.encode_plus(
        prompt,
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

        # move static cache to device after host-side initialization
        static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
        static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]

    # Move inputs and model to device.
    input_ids = inputs.input_ids.to(device)
    model = model.to(device)

    # mark shard specs for input tensors
    xs.mark_sharding(input_ids, mesh, (None, None))

    # apply shardings to static cache
    for i, (key, value) in enumerate(
        zip(
            static_cache.key_cache,
            static_cache.value_cache,
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

    # Compile model with tt backend
    model.compile(backend="tt")

    with torch.no_grad():
        # Use model.generate() with static cache
        tokens_to_generate = 1
        output_ids = model.generate(
            input_ids,
            past_key_values=static_cache,
            max_new_tokens=tokens_to_generate,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode generated tokens
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_tokens = tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        print(f"Input prompt: {prompt}")
        print(f"Full generated text: {generated_text}")
        print(f"Generated tokens: {new_tokens}")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama_generate()
