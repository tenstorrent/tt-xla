# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS-20B Inference Example for TT-XLA

This example demonstrates how to run GPT-OSS-20B inference on Tenstorrent hardware
using tensor parallelism across multiple devices. This follows the same pattern as
the working test infrastructure.

Requirements:
- 4+ TT devices (GPT-OSS-20B requires multiple devices due to memory constraints)
- TT-XLA environment activated (source venv/activate)

Usage:
    python examples/pytorch/gpt_oss_20b.py

Note: This example performs a single forward pass for demonstration. For production
      use cases with autoregressive generation, see the Llama example pattern.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    """Main function for GPT-OSS-20B single forward pass inference."""

    model_name = "openai/gpt-oss-20b"

    # Verify we have enough devices
    num_devices = xr.global_runtime_device_count()
    if num_devices < 4:
        print(
            f"ERROR: GPT-OSS-20B requires at least 4 devices, but only {num_devices} available"
        )
        return

    print(f"Using {num_devices} devices for tensor parallelism")

    # Setup SPMD for multi-device execution
    print("Setting up SPMD environment...")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    # Create device mesh
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape}")

    # Get XLA device
    device = torch_xla.device()

    # Load model configuration with GPT-OSS specific settings
    print(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.quantization_config["quant_method"] = "none"
    config.use_cache = False  # Single forward pass doesn't need cache

    # Load model with eager attention for torch.compile compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded successfully")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    messages = [{"role": "user", "content": "I like taking walks in the"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    print(f"\nInput shape: {inputs['input_ids'].shape}")
    print(f"Input text: {tokenizer.decode(inputs['input_ids'][0])}")

    # Move model and inputs to device
    print("\nMoving model to device...")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Mark sharding for tensor parallelism
    print("Marking model sharding...")
    for layer in model.model.layers:
        # Self-attention: column-wise sharding for q,k,v; row-wise for o
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

        # MoE MLP components
        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, None))
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", None, None))
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, None))
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", None))

    # Compile model with TT backend
    print("Compiling model with TT backend...")
    compiled_model = torch.compile(model, backend="tt")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = compiled_model(**inputs)

    # Get predictions
    logits = outputs.logits.to("cpu")
    next_token_id = logits[:, -1].argmax(dim=-1)
    predicted_token = tokenizer.decode(next_token_id[0])

    print(f"\n{'='*80}")
    print("Inference Results:")
    print(f"{'='*80}")
    print(
        f"Input: {tokenizer.decode(inputs['input_ids'][0].to('cpu'), skip_special_tokens=True)}"
    )
    print(f"Predicted next token: '{predicted_token}'")
    print(f"{'='*80}\n")

    print("✓ GPT-OSS-20B inference completed successfully!")


if __name__ == "__main__":
    # Set device type to TT
    xr.set_device_type("TT")

    main()
