# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS-20B Multi-Token Generation Example for TT-XLA

This example demonstrates how to run GPT-OSS-20B multi-token generation on Tenstorrent
hardware using tensor parallelism across multiple devices.

Requirements:
- 4+ TT devices (GPT-OSS-20B requires multiple devices due to memory constraints)
- TT-XLA environment activated (source venv/activate)

Usage:
    python examples/pytorch/gpt_oss_20b_simple.py

Note: This example generates multiple tokens autoregressively.
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
    """Main function for GPT-OSS-20B multi-token generation inference."""

    model_name = "openai/gpt-oss-20b"
    max_tokens_to_generate = 15  # Number of tokens to generate
    max_sequence_length = 128  # Pre-allocate sequence to avoid recompilation

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
    config.use_cache = False  # Keep cache disabled as in test framework

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

    # Prepare input with padding to fixed length
    messages = [{"role": "user", "content": "I like taking walks in the"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        max_length=max_sequence_length,
    )

    # Find the actual sequence length (before padding)
    attention_mask = inputs["attention_mask"]
    original_input_length = attention_mask[0].sum().item()

    # Save original input text for display
    original_input_text = tokenizer.decode(
        inputs["input_ids"][0, :original_input_length], skip_special_tokens=True
    )

    print(f"\nInput shape (padded): {inputs['input_ids'].shape}")
    print(f"Original input length: {original_input_length}")
    print(
        f"Input text: {tokenizer.decode(inputs['input_ids'][0, :original_input_length])}"
    )

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

    # Run multi-token generation with pre-padded inputs
    print(f"\nRunning generation for {max_tokens_to_generate} tokens...")
    print("Using pre-padded inputs to avoid recompilation")
    generated_tokens = []
    current_pos = original_input_length

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            # Check we have space
            if current_pos >= max_sequence_length:
                print("Reached maximum sequence length, stopping.")
                break

            # Run forward pass - shape stays constant [1, max_sequence_length]
            outputs = compiled_model(**inputs)

            # Get next token from the last valid position
            logits = outputs.logits.to("cpu")
            next_token_id = logits[0, current_pos - 1].argmax(dim=-1)
            next_token_text = tokenizer.decode(next_token_id)
            generated_tokens.append(next_token_text)

            print(
                f"Step {step + 1}/{max_tokens_to_generate}: Generated token '{next_token_text}'"
            )

            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                print("EOS token generated, stopping early.")
                break

            # Update input_ids and attention_mask in-place (no shape change!)
            inputs["input_ids"][0, current_pos] = next_token_id.to(device)
            inputs["attention_mask"][0, current_pos] = 1

            current_pos += 1

    # Print results
    print(f"\n{'='*80}")
    print("Generation Results:")
    print(f"{'='*80}")
    full_output = original_input_text + "".join(generated_tokens)

    print(f"Input: {original_input_text}")
    print(f"Generated tokens: {''.join(generated_tokens)}")
    print(f"Full output: {full_output}")
    print(f"{'='*80}\n")

    print(
        f"✓ GPT-OSS-20B generation completed successfully! Generated {len(generated_tokens)} tokens."
    )


if __name__ == "__main__":
    # Set device type to TT
    xr.set_device_type("TT")

    main()
