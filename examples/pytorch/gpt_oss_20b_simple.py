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

# def parse_harmony_output(text: str, tokenizer) -> str:
#     """
#     Parse harmony format output to extract the final response.

#     GPT-OSS uses harmony format with channels:
#     - analysis: internal reasoning
#     - commentary: tool usage
#     - final: the actual response to user

#     We extract and return just the final channel content.
#     """
#     # The model outputs in format: <|channel|>final<|message|>actual response
#     # We want to extract just the "actual response" part

#     # Try to find the final channel message
#     if "<|channel|>final<|message|>" in text:
#         # Extract everything after final channel marker
#         final_start = text.find("<|channel|>final<|message|>") + len("<|channel|>final<|message|>")
#         final_text = text[final_start:]
#         # Remove any end markers
#         if "<|end|>" in final_text:
#             final_text = final_text[:final_text.find("<|end|>")]
#         return final_text.strip()

#     # If no final channel, try to extract from analysis channel as fallback
#     if "<|message|>" in text:
#         # Find the last message content
#         parts = text.split("<|message|>")
#         if len(parts) > 1:
#             response = parts[-1]
#             # Clean up any end markers
#             if "<|end|>" in response:
#                 response = response[:response.find("<|end|>")]
#             return response.strip()

#     # If no structured format found, return the text as-is (decode with skip_special_tokens)
#     # This shouldn't happen with proper harmony format, but provides a fallback
#     return text


def main():
    """Main function for GPT-OSS-20B multi-token generation inference."""

    # Use BF16 version to avoid MXFP4 quantization issues
    # TODO: For now we use unsloth's BF16 version to get around MXFP4 quantization issue and the unintialized weights issue.
    model_name = "unsloth/gpt-oss-20b-BF16"
    max_tokens_to_generate = 50  # Number of tokens to generate
    max_sequence_length = 256  # Pre-allocate sequence to avoid recompilation

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
    # No need to modify quantization_config for BF16 version
    config.use_cache = False  # Keep cache disabled as in test framework

    # Load model with eager attention for torch.compile compatibility
    # Using BF16 version which has properly initialized weights
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

    # Extract just the user's prompt for clean display
    user_prompt = messages[0]["content"]

    print(f"\nInput shape (padded): {inputs['input_ids'].shape}")
    print(f"Original input length: {original_input_length}")
    print(f'User prompt: "{user_prompt}"')

    # Move model and inputs to device
    print("\nMoving model to TT device...")
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

    # Parse the harmony format output to extract clean response
    raw_output = "".join(generated_tokens)
    # parsed_output = parse_harmony_output(raw_output, tokenizer)

    print(f'User prompt: "{user_prompt}"')
    print(f"\nRaw generated tokens ({len(generated_tokens)} tokens):")
    print(f"  {raw_output}")
    # print(f"\nParsed response:")
    # print(f"  {parsed_output}")
    print(f"{'='*80}\n")

    print(
        f"✓ GPT-OSS-20B generation completed successfully! Generated {len(generated_tokens)} tokens."
    )


if __name__ == "__main__":
    # Set device type to TT
    xr.set_device_type("TT")

    main()
