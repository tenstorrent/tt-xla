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
from transformers.cache_utils import StaticCache

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
    config.use_cache = True  # Enable cache for static cache support

    # Increase sliding window size to avoid 128-token restriction
    # Keep layer_types unchanged to preserve trained architecture
    config.sliding_window = (
        config.max_position_embeddings
    )  # Set to max sequence length (131072)
    print(
        f"Modified config: sliding_window increased from 128 to {config.sliding_window} (preserving trained layer types)"
    )

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
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare input with padding to fixed length
    messages = [{"role": "user", "content": "Explain quantum mechanics"}]
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

    # Initialize static cache on CPU (will be transferred to device later)
    batch_size = 1
    max_cache_len = max_sequence_length
    static_cache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    # Create cache position tensor
    cache_position = torch.arange(0, original_input_length)

    # Create 4D causal attention mask for static cache compatibility
    # Shape: (batch_size, num_heads, query_length, key_value_length)
    # Prefill phase: (1, 1, prompt_length, max_cache_size)
    # Values: 0.0 for unmasked positions (can attend), -inf for masked positions (cannot attend)
    # For causal masking: each query position i can only attend to positions 0..i (inclusive)
    full_attention_mask = torch.full(
        (batch_size, 1, original_input_length, max_cache_len),
        float("-inf"),
        dtype=torch.float32,
    )
    # Create lower triangular causal mask for valid input tokens
    # torch.tril creates a lower triangular matrix (1s below diagonal, 0s above)
    causal_mask = torch.tril(
        torch.ones((original_input_length, original_input_length), dtype=torch.float32)
    )
    # Convert: 1 -> 0.0 (unmasked), 0 -> -inf (masked)
    causal_mask = torch.where(causal_mask == 1, 0.0, float("-inf"))
    full_attention_mask[0, 0, :, :original_input_length] = causal_mask
    # Future cache positions remain masked with -inf (already initialized above)

    # Slice input_ids to only include valid tokens (remove padding for prefill)
    # For left-padded sequences, valid tokens are at the end
    inputs["input_ids"] = inputs["input_ids"][:, -original_input_length:]

    # Update inputs dict to include cache
    inputs["past_key_values"] = static_cache
    inputs["cache_position"] = cache_position
    inputs["use_cache"] = True
    inputs["attention_mask"] = full_attention_mask

    # Move model and inputs to device
    print("\nMoving model to TT device...")
    model = model.to(device)

    # Transfer cache to device separately
    for layer in inputs["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)

    # Transfer other inputs to device
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["cache_position"] = inputs["cache_position"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)

    # Mark sharding for tensor parallelism
    print("Marking model sharding...")

    # Mark sharding on cache tensors
    for layer in inputs["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    # Mark sharding on model weights
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

    # Run multi-token generation with static cache
    print(f"\nRunning generation for {max_tokens_to_generate} tokens...")
    print("Using static cache to avoid recompilation")
    generated_tokens = []
    current_pos = original_input_length

    decode_attention_mask = torch.full(
        (batch_size, 1, 1, max_cache_len),
        float("-inf"),
        dtype=torch.float32,
    )
    # TODO: need to mask padding?
    decode_attention_mask = decode_attention_mask.to(device)

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("RUNNING PREFILL")

            # Check we have space
            if current_pos >= max_sequence_length:
                print("Reached maximum sequence length, stopping.")
                break

            # Run forward pass
            outputs = compiled_model(**inputs)

            # Get next token from the last output position
            logits = outputs.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            next_token_text = tokenizer.decode(next_token_id[0])
            generated_tokens.append(next_token_text)

            print(
                f"Step {step + 1}/{max_tokens_to_generate}: Generated token '{next_token_text}'"
            )

            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                print("EOS token generated, stopping early.")
                break

            # Update inputs for next iteration
            # For decode steps, input_ids becomes just the next token [1, 1]
            inputs["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            # Update cache_position to next position
            host_cache_pos = inputs["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            inputs["cache_position"] = host_cache_pos.to(device)

            # Update attention mask for decode step
            # Shape: (batch_size, num_heads, query_length, key_value_length)
            # Decode phase: (1, 1, 1, max_cache_size)
            # During decode, the single new query token can attend to all previous cached tokens
            # Values: 0.0 for unmasked positions, -inf for masked positions
            if step == 0:
                inputs["attention_mask"] = decode_attention_mask

            # inputs["attention_mask"][0, 0, 0, current_pos + 1] = 0.0

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
