# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Text Encoder (Gemma 3) — standalone bringup script with 4-chip tensor parallelism.

Component: Gemma3ForConditionalGeneration (text-only, vision_config=None)
Memory: 24.4 GiB (bf16) — must be sharded across multiple chips
Sharding: 4-way tensor parallel across bhqb (4 x p150)
  - Per-device: 24.4/4 = 6.1 GiB weights, leaving ~25.9 GiB for activations
  - Attention: 16 heads / 4 = 4 heads/device, 8 KV heads / 4 = 2 KV heads/device (GQA)
  - MLP: gate_proj/up_proj column-parallel, down_proj row-parallel

Architecture: Gemma 3 12B text model
  - 48 layers, hidden_size=3840, intermediate_size=15360
  - 16 attention heads (head_dim=256), 8 KV heads (GQA)
  - vocab_size=262208

Input:  input_ids [B, seq_len], attention_mask [B, seq_len]
Output: hidden_states from all layers (49 total: 1 embedding + 48 layers)
        Each: [B, seq_len, 3840]
"""

import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast


def run_ltx2_text_encoder():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 4, f"Text encoder requires 4 devices, found {num_devices}"

    # Create mesh for tensor parallelism
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()

    # Load pretrained Gemma 3 text encoder from LTX-2
    print("Loading Gemma 3 text encoder from Lightricks/LTX-2...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = model.eval()

    # Load tokenizer
    tokenizer = GemmaTokenizerFast.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="tokenizer",
    )

    # Move model to device
    model = model.to(device)

    # Apply tensor-parallel sharding to the language model layers
    # Gemma3 uses GQA: 16 Q heads, 8 KV heads, head_dim=256
    # With 4 devices: 4 Q heads/device, 2 KV heads/device
    shard_specs = {}
    for layer in model.model.language_model.layers:
        # Attention: column-parallel for Q/K/V, row-parallel for O
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")

        # MLP: column-parallel for gate/up, row-parallel for down
        shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
        shard_specs[layer.mlp.up_proj.weight] = ("model", None)
        shard_specs[layer.mlp.down_proj.weight] = (None, "model")

    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile model
    compiled_model = torch.compile(model, backend="tt")

    # Prepare inputs
    prompt = "A cinematic shot of a futuristic city at sunset with flying vehicles"
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Warm-up pass (compilation)
    print("Text Encoder (Gemma 3, 4-chip TP): warm-up pass (compilation)...")
    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    torch_xla.sync(wait=True)
    hidden_states = output.hidden_states
    print(f"  Number of hidden states: {len(hidden_states)}")
    print(f"  Each hidden state shape: {hidden_states[0].shape}")

    # Timed pass
    print("Text Encoder (Gemma 3, 4-chip TP): timed pass...")
    start = time.time()
    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    hidden_states = output.hidden_states
    print(f"  Number of hidden states: {len(hidden_states)}")
    print(f"  Inference time: {elapsed:.3f}s")

    return hidden_states


if __name__ == "__main__":
    xr.set_device_type("TT")
    run_ltx2_text_encoder()
