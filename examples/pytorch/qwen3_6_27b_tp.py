# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.6-27B inference with tensor parallelism on Tenstorrent hardware.

This script demonstrates running the Qwen3.6-27B model across multiple chips
using PyTorch/XLA SPMD tensor parallelism. The model uses a hybrid architecture
(Gated DeltaNet + Gated Attention) across 64 layers.

Architecture: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- Gated DeltaNet: 48 V heads / 16 QK heads, head_dim=128
- Gated Attention: 24 Q heads / 4 KV heads, head_dim=256
- MLP intermediate_size: 17408

Usage:
    python examples/pytorch/qwen3_6_27b_tp.py
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer


def qwen3_6_27b_tp():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 2, (
        f"This script requires at least 2 devices, but found {num_devices}. "
        f"Use the single-chip script for single-device inference."
    )

    model_id = "Qwen/Qwen3.6-27B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    model.eval()

    # Create device mesh for tensor parallelism
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()
    model = model.to(device)

    # Shard model weights across devices.
    # Qwen3.6-27B has a hybrid architecture: some layers are Gated DeltaNet
    # (linear attention) and some are standard Gated Attention. Both have MLPs.
    # We use hasattr checks to handle both layer types safely.
    shard_specs = {}

    for layer in model.model.layers:
        # MLP sharding (present in all layers): Megatron column/row parallel
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj"):
                shard_specs[mlp.gate_proj.weight] = ("model", None)
            if hasattr(mlp, "up_proj"):
                shard_specs[mlp.up_proj.weight] = ("model", None)
            if hasattr(mlp, "down_proj"):
                shard_specs[mlp.down_proj.weight] = (None, "model")

        # Attention sharding (works for both Gated Attention and DeltaNet
        # layers, as both expose q/k/v/o projection weights in HuggingFace)
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        if hasattr(attn, "q_proj"):
            shard_specs[attn.q_proj.weight] = ("model", None)
        if hasattr(attn, "k_proj"):
            shard_specs[attn.k_proj.weight] = ("model", None)
        if hasattr(attn, "v_proj"):
            shard_specs[attn.v_proj.weight] = ("model", None)
        if hasattr(attn, "o_proj"):
            shard_specs[attn.o_proj.weight] = (None, "model")

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    compiled_model = torch.compile(model, backend="tt")

    # Run inference
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = compiled_model(input_ids, attention_mask=attention_mask)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        decoded = tokenizer.decode(next_token[0])
        print(f"Prompt: {prompt}")
        print(f"Next token: {decoded}")

    return decoded


if __name__ == "__main__":
    xr.set_device_type("TT")
    qwen3_6_27b_tp()
