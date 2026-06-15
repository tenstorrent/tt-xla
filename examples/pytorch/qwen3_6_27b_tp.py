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
import resource

# Cap virtual memory at 200 GB to prevent OOM from killing the entire machine.
# The process will get a MemoryError instead of starving SSH/system services.
_MEM_LIMIT_GB = int(os.environ.get("TTXLA_MEM_LIMIT_GB", "230"))
_MEM_LIMIT_BYTES = _MEM_LIMIT_GB * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))

import time

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def qwen3_6_27b_tp():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 2, (
        f"This script requires at least 2 devices, but found {num_devices}. "
        f"Use the single-chip script for single-device inference."
    )

    model_id = "Qwen/Qwen3.6-27B"

    log(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log(f"Loading model weights (bfloat16, low_cpu_mem_usage=True)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")

    # Create device mesh for tensor parallelism
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()
    log(f"Moving model to XLA device ({device})...")
    t0 = time.time()
    model = model.to(device)
    log(f"Model moved to device in {time.time() - t0:.1f}s")

    # Shard model weights across devices.
    # Qwen3.6-27B has a hybrid architecture: some layers are Gated DeltaNet
    # (linear attention) and some are standard Gated Attention. Both have MLPs.
    # We use hasattr checks to handle both layer types safely.
    #
    # IMPORTANT: Do NOT shard conv1d/depthwise convolution weights in DeltaNet
    # layers. Sharding the feature dim of a depthwise conv causes
    # feature_group_count mismatch in StableHLO verification
    # (see https://github.com/tenstorrent/tt-xla/issues/3508).
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

        # Skip conv1d weights in DeltaNet layers — sharding depthwise
        # convolutions breaks feature_group_count verification in StableHLO.
        if hasattr(attn, "conv1d"):
            pass  # explicitly do not shard attn.conv1d
        if hasattr(attn, "q_conv1d"):
            pass  # explicitly do not shard
        if hasattr(attn, "k_conv1d"):
            pass  # explicitly do not shard

        if hasattr(attn, "q_proj"):
            shard_specs[attn.q_proj.weight] = ("model", None)
        if hasattr(attn, "k_proj"):
            shard_specs[attn.k_proj.weight] = ("model", None)
        if hasattr(attn, "v_proj"):
            shard_specs[attn.v_proj.weight] = ("model", None)
        if hasattr(attn, "o_proj"):
            shard_specs[attn.o_proj.weight] = (None, "model")

    log(f"Applying sharding annotations ({len(shard_specs)} tensors)...")
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    log("Sharding annotations applied.")

    log("Compiling model with torch.compile(backend='tt')...")
    t0 = time.time()
    compiled_model = torch.compile(model, backend="tt")
    log(f"torch.compile returned in {time.time() - t0:.1f}s (lazy — actual compilation on first forward)")

    # Run inference — pad sequence length to a multiple of 128 for tile alignment.
    # TT hardware operates on 32×32 tiles; seq_len % 128 == 0 avoids misalignment
    # crashes (SIGABRT/SIGSEGV) observed during Qwen3.5 bringup.
    prompt = "The capital of France is"
    inputs = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=128
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    log("Running first forward pass (triggers actual compilation)...")
    t0 = time.time()
    with torch.no_grad():
        outputs = compiled_model(input_ids, attention_mask=attention_mask)
    log(f"First forward pass completed in {time.time() - t0:.1f}s")

    log("Decoding output token...")
    with torch.no_grad():
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        decoded = tokenizer.decode(next_token[0])
        print(f"Prompt: {prompt}")
        print(f"Next token: {decoded}")

    return decoded


if __name__ == "__main__":
    xr.set_device_type("TT")
    qwen3_6_27b_tp()
