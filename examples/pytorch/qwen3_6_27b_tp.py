# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.6-27B multi-token generation with tensor parallelism on Tenstorrent hardware.

This script demonstrates autoregressive token generation using the Qwen3.6-27B
model across multiple chips with PyTorch/XLA SPMD tensor parallelism.
No KV cache is used — each forward pass processes the full (padded) sequence
with newly generated tokens appended into the padding slots, keeping tensor
shapes fixed to avoid recompilation.

Architecture: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- Gated DeltaNet: 48 V heads / 16 QK heads, head_dim=128
- Gated Attention: 24 Q heads / 4 KV heads, head_dim=256
- MLP intermediate_size: 17408

Environment variables:
    TTXLA_MAX_NEW_TOKENS: Number of tokens to generate (default: 20)

Usage:
    python examples/pytorch/qwen3_6_27b_tp.py
"""

import math
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
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Monkey patches for graph-break-free compilation on TT hardware.
#
# 1. _update_linear_attn_mask uses torch.all(attention_mask == 1) in a Python
#    if-statement, triggering aten._local_scalar_dense (same root cause as the
#    .tolist() graph breaks in Qwen3.5 — see github.com/tenstorrent/tt-xla/issues/3508).
#
# 2. torch_chunk_gated_delta_rule builds the resolvent matrix (I-A)^{-1}
#    row-by-row in a sequential Python loop (chunk_size-1 = 63 iterations).
#    When TorchDynamo unrolls this, it creates thousands of FX nodes that OOM
#    the host during graph partitioning. We replace it with a parallel Neumann
#    series using the doubling trick:
#      (I-A)^{-1} = (I+A)(I+A²)(I+A⁴)(I+A⁸)(I+A¹⁶)(I+A³²)
#    reducing 63 sequential ops to 11 matrix multiplications.
# =============================================================================


def _patched_update_linear_attn_mask(self, attention_mask, past_key_values):
    """Compilable replacement for Qwen3NextModel._update_linear_attn_mask.

    The original uses `torch.all(attention_mask == 1)` in a Python branch,
    which forces device→host sync and triggers a graph break via
    aten._local_scalar_dense. For non-cached inference we simply return the
    mask as-is — multiplying all-ones by hidden states is a no-op, so the
    result is mathematically identical.
    """
    if past_key_values is not None and past_key_values.has_previous_state():
        return None
    return attention_mask


def _neumann_resolvent(A, n_iters):
    """Compute (I - A)^{-1} for strict lower triangular nilpotent A.

    Uses the factorization:
        (I-A)^{-1} = (I+A)(I+A²)(I+A⁴)...(I+A^{2^{n-1}})

    This replaces 2^n - 1 sequential row-by-row iterations with n squarings
    and n multiplications — fully parallelizable and produces a compact graph.
    """
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    result = I + A
    Ak = A @ A
    for _ in range(n_iters - 1):
        result = result @ (I + Ak)
        Ak = Ak @ Ak
    return result


def _patched_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Compilable replacement for torch_chunk_gated_delta_rule.

    The original builds the resolvent matrix row-by-row in a Python for-loop
    with chunk_size-1 iterations. This version uses the parallel Neumann series
    factorization, reducing graph depth from O(chunk_size) to O(log(chunk_size)).
    """
    from transformers.models.qwen3_next.modeling_qwen3_next import l2norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # Build strict lower triangular matrix A, then compute resolvent (I-A)^{-1}
    # via parallel Neumann series instead of the original 63-iteration loop.
    A = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    n_iters = int(math.ceil(math.log2(chunk_size)))
    attn = _neumann_resolvent(A, n_iters)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
            mask, 0
        )
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_graph_break_patches(model):
    """Apply monkey patches to eliminate graph breaks in Qwen3.6 on TT hardware.

    Patches:
    1. _update_linear_attn_mask: removes torch.all() → Python branch
    2. GatedDeltaNet.chunk_gated_delta_rule: parallel Neumann series resolvent
    """
    model_cls = type(model.model)
    model_cls._update_linear_attn_mask = _patched_update_linear_attn_mask

    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule

    log("Applied graph-break patches (linear_attn_mask + chunk_gated_delta_rule)")


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

    log(f"Loading model weights (bfloat16, low_cpu_mem_usage=True, 8 layers)...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_id)
    # num_layers = 4
    # text_cfg = config.text_config if hasattr(config, "text_config") else config
    # text_cfg.num_hidden_layers = num_layers
    # if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
    #     text_cfg.layer_types = text_cfg.layer_types[:num_layers]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")

    apply_graph_break_patches(model)

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
    max_seq_len = 128
    max_new_tokens = int(os.environ.get("TTXLA_MAX_NEW_TOKENS", "20"))

    prompt = "The capital of France is"
    inputs = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=max_seq_len
    )
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    prompt_len = int(attention_mask.sum(dim=-1).item())

    generated_tokens = []

    log(f"Generating up to {max_new_tokens} tokens (prompt_len={prompt_len}, max_seq_len={max_seq_len})...")
    with torch.no_grad():
        for step in range(max_new_tokens):
            ids_device = input_ids.to(device)
            mask_device = attention_mask.to(device)

            if step == 0:
                log("Running first forward pass (triggers actual compilation)...")
                t0 = time.time()

            outputs = compiled_model(ids_device, attention_mask=mask_device)

            if step == 0:
                log(f"First forward pass completed in {time.time() - t0:.1f}s")
                t_decode = time.time()

            cur_pos = prompt_len + step - 1
            next_token_id = outputs.logits[0, cur_pos, :].argmax(dim=-1)
            token_id = next_token_id.item()
            generated_tokens.append(token_id)

            decoded_so_far = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            log(f"Step {step + 1}: token={token_id} | generated so far: {decoded_so_far!r}")

            if token_id == tokenizer.eos_token_id:
                log("EOS token generated, stopping.")
                break

            if prompt_len + step >= max_seq_len:
                log(f"Reached max sequence length ({max_seq_len}), stopping.")
                break

            input_ids[0, prompt_len + step] = token_id
            attention_mask[0, prompt_len + step] = 1

    total_time = time.time() - t_decode
    num_decode_steps = len(generated_tokens)
    if num_decode_steps > 1:
        tok_per_sec = (num_decode_steps - 1) / total_time
        log(f"Generation complete: {num_decode_steps} tokens in {total_time:.1f}s ({tok_per_sec:.2f} tok/s)")
    else:
        log(f"Generation complete: {num_decode_steps} token in {total_time:.1f}s")

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    print(f"Full: {prompt}{generated_text}")

    return generated_text


if __name__ == "__main__":
    xr.set_device_type("TT")
    qwen3_6_27b_tp()
