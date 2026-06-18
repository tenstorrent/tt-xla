# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.6-27B autoregressive decoding with tensor parallelism on Tenstorrent hardware.

This script demonstrates token-by-token generation using a static KV cache and
fixed tensor shapes to avoid recompilation between decode steps.

Architecture: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- DeltaNet layers: recurrent state (fixed size, no cache growth)
- Attention layers: static KV cache (pre-allocated to max_cache_len)

Key patterns for avoiding recompilation on TT hardware:
1. StaticCache with fixed max_cache_len — pre-allocates KV buffers
2. Fixed input shape [batch, 1] in decode — no shape changes between steps
3. Explicit position_ids — avoids per-step Python int baked into graph
4. cache_position tracking — tells the cache where to write

Usage:
    python examples/pytorch/qwen3_6_27b_tp_decode.py
"""

import math
import os
import resource

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
from transformers.cache_utils import StaticCache
from tt_torch.transformers_overrides import override_cache_static_layers


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Monkey patches for graph-break-free compilation on TT hardware.
# See github.com/tenstorrent/tt-xla/issues/3508 for background.
# =============================================================================


def _patched_update_linear_attn_mask(self, attention_mask, past_key_values):
    """Compilable replacement for Qwen3NextModel._update_linear_attn_mask.

    The original uses `torch.all(attention_mask == 1)` in a Python branch,
    which forces device→host sync and triggers a graph break via
    aten._local_scalar_dense.
    """
    if past_key_values is not None and past_key_values.has_previous_state():
        return None
    return attention_mask


def _neumann_resolvent(A, n_iters):
    """Compute (I - A)^{-1} for strict lower triangular nilpotent A.

    Uses: (I-A)^{-1} = (I+A)(I+A²)(I+A⁴)...(I+A^{2^{n-1}})
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

    Replaces the 63-iteration sequential loop with parallel Neumann series,
    reducing FX graph depth from O(chunk_size) to O(log(chunk_size)).
    Only used during prefill (seq_len > 1). Decode uses the recurrent path.
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
    """Apply monkey patches to eliminate graph breaks in Qwen3.6 on TT hardware."""
    model_cls = type(model.model)
    model_cls._update_linear_attn_mask = _patched_update_linear_attn_mask

    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule

    log("Applied graph-break patches (linear_attn_mask + chunk_gated_delta_rule)")


def qwen3_6_27b_tp_decode():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 2, (
        f"This script requires at least 2 devices, but found {num_devices}. "
        f"Use the single-chip script for single-device inference."
    )

    # --- Configuration ---
    model_id = "Qwen/Qwen3.6-27B"
    max_cache_len = 256
    max_new_tokens = 50
    batch_size = 1

    # --- Load model and tokenizer ---
    log(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading model weights (bfloat16)...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 4
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")

    apply_graph_break_patches(model)

    # --- Construct inputs with static cache ---
    prompt = "The capital of France is"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        return_attention_mask=True,
    )
    prompt_len = inputs["input_ids"].shape[1]

    # StaticCache pre-allocates KV buffers for attention layers (StaticLayer)
    # and uses LinearAttentionLayer for DeltaNet layers (inherently static —
    # fixed-size recurrent state that doesn't grow with sequence length).
    # Both layer types lazily initialize their buffers during the first forward
    # pass, so they'll be created directly on the XLA device.
    static_cache = StaticCache(
        config=model.config,
        max_cache_len=max_cache_len,
    )
    override_cache_static_layers(static_cache)

    # Pre-allocate attention mask to max_cache_len to avoid shape changes
    # during decode steps. Positions beyond current sequence are masked as 1
    # (will be filled as we generate).
    full_attention_mask = torch.zeros(
        (batch_size, max_cache_len), dtype=inputs["attention_mask"].dtype
    )
    full_attention_mask[:, :prompt_len] = inputs["attention_mask"]

    # cache_position tells the model which positions are being processed
    cache_position = torch.arange(0, prompt_len)

    # Explicit position_ids prevent the model from calling
    # past_key_values.get_seq_length() internally (which bakes a per-step
    # Python int into the graph, causing recompilation every decode step).
    position_ids = cache_position.unsqueeze(0)

    input_args = {
        "input_ids": inputs["input_ids"],
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    # --- Create device mesh ---
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    # --- Transfer to device ---
    device = torch_xla.device()
    log(f"Moving model to XLA device ({device})...")
    t0 = time.time()
    model = model.to(device)

    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["position_ids"] = input_args["position_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    # Cache KV buffers lazily initialize on-device during the first forward pass.
    # However, StaticLayer.cumulative_length is a CPU tensor created at __init__
    # time — move it to device to avoid the "non-XLA tensor" warning.
    for layer in input_args["past_key_values"].layers:
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)

    log(f"Transferred to device in {time.time() - t0:.1f}s")

    # --- Shard model weights ---
    shard_specs = {}
    for layer in model.model.layers:
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj"):
                shard_specs[mlp.gate_proj.weight] = ("model", None)
            if hasattr(mlp, "up_proj"):
                shard_specs[mlp.up_proj.weight] = ("model", None)
            if hasattr(mlp, "down_proj"):
                shard_specs[mlp.down_proj.weight] = (None, "model")

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

    log(f"Applying sharding annotations ({len(shard_specs)} tensors)...")
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    log("Sharding annotations applied.")

    # --- Compile ---
    log("Compiling model with torch.compile(backend='tt')...")
    compiled_model = torch.compile(model, backend="tt")

    # --- Generation loop ---
    generated_tokens = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                log("Running prefill (triggers compilation for prefill graph)...")
                t0 = time.time()

            output = compiled_model(**input_args)

            if step == 0:
                log(f"Prefill completed in {time.time() - t0:.1f}s")
                log("Starting decode...")
                t_decode = time.time()

            # Get next token from last non-padding position (prefill) or
            # position 0 (decode, since input is [batch, 1])
            logits = output.logits
            if step == 0:
                last_real_pos = inputs["attention_mask"].sum(dim=-1).to(device) - 1
                next_token_id = logits[0, last_real_pos[0], :].argmax(dim=-1)
            else:
                next_token_id = logits[:, -1, :].argmax(dim=-1)

            generated_tokens.append(next_token_id.item())

            # Check EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # --- Update inputs for next decode step (fixed shapes) ---
            # Input is just the new token: shape [batch, 1]
            input_args["input_ids"] = next_token_id.unsqueeze(0).unsqueeze(0).to(device)

            # Advance cache_position by 1: shape [1]
            new_cache_pos = torch.tensor([prompt_len + step], dtype=torch.long)
            input_args["cache_position"] = new_cache_pos.to(device)
            input_args["position_ids"] = new_cache_pos.unsqueeze(0).to(device)

            # Update attention mask at the new position
            full_attention_mask[:, prompt_len + step] = 1
            input_args["attention_mask"] = full_attention_mask.to(device)

    if generated_tokens:
        decode_time = time.time() - t_decode
        tokens_per_sec = (len(generated_tokens) - 1) / decode_time if decode_time > 0 else 0
        log(f"Decode completed: {len(generated_tokens)} tokens in {decode_time:.1f}s "
            f"({tokens_per_sec:.1f} tok/s)")

    # --- Print result ---
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

    return generated_text


if __name__ == "__main__":
    xr.set_device_type("TT")
    qwen3_6_27b_tp_decode()
