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
from tt_torch.transformers_overrides import TTStaticLayer, override_cache_static_layers


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


def _depthwise_conv1d_as_matmul_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    """Replace depthwise conv1d in decode path with element-wise multiply + sum.

    The conv_state stores the last conv_kernel_size tokens, so after concatenation
    with the new token(s), the input has length (state_len + seq_len). The depthwise
    conv1d (groups=C, kernel_size=K, padding=0) slides over this producing
    (state_len + seq_len - K + 1) outputs; we take the last seq_len.

    For decode (seq_len=1, state_len=K=4): input is [B, C, 5], conv produces [B, C, 2],
    we take the last 1. The last output position is the dot product of weight with the
    last K elements of the concatenated input.
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    K = weight.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])

    # For each output position t in [0, seq_len):
    #   out[b, c, t] = sum_k weight[c, k] * input[b, c, state_len - K + 1 + t + k]
    # For seq_len=1: just dot the weight with input[:, :, -K:]
    # For general seq_len: unroll (same as prefill logic but on the short concatenated input)
    n_out = state_len + seq_len - K + 1
    if seq_len == 1:
        out = (hidden_states_new[:, :, -K:] * weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
    else:
        out = torch.zeros(
            hidden_states_new.shape[0], hidden_size, n_out,
            dtype=hidden_states_new.dtype, device=hidden_states_new.device,
        )
        for k in range(K):
            out = out + hidden_states_new[:, :, k : k + n_out] * weight[:, k].unsqueeze(0).unsqueeze(-1)
        out = out[:, :, -seq_len:]

    if bias is not None:
        out = out + bias.unsqueeze(0).unsqueeze(-1)
    out = F.silu(out)
    out = out.to(hidden_states.dtype)
    return out


def _depthwise_conv1d_as_matmul_prefill(conv1d_module, mixed_qkv, seq_len):
    """Replace depthwise conv1d in prefill path with unrolled shifted multiplies.

    nn.Conv1d(padding=K-1) pads both sides by K-1, then we slice [:seq_len].
    This is equivalent to left-padding by K-1 and computing cross-correlation:
        y[t] = w[0]*x[t-(K-1)] + w[1]*x[t-(K-2)] + ... + w[K-1]*x[t]
    (with x[i]=0 for i<0)
    """
    weight = conv1d_module.weight.squeeze(1)  # [C, K]
    bias = conv1d_module.bias
    K = weight.shape[-1]

    x_padded = F.pad(mixed_qkv, (K - 1, 0))  # [B, C, L + K - 1], causal left-pad
    out = torch.zeros_like(mixed_qkv[:, :, :seq_len])
    for k in range(K):
        # x_padded[:, :, k : k+seq_len] at position t gives x_padded[k+t] = x[t+k-(K-1)]
        # so out[t] += w[k] * x[t + k - (K-1)], matching F.conv1d cross-correlation
        out = out + x_padded[:, :, k : k + seq_len] * weight[:, k].unsqueeze(0).unsqueeze(-1)
    if bias is not None:
        out = out + bias.unsqueeze(0).unsqueeze(-1)
    return F.silu(out)


def _patch_conv1d_forward(layer):
    """Monkey-patch a GatedDeltaNet layer to replace conv1d with matmul equivalents.

    Supports both Qwen3_5GatedDeltaNet (in_proj_qkv + in_proj_z/b/a)
    and Qwen3NextGatedDeltaNet (in_proj_qkvz + in_proj_ba).
    """
    is_qwen3_5 = hasattr(layer, "in_proj_qkv")

    def patched_forward(hidden_states, cache_params=None, attention_mask=None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(layer.layer_idx) and seq_len == 1
        )

        if use_precomputed_states:
            conv_state = cache_params.layers[layer.layer_idx].conv_states
            recurrent_state = cache_params.layers[layer.layer_idx].recurrent_states

        if is_qwen3_5:
            mixed_qkv = layer.in_proj_qkv(hidden_states)
            mixed_qkv = mixed_qkv.transpose(1, 2)
            z = layer.in_proj_z(hidden_states)
            z = z.reshape(batch_size, seq_len, -1, layer.head_v_dim)
            b = layer.in_proj_b(hidden_states)
            a = layer.in_proj_a(hidden_states)
        else:
            projected_states_qkvz = layer.in_proj_qkvz(hidden_states)
            projected_states_ba = layer.in_proj_ba(hidden_states)
            query, key, value, z, b, a = layer.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
            query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
            mixed_qkv = torch.cat((query, key, value), dim=-1)
            mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            mixed_qkv = _depthwise_conv1d_as_matmul_update(
                mixed_qkv,
                conv_state,
                layer.conv1d.weight.squeeze(1),
                layer.conv1d.bias,
                layer.activation,
            )
        else:
            if cache_params is not None:
                conv_state_val = F.pad(mixed_qkv, (layer.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state_val = cache_params.update_conv_state(conv_state_val, layer.layer_idx)
            mixed_qkv = _depthwise_conv1d_as_matmul_prefill(layer.conv1d, mixed_qkv, seq_len)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [layer.key_dim, layer.key_dim, layer.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)

        beta = b.sigmoid()
        g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
        if layer.num_v_heads // layer.num_k_heads > 1:
            query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
            key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = layer.chunk_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = layer.recurrent_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, layer.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
        z = z.reshape(-1, layer.head_v_dim)
        core_attn_out = layer.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        hidden_states = layer.out_proj(core_attn_out)
        return hidden_states

    layer.forward = patched_forward


def apply_graph_break_patches(model):
    """Apply monkey patches to eliminate graph breaks in Qwen3.6 on TT hardware."""
    model_cls = type(model.model)
    model_cls._update_linear_attn_mask = _patched_update_linear_attn_mask

    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule
            _patch_conv1d_forward(layer.linear_attn)

    log("Applied graph-break patches (linear_attn_mask + chunk_gated_delta_rule + conv1d→matmul)")


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
    max_new_tokens = 10
    batch_size = 1

    # --- Load model and tokenizer ---
    log(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading model weights (bfloat16)...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_id)
    # Limiting the number of layers to 4
    # num_layers = 4
    # text_cfg = config.text_config if hasattr(config, "text_config") else config
    # text_cfg.num_hidden_layers = num_layers
    # if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
    #     text_cfg.layer_types = text_cfg.layer_types[:num_layers]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")
    log(f"Model layers: {len(model.model.layers)}, params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

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

    # StaticCache pre-allocates KV buffers for attention layers (StaticLayer).
    # DeltaNet layers use LinearAttentionLayer (fixed-size recurrent state) and
    # lazily initialize under torch.compile with is_torchdynamo_compiling() guards.
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    override_cache_static_layers(static_cache)

    # Eagerly initialize TTStaticLayer (attention) cache layers on CPU to prevent
    # lazy_initialization from firing inside the torch.compile trace, which would
    # bake tensor allocation ops into the compiled StableHLO graph.
    # LinearAttentionLayer (DeltaNet) layers are left to lazily initialize since
    # they have is_torchdynamo_compiling() guards and use different state shapes.
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    for layer in static_cache.layers:
        if isinstance(layer, TTStaticLayer) and not layer.is_initialized:
            fake_kv = torch.zeros(
                (batch_size, num_key_value_heads, 0, head_dim),
                dtype=torch.bfloat16,
                device="cpu",
            )
            layer.lazy_initialization(fake_kv, fake_kv)

    # Pre-allocate attention mask to max_cache_len with all 1s to avoid per-step
    # mask updates. Zero-initialized cache slots contribute nothing to attention
    # output, so attending to unfilled positions is harmless.
    full_attention_mask = torch.ones(
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

    # Transfer eagerly-initialized cache tensors (keys, values, cumulative_length)
    # to device. LinearAttentionLayer (DeltaNet) layers lazily initialize on-device
    # during the first forward pass and are skipped here.
    for layer in input_args["past_key_values"].layers:
        if isinstance(layer, TTStaticLayer):
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
            layer.device = device
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
        if attn is not None:
            if hasattr(attn, "q_proj"):
                shard_specs[attn.q_proj.weight] = ("model", None)
            if hasattr(attn, "k_proj"):
                shard_specs[attn.k_proj.weight] = ("model", None)
            if hasattr(attn, "v_proj"):
                shard_specs[attn.v_proj.weight] = ("model", None)
            if hasattr(attn, "o_proj"):
                shard_specs[attn.o_proj.weight] = (None, "model")

        # DeltaNet (linear_attn) projections can now be sharded because the
        # depthwise conv1d has been replaced with element-wise ops (see
        # _patch_conv1d_forward). Previously, groups= partitioning caused a
        # feature_group_count mismatch (tt-xla#3508).
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is not None:
            if hasattr(linear_attn, "in_proj_qkv"):
                shard_specs[linear_attn.in_proj_qkv.weight] = ("model", None)
            if hasattr(linear_attn, "in_proj_qkvz"):
                shard_specs[linear_attn.in_proj_qkvz.weight] = ("model", None)
            if hasattr(linear_attn, "in_proj_ba"):
                shard_specs[linear_attn.in_proj_ba.weight] = ("model", None)
            if hasattr(linear_attn, "out_proj"):
                shard_specs[linear_attn.out_proj.weight] = (None, "model")

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        shard_specs[model.lm_head.weight] = ("model", None)

    log(f"Applying sharding annotations ({len(shard_specs)} tensors)...")
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    # Shard KV cache along the head dimension to match attention weight sharding.
    # Only TTStaticLayer (attention) layers have KV cache; DeltaNet layers use
    # recurrent state that doesn't need the same head-parallel sharding.
    for layer in input_args["past_key_values"].layers:
        if isinstance(layer, TTStaticLayer):
            xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
            xs.mark_sharding(layer.values, mesh, (None, "model", None, None))
    log("Sharding annotations applied.")

    # --- Compile ---
    log("Compiling model with torch.compile(backend='tt')...")
    compiled_model = torch.compile(model, backend="tt")

    # --- Generation loop ---
    generated_tokens = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            t_step = time.time()

            if step == 0:
                log("Running prefill (triggers compilation for prefill graph)...")
                t0 = time.time()

            output = compiled_model(**input_args)

            t_fwd = time.time()

            # Argmax on device, then transfer only the token ID (1 scalar)
            # instead of the full (1, seq, 248320) logits tensor.
            if step == 0:
                next_token_id = output.logits[0, prompt_len - 1, :].argmax(dim=-1)
            else:
                next_token_id = output.logits[0, -1, :].argmax(dim=-1)

            next_token_cpu = next_token_id.to("cpu")
            t_sync = time.time()

            if step == 0:
                log(f"Prefill completed in {t_sync - t0:.1f}s")
                log("Starting decode...")
                t_decode = time.time()

            token_val = next_token_cpu.item()
            generated_tokens.append(token_val)

            step_elapsed = time.time() - t_step
            log(f"Step {step}: fwd={t_fwd - t_step:.3f}s sync={t_sync - t_fwd:.3f}s total={step_elapsed:.3f}s")

            if token_val == tokenizer.eos_token_id:
                break

            # --- Update inputs for next decode step (fixed shapes) ---
            input_args["input_ids"] = next_token_cpu.view(1, 1).to(device)

            new_cache_pos = torch.tensor([prompt_len + step], dtype=torch.long)
            input_args["cache_position"] = new_cache_pos.to(device)
            input_args["position_ids"] = new_cache_pos.unsqueeze(0).to(device)

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
