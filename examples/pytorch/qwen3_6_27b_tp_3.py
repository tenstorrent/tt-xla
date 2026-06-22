# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.6-27B generation with tensor parallelism on Tenstorrent hardware.

This script reuses the modular generation-loop structure from the GPT-OSS 20B
example (gpt_oss_20b.py) and adapts it to Qwen3.6-27B:
- The graph-break-free monkey patches are copied from qwen3_6_27b_tp_decode.py.
- The tensor-parallel sharding strategy follows the Qwen decode reference
  (pure model-axis sharding of attention/MLP projections + KV cache).

Architecture: mixture of Gated DeltaNet (linear attention, recurrent state) and
Gated Attention (static KV cache) layers.

Usage:
    python examples/pytorch/qwen3_6_27b_tp_3.py
    python examples/pytorch/qwen3_6_27b_tp_3.py --interactive
"""

import argparse
import math
import os
import resource

# Cap the process address space to avoid OOM-killing the host while loading and
# tracing the large Qwen3.6 model on CPU before the device transfer.
_MEM_LIMIT_GB = int(os.environ.get("TTXLA_MEM_LIMIT_GB", "230"))
_MEM_LIMIT_BYTES = _MEM_LIMIT_GB * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from tt_torch.transformers_overrides import TTStaticLayer, override_cache_static_layers

DEFAULT_PROMPTS = ["The capital of France is"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Monkey patches for graph-break-free compilation on TT hardware.
# Copied from qwen3_6_27b_tp_decode.py.
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
                # Save conv_state before masking — with left-padding the last K
                # positions are real tokens so conv_state is correct as-is.
                conv_state_val = F.pad(mixed_qkv, (layer.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state_val = cache_params.update_conv_state(conv_state_val, layer.layer_idx)
            mixed_qkv = _depthwise_conv1d_as_matmul_prefill(layer.conv1d, mixed_qkv, seq_len)

        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Zero query/key/value at PAD positions AFTER conv. Both the linear
        # projection bias and the conv1d bias produce non-zero outputs at PAD
        # positions. With key=0 and value=0, the delta-rule update at PAD
        # positions is beta * 0^T * 0 = 0, leaving recurrent_state unaffected.
        if attention_mask is not None and seq_len > 1 and not use_precomputed_states:
            pad_mask = attention_mask[:, :seq_len, None]  # [B, L, 1]
            mixed_qkv = mixed_qkv * pad_mask

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


# --------------------------------
# Qwen3.6 27B Generation Loop Example
# --------------------------------
def qwen3_6_27b_tp(interactive: bool = False):

    # Set up config variables.
    max_cache_len: int = 256
    model_name: str = "Qwen/Qwen3.6-27B"

    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    while True:
        if interactive:
            user_prompt = input("Enter your prompt or quit() to exit: ")
            batch_size: int = 1
            if user_prompt.lower() == "quit()":
                break
            user_prompt = [user_prompt]
        else:
            user_prompt = DEFAULT_PROMPTS
            batch_size: int = len(user_prompt)

        # Construct inputs, including static cache
        input_args, formatted_prompts = construct_inputs(
            user_prompt, tokenizer, model.config, batch_size, max_cache_len
        )

        # Limit maximum generation count to fit within preallocated static cache
        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        # Transfer model and inputs to device
        model, input_args = transfer_to_device(model, input_args, device)

        # Mark sharding on inputs and model internals
        mark_sharding_on_inputs_and_model(model, input_args, mesh)

        # Compile model
        compiled_model = torch.compile(model, backend="tt")

        # Run generation loop until EOS token generated or max tokens reached
        run_generate(
            compiled_model,
            input_args,
            tokenizer,
            device,
            mesh,
            max_tokens_to_generate,
            formatted_prompts,
            interactive,
        )

        if not interactive:
            break


def setup_spmd():
    """
    Initializes SPMD mode in torch_xla.
    """

    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Qwen3.6 uses pure model-axis (tensor parallel) sharding, so the mesh keeps a
    singleton batch axis and spreads all devices across the model axis.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 2, (
        f"This script requires at least 2 devices, but found {num_devices}. "
        f"Use the single-chip script for single-device inference."
    )

    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def setup_model_and_tokenizer(
    model_name: str,
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    log(f"Loading tokenizer for {model_name}...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for DeltaNet (linear attention) layers.
    # With right-padding, conv_state and recurrent_state would accumulate
    # PAD-token projections at the END of the sequence, corrupting decode.
    # With left-padding, real tokens are last so the saved states reflect them.
    tokenizer.padding_side = "left"

    log(f"Loading model weights (bfloat16)...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_name)
    # Limiting the number of layers to keep host memory and compile time bounded.
    # num_layers = int(os.environ.get("TTXLA_NUM_LAYERS", "4"))
    # text_cfg = config.text_config if hasattr(config, "text_config") else config
    # text_cfg.num_hidden_layers = num_layers
    # if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
    #     text_cfg.layer_types = text_cfg.layer_types[:num_layers]

    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")
    log(
        f"Model layers: {len(model.model.layers)}, "
        f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B"
    )

    # Apply TT-friendly monkey patches (graph-break-free DeltaNet/conv1d/mask).
    apply_graph_break_patches(model)

    return model, tokenizer


def construct_inputs(
    input_prompt: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
) -> tuple[dict, List[str]]:
    """
    Construct inputs including static cache.

    Args:
        input_prompt: Input text prompt(s) - can be a single string or list of strings
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length

    Returns:
        Tuple of (input_args dictionary, formatted_prompts list)
    """

    # Apply chat template to format prompts
    formatted_prompts = []
    for prompt in input_prompt:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    prompt_lengths = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in formatted_prompts
    ]
    max_length = max(prompt_lengths)

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # StaticCache pre-allocates KV buffers for attention layers (TTStaticLayer).
    # DeltaNet layers use LinearAttentionLayer (fixed-size recurrent state) and
    # lazily initialize under torch.compile with is_torchdynamo_compiling() guards.
    # Static cache should be initialized on CPU and separately transferred to device
    # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache: StaticCache = StaticCache(
        config=model_config,
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
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    for layer in static_cache.layers:
        if isinstance(layer, TTStaticLayer) and not layer.is_initialized:
            fake_kv = torch.zeros(
                (batch_size, num_key_value_heads, 0, head_dim),
                dtype=torch.bfloat16,
                device="cpu",
            )
            layer.lazy_initialization(fake_kv, fake_kv)

    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])
    # Pass position_ids explicitly so the forward never enters the
    # "position_ids is None" branch (which calls past_key_values.get_seq_length()
    # and bakes a per-step Python int into the graph, causing dynamo to
    # recompile on every decode step).
    position_ids = cache_position.unsqueeze(0)

    # Attention mask is needed to ignore padding tokens in left-padded batches. The mask should match max_cache_len
    # to prevent recompilation or implicit padding by transformers, which can cause degenerate output.
    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    #   Debug prints
    print("\n=== DEBUG: construct_inputs ===")
    print(f"Original prompts: {input_prompt}")
    print(f"Formatted prompts (with chat template): {formatted_prompts}")
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs: {inputs.input_ids}")
    print(f"Input attention mask shape: {inputs.attention_mask.shape}")
    print(f"Full attention mask shape (pre-allocated): {full_attention_mask.shape}")
    print(f"Full attention mask: {full_attention_mask}")
    print(f"Cache position shape: {cache_position.shape}")
    print(f"Cache position: {cache_position}")
    print(f"Actual sequence length (non-padding): {inputs.attention_mask.sum().item()}")
    print("=" * 50)

    return input_args, formatted_prompts


def transfer_to_device(
    model: torch.nn.Module, input_args: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """
    Transfer model and inputs to device.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple of (model, input_args) on device
    """
    # Transfer eagerly-initialized cache tensors (keys, values, cumulative_length)
    # to device. LinearAttentionLayer (DeltaNet) layers lazily initialize on-device
    # during the first forward pass and are skipped here.
    for layer in input_args["past_key_values"].layers:
        if isinstance(layer, TTStaticLayer):
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
            # StaticLayer.update builds a fresh `cache_position` each call as
            # `torch.arange(kv_length, device=self.device) + self.cumulative_length`,
            # then passes it to `self.keys.index_copy_(2, cache_position, ...)`.
            # If `self.device` is still "cpu" or `cumulative_length` is a CPU tensor,
            # the resulting index is CPU and `index_copy_` on an XLA `keys` tensor
            # raises `Check failed: xtensor: Input tensor is not an XLA tensor`.
            if hasattr(layer, "device"):
                layer.device = device
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)

    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["position_ids"] = input_args["position_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    model = model.to(device)

    return model, input_args


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module, input_args: dict, mesh: Mesh
):
    """
    Mark sharding on inputs and model internals.
    If mark_sharding is not called on a tensor, it is fully replicated across all devices.
        i.e. on cache_positions, input_ids

    Uses Qwen3.6 tensor-parallel sharding: attention/MLP projections are split
    along the model axis. DeltaNet (linear_attn) layers stay replicated.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        mesh: Device mesh for SPMD operations
    """

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

        # DeltaNet (linear_attn) projections are NOT sharded yet.
        # The conv1d has been replaced with element-wise ops (_patch_conv1d_forward),
        # removing the feature_group_count blocker. However, full TP sharding also
        # requires sharding in_proj_z/b/a and the conv weight to match in_proj_qkv,
        # which needs more work. For now, DeltaNet layers remain replicated.

    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        shard_specs[model.lm_head.weight] = ("model", None)

    print(f"Applying sharding annotations ({len(shard_specs)} tensors)...")
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    # Shard KV cache along the head dimension to match attention weight sharding.
    # Only TTStaticLayer (attention) layers have KV cache; DeltaNet layers use
    # recurrent state that doesn't need the same head-parallel sharding.
    for layer in input_args["past_key_values"].layers:
        if isinstance(layer, TTStaticLayer):
            xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
            xs.mark_sharding(layer.values, mesh, (None, "model", None, None))
    print("Sharding annotations applied.")


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh = None,
    max_tokens_to_generate: int = 128,
    formatted_prompts: List[str] = [""],
    is_interactive: bool = False,
):
    """
    Run the generation loop.

    Args:
        compiled_model: Compiled model instance
        input_args: Input arguments dictionary
        tokenizer: Tokenizer instance
        device: Device
        mesh: Device mesh for SPMD operations (optional)
        max_tokens_to_generate: Maximum number of tokens to generate
        formatted_prompts: Formatted prompts with chat template applied
        is_interactive: Whether running in interactive mode
    """
    num_users = input_args["input_ids"].shape[0]
    output_tokens: List[List[str]] = [[] for _ in range(num_users)]

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("RUNNING PREFILL")
                if is_interactive:
                    print("=" * 80)
                    print("PROMPT:")
                    print(formatted_prompts[0])
                    print("-" * 80)
                    print("GENERATED:", end="", flush=True)

            # Run forward pass
            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")
            next_token_id = output_logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[i]) for i in range(num_users)]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])
                if is_interactive:
                    print(output_text[i], end="", flush=True)

            # Check for EOS token and early exit
            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # Add newline after generation completes
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)
            # keep position_ids in sync with cache_position (see construct_inputs)
            input_args["position_ids"] = host_cache_pos.unsqueeze(0).to(device)

    print()
    if not is_interactive:
        for i in range(num_users):
            print(f"=" * 80)
            print(f"Result for user {i}:")
            print(f"-" * 80)
            print("PROMPT:")
            print(formatted_prompts[i])
            print(f"-" * 80)
            print("GENERATED:")
            print("".join(output_tokens[i]))
            print(f"=" * 80)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.6 27B generation example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Enable interactive mode for entering custom prompts",
    )
    args = parser.parse_args()

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    qwen3_6_27b_tp(interactive=args.interactive)
