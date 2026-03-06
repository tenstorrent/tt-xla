#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Core model implementation for Qwen2.5-7B with tensor parallelism in JAX/Flax.
"""

import gc
import os
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from safetensors import safe_open

# Global mesh (set externally, e.g., in generate scripts)
mesh = None


def setup_device_mesh():
    """Setup device mesh for tensor parallelism."""
    global mesh
    from jax.sharding import Mesh

    devices = np.array(jax.devices())
    print(f"Available devices: {len(devices)}")
    # Create 1D mesh for tensor parallelism
    mesh = Mesh(devices, ("mp",))
    print(f"Created mesh: {mesh}")
    return mesh


def sample_next_token(logits, temperature=0.0):
    """Sample next token from logits using greedy sampling."""
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1).item()
    else:
        scaled_logits = logits / temperature
        probs = jax.nn.softmax(scaled_logits, axis=-1)
        return jax.random.categorical(
            jax.random.PRNGKey(0), scaled_logits, axis=-1
        ).item()


# --- Model Code ---
class FullyParallelQwenAttention(nn.Module):
    """Full parallel attention with all projections using ParallelDense."""

    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = c.get("rope_theta", 1000000.0)

        # All projections use ParallelDense for full tensor parallelism
        self.q_proj = ParallelDense(
            self.hidden_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=True,
            name="q_proj",
        )
        self.k_proj = ParallelDense(
            self.kv_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=True,
            name="k_proj",
        )
        self.v_proj = ParallelDense(
            self.kv_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=True,
            name="v_proj",
        )
        self.o_proj = ParallelDense(
            self.hidden_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            name="o_proj",
        )

    def __call__(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None
    ):
        batch, seq, _ = hidden_states.shape

        # Project inputs using FULL PARALLEL approach
        q = self.q_proj(hidden_states).reshape(
            batch, seq, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).reshape(
            batch, seq, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).reshape(
            batch, seq, self.num_kv_heads, self.head_dim
        )

        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(
                position_ids, self.head_dim, self.rope_theta
            )
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)

        cache_k, cache_v = k, v

        # GQA: Repeat k/v to match query heads
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Attention computation
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attention_mask is not None:
            scores += attention_mask

        # Attention computation
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        attn_out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)

        # Output projection using ParallelDense
        attn_out = self.o_proj(attn_out)

        return attn_out, (cache_k, cache_v)


def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    pos = position_ids.astype(jnp.float32)  # [batch, seq]
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    # Expand for broadcasting: [batch, seq, 1, dim]
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return cos, sin


def apply_rotary_emb(q, k, cos, sin):
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    # cos and sin are already [batch, seq, 1, dim], so they broadcast correctly
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot


def make_causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return jnp.where(i >= j - (k_len - q_len), 0, -1e9)


class ParallelEmbed(nn.Module):
    """Tensor parallel embedding layer that shards embeddings across vocab dimension"""

    num_embeddings: int
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    name: str = None

    def setup(self):
        # For embeddings, we typically replicate rather than shard
        # Using standard setup pattern to avoid scope issues
        self.embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs):
        # Standard embedding lookup
        embedding = jnp.asarray(self.embedding, self.dtype)
        return embedding[inputs.astype("i4")]


class ParallelDense(nn.Module):
    """Full parallel dense layer with tensor parallelism."""

    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features

        # Load full-size parameters (compatible with weight loading)
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_dim, out_dim),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros, (out_dim,), self.param_dtype
            )
        else:
            bias = None

        def matmul_fn(x, k, b=None):
            # Kernel is already sharded by input spec P(None, "mp")
            # k is already the shard for this device
            local_out = jnp.einsum("bsd,df->bsf", x, k)

            # Apply bias if provided (bias is also sharded by input spec)
            if b is not None:
                local_out = local_out + b

            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)

            # Reshape to combine all device outputs - use transpose like Llama
            result = jnp.reshape(
                jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1)
            )
            return result

        if bias is not None:
            output = shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(
                    None,
                    P(None, "mp"),
                    P(
                        "mp",
                    ),
                ),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel, bias)
        else:
            output = shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp")),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel)

        return output


class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        # Use ParallelDense for tensor parallelism
        self.gate_proj = ParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="gate_proj",
        )
        self.up_proj = ParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="up_proj",
        )
        self.down_proj = ParallelDense(
            c["hidden_size"], dtype=self.dtype, param_dtype=self.dtype, name="down_proj"
        )

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-6),
            dtype=jnp.bfloat16,
            name="input_layernorm",
        )
        self.self_attn = FullyParallelQwenAttention(config=c, dtype=jnp.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-6),
            dtype=jnp.bfloat16,
            name="post_attention_layernorm",
        )
        self.mlp = QwenMLP(config=c, dtype=jnp.bfloat16)

    def __call__(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, past_key_value


class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.embed_tokens = ParallelEmbed(
            c["vocab_size"],
            c["hidden_size"],
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            name="embed_tokens",
        )
        self.layers = [
            QwenDecoderLayer(config=c, dtype=jnp.bfloat16, name=f"layers_{i}")
            for i in range(c["num_hidden_layers"])
        ]
        self.norm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="norm"
        )
        # Use ParallelDense for tensor parallelism (rationale: sharded for TP efficiency, but note: large vocab can cause bottlenecks; non-parallel alternative considered but kept for consistency)
        self.lm_head = ParallelDense(
            c["vocab_size"],
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        return_dict=True,
    ):
        batch, seq = input_ids.shape
        key_len = (
            seq
            if past_key_values is None or past_key_values[0] is None
            else past_key_values[0][0].shape[1] + seq
        )

        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=self.dtype)
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        attention_bias = jnp.where(attention_mask == 0, -1e9, 0) + causal_mask

        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_key_values = []

        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, new_kv = layer(
                hidden_states, attention_bias, position_ids, past_kv
            )
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return logits


# --- Weight Loading ---
def get_param_path(name):
    mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    if name in mapping:
        return mapping[name]
    import re

    if m := re.match(
        r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight", name
    ):
        return (f"layers_{m.group(1)}", f"{m.group(2)}_layernorm", "scale")
    if m := re.match(
        r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)", name
    ):
        return (
            f"layers_{m.group(1)}",
            "self_attn",
            f"{m.group(2)}_proj",
            "kernel" if m.group(3) == "weight" else "bias",
        )
    if m := re.match(r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", name):
        return (f"layers_{m.group(1)}", "mlp", f"{m.group(2)}_proj", "kernel")
    return None


def transpose_if_needed(name, param):
    if "weight" in name and "layernorm" not in name and "embed_tokens" not in name:
        return param.T
    return param


def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files."""
    print(f"Loading JAX model weights from {model_path}...")
    params = {"params": {}}
    loaded_count = 0
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(
                            param, dtype=jnp.bfloat16
                        )  # Always load as bfloat16
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
                        loaded_count += 1
    gc.collect()
    print(f"Weight loading completed. Loaded {loaded_count} parameters.")
    return params
