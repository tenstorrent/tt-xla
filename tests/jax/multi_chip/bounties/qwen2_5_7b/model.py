#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen2.5-7B with Megatron-LM style tensor parallelism in JAX/Flax.

Implements the canonical column-parallel / row-parallel pattern
(arXiv:1909.08053) so that each decoder layer uses exactly two
collective operations (one all_reduce after attention, one after MLP).

Attention sublayer:
  q/k/v projections  -> column-parallel (output stays sharded by heads)
  attention kernel    -> runs locally per device on its head partition
  o_proj             -> row-parallel   (psum all_reduce at the end)

MLP sublayer:
  gate/up projections -> column-parallel (output stays sharded)
  silu(gate) * up     -> elementwise, runs locally on each device
  down_proj           -> row-parallel   (psum all_reduce at the end)
"""

import gc
import os
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors import safe_open

mesh = None


def setup_device_mesh(config=None):
    """Setup device mesh for tensor parallelism.

    Validates that the TP size (number of devices) cleanly divides the
    model's head counts so that each device owns an integer number of
    attention heads.
    """
    global mesh
    devices = np.array(jax.devices())
    tp_size = len(devices)
    print(f"Available devices: {tp_size}")

    if config is not None:
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        assert num_heads % tp_size == 0, (
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"TP size ({tp_size}). Valid TP sizes for this model: "
            f"{[d for d in range(1, num_heads + 1) if num_heads % d == 0 and num_kv_heads % d == 0]}"
        )
        assert num_kv_heads % tp_size == 0, (
            f"num_key_value_heads ({num_kv_heads}) must be divisible by "
            f"TP size ({tp_size})."
        )

    mesh = Mesh(devices, ("mp",))
    print(f"Created mesh: {mesh}")
    return mesh


def sample_next_token(logits, temperature=0.0):
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1).item()
    else:
        scaled_logits = logits / temperature
        return jax.random.categorical(
            jax.random.PRNGKey(0), scaled_logits, axis=-1
        ).item()


# ---------------------------------------------------------------------------
# Megatron-LM parallel dense layers
# ---------------------------------------------------------------------------


class ColumnParallelDense(nn.Module):
    """Column-parallel dense: shards the *output* dimension across devices.

    Each device computes x @ kernel_shard (+ bias_shard) and keeps its
    local result.  **No all_gather** -- the output stays sharded so
    downstream work (attention, silu*gate) runs locally.
    """

    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features

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

            def matmul_bias_fn(x_loc, k_loc, b_loc):
                return jnp.einsum("bsd,df->bsf", x_loc, k_loc) + b_loc

            return shard_map(
                matmul_bias_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp"), P("mp")),
                out_specs=P(None, None, "mp"),
                check_rep=False,
            )(x, kernel, bias)
        else:

            def matmul_fn(x_loc, k_loc):
                return jnp.einsum("bsd,df->bsf", x_loc, k_loc)

            return shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp")),
                out_specs=P(None, None, "mp"),
                check_rep=False,
            )(x, kernel)


class RowParallelDense(nn.Module):
    """Row-parallel dense: shards the *input* dimension across devices.

    Each device computes x_shard @ kernel_shard (partial sum), then a
    single ``psum`` (all_reduce) produces the final replicated result.
    Bias (if any) is added *after* the reduce on replicated data.
    """

    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_dim, out_dim),
            self.param_dtype,
        )

        def matmul_reduce_fn(x_loc, k_loc):
            local_out = jnp.einsum("bsd,df->bsf", x_loc, k_loc)
            return jax.lax.psum(local_out, axis_name="mp")

        output = shard_map(
            matmul_reduce_fn,
            mesh=mesh,
            in_specs=(P(None, None, "mp"), P("mp", None)),
            out_specs=P(None, None, None),
            check_rep=False,
        )(x, kernel)

        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros, (out_dim,), self.param_dtype
            )
            output = output + bias

        return output


class ParallelDense(nn.Module):
    """Column-parallel dense with all_gather (used only for lm_head
    where full logits are needed for argmax)."""

    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features

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

        def matmul_fn(x_loc, k_loc, b_loc=None):
            local_out = jnp.einsum("bsd,df->bsf", x_loc, k_loc)
            if b_loc is not None:
                local_out = local_out + b_loc
            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
            return jnp.reshape(
                jnp.transpose(full_out, (1, 2, 0, 3)),
                (x_loc.shape[0], x_loc.shape[1], -1),
            )

        if bias is not None:
            return shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp"), P("mp")),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel, bias)
        else:
            return shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp")),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel)


# ---------------------------------------------------------------------------
# Embedding (replicated -- standard TP practice)
# ---------------------------------------------------------------------------


class ParallelEmbed(nn.Module):
    num_embeddings: int
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs):
        embedding = jnp.asarray(self.embedding, self.dtype)
        return embedding[inputs.astype("i4")]


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    pos = position_ids.astype(jnp.float32)
    dim = head_dim // 2
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    t = pos[..., None] * freqs[None, None, :]
    cos = jnp.cos(t)[..., None, :]
    sin = jnp.sin(t)[..., None, :]
    return cos, sin


def apply_rotary_emb(q, k, cos, sin):
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot


def make_causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return jnp.where(i >= j - (k_len - q_len), 0, -1e9)


# ---------------------------------------------------------------------------
# Attention (Megatron-LM: column-parallel q/k/v, local attention, row-parallel o)
# ---------------------------------------------------------------------------


class MegatronQwenAttention(nn.Module):
    """Megatron-LM style tensor-parallel attention.

    * q/k/v projections are **column-parallel** (output sharded by heads).
    * The attention kernel runs inside ``shard_map`` so each device works
      only on its local head partition -- no cross-device data movement.
    * ``o_proj`` is **row-parallel** (one ``psum`` all_reduce at the end).

    Net collectives per call: **one all_reduce** (inside ``o_proj``).
    """

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

        self.q_proj = ColumnParallelDense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
            name="q_proj",
        )
        self.k_proj = ColumnParallelDense(
            self.kv_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
            name="k_proj",
        )
        self.v_proj = ColumnParallelDense(
            self.kv_dim,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
            name="v_proj",
        )
        self.o_proj = RowParallelDense(
            self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=False,
            name="o_proj",
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
    ):
        batch, seq, _ = hidden_states.shape

        # Column-parallel projections: output is sharded P(None,None,"mp")
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        head_dim = self.head_dim
        rope_theta = self.rope_theta
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads

        if past_key_value is not None:
            past_k, past_v = past_key_value
        else:
            past_k = jnp.zeros(
                (batch, 0, self.num_kv_heads, head_dim), dtype=self.dtype
            )
            past_v = jnp.zeros(
                (batch, 0, self.num_kv_heads, head_dim), dtype=self.dtype
            )

        def _local_attention(q_loc, k_loc, v_loc, mask, pos_ids, pk, pv):
            local_q_heads = q_loc.shape[-1] // head_dim
            local_kv_heads = k_loc.shape[-1] // head_dim
            b, s = q_loc.shape[0], q_loc.shape[1]

            q_loc = q_loc.reshape(b, s, local_q_heads, head_dim)
            k_loc = k_loc.reshape(b, s, local_kv_heads, head_dim)
            v_loc = v_loc.reshape(b, s, local_kv_heads, head_dim)

            cos, sin = compute_cos_sin_cache(pos_ids, head_dim, rope_theta)
            q_loc, k_loc = apply_rotary_emb(q_loc, k_loc, cos, sin)

            # KV cache: concatenate along seq dim (head sharding preserved)
            k_loc = jnp.concatenate([pk, k_loc], axis=1)
            v_loc = jnp.concatenate([pv, v_loc], axis=1)

            cache_k, cache_v = k_loc, v_loc

            # GQA: repeat KV heads to match local Q heads
            repeat_factor = local_q_heads // local_kv_heads
            if repeat_factor > 1:
                k_exp = jnp.repeat(k_loc, repeat_factor, axis=2)
                v_exp = jnp.repeat(v_loc, repeat_factor, axis=2)
            else:
                k_exp, v_exp = k_loc, v_loc

            # Standard scaled dot-product attention on local heads
            qt = q_loc.transpose(0, 2, 1, 3)
            kt = k_exp.transpose(0, 2, 1, 3)
            vt = v_exp.transpose(0, 2, 1, 3)

            scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
            scores = jnp.einsum("bhqd,bhkd->bhqk", qt, kt) * scale
            if mask is not None:
                scores = scores + mask
            probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
            attn = jnp.einsum("bhqk,bhkd->bhqd", probs, vt)
            attn = attn.transpose(0, 2, 1, 3).reshape(b, s, local_q_heads * head_dim)
            return attn, cache_k, cache_v

        attn_out, cache_k, cache_v = shard_map(
            _local_attention,
            mesh=mesh,
            in_specs=(
                P(None, None, "mp"),  # q  (sharded features)
                P(None, None, "mp"),  # k
                P(None, None, "mp"),  # v
                None,  # attention_mask (replicated)
                None,  # position_ids   (replicated)
                P(None, None, "mp", None),  # past_k (sharded heads)
                P(None, None, "mp", None),  # past_v
            ),
            out_specs=(
                P(None, None, "mp"),  # attn_out (sharded features)
                P(None, None, "mp", None),  # cache_k  (sharded heads)
                P(None, None, "mp", None),  # cache_v
            ),
            check_rep=False,
        )(q, k, v, attention_mask, position_ids, past_k, past_v)

        # Row-parallel output projection: one all_reduce -> replicated
        attn_out = self.o_proj(attn_out)
        return attn_out, (cache_k, cache_v)


# ---------------------------------------------------------------------------
# MLP (Megatron-LM: column-parallel gate/up, local silu*gate, row-parallel down)
# ---------------------------------------------------------------------------


class QwenMLP(nn.Module):
    """Megatron-LM style tensor-parallel MLP.

    * gate and up projections are **column-parallel** (output sharded).
    * ``silu(gate) * up`` runs locally on each device.
    * ``down_proj`` is **row-parallel** (one ``psum`` all_reduce).

    Net collectives per call: **one all_reduce** (inside ``down_proj``).
    """

    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])

        self.gate_proj = ColumnParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="gate_proj",
        )
        self.up_proj = ColumnParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="up_proj",
        )
        self.down_proj = RowParallelDense(
            c["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="down_proj",
        )

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        def _local_silu_mul(g, u):
            return jax.nn.silu(g) * u

        intermediate = shard_map(
            _local_silu_mul,
            mesh=mesh,
            in_specs=(P(None, None, "mp"), P(None, None, "mp")),
            out_specs=P(None, None, "mp"),
            check_rep=False,
        )(gate, up)

        return self.down_proj(intermediate)


# ---------------------------------------------------------------------------
# Decoder layer and full model
# ---------------------------------------------------------------------------


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
        self.self_attn = MegatronQwenAttention(config=c, dtype=jnp.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-6),
            dtype=jnp.bfloat16,
            name="post_attention_layernorm",
        )
        self.mlp = QwenMLP(config=c, dtype=jnp.bfloat16)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
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
        # lm_head keeps the original all_gather pattern because the full
        # vocab logits are needed for argmax / sampling after generation.
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
                hidden_states,
                attention_bias,
                position_ids,
                past_kv,
            )
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return logits


# ---------------------------------------------------------------------------
# Weight loading (unchanged -- param hierarchy matches Flax module names)
# ---------------------------------------------------------------------------


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
                        param = jnp.array(param, dtype=jnp.bfloat16)
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
                        loaded_count += 1
    gc.collect()
    print(f"Weight loading completed. Loaded {loaded_count} parameters.")
    return params
