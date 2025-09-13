# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gemma 3 model implementation for JAXgarden, based on the implementation in Transformers
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

# from transformers import Gemma3Config, SiglipVisionConfig, Gemma3TextConfig
from gemma3.base import BaseConfig, BaseModel


@dataclass
class Gemma3Config(BaseConfig):
    """Configuration for Gemma3 model.

    This configuration class extends BaseConfig and contains all the parameters
    required to initialize a Gemma3 model. It includes settings for model architecture,
    attention mechanisms, and other hyperparameters.

    Attributes:
        vocab_size: Size of vocabulary
        hidden_size: Size of hidden states
        intermediate_size: Size of MLP intermediate layer
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key/value heads (for group query attention)
        head_dim: Dimension of each attention head
        rms_norm_eps: Epsilon for RMS normalization
        rope_theta: Base period for rotary position embeddings
        rope_local_base_freq: Base frequency for RoPE in local attention
        max_position_embeddings: Maximum sequence length that this model might ever be used with
        initializer_range: Standard deviation for weight initialization
        attention_bias: Whether to use bias in attention projections
        attention_dropout: Dropout probability for attention weights
        query_pre_attn_scalar: Scaling factor for attention scores before softmax
        sliding_window: Size of sliding window for local attention
        sliding_window_pattern: Pattern for alternating global/local attention (every N layers)
        final_logit_soft_cap: Soft cap for final logits
        attn_logit_soft_cap: Soft cap for attention logits
        pad_token_id: ID of padding token
        eos_token_id: ID of end-of-sequence token
        bos_token_id: ID of beginning-of-sequence token
        tie_word_embeddings: Whether to tie input and output embeddings
        rope_scaling: Configuration for RoPE scaling
        hidden_activation: Activation function in MLP
        use_cache: Whether to use KV cache during generation
    """

    vocab_size: int = 262_208
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    attention_bias: bool = False
    attention_dropout: float = 0.0
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 4096
    sliding_window_pattern: int = 6
    final_logit_soft_cap: float | None = None
    attn_logit_soft_cap: float | None = None
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    rope_scaling: dict[str, Any] | None = None
    hidden_activation: str = "gelu_pytorch_tanh"
    use_cache: bool = True
    param_dtype: Any = field(default=jnp.float32, metadata={"dtype": True})
    dtype: Any = field(default=jnp.float32, metadata={"dtype": True})

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads // 2
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "GQA: num_attention_heads must be divisible by num_key_value_heads"
        )

        # Validate RoPE scaling config if provided
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", "default")
            if rope_type not in ["default", "linear", "dynamic", "yarn", "longrope", "llama3"]:
                raise ValueError(f"Unknown RoPE scaling type: {rope_type}")

            if rope_type != "default" and "factor" not in self.rope_scaling:
                raise ValueError(f"RoPE scaling type {rope_type} requires 'factor' parameter")

            if (
                rope_type in ["dynamic", "longrope", "llama3"]
                and "original_max_position_embeddings" not in self.rope_scaling
            ):
                raise ValueError(
                    f"RoPE scaling type {rope_type} requires "
                    + "'original_max_position_embeddings' parameter"
                )

            if rope_type == "yarn":
                self.rope_scaling.setdefault("beta_fast", 32.0)
                self.rope_scaling.setdefault("beta_slow", 1.0)

            if rope_type == "longrope":
                head_dim = self.head_dim // 2  # RoPE applies to half of head dim
                if (
                    "short_factor" not in self.rope_scaling
                    or "long_factor" not in self.rope_scaling
                ):
                    raise ValueError(
                        "LongRoPE requires both 'short_factor' and 'long_factor' lists"
                    )
                if len(self.rope_scaling["short_factor"]) != head_dim:
                    raise ValueError(
                        f"LongRoPE short_factor length {len(self.rope_scaling['short_factor'])} "
                        f"does not match head_dim/2 {head_dim}"
                    )
                if len(self.rope_scaling["long_factor"]) != head_dim:
                    raise ValueError(
                        f"LongRoPE long_factor length {len(self.rope_scaling['long_factor'])} "
                        f"does not match head_dim/2 {head_dim}"
                    )

            if rope_type == "llama3":
                if "low_freq_factor" not in self.rope_scaling:
                    raise ValueError("Llama3 RoPE requires 'low_freq_factor' parameter")
                if "high_freq_factor" not in self.rope_scaling:
                    raise ValueError("Llama3 RoPE requires 'high_freq_factor' parameter")

        # Generate layer types based on sliding window pattern
        self.layer_types = [
            "sliding_attention" if bool((i + 1) % self.sliding_window_pattern) else "full_attention"
            for i in range(self.num_hidden_layers)
        ]



class Gemma3RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    This implementation follows the RMSNorm implementation in Transformers.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        super().__init__()
        # Initialize weight/scale parameter to zeros, following the (1 + scale) pattern
        self.weight = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)

        # Reshape weight to match rank of x_norm for explicit broadcasting
        weight_f32 = self.weight.astype(jnp.float32)
        reshaped_weight = jnp.expand_dims(weight_f32, axis=range(x_norm.ndim - 1))

        # Apply scale as (1 + weight)
        scaled_x = x_norm * (1 + reshaped_weight)
        return scaled_x.astype(orig_dtype)


class Gemma3RotaryEmbedding(nnx.Module):
    """Applies Rotary Position Embedding (RoPE) to input tensors."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,  # Default from Gemma3
        theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        is_local_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.rope_scaling = rope_scaling
        self.is_local_attention = is_local_attention

        # Precompute inverse frequency
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            # Llama3 RoPE uses separate scaling for low and high frequencies
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            half_dim = self.dim // 2
            quarter_dim = half_dim // 2
            # First quarter: low frequency
            inv_freq_low = 1.0 / (
                (self.theta ** (jnp.arange(0, quarter_dim, dtype=jnp.float32) / quarter_dim))
                * low_freq_factor
            )
            # Second quarter: high frequency
            inv_freq_high = 1.0 / (
                (self.theta ** (jnp.arange(0, quarter_dim, dtype=jnp.float32) / quarter_dim))
                * high_freq_factor
            )
            self.inv_freq = jnp.concatenate([inv_freq_low, inv_freq_high])
        else:
            # Default RoPE or other scaling types
            base_freq = self.theta
            self.inv_freq = 1.0 / (
                base_freq ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
            )
            if not self.is_local_attention:
                scaling_factor = rope_scaling["factor"]
                self.inv_freq /= scaling_factor

        # Build cos and sin cached up to max_position_embeddings
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        # Use the direct attribute value
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        # For longrope, apply separate scaling factors
        if rope_scaling is not None and rope_scaling.get("rope_type") == "longrope":
            short_factor = jnp.array(rope_scaling["short_factor"])
            long_factor = jnp.array(rope_scaling["long_factor"])
            orig_max_pos = rope_scaling["original_max_position_embeddings"]
            # Apply short/long scaling based on position
            scaling = jnp.where(
                t[:, None] < orig_max_pos,
                short_factor[None, :],
                long_factor[None, :],
            )
            freqs = freqs * scaling

        self._cos_cached = jnp.cos(freqs)
        self._sin_cached = jnp.sin(freqs)

    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray) -> jnp.ndarray:
        """Applies RoPE to the input tensor using cached sin/cos values.

        Args:
            x: Input tensor of shape [B, L, N, H].
            position_ids: Position indices of shape [B, L].

        Returns:
            Rotated tensor of the same shape as x.
        """
        orig_dtype = x.dtype
        # x shape: [B, L, N, H]
        # position_ids shape: [B, L]

        # --- Fetch cos and sin from cache ---
        # Access direct attributes
        # self._cos_cached shape: [max_pos, H/2]
        # self._sin_cached shape: [max_pos, H/2]

        # Gather from cache: shapes become [B, L, H/2]
        # Access direct attributes
        cos_gathered = self._cos_cached[position_ids]
        sin_gathered = self._sin_cached[position_ids]

        # Expand dims for broadcasting over heads: [B, L, 1, H/2]
        cos = cos_gathered[:, :, None, :].astype(orig_dtype)
        sin = sin_gathered[:, :, None, :].astype(orig_dtype)

        # --- Apply rotation ---
        x1, x2 = jnp.split(x, 2, axis=-1)  # x1, x2 shape: [B, L, N, H/2]
        # cos, sin shape: [B, L, 1, H/2] - Broadcasts over N dimension
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        x_embed = jnp.concatenate((rotated_x1, rotated_x2), axis=-1)

        return x_embed.astype(orig_dtype)


class Gemma3Attention(nnx.Module):
    """Gemma3 attention module with support for GQA and sliding window."""

    def __init__(
        self,
        layer_idx: int,
        config: Gemma3Config,
        *,
        attention_type: str,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = attention_type
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attn_logit_soft_cap = config.attn_logit_soft_cap
        self.is_local_attn = (
            self.config.layer_types[layer_idx] == "sliding_attention"
            if self.config.layer_types
            else (layer_idx % 2 != 0) and (config.sliding_window is not None)
        )

        self.q_norm = Gemma3RMSNorm(config.head_dim, eps=config.rms_norm_eps, rngs=rngs)
        self.k_norm = Gemma3RMSNorm(config.head_dim, eps=config.rms_norm_eps, rngs=rngs)

        self.q_proj = nnx.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            use_bias=config.attention_bias,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            use_bias=config.attention_bias,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

        self.rope = Gemma3RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_local_base_freq if self.is_local_attn else config.rope_theta,
            rope_scaling=config.rope_scaling,
            is_local_attention=self.is_local_attn,
        )

        # Create attention dropout layer
        self.attn_dropout = nnx.Dropout(
            rate=config.attention_dropout,
            broadcast_dims=(1,),  # Broadcast over head dimension
            rngs=rngs,
        )

    def _make_sliding_window_mask(self, q_len: int, kv_len: int, dtype: jnp.dtype) -> jnp.ndarray:
        """Creates a combined causal and sliding window mask. True allows attention."""
        # Creates the lower triangular part for causality
        causal_mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=jnp.bool_))

        # If global attention or no sliding window, causal mask is sufficient
        if not self.is_local_attn or self.config.sliding_window is None:
            return causal_mask[None, None, :, :]  # Add batch and head dims

        window = self.config.sliding_window

        # Position indices for query and key/value sequences
        # Query positions are relative to the end of the kv sequence
        # Key positions range from 0 to kv_len - 1
        q_pos = jnp.arange(kv_len - q_len, kv_len)[:, None]  # Shape [q_len, 1]
        kv_pos = jnp.arange(kv_len)[None, :]  # Shape [1, kv_len]

        # Sliding window constraint: key_pos > query_pos - window
        window_mask = kv_pos > (q_pos - window)  # Shape [q_len, kv_len]

        # Combine causal and sliding window
        final_mask = jnp.logical_and(causal_mask, window_mask)

        # Expand dims: [q_len, kv_len] -> [1, 1, q_len, kv_len]
        return final_mask[None, None, :, :].astype(dtype)

    def _repeat_kv(self, x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
        """Repeats the key/value heads along the head dimension (axis=1) for GQA."""
        # x shape: [B, N_kv, S, H]
        if n_rep == 1:
            return x
        # Repeat along the N_kv dimension (axis=1)
        # Output shape: [B, N_kv * n_rep, S, H] = [B, N_q, S, H]
        return jnp.repeat(x, n_rep, axis=1)

    @staticmethod
    def apply_soft_cap(x: jnp.ndarray, cap: float) -> jnp.ndarray:
        return cap * jnp.tanh(x / cap)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,  # Boolean Input Padding Mask [B, kv_len]
        cache: tuple[jnp.ndarray, jnp.ndarray] | None = None,  # (k_cache, v_cache)
        deterministic: bool = True,  # Used for dropout in training
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        batch_size, q_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        )

        # Apply RMSNorm to Q, K
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE to Q, K
        query_states = self.rope(query_states, position_ids)
        key_states = self.rope(key_states, position_ids)

        # Transpose to [B, heads, seq, head_dim]
        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        # KV caching: concatenate along sequence axis (axis=2)
        if cache is not None:
            k_cache, v_cache = cache  # both [B, num_heads, cache_len, head_dim]
            key_states = jnp.concatenate([k_cache, key_states], axis=2)
            value_states = jnp.concatenate([v_cache, value_states], axis=2)
        updated_cache = (key_states, value_states)
        kv_seq_len = key_states.shape[2]  # Total sequence length including cache

        # Repeat K/V heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights with scaling
        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling

        # Apply attention logits soft cap (BEFORE masking)
        if self.attn_logit_soft_cap is not None:
            attn_weights = self.apply_soft_cap(attn_weights, self.attn_logit_soft_cap)

        # --- Apply Mask ---
        if attention_mask is not None and attention_mask.ndim == 4:
            # Direct additive mask: shape [B, 1, q_len, kv_seq_len]
            attn_weights = attn_weights + attention_mask.astype(self.config.dtype)
        else:
            # 1. Causal/Sliding Window Mask [1, 1, q_len, kv_seq_len]
            attn_internal_mask = self._make_sliding_window_mask(kv_seq_len, kv_seq_len, dtype=jnp.bool_)
            attn_internal_mask = attn_internal_mask[:, :, -q_len:, :]
            # 2. Combine with padding mask if provided (boolean 2D mask [B, kv_seq_len])
            if attention_mask is not None:
                # Ensure shape [B, kv_seq_len]
                assert attention_mask.shape == (batch_size, kv_seq_len), (
                    f"Input attention_mask shape {attention_mask.shape} does not match "
                    f"expected ({batch_size}, {kv_seq_len})"
                )
                padding_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
                final_mask = jnp.logical_and(attn_internal_mask, padding_mask)
            else:
                # Broadcast internal mask to batch
                final_mask = jnp.broadcast_to(
                    attn_internal_mask, (batch_size, 1, q_len, kv_seq_len)
                )
            # Apply additive mask bias: 0 for keep, large negative for mask
            neg_inf = jnp.finfo(self.config.dtype).min
            attention_bias = jnp.where(final_mask, 0.0, neg_inf).astype(self.config.dtype)
            attn_weights = attn_weights + attention_bias

        # --- Softmax & Output ---
        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(self.config.dtype)

        # Apply attention dropout during training
        if not deterministic and self.config.attention_dropout > 0:
            attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)

        # Apply attention weights to value states
        attn_output = jnp.matmul(attn_weights, value_states)

        # --- Reshape and Output --- #
        # Transpose back to [B, q_len, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        # Reshape to [B, q_len, hidden_size]
        attn_output = attn_output.reshape(batch_size, q_len, -1)

        # Apply output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, updated_cache


class Gemma3MLP(nnx.Module):
    """Gemma3 MLP module with GeGLU activation."""

    def __init__(self, config: Gemma3Config, *, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.gate_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def _gelu_pytorch_tanh(self, x: jnp.ndarray) -> jnp.ndarray:
        """PyTorch's approximate GELU with tanh activation."""
        # Constant values from PyTorch's implementation
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
        k_beta = 0.79788456  # sqrt(2/pi)
        k_kappa = 0.044715
        x_cube = x * x * x
        inner = k_beta * (x + k_kappa * x_cube)
        return 0.5 * x * (1.0 + jnp.tanh(inner))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = (
            self._gelu_pytorch_tanh(self.gate_proj(x))
            if self.config.hidden_activation == "gelu_pytorch_tanh"
            else jax.nn.gelu(self.gate_proj(x))
        )
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Gemma3DecoderLayer(nnx.Module):
    """Gemma3 decoder layer combining attention and MLP."""

    def __init__(self, layer_idx: int, config: Gemma3Config, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        # Alternate global/local per layer
        attention_type = "global" if layer_idx % 6 == 5 else "local"
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, rngs=rngs
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, rngs=rngs
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, rngs=rngs
        )

        self.self_attn = Gemma3Attention(
            layer_idx, config, attention_type=attention_type, rngs=rngs
        )
        self.mlp = Gemma3MLP(config, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        position_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,  # Boolean Input Padding Mask [B, kv_len]
        cache: tuple[jnp.ndarray, jnp.ndarray] | None = None,  # (k_cache, v_cache)
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1. Pre-Attention Norm and Attention
        residual = x
        hidden_states = self.input_layernorm(x)
        attn_output, updated_cache = self.self_attn(
            hidden_states, position_ids, attention_mask, cache, deterministic=deterministic
        )
        hidden_states = self.post_attention_layernorm(attn_output)
        hidden_states = residual + hidden_states

        # 2. Pre-MLP Norm and MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, updated_cache


class Gemma3ForCausalLM(BaseModel):
    """Gemma3 model for causal language modeling."""

    config: Gemma3Config  # This helps to fix a mypy issue

    def __init__(self, config: Gemma3Config, *, rngs: nnx.Rngs) -> None:
        super().__init__(config, dtype=config.dtype, param_dtype=config.param_dtype, rngs=rngs)
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.layers = [
            Gemma3DecoderLayer(idx, config, rngs=rngs) for idx in range(config.num_hidden_layers)
        ]
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        input_ids: jnp.ndarray,  # [B, S]
        position_ids: jnp.ndarray | None = None,
        attention_mask: jnp.ndarray
        | None = None,  # [B, S], True for valid tokens, used for padding
        cache: list[tuple[jnp.ndarray, jnp.ndarray]]
        | None = None,  # List of (k_cache, v_cache) per layer
        deterministic: bool = True,
        use_cache: bool | None = None,
    ) -> tuple[jnp.ndarray, list[tuple[jnp.ndarray, jnp.ndarray]] | None]:
        batch_size, seq_length = input_ids.shape

        # --- Input Embeddings ---
        hidden_states = self.embed_tokens(input_ids)
        # Scale embeddings by sqrt(hidden_size)
        hidden_states = hidden_states * jnp.sqrt(float(self.config.hidden_size))
        hidden_states = hidden_states.astype(self.config.dtype)

        # --- Prepare Inputs for Layers ---
        # Compute cache and kv sequence lengths
        cache_len = cache[0][0].shape[2] if cache is not None else 0
        kv_seq_len = cache_len + seq_length

        # Prepare position_ids
        if position_ids is None:
            # If no cache, positions are [0, ..., S-1]
            # If cache, positions are [cache_len, ..., cache_len + S - 1]
            position_ids = jnp.arange(cache_len, kv_seq_len, dtype=jnp.int32)[None, :]
            # position_ids should match the query sequence length (seq_length)
            position_ids = position_ids[:, -seq_length:]  # Ensure shape [1, S]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_length))
        elif position_ids.shape[-1] != seq_length:
            raise ValueError(
                "position_ids shape does not match input_ids shape "
                f"(position_ids: {position_ids.shape}, input_ids: {input_ids.shape})"
            )

        # Prepare attention_mask (padding mask)
        # This mask should cover the entire kv_seq_len for keys/values
        if attention_mask is not None:
            # The input attention_mask corresponds to input_ids [B, S]
            # We need to extend it for the cached keys/values.
            # Assume cached tokens were valid.
            if cache is not None:
                # Create mask for cached part (all True)
                cache_mask = jnp.ones((batch_size, cache_len), dtype=jnp.bool_)
                # Concatenate with the input mask
                padding_mask_2d = jnp.concatenate([cache_mask, attention_mask], axis=1)
            else:
                padding_mask_2d = attention_mask  # No cache, use input mask directly

            # Final shape check for the 2D padding mask
            if padding_mask_2d.shape != (batch_size, kv_seq_len):
                raise ValueError(
                    f"Constructed 2D padding mask shape {padding_mask_2d.shape} "
                    f"does not match expected ({batch_size}, {kv_seq_len})"
                )
        else:
            # If no mask provided, assume all tokens are valid
            padding_mask_2d = jnp.ones((batch_size, kv_seq_len), dtype=jnp.bool_)

        # Reshape the 2D boolean padding mask to 4D log-mask for attention calculation
        # Shape: [B, 1, 1, kv_len]. Log-mask: 0.0 for attend, -inf for ignore.
        # Use the validated `padding_mask_2d` here
        attn_mask_4d = jnp.where(
            padding_mask_2d[:, None, None, :], 0.0, jnp.finfo(self.config.dtype).min
        )

        # --- Pass through Decoder Layers ---
        next_cache_list = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states, updated_layer_cache = layer(
                hidden_states,
                position_ids,
                attention_mask=None,  # Pass the 4D log-mask [B, 1, 1, kv_len]
                cache=layer_cache,
                deterministic=deterministic,
            )
            # Always append the updated cache from the layer
            next_cache_list.append(updated_layer_cache)

        hidden_states = self.norm(hidden_states)

        # --- Logits Calculation --- #
        # Final projection using embedding weights (weight tying)
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.embed_tokens.embedding.T
        else:
            # TODO: Add final LM head linear layer if not tied
            raise NotImplementedError("Separate LM head not implemented yet.")

        # Apply final logit soft capping if specified
        if self.config.final_logit_soft_cap is not None:
            logits = logits / self.config.final_logit_soft_cap
            logits = jnp.tanh(logits)
            logits = logits * self.config.final_logit_soft_cap

        # Return cache only if requested
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return logits, next_cache_list if use_cache else None

    def convert_weights_from_hf(
        self, state: nnx.State | dict[str, jnp.ndarray], weights: Iterator[tuple[Any, Any]]
    ) -> None:
        num_layers = len(state["layers"])
        for wholekey, tensor in weights:
            keys = wholekey.split(".")
            if keys[0] != "language_model":
                continue
            if keys[2] == "layers" and int(keys[3]) < num_layers:
                if keys[4] == "self_attn":
                    if keys[5] == "k_norm" or keys[5] == "q_norm":
                        state["layers"][int(keys[3])][keys[4]][keys[5]][
                            "weight"
                        ].value = tensor.astype(self.config.param_dtype)
                    else:
                        state["layers"][int(keys[3])][keys[4]][keys[5]][
                            "kernel"
                        ].value = tensor.T.astype(self.config.param_dtype)
                elif keys[4] == "mlp":
                    state["layers"][int(keys[3])][keys[4]][keys[5]]["kernel"].value = (
                        tensor.T.astype(self.config.param_dtype)
                    )
                elif (
                    keys[4] == "input_layernorm"
                    or keys[4] == "post_attention_layernorm"
                    or keys[4] == "post_feedforward_layernorm"
                    or keys[4] == "pre_feedforward_layernorm"
                ):
                    state["layers"][int(keys[3])][keys[4]]["weight"].value = (
                        tensor.astype(self.config.param_dtype)
                    )
            elif keys[2] == "embed_tokens":
                state["embed_tokens"].embedding.value = tensor.astype(
                    self.config.param_dtype
                )
            elif keys[2] == "norm":
                state["norm"].weight.value = tensor.astype(self.config.param_dtype)

    def generate(
        self,
        input_ids,
        max_new_tokens=20,
        eos_token_id=None,
    ):
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", None)
        )
        # Initialize cache for faster generation
        cache = None
        generated = input_ids
        next_token = input_ids

        for _ in range(max_new_tokens):
            # Get logits and updated cache
            logits, cache = self(
                input_ids=next_token,
                cache=cache,
                use_cache=True,  # Enable KV caching
                deterministic=True,  # No dropout during inference
            )
            # Get next token (use argmax for simplicity)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)
            # Check if we hit the end of sequence
            if next_token[0] == eos_token_id:
                break
            next_token = next_token[:, None]  # Add sequence dimension
            # Append next token
            generated = jnp.concatenate([generated, next_token], axis=1)

        return generated
