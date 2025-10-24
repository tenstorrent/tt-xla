# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen import combine_masks, make_causal_mask
from jax import Array, lax
from transformers.modeling_flax_utils import ACT2FN
from transformers.models.mixtral.configuration_mixtral import MixtralConfig


@dataclass
class FlaxMoeModelOutputWithPast:
    last_hidden_state: Array
    hidden_states: Optional[Tuple[Array]] = None
    attentions: Optional[Tuple[Array]] = None
    router_logits: Optional[Tuple[Array]] = None


@dataclass
class FlaxMoeCausalLMOutput:
    logits: jnp.ndarray
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    router_logits: Optional[Tuple[jnp.ndarray]] = None


class MixtralBlockSparseTop2MLP(nnx.Module):
    """MLP module with sparse routing for Mixtral architecture."""

    def __init__(self, config: MixtralConfig, rngs: nnx.Rngs):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = (
            config.intermediate_size
            if config.intermediate_size is not None
            else 4 * embed_dim
        )
        self.up_proj = nnx.Linear(
            embed_dim,
            inner_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.lecun_normal(),
        )
        self.gate_proj = nnx.Linear(
            embed_dim,
            inner_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.lecun_normal(),
        )
        self.down_proj = nnx.Linear(
            inner_dim,
            embed_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.lecun_normal(),
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        gate_states = self.act_fn(self.up_proj(hidden_states)) * self.gate_proj(
            hidden_states
        )
        hidden_states = self.down_proj(gate_states)
        return hidden_states


class MixtralSparseMoeBlock(nnx.Module):
    """Sparse Mixture of Experts block for Mixtral."""

    def __init__(self, config: MixtralConfig, dtype, rngs: nnx.Rngs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = dtype
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        for i in range(self.num_experts):
            setattr(self, f"experts_{i}", MixtralBlockSparseTop2MLP(config, rngs=rngs))
        self.jitter_noise = config.router_jitter_noise

    def _get_expert(self, idx):
        """Helper to get expert by index"""
        return getattr(self, f"experts_{idx}")

    def __call__(self, hidden_states: Array):
        batch_size, seq_len, hid_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hid_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = jax.nn.softmax(router_logits, axis=1)
        routing_weights, selected_experts = lax.top_k(routing_weights, self.top_k)
        routing_weights /= jnp.sum(routing_weights, axis=-1, keepdims=True)
        # print(routing_weights, selected_experts)
        routing_weights = routing_weights.astype(hidden_states.dtype)

        final_hidden_states = jnp.zeros(
            (batch_size * seq_len, hid_dim), dtype=hidden_states.dtype
        )

        # Create one-hot representation of selected experts
        expert_mask = jax.nn.one_hot(
            selected_experts, num_classes=self.num_experts, dtype=jnp.int8
        ).transpose(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self._get_expert(expert_idx)
            idx, top_x = jnp.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hid_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            final_hidden_states = final_hidden_states.at[top_x].add(
                current_hidden_states
            )

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hid_dim)
        return final_hidden_states, router_logits


class MixtralRMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization for Mixtral."""

    def __init__(self, config: MixtralConfig, dtype=jnp.float32):
        super().__init__()
        self.epsilon = config.rms_norm_eps
        self.weight = nnx.Param(jnp.ones(config.hidden_size, dtype=dtype))

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.epsilon)

        return self.weight * hidden_states.astype(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """Apply rotary position embeddings to query and key tensors."""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def create_sinusoidal_positions(dim, rope_theta):
    inv_freq = 1 / (rope_theta ** (jnp.arange(0, dim, 2) / dim))
    return inv_freq, 1


class MixtralRotaryEmbedding(nnx.Module):
    def __init__(self, config: MixtralConfig, dtype: jnp.dtype = jnp.float32):
        self.config = config
        self.dtype = dtype
        self.rope_theta = self.config.rope_theta
        self.rope_type = "default"

        head_dim = (
            getattr(self.config, "head_dim", None)
            or self.config.hidden_size // self.config.num_attention_heads
        )
        self.inv_freq, self.attention_scaling = create_sinusoidal_positions(
            head_dim, self.rope_theta
        )

    def __call__(self, x, position_ids=None):
        if position_ids is None:
            position_ids = jnp.tile(jnp.arange(x.shape[-2]), (x.shape[0], 1))
        # (B, T, S), (B, T)
        inv_freq_expanded = jnp.expand_dims(self.inv_freq, axis=(0, 2))
        inv_freq_expanded = jnp.tile(inv_freq_expanded, (position_ids.shape[0], 1, 1))
        # print(inv_freq_expanded.shape)
        # inv_freq_expanded = jnp.repeat(inv_freq_expanded, position_ids.shape[0], axis=0)
        # Create expanded position IDs tensor
        position_ids_expanded = jnp.expand_dims(position_ids, axis=1)

        # Compute frequencies
        # Force float32 precision for this computation
        orig_dtype = x.dtype
        freqs = jnp.matmul(
            inv_freq_expanded.astype(jnp.float32),
            position_ids_expanded.astype(jnp.float32),
        )
        freqs = jnp.transpose(freqs, (0, 2, 1))
        # Concatenate frequencies for sin and cos
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        # Compute cos and sin with scaling
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        # Return with original dtype
        return cos.astype(orig_dtype), sin.astype(orig_dtype)


class MixtralAttention(nnx.Module):
    def __init__(self, config: MixtralConfig, dtype: jnp.dtype, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = (
            getattr(self.config, "head_dim", None)
            or self.config.hidden_size // self.config.num_attention_heads
        )
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.num_key_value_heads = self.config.num_key_value_heads

        self.q_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # Key projection
        self.k_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # Value projection
        self.v_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # Output projection
        self.o_proj = nnx.Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Create rotary embeddings and causal mask
        casual_mask = make_causal_mask(
            jnp.ones((1, self.config.max_position_embeddings), dtype="bool"),
            dtype="bool",
        )
        self.causal_mask = jnp.tril(casual_mask)
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        # Initialize cache variables
        self.cached_key = None
        self.cached_value = None
        self.cache_index = None

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        *batch_dims, max_length, num_heads, depth_per_head = self.cached_key.shape
        # update key, value caches with our new 1d spatial slices
        cur_index = self.cache_index
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(self.cached_key, key, indices)
        value = lax.dynamic_update_slice(self.cached_value, value, indices)
        self.cached_key = key
        self.cached_value = value
        num_updated_cache_vectors = query.shape[1]
        self.cache_index = self.cache_index + num_updated_cache_vectors
        # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
            tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
        )
        attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,  # (B, T, embed)
        position_ids,  # (1, T)
        attention_mask,
        position_embeddings,
        deterministic: bool = False,
        init_cache: Optional[bool] = True,
        past_key_value=None,
        **kwargs,
    ):
        batch_size, seq_len, embed = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        # cos, sin = position_ids
        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        # print(query_states.shape, key_states.shape, value_states.shape)
        query_length, key_length = query_states.shape[1], key_states.shape[1]
        mask_shift = self.cache_index
        max_decoder_length = self.cached_key.shape[1]
        causal_mask = lax.dynamic_slice(
            self.causal_mask,
            (0, 0, mask_shift, 0),
            (1, 1, query_length, max_decoder_length),
        )
        # print(layer_idx)
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )
        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)
        # Later in the code:

        key_states, value_states, attention_mask = self._concatenate_to_cache(
            key_states, value_states, query_states, attention_mask
        )
        # print(query_states.shape, key_states.shape, value_states.shape)
        key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
        value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )

        def attention_fn(q, k, v, bias):
            # Transpose for attention

            q = jnp.transpose(q, (0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]

            k = jnp.transpose(k, (0, 2, 1, 3))

            v = jnp.transpose(v, (0, 2, 1, 3))

            attention_scores = (
                jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scaling
            )

            attention_scores = attention_scores + bias

            attn_weights = jax.nn.softmax(attention_scores, axis=-1).astype(self.dtype)

            attn_output = jnp.matmul(attn_weights, v)

            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

            return attn_output, attn_weights

        attn_output, attn_weights = attention_fn(
            query_states, key_states, value_states, attention_bias
        )

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        # attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, None, 'tp'))
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class MixtralDecoderLayer(nnx.Module):
    def __init__(
        self, config: MixtralConfig, rngs: nnx.Rngs, layer_idx: int, dtype=jnp.float32
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype

        self.input_norm = MixtralRMSNorm(config, dtype=dtype)
        self.attn = MixtralAttention(config, dtype=dtype, rngs=rngs)
        self.block_sparse_moe = MixtralSparseMoeBlock(config, dtype=dtype, rngs=rngs)
        self.attn_norm = MixtralRMSNorm(config, dtype=dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        deterministic: bool = False,
        output_router_logits: Optional[bool] = False,
        init_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[Array, Array]] = None,
        **kwargs,
    ):
        residual = hidden_states
        # print(hidden_states)
        hidden_states = self.input_norm(hidden_states)
        # Self Attentio
        hidden_states, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            deterministic=deterministic,
            past_key_value=past_key_value,
            init_cache=init_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.attn_norm(hidden_states)

        hidden_states, router_logits = self.block_sparse_moe(hidden_states)

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class MixtralModel(nnx.Module):
    """Mixtral model implementation using NNX."""

    def __init__(self, config: MixtralConfig, dtype=jnp.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embedding_init=nnx.initializers.lecun_normal(),
            rngs=nnx.Rngs(0),
        )

        self.layers = [
            MixtralDecoderLayer(
                config=config, layer_idx=layer_idx, dtype=dtype, rngs=nnx.Rngs(0)
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm = MixtralRMSNorm(config, dtype=dtype)
        self.rotary_emb = MixtralRotaryEmbedding(config, dtype=dtype)

    def __call__(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        deterministic: bool = False,
        input_embeds: Optional[Array] = None,
        init_cache: Optional[bool] = None,
        cache=None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        init_cache = init_cache if init_cache is not None else self.config.init_cache

        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
            if self.padding_idx is not None:
                mask = (input_ids != self.padding_idx).astype(jnp.float32)
                embeddings = input_embeds * mask[..., None]

        if cache is None and init_cache:
            cache = [None for _ in range(len(self.layers))]
        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                init_cache=init_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return FlaxMoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class FlaxMixtralForCausalLM(nnx.Module):
    """Mixtral model with a language modeling head implemented in NNX."""

    def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=None):
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Save configuration
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype

        # Create model components
        self.model = MixtralModel(
            config=config,
            dtype=dtype,
        )

        # Create LM head as a linear layer
        self.lm_head = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        inputs_embeds=None,
        init_cache=True,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
    ):
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )

        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return FlaxMoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask=None):
        batch_size, _ = input_ids.shape

        for i in range(self.config.num_hidden_layers):
            self.model.layers[i].attn.cached_key = jnp.zeros(
                (
                    batch_size,
                    max_length,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ),
                dtype=jnp.float32,
            )
            self.model.layers[i].attn.cached_value = jnp.zeros(
                (
                    batch_size,
                    max_length,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ),
                dtype=jnp.float32,
            )
            self.model.layers[i].attn.cache_index = jnp.array(0, dtype=jnp.int32)

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        if attention_mask is not None:
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )

        position_ids = jnp.cumsum(extended_attention_mask, axis=-1) - 1

        return input_ids, extended_attention_mask, position_ids

    def generate(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        max_new_tokens=20,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """Generate text efficiently using properly initialized KV caching in NNX."""
        # Set defaults for special tokens
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", None)
        )

        # Initialize generation variables
        batch_size, seq_length = input_ids.shape
        has_reached_eos = jnp.zeros(batch_size, dtype=jnp.bool_)

        if position_ids is None:
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1

        # Now process the real prompt to fill in the cache for actual tokens
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            init_cache=True,
        )
        # Get next token prediction
        next_token_logits = outputs.logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = next_token[:, None]  # Add sequence dimension

        all_token_ids = jnp.concatenate([input_ids, next_token], axis=1)

        # Check early stopping conditions
        if eos_token_id is not None:
            has_reached_eos = has_reached_eos | (next_token[:, 0] == eos_token_id)

        cur_len = seq_length
        # Start auto-regressive generation loop
        for i in range(1, max_new_tokens):
            # Early exit if all sequences have reached EOS
            if eos_token_id is not None and jnp.all(has_reached_eos):
                break

            # Generate next token using cache
            new_position_ids = position_ids[:, cur_len].reshape(-1, 1)
            outputs = self(
                input_ids=next_token,  # Only process the new token
                attention_mask=attention_mask,
                init_cache=True,
                position_ids=new_position_ids,
            )
            cur_len += 1
            # Get logits and predict next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = jnp.argmax(next_token_logits, axis=-1)
            next_token = next_token[:, None]  # Add sequence dimension
            # Add new token to results
            all_token_ids = jnp.concatenate([all_token_ids, next_token], axis=1)

            # Update EOS tracking
            if eos_token_id is not None:
                has_reached_eos = has_reached_eos | (next_token[:, 0] == eos_token_id)

        return all_token_ids
