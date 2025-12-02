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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_config import axis_name, cpu_devices, device_mesh, num_devices
from transformers.modeling_flax_utils import ACT2FN
from transformers.models.mixtral.configuration_mixtral import MixtralConfig


@dataclass
class FlaxMoeModelOutputWithPast:
    last_hidden_state: Array
    hidden_states: Optional[Tuple[Array]] = None
    attentions: Optional[Tuple[Array]] = None
    router_logits: Optional[Tuple[Array]] = None
    past_key_values: Optional[Tuple[Tuple[Array]]] = None


@dataclass
class FlaxMoeCausalLMOutput:
    logits: jnp.ndarray
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    router_logits: Optional[Tuple[jnp.ndarray]] = None
    past_key_values: Optional[Tuple[Tuple[Array]]] = None


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
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
        )
        self.gate_proj = nnx.Linear(
            embed_dim,
            inner_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
        )
        self.down_proj = nnx.Linear(
            inner_dim,
            embed_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("X", None)
            ),
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states):
        gate_states = self.act_fn(self.up_proj(hidden_states)) * self.gate_proj(
            hidden_states
        )
        hidden_states = self.down_proj(gate_states)
        return hidden_states


class MixtralSparseMoeBlock(nnx.Module):
    """Sparse Mixture of Experts block for Mixtral with expert parallelism."""

    def __init__(self, config: MixtralConfig, dtype, rngs: nnx.Rngs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = dtype
        # Router (gate) is replicated across all devices
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), P()),
            rngs=rngs,
        )

        self.experts = []
        for i in range(self.num_experts):
            expert = MixtralBlockSparseTop2MLP(config, rngs=rngs)
            self.experts.append(expert)

        self.jitter_noise = config.router_jitter_noise

    def __call__(self, hidden_states):
        batch_size, seq_len, hid_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hid_dim)

        router_logits = self.gate(hidden_states)

        routing_weights = jax.nn.softmax(router_logits, axis=1)
        routing_weights, selected_experts = lax.top_k(routing_weights, self.top_k)
        routing_weights /= jnp.sum(routing_weights, axis=-1, keepdims=True)

        routing_weights = routing_weights.astype(hidden_states.dtype)
        final_hidden_states = jnp.zeros(
            (batch_size * seq_len, hid_dim), dtype=hidden_states.dtype
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            token_masks = jnp.zeros((self.top_k, batch_size * seq_len), dtype=jnp.bool_)
            for k in range(self.top_k):
                token_masks = token_masks.at[k].set(
                    selected_experts[:, k] == expert_idx
                )

            token_mask = jnp.any(token_masks, axis=0)

            expert_weights = jnp.zeros(
                (batch_size * seq_len), dtype=routing_weights.dtype
            )

            for k in range(self.top_k):
                expert_weights = expert_weights + jnp.where(
                    selected_experts[:, k] == expert_idx, routing_weights[:, k], 0.0
                )

            expert_output = expert_layer(hidden_states)

            masked_output = (
                jnp.where(
                    token_mask[:, None],  # Broadcast mask across hidden dimension
                    expert_output,
                    0.0,
                )
                * expert_weights[:, None]
            )

            final_hidden_states = final_hidden_states + masked_output

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hid_dim)

        final_hidden_states = jax.lax.psum(final_hidden_states, axis_name="X")

        return final_hidden_states, router_logits


class MixtralRMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization for Mixtral."""

    def __init__(self, config: MixtralConfig, dtype=jnp.float32):
        super().__init__()
        self.epsilon = config.rms_norm_eps
        self.weight = nnx.Param(jnp.ones(config.hidden_size, dtype=dtype), sharding=P())

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
        inv_freq, attention_scaling = create_sinusoidal_positions(
            head_dim, self.rope_theta
        )
        self.inv_freq = nnx.Variable(inv_freq)
        self.attention_scaling = attention_scaling

    def __call__(self, x, position_ids=None):
        if position_ids is None:
            position_ids = jnp.tile(jnp.arange(x.shape[-2]), (x.shape[0], 1))
        inv_freq_expanded = jnp.expand_dims(self.inv_freq.value, axis=(0, 2))
        inv_freq_expanded = jnp.tile(inv_freq_expanded, (position_ids.shape[0], 1, 1))
        position_ids_expanded = jnp.expand_dims(position_ids, axis=1)

        orig_dtype = x.dtype
        freqs = jnp.matmul(
            inv_freq_expanded.astype(jnp.float32),
            position_ids_expanded.astype(jnp.float32),
        )
        freqs = jnp.transpose(freqs, (0, 2, 1))

        emb = jnp.concatenate([freqs, freqs], axis=-1)
        # Compute cos and sin with scaling
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        # Return with original dtype
        return cos.astype(orig_dtype), sin.astype(orig_dtype)


class MixtralAttention(nnx.Module):
    def __init__(
        self, layer_idx: int, config: MixtralConfig, dtype: jnp.dtype, rngs: nnx.Rngs
    ):
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
        self.layer_idx = layer_idx

        self.q_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
            rngs=rngs,
        )

        # Key projection
        self.k_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
            rngs=rngs,
        )

        # Value projection
        self.v_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
            rngs=rngs,
        )

        # Output projection
        self.o_proj = nnx.Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("X", None)
            ),
            rngs=rngs,
        )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        casual_mask = make_causal_mask(
            jnp.ones(
                (1, 100), dtype="bool"
            ),  # TODO:this has to be self.config.max_position_embeddings but i need it smaller
            dtype="bool",
        )
        self.causal_mask = nnx.Variable(jnp.tril(casual_mask))
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

    def _concatenate_to_cache(self, key, value, query, attention_mask, past_key_values):
        idx = self.layer_idx
        cached_key = past_key_values[f"layer_{idx}"]["cached_key"]
        cached_value = past_key_values[f"layer_{idx}"]["cached_value"]
        cache_index = past_key_values[f"layer_{idx}"]["cache_index"]
        *batch_dims, max_length, num_heads, depth_per_head = cached_key.shape
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key, key, indices)
        value = lax.dynamic_update_slice(cached_value, value, indices)
        cached_key = key
        cached_value = value
        num_updated_cache_vectors = query.shape[1]
        cache_index += num_updated_cache_vectors
        # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
            tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
        )
        attention_mask = combine_masks(pad_mask, attention_mask)
        past_key_values[f"layer_{idx}"] = {
            "cached_key": key,
            "cached_value": value,
            "cache_index": cache_index,
        }
        return key, value, attention_mask, past_key_values

    def __call__(
        self,
        hidden_states,  # (B, T, embed)
        position_ids,  # (1, T)
        attention_mask,
        position_embeddings,
        past_key_values,
        init_cache: Optional[bool] = True,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = jax.lax.all_gather(
            jnp.array(query_states), axis_name="X", axis=2, tiled=True
        )
        key_states = jax.lax.all_gather(
            jnp.array(key_states), axis_name="X", axis=2, tiled=True
        )
        value_states = jax.lax.all_gather(
            jnp.array(value_states), axis_name="X", axis=2, tiled=True
        )

        query_states = query_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        query_length = query_states.shape[1]
        idx = self.layer_idx
        cached_key = past_key_values[f"layer_{idx}"]["cached_key"]
        cache_index = past_key_values[f"layer_{idx}"]["cache_index"]

        mask_shift = cache_index
        max_decoder_length = cached_key.shape[1]
        causal_mask = lax.dynamic_slice(
            self.causal_mask.value,
            (0, 0, mask_shift, 0),
            (1, 1, query_length, max_decoder_length),
        )

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )
        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
        )
        attention_mask = combine_masks(attention_mask, causal_mask)
        (
            key_states,
            value_states,
            attention_mask,
            past_key_values,
        ) = self._concatenate_to_cache(
            key_states, value_states, query_states, attention_mask, past_key_values
        )

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

        local_dim = 4096 // 8
        start_idx = jax.lax.axis_index("X") * local_dim
        attn_output_local = jax.lax.dynamic_slice(
            attn_output, (0, 0, start_idx), (batch_size, seq_len, local_dim)
        )

        attn_output = self.o_proj(attn_output_local)
        attn_output = jax.lax.psum(attn_output, axis_name="X")
        return attn_output, attn_weights, past_key_values


class MixtralDecoderLayer(nnx.Module):
    def __init__(
        self, config: MixtralConfig, rngs: nnx.Rngs, layer_idx: int, dtype=jnp.float32
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype

        self.input_norm = MixtralRMSNorm(config, dtype=dtype)
        self.attn = MixtralAttention(
            config=config, dtype=dtype, rngs=rngs, layer_idx=layer_idx
        )
        self.block_sparse_moe = MixtralSparseMoeBlock(config, dtype=dtype, rngs=rngs)
        self.attn_norm = MixtralRMSNorm(config, dtype=dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        init_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[Array, Array]] = None,
        **kwargs,
    ):

        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, self_attn_weights, past_key_values = self.attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            init_cache=init_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        hidden_states, router_logits = self.block_sparse_moe(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        outputs += (past_key_values,)

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
            embedding_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
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
        past_key_values=None,
        input_embeds: Optional[Array] = None,
        init_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
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
        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        hidden_states = jax.lax.all_gather(
            hidden_states, axis_name="X", axis=-1, tiled=True
        )

        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                init_cache=init_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            past_key_values = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

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
            past_key_values=past_key_values,
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
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "X")
            ),
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        init_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
    ):
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            init_cache=init_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )

        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        # return logits, past_key_values
        return FlaxMoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            past_key_values=past_key_values,
        )

    @staticmethod
    def load_hf_params(state, pt_model, config) -> nnx.state:
        embeddings = pt_model.model.embed_tokens.weight.detach().cpu().numpy()
        # Save LM head
        lm_head = pt_model.lm_head.weight.detach().cpu().numpy()
        state["model"]["embed_tokens"]["embedding"].value = jnp.array(embeddings)
        state["lm_head"]["kernel"].value = jnp.array(lm_head.T)

        final_norm = pt_model.model.norm.weight.detach().cpu().numpy()
        state["model"]["norm"]["weight"].value = jnp.array(final_norm)

        for i in range(config.num_hidden_layers):
            layer = pt_model.model.layers[i]
            layerJax = state["model"]["layers"][i]
            input_layernorm = layer.input_layernorm.weight.detach().cpu().numpy()
            post_attention_layernorm = (
                layer.post_attention_layernorm.weight.detach().cpu().numpy()
            )

            attn_q = layer.self_attn.q_proj.weight.detach().cpu().numpy()
            attn_k = layer.self_attn.k_proj.weight.detach().cpu().numpy()
            attn_v = layer.self_attn.v_proj.weight.detach().cpu().numpy()
            attn_o = layer.self_attn.o_proj.weight.detach().cpu().numpy()

            layerJax["input_norm"]["weight"].value = jnp.array(input_layernorm)
            layerJax["attn_norm"]["weight"].value = jnp.array(post_attention_layernorm)

            layerJax["attn"]["q_proj"]["kernel"].value = jnp.array(attn_q.T)
            layerJax["attn"]["k_proj"]["kernel"].value = jnp.array(attn_k.T)
            layerJax["attn"]["v_proj"]["kernel"].value = jnp.array(attn_v.T)
            layerJax["attn"]["o_proj"]["kernel"].value = jnp.array(attn_o.T)

            moe = layer.block_sparse_moe
            moe_gate = moe.gate.weight.detach().cpu().numpy()

            layerJax["block_sparse_moe"]["gate"]["kernel"].value = jnp.array(moe_gate.T)

            num_experts = config.num_local_experts

            for j in range(num_experts):
                w1 = moe.experts[j].w1.weight.detach().cpu().numpy()
                w2 = moe.experts[j].w2.weight.detach().cpu().numpy()
                w3 = moe.experts[j].w3.weight.detach().cpu().numpy()
                expert = layerJax["block_sparse_moe"]["experts"][j]

                expert["gate_proj"]["kernel"].value = jnp.array(w3.T)
                expert["up_proj"]["kernel"].value = jnp.array(w1.T)
                expert["down_proj"]["kernel"].value = jnp.array(w2.T)

        return state

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask=None):
        batch_size, _ = input_ids.shape
        past_key_values = {}

        for i in range(self.config.num_hidden_layers):
            layer_key = f"layer_{i}"
            past_key_values[layer_key] = {
                "cached_key": jnp.zeros(
                    (
                        batch_size,
                        max_length,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ),
                    dtype=jnp.float32,
                ),
                "cached_value": jnp.zeros(
                    (
                        batch_size,
                        max_length,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ),
                    dtype=jnp.float32,
                ),
                "cache_index": jnp.array(0, dtype=jnp.int32),
            }

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")

        if attention_mask is not None:
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )

        position_ids = jnp.cumsum(extended_attention_mask, axis=-1) - 1

        return input_ids, extended_attention_mask, position_ids, past_key_values

    def prepare_sharding(self):

        state = nnx.state(self)

        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)

        nnx.update(self, sharded_state)

        cache_specs = {}
        for i in range(self.config.num_hidden_layers):
            layer_key = f"layer_{i}"
            cache_specs[layer_key] = {
                "cached_key": P(),  # Replicated
                "cached_value": P(),  # Replicated
                "cache_index": P(),  # Replicated
            }

        return cache_specs

    def generate(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        max_new_tokens=20,
        pad_token_id=None,
        eos_token_id=None,
    ):
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.config, "eos_token_id", None)
        )

        batch_size, seq_len = input_ids.shape
        max_len = max_new_tokens + seq_len

        has_reached_eos = jnp.zeros(batch_size, dtype=jnp.bool_)

        if position_ids is None:
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1

        def model_forward(
            model_state, input_ids, attention_mask, position_ids, past_key_values
        ):
            # Call the model
            model = nnx.merge(nnx.graphdef(self), model_state)

            outputs = model(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
            )
            return outputs.logits, outputs.past_key_values

        with device_mesh:
            cache_specs = self.prepare_sharding()

            model_state = nnx.state(self)
            model_specs = nnx.get_partition_spec(model_state)

            sharded_forward = nnx.shard_map(
                model_forward,
                mesh=device_mesh,
                in_specs=(model_specs, P(), P(), P(), cache_specs),
                out_specs=(P(None, None, "X"), cache_specs),
                check_rep=False,
            )

            jit_forward = nnx.jit(sharded_forward, donate_argnums=(4,))

        all_token_ids = jnp.zeros((batch_size, max_len), dtype=input_ids.dtype)
        all_token_ids = all_token_ids.at[:, :seq_len].set(input_ids)

        # First forward pass
        logits, past_key_values = jit_forward(
            model_state,
            input_ids,
            attention_mask,
            position_ids[:, :seq_len],
            past_key_values,
        )

        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        all_token_ids = all_token_ids.at[:, seq_len].set(next_token)

        cur_len = seq_len
        for step in range(1, max_new_tokens):
            if eos_token_id is not None and jnp.all(has_reached_eos):
                break

            next_token = next_token[:, None]

            new_position_ids = position_ids[:, cur_len].reshape(-1, 1)

            logits_step, past_key_values = jit_forward(
                model_state,
                next_token,
                attention_mask,
                new_position_ids,
                past_key_values,
            )

            cur_len += 1
            next_token_logits = logits_step[:, 0, :]
            next_token = jnp.argmax(next_token_logits, axis=-1)

            all_token_ids = lax.dynamic_update_slice(
                all_token_ids, next_token[:, None], (0, cur_len)
            )

            if eos_token_id is not None:
                has_reached_eos = has_reached_eos | (next_token == eos_token_id)

        return all_token_ids
