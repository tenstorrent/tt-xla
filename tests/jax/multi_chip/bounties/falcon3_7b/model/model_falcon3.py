# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# coding=utf-8
"""Flax Falcon3 model."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from model.configuration_falcon3 import Falcon3Config
from safetensors.flax import load_file
from transformers.models.llama.configuration_llama import LlamaConfig


def create_sinusoidal_positions(num_pos, theta, dim):
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype(jnp.float32)

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]),
        axis=-1,
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class FlaxFalcon3RMSNorm(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param(
            "weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size
        )

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxFalcon3RotaryEmbedding(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sincos = create_sinusoidal_positions(
            self.config.max_position_embeddings, self.config.rope_theta, head_dim
        )

    def __call__(self, key, query, position_ids):
        sincos = self.sincos[position_ids]
        sin_pos, cos_pos = jnp.split(sincos, 2, axis=-1)
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)

        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)

        return key, query


class FlaxFalcon3Attention(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        dense = partial(
            nn.Dense,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj = dense(self.num_heads * self.head_dim)
        self.k_proj = dense(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense(self.embed_dim)
        self.causal_mask = make_causal_mask(
            jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool"
        )
        self.rotary_emb = FlaxFalcon3RotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if is_initialized:
            *batch_dims, max_length, _, _ = cached_key.value.shape

            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)

            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors

            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[jax.Array, ...]:

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)

        key, query = self.rotary_emb(key, query, position_ids)
        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(
            causal_mask, (batch_size,) + causal_mask.shape[1:]
        )
        if attention_mask is not None:
            attention_mask = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
            )
            attention_mask = combine_masks(attention_mask, causal_mask)
        else:
            attention_mask = causal_mask

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(
                key, value, query, attention_mask
            )

        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype

        # compute attention weights using dot product attention
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxFalcon3MLP(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = (
            self.config.intermediate_size
            if self.config.intermediate_size is not None
            else 4 * embed_dim
        )

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.act = nn.silu

        self.gate_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.down_proj = nn.Dense(
            embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.up_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:

        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class FlaxFalcon3DecoderLayer(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxFalcon3RMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxFalcon3Attention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxFalcon3RMSNorm(
            self.config, dtype=self.dtype
        )
        self.mlp = FlaxFalcon3MLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[jax.Array, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states

        return (hidden_states,) + outputs[1:]


class FlaxFalcon3LayerCollection(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [
            FlaxFalcon3DecoderLayer(self.config, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[jax.Array, ...]:

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            print(f"Processing block {block.name}...")  # Debugging output
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this can hold None values, but `FlaxFalcon3Model` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxFalcon3Model(nn.Module):
    config: Falcon3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(
            stddev=self.config.initializer_range
        )
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        self.layers = FlaxFalcon3LayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxFalcon3RMSNorm(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids: jax.Array = None,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        deterministic: bool = True,
        init_cache: bool = False,
        logits_to_keep: Union[int, jax.Array] = 0,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[jax.Array, ...], dict]:

        input_embeds = self.embed_tokens(input_ids.astype("i4"))
        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        hidden_states = self.norm(hidden_states[:, slice_indices, :])

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        else:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": outputs[1],
                "attentions": outputs[-1],
            }


class FlaxFalcon3ForCausalLMModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxFalcon3Model(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )

    def __call__(
        self,
        input_ids: jax.Array = None,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[jax.Array, ...], dict]:
        """
        Forward pass for the Falcon3 model for causal language modeling.

        Args:
            input_ids (jax.Array): Input token IDs of shape (batch_size, seq_len).
            attention_mask (jax.Array): Attention mask of shape (batch_size, seq_len).
            position_ids (jax.Array): Position IDs of shape (batch_size, seq_len).
            deterministic (bool): Whether to use dropout or not.
            init_cache (bool): Whether to initialize the cache for fast autoregressive decoding.
            output_attentions (bool): Whether to return attention weights across the layers.
            output_hidden_states (bool): Whether to return hidden states across the layers.
            return_dict (bool): Whether to return a dictionary or a tuple.
        Returns:
            dict or tuple: If `return_dict` is True, returns a dictionary with keys 'logits', 'hidden_states', and 'attentions'.
        """

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs["last_hidden_state"]

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]
        else:
            return {
                "logits": lm_logits,
                "hidden_states": outputs["hidden_states"],
                "attentions": outputs["attentions"],
            }


class FlaxFalcon3ForCausalLM:
    config_class = Falcon3Config
    base_model_prefix = "falcon3"

    def __init__(self, config: Falcon3Config, dtype: jnp.dtype = jnp.float32):
        self.config = config
        self.model = FlaxFalcon3ForCausalLMModule(config, dtype=dtype)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def init_weights(
        self,
        rng: Optional[Union[int, jax.random.PRNGKey]] = None,
        input_ids: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        init_cache: bool = True,
    ) -> FrozenDict:
        """
        Initialize the model weights.

        Args:
            rng (Optional[jax.random.PRNGKey]): Random key for initialization.
            input_ids (Optional[jax.Array]): Input token IDs for the initialization of the model.
            attention_mask (Optional[jax.Array]): Attention mask for the initialization of the model.
            position_ids (Optional[jax.Array]): Position IDs for the initialization of the model.
            init_cache (bool): Whether to initialize the cache in initializationfor fast autoregressive decoding.
        Returns:
            dict: Initialized model parameters.
        """
        if rng is None:
            rng = jax.random.PRNGKey(42)
        elif isinstance(rng, int):
            rng = jax.random.PRNGKey(rng)

        input_ids = (
            input_ids if input_ids is not None else jnp.zeros((2, 4), dtype=jnp.int32)
        )
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else jnp.ones((2, 4), dtype=jnp.int32)
        )
        position_ids = (
            position_ids
            if position_ids is not None
            else jnp.arange(4, dtype=jnp.int32)[None, :].repeat(2, axis=0)
        )

        model_params = self.model.init(
            rng, input_ids, attention_mask, position_ids, init_cache=init_cache
        )

        return model_params

    def _download_weights(self) -> Path:

        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id="tiiuae/Falcon3-7B-Instruct",
            local_dir="./hf_weights",
        )

        return Path(path)

    def convert_from_hf_weights(
        self,
        config: Falcon3Config,
        batch_size: int,
        max_len: int,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """
        Convert weights from a Hugging Face checkpoint to a Flax model.

        Args:
            config (Falcon3Config): Configuration for the model.
            batch_size (int): Batch size for the model.
            max_len (int): Maximum sequence length for the model (including future auto-regressive generated tokens).
            checkpoint_path (str or Path): Path to the Hugging Face checkpoint directory.
        Returns:
            dict: Flax parameters converted from the Hugging Face checkpoint.
        """

        if checkpoint_path is None:
            # If no checkpoint path is provided, download the Falcon3-7B-Instruct weights
            # in the default directory `./hf_weights`.
            checkpoint_path = self._download_weights()
        elif isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        ckpt_paths = sorted(checkpoint_path.glob("*.safetensors"))
        ckpts = []
        for ckpt_path in ckpt_paths:
            checkpoint = load_file(str(ckpt_path))
            ckpts.append(checkpoint)

        def from_checkpoint(key, axis=0):
            weights = jnp.concatenate(
                [ckpt[key] for ckpt in ckpts if key in ckpt], axis=axis
            )
            # Convert weights to float32 if they are bfloat16
            if weights.dtype == jnp.bfloat16:
                weights = weights.astype(jnp.float32)
            return weights

        print(f"Loaded {len(ckpts)} checkpoints from {checkpoint_path}")
        flax_params = {
            "params": {
                "model": {
                    "embed_tokens": {
                        "embedding": from_checkpoint("model.embed_tokens.weight")
                    },
                    "norm": {"weight": from_checkpoint("model.norm.weight")},
                    "layers": {
                        f"{layer}": {
                            "input_layernorm": {
                                "weight": from_checkpoint(
                                    f"model.layers.{layer}.input_layernorm.weight"
                                )
                            },
                            "self_attn": {
                                "q_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.self_attn.q_proj.weight",
                                        axis=0,
                                    ).transpose()
                                },
                                "k_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.self_attn.k_proj.weight",
                                        axis=0,
                                    ).transpose()
                                },
                                "v_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.self_attn.v_proj.weight",
                                        axis=0,
                                    ).transpose()
                                },
                                "o_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.self_attn.o_proj.weight",
                                        axis=1,
                                    ).transpose()
                                },
                            },
                            "post_attention_layernorm": {
                                "weight": from_checkpoint(
                                    f"model.layers.{layer}.post_attention_layernorm.weight"
                                )
                            },
                            "mlp": {
                                "up_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.mlp.up_proj.weight",
                                        axis=0,
                                    ).transpose()
                                },
                                "gate_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.mlp.gate_proj.weight",
                                        axis=0,
                                    ).transpose()
                                },
                                "down_proj": {
                                    "kernel": from_checkpoint(
                                        f"model.layers.{layer}.mlp.down_proj.weight",
                                        axis=1,
                                    ).transpose()
                                },
                            },
                        }
                        for layer in range(config.num_hidden_layers)
                    },
                },
                "lm_head": {"kernel": from_checkpoint("lm_head.weight").transpose()},
            },
            "cache": {
                "model": {
                    "layers": {
                        f"{layer}": {
                            "self_attn": {
                                "cached_key": jnp.zeros(
                                    (
                                        batch_size,
                                        max_len,
                                        config.num_key_value_heads,
                                        config.head_dim,
                                    ),
                                    dtype=jnp.float32,
                                ),
                                "cached_value": jnp.zeros(
                                    (
                                        batch_size,
                                        max_len,
                                        config.num_key_value_heads,
                                        config.head_dim,
                                    ),
                                    dtype=jnp.float32,
                                ),
                                "cache_index": jnp.array(0, dtype=jnp.int32),
                            }
                        }
                        for layer in range(self.config.num_hidden_layers)
                    },
                }
            },
        }
        del ckpts
        return flax_params

    def get_partitioning_rules(self) -> dict:

        partitioning_rules = {
            "params": {
                "model": {
                    "embed_tokens": {"embedding": P("tp", None)},
                    "norm": {"weight": P()},
                    "layers": {
                        f"{layer}": {
                            "input_layernorm": {"weight": P()},
                            "self_attn": {
                                "q_proj": {"kernel": P(None, "tp")},
                                "k_proj": {"kernel": P(None, "tp")},
                                "v_proj": {"kernel": P(None, "tp")},
                                "o_proj": {"kernel": P("tp", None)},
                            },
                            "post_attention_layernorm": {"weight": P()},
                            "mlp": {
                                "up_proj": {"kernel": P(None, "tp")},
                                "gate_proj": {"kernel": P(None, "tp")},
                                "down_proj": {"kernel": P("tp", None)},
                            },
                        }
                        for layer in range(self.config.num_hidden_layers)
                    },
                },
                "lm_head": {"kernel": P(None, "tp")},
            },
            "cache": {
                "model": {
                    "layers": {
                        f"{layer}": {
                            "self_attn": {
                                "cached_key": P(),
                                "cached_value": P(),
                                "cache_index": P(),
                            }
                        }
                        for layer in range(self.config.num_hidden_layers)
                    },
                }
            },
        }
        return partitioning_rules

    def prepare_inputs_for_generation(
        self,
        input_ids: jax.Array,
        max_length: int,
        attention_mask: Optional[jax.Array] = None,
    ) -> dict:
        """
        Prepare inputs for generation by extending the attention mask and corresponding position IDs.

        Args:
            input_ids (jax.Array): Input token IDs of shape (batch_size, seq_len).
            max_length (int): Maximum length of the sequence.
            attention_mask (Optional[jax.Array]): Attention mask of shape (batch_size, seq_len).
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and position_ids.
        """

        batch_size, seq_length = input_ids.shape

        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = extended_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "input_ids": input_ids,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def shard_parameters(
        self, params: dict, device_mesh: Mesh, rules: Optional[dict] = None
    ) -> dict:
        """
        Apply sharding to loaded parameters based on partitioning rules.

        Args:
            params (dict): Flax parameters for the model.
            device_mesh (Mesh): JAX device mesh for sharding.
            rules (Optional[dict]): Partitioning rules for sharding.
                If None, uses default rules, defined inside model class.
        Returns:
            dict: Sharded parameters.
        """
        params = flatten_dict(params)
        rules = rules if rules is not None else self.get_partitioning_rules()
        rules = flatten_dict(rules)

        sharded_params = {}

        for param_key, param_value in params.items():
            # Find the corresponding rule
            rule_key = param_key  # Adjust if your rules have different structure
            if rule_key in rules:
                partition_spec = rules[rule_key]

                sharding = NamedSharding(device_mesh, partition_spec)
                sharded_param = jax.device_put(param_value, sharding)
                sharded_params[param_key] = sharded_param
            else:
                sharded_params[param_key] = param_value

        return unflatten_dict(sharded_params)

    def shard_inputs(
        self,
        device_mesh: Mesh,
        input_ids: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, ...]:
        """
        Shard inputs for the model based on the device mesh.

        Args:
            device_mesh (Mesh): JAX device mesh for sharding.
            input_ids (jax.Array): Input token IDs of shape (batch_size, seq_len).
            attention_mask (jax.Array): Attention mask of shape (batch_size, seq_len).
            position_ids (jax.Array): Position IDs of shape (batch_size, seq_len).
        Returns:
            Tuple[jax.Array, ...]: Sharded given inputs as a tuple.
        """

        sharding = NamedSharding(device_mesh, P("dp", None))
        outputs = ()

        if input_ids is not None:
            input_ids = jax.lax.with_sharding_constraint(input_ids, sharding)
            outputs += (input_ids,)
        if attention_mask is not None:
            attention_mask = jax.lax.with_sharding_constraint(attention_mask, sharding)
            outputs += (attention_mask,)
        if position_ids is not None:
            position_ids = jax.lax.with_sharding_constraint(position_ids, sharding)
            outputs += (position_ids,)

        return outputs

    def generate(
        self,
        params: dict = None,
        input_ids: jax.Array = None,
        attention_mask: jax.Array = None,
        position_ids: jax.Array = None,
        max_new_tokens: int = 20,
        return_dict: bool = True,
    ) -> jax.Array:
        """
        Generate causal tokens from input using JIT and Key&Value Caching in NN.

        Args:
            params (dict): Flax parameters for the model.
            input_ids (jax.Array): Input token IDs of shape (batch_size, seq_len).
            attention_mask (jax.Array): Attention mask of shape (batch_size, seq_len).
            position_ids (jax.Array): Position IDs of shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.
        Returns:

            jax.Array: Generated token IDs of shape (batch_size, max_length).
        """

        batch_size, seq_len = input_ids.shape
        max_len = seq_len + max_new_tokens

        all_token_ids = input_ids.copy()

        jit_apply = jax.jit(self.model.apply, static_argnames=("return_dict"))

        print(f"Initial pass for loading cache into model...")
        outputs, cache = self.model.apply(
            params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids[:, :seq_len],
            return_dict=return_dict,
            mutable=["cache"],
        )
        params["cache"] = cache["cache"]

        next_token_logits = outputs["logits"][:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)[:, None]
        all_token_ids = jnp.concatenate((all_token_ids, next_token), axis=1)

        while seq_len < max_len:
            print("------", seq_len, "------")  # just to track tokens

            outputs, cache = self.model.apply(
                params,
                input_ids=next_token,  # Only process the new token
                attention_mask=attention_mask,
                position_ids=position_ids[:, seq_len].reshape(-1, 1),
                return_dict=return_dict,
                mutable=["cache"],
            )
            params["cache"] = cache["cache"]
            seq_len += 1

            next_token_logits = outputs["logits"][:, -1, :]
            next_token = jnp.argmax(next_token_logits, axis=-1)[:, None]
            all_token_ids = jnp.concatenate((all_token_ids, next_token), axis=1)

        return all_token_ids[:, :-1]
