# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import PyTree
from transformers import SqueezeBertConfig

from .utils import torch_statedict_to_pytree


class SqueezeBertEmbedding(nn.Module):
    """Embedding layer for SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.word_embedding = nn.Embed(
            self.config.vocab_size, self.config.embedding_size
        )
        self.position_embedding = nn.Embed(
            self.config.max_position_embeddings,
            self.config.embedding_size,
        )
        self.token_type_embedding = nn.Embed(
            self.config.type_vocab_size,
            self.config.embedding_size,
        )

        self.layernorm = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids: jax.Array,
        token_type_ids: jax.Array = None,
        position_ids: jax.Array = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if position_ids is None:
            position_ids = jax.numpy.arange(input_ids.shape[1])
        if token_type_ids is None:
            token_type_ids = jax.numpy.zeros_like(input_ids)

        word_embeddings = self.word_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


class SqueezeBertSelfAttention(nn.Module):
    """Self-attention layer for SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.query = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(1,),
            feature_group_count=self.config.q_groups,
        )
        self.key = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(1,),
            feature_group_count=self.config.k_groups,
        )
        self.value = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(1,),
            feature_group_count=self.config.v_groups,
        )
        self.output = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(1,),
            feature_group_count=self.config.post_attention_groups,
        )

        self.attn_dropout = nn.Dropout(rate=self.config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm()

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = einops.rearrange(
            query,
            "b s (H d) -> b s H d",  # batch sequence Heads dim_head
            H=self.config.num_attention_heads,
            d=head_dim,
        )
        key = einops.rearrange(
            key,
            "b s (H d) -> b s H d",
            H=self.config.num_attention_heads,
            d=head_dim,
        )
        value = einops.rearrange(
            value,
            "b s (H d) -> b s H d",
            H=self.config.num_attention_heads,
            d=head_dim,
        )

        attention_scores = jnp.einsum("B s H d ,B S H d -> B H s S", query, key)
        attention_scores = attention_scores / jnp.sqrt(head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.activation.softmax(attention_scores, axis=-1)
        attention_probs = self.attn_dropout(
            attention_probs, deterministic=deterministic
        )

        context = jnp.einsum("B H s S, B S H d -> B s H d", attention_probs, value)
        context = einops.rearrange(context, "b s H d -> b s (H d)")

        output = self.output(context)
        output = self.resid_dropout(output, deterministic=deterministic)
        output = hidden_states + output
        output = self.layernorm(output)
        return output


class SqueezeBertMLP(nn.Module):
    """MLP layer for SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.w1 = nn.Conv(
            features=self.config.intermediate_size,
            kernel_size=(1,),
            feature_group_count=self.config.intermediate_groups,
        )
        if self.config.hidden_act == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(
                f"Activation function {self.config.hidden_act} not supported."
            )
        self.w2 = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(1,),
            feature_group_count=self.config.output_groups,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm()

    def __call__(
        self, hidden_states: jax.Array, deterministic: bool = False
    ) -> jax.Array:
        x = self.w1(hidden_states)
        x = self.act(x)
        x = self.w2(x)
        x = self.dropout(x, deterministic=deterministic)
        output = hidden_states + x
        output = self.layernorm(output)
        return output


class SqueezeBertLayer(nn.Module):
    """Layer for SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.attention = SqueezeBertSelfAttention(self.config)
        self.mlp = SqueezeBertMLP(self.config)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        attention_output = self.attention(
            hidden_states, attention_mask, deterministic=deterministic
        )
        output = self.mlp(attention_output, deterministic=deterministic)
        return output


class SqueezeBertEncoder(nn.Module):
    """Encoder for SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.layers = [
            SqueezeBertLayer(self.config) for _ in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, deterministic=deterministic
            )
        return hidden_states


class SqueezeBertPooler(nn.Module):
    """Pooler layer for SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size)
        self.activation = nn.tanh

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SqueezeBertModel(nn.Module):
    """SqueezeBERT model."""

    config: SqueezeBertConfig

    def setup(self):
        self.embeddings = SqueezeBertEmbedding(self.config)
        self.encoder = SqueezeBertEncoder(self.config)
        self.pooler = SqueezeBertPooler(self.config)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        token_type_ids: jax.Array = None,
        position_ids: jax.Array = None,
        *,
        train: bool,
    ) -> Tuple[jax.Array, jax.Array]:
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        embeddings = self.embeddings(
            input_ids, token_type_ids, position_ids, deterministic=not train
        )
        encoder_output = self.encoder(
            embeddings, attention_mask, deterministic=not train
        )
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


class SqueezeBertForMaskedLM(nn.Module):
    """SqueezeBERT model with masked language modeling head."""

    config: SqueezeBertConfig

    def setup(self):
        self.squeezebert = SqueezeBertModel(self.config)
        self.transform_dense = nn.Dense(self.config.hidden_size)

        if self.config.hidden_act == "gelu":
            self.transform_act = nn.gelu
        else:
            raise ValueError(
                f"Activation function {self.config.hidden_act} not supported."
            )

        self.transform_layernorm = nn.LayerNorm()

        self.decoder = nn.Dense(self.config.vocab_size)
        # TODO(stefan): Figure out if SqueezeBERT uses tied weights for embeddings and output layer
        # that is only relevant for training

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        token_type_ids: jax.Array = None,
        position_ids: jax.Array = None,
        *,
        train: bool,
    ) -> jax.Array:
        hidden_states, _ = self.squeezebert(
            input_ids, attention_mask, token_type_ids, position_ids, train=train
        )
        hidden_states = self.transform_dense(hidden_states)
        hidden_states = self.transform_act(hidden_states)
        hidden_states = self.transform_layernorm(hidden_states)

        prediction_scores = self.decoder(hidden_states)
        return prediction_scores

    @staticmethod
    def _get_renaming_patterns() -> List[Tuple[str, str]]:
        return [
            ("transformer.", "squeezebert."),
            ("LayerNorm", "layernorm"),
            ("layernorm.weight", "layernorm.scale"),
            ("_embeddings.weight", "_embedding.embedding"),
            ("encoder.layers.", "encoder.layers_"),
            ("attention.query.weight", "attention.query.kernel"),
            ("attention.key.weight", "attention.key.kernel"),
            ("attention.value.weight", "attention.value.kernel"),
            ("post_attention.conv1d.weight", "attention.output.kernel"),
            ("post_attention.conv1d.bias", "attention.output.bias"),
            ("post_attention.layernorm", "attention.layernorm"),
            ("intermediate.conv1d.weight", "mlp.w1.kernel"),
            ("intermediate.conv1d.bias", "mlp.w1.bias"),
            ("output.conv1d.weight", "mlp.w2.kernel"),
            ("output.conv1d.bias", "mlp.w2.bias"),
            ("output.layernorm", "mlp.layernorm"),
            ("pooler.dense.weight", "pooler.dense.kernel"),
            ("cls.predictions.transform.dense.weight", "transform_dense.kernel"),
            ("cls.predictions.transform.dense.bias", "transform_dense.bias"),
            ("cls.predictions.transform.layernorm", "transform_layernorm"),
            ("cls.predictions.decoder.weight", "decoder.kernel"),
            ("cls.predictions.bias", "decoder.bias"),
        ]

    @staticmethod
    def _get_banned_subkeys() -> List[str]:
        return ["cls.seq_relationship"]

    @staticmethod
    def init_from_pytorch_statedict(
        state_dict: Dict[str, Any], dtype: Optional[jnp.dtype] = None
    ) -> PyTree:
        return torch_statedict_to_pytree(
            state_dict,
            patterns=SqueezeBertForMaskedLM._get_renaming_patterns(),
            banned_subkeys=SqueezeBertForMaskedLM._get_banned_subkeys(),
            dtype=dtype,
        )
