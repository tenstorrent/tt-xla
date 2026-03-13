# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from flax import linen as nn
import jax.numpy as jnp
from transformers import FlaxBertModel, BertConfig


class PoolingMode(Enum):
    # Use the first token (CLS token) as text representations.
    CLS_TOKEN = "cls_token"
    # Use max in each dimension over all tokens.
    MAX_TOKENS = "max_tokens"
    # Perform mean-pooling.
    MEAN_TOKENS = "mean_tokens"
    # Perform mean-pooling, but divide by sqrt(input_length).
    MEAN_SQRT_LEN_TOKENS = "mean_sqrt_len_tokens"
    # Perform (position) weighted mean pooling (https://arxiv.org/abs/2202.08904).
    WEIGHTEDMEAN_TOKENS = "weightedmean_tokens"
    # Perform last token pooling (https://arxiv.org/abs/2201.10005).
    POOLING_MODE_LASTTOKEN = "pooling_mode_lasttoken"


class FlaxSentenceTransformer(nn.Module):
    config: BertConfig
    pooling_mode: PoolingMode = PoolingMode
    normalize: bool = False
    pooling_include_prompt: bool = True
    prompt_length: int = 0

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_type_ids: jnp.ndarray = None,
    ) -> jnp.ndarray:
        transformer = FlaxBertModel(self.config, dtype=jnp.float32)
        transformer_outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        token_embeddings = transformer_outputs.last_hidden_state

        if not self.pooling_include_prompt and self.prompt_length > 0:
            attention_mask = attention_mask.at[:, : self.prompt_length].set(0)

        input_mask_expanded = jnp.expand_dims(attention_mask, axis=-1)
        sum_embeddings = jnp.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = jnp.clip(jnp.sum(input_mask_expanded, axis=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        if self.normalize:
            norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norm

        return embeddings


class FlaxSentenceTransformerBERT(FlaxSentenceTransformer):
    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_type_ids: jnp.ndarray = None,
    ) -> jnp.ndarray:
        sentence_transformer = FlaxSentenceTransformer(
            config=self.config,
            pooling_mode=PoolingMode.MEAN_TOKENS,
            normalize=True,
            pooling_include_prompt=True,
            prompt_length=0,
        )
        return sentence_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
