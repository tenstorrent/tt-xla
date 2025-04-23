# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from flax import linen as nn
import jax.numpy as jnp
from transformers import FlaxBertModel, BertConfig


class FlaxSentenceTransformerBERT(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        token_type_ids: jnp.ndarray = None,
        normalize: bool = True,
        include_prompt: bool = True,
        prompt_length: int = 0,
    ) -> jnp.ndarray:
        if not include_prompt and prompt_length > 0:
            attention_mask = attention_mask.at[:, :prompt_length].set(0)

        model = FlaxBertModel(self.config, dtype=jnp.float32)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        token_embeddings = output.last_hidden_state

        # Mean pooling
        input_mask_expanded = jnp.expand_dims(attention_mask, axis=-1)
        sum_embeddings = jnp.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = jnp.clip(jnp.sum(input_mask_expanded, axis=1), a_min=1e-9)
        embeddings = sum_embeddings / sum_mask

        if normalize:
            norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norm

        return self.cosine_similarity(embeddings[0], embeddings[1])

    def cosine_similarity(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(a, b) / jnp.maximum(
            jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-9
        )
