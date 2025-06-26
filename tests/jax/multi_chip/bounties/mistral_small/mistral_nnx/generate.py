# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.sharding import Mesh
from jaxtyping import Float, Integer

from .model import MistralModel


def sample_best(logits: Float[Array, "*B V"]) -> Integer[Array, "*B"]:
    return jnp.argmax(logits, axis=-1)


def sample_top_p(
    logits: Float[Array, "*B V"],
    *,
    temperature: float,
    top_p: float,
    key: Array,
) -> Integer[Array, "*B"]:
    """Top-p sampling.

    Args:
        key: RNG key
    """
    probs = nnx.softmax(logits / temperature, axis=-1)
    return _sample_top_p(probs, top_p, key=key)


def _sample_top_p(
    probs: Float[Array, "*B V"], p: float, key: Array
) -> Float[Array, "*B"]:
    """Sample a token using top-p sampling."""
    # From flax/examples/gemma/sampler.py
    probs_sorted, indices = jax.lax.top_k(probs, k=probs.shape[-1])
    cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
    mask = cumsum_probs - probs_sorted > p
    probs_sorted = jnp.where(mask, 0.0, probs_sorted)
    probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

    next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

    next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
    next_token = jnp.squeeze(next_token, axis=-1)
    return next_token


@dataclass
class GenerateResult:
    tokens: list[int]

    # all logits, including those from input tokens
    logits: Float[Array, "S V"]


class Generator:
    """Token generator using the mistral model.

    - Holds on to a jitted function for running the model.
    - Instantiates and uses KV cache to do incremental decoding.
    """

    def __init__(self, model: MistralModel, max_seqlen: int):
        """
        Args:
            model: Mistral model.
            max_seqlen: Maximum sequence length (input + max_tokens). This
                controls the size of the allocated KV cache.

        Note: max_seqlen is fixed at startup to prevent excessive jitting with
        different kv cache sizes.
        """
        self.model = model
        self.max_seqlen = max_seqlen

        # Use jax.jit with pre-split model to avoid nnx.jit's cpu and memory
        # overhead.
        self.graphdef, self.state = nnx.split(self.model)
        self._jit_decode = jax.jit(
            self._jit_decode_impl,
            donate_argnames=("cache",),
        )

        # Use nnx jit for this one to handle rngs, which is simple enough.
        self._jit_sample_top_p = nnx.jit(sample_top_p)

    @staticmethod
    def _jit_decode_impl(graphdef, state, input, cache):
        model = nnx.merge(graphdef, state)
        logits, cache = model.decode(input, cache)
        return logits, cache

    def generate(
        self,
        input_ids: list[int],
        *,
        rngs: nnx.Rngs,
        max_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.8,
        eos_id: int | None = 2,
        mesh: Mesh | None = None,
    ) -> GenerateResult:
        """Generate output with simple greedy search.

        Args:
            input_ids: Sequence of tokens to generate from.
            rngs: rng. The stream named 'sample' is used.
            mesh: mesh used to shard kv cache, should be the same as what's used for the model.

        Returns:
            sequnece of generated tokens

        """
        max_tokens = max(len(input_ids), max_tokens)
        assert max_tokens < self.max_seqlen
        result = list(input_ids)

        cache = self.model.create_cache(1, self.max_seqlen, mesh)
        all_logits: list[Float[Array, "1 1 V"]] = []

        # prefill cache and get first token
        input = jnp.array(input_ids, dtype="int32")
        input = input.reshape(1, -1)
        logits = None
        for i in range(input.shape[1]):
            logits, cache = self._jit_decode(
                self.graphdef,
                self.state,
                input[1, i].reshape(1, 1),
                cache,
            )
            all_logits.append(logits)
        assert logits is not None

        for _ in range(max_tokens):
            last_chosen = self._jit_sample_top_p(
                logits[0, -1, :],
                temperature=temperature,
                top_p=top_p,
                key=rngs.sample(),
            )
            assert len(last_chosen.shape) == 0
            result.append(last_chosen.item())
            if eos_id != None and last_chosen.item() == eos_id:
                break

            logits, cache = self._jit_decode(
                self.graphdef,
                self.state,
                last_chosen.reshape(1, 1),
                cache,
            )
            all_logits.append(logits)

        return GenerateResult(
            tokens=result,
            logits=jnp.stack([x.reshape(-1) for x in all_logits]),
        )
