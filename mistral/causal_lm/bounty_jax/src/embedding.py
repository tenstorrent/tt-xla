# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright 2024 Google LLC.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding implementation.

This implementation expects the features to be ordered in odds and evens.

From flaxformers/components/embedding.py
"""

import functools

import jax
import jax.numpy as jnp


def rotate_half(x):
    """Helper that splits a tensor at last dim into half and rotate it."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


@functools.partial(jax.jit, static_argnums=(4,))
def apply_rotary_embedding(q, k, cos, sin, decode=False, rotary_index=None):
    """Helper function to apply Rotary Embeddings."""
    if len(k.shape) == 3:
        # for multi query attention
        k = jnp.expand_dims(k, 2)
        multiquery = True
    else:
        multiquery = False

    batch, qlen, qheads, d = q.shape
    kbatch, klen, kheads, kd = k.shape
    assert batch == kbatch, f"{batch} != {kbatch}"
    assert d == kd, f"{d} != {kd}"

    # cos: [len, d]
    # sin: [len, d]
    # rotary_index: [batch]

    if decode and qlen == 1 and rotary_index is not None:
        # we check qlen == 1 so that we don't do this when initializing cache.
        qcos = cos[rotary_index, :]
        qsin = sin[rotary_index, :]
        # qcos, qsin: [batch, d]
        qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (0, 3))
        qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (0, 3))
        # qcos, qsin: [batch, qlen, qheads, d]
    else:
        qcos, qsin = cos[:qlen, :], sin[:qlen, :]
        # qcos, qsin: [qlen, d]
        qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (1, 3))
        qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (1, 3))
        # qcos, qsin: [batch, qlen, qheads, d]

    kcos, ksin = cos[:klen, :], sin[:klen, :]
    # kcos, ksin: [klen, d]
    kcos = jax.lax.broadcast_in_dim(kcos, (batch, klen, kheads, d), (1, 3))
    ksin = jax.lax.broadcast_in_dim(ksin, (batch, klen, kheads, d), (1, 3))
    # kcos, ksin: [batch, klen, kheads, d]

    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)
    if multiquery:
        out_k = jnp.squeeze(out_k, 2)
    return out_q, out_k


def generate_fixed_pos_embedding(
    features, length, min_timescale=1.0, max_timescale=10000.0
):
    """Generate Sin/Cos for Rotary Embeddings.

    Generates sinusoids at (features//2) different timescales, where the
    timescales form a gemetric series from min_timescale to max_timescale
    (max_timescale is not included, but would be the next element in the series).

    Sinusoids are evaluated at integer positions i in [0, length).

    The outputs are computed as:

      output_sin[i, j] = sin(i / timescale[j])
      output_cos[i, j] = cos(i / timescale[j])

    Finally, the outputs are tiled twice in the features dimension.

    Args:
      features: an integer
      length: an integer
      min_timescale: an optional float
      max_timescale: an optional float

    Returns:
      output_sin: a float32 Tensor with shape [length, features]
      output_cos: a float32 Tensor with shape [length, features]
    """
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, since rounding off to a bfloat16 is
    # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
    # from sin(256).
    sinusoid_inp = jnp.einsum(
        "i , j -> i j",
        jnp.arange(length, dtype=jnp.float32),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)

    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
