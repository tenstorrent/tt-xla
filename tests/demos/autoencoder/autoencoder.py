# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax._src.xla_bridge as xb
import jax.numpy as jnp
import numpy
import os

path = os.path.join(os.path.dirname(__file__), "../../../build/src/tt/pjrt_plugin_tt.so")
if not os.path.exists(path):
    raise FileNotFoundError(
        f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?"
    )

plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
jax.config.update("jax_platforms", "tt,cpu")

from infrastructure import verify_module

def test_encoder_linear_one():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 784), (784, 128), (1, 128), (1, 128)])

def test_encoder_linear_two():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 128), (128, 64), (1, 64), (1, 64)])

def test_encoder_linear_three():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 64), (64, 12), (1, 12), (1, 12)])

def test_encoder_linear_four():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 12), (12, 3), (1, 3), (1, 3)])

def test_decoder_linear_one():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 3), (3, 12), (1, 12), (1, 12)])

def test_decoder_linear_two():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 12), (12, 64), (1, 64), (1, 64)])

def test_decoder_linear_three():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 64), (64, 128), (1, 128), (1, 128)])

def test_decoder_linear_four():
    def module(a, b, c, d):
        act = jnp.dot(a, b)
        act = jnp.multiply(act, c)
        return jnp.add(act, d)

    verify_module(module, input_shapes=[(1, 128), (128, 784), (1, 784), (1, 784)])

def test_relu():
    def module(a):
        return jnp.maximum(a, 0)

    verify_module(module, input_shapes=[(1, 12)])
    verify_module(module, input_shapes=[(1, 64)])
    verify_module(module, input_shapes=[(1, 128)])

test_encoder_linear_two()
