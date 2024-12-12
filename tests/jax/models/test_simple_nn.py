# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
import jax.numpy as jnp
import pytest
from infra.module_tester import test, test_with_random_inputs
from infra.test_model import TestModel


class SimpleNN(TestModel):  # TODO what's the benefit of inheriting nnx.Module?
    # TODO upgrade to python 3.13 to enable this decorator
    # @override
    def __call__(
        self, act: jax.Array, w0: jax.Array, b0: jax.Array, w1: jax.Array, b1: jax.Array
    ) -> jax.Array:
        x = jnp.matmul(act, w0) + b0
        x = jnp.matmul(x, w1) + b1
        return x

    # @override
    @staticmethod
    def get_model() -> TestModel:
        return SimpleNN()

    # @override
    @staticmethod
    def get_model_inputs() -> Sequence[jax.Array]:
        act_shape, w0_shape, b0_shape, w1_shape, b1_shape = (
            (32, 784),
            (784, 128),
            (1, 128),
            (128, 128),
            (1, 128),
        )

        act = jax.numpy.ones(act_shape)
        w0 = jax.numpy.ones(w0_shape)
        b0 = jax.numpy.ones(b0_shape)
        w1 = jax.numpy.ones(w1_shape)
        b1 = jax.numpy.zeros(b1_shape)

        return [act, w0, b0, w1, b1]


@pytest.fixture
def model() -> SimpleNN:
    return SimpleNN.get_model()


@pytest.fixture
def inputs() -> Sequence[jax.Array]:
    return SimpleNN.get_model_inputs()


def test_simple_nn(model: SimpleNN, inputs: Sequence[jax.Array]):
    assert test(model, inputs)


@pytest.mark.parametrize(
    ["act", "w0", "b0", "w1", "b1"],
    [
        [(32, 784), (784, 128), (1, 128), (128, 128), (1, 128)],
    ],
)
def test_simple_nn_with_random_inputs(
    model: SimpleNN, act: tuple, w0: tuple, b0: tuple, w1: tuple, b1: tuple
):
    assert test_with_random_inputs(model, [act, w0, b0, w1, b1])


if __name__ == "__main__":
    test_simple_nn()
