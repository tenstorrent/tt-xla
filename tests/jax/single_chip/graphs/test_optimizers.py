# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax
import pytest
from infra import run_graph_test_with_random_inputs
from tests.utils import Category, incorrect_result


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("param_shape", [(32, 32), (64, 64)])
def test_sgd(param_shape: tuple):
    """Test SGD optimizer update step."""

    def sgd_update(params: jax.Array, grads: jax.Array) -> jax.Array:
        learning_rate = 0.01
        grads = grads * 0.01
        optimizer = optax.sgd(learning_rate=learning_rate)
        updates, _ = optimizer.update(grads, optimizer.init(params))
        return optax.apply_updates(params, updates)

    run_graph_test_with_random_inputs(sgd_update, [param_shape, param_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("param_shape", [(32, 32), (64, 64)])
def test_adam(param_shape: tuple):
    """Test Adam optimizer update step."""

    def adam_update(params: jax.Array, grads: jax.Array) -> jax.Array:
        learning_rate = 0.01
        grads = grads * 0.01
        optimizer = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8)
        updates, _ = optimizer.update(grads, optimizer.init(params))
        return optax.apply_updates(params, updates)

    run_graph_test_with_random_inputs(adam_update, [param_shape, param_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("param_shape", [(4096, 4096), (4096, 11008)])
def test_llama_adamw(param_shape: tuple):
    """Test AdamW optimizer with LLaMA-like shape of parameters."""

    def llama_adamw_update(params: jax.Array, grads: jax.Array) -> jax.Array:
        grads = grads * 0.01
        learning_rate = 3e-4
        weight_decay = 0.1
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
        )
        opt_state = optimizer.init(params)
        updates, _ = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates)

    run_graph_test_with_random_inputs(llama_adamw_update, [param_shape, param_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("param_shape", [(768, 768), (768, 3072)])
def test_bert_adamw(param_shape: tuple):
    """Test AdamW optimizer with BERT-like shape of parameters."""

    def bert_adamw_update(params: jax.Array, grads: jax.Array) -> jax.Array:
        grads = grads * 0.01
        learning_rate = 5e-5
        weight_decay = 0.01
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-6,
            weight_decay=weight_decay,
        )
        opt_state = optimizer.init(params)
        updates, _ = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates)

    run_graph_test_with_random_inputs(bert_adamw_update, [param_shape, param_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("param_shape", [(256, 1152), (512, 2304)])
def test_resnet18_sgd(param_shape: tuple):
    """Test SGD optimizer with actual ResNet18 convolutional layer shapes.

    (256, 1152): Layer3 conv layer (256, 128, 3, 3) flattened -> 256 * (128*3*3) = 256 * 1152
    (512, 2304): Layer4 conv layer (512, 256, 3, 3) flattened -> 512 * (256*3*3) = 512 * 2304
    """

    def resnet18_sgd_update(params: jax.Array, grads: jax.Array) -> jax.Array:
        grads = grads * 0.01
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
        grads_with_decay = grads + weight_decay * params
        updates, _ = optimizer.update(grads_with_decay, optimizer.init(params))
        return optax.apply_updates(params, updates)

    run_graph_test_with_random_inputs(resnet18_sgd_update, [param_shape, param_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("param_shape", [(128, 128), (256, 256)])
def test_radam(param_shape: tuple):
    """Test RAdam optimizer with EfficientNeRF shapes."""

    def radam_update(params: jax.Array, grads: jax.Array) -> jax.Array:
        grads = grads * 0.01
        learning_rate = 8e-4
        optimizer = optax.radam(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        )
        opt_state = optimizer.init(params)
        updates, _ = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates)

    run_graph_test_with_random_inputs(radam_update, [param_shape, param_shape])
