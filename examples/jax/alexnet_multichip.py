# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Sequence

import jax
import jax._src.xla_bridge as xb
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from third_party.tt_forge_models.alexnet.image_classification.jax.src.model_implementation import (
    AlexNetMultichipModel,
)

# Allocating enough CPU devices so we can create various meshes depending on which TT
# device this example is running. Can't be set to exact number of TT devices because
# after calling `jax.devices` function this config update doesn't work anymore.
jax.config.update("jax_num_cpu_devices", 2)

# Change if you want to use shardy.
jax.config.update("jax_use_shardy_partitioner", False)


def generate_inputs_on_cpu(prng_key: jax.Array) -> Sequence[jax.Array]:
    """Generates inputs on CPU."""

    with jax.default_device(jax.devices("cpu")[0]):
        return jax.random.randint(
            key=prng_key,
            # B, H, W, C
            shape=(32, 224, 224, 3),
            # In the original paper inputs are normalized with individual channel
            # values learned from training set.
            minval=-128,
            maxval=128,
        )


def initialize_parameters(
    model, inputs_specs, cpu_inputs, params_specs, device_mesh, prng_key
):
    """Initializes model parameters. Currently we don't support RNG on device, so inputs
    are initialized on CPU and moved to device."""

    # Initializing parameters on CPU.
    cpu_mesh = jax.make_mesh(
        device_mesh.axis_sizes, device_mesh.axis_names, devices=jax.devices("cpu")
    )
    init_fn = shard_map(
        model.init, cpu_mesh, in_specs=(None, inputs_specs), out_specs=params_specs
    )
    params = init_fn(prng_key, cpu_inputs)

    # Moving parameters to device.
    params = jax.tree.map(
        lambda spec, param: jax.device_put(param, NamedSharding(device_mesh, spec)),
        params_specs,
        params,
        is_leaf=lambda x: isinstance(x, nn.Partitioned) or isinstance(x, P),
    )

    return params


def run_alexnet():
    tt_devices = jax.devices("tt")
    num_tt_devices = len(jax.devices("tt"))

    axis_name = "X"
    device_mesh = jax.make_mesh((num_tt_devices,), (axis_name), devices=tt_devices)

    model = AlexNetMultichipModel(
        axis_name=axis_name, num_devices=num_tt_devices, train_mode=False
    )

    prng_key = jax.random.PRNGKey(23)

    # Sharding data on batch axis since data parallelism is utilized for the convolutional
    # layers.
    inputs_specs = P(axis_name)
    cpu_inputs = generate_inputs_on_cpu(prng_key)

    # Have to use shard_map because CCL ops need a mapped axis for tracing to work.
    params_specs = nn.get_partition_spec(
        jax.eval_shape(
            shard_map(
                model.init, device_mesh, in_specs=(None, inputs_specs), out_specs=P()
            ),
            prng_key,
            cpu_inputs,
        )
    )
    params = initialize_parameters(
        model, inputs_specs, cpu_inputs, params_specs, device_mesh, prng_key
    )

    # Now we can move inputs to device, needed them on CPU to initialize the parameters.
    device_inputs = jax.device_put(cpu_inputs, NamedSharding(device_mesh, inputs_specs))

    # Outputs will be replicated on each device.
    out_spec = P()

    compiled_apply = jax.jit(
        shard_map(
            model.apply,
            device_mesh,
            in_specs=(params_specs, inputs_specs),
            out_specs=out_spec,
            check_rep=False,
        ),
        out_shardings=NamedSharding(device_mesh, out_spec),
    )
    results = compiled_apply(params, device_inputs)

    return results


def run_alexnet_cpu():
    """Run the same sharded AlexNet on CPU devices for comparison."""
    cpu_devices = jax.devices("cpu")
    num_cpu_devices = len(cpu_devices)

    axis_name = "X"
    device_mesh = jax.make_mesh((num_cpu_devices,), (axis_name,), devices=cpu_devices)

    model = AlexNetMultichipModel(
        axis_name=axis_name, num_devices=num_cpu_devices, train_mode=False
    )

    prng_key = jax.random.PRNGKey(23)

    inputs_specs = P(axis_name)
    cpu_inputs = generate_inputs_on_cpu(prng_key)

    params_specs = nn.get_partition_spec(
        jax.eval_shape(
            shard_map(
                model.init, device_mesh, in_specs=(None, inputs_specs), out_specs=P()
            ),
            prng_key,
            cpu_inputs,
        )
    )
    params = initialize_parameters(
        model, inputs_specs, cpu_inputs, params_specs, device_mesh, prng_key
    )

    device_inputs = jax.device_put(cpu_inputs, NamedSharding(device_mesh, inputs_specs))

    out_spec = P()

    compiled_apply = jax.jit(
        shard_map(
            model.apply,
            device_mesh,
            in_specs=(params_specs, inputs_specs),
            out_specs=out_spec,
            check_rep=False,
        ),
        out_shardings=NamedSharding(device_mesh, out_spec),
    )
    results = compiled_apply(params, device_inputs)

    return results


def test_alexnet_multichip():
    """Test AlexNet multichip output against CPU reference."""
    tt_results = run_alexnet()
    cpu_results = run_alexnet_cpu()

    tt_results_cpu = jax.device_get(tt_results)
    cpu_results_cpu = jax.device_get(cpu_results)

    assert jnp.allclose(
        tt_results_cpu, cpu_results_cpu, atol=0.05
    ), f"AlexNet multichip mismatch. Max diff: {jnp.abs(tt_results_cpu - cpu_results_cpu).max()}"


if __name__ == "__main__":
    results = run_alexnet()
    print(results)
