# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Sequence

import jax
import jax._src.xla_bridge as xb
from flax import linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# TODO Move this to TT models repo.
from tests.jax.multi_chip.n300.models.tensor_parallel.alexnet.model_implementation import (
    AlexNetMultichipModel,
)

# Allocating enough CPU devices so we can create various meshes depending on which TT
# device this example is running. Can't be set to exact number of TT devices because
# after calling `jax.devices` function this config update doesn't work anymore.
jax.config.update("jax_num_cpu_devices", 8)

# Change if you want to use shardy.
jax.config.update("jax_use_shardy_partitioner", False)


def register_pjrt_plugin():
    """Registers TT PJRT plugin."""

    plugin_path = os.path.join(
        os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
    )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Could not find TT PJRT plugin at {plugin_path}")

    xb.register_plugin("tt", library_path=plugin_path)
    jax.config.update("jax_platforms", "cpu,tt")


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


register_pjrt_plugin()

axis_name = "X"
# Currently we support only 2D mesh with shardy enabled.
device_mesh = jax.make_mesh(
    (1, len(jax.devices("tt"))), ("Y", axis_name), devices=jax.devices("tt")
)

model = AlexNetMultichipModel(
    axis_name=axis_name, num_devices=len(jax.devices("tt")), train_mode=False
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

compiled_apply = jax.jit(
    shard_map(
        model.apply,
        device_mesh,
        in_specs=(params_specs, inputs_specs),
        out_specs=P(),
        check_rep=False,
    )
)
results = compiled_apply(params, device_inputs)

print(results)
