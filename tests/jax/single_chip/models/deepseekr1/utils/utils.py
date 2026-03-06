# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import flax
import numpy as np
from rich import print

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9375"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from deepseekr1.utils.torch_to_flax import torch_to_flax
from deepseekr1.utils.model_implementation import (
    get_partition_rules,
    Qwen2Config,
    Qwen2ForCausalLM,
)


def create_sharding_specs(params):
    devices = jax.devices()
    device_mesh = np.array(devices).reshape(-1)
    mesh = Mesh(device_mesh, ("mp",))
    print("Device mesh:", mesh)

    partition_rules = get_partition_rules()

    def assign_spec(path, _):
        path_str = "/".join(map(str, path))
        for rule_path, spec in partition_rules:
            if rule_path in path_str:
                return NamedSharding(mesh, spec)
        return NamedSharding(mesh, P())

    return jax.tree_util.tree_map_with_path(assign_spec, params)


def load_model(SHARD_MODEL: bool = True):
    config = Qwen2Config()
    model = Qwen2ForCausalLM(config=config)

    rng = jax.random.PRNGKey(0)
    input_shape = (1, 32)

    with jax.default_device(jax.devices("cpu")[0]):
        try:
            params = model.init(rng, jnp.ones(input_shape, dtype=jnp.int4))
        except Exception:
            params = model.init(rng, jnp.ones(input_shape, dtype=jnp.int32))

    try:
        with open("flax_params.msgpack", "rb") as f:
            params = {
                "params": flax.serialization.from_bytes(params["params"], f.read())
            }
    except FileNotFoundError:
        print("[yellow]Checkpoint not found. Running torch_to_flax()[/yellow]")
        torch_to_flax()
        with open("flax_params.msgpack", "rb") as f:
            params = {
                "params": flax.serialization.from_bytes(params["params"], f.read())
            }

    if SHARD_MODEL:
        sharding_specs = create_sharding_specs(params)
        params = jax.tree_util.tree_map(
            lambda x, spec: jax.device_put(x, spec), params, sharding_specs
        )
        print("[green]Model parameters sharded successfully.[/green]")
    else:
        params = jax.device_put(params)

    return model, params
