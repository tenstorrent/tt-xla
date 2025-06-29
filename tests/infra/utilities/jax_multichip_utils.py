# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from enum import Enum

import jax
from flax import linen
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree

from .types import Tensor


class ShardingMode(Enum):
    INPUTS_AND_MODULE = 1  # Uses both shard_map() and jax.device_put() (inputs are sharded and function has sharding ops)
    MODULE = 2  # Uses only shard_map() (function has sharding ops)
    INPUTS = 3  # Uses only jax.device_put() (inputs are sharded)

    @property
    def requires_shard_map(self) -> bool:
        return self == ShardingMode.INPUTS_AND_MODULE or self == ShardingMode.MODULE

    @property
    def requires_device_put(self) -> bool:
        return self == ShardingMode.INPUTS_AND_MODULE or self == ShardingMode.INPUTS


@contextmanager
def enable_shardy(use_shardy: bool):
    """
    Context manager that temporarily enables shardy in jax.config.

    Isolated as a context manager so that it doesn't change global config for all jax
    imports and cause unexpected fails elsewhere.
    """
    try:
        # Set the config to True within this block, and yield back control.
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        yield
    finally:
        # After `with` statement ends, turn it off again.
        jax.config.update("jax_use_shardy_partitioner", False)


def make_partition_spec(axis_names: tuple) -> PartitionSpec:
    """
    Returns a PartitionSpec object for the given `axis_names`.
    """
    return PartitionSpec(*axis_names)


def make_flax_linen_parameters_partition_specs_on_cpu(
    model: linen.Module, cpu_mesh: Mesh, inputs_specs: PartitionSpec, cpu_inputs: Tensor
):
    """Makes partition specs for Flax linen model parameters on CPU."""
    # Have to use shard_map because CCL ops need a mapped axis for tracing to work.
    return linen.get_partition_spec(
        jax.eval_shape(
            shard_map(
                model.init,
                cpu_mesh,
                in_specs=(None, inputs_specs),
                out_specs=PartitionSpec(),
            ),
            jax.random.PRNGKey(21),
            cpu_inputs,
        )
    )


def initialize_flax_linen_parameters_on_cpu(
    model: linen.Module,
    inputs_specs: PartitionSpec,
    cpu_inputs: Tensor,
    params_specs: PyTree,
    cpu_mesh: Mesh,
    rng_seed: int,
):
    """Initializes Flax linen model parameters on CPU."""
    init_fn = shard_map(
        model.init, cpu_mesh, in_specs=(None, inputs_specs), out_specs=params_specs
    )
    return init_fn(jax.random.PRNGKey(rng_seed), cpu_inputs)
