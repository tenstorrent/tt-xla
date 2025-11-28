# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from enum import Enum
from typing import Tuple

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


def make_easydel_parameters_partition_specs(
    model_state: PyTree,
    partition_rules: Tuple[Tuple[str, PartitionSpec], ...],
) -> PyTree:
    """
    Creates partition specs for EasyDel/NNX model parameters based on partition rules.

    Args:
        model_state: The NNX model state (from nnx.split(model)[1])
        partition_rules: Tuple of (regex_pattern, PartitionSpec) pairs

    Returns:
        PyTree of partition specs matching the model_state structure
    """
    import re

    def jax_path_to_string(path) -> str:
        """Convert JAX tree path to string representation.

        Args:
            path: JAX tree path from tree_flatten_with_path

        Returns:
            String representation of the path (e.g., "transformer/h/0/attn/c_attn/kernel")
        """
        return "/".join(
            (
                key.name
                if hasattr(key, "name")
                else (
                    str(key.key)
                    if hasattr(key, "key")
                    else str(key.idx) if hasattr(key, "idx") else str(key)
                )
            )
            for key in path
        )

    def get_partition_spec_for_param(param_path: str) -> PartitionSpec:
        """Match param path against partition rules and return appropriate spec.

        Note: Partition rules should be ordered from most specific to most general.
        Put general rules (like r".*") at the end to avoid conflicts between rules.
        The first matching pattern will be used.
        """
        for pattern, spec in partition_rules:
            if re.match(pattern, param_path):
                return spec
        # Default to replicated if no match
        return PartitionSpec()

    # Get flattened state with paths
    flat_state, tree_def = jax.tree_util.tree_flatten_with_path(model_state)

    # Create partition specs for each parameter
    partition_specs = []
    for path, _ in flat_state:
        param_path = jax_path_to_string(path)
        spec = get_partition_spec_for_param(param_path)
        partition_specs.append(spec)

    # Reconstruct PyTree with same structure
    return jax.tree_util.tree_unflatten(tree_def, partition_specs)


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
