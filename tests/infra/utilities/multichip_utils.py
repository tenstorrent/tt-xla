# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
from jax.sharding import Mesh, PartitionSpec

from .workloads.jax_workload import JaxWorkload


@dataclass
class MultichipWorkload(JaxWorkload):
    """
    An extension of the JaxWorkload dataclass that includes a mesh and partition specs,
    necessary for multichip sharding.

    TODO shouldn't inherit from JaxWorkload, but for now only used with jax.
    """

    def __init__(
        self,
        executable: Callable,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
        device_mesh: Optional[Mesh] = None,
        in_specs: Optional[Sequence[PartitionSpec]] = None,
    ) -> None:
        super().__init__(executable, args, kwargs, static_argnames)

        self.device_mesh = device_mesh
        self.in_specs = in_specs


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


def make_partition_spec(axis_names: tuple) -> jax.sharding.PartitionSpec:
    """
    Returns a PartitionSpec object for the given `axis_names`.
    """
    return jax.sharding.PartitionSpec(*axis_names)
