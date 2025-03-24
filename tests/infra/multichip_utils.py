# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from enum import Enum
import jax


class MultichipMode(Enum):
    FULLY_MANUAL = 1  # Uses both shard_map() and jax.device_put() (inputs are sharded and function has sharding ops)
    MANUAL = 2  # Uses only shard_map() (function has sharding ops)
    AUTOMATIC = 3  # Uses only jax.device_put() (inputs are sharded)

    @property
    def requires_shard_map(self) -> bool:
        return self == MultichipMode.FULLY_MANUAL or self == MultichipMode.MANUAL

    @property
    def requires_device_put(self) -> bool:
        return self == MultichipMode.FULLY_MANUAL or self == MultichipMode.AUTOMATIC


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
