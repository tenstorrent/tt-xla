# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
import jax
import jax.numpy as jnp
from jax import export
from jax._src.typing import DTypeLike
from typing import Union

from .device_runner import run_on_cpu
from .types import Framework, Tensor
from .workload import Workload


def _str_to_dtype(dtype_str: str, framework: Framework = Framework.JAX):
    """Convert a string dtype to the corresponding framework-specific dtype."""
    if framework == Framework.JAX:
        return jnp.dtype(dtype_str)
    else:
        raise ValueError(f"Unsupported framework: {framework.value}.")


@run_on_cpu
def random_tensor(
    shape: tuple,
    dtype: Union[str, DTypeLike] = jnp.float32,
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
    framework: Framework = Framework.JAX,
) -> Tensor:
    """
    Generates a random tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`) for the desired `framework`.
    """
    dtype_converted = (
        _str_to_dtype(dtype, framework) if isinstance(dtype, str) else dtype
    )

    # Generate random tensor based on framework type.
    if framework == Framework.JAX:
        prng_key = jax.random.PRNGKey(random_seed)

        if jnp.issubdtype(dtype_converted, jnp.integer):
            return jax.random.randint(
                key=prng_key,
                shape=shape,
                dtype=dtype_converted,
                minval=int(minval),
                maxval=int(maxval),
            )
        elif jnp.issubdtype(dtype_converted, jnp.floating):
            return jax.random.uniform(
                key=prng_key,
                shape=shape,
                dtype=dtype_converted,
                minval=minval,
                maxval=maxval,
            )
        elif jnp.issubdtype(dtype_converted, jnp.bool):
            # Generate random tensor of type bool.
            return jax.random.choice(
                key=prng_key, a=jnp.array([False, True]), shape=shape
            )
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")
    else:
        raise ValueError(f"Unsupported framework: {framework.value}.")


def workload_as_mlir_module(
    workload: Workload, framework: Framework = Framework.JAX
) -> str:
    """
    Returns workload as mlir module string.

    Note that in case of jax, workload.executable must be the result of jit, otherwise
    empty string will be returned.
    """

    if framework == Framework.JAX:
        try:
            s = export.export(workload.executable)(
                *workload.args, **workload.kwargs
            ).mlir_module()

            # Remove all lines that start with "#loc" for cleaner output.
            return "\n".join(
                line for line in s.splitlines() if not line.startswith("#loc")
            )

        except ValueError:
            return ""
    else:
        raise ValueError(f"Unsupported framework: {framework.value}.")


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
