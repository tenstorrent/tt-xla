# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

import jax
import jax.numpy as jnp
import torch
from jax import export
from jax._src.typing import DTypeLike
from runners.utils import run_on_cpu

from .types import Framework, Tensor
from .workloads import Workload


def random_tensor(
    shape: Tuple,
    dtype: str | DTypeLike | torch.dtype = "float32",
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
    framework: Framework = Framework.JAX,
) -> Tensor:
    """
    Generates a random tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`) for the desired `framework`.
    """
    return (
        random_jax_tensor(shape, dtype, random_seed, minval, maxval)
        if framework == Framework.JAX
        else random_torch_tensor(shape, dtype, random_seed, minval, maxval)
    )


@run_on_cpu(Framework.JAX)
def random_jax_tensor(
    shape: Tuple,
    dtype: Union[str, DTypeLike] = jnp.float32,
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> Tensor:
    """
    Generates a random jax tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`).
    """

    def _str_to_dtype(dtype_str: str):
        return jnp.dtype(dtype_str)

    dtype_converted = _str_to_dtype(dtype) if isinstance(dtype, str) else dtype

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
        return jax.random.choice(key=prng_key, a=jnp.array([False, True]), shape=shape)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")


@run_on_cpu(Framework.TORCH)
def random_torch_tensor(
    shape: Tuple,
    dtype: Union[str, torch.dtype] = torch.float32,
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> torch.Tensor:
    """
    Generates a random torch tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`).
    """

    def _str_to_dtype(dtype_str: str):
        return getattr(torch, dtype_str)

    dtype_converted = _str_to_dtype(dtype) if isinstance(dtype, str) else dtype

    torch.manual_seed(random_seed)

    if dtype_converted in (
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return torch.empty(shape, dtype=dtype_converted).uniform_(minval, maxval)
    elif dtype_converted in (
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        return torch.randint(int(minval), int(maxval), shape, dtype=dtype_converted)
    elif dtype_converted == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool)
    else:
        raise TypeError(f"Unsupported dtype: {dtype_converted}")


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
