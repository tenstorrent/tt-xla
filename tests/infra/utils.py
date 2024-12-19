# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from flax import linen, nnx
from jax import export


@dataclass
class Workload:
    executable: Callable
    args: Sequence[Any]
    kwargs: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        # If kwargs is None, initialize it to an empty dictionary.
        if self.kwargs is None:
            self.kwargs = {}

    def execute(self) -> Any:
        return self.executable(*self.args, **self.kwargs)


class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"
    NUMPY = "numpy"


# Convenience alias. Could be used to represent jax.Array, torch.Tensor, np.ndarray, etc.
Tensor = Union[jax.Array]

# Convenience alias. Could be used to represent nnx.Module, torch.nn.Module, etc.
# NOTE nnx.Module is the newest API, linen.Module is legacy but it is used in some
# huggingface models.
Model = Union[nnx.Module, linen.Module]


def _str_to_dtype(dtype_str: str, framework: Framework = Framework.JAX):
    """Convert a string dtype to the corresponding framework-specific dtype."""
    if framework == Framework.JAX:
        return jnp.dtype(dtype_str)
    else:
        raise ValueError(f"Unsupported framework: {framework.value}.")


def random_tensor(
    shape: tuple,
    dtype: str = "float32",
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
    framework: Framework = Framework.JAX,
) -> Tensor:
    """
    Generates a random tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`) for the desired `framework`.
    """
    # Convert dtype string to actual dtype for the selected framework.
    dtype_converted = _str_to_dtype(dtype, framework)

    # Generate random tensor based on framework type
    if framework == Framework.JAX:
        prng_key = jax.random.PRNGKey(random_seed)

        return jax.random.uniform(
            key=prng_key,
            shape=shape,
            dtype=dtype_converted,
            minval=minval,
            maxval=maxval,
        )
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
