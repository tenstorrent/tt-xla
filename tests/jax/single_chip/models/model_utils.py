# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Dict, List, Optional, Tuple

import flax.traverse_util
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def _process_value(k: str, v: jax.Array) -> jax.Array:
    """
    Helper function to convert between conventions of PyTorch and JAX.
    For now, this function only transposes kernels.
    Args:
    k: Name of the weight inside a flattened pytree.
        Used as a hint which conversion to apply.
    v: The value to convert.
    """
    if k.endswith(".kernel"):
        if len(v.shape) == 2:
            return jnp.transpose(v)
        if len(v.shape) == 3:
            return jnp.transpose(v, (2, 1, 0))
        if len(v.shape) == 4:
            return jnp.transpose(v, (2, 3, 1, 0))
        raise ValueError(f"Unexpected shape for kernel: {v.shape}")
    return v


def torch_statedict_to_pytree(
    state_dict: Dict[str, Any],
    patterns: List[Tuple[str, str]],
    banned_subkeys: List[str],
    dtype: Optional[jnp.dtype] = None,
) -> PyTree:
    """
    Helper function to convert a PyTorch state dict to a JAX pytree.

    Args:
    state_dict: The PyTorch state dict to convert.
    patterns: Key renamings to apply to flattened(dot separated) keys.
    banned_subkeys: Keys to exclude from the result.
    """

    # Note that is_banned_key and rewrite_key capture arguments from the outer scope
    def is_banned_key(key: str) -> bool:
        return any(banned_subkey in key for banned_subkey in banned_subkeys)

    def rewrite_key(key: str) -> str:
        is_batch_stat = "running_" in key
        prefix = "batch_stats." if is_batch_stat else "params."
        for pattern in patterns:
            key = re.sub(pattern[0], pattern[1], key)
        return prefix + key

    # ---- Logic starts here ----

    state_dict = {
        rewrite_key(k): jnp.array(v, dtype=dtype) if dtype is not None else jnp.array(v)
        for k, v in state_dict.items()
        if not is_banned_key(k)
    }
    # Conversion is done in two steps because process_value expects renamed keys
    state_dict = {k: _process_value(k, v) for k, v in state_dict.items()}
    return flax.traverse_util.unflatten_dict(state_dict, sep=".")
