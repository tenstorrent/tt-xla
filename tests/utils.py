# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Callable

import jax
import jax.lax as jlx
import jax.numpy as jnp


def compile_fail(reason: str) -> str:
    return f"Compile failed: {reason}"


def runtime_fail(reason: str) -> str:
    return f"Runtime failed: {reason}"


# NOTE TTNN does not support boolean data type, so bfloat16 is used instead.
# The output of logical operation (and other similar ops) is bfloat16. JAX can
# not perform any computation due to mismatch in output data type (in testing
# infrastructure). The following tests explicitly convert data type of logical
# operation output for the verification purposes.
# TODO Remove this workaround once the data type issue is resolved.
# https://github.com/tenstorrent/tt-xla/issues/93
# TODO investigate why this decorator cannot be removed. See issue
# https://github.com/tenstorrent/tt-xla/issues/156
def convert_output_to_bfloat16(f: Callable):
    """Decorator to work around the mentioned issue."""

    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        return jlx.convert_element_type(res, jnp.bfloat16)

    return wrapper


@contextmanager
def enable_x64():
    """
    Context manager that temporarily enables x64 in jax.config.

    Isolated as a context manager so that it doesn't change global config for all jax
    imports and cause unexpected fails elsewhere.
    """
    try:
        # Set the config to True within this block, and yield back control.
        jax.config.update("jax_enable_x64", True)
        yield
    finally:
        # After `with` statement ends, turn it off again.
        jax.config.update("jax_enable_x64", False)
