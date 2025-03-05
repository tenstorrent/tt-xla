# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import jax


def compile_fail(reason: str) -> str:
    return f"Compile failed: {reason}"


def runtime_fail(reason: str) -> str:
    return f"Runtime failed: {reason}"


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
