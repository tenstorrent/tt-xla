# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Serialization tools specific to JAX.
"""

import io
import os
import pickle
from typing import Callable

import jax
from jax.experimental import serialize_executable
from ttxla_tools import parse_executable


def serialize_compiled_artifacts(func: Callable, *args, **kwargs):
    """
    Compiles JAX function for invocation with the provided arguments and serializes the compilation artifacts (TTIR graph, TTNN graph and Flatbuffer binary).
    Returns tuple of serialized artifacts in-memory.
    Please remember to provide sample arguments for compilation.

    Args:
        func: The function to serialize
        *args: Sample arguments for compilation
        **kwargs: Sample keyword arguments for compilation

    Returns:
        tuple: (ttir_mlir, ttnn_mlir, flatbuffer_binary)
    """

    def persistent_load(pid):
        """
        Custom function used during unpickling of the serialized executable.
        When JAX serializes a compiled computation (via serialize_executable.serialize()),
        it stores a set of persistent identifiers (pids) that refer to objects.

        Each pid is typically a tuple where the second element (pid[1]) is the actual
        object to be reloaded (e.g., device buffers, constants, or compilation artifacts),
        while the first element (pid[0]) is a fallback identifier.

        Args:
            pid: Persistent identifier tuple

        Returns:
            bytes: Object value to be used in deserialization
        """
        if len(pid) < 2:
            return pid[0]

        tag, data = pid[0], pid[1]
        if tag == "device":
            return jax.devices("tt")[data]

        return data

    jitted_func = jax.jit(func)
    compiled = jitted_func.lower(*args, **kwargs).compile()
    payload, _, _ = serialize_executable.serialize(compiled)

    # Extract the binary from the payload
    payload_io = io.BytesIO(payload)
    unpickler = pickle.Unpickler(payload_io)
    unpickler.persistent_load = persistent_load
    unloaded_executable, _, _ = unpickler.load()

    executable_io = io.BytesIO(unloaded_executable.xla_executable)

    return parse_executable(executable_io)


def serialize_compiled_artifacts_to_disk(
    func: Callable, *args, output_prefix: str, **kwargs
):
    """
    Compiles JAX function for invocation with the provided arguments and serializes the compilation artifacts (TTIR graph, TTNN graph and Flatbuffer binary).
    Saves the artifacts to disk using the provided output_prefix.
    Please remember to provide sample arguments for compilation.

    Creates three files: {output_prefix}_ttir.mlir, {output_prefix}_ttnn.mlir, and {output_prefix}.ttnn.
    Output directory is created if it doesn't exist.

    Example:
    ```
        serialize_function_to_disk("output/my_fn", fn, x, params=params)
        # This creates: output/my_fn_ttir.mlir, output/my_fn_ttnn.mlir, output/my_fn.ttnn
    ```

    Args:
        func (callable): The function to serialize
        *args: Positional arguments for compilation
        **kwargs: Keyword arguments for compilation
        output_prefix (str): Base path and filename prefix for output files
    """

    ttir_mlir, ttnn_mlir, flatbuffer_binary = serialize_compiled_artifacts(
        func, *args, **kwargs
    )

    dirname = os.path.dirname(output_prefix)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    ttir_path = f"{output_prefix}_ttir.mlir"
    with open(ttir_path, "w", encoding="utf-8") as f:
        f.write(ttir_mlir)

    ttnn_path = f"{output_prefix}_ttnn.mlir"
    with open(ttnn_path, "w", encoding="utf-8") as f:
        f.write(ttnn_mlir)

    flatbuffer_path = f"{output_prefix}.ttnn"
    with open(flatbuffer_path, "wb") as f:
        f.write(flatbuffer_binary)
