# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import io
import jax
from jax.experimental import serialize_executable
import pickle


def serialize_function_to_binary(func, binary_file_path, *args, **kwargs):
    """
    Serialize a JAX function to binary format.

    Args:
        func: The function to serialize
        binary_file_path: Path to save the serialized binary
        *args: Sample arguments to trigger compilation
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
        return pid[1]

    # JIT compile the function
    jitted_func = jax.jit(func)

    # Compile with the provided arguments
    compiled = jitted_func.lower(*args, **kwargs).compile()

    # Serialize the compiled executable
    payload, _, _ = serialize_executable.serialize(compiled)

    # Extract the binary from the payload
    payload_io = io.BytesIO(payload)
    unpickler = pickle.Unpickler(payload_io)
    unpickler.persistent_load = persistent_load
    unloaded_executable, _, _ = unpickler.load()

    with open(binary_file_path, "wb") as f:
        f.write(unloaded_executable.xla_executable)
