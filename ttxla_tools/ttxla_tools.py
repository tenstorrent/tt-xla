# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import io
import jax
from jax.experimental import serialize_executable
import os
import pickle


def serialize_function(func, output_prefix, *args, **kwargs):
    """
    Serialize a JAX function and extract TTIR, TTNN, and flatbuffer components.

    Args:
        func: The function to serialize
        output_prefix: Path + base name for output files
        *args: Sample arguments for compilation
        **kwargs: Sample keyword arguments for compilation
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

    ttir_mlir, ttnn_mlir, flatbuffer_binary = _parse_executable(
        unloaded_executable.xla_executable
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


def _parse_executable(executable):
    """
    Parse the serialized payload to extract TTIR, TTNN, and flatbuffer components.

    Format:
    - MAGIC: TTSERv00, 8 bytes
    - Offset + size for TTIR string, TTNN string, flatbuffer binary, 48 bytes total
    - Arbitrary data starts after the header

    Args:
        extracted_payload: Raw binary payload

    Returns:
        tuple: (ttir_mlir, ttnn_mlir, flatbuffer_binary)
    """
    executable_io = io.BytesIO(executable)

    # Verify magic number
    magic = executable_io.read(8)
    if magic != b"TTSERv00":
        raise ValueError(f"Invalid magic number in extracted payload: {magic}")

    # Read offsets and sizes (6 * 8 bytes = 48 bytes)
    offset_ttir = int.from_bytes(executable_io.read(8), "little")
    size_ttir = int.from_bytes(executable_io.read(8), "little")
    offset_ttnn = int.from_bytes(executable_io.read(8), "little")
    size_ttnn = int.from_bytes(executable_io.read(8), "little")
    offset_flatbuffer = int.from_bytes(executable_io.read(8), "little")
    size_flatbuffer = int.from_bytes(executable_io.read(8), "little")

    # Read TTIR MLIR
    executable_io.seek(8 + 48 + offset_ttir)
    ttir_mlir = executable_io.read(size_ttir).decode("utf-8")

    # Read TTNN MLIR
    executable_io.seek(8 + 48 + offset_ttnn)
    ttnn_mlir = executable_io.read(size_ttnn).decode("utf-8")

    # Read flatbuffer binary
    executable_io.seek(8 + 48 + offset_flatbuffer)
    flatbuffer_binary = executable_io.read(size_flatbuffer)

    return ttir_mlir, ttnn_mlir, flatbuffer_binary
