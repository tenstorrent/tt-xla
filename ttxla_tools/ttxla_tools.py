# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import io


def parse_executable(executable):
    """
    Parse the serialized executable to extract TTIR, TTNN, and flatbuffer components.

    This an advanced function, for users that already got to a SerializedExecutable bytes object.
    Most users should use helper functions in jax or torch submodules.

    Format:
    - MAGIC: TTSERv00, 8 bytes
    - Offset + size for TTIR string, TTNN string, flatbuffer binary, 48 bytes total
    - Arbitrary data starts after the header

    Args:
        executable: Raw binary executable

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
