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

    MAGIC_SIZE = 8
    HEADER_SIZE = MAGIC_SIZE + 6 * 8

    # Verify magic number
    magic = executable_io.read(MAGIC_SIZE)
    if magic != b"TTSERv00":
        raise ValueError(f"Invalid magic number in extracted payload: {magic}")

    # Read offsets and sizes (6 * 8 bytes = 48 bytes)
    offset_ttir = int.from_bytes(executable_io.read(8), "little")
    size_ttir = int.from_bytes(executable_io.read(8), "little")
    offset_ttnn = int.from_bytes(executable_io.read(8), "little")
    size_ttnn = int.from_bytes(executable_io.read(8), "little")
    offset_flatbuffer = int.from_bytes(executable_io.read(8), "little")
    size_flatbuffer = int.from_bytes(executable_io.read(8), "little")

    assert (
        offset_ttir <= offset_ttnn <= offset_flatbuffer
    ), "Unexpected order of sections in body"
    assert offset_ttir + size_ttir <= offset_ttnn, "TTIR and TTNN segments overlap"
    assert (
        offset_ttnn + size_ttnn <= offset_flatbuffer
    ), "TTNN and flatbuffer segments overlap"
    assert (
        offset_flatbuffer + size_flatbuffer <= len(executable) - HEADER_SIZE
    ), "Flatbuffer segment exceeds executable size"

    executable_io.seek(HEADER_SIZE + offset_ttir)
    ttir_mlir = executable_io.read(size_ttir).decode("utf-8")

    executable_io.seek(HEADER_SIZE + offset_ttnn)
    ttnn_mlir = executable_io.read(size_ttnn).decode("utf-8")

    executable_io.seek(HEADER_SIZE + offset_flatbuffer)
    flatbuffer_binary = executable_io.read(size_flatbuffer)

    return ttir_mlir, ttnn_mlir, flatbuffer_binary
