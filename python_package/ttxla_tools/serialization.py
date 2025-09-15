# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import io


def parse_executable(executable_io: io.BytesIO):
    """
    Parse the serialized executable to extract TTIR, TTNN, and flatbuffer components.

    This an advanced function, for users that already got to a SerializedExecutable bytes object.
    Most users should use helper functions in jax or torch submodules.

    Serialization format:
      1. Magic string: "TTSERv00", 8 bytes
      2. Header: 48 bytes, 3*2*sizeof(u64)
        2.1. Offset + size for TTIR, 2*sizeof(u64)
        2.2. Offset + size for TTNN, 2*sizeof(u64)
        2.3. Offset + size for Flatbuffer, 2*sizeof(u64)
      3. Body: variable size
        3.1. TTIR, variable size
        3.2. TTNN, variable size
        3.3. Flatbuffer, variable size
    Total size: 8 + 48 + TTIR.size() + TTNN.size() + flatbuffer_data.size()

    Serialization format:
      1. Header, 56 bytes
        1.1. Magic string: "TTSERv00", 8 bytes
        1.2  (Offset, Size) pairs for each section in data, 48 bytes
          1.2.1. Offset + size for TTIR, 2*sizeof(u64)
          1.2.2. Offset + size for TTNN, 2*sizeof(u64)
          1.2.3. Offset + size for Flatbuffer, 2*sizeof(u64)
      2. Body: variable size
        2.1. TTIR, variable size
        2.2. TTNN, variable size
        2.3. Flatbuffer, variable size
    The offsets are relative to the start of the body.
    Subsections in 2. are **NOT** assumed to be contiguous.
    However the current serializer does pack them contiguously.
    Subsections in 2. are assumed to be in the order defined above.
    Total size: 56: + (flatbuffer_offset + flatbuffer_size)
    This struct contains only the header part.

    Args:
        executable: Raw binary executable

    Returns:
        tuple: (ttir_mlir, ttnn_mlir, flatbuffer_binary)
    """

    MAGIC_SIZE = 8
    HEADER_SIZE = MAGIC_SIZE + 6 * 8

    # Verify magic number
    magic = executable_io.read(MAGIC_SIZE)
    if magic != b"TTSERv00":
        raise ValueError(f"Invalid magic number in extracted payload: {magic}")

    total_size = executable_io.getbuffer().nbytes

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
        offset_flatbuffer + size_flatbuffer <= total_size - HEADER_SIZE
    ), "Flatbuffer segment exceeds executable size"

    executable_io.seek(HEADER_SIZE + offset_ttir)
    ttir_mlir = executable_io.read(size_ttir).decode("utf-8")

    executable_io.seek(HEADER_SIZE + offset_ttnn)
    ttnn_mlir = executable_io.read(size_ttnn).decode("utf-8")

    executable_io.seek(HEADER_SIZE + offset_flatbuffer)
    flatbuffer_binary = executable_io.read(size_flatbuffer)

    return ttir_mlir, ttnn_mlir, flatbuffer_binary
