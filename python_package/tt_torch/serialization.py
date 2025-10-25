# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Serialization tools specific to PyTorch.
"""

import io
import os
import shutil

from ttxla_tools import parse_executable


def parse_compiled_artifacts_from_cache(cache_path: str):
    """
    Load a serialized executable from PyTorch persistent cache and parse it.
    Expects there to be only one file in the cache directory.

    Args:
        cache_path (str): Path to the persistent cache folder

    Returns:
        tuple: (ttir_mlir, ttnn_mlir, flatbuffer_binary)

    Raises:
        FileNotFoundError: If cache_path doesn't exist or is empty
        AssertionError: If multiple files are found in cache_path
        ValueError: If no files are found in cache_path
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache path does not exist: {cache_path}")

    if not os.path.isdir(cache_path):
        raise ValueError(f"Cache path is not a directory: {cache_path}")

    # Find all files in the cache directory
    files = [
        f for f in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, f))
    ]

    if len(files) == 0:
        raise ValueError(f"No files found in cache directory: {cache_path}")

    assert len(files) == 1, (
        f"Expected exactly one file in cache directory {cache_path}, found {len(files)}: {files}. \n"
        f"You can manually load the right file and pass it to parse_executable from the parent module"
    )

    cache_file_path = os.path.join(cache_path, files[0])

    # Load the cached executable
    with open(cache_file_path, "rb") as f:
        executable_data = f.read()

    executable_io = io.BytesIO(executable_data)

    return parse_executable(executable_io)


def parse_compiled_artifacts_from_cache_to_disk(cache_path: str, output_prefix: str):
    """
    Load a serialized executable from PyTorch persistent cache and save components to disk.
    Expects there to be only one file in the cache directory.

    Creates three files: {output_prefix}_ttir.mlir, {output_prefix}_ttnn.mlir, and {output_prefix}.ttnn.
    Output directory is created if it doesn't exist.

    Example:
    ```
        parse_from_cache_to_disk("/path/to/cache", "output/my_model")
        # This creates: output/my_model_ttir.mlir, output/my_model_ttnn.mlir, output/my_model.ttnn
    ```

    Args:
        cache_path (str): Path to the persistent cache folder
        output_prefix (str): Base path and filename prefix for output files


    Raises:
        FileNotFoundError: If cache_path doesn't exist or is empty
        AssertionError: If multiple files are found in cache_path
        ValueError: If no files are found in cache_path
    """
    ttir_mlir, ttnn_mlir, flatbuffer_binary = parse_compiled_artifacts_from_cache(
        cache_path
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

    shutil.rmtree(cache_path)
