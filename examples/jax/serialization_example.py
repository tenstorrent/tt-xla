# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import jax.numpy as jnp
from tt_jax import serialize_compiled_artifacts_to_disk


def add(x, y):
    return x + y


def main():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])

    serialize_compiled_artifacts_to_disk(add, a, b, output_prefix="output/add")


def test_serialization_artifacts():
    """Test that serialization creates expected output files."""
    output_dir = Path("output")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        main()

        expected_files = [
            Path("output/add.ttnn"),
            Path("output/add_ttnn.mlir"),
            Path("output/add_ttir.mlir"),
        ]

        for filepath in expected_files:
            assert filepath.exists(), f"Expected file {filepath} was not created"

        print("All expected files were created successfully.")

    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Cleaned up {output_dir}")


if __name__ == "__main__":
    main()
