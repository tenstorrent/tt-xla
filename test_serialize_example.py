#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Example of using the serialization functionality through op_tester."""

import sys
import os

# Add the tests directory to the path to import infra modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import jax
import jax.numpy as jnp
from infra.testers.single_chip.op.op_tester import serialize_op


def add(x, y):
    """Simple addition function."""
    return x + y


def sub(x, y):
    """Matrix multiplication function."""
    return x - y


def main():
    # Example 1: Serialize with specific inputs
    print("Example 1: Serialize add operation with specific inputs")
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])

    serialize_op(add, [a, b], "output/test_add")
    print(
        "  Created: output/test_add_ttir.mlir, output/test_add_ttnn.mlir, output/test_add.ttnn"
    )

    serialize_op(sub, [a, b], "output/test_sub")
    print(
        "  Created: output/test_sub_ttir.mlir, output/test_sub_ttnn.mlir, output/test_sub.ttnn"
    )


if __name__ == "__main__":
    main()
