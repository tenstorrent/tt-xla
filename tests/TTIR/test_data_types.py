# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %PYTHON %s | FileCheck %s


import jax.numpy as jnp

# Currently, tt::runtime only support float32, bfloat16, uint16, and uint32


def test_data_types(capfd):
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.bfloat16)
    c = jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32)
    d = jnp.array([[5, 6], [7, 8]], dtype=jnp.uint16)
    print(a)
    out, _ = capfd.readouterr()
    assert "[[1. 2.]\n [3. 4.]]" in out

    print(b)
    out, _ = capfd.readouterr()
    assert "[[5 6]\n [7 8]]" in out

    print(c)
    out, _ = capfd.readouterr()
    assert "[[1 2]\n [3 4]]" in out

    print(d)
    out, _ = capfd.readouterr()
    assert "[[5 6]\n [7 8]]" in out
