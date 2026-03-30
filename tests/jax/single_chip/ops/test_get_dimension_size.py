# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from jax.export import export, symbolic_args_specs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.shape",
    shlo_op_name="stablehlo.get_dimension_size",
)
@pytest.mark.parametrize(
    ["x_shape", "dimension"],
    [
        ((32, 64), 0),
        ((32, 64), 1),
        ((8, 32, 256), 0),
        ((8, 32, 256), 1),
        ((8, 32, 256), 2),
    ],
    ids=lambda val: f"{val}",
)
def test_get_dimension_size(x_shape: tuple, dimension: int):
    """Tests get_dimension_size using jax.export with dynamic shapes.

    In standard JAX tracing shapes are static so x.shape[i] is a constant.
    Using jax.export with polymorphic shapes forces the compiler to emit
    stablehlo.get_dimension_size to query the dimension at runtime.
    """

    def f(x: jax.Array) -> jax.Array:
        return x + x.shape[dimension]

    # Build symbolic shape spec with a dynamic variable per dimension.
    dim_vars = ", ".join(f"d{i}" for i in range(len(x_shape)))
    dummy = jnp.zeros(x_shape, dtype=jnp.float32)
    args_specs = symbolic_args_specs((dummy,), (dim_vars,))

    # Export with dynamic shapes — this produces get_dimension_size in StableHLO.
    exported = export(jax.jit(f), platforms=["tt"])(*args_specs)

    # Run on CPU for reference.
    cpu_device = jax.devices("cpu")[0]
    cpu_x = jax.device_put(dummy, cpu_device)
    cpu_res = jax.jit(f)(cpu_x)

    # Run exported module on TT device.
    tt_device = jax.devices("tt")[0]
    tt_x = jax.device_put(dummy, tt_device)
    tt_res = exported.call(tt_x)
    tt_res = jax.device_put(tt_res, cpu_device)

    assert jnp.allclose(cpu_res, tt_res, atol=1e-2)
