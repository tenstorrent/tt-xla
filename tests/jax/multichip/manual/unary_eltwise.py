# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from infra import run_multichip_test_with_random_inputs, make_partition_spec
import pytest
from utils import compile_fail
from tests.utils import make_partition_spec


@pytest.mark.parametrize(("x_shape", "axis_names"), [((256, 256), ("x", "y"))])
@pytest.mark.skip(reason=compile_fail("Multichip still in development"))
def test_unary_eltwise(x_shape: tuple, axis_names: tuple):
    def fwd(a_block):
        b_block = jnp.negative(a_block)
        stitched_result = jax.lax.psum(b_block, axis_names)
        return stitched_result

    def fwd_single_device(a_block):
        a1, a2 = jnp.split(a_block, 2, axis=1)

        b1, b2 = jnp.negative(a1), jnp.negative(a2)

        stitched_result = b1 + b2
        return stitched_result

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec((None, None))

    run_multichip_test_with_random_inputs(
        fwd, fwd_single_device, [x_shape], (1, 2), axis_names, in_specs, out_specs
    )
