# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import make_partition_spec, run_multichip_test_with_random_inputs

from tests.utils import failed_fe_compilation


@pytest.mark.parametrize(
    ("input_shape", "mesh_shape", "axis_names"), [((256, 256), (1, 2), ("x", "y"))]
)
@pytest.mark.skip(reason=failed_fe_compilation("Multichip still in development"))
def test_unary_eltwise(input_shape: tuple, mesh_shape: tuple, axis_names: tuple):
    def fwd(a_block):
        b_block = jnp.negative(a_block)
        return b_block

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd, [input_shape], mesh_shape, axis_names, in_specs, out_specs
    )
