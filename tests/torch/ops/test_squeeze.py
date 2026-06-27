# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


# A no-op ``squeeze(dim)`` (where ``dim`` is not of size 1) lowers to
# ``prims.view_of``, an identity alias-view that the functionalization pass could
# not handle, breaking compilation (tt-xla #5375). These cases exercise both the
# no-op path (the regression) and ordinary size-1 squeezes to guard the fix.
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "input_shape,dim",
    [
        # No-op squeezes: dim is NOT size 1 -> previously failed on prims.view_of
        ((1, 46, 1024), 1),
        ((1, 46, 1024), 2),
        ((4, 8), 0),
        ((3, 5, 7), -1),
        # Real squeezes: dim IS size 1
        ((1, 46, 1024), 0),
        ((4, 1, 8), 1),
        ((2, 3, 1), 2),
    ],
)
def test_squeeze_dim(input_shape, dim):
    """Squeeze on a given dim (no-op when that dim != 1, real otherwise)."""

    class Squeeze(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            # Add an op so the squeezed (possibly aliasing) tensor is consumed,
            # mirroring how no-op squeezes appear inside real model graphs.
            return x.squeeze(self.dim) + 1.0

    run_op_test_with_random_inputs(
        Squeeze(dim), [input_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 46, 1024),  # only one size-1 dim
        (1, 1, 46),  # multiple size-1 dims
        (4, 8),  # no size-1 dims -> squeeze() is a pure no-op
    ],
)
def test_squeeze_all(input_shape):
    """Dimensionless squeeze() removes all size-1 dims (no-op if none)."""

    class SqueezeAll(torch.nn.Module):
        def forward(self, x):
            return x.squeeze() + 1.0

    run_op_test_with_random_inputs(
        SqueezeAll(), [input_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )
