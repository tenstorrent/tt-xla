# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category, TTArch, get_tt_device_arch


def check_device_compatibility(input_shape: tuple, k: int):
    """
    Check device architecture and conditionally xfail or skip tests.
    - All tests are xfailed on Blackhole
    - Last only some are skipped on Wormhole
    """
    device_arch = get_tt_device_arch()

    if device_arch == TTArch.BLACKHOLE:
        pytest.xfail(
            "All topk tests are xfailed on Blackhole architecture with bad PCC - https://github.com/tenstorrent/tt-xla/issues/1797"
        )
    elif device_arch == TTArch.WORMHOLE_B0:
        if (input_shape == (1, 40) and k == 5) or (
            input_shape == (1, 8400) and k == 300
        ):
            pytest.xfail(
                "Test skipped on Wormhole architecture with bad PCC - https://github.com/tenstorrent/tt-xla/issues/1797"
            )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    ["input_shape", "k"],
    [
        ((1, 10), 5),
        ((1, 20), 5),
        ((1, 30), 5),
        ((1, 40), 5),
        ((1, 8400), 300),
    ],
)
def test_topk_indices(input_shape: tuple, k: int):
    """Test topk operation returning indices."""

    check_device_compatibility(input_shape, k)

    class TopKIndices(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k)[1]

    model = TopKIndices(k)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.float32, framework=Framework.TORCH
    )
