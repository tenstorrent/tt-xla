# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_graph_test
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pixtral():
    class Pixtral(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sliding_window = 4096

        def forward(self, full_key_states):

            keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
            return keys

    op = Pixtral()

    full_key_states = torch.randn(1, 32, 16, 128)
    run_graph_test(
        op,
        [full_key_states],
        framework=Framework.TORCH,
    )
