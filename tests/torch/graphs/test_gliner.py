# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_graph_test, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_gliner():
    class Matmul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, lengths=torch.tensor([49]), hidden=None):
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            packed_x = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            # packed_output, hidden = self.lstm(packed_x, hidden)
            # output, _ = pad_packed_sequence(packed_output, batch_first=True)
            # return output
            return packed_x

    op = Matmul()
    x = torch.rand(1, 49, 512)
    run_graph_test(
        op,
        [x],
        framework=Framework.TORCH,
    )
