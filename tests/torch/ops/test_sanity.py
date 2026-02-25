# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
BiLSTM-CRF sanity tests.

- test_sanity_bilstm_padded: LSTM on padded input (works with torch.compile/XLA)
- test_sanity_bilstm_packed: pack_padded_sequence + LSTM (fails - "data is not allocated")
"""

import pytest
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from infra.comparators import ComparisonConfig
from infra.utilities.types import Framework
from infra.workloads import Workload
from loguru import logger

from tests.infra.testers.single_chip.op.op_tester import OpTester



@pytest.mark.single_device

def test_sanity_bilstm_packed():
    """pack_padded_sequence + LSTM - fails with 'data is not allocated' (expected)."""

    class PackedLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 4)
            self.lstm = torch.nn.LSTM(4, 4, batch_first=True, bidirectional=True)

        def forward(self, x):
            # Must be descending for pack_padded_sequence (enforce_sorted=True by default)
            lengths = torch.tensor([5, 3], device=x.device, dtype=torch.long)
            embeds = self.emb(x)
            pack = pack_padded_sequence(embeds, lengths=lengths, batch_first=True)
            out, _ = self.lstm(pack)
            out, _ = pad_packed_sequence(out, batch_first=True)
            return out

    model = PackedLSTM()
    x = torch.randint(0, 10, (2, 5))

    logger.info("PackedLSTM input shape={}", x.shape)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[x],
    )

    tester.test(workload)
