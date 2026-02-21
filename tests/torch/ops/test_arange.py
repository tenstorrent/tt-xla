# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test


class ArangeOp(torch.nn.Module):
    def forward(self):
        return torch.arange(0, 913, dtype=torch.long)


def test_arange():
    run_op_test(
        ArangeOp(),
        inputs=[],
        framework=Framework.TORCH,
    )