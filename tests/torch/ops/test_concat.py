# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
from utils import Category


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.concat",
)
def test_concat():

    class concat(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, ip):

            x = torch.cat(ip, dim=1)

            return x

    model = concat()
    model = model.to(torch.bflloat16)

    ip0 = torch.load("ip0.pt", map_location="cpu")
    ip1 = torch.load("ip1.pt", map_location="cpu")
    ip2 = torch.load("ip2.pt", map_location="cpu")
    ip3 = torch.load("ip3.pt", map_location="cpu")
    ip4 = torch.load("ip4.pt", map_location="cpu")
    ip5 = torch.load("ip5.pt", map_location="cpu")

    inputs = [[ip0, ip1, ip2, ip3, ip4, ip5]]

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=inputs,
    )

    tester.test(workload)
