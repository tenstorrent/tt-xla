# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from utils import Category
from loguru import logger

@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.reshape",
)
def test_sanity1():

    class s1(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,  mask_gt, outputs, op_threshs):
            
            # op = ((1.0 - outputs[mask_gt]) / ((1 - op_threshs[mask_gt]) * 2))
            
            return outputs[mask_gt], op_threshs[mask_gt], 1.0 - outputs[mask_gt], 1 - op_threshs[mask_gt] ,  1 - op_threshs[mask_gt] * 2 
            

    model = s1()
    
    mask_gt = torch.tensor([[False, False,  True, False, False,  True,  True, False, False, False,
         False,  True,  True, False, False,  True,  True, False]])
    outputs = torch.tensor([[0.0487, 0.0329, 0.1553, 0.0057, 0.0011, 0.0046, 0.0474, 0.0574, 0.0211,
         0.0131, 0.0367, 0.3945, 0.2926, 0.0009, 0.0012, 0.0729, 0.3469, 0.0273]])
    op_threshs = torch.tensor([[0.0742, 0.0383, 0.0981, 0.0098, 0.0236, 0.0022, 0.0101, 0.1032, 0.0568,
         0.0268, 0.0503, 0.0240, 0.0194, 0.0429, 0.0534, 0.0360, 0.2021, 0.0501]])

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[mask_gt, outputs,op_threshs],
    )

    tester.test(workload)