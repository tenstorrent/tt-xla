# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


# duplicated to have all fusion tests in the same place, also it uses the F.rms_norm rather than pulling from HF model code
@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["rms_norm.ttir.mlir"])
def test_rmsnorm(request):
    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(x, (256,), gamma)

    run_graph_test_with_random_inputs(
        rmsnorm,
        [(64, 256), (256,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
