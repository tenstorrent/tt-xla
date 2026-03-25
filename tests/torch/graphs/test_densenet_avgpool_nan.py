# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer for DenseNet121 PCC drop.

Root cause: AvgPool2d(kernel_size=2, stride=2) in transition1 produces NaN/Inf
on TT hardware when preceded by the full stem+denseblock1+transition1.conv graph.

Test runs features through transition1.pool and asserts no NaN in TT output.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig
from tests.infra.evaluators.evaluation_config import PccConfig

from third_party.tt_forge_models.densenet.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


class Transition1Wrapper(nn.Module):
    """stem → denseblock1 → transition1 (full, including AvgPool2d)."""

    def __init__(self, model):
        super().__init__()
        f = model.features
        self.conv0 = f.conv0
        self.norm0 = f.norm0
        self.relu0 = f.relu0
        self.pool0 = f.pool0
        self.denseblock1 = f.denseblock1
        self.transition1 = f.transition1

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        return x


@pytest.mark.single_device
def test_densenet121_transition1_avgpool_nan(request):
    """
    Reproduces the NaN/Inf produced by AvgPool2d in transition1 on TT hardware.

    Expected: TT output matches CPU (PCC >= 0.99) — no NaN or Inf.
    Actual:   TT produces NaN/Inf from AvgPool2d(kernel_size=2, stride=2).
    """
    loader = ModelLoader(variant=ModelVariant.DENSENET121)
    model = loader.load_model().eval()
    inputs = [loader.load_inputs()]

    run_op_test(
        Transition1Wrapper(model),
        inputs,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
        framework=Framework.TORCH,
        request=request,
    )
