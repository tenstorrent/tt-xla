# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig
from tests.infra.evaluators.evaluation_config import PccConfig

from third_party.tt_forge_models.centernet.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


class RegWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.reg = model.reg
        self.first_level = model.first_level
        self.last_level = model.last_level

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return self.reg(y[-1])


@pytest.mark.single_device
def test_centernet_reg_head_pcc_drop(request):
    loader = ModelLoader(variant=ModelVariant.DLA_1X_COCO)
    model = loader.load_model().eval()
    inputs = [loader.load_inputs()]

    run_op_test(
        RegWrapper(model),
        inputs,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.97)),
        framework=Framework.TORCH,
        request=request,
    )
