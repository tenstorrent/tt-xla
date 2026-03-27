# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import copy

def test_masked_scatter():

    class masked_scatter(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs_embeds,special_image_mask,image_features):
            return inputs_embeds.masked_scatter(special_image_mask, image_features)


    # model
    model = masked_scatter()
    model.eval()

    # inputs
    inputs_embeds = torch.load('inputs_embeds.pt',map_location="cpu")
    special_image_mask = torch.load('special_image_mask.pt',map_location="cpu")
    image_features = torch.load('image_features.pt',map_location="cpu")

    logger.info("inputs_embeds={}",inputs_embeds)
    logger.info("inputs_embeds.shape={}",inputs_embeds.shape)
    logger.info("inputs_embeds.dtype={}",inputs_embeds.dtype)

    logger.info("special_image_mask={}",special_image_mask)
    logger.info("special_image_mask.shape={}",special_image_mask.shape)
    logger.info("special_image_mask.dtype={}",special_image_mask.dtype)

    logger.info("image_features={}",image_features)
    logger.info("image_features.shape={}",image_features.shape)
    logger.info("image_features.dtype={}",image_features.dtype)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs_embeds,special_image_mask,image_features],
    )

    tester.test(workload)