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

        def forward(self,inputs_embeds,special_image_mask,image_features):
        
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            return inputs_embeds

    op = Pixtral()
   
    inputs_embeds = torch.load("inputs_embeds.pt")
    special_image_mask = torch.load("special_image_mask.pt")
    image_features = torch.load("image_features.pt")
    run_graph_test(
        op,
        [inputs_embeds, special_image_mask, image_features],
        framework=Framework.TORCH,
    )