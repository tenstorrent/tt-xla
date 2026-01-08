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
            self.patch_size = 14

        def forward(self, patch_embeds, image_sizes):
            patch_embeds_list = [
                embed[
                    ..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)
                ]
                for embed, size in zip(patch_embeds, image_sizes)
            ]

            return patch_embeds_list

    op = Matmul()
    x = torch.rand(1, 1024, 28, 38)
    image_sizes = torch.tensor([[392, 532]])
    run_graph_test(
        op,
        [x, image_sizes],
        framework=Framework.TORCH,
    )
