# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_graph_test
from utils import Category


def generate_block_attention_mask(patch_embeds_list):
    print("patch_embeds_list", patch_embeds_list,patch_embeds_list[0])
    print("length of patch_embeds_list",len(patch_embeds_list))
    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    return block_end_idx, block_start_idx

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pixtral():
    class Pixtral(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,patch_embeds_list):
            return  generate_block_attention_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list]
            )
    op = Pixtral()
   
    patch_embeds_list = torch.load("patch_embeds_list.pt")
    run_graph_test(
        op,
        [patch_embeds_list],
        framework=Framework.TORCH,
    )