# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
import torch
from third_party.tt_forge_models.olm_ocr.image_text_generation.pytorch.loader import ModelLoader,ModelVariant
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs
from ttxla_tools.logging import logger


class visual_blk(torch.nn.Module):
    def __init__(self,module):
        super().__init__()
        self.blocks = module.model.visual.blocks
        self.fullatt_block_indexes = module.model.visual.fullatt_block_indexes


    def forward(self,hidden_states,cu_seqlens,cu_window_seqlens,position_embeddings):
        logger.info(f"self.fullatt_block_indexes is {self.fullatt_block_indexes}")
        for layer_num, blk in enumerate(self.blocks):
            logger.info(f"layer_num is {layer_num,hidden_states.shape}")
            # if layer_num==14: # OOM starts with layer_num - 15
            #     break

            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        return hidden_states

variants = [
    ModelVariant.OLM_OCR_7B_0725,
    ModelVariant.OLM_OCR_7B_0825,
    ModelVariant.OLM_OCR_2_7B_1025
           ]
@pytest.mark.parametrize("variant",variants)
def test_visual_blk(variant):
    loader = ModelLoader(variant)
    module = loader.load_model()
    data = torch.load(f"block_inputs.pt", map_location="cpu")
    run_op_test_with_saved_inputs(
        visual_blk(module),
        [
            data["hidden_states"],
            data["cu_seqlens"],
            data["cu_window_seqlens"],
            data["position_embeddings"],
        ],
        framework=Framework.TORCH,
    )