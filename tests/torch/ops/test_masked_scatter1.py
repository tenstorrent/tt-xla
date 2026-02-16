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

        def forward(self, inputs_embeds, images_seq_mask, images_in_this_batch):
            # In the real model, masked_scatter_ is inside complex
            # control flow (for loops, conditionals, list ops) that
            # Dynamo cannot trace. This forces it to run eagerly on
            # XLA lazy tensors, bypassing decompositions.py entirely.
            # @torch.compiler.disable mimics that exact behavior.
            @torch.compiler.disable
            def eager_masked_scatter(embeds, mask, source):
                return embeds.masked_scatter_(mask, source)

            output = eager_masked_scatter(
                inputs_embeds, images_seq_mask, images_in_this_batch
            )
            return output


    # model
    model = masked_scatter()
    model.eval()
    
    # inputs
    inputs_embeds = torch.load('input_embeds.pt',map_location="cpu")
    images_seq_mask = torch.load('images_seq_mask.pt',map_location="cpu")
    images_in_this_batch = torch.load('images_in_this_batch.pt',map_location="cpu")
    
    # logger.info("inputs_embeds={}",inputs_embeds)
    # logger.info("inputs_embeds.shape={}",inputs_embeds.shape)
    # logger.info("inputs_embeds.dtype={}",inputs_embeds.dtype)
    
    # logger.info("images_seq_mask={}",images_seq_mask)
    # logger.info("images_seq_mask.shape={}",images_seq_mask.shape)
    # logger.info("images_seq_mask.dtype={}",images_seq_mask.dtype)
    
    # logger.info("images_in_this_batch={}",images_in_this_batch)
    # logger.info("images_in_this_batch.shape={}",images_in_this_batch.shape)
    # logger.info("images_in_this_batch.dtype={}",images_in_this_batch.dtype)
    

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs_embeds,images_seq_mask,images_in_this_batch],
    )

    tester.test(workload)