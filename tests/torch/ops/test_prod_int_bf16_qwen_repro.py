# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Sanity reproducer for the Qwen 2.5 VL `split_sizes` mismatch: an i64
# `torch.prod` (stablehlo.reduce with mul body) over `image_grid_thw`
# is lowered to ttnn.prod, which runs on tile math units in bf16 and
# silently downcasts the integer operand. For grid_thw=(1, 38, 58) the
# true product is 2204, but bf16 rounds 2204 -> 2208 (step is 16 in
# [2048, 4096]), so split_sizes becomes 552 instead of 551 and
# torch.split mismatches the (551, 2048) pooler_output.
# Pins this scenario for the follow-up tt-mlir fix.

import torch
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, run_graph_test
from torch import nn


class IntProd(nn.Module):
    def forward(self, grid_thw):
        x = grid_thw.prod(-1)
        print("prod:", x)
        return x

def test_prod_int_bf16_qwen_repro():
    xr.set_device_type("TT")

    # Same shape Qwen2.5-VL feeds at this line:
    #   split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
    # grid_thw = (T, H, W) = (1, 38, 58) -> prod 2204. bf16 round-trip
    # of 2204 yields 2208, exposing the bug.
    grid_thw = torch.tensor([[1, 38, 58]], dtype=torch.int64)

    # Default ComparisonConfig only checks PCC, which falls back to
    # allclose for near-equal scalars (2204 vs 2208 differ by ~0.18%
    # and would silently pass). Force strict equality for this integer
    # correctness test.
    config = ComparisonConfig()
    config.equal.enable()
    config.pcc.disable()

    run_graph_test(
        IntProd(),
        [grid_thw],
        comparison_config=config,
        framework=Framework.TORCH,
    )
