# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Codegen reproducer for the Qwen 2.5 VL `split_sizes` mismatch.
###
### Qwen2.5-VL computes
###     split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
### at modeling_qwen2_5_vl.py:1208. With grid_thw = (1, 38, 58) the true
### product is 2204, but `ttnn.prod` runs on tile math units in bf16 and
### silently downcasts the i64 operand. In bf16 the step in [2048, 4096]
### is 16, so 2204 rounds up to 2208, and split_sizes becomes [552]
### instead of [551], mismatching the (551, 2048) pooler_output.
###
### This example emits the lowered TT code so the bf16-cast pattern can
### be inspected directly in the generated kernel.

import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

EXPORT_PATH = "prod_int_bf16_qwen_repro_codegen"


class IntProd(nn.Module):
    def forward(self, grid_thw):
        return grid_thw.prod(-1)


def main():
    xr.set_device_type("TT")

    # Same shape Qwen2.5-VL feeds in:
    # grid_thw = (T, H, W) = (1, 38, 58) -> prod 2204 (true) / 2208 (bf16).
    grid_thw = torch.tensor([[1, 38, 58]], dtype=torch.int64)

    codegen_py(IntProd(), grid_thw, export_path=EXPORT_PATH)


def test_prod_int_bf16_qwen_repro_codegen():
    """Codegen the i64 prod to surface the bf16 downcast."""
    output_dir = Path(EXPORT_PATH)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert output_dir.exists(), f"Expected '{EXPORT_PATH}' directory to be created"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
