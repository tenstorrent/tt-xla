# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Codegen for torch.gather with DeepSeek OCR exact shapes.

Reproduces the failing gather call from the gather-based masked_scatter
decomposition in DeepseekOCRModel.forward:

    source       : [903, 1280] bfloat16  (image features)
    source_idx_2d: [913, 1280] int64     (from cumsum on images_seq_mask)
    dim          : 0

The index is built the same way as the model: cumsum → clamp → unsqueeze → expand.

Related issues:
  - https://github.com/tenstorrent/tt-xla/issues/3316
  - https://github.com/tenstorrent/tt-xla/issues/3412
  - https://github.com/tenstorrent/tt-xla/issues/4167
"""

import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

S = 913
D = 1280
N = 903


class GatherDim0(nn.Module):
    """torch.gather(input, 0, index) — the exact op from the decomposition."""

    def forward(self, input, index):
        return torch.gather(input, 0, index)


def _build_inputs(seed=42):
    """Build gather inputs matching exactly what the DeepSeek OCR
    gather-based masked_scatter decomposition produces.

    Pipeline on CPU:
        mask_1d       = bool [913], 903 True / 10 False (shuffled)
        mask_i        = mask_1d.long()                           # [913] int64
        source_idx    = cumsum(mask_i, 0) - 1                    # [913] int64
        source_idx    = clamp(source_idx, 0, 902)                # [913] int64
        source_idx_2d = source_idx.unsqueeze(-1).expand(913,1280)# [913,1280] int64
        gathered      = torch.gather(source, 0, source_idx_2d)  # [913,1280] bf16
    """
    torch.manual_seed(seed)

    source = torch.randn(N, D, dtype=torch.bfloat16)

    mask_1d = torch.zeros(S, dtype=torch.bool)
    mask_1d[:N] = True
    mask_1d = mask_1d[torch.randperm(S)]

    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, N - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand(S, D)

    return source, source_idx_2d


def main():
    xr.set_device_type("TT")

    model = GatherDim0()
    model.eval()

    source, index = _build_inputs()

    codegen_py(model, source, index, export_path="gather_deepseek_ocr_codegen")


def test_gather_deepseek_ocr_codegen():
    """Test that codegen creates the expected output folder."""
    output_dir = Path("gather_deepseek_ocr_codegen")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert output_dir.exists(), (
            f"Expected output folder '{output_dir}' was not created"
        )
        assert output_dir.is_dir(), f"'{output_dir}' exists but is not a directory"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
