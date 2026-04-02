# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Op-level sanity test for stablehlo.scatter (torch.index_put with boolean mask).

Reproduces the compilation failure observed in the SmolVLA model run:
  loc("scatter.42"): error: failed to legalize operation 'stablehlo.scatter'

Root cause:
  SmolVLMVisionEmbeddings.forward()
  (transformers/models/smolvlm/modeling_smolvlm.py:160) builds position_ids
  for variable-resolution patches using a boolean-indexed assignment:

      position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

  PyTorch lowers this to aten::index_put_ → stablehlo.scatter, which the TT
  backend currently fails to legalize.

SmolVLM2-500M-Video-Instruct vision config (used by smolvla_base):
  image_size=512, patch_size=16, num_patches_per_side=32, max_patches=1024
  position_ids dtype: int64, mask dtype: bool, values dtype: int64
"""

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


class ScatterBoolMask(torch.nn.Module):
    """Applies torch.index_put with a boolean mask: target[mask] = values.

    This is the exact pattern from SmolVLMVisionEmbeddings.forward():
        position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
    which lowers to stablehlo.scatter.
    """

    def forward(
        self,
        target: torch.Tensor,
        mask: torch.BoolTensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return torch.index_put(target, (mask,), values)


def test_scatter_bool_mask_smolvlm_full():
    """Replicates stablehlo.scatter failure from SmolVLA/SmolVLM2-500M run.

    Full-image case: all 1024 patch positions written (no padding).
    Dims: image_size=512, patch_size=16 → num_patches_per_side=32,
          position_ids shape=(1024,), dtype=int64.

    Failure mode: stablehlo.scatter fails to legalize on the TT backend.
      loc("scatter.22"): error: failed to legalize operation 'stablehlo.scatter'
      RuntimeError: Error code: 13
    The mask is all-True so the scatter is statically-shaped (always 1024
    writes), but TT has no lowering for scatter regardless.
    """
    num_patches = 1024  # 32 * 32

    target = torch.zeros(num_patches, dtype=torch.int64)
    mask = torch.ones(num_patches, dtype=torch.bool)
    values = torch.arange(num_patches, dtype=torch.int64)

    run_op_test(ScatterBoolMask(), [target, mask, values], framework=Framework.TORCH)
