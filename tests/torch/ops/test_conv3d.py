# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category

# Qwen2.5-VL-3B vision patch-embed config (from its vision_config).
IN_CHANNELS = 3
EMBED_DIM = 1280
TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 14

# Number of patches the demo image produces: image_grid_thw=[[1, 38, 58]]
# -> 1 * 38 * 58 = 2204 patches, each a flattened [C * T * P * P] = 1176 vector.
NUM_PATCHES = 1 * 38 * 58
PATCH_FEATURES = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE


class Qwen25VLPatchEmbed(torch.nn.Module):
    """Mirror of Qwen2_5_VisionPatchEmbed.forward (modeling_qwen2_5_vl.py:108-114).

    Conv3d with stride == kernel == [T, P, P] is a non-overlapping patchify; the
    input is pre-reshaped to one patch per row and the output flattened to
    [num_patches, embed_dim].
    """

    def __init__(self):
        super().__init__()
        kernel = [TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE]
        self.proj = torch.nn.Conv3d(
            IN_CHANNELS, EMBED_DIM, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, x):
        x = x.view(-1, IN_CHANNELS, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
        return self.proj(x).view(-1, EMBED_DIM)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.Conv3d",
)
def test_conv3d_qwen_vl_patch_embed():
    """Conv3d patch-embed of Qwen2.5-VL-3B on the real demo-image patch count.

    Guards the patchify-conv3d device bug (tt-xla #1662): this is the vision
    tower's first op and a non-overlapping conv (stride == kernel) that ttnn.conv3d
    mis-lowers (device PCC ~0). tt-mlir rewrites such convs to an equivalent
    matmul, so this should pass. Run in f32 so a regression is unambiguously a
    lowering issue, not bf16 rounding; default PCC>=0.99 comparator.
    """
    x = torch.randn(NUM_PATCHES, PATCH_FEATURES, dtype=torch.float32)
    run_op_test(
        Qwen25VLPatchEmbed(),
        [x],
        framework=Framework.TORCH,
    )
