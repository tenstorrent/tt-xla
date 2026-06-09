# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test for HiDream's interleaved-pair RoPE through the tt-xla pipeline.

Mimics HiDream's `apply_rope` (from HuggingFace diffusers) using the EXACT
shapes and dtypes captured from the actual model run
(jun7_hidream_transformer_4_chips_rope_info.log):

    query.shape           = (1, 4480, 20, 128)
    key.shape             = (1, 4480, 20, 128)
    image_rotary_emb.shape = (1, 4480, 1, 64, 2, 2)   # heads=1, broadcasts
    all dtypes            = torch.float32 (in model);
                            also tested with bf16 here

Purpose:
  - Reproduce the interleaved-pair RoPE OOM in isolation (small, fast,
    independent of the full HiDream transformer model)
  - Gating signal for the TTIR fusing matcher work — fails today with the
    same 23.5 GB allocation we see in the full model
  - Once the matcher fires correctly, this test should pass AND the
    lowered IR should contain `ttir.rotary_embedding` / `ttnn.rotary_embedding`
"""

import pytest
import torch
from infra import Framework, run_graph_test


def apply_rope(xq, xk, freqs_cis):
    """HiDream's interleaved-pair RoPE — verbatim from the diffusers source.

    Reference:
    https://github.com/huggingface/diffusers/blob/ad3a3afc3a4d3068bbb12f58129c855087ffc6d6/src/diffusers/models/transformers/transformer_hidream_image.py#L131-L137
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return (
        xq_out.reshape(*xq.shape).type_as(xq),
        xk_out.reshape(*xk.shape).type_as(xk),
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hidream_rope(dtype):
    torch.manual_seed(0)

    # Actual HiDream model shapes
    bsz, seq, heads, D = 1, 4480, 20, 128

    query = torch.randn(bsz, seq, heads, D, dtype=dtype)
    key = torch.randn(bsz, seq, heads, D, dtype=dtype)

    # freqs_cis: per pair, 2x2 = [[cos, -sin], [sin, cos]]; heads dim = 1 (broadcasts).
    angles = torch.randn(bsz, seq, 1, D // 2)
    c, s = torch.cos(angles), torch.sin(angles)
    freqs_cis = torch.stack([c, -s, s, c], dim=-1).reshape(
        bsz, seq, 1, D // 2, 2, 2
    )

    run_graph_test(
        apply_rope, [query, key, freqs_cis], framework=Framework.TORCH
    )
