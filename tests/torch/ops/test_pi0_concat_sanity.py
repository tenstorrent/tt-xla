# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity `run_op_test` cases for PI0 / multimodal-style `torch.cat` patterns.

The PI0 inference failure (see pi_0 logs) surfaces in TT-Metal as::

    ShapeBase[] index out of range. 2 not in [-4, 2)
    ttnn::concat(...)

That means a concat was lowered with **axis 2** while at least one operand was
**rank-2** (only axes -2,-1,0,1). These tests do **not** import lerobot; they
isolate concat shapes/dims that often appear around vision + language fusion.

Narrowing the real submodule:
- PI0 forward path: ``PI0Policy.model.sample_actions`` (lerobot ``modeling_pi0``).
- Forge wrapper: ``third_party/tt_forge_models/pi_0/pytorch/src/model.py`` calls
  ``self.model.sample_actions(...)``.
- Next step: run the model test with ``--dump-irs`` and/or
  ``test_all_models_op_by_op`` for this node id, then grep MLIR for
  ``stablehlo.concatenate`` / ``ttnn.concat`` to map to a subgraph.

Usage::

    pytest -svv tests/torch/ops/test_pi0_concat_sanity.py -k concat
"""

from __future__ import annotations

import pytest
import torch
from infra import ComparisonConfig, Framework, run_op_test
from utils import Category


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "name,batch,seq,h_a,h_b,dim",
    [
        ("cat_dim1_merge_seq", 1, 32, 64, 64, 1),
        ("cat_dim2_merge_hidden", 1, 16, 64, 48, 2),
        ("cat_dim0_merge_batch", 2, 32, 64, 64, 0),
    ],
)
def test_pi0_style_cat_rank3(request, name, batch, seq, h_a, h_b, dim):
    """Concat rank-3 tensors on dim 0/1/2 with valid PyTorch shapes."""

    class Cat3(torch.nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.d = d

        def forward(self, a, b):
            return torch.cat([a, b], dim=self.d)

    if dim == 0:
        a = torch.randn(batch, seq, h_a, dtype=torch.bfloat16)
        b = torch.randn(batch, seq, h_b, dtype=torch.bfloat16)
    elif dim == 1:
        a = torch.randn(batch, seq, h_a, dtype=torch.bfloat16)
        b = torch.randn(batch, seq + 8, h_a, dtype=torch.bfloat16)
    else:
        a = torch.randn(batch, seq, h_a, dtype=torch.bfloat16)
        b = torch.randn(batch, seq, h_b, dtype=torch.bfloat16)

    run_op_test(
        Cat3(dim),
        [a, b],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "name,h1,w1,h2,w2,dim",
    [
        ("two_views_along_channels", 224, 224, 224, 224, 1),
        ("two_views_along_batch", 96, 96, 96, 96, 0),
    ],
)
def test_pi0_style_cat_rank4_nchw(request, name, h1, w1, h2, w2, dim):
    """Typical dual-camera NCHW stacks: cat along C (1) or B (0)."""

    class Cat4(torch.nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.d = d

        def forward(self, a, b):
            return torch.cat([a, b], dim=self.d)

    bsz = 1
    c = 3
    a = torch.randn(bsz, c, h1, w1, dtype=torch.bfloat16)
    b = torch.randn(bsz, c, h2, w2, dtype=torch.bfloat16)
    run_op_test(
        Cat4(dim),
        [a, b],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_style_cat_rank2_features(request):
    """Rank-2 feature concat along dim=1 (projection / MLP heads)."""

    class Cat2(torch.nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=1)

    a = torch.randn(1, 128, dtype=torch.bfloat16)
    b = torch.randn(1, 128, dtype=torch.bfloat16)
    run_op_test(
        Cat2(),
        [a, b],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_style_cat_rank2_then_unsqueeze_chain(request):
    """
    Chain: concat rank-2 on dim=1, then unsqueeze to rank-3.
    Catches lowering that keeps concat axis as "2" after a reshape bug.
    """

    class Cat2Unsqueeze(torch.nn.Module):
        def forward(self, a, b):
            x = torch.cat([a, b], dim=1)
            return x.unsqueeze(-1)

    a = torch.randn(2, 32, dtype=torch.bfloat16)
    b = torch.randn(2, 32, dtype=torch.bfloat16)
    run_op_test(
        Cat2Unsqueeze(),
        [a, b],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
        request=request,
    )
