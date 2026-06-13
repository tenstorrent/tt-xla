# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — QwenImageTransformer2DModel (transformer) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.qwen_image.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="Out of Memory: Not enough space to allocate 150994944 B DRAM buffer across 12 banks, where each bank needs to store 12582912 B, but bank size is 1071821792 B (allocated: 1056285728 B, free: 15536064 B) — 20B QwenImageTransformer2DModel does not fit a single device; needs multi-chip tensor-parallel — https://github.com/tenstorrent/tt-xla/issues/TBD-TRANSFORMER"
)
def test_transformer():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.float32)
    inputs = loader.load_inputs(dtype_override=torch.float32)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
