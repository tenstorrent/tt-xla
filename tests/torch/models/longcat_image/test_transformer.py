# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image — LongCatImageTransformer2DModel (transformer) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.longcat_image.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="Out of Memory: ~6B LongCatImageTransformer2DModel (bf16) does not fit a "
    "single n150 — needs a 6870269952 B DRAM buffer (each of 12 banks would store "
    "572522496 B, but bank size is only 1070773184 B). Needs multi-chip "
    "tensor-parallel (mesh [1,2]); skipped to avoid wasting compute on a known "
    "single-chip OOM — https://github.com/tenstorrent/tt-xla/issues/5169"
)
def test_transformer():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
