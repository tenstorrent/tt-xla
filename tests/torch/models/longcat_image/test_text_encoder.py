# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image — Qwen2_5_VLForConditionalGeneration (text_encoder) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.longcat_image.pytorch import ModelLoader, ModelVariant


@pytest.mark.skip(
    reason="Out of Memory: ~7.7B Qwen2.5-VL text encoder (bf16) does not fit a "
    "single n150 — device DRAM fills up (1057376288 B allocated of 1070773184 B "
    "bank size) then a further 135790592 B buffer cannot be allocated (only "
    "13396896 B free). Needs multi-chip tensor-parallel (mesh [1,2]); skipped to "
    "avoid wasting compute on a known single-chip OOM — "
    "https://github.com/tenstorrent/tt-xla/issues/5168"
)
def test_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
