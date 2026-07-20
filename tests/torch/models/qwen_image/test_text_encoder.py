# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — Qwen2.5-VL text encoder component test."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from third_party.tt_forge_models.qwen_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.qwen_image.pytorch.src.model_utils import SEED

from . import skip_on_wormhole


@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_text_encoder():
    skip_on_wormhole("text encoder")
    torch.manual_seed(SEED)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Qwen text encoders land ~0.98 in bf16 on TT (hidden_states[-1]); relax the
    # gate to 0.97 to accept the bf16 accumulation error (cf. Z-Image encoder).
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.97))

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )
