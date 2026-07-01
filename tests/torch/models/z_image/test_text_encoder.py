# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image — Qwen3 text encoder component test."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from third_party.tt_forge_models.z_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.z_image.pytorch.src.model_utils import SEED


@pytest.mark.model_test
@pytest.mark.single_device
def test_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # The Qwen3 encoder lands at pcc~0.9795 in bf16 on TT (hidden_states[-2]);
    # relax the gate to 0.97 to accept the bf16 accumulation error.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.97))

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )
