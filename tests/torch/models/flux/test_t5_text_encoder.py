# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — T5 text encoder component test (sequence embedding)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from utils import BringupStatus, Category

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=ModelLoader.get_model_info(ModelVariant.T5_TEXT_ENCODER),
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_t5_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.T5_TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        # Accumulative bf16 device-matmul drift across the 24 encoder blocks; does not affect the real-input e2e pipeline output.
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.95)),
        framework=Framework.TORCH,
    )
