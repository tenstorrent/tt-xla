# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro — GenVisionDecode (gen_vision_model.decode_code)."""

import inspect

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_gen_vision_decode_pro_1b():
    _run(ModelVariant.GEN_VISION_DECODE)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_gen_vision_decode_pro_7b():
    _run(ModelVariant.GEN_VISION_DECODE_7B)


def _run(variant: ModelVariant):
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        xr.set_device_type("TT")
        torch.manual_seed(42)

        loader = ModelLoader(variant)
        model = loader.load_model(dtype_override=torch.bfloat16)
        inputs = loader.load_inputs(dtype_override=torch.bfloat16)

        run_graph_test(
            model,
            [inputs["generated_tokens"]],
            framework=Framework.TORCH,
        )
