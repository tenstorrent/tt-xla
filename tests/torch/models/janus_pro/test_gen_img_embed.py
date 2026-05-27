# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro — GenImgEmbed (gen_embed + gen_aligner)."""

import inspect

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from tests.runner.requirements import RequirementsManager

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_gen_img_embed_pro_1b():
    _run("GenImgEmbed")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_gen_img_embed_pro_7b():
    _run("GenImgEmbed_7B")


def _run(variant_name: str):
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
            ModelLoader,
            ModelVariant,
        )

        variant = ModelVariant(variant_name)
        xr.set_device_type("TT")
        torch.manual_seed(42)

        loader = ModelLoader(variant)
        model = loader.load_model(dtype_override=torch.bfloat16)
        inputs = loader.load_inputs(dtype_override=torch.bfloat16)

        run_graph_test(
            model,
            [inputs["image_ids"]],
            framework=Framework.TORCH,
        )
