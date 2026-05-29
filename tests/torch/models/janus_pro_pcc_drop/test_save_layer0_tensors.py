# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""One-shot CPU capture of layer-0 layernorm I/O from ImageTokenStep decode (no TT)."""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from tests.runner.requirements import RequirementsManager
from tests.torch.models.janus_pro_pcc_drop.layer0_tensor_capture import (
    DEFAULT_LAYER0_TENSOR_DIR,
    save_layer0_input_layernorm_tensors_from_decode_step,
)

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)


def _capture_layer0_tensors(variant_name: str) -> None:
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()

        loader = ModelLoader(ModelVariant(variant_name))
        inputs = loader.load_inputs(dtype_override=torch.bfloat16, prefill=False)
        step = loader.load_model(dtype_override=torch.bfloat16)

        out_dir = os.environ.get("JANUS_LAYER0_TENSOR_DIR", str(DEFAULT_LAYER0_TENSOR_DIR))
        save_layer0_input_layernorm_tensors_from_decode_step(
            step,
            inputs,
            out_dir,
            variant_label=variant_name,
        )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
def test_save_layer0_input_layernorm_tensors_from_decode_step_pro_1b():
    """CPU-only: write layer-0 tensors (same loader path as ``test_image_token_decode_pro_1b``)."""
    _capture_layer0_tensors("Pro_1B")
