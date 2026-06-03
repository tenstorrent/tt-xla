# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro — ImageTokenStep (language_model.model + gen_head).

Pro-7B prefill/decode skips on wormhole (n150) at runtime due to DRAM OOM.
Pro-1B and gen components run on both n150 and p150 via standard component CI
(``model_test and single_device``).
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from utils import BringupStatus, Category

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from tests.runner.requirements import RequirementsManager

from . import skip_pro_7b_image_token_on_wormhole
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)

MODEL_INFO_PRO_1B = ModelLoader.get_model_info(ModelVariant.PRO_1B)
MODEL_INFO_PRO_7B = ModelLoader.get_model_info(ModelVariant.PRO_7B)


class ImageTokenPrefillWrapper(nn.Module):
    """Return logits only (drop KV cache) for run_graph_test."""

    def __init__(self, step: nn.Module):
        super().__init__()
        self.step = step

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        logits, _ = self.step(inputs_embeds, None)
        return logits


class ImageTokenDecodeWrapper(nn.Module):
    """Decode step: single-token embeds + prefill KV for run_graph_test."""

    def __init__(self, step: nn.Module, kv_template: Any):
        super().__init__()
        self.step = step
        self._kv_template = kv_template

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # run_graph_test runs CPU then Forge on one module; attention mutates KV
        # in-place. Clone prefill KV each forward so both sides start clean (#4968).
        past_key_values = copy.deepcopy(self._kv_template)
        logits, _ = self.step(inputs_embeds, past_key_values)
        return logits


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_image_token_prefill_pro_1b():
    _run_prefill("Pro_1B")


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO_PRO_1B,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_image_token_decode_pro_1b():
    _run_decode("Pro_1B")


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_image_token_prefill_pro_7b():
    skip_pro_7b_image_token_on_wormhole()
    _run_prefill("Pro_7B")


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO_PRO_7B,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_image_token_decode_pro_7b():
    skip_pro_7b_image_token_on_wormhole()
    _run_decode("Pro_7B")


def _run_prefill(variant_name: str):
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        variant = ModelVariant(variant_name)
        torch.manual_seed(42)

        loader = ModelLoader(variant)
        # Build forge inputs on CPU before TT torch overrides are active.
        inputs = loader.load_inputs(dtype_override=torch.bfloat16, prefill=True)
        step = loader.load_model(dtype_override=torch.bfloat16)
        xr.set_device_type("TT")

        run_graph_test(
            ImageTokenPrefillWrapper(step),
            [inputs["inputs_embeds"]],
            framework=Framework.TORCH,
        )


def _run_decode(variant_name: str):
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        variant = ModelVariant(variant_name)
        torch.manual_seed(42)

        # Decode inputs run a CPU prefill through the full MMGPT; avoid a cached
        # copy that prior TT compile may have left on XLA.
        model_utils._mmgpt_cache.clear()
        loader = ModelLoader(variant)
        inputs = loader.load_inputs(dtype_override=torch.bfloat16, prefill=False)
        step = loader.load_model(dtype_override=torch.bfloat16)
        xr.set_device_type("TT")

        run_graph_test(
            ImageTokenDecodeWrapper(step, inputs["past_key_values"]),
            [inputs["inputs_embeds"]],
            framework=Framework.TORCH,
        )
