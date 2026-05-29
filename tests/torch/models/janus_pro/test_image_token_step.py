# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro — ImageTokenStep (language_model.model + gen_head).

Arch selection uses pytest markers (same as other tt-xla model tests):
  - Pro-1B: ``n150`` + ``p150`` (single_device)
  - Pro-7B: ``p150`` only (DRAM OOM on n150)

On wormhole (n150), run with a marker so Pro-7B is not collected::

  pytest -m "n150 and single_device" tests/torch/models/janus_pro/

On blackhole (p150)::

  pytest -m "p150 and single_device" tests/torch/models/janus_pro/
"""

import inspect
import os

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from tests.runner.requirements import RequirementsManager
from utils import BringupStatus, Category, incorrect_result

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
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
    """Decode step: single-token embeds; KV cache set before run_graph_test."""

    def __init__(self, step: nn.Module):
        super().__init__()
        self.step = step
        self.past_key_values = None

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        logits, _ = self.step(inputs_embeds, self.past_key_values)
        return logits


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_prefill_pro_1b():
    _run_prefill("Pro_1B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO_PRO_1B,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
# @pytest.mark.xfail(
#     reason=incorrect_result(
#         "ImageTokenStep decode (Pro-1B): observed pcc=0.94 vs required 0.99"
#     ),
#     strict=False,
# )
def test_image_token_decode_pro_1b():
    _run_decode("Pro_1B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
def test_image_token_prefill_pro_7b():
    _run_prefill("Pro_7B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO_PRO_7B,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
# @pytest.mark.xfail(
#     reason=incorrect_result(
#         "ImageTokenStep decode (Pro-7B): observed pcc=0.82 vs required 0.99"
#     ),
#     strict=False,
# )
def test_image_token_decode_pro_7b():
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

        if os.environ.get("JANUS_SAVE_LAYER0_TENSORS") == "1":
            from tests.torch.models.janus_pro_pcc_drop.layer0_tensor_capture import (
                DEFAULT_LAYER0_TENSOR_DIR,
                save_layer0_input_layernorm_tensors_from_decode_step,
            )

            out_dir = os.environ.get(
                "JANUS_LAYER0_TENSOR_DIR", str(DEFAULT_LAYER0_TENSOR_DIR)
            )
            save_layer0_input_layernorm_tensors_from_decode_step(
                step,
                inputs,
                out_dir,
                variant_label=variant_name,
            )
            pytest.exit(
                f"Saved layer-0 layernorm tensors under {out_dir}", returncode=0
            )

        xr.set_device_type("TT")

        wrapper = ImageTokenDecodeWrapper(step)
        wrapper.past_key_values = inputs["past_key_values"]
        run_graph_test(
            wrapper,
            [inputs["inputs_embeds"]],
            framework=Framework.TORCH,
        )
