# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OOM bisection for LLaVA 1.5-7B with NEW optimized masked_scatter decomposition.

Incrementally adds decoder layers to pinpoint where DRAM OOM occurs.
The NEW decomposition does cumsum on S = 596 elements (row-level),
compared to the OLD decomposition which does cumsum on S*D = 2,441,216 elements.

The wrapper calls the REAL LlavaForConditionalGeneration.forward — matching
the whole-model test runner (test_all_models_torch) exactly:
  - Model and inputs loaded in bfloat16 (same as TorchDynamicLoader)
  - Inputs passed as kwargs (same as TorchModelTester)
  - Only two changes: monkey-patched masked_scatter + num_hidden_layers patch

Tests:
  test_pre_decoder       — 0 decoder layers
  test_decoder_1_layer   — 1 decoder layer
  test_decoder_4_layers  — 4 decoder layers
  test_decoder_16_layers — 16 decoder layers
  test_decoder_32_layers — 32 decoder layers (full model) — expected to OOM

Usage:
    pytest tests/torch/models/llava_oom_bisect/test_llava_new_decomp.py -svv
"""

import pytest
import torch
import torch.nn as nn
from infra.utilities.types import Framework
from tests.infra.testers.single_chip.op.op_tester import run_op_test

from python_package.tt_torch.backend.decompositions import masked_scatter_optimized
from third_party.tt_forge_models.llava.pytorch.loader import ModelLoader, ModelVariant


class LlavaWrapper(nn.Module):
    """Thin wrapper around the real LlavaForConditionalGeneration.

    Calls full_model.forward() directly so every tensor creation, module
    call, decorator, DynamicCache, causal mask, etc. is identical to the
    whole-model run.  masked_scatter is monkey-patched to use the chosen
    decomposition so Dynamo traces that instead of the native op.
    """

    def __init__(self, full_model):
        super().__init__()
        self.full_model = full_model

    def forward(self, input_ids, pixel_values, attention_mask):
        _orig = torch.Tensor.masked_scatter
        torch.Tensor.masked_scatter = lambda self, mask, source: masked_scatter_optimized(
            self, mask, source
        )
        try:
            outputs = self.full_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
        finally:
            torch.Tensor.masked_scatter = _orig
        return outputs.logits


@pytest.fixture(scope="module")
def model_and_inputs():
    loader = ModelLoader(variant=ModelVariant.LLAVA_1_5_7B)
    full_model = loader.load_model(dtype_override=torch.bfloat16)
    full_model.eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    return full_model, inputs


def _run_test(model, inputs):
    input_list = [inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]]
    run_op_test(model, input_list, framework=Framework.TORCH)


def _run_n_layers(model_and_inputs, num_layers):
    full_model, inputs = model_and_inputs
    llm = full_model.model.language_model
    orig = llm.config.num_hidden_layers
    llm.config.num_hidden_layers = num_layers
    try:
        wrapper = LlavaWrapper(full_model)
        wrapper.eval()
        _run_test(wrapper, inputs)
    finally:
        llm.config.num_hidden_layers = orig


def test_pre_decoder(model_and_inputs):
    _run_n_layers(model_and_inputs, 0)


def test_decoder_1_layer(model_and_inputs):
    _run_n_layers(model_and_inputs, 1)


def test_decoder_4_layers(model_and_inputs):
    _run_n_layers(model_and_inputs, 4)


def test_decoder_16_layers(model_and_inputs):
    _run_n_layers(model_and_inputs, 16)


def test_decoder_32_layers(model_and_inputs):
    _run_n_layers(model_and_inputs, 32)
