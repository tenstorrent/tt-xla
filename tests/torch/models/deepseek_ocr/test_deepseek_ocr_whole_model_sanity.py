# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Whole-model sanity for ``DeepseekOCRForCausalLM`` — two ``run_op_test`` tests.

OOM was observed when ``DeepseekOCRModel.forward`` finishes the vision+scatter
path and hands off to ``DeepseekV2Model.forward``.  Individual sub-modules
(cumsum, scatter, vision stages) pass in isolation; the OOM only appears once
the full ``DeepseekV2Model`` graph is included.

Two tests use the **same** input configuration:

1. ``test_…_early_stop`` — early-returns from ``DeepseekOCRModel.forward``
   right before calling into ``DeepseekV2Model.forward`` (``early_stop=True``).
   Smallest graph that reproduces the OOM.
2. ``test_…_full_forward`` — runs the full decoder + ``lm_head``.

Input geometry (same as the whole-model / incremental-pipeline tests):

- ``patches``           : ``[6, 3, 640, 640]``   (``crop_shape = [2, 3]``)
- ``image_ori``         : ``[1, 3, 1024, 1024]``
- ``images_seq_mask``   : 903 ``True`` positions  (630 local + 272 global + 1 sep)
- ``seq_len``           : 1031  (``903 + 128``)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from utils import Category

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.modeling_deepseekocr import (
    DeepseekOCRConfig,
    DeepseekOCRForCausalLM,
)

DTYPE = torch.bfloat16

CROP_W = 2
CROP_H = 3
N_CROPS = CROP_W * CROP_H
PATCH_HW = 640
GLOBAL_HW = 1024

N_IMAGE_TOKENS = 903
SEQ_LEN = N_IMAGE_TOKENS + 128  # 1031


WEIGHTS_DIR = "DeepSeek_OCR_weights"


def _make_config() -> DeepseekOCRConfig:
    """Load config directly from ``DeepSeek_OCR_weights/config.json``."""
    config = DeepseekOCRConfig.from_pretrained(WEIGHTS_DIR)
    config.use_cache = False
    config._attn_implementation = "eager"
    config.head_dim = config.hidden_size // config.num_attention_heads
    return config


def _make_inputs(seed: int):
    """Synthetic inputs matching the whole-model forward signature."""
    g = torch.Generator().manual_seed(seed)

    input_ids = torch.randint(0, 129280, (1, SEQ_LEN), generator=g)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

    images_seq_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    idx = torch.randperm(SEQ_LEN, generator=g)[:N_IMAGE_TOKENS]
    images_seq_mask[0, idx] = True

    images_spatial_crop = torch.tensor([[CROP_W, CROP_H]], dtype=torch.long)

    patches = torch.randn(N_CROPS, 3, PATCH_HW, PATCH_HW, dtype=DTYPE, generator=g)
    image_ori = torch.randn(1, 3, GLOBAL_HW, GLOBAL_HW, dtype=DTYPE, generator=g)

    return input_ids, attention_mask, images_seq_mask, images_spatial_crop, patches, image_ori


class WholeModelWrapper(nn.Module):
    """Thin wrapper that flattens ``images`` from positional tensors for ``run_op_test``."""

    def __init__(self, model: DeepseekOCRForCausalLM, early_stop: bool = False):
        super().__init__()
        self.causal_lm = model
        self.early_stop = early_stop

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images_seq_mask: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        patches: torch.Tensor,
        image_ori: torch.Tensor,
    ) -> torch.Tensor:
        images = [(patches, image_ori)]
        result = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=False,
            early_stop=self.early_stop,
        )
        return result[0]


@pytest.fixture()
def causal_lm_model():
    """Fresh model per test — ``run_op_test`` mutates the model in-place via ``.to(device)``."""
    config = _make_config()
    model = DeepseekOCRForCausalLM(config)
    model.config.use_cache = False
    model.eval()
    model = model.to(DTYPE)
    return model


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_whole_model_early_stop(causal_lm_model):
    """Vision+scatter only — early-return before ``DeepseekV2Model.forward``."""
    xr.set_device_type("TT")
    inputs = _make_inputs(seed=42)
    run_op_test(
        WholeModelWrapper(causal_lm_model, early_stop=True),
        list(inputs),
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_whole_model_full_forward(causal_lm_model):
    """Full ``DeepseekOCRForCausalLM`` forward through decoder layers + lm_head (known OOM)."""
    xr.set_device_type("TT")
    inputs = _make_inputs(seed=42)
    run_op_test(
        WholeModelWrapper(causal_lm_model),
        list(inputs),
        framework=Framework.TORCH,
    )
