# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolate whether ``torch.arange`` INSIDE ``DeepseekV2Model.forward`` triggers OOM.

Pre-compute ``position_ids`` in ``DeepseekOCRModel.forward`` (via torch.arange),
then pass them to ``super().forward(position_ids=precomputed, early_stop="after_position_ids")``.

Inside ``DeepseekV2Model.forward``, the ``if position_ids is None:`` branch is
**skipped** — no ``torch.arange`` executes inside V2.  V2 simply proceeds past
the ``"before_position_ids"`` checkpoint and returns at ``"after_position_ids"``.

Comparison with previous tests:
- Test 07 (super_early_then_posids): super() returns at "before_position_ids",
  posids created AFTER in OCR                                     → PASS
- Test 09 (after_position_ids): super() creates posids inside V2,
  returns at "after_position_ids"                                  → OOM
- Test 10 (this): super() goes PAST "before_position_ids" but SKIPS
  the arange (posids pre-computed), returns at "after_position_ids" → ???

If PASS → torch.arange inside V2.forward is the specific trigger.
If OOM  → V2.forward proceeding past the "before_position_ids" checkpoint
          (even without torch.arange) is enough to change graph compilation.
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
CROP_W, CROP_H = 2, 3
N_CROPS = CROP_W * CROP_H
PATCH_HW, GLOBAL_HW = 640, 1024
N_IMAGE_TOKENS, N_TEXT_TOKENS = 903, 10
SEQ_LEN = N_IMAGE_TOKENS + N_TEXT_TOKENS  # 913
WEIGHTS_DIR = "DeepSeek_OCR_weights"


def _make_config() -> DeepseekOCRConfig:
    config = DeepseekOCRConfig.from_pretrained(WEIGHTS_DIR)
    config.use_cache = False
    config._attn_implementation = "eager"
    config.head_dim = config.hidden_size // config.num_attention_heads
    return config


def _make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 129280, (1, SEQ_LEN), generator=g)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    images_seq_mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    images_seq_mask[0, torch.randperm(SEQ_LEN, generator=g)[:N_IMAGE_TOKENS]] = True
    images_spatial_crop = torch.tensor([[CROP_W, CROP_H]], dtype=torch.long)
    patches = torch.randn(N_CROPS, 3, PATCH_HW, PATCH_HW, dtype=DTYPE, generator=g)
    image_ori = torch.randn(1, 3, GLOBAL_HW, GLOBAL_HW, dtype=DTYPE, generator=g)
    return input_ids, attention_mask, images_seq_mask, images_spatial_crop, patches, image_ori


class WholeModelWrapper(nn.Module):
    def __init__(self, model: DeepseekOCRForCausalLM, early_stop: str = ""):
        super().__init__()
        self.causal_lm = model
        self.early_stop = early_stop

    def forward(self, input_ids, attention_mask, images_seq_mask,
                images_spatial_crop, patches, image_ori):
        result = self.causal_lm(
            input_ids=input_ids, attention_mask=attention_mask,
            images=[(patches, image_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=False, early_stop=self.early_stop,
        )
        return result[0]


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_super_with_precomputed_posids():
    """Pre-compute posids in OCR, pass to super() — V2 skips torch.arange."""
    xr.set_device_type("TT")
    config = _make_config()
    model = DeepseekOCRForCausalLM(config)
    model.config.use_cache = False
    model.eval()
    model = model.to(DTYPE)
    run_op_test(
        WholeModelWrapper(model, early_stop="super_with_precomputed_posids"),
        list(_make_inputs()),
        framework=Framework.TORCH,
    )
