# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Replicate all DeepseekV2Model.forward ops (up to position_ids) inside
DeepseekOCRModel.forward — expected PASS.

Copies the exact sequence of operations from ``DeepseekV2Model.forward``
before and including ``position_ids`` creation (shape extraction,
``use_cache`` / ``past_key_values`` handling, ``torch.arange``) into
``DeepseekOCRModel.forward``, returning before ``super().forward()``.

- If this **passes** → the operations themselves are NOT the OOM trigger;
  entering ``DeepseekV2Model.forward`` (which exposes all decoder layer
  parameters to torch_dynamo's graph) is what causes OOM.
- If this **OOMs** → the operations are genuinely the trigger.

Compare:
- ``posids_05`` (position_ids only, no V2 ops) → PASS
- ``posids_04`` (entering V2 forward, returning after position_ids) → OOM
- **This test** (V2 ops replicated in OCR forward, no super()) → ???

See ``test_deepseek_ocr_posids_01_whole_model.py`` for suite overview.
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
def test_early_stop_before_super_with_v2_ops():
    """Vision+scatter+V2 ops (up to posids), no super() — isolates ops vs entering V2."""
    xr.set_device_type("TT")
    config = _make_config()
    model = DeepseekOCRForCausalLM(config)
    model.config.use_cache = False
    model.eval()
    model = model.to(DTYPE)
    run_op_test(
        WholeModelWrapper(model, early_stop="before_super_with_v2_ops"),
        list(_make_inputs()),
        framework=Framework.TORCH,
    )
