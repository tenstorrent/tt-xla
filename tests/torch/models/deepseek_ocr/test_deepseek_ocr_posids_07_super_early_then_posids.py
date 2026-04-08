# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Enter ``super().forward()`` (exposing decoder params), return immediately,
then create ``position_ids`` back in ``DeepseekOCRModel.forward``.

This is the **smoking gun** test.  It combines:

1. Calling ``super().forward()`` with ``early_stop="before_position_ids"``
   → torch_dynamo traces into ``DeepseekV2Model.forward``, exposing all
   decoder layer parameters to the XLA graph, then returns immediately.
2. Creating ``position_ids = torch.arange(...)`` **after** ``super()`` returns,
   back inside ``DeepseekOCRModel.forward``.

Compare:

- ``posids_03`` (enters super, no posids)         → PASS (decoder params exposed, small graph)
- ``posids_06`` (no super, with V2 ops + posids)   → PASS (no decoder params exposed)
- ``posids_04`` (enters super, posids inside V2)   → OOM
- **This test** (enters super + posids outside V2) → **OOM if decoder param exposure is the trigger**

If this OOMs: entering ``super().forward()`` makes torch_dynamo include
decoder layer parameters in the compiled graph, and adding ``position_ids``
to the same graph pushes total DRAM allocation over the limit.

If this passes: something specific to creating ``position_ids`` *inside*
``DeepseekV2Model.forward`` (vs outside) triggers the OOM.
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
def test_super_early_then_posids():
    """Enter super().forward() (decoder params exposed) + posids after return."""
    xr.set_device_type("TT")
    config = _make_config()
    model = DeepseekOCRForCausalLM(config)
    model.config.use_cache = False
    model.eval()
    model = model.to(DTYPE)
    run_op_test(
        WholeModelWrapper(model, early_stop="super_early_then_posids"),
        list(_make_inputs()),
        framework=Framework.TORCH,
    )
