# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone DeepseekV2 sanity test to isolate OOM behavior.

Loads the full DeepseekOCR model, removes vision components (SAM, CLIP,
projector), and calls DeepseekV2Model.forward directly with the exact same
input shapes/dtypes the full model would pass after vision processing.

If this test PASSES but the full DeepseekOCR model OOMs, then the OOM is
caused by cumulative DRAM usage from prior model components (SAM + vision +
projector), not by DeepseekV2 alone.

Input shapes captured from the full model run log:
  - inputs_embeds: [1, 913, 1280] float32
  - attention_mask: [1, 913] long
  - position_ids: None (created internally as torch.arange(0, 913))
  - use_cache: False, return_dict: False
"""

import torch
from infra import Framework, run_op_test
from tests.runner.requirements import RequirementsManager
import inspect
from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader
from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.src.modeling_deepseekv2 import DeepseekV2Model


class DeepseekV2Wrapper(torch.nn.Module):
    """Loads the full OCR model, strips vision components, and exposes only
    the DeepseekV2 forward path (embed_tokens, decoder layers, norm, lm_head).
    """

    def __init__(self):
        super().__init__()
        loader = ModelLoader()
        loader_path = inspect.getsourcefile(ModelLoader)
        with RequirementsManager.for_loader(loader_path):
            ocr_model = loader.load_model(dtype_override=torch.bfloat16)
        ocr_model.config.return_dict = False
        ocr_model.config.use_cache = False
        ocr_model.eval()

        # ocr_model.model is DeepseekOCRModel which extends DeepseekV2Model.
        # Remove vision components so only V2 weights hit TT DRAM.
        del ocr_model.model.sam_model
        del ocr_model.model.vision_model
        del ocr_model.model.projector
        del ocr_model.model.image_newline
        del ocr_model.model.view_seperator

        self.ocr_model = ocr_model

    def forward(self, inputs_embeds):
        # Call DeepseekV2Model.forward directly (parent class),
        # bypassing DeepseekOCRModel.forward which runs vision processing.
        outputs = DeepseekV2Model.forward(
            self.ocr_model.model,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        hidden_states = outputs[0]
        logits = self.ocr_model.lm_head(hidden_states)
        return logits.float()
# tt_forge_models.deepseek.deepseek_ocr.pytorch.src.modeling_deepseekocr:torch_dynamo_resume_in_forward_at_67:214 - DeepseekV2Model.forward inputs => input_ids=None, attention_mask: shape=None, dtype=None, inputs_embeds: shape=torch.Size([1, 913, 1280]), dtype=torch.bfloat16, position_ids: shape=None, dtype=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=False

def test_deepseekv2_sanity():
    """Test DeepseekV2 forward alone with the same inputs that the full OCR
    model would pass after vision processing.

    If this passes on TT device, the full-model OOM is caused by cumulative
    DRAM pressure from SAM + vision + projector, not by DeepseekV2 itself.
    """
    wrapper = DeepseekV2Wrapper()

    inputs_embeds = torch.randn(1, 913, 1280, dtype=torch.bfloat16)

    run_op_test(
        wrapper,
        inputs=[inputs_embeds,],
        framework=Framework.TORCH,
    )
