# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2.5 ONNX model loader implementation for causal language modeling.

Loads LiquidAI's LFM2.5-1.2B-Thinking model in ONNX format, a 1.2B-parameter
reasoning model that generates step-by-step thinking before producing answers.
"""

import onnx
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_REPO_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX"


class ModelVariant(StrEnum):
    """Available LFM2.5 ONNX model variants for causal language modeling."""

    LFM2_5_1_2B_THINKING_Q4 = "1_2B_Thinking_Q4"
    LFM2_5_1_2B_THINKING_FP16 = "1_2B_Thinking_FP16"


class ModelLoader(ForgeModel):
    """LFM2.5 ONNX model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_5_1_2B_THINKING_Q4: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
        ModelVariant.LFM2_5_1_2B_THINKING_FP16: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
    }

    _ONNX_FILES = {
        ModelVariant.LFM2_5_1_2B_THINKING_Q4: "onnx/model_q4.onnx",
        ModelVariant.LFM2_5_1_2B_THINKING_FP16: "onnx/model_fp16.onnx",
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_5_1_2B_THINKING_Q4

    sample_text = "What is 25 times 37?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LFM2.5 ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the LFM2.5 model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            _REPO_ID,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, **kwargs):
        """Load and return the LFM2.5 ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        onnx_file = self._ONNX_FILES[self._variant]
        onnx_path = hf_hub_download(repo_id=_REPO_ID, filename=onnx_file)
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, batch_size=1, **kwargs):
        """Load and return sample inputs for the LFM2.5 model.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        messages = [
            {"role": "user", "content": self.sample_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
