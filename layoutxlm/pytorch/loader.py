# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutXLM document understanding model loader implementation (PyTorch).
"""

import torch
from transformers import LayoutXLMModel, LayoutXLMTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available LayoutXLM model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """LayoutXLM document understanding model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="microsoft/layoutxlm-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LayoutXLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = LayoutXLMTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutXLMModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        words = ["Invoice", "Number:", "12345", "Date:", "2024-01-15"]
        boxes = [
            [100, 50, 200, 80],
            [210, 50, 330, 80],
            [340, 50, 420, 80],
            [100, 100, 180, 130],
            [190, 100, 340, 130],
        ]

        encoding = self.tokenizer(
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        bbox = encoding["bbox"]

        image = torch.zeros(1, 3, 224, 224)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "image": image,
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "last_hidden_state"):
            return fwd_output.last_hidden_state
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output
