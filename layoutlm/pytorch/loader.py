# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutLM document understanding model loader implementation (PyTorch).
"""

import torch
from transformers import LayoutLMModel, LayoutLMTokenizer
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
    """Available LayoutLM model variants."""

    BASE_CASED = "Base Cased"


class ModelLoader(ForgeModel):
    """LayoutLM document understanding model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE_CASED: ModelConfig(
            pretrained_model_name="microsoft/layoutlm-base-cased",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_CASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LayoutLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = LayoutLMTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutLMModel.from_pretrained(pretrained_model_name, **model_kwargs)
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

        max_length = 512

        # LayoutLM v1 tokenizer does not accept boxes; tokenize words and
        # manually align bounding boxes to the resulting word-piece tokens.
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )

        # Build bbox tensor aligned with tokenized output.
        # [CLS] and [SEP] get [0,0,0,0]; each word-piece inherits its word's box.
        word_ids = encoding.word_ids(batch_index=0)
        bbox = []
        for wid in word_ids:
            if wid is None:
                bbox.append([0, 0, 0, 0])
            else:
                bbox.append(boxes[wid])
        bbox = torch.tensor([bbox], dtype=torch.long)

        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": bbox,
            "token_type_ids": encoding["token_type_ids"],
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
