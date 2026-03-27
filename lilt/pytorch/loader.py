# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LiLT (Language-Independent Layout Transformer) model loader implementation.
"""

import torch
from transformers import AutoTokenizer, LiltModel
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
    """Available LiLT model variants."""

    ROBERTA_EN_BASE = "RoBERTa EN Base"


class ModelLoader(ForgeModel):
    """LiLT model loader implementation for document understanding."""

    _VARIANTS = {
        ModelVariant.ROBERTA_EN_BASE: ModelConfig(
            pretrained_model_name="SCUT-DLVCLab/lilt-roberta-en-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBERTA_EN_BASE

    # Sample document words and their bounding boxes (normalized 0-1000)
    words = ["Invoice", "Number:", "12345", "Date:", "2024-01-15"]
    boxes = [
        [100, 50, 200, 80],
        [210, 50, 330, 80],
        [340, 50, 420, 80],
        [100, 100, 180, 130],
        [190, 100, 340, 130],
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LiLT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
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

        model = LiltModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        encoding = self.tokenizer(
            self.words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
        )

        # Build bbox tensor aligned to tokenizer output (including special tokens)
        word_ids = encoding.word_ids()
        bbox = []
        for wid in word_ids:
            if wid is None:
                bbox.append([0, 0, 0, 0])
            else:
                bbox.append(self.boxes[wid])

        encoding["bbox"] = torch.tensor([bbox])

        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
