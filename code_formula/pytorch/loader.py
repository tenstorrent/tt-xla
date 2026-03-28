# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CodeFormula model loader implementation for code and formula OCR from images.
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
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
    """Available CodeFormula model variants."""

    CODE_FORMULA = "CodeFormula"


class ModelLoader(ForgeModel):
    """CodeFormula model loader for code and formula OCR from images."""

    _VARIANTS = {
        ModelVariant.CODE_FORMULA: ModelConfig(
            pretrained_model_name="docling-project/CodeFormula",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CODE_FORMULA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CodeFormula",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        image = Image.new("RGB", (1024, 1024), color=(255, 255, 255))

        inputs = self.tokenizer(image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
