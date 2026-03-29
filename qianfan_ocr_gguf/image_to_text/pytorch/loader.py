# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Reza2kn Qianfan-OCR GGUF model loader implementation for image to text.
"""

from transformers import AutoModel, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qianfan-OCR GGUF model variants for image to text."""

    QIANFAN_OCR_GGUF = "qianfan_ocr_gguf"


class ModelLoader(ForgeModel):
    """Qianfan-OCR GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QIANFAN_OCR_GGUF: LLMModelConfig(
            pretrained_model_name="Reza2kn/Qianfan-OCR-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QIANFAN_OCR_GGUF

    GGUF_FILE = "Qianfan-OCR-q4_k_m.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qianfan-OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a tokenizer; use the base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "baidu/Qianfan-OCR", trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        prompt = "Parse this document to Markdown."

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        return inputs
