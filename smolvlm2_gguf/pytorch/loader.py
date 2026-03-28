# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLM2 GGUF model loader implementation for multimodal conditional generation.

Loads the GGUF-quantized SmolVLM2 vision-language model from
ggml-org/SmolVLM2-500M-Video-Instruct-GGUF using transformers' GGUF support.

Available variants:
- SMOLVLM2_500M_Q8_0: Q8_0 quantized (437 MB)
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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
from ...tools.utils import get_file

GGUF_REPO = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

_GGUF_FILES = {
    "Q8_0": "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
}


class ModelVariant(StrEnum):
    """Available SmolVLM2 GGUF model variants."""

    SMOLVLM2_500M_Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """SmolVLM2 GGUF model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.SMOLVLM2_500M_Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLVLM2_500M_Q8_0

    sample_text = "Can you describe this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SmolVLM2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_filename(self) -> str:
        return _GGUF_FILES[self._variant.value]

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SmolVLM2 GGUF model instance."""
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._get_gguf_filename()

        model = AutoModelForImageTextToText.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for SmolVLM2."""
        if self.processor is None:
            self._load_processor()

        image_path = get_file(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
        )
        image = Image.open(str(image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            images=[image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
