# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for image to text.
"""

from typing import Optional

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig

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


class ModelVariant(StrEnum):
    """Available Qwen 3.5 model variants for image to text."""

    QWEN_3_5_4B_BASE_GGUF = "4B_Base_GGUF"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_4B_BASE_GGUF: ModelConfig(
            pretrained_model_name="titan0115/Qwen3.5-4B-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_4B_BASE_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_4B_BASE_GGUF: "Qwen3.5-4B-Base-Q4_K_M.gguf",
    }

    _PROCESSOR_NAME = "Qwen/Qwen3.5-4B-Base"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Qwen 3.5 image to text model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_NAME)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 model instance for image to text."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Qwen 3.5 image to text."""
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
