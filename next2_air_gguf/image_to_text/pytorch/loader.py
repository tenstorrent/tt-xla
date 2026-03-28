# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Next2-Air GGUF model loader implementation for image to text.
"""

from typing import Optional

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig

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
    """Available Next2-Air GGUF model variants."""

    NEXT2_AIR_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Next2-Air GGUF model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.NEXT2_AIR_Q4_K_M: ModelConfig(
            pretrained_model_name="thelamapi/next2-air-i1-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEXT2_AIR_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.NEXT2_AIR_Q4_K_M: "next2-air.i1-Q4_K_M.gguf",
    }

    _PROCESSOR_NAME = "thelamapi/next2-air"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Next2-Air GGUF model loader."""
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
            model="Next2-Air GGUF",
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
        """Load and return the Next2-Air GGUF model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Next2-Air GGUF."""
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
