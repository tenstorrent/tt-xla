# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek VL2 model loader implementation for multimodal vision-language tasks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Optional

from ....tools.utils import get_file
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
    """Available DeepSeek VL2 model variants."""

    DEEPSEEK_VL2_TINY = "Tiny"


class ModelLoader(ForgeModel):
    """DeepSeek VL2 model loader implementation for multimodal vision-language tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_VL2_TINY: ModelConfig(
            pretrained_model_name="Isotr0py/deepseek-vl2-tiny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_VL2_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepSeek VL2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        conversation = [
            {
                "role": "user",
                "content": "<image>\nDescribe this image.",
            },
            {"role": "assistant", "content": ""},
        ]

        inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        if dtype_override is not None:
            if inputs.images is not None:
                inputs.images = inputs.images.to(dtype_override)

        return inputs
