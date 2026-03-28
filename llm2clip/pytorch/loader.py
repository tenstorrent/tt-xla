# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLM2CLIP model loader implementation for image feature extraction.
"""

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
    """Available LLM2CLIP model variants."""

    OPENAI_L_14_336 = "Openai-L-14-336"


class ModelLoader(ForgeModel):
    """LLM2CLIP model loader implementation for image feature extraction."""

    _VARIANTS = {
        ModelVariant.OPENAI_L_14_336: ModelConfig(
            pretrained_model_name="microsoft/LLM2CLIP-Openai-L-14-336",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENAI_L_14_336

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LLM2CLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import CLIPImageProcessor

        self._processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        from datasets import load_dataset

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self._processor(images=image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
