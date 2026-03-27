# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chameleon model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import torch
from datasets import load_dataset
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Chameleon model variants."""

    CHAMELEON_7B = "7B"


class ModelLoader(ForgeModel):
    """Chameleon model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.CHAMELEON_7B: ModelConfig(
            pretrained_model_name="facebook/chameleon-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHAMELEON_7B

    sample_text = "What is shown in this image?<image>"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Chameleon model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Chameleon",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ChameleonProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Chameleon model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ChameleonForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Chameleon."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            images=image, text=self.sample_text, return_tensors="pt"
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
