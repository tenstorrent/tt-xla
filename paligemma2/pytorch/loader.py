# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PaliGemma2 model loader implementation for multimodal conditional generation.
"""

from typing import Optional

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

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
    """Available PaliGemma2 model variants."""

    PALIGEMMA2_3B_FT_DOCCI_448 = "3b_ft_docci_448"


class ModelLoader(ForgeModel):
    """PaliGemma2 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.PALIGEMMA2_3B_FT_DOCCI_448: ModelConfig(
            pretrained_model_name="google/paligemma2-3b-ft-docci-448",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PALIGEMMA2_3B_FT_DOCCI_448

    sample_text = "caption en"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize PaliGemma2 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PaliGemma2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = PaliGemmaProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PaliGemma2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for PaliGemma2."""
        if self.processor is None:
            self._load_processor()

        from datasets import load_dataset

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            text=self.sample_text, images=image, return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
