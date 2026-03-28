# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChartGemma model loader implementation for chart understanding and reasoning.
"""

from typing import Optional

from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

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
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available ChartGemma model variants."""

    CHARTGEMMA_3B = "3B"


class ModelLoader(ForgeModel):
    """ChartGemma model loader for chart understanding and reasoning."""

    _VARIANTS = {
        ModelVariant.CHARTGEMMA_3B: ModelConfig(
            pretrained_model_name="ahmed-masry/chartgemma",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHARTGEMMA_3B

    sample_image = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/png/multi_col_1229.png"
    sample_text = "What is the title of the chart?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ChartGemma model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ChartGemma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ChartGemma model instance."""
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
        """Load and return input tensors for ChartGemma."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image)
        image = Image.open(image_file).convert("RGB")

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
