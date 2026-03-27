# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PaliGemma2 model loader implementation for multimodal conditional generation.
"""

from typing import Optional

from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available PaliGemma2 model variants."""

    PALIGEMMA2_3B_MIX_224 = "3B_Mix_224"


class ModelLoader(ForgeModel):
    """PaliGemma2 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.PALIGEMMA2_3B_MIX_224: ModelConfig(
            pretrained_model_name="google/paligemma2-3b-mix-224",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PALIGEMMA2_3B_MIX_224

    sample_text = "describe en"

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
            task=ModelTask.MM_CONDITIONAL_GENERATION,
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(model_name), **model_kwargs
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for PaliGemma2."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        inputs = self.processor(
            images=image, text=self.sample_text, return_tensors="pt"
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
