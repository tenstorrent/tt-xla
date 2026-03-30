# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 39M Vision model loader implementation for multimodal conditional generation.
"""

from typing import Optional

from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

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
    """Available Llama 3.2 39M Vision model variants."""

    LLAMA_3_2_39M_VISION = "39M_Vision"


class ModelLoader(ForgeModel):
    """Llama 3.2 39M Vision model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_39M_VISION: ModelConfig(
            pretrained_model_name="axolotl-ai-co/Llama-3.2-39M-Vision",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_39M_VISION

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Llama 3.2 39M Vision model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama 3.2 39M Vision",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llama 3.2 39M Vision model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = MllamaForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Llama 3.2 39M Vision."""
        if self.processor is None:
            self._load_processor()

        # Load sample image
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        # Build prompt with image token
        prompt = f"<|image|><|begin_of_text|>{self.sample_text}"

        # Preprocess
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return inputs
