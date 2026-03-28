# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-Med model loader implementation for biomedical visual question answering.
"""

from typing import Optional

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

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
from ...tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaVA-Med model variants."""

    LLAVA_MED_V1_5_MISTRAL_7B = "v1.5_Mistral_7B"


class ModelLoader(ForgeModel):
    """LLaVA-Med model loader for biomedical visual question answering."""

    _VARIANTS = {
        ModelVariant.LLAVA_MED_V1_5_MISTRAL_7B: ModelConfig(
            pretrained_model_name="microsoft/llava-med-v1.5-mistral-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_MED_V1_5_MISTRAL_7B

    sample_text = "What organ is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-Med model loader."""
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-Med",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        return self.tokenizer

    def _load_image_processor(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )
        return self.image_processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-Med model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name), trust_remote_code=True, **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        if self.image_processor is None:
            self._load_image_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-Med."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if self.image_processor is None:
            self._load_image_processor()

        # Load sample image
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        # Process image
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        # Tokenize text
        input_ids = self.tokenizer(self.sample_text, return_tensors="pt").input_ids

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "images": pixel_values,
        }
