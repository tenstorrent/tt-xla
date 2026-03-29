# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3n BnB 4-bit model loader implementation for multimodal modeling.
"""

from typing import Optional

from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
)
from PIL import Image

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Gemma 3n BnB 4-bit multimodal model variants."""

    GEMMA_3N_E2B_IT_BNB_4BIT = "E2B_Instruct_BnB_4bit"


class ModelLoader(ForgeModel):
    """Gemma 3n BnB 4-bit model loader implementation for multimodal modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3N_E2B_IT_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3N_E2B_IT_BNB_4BIT

    sample_text = "What do you see in this image?"
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Gemma 3n BnB 4-bit Multimodal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant."""
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name, **kwargs)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma 3n BnB 4-bit multimodal model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Gemma 3n BnB 4-bit model instance for multimodal modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override)

        model_kwargs = {}
        model_kwargs["device_map"] = "cpu"
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Gemma3nForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
    ):
        """Load and return sample inputs for the Gemma 3n BnB 4-bit multimodal model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override)

        image_file = get_file(image_url or self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
