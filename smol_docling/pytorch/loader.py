# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolDocling model loader implementation for document image-to-text conversion.
"""
import torch
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
    """Available SmolDocling model variants."""

    SMOL_DOCLING_256M_PREVIEW = "SmolDocling_256M_preview"


class ModelLoader(ForgeModel):
    """SmolDocling model loader for document image-to-text conversion."""

    _VARIANTS = {
        ModelVariant.SMOL_DOCLING_256M_PREVIEW: ModelConfig(
            pretrained_model_name="docling-project/SmolDocling-256M-preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOL_DOCLING_256M_PREVIEW

    sample_text = "Convert this page to docling."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize SmolDocling model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SmolDocling",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SmolDocling model instance."""
        from transformers import AutoModelForVision2Seq

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForVision2Seq.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for SmolDocling."""
        from PIL import Image

        from ...tools.utils import get_file

        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://huggingface.co/spaces/ds4sd/SmolDocling-256M-Demo/resolve/main/example_images/annual_rep_14.png"
        )
        image = Image.open(image_file).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            },
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
