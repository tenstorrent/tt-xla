# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo model loader implementation for multimodal visual question answering.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Molmo model variants."""

    MOLMO_72B_0924 = "72B_0924"


class ModelLoader(ForgeModel):
    """Molmo model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.MOLMO_72B_0924: LLMModelConfig(
            pretrained_model_name="allenai/Molmo-72B-0924",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO_72B_0924

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Molmo model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Molmo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Molmo model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(str(model_name), **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Molmo."""
        if self.processor is None:
            self._load_processor()

        # Load sample image
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        # Process inputs using Molmo's custom processor
        inputs = self.processor.process(images=[image], text=self.sample_text)

        # Add batch dimension
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return inputs
