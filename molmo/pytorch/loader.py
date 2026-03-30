# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo model loader implementation for multimodal visual question answering.
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Optional

from ...tools.utils import get_file
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
    """Available Molmo model variants."""

    MOLMO_7B_O = "Molmo-7B-O-0924"


class ModelLoader(ForgeModel):
    """Molmo model loader for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.MOLMO_7B_O: ModelConfig(
            pretrained_model_name="allenai/Molmo-7B-O-0924",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO_7B_O

    def __init__(self, variant: Optional[ModelVariant] = None):
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if dtype_override:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
            **kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Molmo model."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor.process(
            images=[image],
            text="Describe this image.",
        )

        # Add batch dimension
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

        if batch_size > 1:
            inputs = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in inputs.items()
            }

        if dtype_override:
            inputs = {
                k: v.to(dtype_override) if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        return inputs
