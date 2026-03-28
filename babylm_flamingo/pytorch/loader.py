# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM Flamingo model loader implementation for multimodal causal language modeling.
"""

from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

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
    """Available BabyLM Flamingo model variants."""

    MULTIMODAL_BASELINE = "multimodal_baseline"


class ModelLoader(ForgeModel):
    """BabyLM Flamingo model loader for multimodal causal language modeling."""

    _VARIANTS = {
        ModelVariant.MULTIMODAL_BASELINE: ModelConfig(
            pretrained_model_name="BabyLM-community/babylm-multimodal-baseline-flamingo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTIMODAL_BASELINE

    sample_text = "A photo of"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize BabyLM Flamingo model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BabyLM Flamingo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CAUSAL_LM,
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
        """Load and return the BabyLM Flamingo model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for BabyLM Flamingo."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            images=image,
            text=self.sample_text,
            return_tensors="pt",
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return dict(inputs)
