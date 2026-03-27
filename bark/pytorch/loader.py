# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bark text-to-audio model loader implementation.
"""
from transformers import AutoProcessor, BarkModel
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
    """Available Bark model variants."""

    SMALL = "Small"


class ModelLoader(ForgeModel):
    """Bark text-to-audio model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="suno/bark-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    sample_text = "Hello, my dog is cooler than you!"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Bark",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BarkModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.processor(self.sample_text, return_tensors="pt")
        return dict(inputs)
