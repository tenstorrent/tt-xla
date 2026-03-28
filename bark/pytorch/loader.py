# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bark model loader implementation for text-to-audio generation.
"""
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

    BARK = "bark"
    BARK_SMALL = "bark-small"


class ModelLoader(ForgeModel):
    """Bark model loader implementation for text-to-audio generation."""

    _VARIANTS = {
        ModelVariant.BARK: ModelConfig(
            pretrained_model_name="suno/bark",
        ),
        ModelVariant.BARK_SMALL: ModelConfig(
            pretrained_model_name="suno/bark-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARK

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
        from transformers import AutoProcessor, BarkModel

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model = BarkModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype_override,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        inputs = self.processor(
            text=["Hello, this is a test of the Bark text-to-audio model."],
            return_tensors="pt",
        )
        return inputs
