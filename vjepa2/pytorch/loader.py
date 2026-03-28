# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V-JEPA 2 model loader implementation for video classification.
"""

from typing import Optional

import numpy as np
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available V-JEPA 2 model variants."""

    VITL_FPC16_256_SSV2 = "vitl_fpc16_256_ssv2"


class ModelLoader(ForgeModel):
    """V-JEPA 2 model loader for video classification."""

    _VARIANTS = {
        ModelVariant.VITL_FPC16_256_SSV2: ModelConfig(
            pretrained_model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITL_FPC16_256_SSV2

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize V-JEPA 2 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="V-JEPA2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the V-JEPA 2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModelForVideoClassification.from_pretrained(model_name, **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self.processor = AutoVideoProcessor.from_pretrained(model_name)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for V-JEPA 2."""
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = AutoVideoProcessor.from_pretrained(model_name)

        # Create a synthetic video: 16 frames of 256x256 RGB
        num_frames = 16
        video = np.random.randint(0, 255, (num_frames, 256, 256, 3), dtype=np.uint8)

        inputs = self.processor(list(video), return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
