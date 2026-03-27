# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V-JEPA 2 model loader implementation for video feature extraction.
"""

from typing import Optional

import numpy as np
from transformers import AutoModel, AutoVideoProcessor

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


class ModelVariant(StrEnum):
    """Available V-JEPA 2 model variants."""

    VITG_FPC64_256 = "vitg_fpc64_256"


class ModelLoader(ForgeModel):
    """V-JEPA 2 model loader for video feature extraction."""

    _VARIANTS = {
        ModelVariant.VITG_FPC64_256: ModelConfig(
            pretrained_model_name="facebook/vjepa2-vitg-fpc64-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITG_FPC64_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize V-JEPA 2 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="V-JEPA 2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the V-JEPA 2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        if self.processor is None:
            self.processor = AutoVideoProcessor.from_pretrained(model_name)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample video inputs for V-JEPA 2.

        Creates synthetic video frames (64 frames of 256x256 RGB) as input.
        """
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = AutoVideoProcessor.from_pretrained(model_name)

        # Create synthetic video: 64 frames of 256x256 RGB
        video = np.random.randint(0, 255, (64, 256, 256, 3), dtype=np.uint8)

        inputs = self.processor(video, return_tensors="pt")

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        return dict(inputs)
