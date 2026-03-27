# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V-JEPA2 model loader implementation for video classification.
"""

from typing import Optional

import numpy as np
import torch
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
    """Available V-JEPA2 model variants."""

    VITL_FPC64_256 = "vitl_fpc64_256"


class ModelLoader(ForgeModel):
    """V-JEPA2 model loader for video classification."""

    _VARIANTS = {
        ModelVariant.VITL_FPC64_256: ModelConfig(
            pretrained_model_name="facebook/vjepa2-vitl-fpc64-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITL_FPC64_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize V-JEPA2 model loader."""
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
        """Load and return the V-JEPA2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        self.processor = AutoVideoProcessor.from_pretrained(model_name)

        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return input tensors for V-JEPA2."""
        if self.processor is None:
            raise RuntimeError(
                "Model must be loaded first before loading inputs. Call load_model() first."
            )

        # Create synthetic video: 64 frames of 256x256 RGB
        video = np.random.randint(0, 255, (64, 256, 256, 3), dtype=np.uint8)

        inputs = self.processor(video, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: v.to(dtype_override)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v
                for k, v in inputs.items()
            }

        return dict(inputs)
