# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Emu3 VisionTokenizer model loader implementation for vision feature extraction.
"""

import torch
from typing import Optional
from datasets import load_dataset
from loguru import logger

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Emu3 VisionTokenizer model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Emu3 VisionTokenizer model loader implementation for vision feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="BAAI/Emu3-VisionTokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Emu3-VisionTokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel, AutoImageProcessor

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        self.image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self.image_processor is None:
            raise RuntimeError(
                "Model must be loaded first before loading inputs. Call load_model() first."
            )

        try:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
            pixel_values = self.image_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ]
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset ({e}), replacing input with random tensor."
            )
            pixel_values = torch.rand(1, 3, 512, 512).to(torch.float32)

        # Model expects input shape (batch, channels, height, width)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
