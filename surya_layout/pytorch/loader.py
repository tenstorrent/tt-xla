# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Layout model loader implementation for document layout detection.
"""

from typing import Optional, List

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel

from .src.utils import (
    SuryaLayoutWrapper,
    save_outputs_layout,
)


class ModelVariant(StrEnum):
    """Available Surya Layout model variants."""

    LAYOUT = "Layout"


class ModelLoader(ForgeModel):
    """Surya Layout model loader implementation for document layout detection."""

    _VARIANTS = {
        ModelVariant.LAYOUT: ModelConfig(
            pretrained_model_name="datalab-to/surya_layout",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAYOUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._transform = transforms.Compose([transforms.ToTensor()])
        self.image_tensor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SuryaLayout",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Surya Layout wrapper model.

        Returns:
            nn.Module: A wrapper module that calls Surya LayoutPredictor.
        """
        model = SuryaLayoutWrapper()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = torch.float32, batch_size=1
    ):
        """Generate sample inputs for Surya Layout.

        Returns:
            torch.Tensor: Image tensor for layout detection.
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")
        image_tensor = self._transform(image)
        image_tensor = torch.stack([image_tensor])

        self.images: List[Image.Image] = [image]

        if batch_size > 1:
            image_tensor = image_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            image_tensor = image_tensor.to(dtype_override)

        self.image_tensor = image_tensor
        return image_tensor

    def post_process(self, co_out, result_path):
        save_outputs_layout(co_out, self.images, result_path)
