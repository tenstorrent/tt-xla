# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import cv2

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file
from .src.model_utils import create_model, pre_process, load_model


@dataclass
class CenterNetConfig(ModelConfig):
    """Configuration specific to CenterNet models"""

    source: ModelSource
    heads: dict
    head_conv: int


class ModelVariant(StrEnum):
    """Available CenterNet model variants."""

    # Hourglass-based variants
    HOURGLASS_COCO = "hourglass_coco"


class ModelLoader(ForgeModel):
    """CenterNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HOURGLASS_COCO: CenterNetConfig(
            pretrained_model_name="hourglass_coco",
            source=ModelSource.TORCH_HUB,
            heads={"hm": 80, "wh": 2, "reg": 2},  # COCO has 80 classes
            head_conv=64,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HOURGLASS_COCO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="centernet",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained CenterNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The CenterNet model instance.
        """
        # Get the configuration from the instance's variant config
        config = self._variant_config

        # Create model using the heads and head_conv from config
        model = create_model("hourglass", config.heads, config.head_conv)

        # Load model weights
        model = load_model(
            model, get_file("test_files/pytorch/centernet/ctdet_coco_hg.pth")
        )

        # Set model to evaluation mode
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for CenterNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for CenterNet.
        """
        # Get the Image
        image_file = get_file(
            "https://github.com/xingyizhou/CenterNet/raw/master/images/17790319373_bd19b24cfc_k.jpg"
        )
        image = cv2.imread(image_file)

        # Preprocess image
        inputs = pre_process(image)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
