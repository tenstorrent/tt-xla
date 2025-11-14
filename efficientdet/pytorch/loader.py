# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Efficientdet model loader implementation
"""

from typing import Optional
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


class ModelVariant(StrEnum):
    """Available Efficientdet model variants."""

    D0 = "d0"
    D1 = "d1"
    D2 = "d2"
    D3 = "d3"
    D4 = "d4"
    D5 = "d5"
    D6 = "d6"
    D7 = "d7"
    D7X = "d7x"


class ModelLoader(ForgeModel):
    """Efficientdet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.D0: ModelConfig(
            pretrained_model_name="tf_efficientdet_d0",
        ),
        ModelVariant.D1: ModelConfig(
            pretrained_model_name="tf_efficientdet_d1",
        ),
        ModelVariant.D2: ModelConfig(
            pretrained_model_name="tf_efficientdet_d2",
        ),
        ModelVariant.D3: ModelConfig(
            pretrained_model_name="tf_efficientdet_d3",
        ),
        ModelVariant.D4: ModelConfig(
            pretrained_model_name="tf_efficientdet_d4",
        ),
        ModelVariant.D5: ModelConfig(
            pretrained_model_name="tf_efficientdet_d5",
        ),
        ModelVariant.D6: ModelConfig(
            pretrained_model_name="tf_efficientdet_d6",
        ),
        ModelVariant.D7: ModelConfig(
            pretrained_model_name="tf_efficientdet_d7",
        ),
        ModelVariant.D7X: ModelConfig(
            pretrained_model_name="tf_efficientdet_d7x",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.D0

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

        if variant == ModelVariant.D0:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="Efficientdet",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Efficientdet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Efficientdet model instance.
        """
        from .src.model_utils import create_model

        variant_name = self._variant_config.pretrained_model_name

        # Create model
        model = create_model(variant_name, pretrained=True)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        # To get input size for post processing
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return inputs for the Efficientdet model

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for Efficientdet.
        """
        from .src.model_utils import preprocess_image

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/test2017/000000000001.jpg")

        # Preprocess the Image
        inputs = preprocess_image(image_file, target_size=self.model.config.image_size)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
