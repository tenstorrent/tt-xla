# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DLA model loader implementation
"""

from typing import Optional
from PIL import Image
from torchvision import transforms

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
from ...tools.utils import get_file, print_compiled_model_results
from .src import dla_model


class ModelVariant(StrEnum):
    """Available DLA model variants."""

    DLA34 = "dla34"
    DLA46_C = "dla46_c"
    DLA46X_C = "dla46x_c"
    DLA60 = "dla60"
    DLA60X = "dla60x"
    DLA60X_C = "dla60x_c"
    DLA102 = "dla102"
    DLA102X = "dla102x"
    DLA102X2 = "dla102x2"
    DLA169 = "dla169"


class ModelLoader(ForgeModel):
    """DLA model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DLA34: ModelConfig(
            pretrained_model_name="dla34",
        ),
        ModelVariant.DLA46_C: ModelConfig(
            pretrained_model_name="dla46_c",
        ),
        ModelVariant.DLA46X_C: ModelConfig(
            pretrained_model_name="dla46x_c",
        ),
        ModelVariant.DLA60: ModelConfig(
            pretrained_model_name="dla60",
        ),
        ModelVariant.DLA60X: ModelConfig(
            pretrained_model_name="dla60x",
        ),
        ModelVariant.DLA60X_C: ModelConfig(
            pretrained_model_name="dla60x_c",
        ),
        ModelVariant.DLA102: ModelConfig(
            pretrained_model_name="dla102",
        ),
        ModelVariant.DLA102X: ModelConfig(
            pretrained_model_name="dla102x",
        ),
        ModelVariant.DLA102X2: ModelConfig(
            pretrained_model_name="dla102x2",
        ),
        ModelVariant.DLA169: ModelConfig(
            pretrained_model_name="dla169",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DLA34

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
        return ModelInfo(
            model="dla",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained DLA model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DLA model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load model using the dla_model module
        func = getattr(dla_model, model_name)
        model = func(pretrained=None)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for DLA model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for DLA.
        """
        # Get the Image
        image_file = get_file(
            "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        )
        image = Image.open(image_file)

        # Preprocess image
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        """Print classification results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        print_compiled_model_results(compiled_model_out)
