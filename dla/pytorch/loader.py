# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DLA model loader implementation
"""

from typing import Optional
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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


@dataclass
class DLAConfig(ModelConfig):
    """Configuration specific to DLA models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available DLA model variants."""

    # Torchvision variants
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

    # Timm variants
    DLA34_IN1K = "dla34.in1k"


class ModelLoader(ForgeModel):
    """DLA model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torchvision variants
        ModelVariant.DLA34: DLAConfig(
            pretrained_model_name="dla34",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA46_C: DLAConfig(
            pretrained_model_name="dla46_c",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA46X_C: DLAConfig(
            pretrained_model_name="dla46x_c",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA60: DLAConfig(
            pretrained_model_name="dla60",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA60X: DLAConfig(
            pretrained_model_name="dla60x",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA60X_C: DLAConfig(
            pretrained_model_name="dla60x_c",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA102: DLAConfig(
            pretrained_model_name="dla102",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA102X: DLAConfig(
            pretrained_model_name="dla102x",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA102X2: DLAConfig(
            pretrained_model_name="dla102x2",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA169: DLAConfig(
            pretrained_model_name="dla169",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DLA34_IN1K: DLAConfig(
            pretrained_model_name="dla34.in1k",
            source=ModelSource.TIMM,
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
        self._cached_model = None

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
            model="dla",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
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
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)
        else:
            # Load model using the dla_model module (torchvision style)
            func = getattr(dla_model, model_name)
            model = func(pretrained="imagenet")

        model.eval()

        # Cache model for use in load_inputs (to avoid reloading)
        self._cached_model = model

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
        image = Image.open(image_file).convert("RGB")

        source = self._variant_config.source

        if source == ModelSource.TIMM:
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override)

            # Preprocess image using model's data config
            data_config = resolve_data_config({}, model=model_for_config)
            timm_transforms = create_transform(**data_config)
            inputs = timm_transforms(image).unsqueeze(0)
        else:
            # Use standard torchvision preprocessing
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
