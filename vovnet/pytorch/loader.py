# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vovnet model loader implementation for question answering
"""
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms
from PIL import Image
from typing import Optional
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
from ...tools.utils import print_compiled_model_results
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available WideResnet model variants."""

    VOVNET27S = "vovnet27s"
    VOVNET39 = "vovnet39"
    VOVNET57 = "vovnet57"


class ModelLoader(ForgeModel):
    """WideResnet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VOVNET27S: ModelConfig(
            pretrained_model_name="vovnet27s",
        ),
        ModelVariant.VOVNET39: ModelConfig(
            pretrained_model_name="vovnet39",
        ),
        ModelVariant.VOVNET57: ModelConfig(
            pretrained_model_name="vovnet57",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VOVNET27S

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
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
            model="vovnet",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.input_shape = (3, 224, 224)

    def load_model(self, dtype_override=None):
        """Load a Vovnet model from Pytorch CV Model Provider."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        model = ptcv_get_model(pretrained_model_name, pretrained=True)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Vovnet models."""

        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
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
        img_tensor = preprocess(input_image)
        inputs = img_tensor.unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
