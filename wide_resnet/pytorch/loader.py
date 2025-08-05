# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WideResnet model loader implementation for question answering
"""
import torch
from PIL import Image
from typing import Optional
from torchvision import transforms
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
from ...tools.utils import get_file
from ...tools.utils import print_compiled_model_results
from typing import Optional


class ModelVariant(StrEnum):
    """Available WideResnet model variants."""

    WIDE_RESNET50_2 = "wide_resnet50_2"
    WIDE_RESNET101_2 = "wide_resnet101_2"


class ModelLoader(ForgeModel):
    """WideResnet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.WIDE_RESNET50_2: ModelConfig(
            pretrained_model_name="wide_resnet50_2",
        ),
        ModelVariant.WIDE_RESNET101_2: ModelConfig(
            pretrained_model_name="wide_resnet101_2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.WIDE_RESNET50_2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="wide_resnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
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

    def load_model(self, dtype_override=None):
        """Load a WideResnet model from Torch Hub."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", pretrained_model_name, pretrained=True
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for WideResnet models."""
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        input_image = Image.open(image_file)

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
        input_tensor = preprocess(input_image)
        inputs = input_tensor.unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, output, top_k=5):
        probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
        class_file_path = get_file(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )

        with open(class_file_path, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        topk_prob, topk_catid = torch.topk(probabilities, top_k)
        for i in range(topk_prob.size(0)):
            print(categories[topk_catid[i]], topk_prob[i].item())
