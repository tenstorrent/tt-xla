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
from dataclasses import dataclass
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


@dataclass
class WideResnetConfig(ModelConfig):
    source: ModelSource


class ModelVariant(StrEnum):
    """Available WideResnet model variants."""

    # Torch Hub/Torchvision variants
    WIDE_RESNET50_2 = "wide_resnet50_2"
    WIDE_RESNET101_2 = "wide_resnet101_2"

    # TIMM variants
    TIMM_WIDE_RESNET50_2 = "wide_resnet50_2.timm"
    TIMM_WIDE_RESNET101_2 = "wide_resnet101_2.timm"


class ModelLoader(ForgeModel):
    """WideResnet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torch Hub variants
        ModelVariant.WIDE_RESNET50_2: WideResnetConfig(
            pretrained_model_name="wide_resnet50_2",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.WIDE_RESNET101_2: WideResnetConfig(
            pretrained_model_name="wide_resnet101_2",
            source=ModelSource.TORCH_HUB,
        ),
        # TIMM variants
        ModelVariant.TIMM_WIDE_RESNET50_2: WideResnetConfig(
            pretrained_model_name="wide_resnet50_2",
            source=ModelSource.TIMM,
        ),
        ModelVariant.TIMM_WIDE_RESNET101_2: WideResnetConfig(
            pretrained_model_name="wide_resnet101_2",
            source=ModelSource.TIMM,
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
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        source = cls._VARIANTS[variant].source
        return ModelInfo(
            model="wide_resnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

    def load_model(self, dtype_override=None):
        """Load a WideResnet model from Torch Hub or TIMM depending on variant source."""

        # Get the pretrained model name and source from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            model = timm.create_model(pretrained_model_name, pretrained=True)
        else:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", pretrained_model_name, pretrained=True
            )
        model.eval()

        # Cache for preprocessing config
        self._cached_model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for WideResnet models."""
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        input_image = Image.open(image_file).convert("RGB")

        source = self._variant_config.source

        if source == ModelSource.TIMM:
            model_for_config = (
                self._cached_model if self._cached_model is not None else None
            )
            if model_for_config is None:
                model_for_config = self.load_model(dtype_override)
            data_config = resolve_data_config({}, model=model_for_config)
            transform = create_transform(**data_config)
            inputs = transform(input_image).unsqueeze(0)
        else:
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
            inputs = preprocess(input_image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

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
