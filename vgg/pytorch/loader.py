# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGG model loader implementation
"""

import torch
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

from PIL import Image
from ...tools.utils import get_file, print_compiled_model_results
from torchvision import transforms
from dataclasses import dataclass
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import models as tv_models


@dataclass
class VGGConfig(ModelConfig):
    source: ModelSource
    model_function: Optional[str] = None  # for torchvision
    weights_class: Optional[str] = None  # for torchvision


class ModelVariant(StrEnum):
    """Available VGG model variants."""

    # OSMR (pytorchcv) image classification variants
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    VGG19_BN_OSMR = "bn_vgg19"
    VGG19_BNB_OSMR = "bn_vgg19b"

    # TorchHub variant
    VGG19_BN = "vgg19_bn"

    # TIMM variant
    TIMM_VGG19_BN = "timm_vgg19_bn"

    # Torchvision variants
    TV_VGG11 = "torchvision_vgg11"
    TV_VGG11_BN = "torchvision_vgg11_bn"
    TV_VGG13 = "torchvision_vgg13"
    TV_VGG13_BN = "torchvision_vgg13_bn"
    TV_VGG16 = "torchvision_vgg16"
    TV_VGG16_BN = "torchvision_vgg16_bn"
    TV_VGG19 = "torchvision_vgg19"
    TV_VGG19_BN = "torchvision_vgg19_bn"

    # HuggingFace vgg-pytorch
    HF_VGG19 = "hf_vgg19"


class ModelLoader(ForgeModel):
    """VGG model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # OSMR variants
        ModelVariant.VGG11: VGGConfig(
            pretrained_model_name="vgg11", source=ModelSource.OSMR
        ),
        ModelVariant.VGG13: VGGConfig(
            pretrained_model_name="vgg13", source=ModelSource.OSMR
        ),
        ModelVariant.VGG16: VGGConfig(
            pretrained_model_name="vgg16", source=ModelSource.OSMR
        ),
        ModelVariant.VGG19: VGGConfig(
            pretrained_model_name="vgg19", source=ModelSource.OSMR
        ),
        ModelVariant.VGG19_BN_OSMR: VGGConfig(
            pretrained_model_name="bn_vgg19", source=ModelSource.OSMR
        ),
        ModelVariant.VGG19_BNB_OSMR: VGGConfig(
            pretrained_model_name="bn_vgg19b", source=ModelSource.OSMR
        ),
        # TorchHub
        ModelVariant.VGG19_BN: VGGConfig(
            pretrained_model_name="vgg19_bn", source=ModelSource.TORCH_HUB
        ),
        # TIMM
        ModelVariant.TIMM_VGG19_BN: VGGConfig(
            pretrained_model_name="vgg19_bn", source=ModelSource.TIMM
        ),
        # Torchvision
        ModelVariant.TV_VGG11: VGGConfig(
            pretrained_model_name="vgg11",
            source=ModelSource.TORCHVISION,
            model_function="vgg11",
            weights_class="VGG11_Weights",
        ),
        ModelVariant.TV_VGG11_BN: VGGConfig(
            pretrained_model_name="vgg11_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg11_bn",
            weights_class="VGG11_BN_Weights",
        ),
        ModelVariant.TV_VGG13: VGGConfig(
            pretrained_model_name="vgg13",
            source=ModelSource.TORCHVISION,
            model_function="vgg13",
            weights_class="VGG13_Weights",
        ),
        ModelVariant.TV_VGG13_BN: VGGConfig(
            pretrained_model_name="vgg13_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg13_bn",
            weights_class="VGG13_BN_Weights",
        ),
        ModelVariant.TV_VGG16: VGGConfig(
            pretrained_model_name="vgg16",
            source=ModelSource.TORCHVISION,
            model_function="vgg16",
            weights_class="VGG16_Weights",
        ),
        ModelVariant.TV_VGG16_BN: VGGConfig(
            pretrained_model_name="vgg16_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg16_bn",
            weights_class="VGG16_BN_Weights",
        ),
        ModelVariant.TV_VGG19: VGGConfig(
            pretrained_model_name="vgg19",
            source=ModelSource.TORCHVISION,
            model_function="vgg19",
            weights_class="VGG19_Weights",
        ),
        ModelVariant.TV_VGG19_BN: VGGConfig(
            pretrained_model_name="vgg19_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg19_bn",
            weights_class="VGG19_BN_Weights",
        ),
        # HuggingFace
        ModelVariant.HF_VGG19: VGGConfig(
            pretrained_model_name="vgg19", source=ModelSource.HUGGING_FACE
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TV_VGG19_BN

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
        source = cls._VARIANTS[variant].source
        return ModelInfo(
            model="vgg",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the VGG model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The VGG model instance.
        """

        from vgg_pytorch import VGG as HFVGG

        # Get the pretrained model name from the instance's variant config
        cfg = self._variant_config
        model_name = cfg.pretrained_model_name
        source = cfg.source

        if source == ModelSource.TORCH_HUB:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )
        elif source == ModelSource.TIMM:
            model = timm.create_model(model_name, pretrained=True)
        elif source == ModelSource.OSMR:
            model = ptcv_get_model(model_name, pretrained=True)
        elif source == ModelSource.HUGGING_FACE:
            model = HFVGG.from_pretrained(model_name)
        elif source == ModelSource.TORCHVISION:
            weights = getattr(tv_models, self._variant_config.weights_class).DEFAULT
            model = getattr(tv_models, self._variant_config.model_function)(
                weights=weights
            )
        else:
            raise ValueError(f"Unsupported source: {source}")

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        # Cache model for timm input config if needed
        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the VGG model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for VGG.
        """
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        if self._variant_config.source == ModelSource.TIMM:
            # Use cached model if available, otherwise load it
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override)

            data_config = resolve_data_config({}, model=model_for_config)
            data_transforms = create_transform(**data_config)
            inputs = data_transforms(image).unsqueeze(0)
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
