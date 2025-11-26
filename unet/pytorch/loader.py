# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet model loader implementation with multiple sources (OSMR, TorchHub, SMP).
"""
import numpy as np
import torch
from typing import Optional, Callable
from dataclasses import dataclass
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
from ...tools.utils import get_file, VisionPreprocessor


@dataclass
class UnetConfig(ModelConfig):
    source: ModelSource
    # TorchHub-specific fields
    hub_repo: Optional[str] = None
    hub_model: Optional[str] = None
    # SMP-specific fields
    smp_encoder_name: Optional[str] = None


class ModelVariant(StrEnum):
    """Available UNet model variants."""

    # OSMR (pytorchcv)
    OSMR_CITYSCAPES = "unet_cityscapes"

    # Qubvel SMP (segmentation_models_pytorch)
    SMP_UNET_RESNET101 = "smp_unet_resnet101"

    # TorchHub brain segmentation UNet
    TORCHHUB_BRAIN_UNET = "torchhub_brain_unet"

    # Carvana UNet (in-repo fallback)
    CARVANA_UNET = "carvana_unet"
    CARVANA_UNET_480x640 = "carvana_unet_480x640"


class ModelLoader(ForgeModel):
    """UNet model loader implementation supporting multiple sources."""

    _VARIANTS = {
        ModelVariant.OSMR_CITYSCAPES: UnetConfig(
            pretrained_model_name="unet_cityscapes",
            source=ModelSource.OSMR,
        ),
        ModelVariant.SMP_UNET_RESNET101: UnetConfig(
            pretrained_model_name="unet",
            source=ModelSource.TORCH_HUB,  # Match original test property even though loaded via SMP
            smp_encoder_name="resnet101",
        ),
        ModelVariant.TORCHHUB_BRAIN_UNET: UnetConfig(
            pretrained_model_name="unet",
            source=ModelSource.TORCH_HUB,
            hub_repo="mateuszbuda/brain-segmentation-pytorch",
            hub_model="unet",
        ),
        ModelVariant.CARVANA_UNET: UnetConfig(
            pretrained_model_name="carvana_unet",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.CARVANA_UNET_480x640: UnetConfig(
            pretrained_model_name="carvana_unet_480x640",
            source=ModelSource.CUSTOM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OSMR_CITYSCAPES

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="unet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    def load_model(self, dtype_override=None):
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.OSMR:
            from pytorchcv.model_provider import get_model as ptcv_get_model

            model = ptcv_get_model(cfg.pretrained_model_name, pretrained=False)

        elif source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:
            # TorchHub brain segmentation UNet
            model = torch.hub.load(
                cfg.hub_repo,
                cfg.hub_model,
                in_channels=3,
                out_channels=1,
                init_features=32,
                pretrained=True,
            )
            model.eval()

        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            # Qubvel SMP Unet
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name=cfg.smp_encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
            model.eval()

        else:
            # Fallback to a simple in-repo UNET (if needed)
            from .src.unet import UNET

            model = UNET(in_channels=3, out_channels=1)

        model.eval()

        # Store model for potential use in input preprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def _create_custom_preprocess_fn(self) -> Callable:
        """Create a custom preprocessing function based on the variant."""
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.OSMR:
            # Random input consistent with previous OSMR test
            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                return torch.randn(1, 3, 224, 224)

        elif source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:
            # TorchHub brain segmentation sample preprocessing
            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
                preprocess = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(mean=m, std=s),
                    ]
                )
                return preprocess(image)

        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            # SMP preprocessing using encoder params
            import segmentation_models_pytorch as smp

            params = smp.encoders.get_preprocessing_params(cfg.smp_encoder_name)
            std = torch.tensor(params["std"]).view(1, 3, 1, 1)
            mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                img = image.convert("RGB")
                img_tensor = transforms.ToTensor()(img).unsqueeze(0)

                # Ensure dimensions are divisible by 32 (UNet output stride requirement)
                # Pad the image to the next multiple of 32
                _, _, h, w = img_tensor.shape
                output_stride = 32
                new_h = ((h - 1) // output_stride + 1) * output_stride
                new_w = ((w - 1) // output_stride + 1) * output_stride

                # Pad if needed
                if h != new_h or w != new_w:
                    pad_h = new_h - h
                    pad_w = new_w - w
                    # Pad: (left, right, top, bottom)
                    img_tensor = torch.nn.functional.pad(
                        img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
                    )

                return (img_tensor - mean) / std

        elif self._variant == ModelVariant.CARVANA_UNET_480x640:

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                return torch.rand(1, 3, 480, 640)

        else:

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                return torch.rand(1, 3, 224, 224)

        return preprocess_fn

    def _get_default_image_for_variant(self) -> Optional[str]:
        """Get default image URL for variants that need it."""
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:
            return "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png"
        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            return "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        else:
            # For random input variants, return None to use random generation
            return None

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default for variant).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            cfg = self._variant_config
            source = cfg.source

            # Get default image URL for this variant
            default_image_url = self._get_default_image_for_variant()

            # Create custom preprocessing function
            custom_preprocess_fn = self._create_custom_preprocess_fn()

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name=cfg.pretrained_model_name,
                default_image_url=default_image_url,
                custom_preprocess_fn=custom_preprocess_fn,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        # For variants with None default_image_url (random input generation),
        # provide a dummy PIL Image when image=None to avoid errors in preprocessor
        if image is None and self._get_default_image_for_variant() is None:
            # Create a dummy image - the custom preprocessor will generate random tensors anyway
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )
