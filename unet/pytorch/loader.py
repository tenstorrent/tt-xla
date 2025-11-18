# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet model loader implementation with multiple sources (OSMR, TorchHub, SMP).
"""
import numpy as np
import torch
from typing import Optional
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
from ...tools.utils import get_file


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

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.OSMR:
            # Random input consistent with previous OSMR test
            inputs = torch.randn(1, 3, 224, 224)

        elif source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:
            # TorchHub brain segmentation sample preprocessing
            file_path = get_file(
                "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
            )
            input_image = Image.open(file_path)
            m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=m, std=s),
                ]
            )
            input_tensor = preprocess(input_image)
            inputs = input_tensor.unsqueeze(0)

        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            # SMP preprocessing using encoder params
            import segmentation_models_pytorch as smp

            file_path = get_file(
                "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
            )
            img = Image.open(file_path).convert("RGB")

            params = smp.encoders.get_preprocessing_params(cfg.smp_encoder_name)
            std = torch.tensor(params["std"]).view(1, 3, 1, 1)
            mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

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

            inputs = (img_tensor - mean) / std

        elif self._variant == ModelVariant.CARVANA_UNET_480x640:
            inputs = torch.rand(1, 3, 480, 640)

        else:
            inputs = torch.rand(1, 3, 224, 224)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        return inputs
