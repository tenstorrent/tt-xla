# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetinaNet model loader implementation
"""
import os
import shutil
import zipfile
import requests
import torch
from PIL import Image
from torchvision import transforms, models
from typing import Optional
from dataclasses import dataclass
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
from .src.model import Model
from ...tools.utils import get_file


@dataclass
class RetinaNetConfig(ModelConfig):
    """Configuration specific to RetinaNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available RetinaNet model variants."""

    # NVIDIA variants (custom model implementation)
    RETINANET_RN18FPN = "retinanet_rn18fpn"
    RETINANET_RN34FPN = "retinanet_rn34fpn"
    RETINANET_RN50FPN = "retinanet_rn50fpn"
    RETINANET_RN101FPN = "retinanet_rn101fpn"
    RETINANET_RN152FPN = "retinanet_rn152fpn"

    # Torchvision variants
    RETINANET_RESNET50_FPN_V2 = "retinanet_resnet50_fpn_v2"


class ModelLoader(ForgeModel):
    """RetinaNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Github variants
        ModelVariant.RETINANET_RN18FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn18fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN34FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn34fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN50FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn50fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN101FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn101fpn",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.RETINANET_RN152FPN: RetinaNetConfig(
            pretrained_model_name="retinanet_rn152fpn",
            source=ModelSource.CUSTOM,
        ),
        # Torchvision variants
        ModelVariant.RETINANET_RESNET50_FPN_V2: RetinaNetConfig(
            pretrained_model_name="retinanet_resnet50_fpn_v2",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RETINANET_RN50FPN

    # Weight mappings for torchvision variants
    _TORCHVISION_WEIGHTS = {
        ModelVariant.RETINANET_RESNET50_FPN_V2: "RetinaNet_ResNet50_FPN_V2_Weights",
    }

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
            model="retinanet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
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

        # Configuration parameters
        self._cleanup_files = []  # Track files to cleanup

    def _download_nvidia_model(self, variant_name):
        """Download and extract NVIDIA RetinaNet model."""
        url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant_name}.zip"
        local_zip_path = f"{variant_name}.zip"

        # Download the model
        response = requests.get(url)
        with open(local_zip_path, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        extracted_path = variant_name
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        # Find the .pth file
        checkpoint_path = ""
        for root, _, files in os.walk(extracted_path):
            for file in files:
                if file.endswith(".pth"):
                    checkpoint_path = os.path.join(root, file)
                    break

        # Track files for cleanup
        self._cleanup_files.extend([local_zip_path, extracted_path])

        return checkpoint_path

    def load_model(self, dtype_override=None):
        """Load and return the RetinaNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
                           NOTE: This parameter is currently ignored for retinanet_resnet50_fpn_v2(model always uses float32).
                           TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16

        Returns:
            torch.nn.Module: The RetinaNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCHVISION:
            # Load torchvision model
            weight_name = self._TORCHVISION_WEIGHTS[self._variant]
            weights = getattr(models.detection, weight_name).DEFAULT
            model = getattr(models.detection, model_name)(weights=weights)
        elif source == ModelSource.CUSTOM:
            # Load custom model
            checkpoint_path = self._download_nvidia_model(model_name)
            model = Model.load(checkpoint_path)

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            if model_name == "retinanet_resnet50_fpn_v2":
                # TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16
                print(
                    "NOTE: dtype_override ignored - batched_nms lacks BFloat16 support"
                )
            else:
                model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RetinaNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
                           NOTE: This parameter is currently ignored for retinanet_resnet50_fpn_v2(model always uses float32).
                           TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for RetinaNet.
        """
        # Get the pretrained model name  amd source from the instance's variant config
        source = self._variant_config.source
        model_name = self._variant_config.pretrained_model_name

        if source == ModelSource.TORCHVISION:
            weight_name = self._TORCHVISION_WEIGHTS[self._variant]
            weights = getattr(models.detection, weight_name).DEFAULT
            preprocess = weights.transforms()

            # Load COCO image
            input_image = get_file(
                "http://images.cocodataset.org/val2017/000000039769.jpg"
            )
            image = Image.open(str(input_image)).convert("RGB")
            img_t = preprocess(image)
            batch_t = torch.unsqueeze(img_t, 0).contiguous()
        elif source == ModelSource.CUSTOM:
            url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
            pil_img = Image.open(requests.get(url, stream=True).raw)
            new_size = (640, 480)
            pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)

            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            img = preprocess(pil_img)
            batch_t = img.unsqueeze(0)

        # Replicate tensors for batch size
        batch_t = batch_t.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            if model_name == "retinanet_resnet50_fpn_v2":
                # TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16
                print(
                    "NOTE: dtype_override ignored - batched_nms lacks BFloat16 support"
                )
            else:
                batch_t = batch_t.to(dtype_override)

        return batch_t

    def cleanup(self):
        """Clean up downloaded files."""
        for file_path in self._cleanup_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not clean up {file_path}: {e}")
        self._cleanup_files.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
