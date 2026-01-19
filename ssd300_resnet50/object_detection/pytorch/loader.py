# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSD300 ResNet50 model loader implementation for object detection
"""
import numpy as np
import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file
from .src.utils import prepare_ssd_input, download_checkpoint


class ModelVariant(StrEnum):
    """Available SSD300 ResNet50 model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """SSD300 ResNet50 model loader implementation for object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="nvidia_ssd",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Shared configuration parameters
    checkpoint_url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"
    sample_image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="ssd300_resnet50",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the SSD300 ResNet50 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The SSD300 ResNet50 model instance for object detection.
        """
        # Load model from torch hub
        model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False
        )

        # Download and load checkpoint using shared utility
        checkpoint_file = download_checkpoint(self.checkpoint_url, self.checkpoint_path)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SSD300 ResNet50 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model.
        """
        # Download sample image
        input_image = get_file(self.sample_image_url)

        # Prepare input using shared utility function
        HWC = prepare_ssd_input(input_image)
        CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
        batch = np.expand_dims(CHW, axis=0)
        input_batch = torch.from_numpy(batch).float()

        # Create batch (default 1)
        input_batch = input_batch.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            input_batch = input_batch.to(dtype_override)

        return input_batch
