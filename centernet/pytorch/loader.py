# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import cv2
import torch

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
from ...tools.utils import get_file, extract_tensors_recursive
from .src.model_utils import create_model, pre_process, load_model


@dataclass
class CenterNetConfig(ModelConfig):
    """Configuration specific to CenterNet models"""

    head_conv: int


class ModelVariant(StrEnum):
    """Available CenterNet model variants."""

    # Hourglass-based variants
    HOURGLASS_COCO = "hourglass_coco"

    # Resnet-based variants
    RESNET_18_COCO = "resnet18_coco"
    RESNET_101_COCO = "resnet101_coco"

    # DLA-based variants
    DLA_1X_COCO = "dla1x_coco"
    DLA_2X_COCO = "dla2x_coco"


class ModelLoader(ForgeModel):
    """CenterNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HOURGLASS_COCO: CenterNetConfig(
            pretrained_model_name="hourglass_coco",
            head_conv=64,
        ),
        ModelVariant.RESNET_18_COCO: CenterNetConfig(
            pretrained_model_name="resnet18_coco",
            head_conv=64,
        ),
        ModelVariant.RESNET_101_COCO: CenterNetConfig(
            pretrained_model_name="resnet101_coco",
            head_conv=64,
        ),
        ModelVariant.DLA_1X_COCO: CenterNetConfig(
            pretrained_model_name="dla1x_coco",
            head_conv=256,
        ),
        ModelVariant.DLA_2X_COCO: CenterNetConfig(
            pretrained_model_name="dla2x_coco",
            head_conv=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HOURGLASS_COCO

    # mapping variant → arch + checkpoint
    ARCH_WEIGHTS = {
        ModelVariant.HOURGLASS_COCO: ("hourglass", "ctdet_coco_hg.pth"),
        ModelVariant.RESNET_18_COCO: ("resdcn_18", "ctdet_coco_resdcn18.pth"),
        ModelVariant.RESNET_101_COCO: ("resdcn_101", "ctdet_coco_resdcn101.pth"),
        ModelVariant.DLA_1X_COCO: ("dla_34", "ctdet_coco_dla_1x.pth"),
        ModelVariant.DLA_2X_COCO: ("dla_34", "ctdet_coco_dla_2x.pth"),
    }

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
            model="centernet",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.HOURGLASS_COCO
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained CenterNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The CenterNet model instance.
        """
        # Get the configuration from the instance's variant config
        config = self._variant_config

        # pick arch + checkpoint based on variant
        arch, checkpoint_name = self.ARCH_WEIGHTS[self._variant]

        # Create the model with the specified head configuration and head_conv value
        head_config = {"hm": 80, "wh": 2, "reg": 2}

        # build model
        model = create_model(arch, head_config, config.head_conv)

        # Load model weights
        model = load_model(
            model, get_file(f"test_files/pytorch/centernet/{checkpoint_name}")
        )

        # Set model to evaluation mode
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for CenterNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for CenterNet.
        """
        # Get the Image
        image_file = get_file(
            "https://github.com/xingyizhou/CenterNet/raw/master/images/17790319373_bd19b24cfc_k.jpg"
        )
        image = cv2.imread(image_file)

        # Preprocess image
        inputs = pre_process(image)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The CenterNet model returns a list of dictionaries, one per stack:
        [{"hm": heatmap_tensor, "wh": wh_tensor, "reg": reg_tensor}, ...]

        For training, we extract all tensor outputs and concatenate them
        to create a single differentiable tensor for backpropagation.

        Args:
            fwd_output: Output from the model's forward pass

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        tensors = []
        extract_tensors_recursive(fwd_output, tensors)

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
