# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSD300 VGG16 model loader implementation for object detection
"""
import torch
from typing import Optional
from torchvision import models
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from PIL import Image

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from datasets import load_dataset
from ...tools.utils import print_compiled_model_results
from .src.utils import SSDPostprocessor, patched_grid_default_boxes, patched_forward


class ModelVariant(StrEnum):
    """Available SSD300 VGG16 model variants."""

    BASE = "Default"


class ModelLoader(ForgeModel):
    """SSD300 VGG16 model loader implementation for object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="ssd300_vgg16",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Shared configuration parameters
    sample_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

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
            model="SSD300-VGG16",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCHVISION,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SSD300 VGG16 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                          NOTE: This parameter is currently ignored (model always uses float32).
                          TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16

        Returns:
            torch.nn.Module: The SSD300 VGG16 model instance for object detection.
        """
        # Monkey-patch DefaultBoxGenerator to propagate device into _grid_default_boxes,
        # so tensors created during forward are on the same device (XLA) as the feature maps
        # instead of defaulting to CPU - https://github.com/tenstorrent/tt-xla/issues/3335
        DefaultBoxGenerator._grid_default_boxes = patched_grid_default_boxes
        DefaultBoxGenerator.forward = patched_forward

        # Load model from torchvision
        weights = models.detection.SSD300_VGG16_Weights.DEFAULT
        model = models.detection.ssd300_vgg16(weights=weights)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            # model = model.to(dtype_override)
            # TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16
            print("NOTE: dtype_override ignored - batched_nms lacks BFloat16 support")

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SSD300 VGG16 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                          NOTE: This parameter is currently ignored (model always uses float32).
                          TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for SSD300 VGG16.
        """
        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        # Load and preprocess image using torchvision weights transforms
        weights = models.detection.SSD300_VGG16_Weights.DEFAULT
        preprocess = weights.transforms()
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.contiguous()

        # Create batch (default 1)
        batch_t = batch_t.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            # batch_t = batch_t.to(dtype_override)
            # TODO (@ppadjinTT): remove this when torchvision starts supporting torchvision.ops.nms for bfloat16
            print("NOTE: dtype_override ignored - batched_nms lacks BFloat16 support")

        return batch_t

    def postprocess_results(self, model, fw_out, co_out, inputs):
        """Postprocess detection results from framework and compiled model outputs and Print classification results

        Args:
            model: The SSD model instance (typically the wrapped model)
            fw_out: Framework model outputs
            co_out: Compiled model outputs
            inputs: Input tensors used for inference

        """
        postprocessor = SSDPostprocessor(model)
        _, detection_co = postprocessor.process(fw_out, co_out, inputs)
        print_compiled_model_results(detection_co)
