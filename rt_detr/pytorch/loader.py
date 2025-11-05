# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RT-DETR (Real-Time DETR) model loader implementation for object detection.
"""
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from typing import Optional
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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available RT-DETR model variants for object detection."""

    RTDETR_R18VD = "rtdetr-r18vd"
    RTDETR_R34VD = "rtdetr-r34vd"
    RTDETR_R50VD = "rtdetr-r50vd"
    RTDETR_R101VD = "rtdetr-r101vd"


class ModelLoader(ForgeModel):
    """RT-DETR model loader implementation for real-time object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RTDETR_R18VD: ModelConfig(
            pretrained_model_name="PekingU/rtdetr_r18vd",
        ),
        ModelVariant.RTDETR_R34VD: ModelConfig(
            pretrained_model_name="PekingU/rtdetr_r34vd",
        ),
        ModelVariant.RTDETR_R50VD: ModelConfig(
            pretrained_model_name="PekingU/rtdetr_r50vd",
        ),
        ModelVariant.RTDETR_R101VD: ModelConfig(
            pretrained_model_name="PekingU/rtdetr_r101vd",
        ),
    }

    # Default variant to use (smallest model for fastest testing)
    DEFAULT_VARIANT = ModelVariant.RTDETR_R18VD

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant in [
            ModelVariant.RTDETR_R18VD,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="rt_detr",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor
        self.processor = RTDetrImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the RT-DETR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The RT-DETR model instance for object detection.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RT-DETR model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        inputs = self.processor(images=image, return_tensors="pt")

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                # Convert the input dtype to dtype_override if specified
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
