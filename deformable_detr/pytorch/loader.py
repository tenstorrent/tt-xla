# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Deformable DETR model loader implementation for object detection.
"""
import torch
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
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
    """Available Deformable DETR model variants for object detection."""

    DEFORMABLE_DETR = "deformable-detr"
    DEFORMABLE_DETR_SINGLE_SCALE = "deformable-detr-single-scale"
    DEFORMABLE_DETR_WITH_BOX_REFINE = "deformable-detr-with-box-refine"
    DEFORMABLE_DETR_WITH_BOX_REFINE_TWO_STAGE = (
        "deformable-detr-with-box-refine-two-stage"
    )


class ModelLoader(ForgeModel):
    """Deformable DETR model loader implementation for object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEFORMABLE_DETR: ModelConfig(
            pretrained_model_name="SenseTime/deformable-detr",
        ),
        ModelVariant.DEFORMABLE_DETR_SINGLE_SCALE: ModelConfig(
            pretrained_model_name="SenseTime/deformable-detr-single-scale",
        ),
        ModelVariant.DEFORMABLE_DETR_WITH_BOX_REFINE: ModelConfig(
            pretrained_model_name="SenseTime/deformable-detr-with-box-refine",
        ),
        ModelVariant.DEFORMABLE_DETR_WITH_BOX_REFINE_TWO_STAGE: ModelConfig(
            pretrained_model_name="SenseTime/deformable-detr-with-box-refine-two-stage",
        ),
    }

    # Default variant to use (base model)
    DEFAULT_VARIANT = ModelVariant.DEFORMABLE_DETR

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

        if variant == ModelVariant.DEFORMABLE_DETR:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="deformable_detr",
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
        self.processor = DeformableDetrImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Deformable DETR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Deformable DETR model instance for object detection.
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

        model = DeformableDetrForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Deformable DETR model with this instance's variant settings.

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
