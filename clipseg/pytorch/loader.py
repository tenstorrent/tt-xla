# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIPSeg model loader implementation for zero-shot image segmentation.
"""
import torch
from PIL import Image
from typing import Optional
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

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


class ModelVariant(StrEnum):
    """Available CLIPSeg model variants for image segmentation."""

    RD64_REFINED = "Rd64_Refined"


class ModelLoader(ForgeModel):
    """CLIPSeg model loader implementation for zero-shot image segmentation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RD64_REFINED: ModelConfig(
            pretrained_model_name="CIDAS/clipseg-rd64-refined",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RD64_REFINED

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
        return ModelInfo(
            model="CLIPSeg",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = CLIPSegProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CLIPSeg model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).

        Returns:
            torch.nn.Module: The CLIPSeg model instance for image segmentation.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # NOTE: Ignoring dtype_override and always using default (fp32) due to dtype mismatch
        # issue with bfloat16. See: https://github.com/tenstorrent/tt-xla/issues/1959
        # if dtype_override is not None:
        #     model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CLIPSegForImageSegmentation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CLIPSeg model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           NOTE: This parameter is currently ignored (inputs always use float32).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Process image with text prompt for segmentation
        inputs = self.processor(
            text=["a cat"],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # NOTE: Ignoring dtype_override and always using default (fp32) due to dtype mismatch
        # issue with bfloat16. See: https://github.com/tenstorrent/tt-xla/issues/1959
        # if dtype_override is not None:
        #     inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
