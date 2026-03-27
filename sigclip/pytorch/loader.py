# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SigCLIP Vision model loader implementation for image feature extraction.

Loads the vision encoder from Comfy-Org/sigclip_vision_384, which is the
SigLIP SO400M vision tower (google/siglip-so400m-patch14-384).
"""

import torch
from transformers import AutoImageProcessor, SiglipVisionModel
from datasets import load_dataset
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available SigCLIP Vision model variants."""

    SO400M_PATCH14_384 = "So400m_Patch14_384"


class ModelLoader(ForgeModel):
    """SigCLIP Vision model loader for image feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.SO400M_PATCH14_384: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch14-384",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SO400M_PATCH14_384

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SigCLIP Vision",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant."""
        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SigCLIP vision encoder model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The SigLIP vision model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = SiglipVisionModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SigCLIP vision model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
