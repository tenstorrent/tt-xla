# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hulu-Med model loader implementation for medical image-to-text generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
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


class ModelVariant(StrEnum):
    """Available Hulu-Med model variants for medical image-to-text."""

    HULU_MED_4B = "4B"


class ModelLoader(ForgeModel):
    """Hulu-Med model loader implementation for medical image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.HULU_MED_4B: ModelConfig(
            pretrained_model_name="ZJU-AI4H/Hulu-Med-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HULU_MED_4B

    sample_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.model = None

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
            model="Hulu-Med",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hulu-Med model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Hulu-Med model instance for medical image-to-text.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Hulu-Med model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": {
                            "image_path": image_file,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image.",
                    },
                ],
            }
        ]

        inputs = self.processor(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output tensors.
            dtype_override: Optional torch.dtype (unused, for API compatibility).
            inputs: Optional input tensors (unused, for API compatibility).

        Returns:
            str: Decoded text output.
        """
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.processor.decode(next_token_id)
