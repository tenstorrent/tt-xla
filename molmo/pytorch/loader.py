# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo model loader implementation for image-text-to-text generation.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Molmo model variants."""

    MOLMO_7B_D_0924 = "7B-D-0924"


class ModelLoader(ForgeModel):
    """Molmo model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.MOLMO_7B_D_0924: ModelConfig(
            pretrained_model_name="allenai/Molmo-7B-D-0924",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO_7B_D_0924

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Molmo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.load_model(dtype_override=dtype_override)

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        inputs = self.processor.process(
            images=[image],
            text="Describe this image.",
        )

        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

        if batch_size > 1:
            inputs = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in inputs.items()
            }

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.processor is None:
            return str(outputs)

        tokenizer = self.processor.tokenizer

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return tokenizer.decode(next_token_id)
