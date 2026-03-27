# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TexTeller model loader implementation for mathematical formula image-to-LaTeX
using PyTorch.
"""

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


class ModelVariant(StrEnum):
    """Available TexTeller PyTorch image-to-text model variants."""

    TEXTELLER = "TexTeller"


class ModelLoader(ForgeModel):
    """TexTeller model loader for mathematical formula recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.TEXTELLER: ModelConfig(
            pretrained_model_name="OleehyO/TexTeller",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEXTELLER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TexTeller",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import VisionEncoderDecoderModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from PIL import Image
        from torchvision import transforms

        # TexTeller expects single-channel (grayscale) 448x448 images
        image = Image.new("L", (448, 448), color=255)

        pixel_values = transforms.ToTensor()(image).unsqueeze(0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        decoder_input_ids = torch.tensor([[2]]).repeat_interleave(batch_size, dim=0)

        return {"pixel_values": pixel_values, "decoder_input_ids": decoder_input_ids}

    def decode_output(self, outputs):
        if self._tokenizer is None:
            self._load_tokenizer()

        if hasattr(outputs, "logits"):
            predicted_ids = outputs.logits.argmax(-1)
        else:
            predicted_ids = outputs[0].argmax(-1)

        generated_text = self._tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return generated_text
