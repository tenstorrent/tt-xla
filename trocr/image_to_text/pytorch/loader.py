# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TrOCR (Transformer-based Optical Character Recognition) model loader implementation
for image-to-text using PyTorch.
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
    """Available TrOCR PyTorch image-to-text model variants."""

    LARGE_PRINTED = "Large_Printed"


class ModelLoader(ForgeModel):
    """TrOCR model loader implementation for image-to-text (PyTorch)."""

    _VARIANTS = {
        ModelVariant.LARGE_PRINTED: ModelConfig(
            pretrained_model_name="microsoft/trocr-large-printed",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PRINTED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TrOCR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import TrOCRProcessor

        self._processor = TrOCRProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self._processor

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

        # Fix sinusoidal positional embeddings left on meta device after loading
        embed_positions = model.decoder.model.decoder.embed_positions
        if hasattr(embed_positions, "weights") and embed_positions.weights is not None:
            if embed_positions.weights.device.type == "meta":
                embed_positions.weights = embed_positions.get_embedding(
                    embed_positions.weights.shape[0],
                    embed_positions.embedding_dim,
                    embed_positions.padding_idx,
                )

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        from PIL import Image
        import requests

        if self._processor is None:
            self._load_processor()

        # Load a sample image of printed text
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        pixel_values = self._processor(
            images=image,
            return_tensors="pt",
        ).pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values

    def decode_output(self, co_out):
        """Decode model outputs into human-readable text.

        Args:
            co_out: Model output (generated token IDs)

        Returns:
            str: Decoded text
        """
        if self._processor is None:
            self._load_processor()

        generated_text = self._processor.batch_decode(co_out, skip_special_tokens=True)[
            0
        ]
        print(f"Generated text: {generated_text}")
        return generated_text
