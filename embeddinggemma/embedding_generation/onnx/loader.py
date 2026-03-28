# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EmbeddingGemma 300M ONNX model loader for text embedding generation.
"""

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
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
    """Available EmbeddingGemma ONNX model variants."""

    EMBEDDINGGEMMA_300M = "EmbeddingGemma-300M"


class ModelLoader(ForgeModel):
    """EmbeddingGemma 300M ONNX model loader for text embedding generation."""

    _VARIANTS = {
        ModelVariant.EMBEDDINGGEMMA_300M: ModelConfig(
            pretrained_model_name="onnx-community/embeddinggemma-300m-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EMBEDDINGGEMMA_300M

    sample_sentences = ["query: This is an example sentence for embedding generation"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="EmbeddingGemma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the EmbeddingGemma ONNX model from Hugging Face.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_path = hf_hub_download(pretrained_model_name, filename="onnx/model.onnx")
        model = onnx.load(model_path)

        return model

    def load_inputs(self, **kwargs):
        """Generate tokenized inputs for the EmbeddingGemma ONNX model.

        Returns:
            numpy.ndarray: Input IDs array suitable for the ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        inputs = self._tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="np",
        )

        return inputs["input_ids"]
