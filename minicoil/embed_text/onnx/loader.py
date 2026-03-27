# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCOIL v1 ONNX model loader implementation for sparse text embedding generation.
"""
import onnx
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
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
    """Available MiniCOIL model variants."""

    MINICOIL_V1 = "minicoil-v1"


class ModelLoader(ForgeModel):
    """MiniCOIL v1 ONNX model loader for sparse text embedding generation."""

    _VARIANTS = {
        ModelVariant.MINICOIL_V1: ModelConfig(
            pretrained_model_name="Qdrant/minicoil-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICOIL_V1

    sample_sentences = [
        "MiniCOIL generates sparse embeddings for efficient text retrieval"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MiniCOIL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        onnx_path = hf_hub_download(
            pretrained_model_name,
            filename="onnx/model.onnx",
        )
        model = onnx.load(onnx_path)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            import torch

            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
