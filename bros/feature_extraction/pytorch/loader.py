# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BROS model loader implementation for feature extraction.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available BROS model variants for feature extraction."""

    NAVER_CLOVA_OCR_BROS_BASE_UNCASED = "naver-clova-ocr/bros-base-uncased"


class ModelLoader(ForgeModel):
    """BROS model loader implementation for feature extraction."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.NAVER_CLOVA_OCR_BROS_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="naver-clova-ocr/bros-base-uncased",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.NAVER_CLOVA_OCR_BROS_BASE_UNCASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BROS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BROS model for feature extraction from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BROS model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BROS feature extraction.

        BROS requires both token inputs and bounding box coordinates.
        Bounding boxes represent the spatial position of each token on the document page.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors including input_ids, attention_mask, and bbox.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        max_length = self._variant_config.max_length
        sample_text = "BROS is a pre-trained language model for document understanding"

        # Tokenize the sample text
        inputs = self.tokenizer(
            sample_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Generate synthetic bounding box coordinates for each token
        # BROS expects bbox of shape (batch_size, seq_len, 8) representing
        # 4 corner points (x1, y1, x2, y2, x3, y3, x4, y4) scaled by bbox_scale (100.0)
        seq_len = inputs["input_ids"].shape[1]
        bbox = torch.zeros(1, seq_len, 8, dtype=torch.long)

        # Assign simple sequential bounding boxes to non-padding tokens
        num_tokens = inputs["attention_mask"].sum().item()
        for i in range(int(num_tokens)):
            x1 = i * 10
            y1 = 0
            x2 = (i + 1) * 10
            y2 = 10
            bbox[0, i] = torch.tensor(
                [x1, y1, x2, y1, x2, y2, x1, y2], dtype=torch.long
            )

        inputs["bbox"] = bbox

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode the model output for feature extraction.

        Args:
            outputs: Model output tuple (last_hidden_state, ...).
            inputs: Optional input tensors.

        Returns:
            torch.Tensor: The last hidden state embeddings.
        """
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs
