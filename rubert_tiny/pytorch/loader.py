# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RuBERT Tiny model loader implementation for sentence embedding generation.
"""

import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available RuBERT Tiny model variants."""

    RUBERT_TINY = "cointegrated_rubert-tiny"


class ModelLoader(ForgeModel):
    """RuBERT Tiny model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.RUBERT_TINY: LLMModelConfig(
            pretrained_model_name="cointegrated/rubert-tiny",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RUBERT_TINY

    # Sample Russian text for embedding generation
    sample_text = "Привет, мир! Это тестовое предложение."

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="RuBERT-Tiny",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RuBERT Tiny model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RuBERT Tiny model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for RuBERT Tiny embedding generation.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for embedding generation.

        Args:
            co_out: Model output (last_hidden_state, pooler_output)
        """
        if isinstance(co_out, (tuple, list)):
            cls_embedding = co_out[0][:, 0, :]
        elif hasattr(co_out, "last_hidden_state"):
            cls_embedding = co_out.last_hidden_state[:, 0, :]
        else:
            cls_embedding = co_out[:, 0, :]

        # Normalize the embedding
        cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=-1)
        print(f"CLS embedding shape: {cls_embedding.shape}")
        print(f"CLS embedding (first 5 values): {cls_embedding[0, :5].tolist()}")
