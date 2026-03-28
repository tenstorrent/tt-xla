# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobileBERT model loader implementation for sentence embedding generation.
"""

import torch
from transformers import AutoTokenizer, MobileBertModel

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
    """Available MobileBERT model variants for sentence embedding generation."""

    OPTIMUM_INTEL_TINY_RANDOM = (
        "optimum-intel-internal-testing/tiny-random-MobileBertModel"
    )


class ModelLoader(ForgeModel):
    """MobileBERT model loader implementation for sentence embedding generation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OPTIMUM_INTEL_TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-MobileBertModel",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPTIMUM_INTEL_TINY_RANDOM

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "This is a sample sentence for embedding generation."
        self.tokenizer = None
        self.model = None

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
            model="MobileBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load MobileBERT model for sentence embedding generation from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MobileBERT model instance.
        """
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MobileBertModel.from_pretrained(self.model_name, **model_kwargs)
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for MobileBERT sentence embedding generation.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

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
        """Decode the model output for sentence embedding generation.

        Uses mean pooling over the last hidden state to produce sentence embeddings.

        Args:
            co_out: Model output tuple (last_hidden_state, ...)
        """
        inputs = self.load_inputs()
        attention_mask = inputs["attention_mask"]

        if isinstance(co_out, (tuple, list)):
            token_embeddings = co_out[0]
        elif hasattr(co_out, "last_hidden_state"):
            token_embeddings = co_out.last_hidden_state
        else:
            token_embeddings = co_out

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        print(f"Sentence embedding shape: {sentence_embeddings.shape}")
