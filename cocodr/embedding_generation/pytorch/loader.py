# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
COCO-DR model loader implementation for embedding generation.

COCO-DR (Contrastive and Distributionally Robust) is a dense retrieval model
built on BERT, fine-tuned on MS MARCO for zero-shot dense retrieval.
Embeddings are extracted from the [CLS] token of the last hidden state.
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
    """Available COCO-DR model variants for embedding generation."""

    COCODR_BASE_MSMARCO = "OpenMatch/cocodr-base-msmarco"


class ModelLoader(ForgeModel):
    """COCO-DR model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.COCODR_BASE_MSMARCO: LLMModelConfig(
            pretrained_model_name="OpenMatch/cocodr-base-msmarco",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COCODR_BASE_MSMARCO

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
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="COCO-DR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load COCO-DR model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The COCO-DR model instance.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, query=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            query: Optional query string. If None, uses a default query.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        if query is None:
            query = "What is the capital of France?"

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode the model output to extract CLS token embeddings.

        COCO-DR embeddings are the [CLS] token hidden state from the last layer.

        Args:
            outputs: Model output tuple or BaseModelOutput.
            inputs: Optional input tensors.

        Returns:
            torch.Tensor: CLS token embeddings (batch_size, hidden_size).
        """
        if isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs

        # Extract [CLS] token embedding (first token)
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
