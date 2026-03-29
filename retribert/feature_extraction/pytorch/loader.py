# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetriBERT model loader for text retrieval feature extraction.
"""
import torch
import torch.nn as nn
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


class RetriBertEmbedWrapper(nn.Module):
    """Wrapper around RetriBertModel that calls embed_questions for inference.

    The native RetriBertModel.forward computes a contrastive loss, which is not
    useful for feature extraction. This wrapper exposes embed_questions as the
    forward pass so the model produces query embeddings (projected to 128-dim).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model.embed_questions(input_ids, attention_mask=attention_mask)


class ModelVariant(StrEnum):
    """Available model variants for RetriBERT."""

    RETRIBERT_BASE_UNCASED = "yjernite/retribert-base-uncased"


class ModelLoader(ForgeModel):
    """RetriBERT model loader for text retrieval feature extraction."""

    _VARIANTS = {
        ModelVariant.RETRIBERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="yjernite/retribert-base-uncased",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RETRIBERT_BASE_UNCASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RetriBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        wrapped = RetriBertEmbedWrapper(model)
        wrapped.eval()

        self.model = wrapped

        return wrapped

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = "How many people live in Berlin?"

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, (tuple, list)):
            return fwd_output[0].flatten()
        return fwd_output.flatten()
