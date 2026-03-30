# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multilingual E5 model loader for sentence embedding generation.

Uses the XLM-RoBERTa architecture via AutoModel for multilingual text embeddings.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available multilingual E5 model variants."""

    MULTILINGUAL_E5_SMALL = "intfloat/multilingual-e5-small"
    MULTILINGUAL_E5_BASE = "intfloat/multilingual-e5-base"
    MULTILINGUAL_E5_LARGE_INSTRUCT = "intfloat/multilingual-e5-large-instruct"
    E5_LARGE_TRM_NL = "clips/e5-large-trm-nl"


class ModelLoader(ForgeModel):
    """Multilingual E5 model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.MULTILINGUAL_E5_SMALL: LLMModelConfig(
            pretrained_model_name="intfloat/multilingual-e5-small",
            max_length=512,
        ),
        ModelVariant.MULTILINGUAL_E5_BASE: LLMModelConfig(
            pretrained_model_name="intfloat/multilingual-e5-base",
            max_length=512,
        ),
        ModelVariant.MULTILINGUAL_E5_LARGE_INSTRUCT: LLMModelConfig(
            pretrained_model_name="intfloat/multilingual-e5-large-instruct",
            max_length=512,
        ),
        ModelVariant.E5_LARGE_TRM_NL: LLMModelConfig(
            pretrained_model_name="clips/e5-large-trm-nl",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL_E5_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="multilingual-e5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            if self._variant == ModelVariant.MULTILINGUAL_E5_LARGE_INSTRUCT:
                sentence = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: How is the weather today?"
            elif self._variant == ModelVariant.E5_LARGE_TRM_NL:
                sentence = "query: Hoe is het weer vandaag?"
            else:
                sentence = "query: How is the weather today?"

        max_length = getattr(self._variant_config, "max_length", 512)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
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
