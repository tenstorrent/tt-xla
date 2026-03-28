# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT MeSH term classification model loader implementation.
"""
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available BERT MeSH classification model variants."""

    MARCMENDEZ_AILY_BERT_MESH_TERMS = "marcmendez_aily_BertMeshTerms"


class ModelLoader(ForgeModel):
    """BERT MeSH term classification model loader implementation."""

    _VARIANTS = {
        ModelVariant.MARCMENDEZ_AILY_BERT_MESH_TERMS: ModelConfig(
            pretrained_model_name="marcmendez-aily/BertMeshTerms",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MARCMENDEZ_AILY_BERT_MESH_TERMS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BERT_MeSH",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        sample_text = "Effect of aspirin on cardiovascular events and bleeding in the healthy elderly."

        inputs = self.tokenizer(
            sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        predicted_indices = (probabilities > 0.5).nonzero(as_tuple=True)[-1]

        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
            predicted_labels = [
                self.model.config.id2label[idx.item()] for idx in predicted_indices
            ]
        else:
            predicted_labels = [str(idx.item()) for idx in predicted_indices]

        print(f"Predicted MeSH Terms: {predicted_labels}")
