# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model2Vec static embedding model loader for sentence embedding generation.

Model2Vec models are distilled static embedding models that provide fast text
embedding computation with minimal resource requirements.
"""
import torch
import torch.nn as nn
from model2vec import StaticModel
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


class Model2VecTorchModel(nn.Module):
    """Wraps a Model2Vec StaticModel as a torch.nn.Module for hardware compilation."""

    def __init__(self, static_model: StaticModel):
        super().__init__()
        embedding_tensor = torch.from_numpy(static_model.embedding.copy()).float()
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        self.normalize = static_model.normalize
        self.dim = static_model.dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        summed = torch.sum(token_embeddings * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        if self.normalize:
            norm = torch.clamp(torch.norm(mean_pooled, dim=1, keepdim=True), min=1e-32)
            mean_pooled = mean_pooled / norm
        return mean_pooled


class ModelVariant(StrEnum):
    """Available model variants for Model2Vec."""

    M2V_BASE_OUTPUT = "minishlab/M2V_base_output"


class ModelLoader(ForgeModel):
    """Model2Vec static embedding model loader."""

    _VARIANTS = {
        ModelVariant.M2V_BASE_OUTPUT: LLMModelConfig(
            pretrained_model_name="minishlab/M2V_base_output",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.M2V_BASE_OUTPUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._static_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="model2vec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_static_model(self):
        if self._static_model is None:
            model_name = self._variant_config.pretrained_model_name
            self._static_model = StaticModel.from_pretrained(model_name)
        return self._static_model

    def load_model(self, *, dtype_override=None, **kwargs):
        static_model = self._load_static_model()
        model = Model2VecTorchModel(static_model)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        static_model = self._load_static_model()

        if sentence is None:
            sentence = "This is an example sentence for embedding generation."

        max_length = getattr(self._variant_config, "max_length", 128)

        # Tokenize using the model2vec tokenizer
        token_ids_list = static_model.tokenize([sentence], max_length=max_length)
        token_ids = token_ids_list[0]

        # Pad or truncate to max_length
        if len(token_ids) >= max_length:
            token_ids = token_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
            token_ids = token_ids + [0] * (max_length - len(token_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        return fwd_output
