# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedCPT-Article-Encoder model loader for embedding generation.
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
    """Available MedCPT-Article-Encoder model variants."""

    MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"


class ModelLoader(ForgeModel):
    """MedCPT-Article-Encoder model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.MEDCPT_ARTICLE_ENCODER: LLMModelConfig(
            pretrained_model_name="ncbi/MedCPT-Article-Encoder",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDCPT_ARTICLE_ENCODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MedCPT-Article-Encoder",
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
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        title = "Chronic kidney disease and cardiovascular outcomes"
        abstract = "Chronic kidney disease is associated with increased cardiovascular risk and mortality."

        max_length = getattr(self._variant_config, "max_length", 512)

        inputs = self.tokenizer(
            [[title, abstract]],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            last_hidden_state = output[0]
        elif hasattr(output, "last_hidden_state"):
            last_hidden_state = output.last_hidden_state
        else:
            last_hidden_state = output

        # MedCPT uses [CLS] token embedding as the article representation
        cls_embedding = last_hidden_state[:, 0, :]

        return cls_embedding

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
