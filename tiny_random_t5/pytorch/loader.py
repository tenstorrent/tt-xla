# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
echarlaix/tiny-random-t5 model loader
for feature extraction (embedding generation).
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
    """Available model variants for tiny-random-t5."""

    TINY_RANDOM_T5 = "echarlaix/tiny-random-t5"


class ModelLoader(ForgeModel):
    """echarlaix/tiny-random-t5 feature extraction model loader."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_T5: LLMModelConfig(
            pretrained_model_name="echarlaix/tiny-random-t5",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_T5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="tiny-random-t5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        sample_text = "This is a sample input for the T5 feature extraction model."

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            sample_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # T5 encoder-decoder model requires decoder_input_ids
        decoder_start_token_id = self._cached_model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
