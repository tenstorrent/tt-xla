# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T5 v1.1 model loader implementation
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available T5 v1.1 model variants."""

    BASE = "Base"
    LARGE = "Large"
    XXL = "XXL"


class ModelLoader(ForgeModel):
    """T5 v1.1 model loader implementation for conditional generation tasks."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="google/t5-v1_1-base",
            max_length=512,
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="google/t5-v1_1-large",
            max_length=512,
        ),
        ModelVariant.XXL: LLMModelConfig(
            pretrained_model_name="google/t5-v1_1-xxl",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    sample_text = """summarize: Researchers have extensively studied the benefits of having pets,
                    particularly dogs, on human health and well-being. Findings suggest that pet ownership
                    can lead to improved mental health, reduced stress levels, and even physical health benefits
                    such as lower blood pressure and increased physical activity levels due to regular walks."""

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="T5_v1_1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.use_cache = False

        if self.num_layers is not None:
            config.num_layers = self.num_layers
            config.num_decoder_layers = self.num_layers

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
