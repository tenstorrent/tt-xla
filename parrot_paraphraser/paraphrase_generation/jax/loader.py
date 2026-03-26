# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parrot Paraphraser on T5 model loader implementation for paraphrase generation."""

from typing import Optional
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available Parrot Paraphraser model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Parrot Paraphraser on T5 model loader implementation for paraphrase generation."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="prithivida/parrot_paraphraser_on_T5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "Can you recommend some upscale restaurants in New York?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""

        return ModelInfo(
            model="Parrot Paraphraser T5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""

        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Parrot Paraphraser T5 model instance."""

        from transformers import FlaxT5ForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"from_pt": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxT5ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Parrot Paraphraser T5 model."""

        from transformers import T5Config

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs_dict = self._tokenizer(self.sample_text, return_tensors="jax")

        config = T5Config.from_pretrained(self._model_name)

        decoder_input_ids = shift_tokens_right(
            inputs_dict["input_ids"],
            config.pad_token_id,
            config.decoder_start_token_id,
        )

        inputs = {
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }

        return inputs
