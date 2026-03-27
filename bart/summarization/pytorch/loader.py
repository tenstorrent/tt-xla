# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BART model loader implementation for summarization.
"""
from typing import Optional

from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

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


class ModelVariant(StrEnum):
    """Available BART model variants for summarization."""

    MEETING_SUMMARY = "Meeting_Summary"


class ModelLoader(ForgeModel):
    """BART model loader implementation for summarization."""

    _VARIANTS = {
        ModelVariant.MEETING_SUMMARY: LLMModelConfig(
            pretrained_model_name="knkarthick/MEETING_SUMMARY",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEETING_SUMMARY

    sample_text = (
        "The tower is 324 metres (1,063 ft) tall, about the same height as an "
        "81-storey building, and the tallest structure in Paris. Its base is square, "
        "measuring 125 metres (410 ft) on each side. It was the first structure to "
        "reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is "
        "the second tallest free-standing structure in France after the Millau Viaduct."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="BART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BART model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BART model instance for summarization.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BART model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs_dict = self._tokenizer(
            self.sample_text,
            truncation=True,
            return_tensors="pt",
        )

        model = self.load_model()
        decoder_input_ids = shift_tokens_right(
            inputs_dict["input_ids"],
            model.config.pad_token_id,
            model.config.decoder_start_token_id,
        )

        inputs = [
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            decoder_input_ids,
        ]
        return inputs
