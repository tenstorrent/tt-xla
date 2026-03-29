# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ALBERT model loader implementation for masked language modeling.
"""
import torch
from transformers import AlbertForMaskedLM, AlbertTokenizer, BertTokenizerFast
from typing import Optional

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
    """Available ALBERT model variants."""

    BASE_V1 = "Base_v1"
    LARGE_V1 = "Large_v1"
    XLARGE_V1 = "Xlarge_v1"
    XXLARGE_V1 = "Xxlarge_v1"
    BASE_V2 = "Base_v2"
    LARGE_V2 = "Large_v2"
    XLARGE_V2 = "Xlarge_v2"
    XXLARGE_V2 = "Xxlarge_v2"
    KOR_BASE = "Kor_Base"


class ModelLoader(ForgeModel):
    """ALBERT model loader implementation for masked language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_V1: LLMModelConfig(
            pretrained_model_name="albert-base-v1",
            max_length=128,
        ),
        ModelVariant.LARGE_V1: LLMModelConfig(
            pretrained_model_name="albert-large-v1",
            max_length=128,
        ),
        ModelVariant.XLARGE_V1: LLMModelConfig(
            pretrained_model_name="albert-xlarge-v1",
            max_length=128,
        ),
        ModelVariant.XXLARGE_V1: LLMModelConfig(
            pretrained_model_name="albert-xxlarge-v1",
            max_length=128,
        ),
        ModelVariant.BASE_V2: LLMModelConfig(
            pretrained_model_name="albert-base-v2",
            max_length=128,
        ),
        ModelVariant.LARGE_V2: LLMModelConfig(
            pretrained_model_name="albert-large-v2",
            max_length=128,
        ),
        ModelVariant.XLARGE_V2: LLMModelConfig(
            pretrained_model_name="albert-xlarge-v2",
            max_length=128,
        ),
        ModelVariant.XXLARGE_V2: LLMModelConfig(
            pretrained_model_name="albert-xxlarge-v2",
            max_length=128,
        ),
        ModelVariant.KOR_BASE: LLMModelConfig(
            pretrained_model_name="kykim/albert-kor-base",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_V2

    # Shared configuration parameters
    sample_text = "The capital of France is [MASK]."

    # Variants that use BertTokenizerFast instead of AlbertTokenizer
    _BERT_TOKENIZER_VARIANTS = {ModelVariant.KOR_BASE}

    # Per-variant sample texts (Korean variant uses Korean text)
    _SAMPLE_TEXTS = {
        ModelVariant.KOR_BASE: "한국의 수도는 [MASK]입니다.",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        group = (
            ModelGroup.VULCAN
            if variant == ModelVariant.KOR_BASE
            else ModelGroup.GENERALITY
        )
        return ModelInfo(
            model="ALBERT",
            variant=variant,
            group=group,
            task=ModelTask.NLP_MASKED_LM,
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

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer (Korean variant uses BertTokenizerFast)
        tokenizer_cls = (
            BertTokenizerFast
            if self._variant in self._BERT_TOKENIZER_VARIANTS
            else AlbertTokenizer
        )
        self.tokenizer = tokenizer_cls.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ALBERT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ALBERT model instance for masked language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AlbertForMaskedLM.from_pretrained(pretrained_model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ALBERT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Create tokenized inputs for the masked language modeling task
        text = self._SAMPLE_TEXTS.get(self._variant, self.sample_text)
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded prediction for the masked token
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        # Get the prediction for the masked token
        logits = outputs[0]
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
