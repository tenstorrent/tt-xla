# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
NLLB-200 model loader implementation for text translation.
"""

import torch
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
    """Available NLLB-200 model variants."""

    DISTILLED_600M = "Distilled_600M"


class ModelLoader(ForgeModel):
    """NLLB-200 model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.DISTILLED_600M: LLMModelConfig(
            pretrained_model_name="facebook/nllb-200-distilled-600M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILLED_600M

    sample_text = "Hello, how are you today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

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
            model="NLLB-200",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NLLB-200 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import M2M100ForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = M2M100ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the NLLB-200 model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        self._tokenizer.src_lang = "eng_Latn"

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Seq2seq models need decoder_input_ids for the forward pass.
        # Use the target language BOS token to start decoding.
        target_lang_id = self._tokenizer.convert_tokens_to_ids("fra_Latn")
        inputs["decoder_input_ids"] = torch.tensor([[target_lang_id]])

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
