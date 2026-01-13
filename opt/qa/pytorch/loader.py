# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OPT model loader implementation for question answering.
"""
import torch
from transformers import OPTForQuestionAnswering, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
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
    """Available OPT model variants."""

    OPT_125M = "facebook/opt-125m"
    OPT_350M = "facebook/opt-350m"
    OPT_1_3B = "facebook/opt-1.3b"


class ModelLoader(ForgeModel):
    """OPT model loader implementation for question answering tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OPT_125M: LLMModelConfig(
            pretrained_model_name="facebook/opt-125m",
            max_length=32,
        ),
        ModelVariant.OPT_350M: LLMModelConfig(
            pretrained_model_name="facebook/opt-350m",
            max_length=32,
        ),
        ModelVariant.OPT_1_3B: LLMModelConfig(
            pretrained_model_name="facebook/opt-1.3b",
            max_length=32,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPT_125M

    # Shared configuration parameters
    sample_question = "Who was Jim Henson?"
    sample_context = "Jim Henson was a nice puppet"

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
        return ModelInfo(
            model="opt",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
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

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the OPT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The OPT model instance for question answering.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = OPTForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the OPT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model [input_ids, attention_mask].
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the question answering task
        input_tokens = self.tokenizer(
            self.sample_question,
            self.sample_context,
            max_length=self._variant_config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Return as list of tensors for the wrapper
        return [input_tokens["input_ids"], input_tokens["attention_mask"]]
