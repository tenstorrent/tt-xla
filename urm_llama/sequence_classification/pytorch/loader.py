# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
URM-LLaMa model loader implementation for sequence classification.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    """Available URM-LLaMa model variants for sequence classification."""

    URM_LLAMA_3_1_8B = "URM-LLaMa-3.1-8B"


class ModelLoader(ForgeModel):
    """URM-LLaMa model loader for sequence classification (reward scoring)."""

    _VARIANTS = {
        ModelVariant.URM_LLAMA_3_1_8B: LLMModelConfig(
            pretrained_model_name="LxzGordon/URM-LLaMa-3.1-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.URM_LLAMA_3_1_8B

    sample_text = "Movie is great"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="URM-LLaMa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the URM-LLaMa model for sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for sequence classification.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model

        return model

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        return self.tokenizer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into the reward score.

        Args:
            outputs: Model output from a forward pass (logits)
            inputs: Optional input tensors used to generate the outputs

        Returns:
            float: The reward score
        """
        logits = outputs[0]
        reward_score = logits[0].item()
        return reward_score
