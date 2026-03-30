# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SoLU (Softmax Linear Unit) causal language model loader implementation.

SoLU models are small transformer language models designed for mechanistic
interpretability research by Neel Nanda. They use the Softmax Linear Unit
activation function to produce more interpretable neurons.

Source: https://huggingface.co/NeelNanda/SoLU_3L512W_C4_Code
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
    """Available SoLU model variants."""

    SOLU_3L512W_C4_CODE = "Default"


class ModelLoader(ForgeModel):
    """SoLU causal language model loader using TransformerLens."""

    _VARIANTS = {
        ModelVariant.SOLU_3L512W_C4_CODE: LLMModelConfig(
            pretrained_model_name="NeelNanda/SoLU_3L512W_C4_Code",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SOLU_3L512W_C4_CODE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SoLU",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SoLU model via TransformerLens.

        Returns:
            torch.nn.Module: The HookedTransformer model instance.
        """
        from transformer_lens import HookedTransformer

        model_name = self._variant_config.pretrained_model_name

        model = HookedTransformer.from_pretrained(model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for the SoLU model.

        Returns:
            dict: Dictionary with 'input_ids' tensor.
        """
        from transformer_lens import HookedTransformer

        model_name = self._variant_config.pretrained_model_name

        tokenizer = HookedTransformer.from_pretrained(model_name).tokenizer
        self.tokenizer = tokenizer

        input_text = "The quick brown fox jumps over the lazy dog"
        tokens = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            truncation=True,
        )

        return {"input_ids": tokens["input_ids"]}

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
