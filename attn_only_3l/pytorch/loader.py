# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Attention-Only 3-Layer Transformer model loader implementation.

This is a small (2.36M parameter) attention-only transformer from Neel Nanda's
mechanistic interpretability research, loaded via the TransformerLens library.
"""

from transformer_lens import HookedTransformer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Attention-Only 3L model variants."""

    ATTN_ONLY_3L = "Attn_Only_3L512W_C4_Code"


class ModelLoader(ForgeModel):
    """Attention-Only 3-Layer Transformer model loader."""

    _VARIANTS = {
        ModelVariant.ATTN_ONLY_3L: ModelConfig(
            pretrained_model_name="NeelNanda/Attn_Only_3L512W_C4_Code",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ATTN_ONLY_3L

    sample_text = "Hello, world! This is a test of"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Attn-Only-3L",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Attention-Only 3L model via TransformerLens."""
        model = HookedTransformer.from_pretrained(
            "attn-only-3l",
            device="cpu",
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample token inputs for the model.

        Returns a tensor of token IDs, as HookedTransformer expects a
        positional tensor input rather than keyword arguments.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        tokens = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128,
        )

        return tokens["input_ids"]

    def _load_tokenizer(self):
        """Load the tokenizer used by this model."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "NeelNanda/gpt-neox-tokenizer-digits"
        )
        return self.tokenizer

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text.

        HookedTransformer returns raw logits as a tensor of shape
        (batch, seq_len, vocab_size).
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
