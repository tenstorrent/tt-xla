# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SaT (Segment any Text) model loader implementation for token classification (sentence segmentation).
"""

import torch
import wtpsplit.configs  # noqa: F401 — registers custom "xlm-token" config type
import wtpsplit.models  # noqa: F401 — registers SubwordXLMForTokenClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SaT model variants."""

    SAT_12L = "sat-12l"


class ModelLoader(ForgeModel):
    """SaT model loader implementation for token classification (sentence segmentation)."""

    _VARIANTS = {
        ModelVariant.SAT_12L: ModelConfig(
            pretrained_model_name="segment-any-text/sat-12l",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAT_12L

    sample_text = (
        "This is a test. This is another test. "
        "The quick brown fox jumps over the lazy dog."
    )

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SaT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        logits = co_out[0]
        probs = torch.sigmoid(logits)

        # Use the first label dimension as the sentence boundary probability
        boundary_probs = probs[0, :, 0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask = inputs["attention_mask"][0].bool()

        boundaries = []
        for i, (token, prob, valid) in enumerate(zip(tokens, boundary_probs, mask)):
            if valid and prob > 0.5:
                boundaries.append((i, token, prob.item()))

        print(f"Input text: {self.sample_text}")
        print(f"Detected sentence boundaries at tokens: {boundaries}")
        return boundaries
